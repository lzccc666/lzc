"""带转角状态、前向控制基元和车身扫掠检查的有界Hybrid A*。"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
import time

import numpy as np
from shapely.geometry import Point, LineString
from shapely.prepared import prep

from conceptgraph.navigation.vehicle_geometry import (
    VehicleSpec, VehicleState, step_vehicle, footprint_polygon,
    swept_body_polygon, angle_difference,
)


@dataclass
class VehiclePlan:
    states: list[VehicleState]
    expanded: int
    duration_s: float
    failure: str | None = None
    algorithm: str = 'forward_steering_state_hybrid_astar'


def _reference_heading(reference,position):
    s=reference.project(Point(position))
    a=reference.interpolate(max(0,s-.2))
    b=reference.interpolate(min(reference.length,s+.2))
    return math.atan2(b.y-a.y,b.x-a.x)


def hybrid_astar(start,goal_xy,goal_yaw,drivable_polygon,obstacles,reference_line,
                 spec=VehicleSpec(),budget_s=10,max_expanded=6000):
    """已知有向单车道走廊；多车道连接仍由上层拓扑提供，禁止逆行和倒车。"""
    started=time.perf_counter()
    road=prep(drivable_polygon)
    obstacle_prepared=[prep(o) for o in obstacles]
    goal=np.asarray(goal_xy,dtype=float)
    reference=reference_line if isinstance(reference_line,LineString) else LineString(reference_line)
    def legal(body):
        return road.covers(body) and not any(o.intersects(body) for o in obstacle_prepared)
    if not legal(footprint_polygon(start,spec,.05)):
        return VehiclePlan([],0,time.perf_counter()-started,'INVALID_INITIAL_BODY')
    def key(state):
        return (round(state.x/.4),round(state.y/.4),round(state.yaw_rad/math.radians(7.5)),
            round(state.steering_rad/math.radians(16)),round(state.speed_mps/.25))
    def heuristic(state):
        return max(0,math.hypot(state.x-goal[0],state.y-goal[1])-.65)
    sequence=itertools.count()
    first=next(sequence)
    nodes={first:(start,None,[],0.0)}
    queue=[(heuristic(start),first)]
    best={key(start):0.0}
    expanded=0
    steering_targets=[0,-spec.max_steering_rad/2,spec.max_steering_rad/2,
        -spec.max_steering_rad,spec.max_steering_rad]
    while queue and expanded<max_expanded:
        if time.perf_counter()-started>budget_s:
            return VehiclePlan([],expanded,time.perf_counter()-started,'PLANNING_TIME_BUDGET')
        _,identity=heapq.heappop(queue)
        state,parent,segment,cost=nodes[identity]
        if cost>best.get(key(state),math.inf)+1e-8:
            continue
        if heuristic(state)<=1e-8 and abs(angle_difference(state.yaw_rad,goal_yaw))<=math.radians(12):
            segments=[]
            current=identity
            while nodes[current][1] is not None:
                segments.append(nodes[current][2])
                current=nodes[current][1]
            path=[start]
            for points in reversed(segments):
                path.extend(points)
            return VehiclePlan(path,expanded,time.perf_counter()-started)
        expanded+=1
        prior_s=reference.project(Point(state.x,state.y))
        for target in steering_targets:
            current=state
            primitive=[]
            valid=True
            for _ in range(10):
                next_state=step_vehicle(current,(1.0-current.speed_mps)/spec.dt_s,target,spec)
                if not legal(footprint_polygon(next_state,spec,.05)):
                    valid=False
                    break
                if not legal(swept_body_polygon(current,next_state,spec,.05)):
                    valid=False
                    break
                current=next_state
                primitive.append(current)
            if not valid:
                continue
            progress=reference.project(Point(current.x,current.y))
            if progress<prior_s-.02:
                continue
            heading=_reference_heading(reference,[current.x,current.y])
            if abs(angle_difference(current.yaw_rad,heading))>math.radians(60):
                continue
            length=sum(math.hypot(b.x-a.x,b.y-a.y) for a,b in zip([state]+primitive,primitive))
            deviation=reference.distance(Point(current.x,current.y))
            new_cost=cost+length+.08*deviation+.05*abs(current.steering_rad-state.steering_rad)
            k=key(current)
            if new_cost>=best.get(k,math.inf)-1e-9:
                continue
            best[k]=new_cost
            child=next(sequence)
            nodes[child]=(current,identity,primitive,new_cost)
            heapq.heappush(queue,(new_cost+heuristic(current),child))
    return VehiclePlan([],expanded,time.perf_counter()-started,
        'EXPANSION_BUDGET' if expanded>=max_expanded else 'NO_FEASIBLE_PRIMITIVE_ROUTE')


def tracking_control(state,path,goal_xy,spec=VehicleSpec(),cruise_speed_mps=1.3):
    """纯跟踪控制器输出加速度/转角；位置只能由车辆积分更新。"""
    points=np.asarray(path,dtype=float)
    if len(points)<2:
        return -spec.max_braking_mps2,0.0,dict(reason='empty_path')
    line=LineString(points)
    progress=line.project(Point(state.x,state.y))
    lookahead=max(2.0,state.speed_mps*1.6)
    target=line.interpolate(min(line.length,progress+lookahead))
    dx,dy=target.x-state.x,target.y-state.y
    distance=max(math.hypot(dx,dy),.05)
    alpha=angle_difference(math.atan2(dy,dx),state.yaw_rad)
    steering=math.atan2(2*spec.wheelbase_m*math.sin(alpha),distance)
    remaining=max(0,line.length-progress)
    goal_distance=float(np.linalg.norm(np.asarray(goal_xy)-[state.x,state.y]))
    desired=min(cruise_speed_mps,math.sqrt(2*.8*max(0,remaining-.2)))
    if remaining<.35 or goal_distance<.5:
        desired=0
    acceleration=(desired-state.speed_mps)*1.5
    return acceleration,steering,dict(progress_m=progress,remaining_m=remaining,
        cross_track_error_m=line.distance(Point(state.x,state.y)),goal_distance_m=goal_distance,
        desired_speed_mps=desired)
