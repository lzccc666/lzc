"""复用已规划路线的形状；从当前状态重新积分和独立审计，不复用旧安全标志。"""
from __future__ import annotations

import math
import time

import numpy as np
from shapely.geometry import Point

from conceptgraph.navigation.vehicle_geometry import (
    angle_difference, step_vehicle, audit_executed_states, footprint_polygon, swept_body_polygon,
)
from conceptgraph.navigation.vehicle_planner import VehiclePlan, tracking_control


def revalidate_route(start, path, goal, goal_yaw, road, obstacles, line, spec, *, budget_s=1.):
    started=time.perf_counter()
    goal=np.asarray(goal,dtype=float)
    if len(path)<2 or np.linalg.norm(np.asarray(path[-1])-goal)>.65:
        return None,'route_endpoint_no_longer_matches_goal'
    states=[start]
    progress=line.project(Point(start.x,start.y))
    for _ in range(600):
        state=states[-1]
        if (np.linalg.norm(np.asarray([state.x,state.y])-goal)<=.65 and state.speed_mps<=.15
                and abs(angle_difference(state.yaw_rad,goal_yaw))<=math.radians(12)):
            break
        if time.perf_counter()-started>budget_s:
            return None,'revalidation_time_budget'
        acceleration,steering,_=tracking_control(state,path,goal,spec)
        following=step_vehicle(state,acceleration,steering,spec)
        next_progress=line.project(Point(following.x,following.y))
        a=line.interpolate(max(0,next_progress-.2))
        b=line.interpolate(min(line.length,next_progress+.2))
        heading=math.atan2(b.y-a.y,b.x-a.x)
        if next_progress<progress-.02 or abs(angle_difference(following.yaw_rad,heading))>math.radians(60):
            return None,'directed_lane_progress_or_heading_changed'
        states.append(following)
        progress=next_progress
    else:
        return None,'tracking_does_not_reach_current_goal'
    # 与原Hybrid规划相同的5cm车身/扫掠余量，不因复用路线放宽保护。
    for i,state in enumerate(states):
        shapes=[footprint_polygon(state,spec,.05)]
        if i:
            shapes.append(swept_body_polygon(states[i-1],state,spec,.05))
        if any(not road.covers(s) or any(s.intersects(o) for o in obstacles) for s in shapes):
            return None,'current_body_or_swept_motion_invalid'
    audit=audit_executed_states(states,road,obstacles,spec)
    if not audit['passed']:
        return None,'current_body_or_swept_motion_invalid'
    if time.perf_counter()-started>budget_s:
        return None,'revalidation_time_budget'
    return VehiclePlan(states,0,time.perf_counter()-started,
        algorithm='current_state_tracking_of_cached_hybrid_route'),'current_state_and_obstacles_revalidated'
