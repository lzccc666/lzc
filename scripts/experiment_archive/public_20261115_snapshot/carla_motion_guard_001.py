"""无有效路线也执行的当前LiDAR运动检查；须在正式接入前完成实际停车标定。"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import MultiPoint, Point
from sklearn.cluster import DBSCAN

from conceptgraph.navigation.vehicle_geometry import footprint_polygon, step_vehicle, swept_body_polygon
from conceptgraph.navigation.vehicle_planner import tracking_control


def current_full_lidar_obstacles(points, pose_carla, state, spec, *, sensor_frame,
                                 snapshot_frame, ground_z=0.):
    """观测来自整圈原始LiDAR，不依据相机视野或目标类别过滤障碍。"""
    points=np.asarray(points,dtype=float)
    matrix=np.asarray(pose_carla,dtype=float)
    evidence=dict(sensor_frame=int(sensor_frame),snapshot_frame=int(snapshot_frame),
        source='current_official_full_360_LiDAR',ground_z_m=float(ground_z),
        ground_scope='显式平坦道路高度先验，尚未对起伏道路验证',GT_in_obstacle_input=False)
    if (sensor_frame!=snapshot_frame or points.ndim!=2 or points.shape[1]<3 or len(points)<100
            or matrix.shape!=(4,4) or not np.isfinite(matrix).all()
            or not np.isfinite(points[:,:3]).all()):
        return [],dict(evidence,valid=False,reason='missing_malformed_or_unsynchronized_LiDAR')
    transform=np.diag([1.,-1.,1.,1.])@matrix
    world=(transform@np.column_stack([points[:,:3],np.ones(len(points))]).T).T[:,:3]
    elevated=world[(world[:,2]>ground_z+.25)&(world[:,2]<ground_z+4.)]
    body=footprint_polygon(state,spec,.15)
    keep=np.array([not body.covers(Point(p[:2])) for p in elevated],dtype=bool)
    xy=elevated[keep,:2]
    if len(xy)<3:
        return [],dict(evidence,valid=True,raw_points=len(points),elevated_non_ego_points=len(xy),clusters=0)
    labels=DBSCAN(eps=.6,min_samples=3).fit_predict(xy)
    obstacles=[MultiPoint(xy[labels==label]).convex_hull.buffer(.2)
               for label in sorted(set(labels)) if label>=0]
    return obstacles,dict(evidence,valid=True,raw_points=len(points),
        elevated_non_ego_points=len(xy),clusters=len(obstacles),
        missing_unobserved_obstacles_still_possible=True)


def nominal_coasting_check(state, road, obstacles, spec, *, observation,
                          path=None, goal=None, passive_deceleration_mps2=.03,
                          low_speed_brake_gate_mps=.15, reaction_s=.3):
    """独立于路径存在性的保守名义滑行检查；不是未经验证的物理安全证明。

    passive_deceleration必须由后续标定协议指定。本函数只检查预测轨迹，实际
    停车距离、切油减速度及到达仍须用官方执行数据单独评价。
    """
    if not math.isfinite(passive_deceleration_mps2) or passive_deceleration_mps2<=0:
        raise ValueError('positive declared passive deceleration required')
    if not observation or not observation.get('valid'):
        return dict(stop_required=True,reason='invalid_current_observation',
            checked_without_route=not bool(path),actual_physical_safety_proven=False)
    if state.speed_mps<0 or not math.isfinite(state.speed_mps):
        return dict(stop_required=True,reason='invalid_current_speed',actual_physical_safety_proven=False)
    states=[state]
    horizon=math.ceil(max(0.,state.speed_mps-low_speed_brake_gate_mps)/passive_deceleration_mps2/spec.dt_s)
    horizon+=math.ceil(reaction_s/spec.dt_s)+2
    reaction_ticks=math.ceil(reaction_s/spec.dt_s)
    violations=[]
    for tick in range(horizon+1):
        current=states[-1]
        body=footprint_polygon(current,spec,.05)
        shapes=[body]
        if len(states)>1:shapes.append(swept_body_polygon(states[-2],current,spec,.05))
        if any(not road.covers(shape) for shape in shapes):
            violations.append(dict(tick=tick,reason='predicted_body_or_sweep_leaves_road'))
            break
        if any(shape.intersects(p) for shape in shapes for p in obstacles):
            violations.append(dict(tick=tick,reason='predicted_body_or_sweep_hits_current_obstacle'))
            break
        if tick==horizon:break
        steer=current.steering_rad
        if path and goal is not None:
            _,steer,_=tracking_control(current,path,goal,spec)
        if tick<reaction_ticks:acceleration=0.
        elif current.speed_mps<low_speed_brake_gate_mps:acceleration=-spec.max_braking_mps2
        else:acceleration=-passive_deceleration_mps2
        states.append(step_vehicle(current,acceleration,steer,spec))
    return dict(stop_required=bool(violations),reason=violations[0]['reason'] if violations else 'nominal_horizon_clear',
        violations=violations,checked_states=len(states),checked_without_route=not bool(path),
        declared_passive_deceleration_mps2=passive_deceleration_mps2,
        declared_reaction_s=reaction_s,predicted_travel_m=sum(
            math.hypot(b.x-a.x,b.y-a.y) for a,b in zip(states,states[1:])),
        actual_physical_safety_proven=False,requires_official_actuator_and_navigation_validation=True)
