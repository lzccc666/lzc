"""Past-only graph/odometry following baseline; no official objects or futures."""
from __future__ import annotations

from collections import deque
import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from conceptgraph.completion.reference_regions import load_reference_prior
from conceptgraph.navigation.vehicle_geometry import angle_difference

POLICY = dict(seed=17, graph='T3_three', horizon_s=3., dt_s=.1,
    maximum_measurement_age_s=1.5, minimum_motion_span_s=.4,
    target_categories=['car','truck','bus'], minimum_forward_m=5., maximum_forward_m=60.,
    maximum_lane_hops=8, maximum_lane_join_gap_m=.75, desired_body_gap_m=15.,
    gap_gain_per_s=.25, max_speed_mps=15.,
    output='timed_xy_yaw_trajectory_only; no acceleration or control claims',
    identity_evaluation='one-to-one Hungarian assignment within vehicle family, 3D center distance <=3m; no GT used by method',
    map='same shared official HD lane/drivable map for all tasks',GPU_calls=0,qwen_calls=0)


def routes_from(start_lane, lanes):
    queue=deque([[start_lane['token']]]); routes={}
    while queue:
        route=queue.popleft();token=route[-1]
        if token in routes:continue
        routes[token]=route
        if len(route)>POLICY['maximum_lane_hops']:continue
        for nxt in sorted(lanes[token]['outgoing']):
            if nxt in lanes and nxt not in route and math.dist(lanes[token]['end_pose'][:2],lanes[nxt]['start_pose'][:2])<=POLICY['maximum_lane_join_gap_m']:
                queue.append(route+[nxt])
    return routes


def infer(task):
    result=dict(task_id=task['task_id'],status='NO_CURRENT_GRAPH',selected_node_id=None,
        selected_center=None,predicted_states=[],GT_in_inference=False,past_only=True)
    history=task['prediction_history']; now=task['issue_time_s']
    if not history or not history[-1]['objects']:return result
    odometry=task['past_rear_axle_odometry']
    if len(odometry)<2:
        result['status']='INSUFFICIENT_PAST_ODOMETRY';return result
    prior=load_reference_prior(task['map_location'])
    lanes={x['token']:x for x in prior['lanes']}
    pose=np.asarray(task['initial_rear_axle_world_pose']);ego=pose[:2,3]
    heading=task['initial_yaw_rad'];seeds=[]
    for lane in lanes.values():
        if Polygon(lane['polygon']).covers(Point(ego)):
            xy=np.asarray(lane['centerline']);index=int(np.argmin(np.linalg.norm(xy[:,:2]-ego,axis=1)))
            if abs(angle_difference(heading,float(xy[index,2])))<=math.radians(30):
                seeds.append((LineString(xy[:,:2]).distance(Point(ego)),lane['token']))
    if not seeds:result['status']='NO_DIRECTED_START_LANE';return result
    routes=routes_from(lanes[min(seeds)[1]],lanes)
    candidates=[]
    result['shape_missing_current_nodes']=sum(o['extent'] is None for o in history[-1]['objects'])
    for obj in history[-1]['objects']:
        if obj['category'] not in POLICY['target_categories']:continue
        if obj['extent'] is None:continue
        age=now-obj['measurement_timestamp_s']
        if not 0<=age<=POLICY['maximum_measurement_age_s']:continue
        for token,route in routes.items():
            if not Polygon(lanes[token]['polygon']).covers(Point(obj['center'][:2])):continue
            line=LineString(np.concatenate([np.asarray(lanes[t]['centerline'])[:,:2] for t in route]))
            distance=line.project(Point(obj['center'][:2]))-line.project(Point(ego))
            if POLICY['minimum_forward_m']<=distance<=POLICY['maximum_forward_m']:
                candidates.append((distance,str(obj['node_id']),obj,route,line))
    if not candidates:result['status']='NO_FORWARD_PREDICTED_VEHICLE';return result
    distance,node,obj,route,line=min(candidates,key=lambda x:x[:2])
    result.update(selected_node_id=node,selected_center=obj['center'],selected_category=obj['category'])
    measurements={}
    for frame in history:
        for record in frame['objects']:
            if str(record['node_id'])==node:
                measurements[record['measurement_timestamp_s']]=np.asarray(record['center'][:2])
    times=sorted(measurements)
    if len(times)<2 or times[-1]-times[0]<POLICY['minimum_motion_span_s']:
        result['status']='INSUFFICIENT_PAST_TARGET_MOTION';return result
    elapsed=times[-1]-times[0]
    target_speed=(line.project(Point(measurements[times[-1]]))-line.project(Point(measurements[times[0]])))/elapsed
    past=odometry[-3:]
    delta=past[-1]['time_s']-past[0]['time_s']
    if delta<=0:result['status']='INVALID_ODOMETRY_TIME';return result
    velocity=(np.asarray(past[-1]['xy'])-past[0]['xy'])/delta
    ego_speed=max(0.,float(velocity@np.array([math.cos(heading),math.sin(heading)])))
    # Continue the chosen map route by the straightest legal successor, without future observations.
    while len(route)<=POLICY['maximum_lane_hops']:
        previous=lanes[route[-1]];choices=[]
        for nxt in previous['outgoing']:
            if nxt in lanes and nxt not in route and math.dist(previous['end_pose'][:2],lanes[nxt]['start_pose'][:2])<=POLICY['maximum_lane_join_gap_m']:
                choices.append((abs(angle_difference(float(lanes[nxt]['start_pose'][2]),float(previous['end_pose'][2]))),nxt))
        if not choices:break
        route=route+[min(choices)[1]]
    line=LineString(np.concatenate([np.asarray(lanes[t]['centerline'])[:,:2] for t in route]))
    start=line.project(Point(ego));closest=line.interpolate(start)
    offset=ego-np.array([closest.x,closest.y])
    length_estimate=max(float(obj['extent'][0]),float(obj['extent'][1]))
    gap=distance-3.7-length_estimate/2
    command=max(0.,min(POLICY['max_speed_mps'],ego_speed,target_speed+POLICY['gap_gain_per_s']*(gap-POLICY['desired_body_gap_m'])))
    states=[]
    for t in np.linspace(0.,POLICY['horizon_s'],31):
        arc=start+command*t;point=line.interpolate(min(arc,line.length))
        a=line.interpolate(max(0.,min(arc,line.length)-.1));b=line.interpolate(min(line.length,arc+.1))
        tangent=math.atan2(b.y-a.y,b.x-a.x)
        blend=min(1.,float(t));xy=np.array([point.x,point.y])+(1-blend)*offset
        direction=heading+blend*angle_difference(tangent,heading)
        states.append(dict(time_s=float(t),x=float(xy[0]),y=float(xy[1]),yaw_rad=direction))
    result.update(status='PREDICTED_GEOMETRY_TIME_TRAJECTORY',predicted_states=states,
        selected_lane_tokens=route,estimated_ego_speed_mps=ego_speed,
        estimated_target_speed_mps=float(target_speed),command_speed_mps=float(command),
        target_history_span_s=elapsed,predicted_initial_body_gap_m=gap)
    return result
