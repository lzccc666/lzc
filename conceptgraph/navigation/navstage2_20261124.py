"""历史对象接近的两阶段开环入口；推理只读预测图、已知后轴和共同HD地图。

目标认证与真值评价在独立脚本中进行。本模块不接收参考轨迹、官方对象或目标区域。
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import math
import time

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from conceptgraph.completion.execution_direction_audit import DirectedPolyline
from conceptgraph.completion.reference_regions import load_reference_prior
from conceptgraph.completion.selection_ablation import select as semantic_candidates
from conceptgraph.completion.topology_vehicle import DirectedVehicleCorridor
from conceptgraph.navigation.reference_guided_vehicle import reference_guided_vehicle
from conceptgraph.navigation.vehicle_geometry import (
    VehicleSpec, VehicleState, angle_difference, audit_executed_states,
    footprint_polygon, step_vehicle,
)

POLICY = dict(seed=17, maximum_objects=3, maximum_goals_per_object=2,
    query_planning_budget_s=3., total_planning_budget_s=300., maximum_expanded=2000,
    maximum_lane_edges=20, maximum_lane_expansions=4000, map_radius_m=100.,
    minimum_approach_distance_m=2., maximum_approach_distance_m=8.,
    terminal_speed_max_mps=.02, terminal_hold_s=2., horizon_s=90.,
    semantic_condition='full_joint_rule', candidate_selection='shortest_verified_timed_route_then_semantic_rank_node_id',
    reverse_allowed=False, qwen_calls=0, GPU_calls=0,
    scope='historical_object_approach_open_loop_regression_not_three_new_tasks')


class MethodMap:
    def __init__(self, location):
        prior = load_reference_prior(location)
        self.sha256, self.path = prior['sha256'], prior['path']
        self.road = prior['drivable']
        self.lanes = {r['token']: r for r in prior['lanes']}
        self.polygons = {t: Polygon(r['polygon']) for t, r in self.lanes.items()}
        self.transitions = {t: r['outgoing'] for t, r in self.lanes.items()}

    def local_paths(self, start, policy):
        point = Point(start.x, start.y)
        window = point.buffer(policy['map_radius_m'])
        available = {t for t, p in self.polygons.items() if p.intersects(window)}
        seeds = []
        for t in sorted(available):
            if self.polygons[t].covers(point):
                c = self.lanes[t]['centerline']
                nearest = int(np.argmin(np.linalg.norm(c[:, :2]-[start.x, start.y], axis=1)))
                if abs(angle_difference(start.yaw_rad, float(c[nearest, 2]))) <= math.radians(60):
                    seeds.append(t)
        queue = deque([[t] for t in seeds]); paths = {}; expanded = 0
        while queue and expanded < policy['maximum_lane_expansions']:
            route = queue.popleft(); expanded += 1; token = route[-1]
            if token in paths:
                continue
            paths[token] = route
            if len(route)-1 >= policy['maximum_lane_edges']:
                continue
            for nxt in sorted(self.transitions[token]):
                if nxt in available and nxt not in route:
                    if math.dist(self.lanes[token]['end_pose'][:2], self.lanes[nxt]['start_pose'][:2]) <= .75:
                        queue.append(route+[nxt])
        return paths


class MethodCorridor(DirectedVehicleCorridor):
    """沿用原有连续有向检查，同时允许合法单车道无连接的任务。"""
    def __init__(self, lanes, transitions, drivable, obstacles):
        if len(lanes) > 1:
            super().__init__(lanes, transitions, drivable, obstacles, endpoint_tolerance_m=.75)
            return
        self.lanes = lanes; self.tokens = [lanes[0]['token']]
        self.polygons = [Polygon(lanes[0]['polygon'])]
        self.road = self.polygons[0].intersection(drivable)
        self.obstacles = obstacles
        self.lines = [DirectedPolyline(np.asarray(lanes[0]['centerline'])[:, :2])]
        self.endpoint_tolerance_m = .75


def prediction_polygon(obj):
    center, extent = np.asarray(obj['center']), np.asarray(obj['extent'])
    if not np.isfinite(np.r_[center, extent]).all() or np.any(extent <= 0):
        raise ValueError('invalid_predicted_object_geometry')
    # 预测extent是观测点云的世界轴包围盒；不注入官方大小或未知朝向。
    return box(center[0]-extent[0]/2, center[1]-extent[1]/2,
               center[0]+extent[0]/2, center[1]+extent[1]/2)


def proposals(obj, paths, common_map, start, obstacles, spec, policy):
    target = prediction_polygon(obj); proposals = []
    for token, route in sorted(paths.items()):
        lane = common_map.lanes[token]
        samples = lane['centerline'][::4]
        for index, pose in enumerate(samples):
            distance = target.distance(Point(pose[:2]))
            if not policy['minimum_approach_distance_m'] <= distance <= policy['maximum_approach_distance_m']:
                continue
            state = VehicleState(float(pose[0]), float(pose[1]), float(pose[2]))
            body = footprint_polygon(state, spec, .05)
            if not common_map.road.covers(body) or any(body.intersects(p) for p in obstacles):
                continue
            chain = [common_map.lanes[t] for t in route]
            arrays = [np.asarray(r['centerline'])[:, :2] for r in chain]
            # HD中心线只来自共同地图；起终点由已知后轴与预测对象几何决定。
            arrays[0] = arrays[0][int(np.argmin(np.linalg.norm(arrays[0]-[start.x, start.y], axis=1))):]
            arrays[-1] = arrays[-1][:int(np.argmin(np.linalg.norm(arrays[-1]-pose[:2], axis=1)))+1]
            xy = np.concatenate(arrays)
            if len(xy) < 2 or LineString(xy).length < .2:
                continue
            line = LineString(xy)
            if line.project(Point(pose[:2])) < line.project(Point(start.x, start.y))+.1:
                continue
            proposals.append(dict(goal_xy=pose[:2].tolist(), goal_yaw=float(pose[2]),
                lane_tokens=route, line=xy.tolist(), map_length_m=line.length,
                distance_to_predicted_target_m=distance))
    proposals.sort(key=lambda p: (p['map_length_m'], p['distance_to_predicted_target_m'], p['lane_tokens'], p['goal_xy']))
    chosen = []
    for proposal in proposals:
        if any(math.dist(proposal['goal_xy'], p['goal_xy']) < 2. for p in chosen):
            continue
        chosen.append(proposal)
        if len(chosen) == policy['maximum_goals_per_object']:
            break
    return chosen


def timed_rows(states, spec):
    rows = [dict(state=asdict(states[0]), control=None)]
    for before, after in zip(states, states[1:]):
        rows.append(dict(state=asdict(after), control=dict(
            acceleration_mps2=(after.speed_mps-before.speed_mps)/spec.dt_s,
            steering_command_rad=after.steering_rad)))
    return rows


def run_task(method_input, common_map, *, policy=POLICY):
    begun = time.monotonic(); deadline = begun+policy['query_planning_budget_s']
    spec = VehicleSpec(); pose = np.asarray(method_input['rear_axle_reference_world_pose'])
    start = VehicleState(float(pose[0, 3]), float(pose[1, 3]), method_input['rear_axle_yaw_rad'])
    frame = method_input['prediction_frame']; objects = frame['objects']
    result = dict(task_id=method_input['task_id'], scene=method_input['scene'],
        frame_idx=method_input['issue_frame_idx'], status='NO_SEMANTIC_CANDIDATES',
        selected_node_id=None, selected_prediction_index=None, selected=None,
        semantic_candidates=[], planning_attempts=[], predicted_rows=[],
        actual_execution=False, actual_arrival_credit=False, gt_in_inference=False,
        qwen_used=False, output_mode='timed_trajectory', map_sha256=common_map.sha256)
    if not objects:
        result.update(status='EMPTY_PREDICTED_GRAPH', duration_s=time.monotonic()-begun)
        return result
    ranked = semantic_candidates(method_input['instruction'], objects,
        np.asarray(frame['camera_pose']), frame['frame_idx'], policy['semantic_condition'])
    result['semantic_candidates'] = [dict(prediction_index=c.node_id, node_id=objects[c.node_id]['node_id'],
        semantic_rank=i+1, semantic_score=c.score) for i, c in enumerate(ranked)]
    if not ranked:
        result['duration_s'] = time.monotonic()-begun
        return result
    obstacles = [prediction_polygon(o) for o in objects]
    paths = common_map.local_paths(start, policy)
    planned = []
    for rank, candidate in enumerate(ranked[:policy['maximum_objects']]):
        obj = objects[candidate.node_id]
        goals = proposals(obj, paths, common_map, start, obstacles, spec, policy)
        if not goals:
            result['planning_attempts'].append(dict(node_id=obj['node_id'], failure='NO_PREDICTED_SAFE_APPROACH_PROPOSAL'))
        for proposal in goals:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                result['planning_attempts'].append(dict(node_id=obj['node_id'], failure='QUERY_PLANNING_BUDGET'))
                break
            try:
                corridor = MethodCorridor([common_map.lanes[t] for t in proposal['lane_tokens']],
                    common_map.transitions, common_map.road, obstacles)
            except ValueError as error:
                result['planning_attempts'].append(dict(node_id=obj['node_id'],
                    proposal={k:v for k,v in proposal.items() if k != 'line'},
                    failure='MAP_GEOMETRY_UNCERTIFIABLE', detail=str(error)))
                continue
            plan = reference_guided_vehicle(start, proposal['goal_xy'], proposal['goal_yaw'],
                corridor, proposal['line'], spec, budget_s=min(remaining, .5), max_expanded=policy['maximum_expanded'])
            attempt = dict(node_id=obj['node_id'], semantic_rank=rank+1,
                proposal={k:v for k,v in proposal.items() if k != 'line'},
                failure=plan.failure, expanded=plan.expanded, duration_s=plan.duration_s,
                independently_verified=False)
            result['planning_attempts'].append(attempt)
            if not plan.states or plan.failure is not None:
                continue
            states = list(plan.states)
            # 路径的末端通过限幅制动得到，不直接改写速度或坐标；2秒静止段也是预测。
            while states[-1].speed_mps > 0:
                states.append(step_vehicle(states[-1], -spec.max_braking_mps2, states[-1].steering_rad, spec))
            for _ in range(math.ceil(policy['terminal_hold_s']/spec.dt_s)):
                states.append(step_vehicle(states[-1], 0., states[-1].steering_rad, spec))
            audit = audit_executed_states(states, corridor.road, obstacles, spec)
            active = corridor.initial_stages(states[0]); direction_legal = bool(active)
            for before, after in zip(states, states[1:]):
                checked = corridor.validate_step(before, after, active, spec)
                active = set(checked['next_stages'])
                direction_legal &= bool(checked['legal'])
            final = states[-1]
            distance = prediction_polygon(obj).distance(Point(final.x, final.y))
            terminal = (2. <= distance <= 8. and abs(angle_difference(final.yaw_rad, proposal['goal_yaw'])) <= math.radians(5.)
                and final.time_s <= policy['horizon_s'] and len(corridor.lanes)-1 in active)
            attempt.update(independent_sweep=audit, directed_route_verified=direction_legal,
                predicted_terminal_verified=terminal,
                independently_verified=bool(audit['passed'] and direction_legal and terminal))
            if attempt['independently_verified']:
                length = sum(math.dist((a.x,a.y),(b.x,b.y)) for a,b in zip(states,states[1:]))
                planned.append((length, rank, str(obj['node_id']), candidate.node_id, states, proposal))
    if planned:
        length, rank, node_id, index, states, proposal = min(planned, key=lambda r: r[:3])
        result.update(status='VERIFIED_PREDICTED_ROUTE', selected_node_id=node_id,
            selected_prediction_index=index, selected=dict(objects[index]),
            path_length_m=length, predicted_rows=timed_rows(states, spec),
            selected_lane_tokens=proposal['lane_tokens'])
    else:
        result['status'] = 'NO_VERIFIED_PREDICTED_ROUTE'
    result['duration_s'] = time.monotonic()-begun
    return result
