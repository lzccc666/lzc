"""固定有向车道链上的运动规划和控制安全门；不接收语言或对象真值。"""
from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from conceptgraph.completion.execution_direction_audit import DirectedPolyline, certify_arc_direction
from conceptgraph.navigation.vehicle_geometry import (
    VehicleSpec, body_is_legal, step_vehicle, swept_body_polygon,
)
from conceptgraph.navigation.vehicle_planner import tracking_control


class DirectedVehicleCorridor:
    """只沿已声明outgoing前进；交叉或邻近的非连接车道不能授权切换。"""

    def __init__(self, lanes, transitions, drivable, obstacles, *, endpoint_tolerance_m=2.):
        if len(lanes) < 2 or not 0 < endpoint_tolerance_m <= 2:
            raise ValueError('需要至少两条车道及不放宽的端点界')
        self.lanes = lanes
        self.tokens = [lane['token'] for lane in lanes]
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError('本诊断的冻结车道链不得包含回路')
        for source, target in zip(lanes, lanes[1:]):
            if target['token'] not in transitions.get(source['token'], ()):
                raise ValueError('车道链包含未声明的有向连接')
            if source.get('end_pose') is None or target.get('start_pose') is None:
                raise ValueError('连接缺少官方端点')
            if math.dist(source['end_pose'][:2], target['start_pose'][:2]) > endpoint_tolerance_m:
                raise ValueError('连接官方端点距离超界，不插线修补')
        self.polygons = [Polygon(lane['polygon']) for lane in lanes]
        if any(not polygon.is_valid or polygon.is_empty for polygon in self.polygons):
            raise ValueError('无效车道几何，不自动修补')
        self.road = unary_union(self.polygons).intersection(drivable)
        self.obstacles = obstacles
        self.lines = [DirectedPolyline(np.asarray(lane['centerline'])[:, :2]) for lane in lanes]
        self.endpoint_tolerance_m = endpoint_tolerance_m

    def initial_stages(self, state):
        # 从被冻结的入口车道开始；不能利用交叉重叠直接初始化到出口。
        return {0} if self.polygons[0].covers(Point(state.x, state.y)) else set()

    def validate_step(self, before, after, stages, spec=VehicleSpec()):
        if not stages:
            return dict(legal=False, next_stages=[], reason='missing_entry_lane_identity')
        sweep = swept_body_polygon(before, after, spec)
        if not self.road.covers(sweep) or any(sweep.intersects(o) for o in self.obstacles):
            return dict(legal=False, next_stages=[], reason='body_sweep_outside_corridor_or_observed_obstacle')
        a, b = np.array([before.x, before.y]), np.array([after.x, after.y])
        travel = .5*(before.speed_mps+after.speed_mps)*spec.dt_s
        turn = math.atan2(math.sin(after.yaw_rad-before.yaw_rad), math.cos(after.yaw_rad-before.yaw_rad))
        # 圆弧与弦的最大弓高；缓冲覆盖后轴轨迹，不以离散端点代替连续检查。
        sag = travel/abs(turn)*(1-math.cos(turn/2)) if abs(turn) > 1e-10 else 0.
        curve = LineString([a, b]) if np.linalg.norm(b-a) > 1e-12 else Point(a)
        tube = curve.buffer(sag+1e-8)
        possible = set()
        witnesses = []
        for source in sorted(stages):
            if not 0 <= source < len(self.lanes):
                raise ValueError('非法车道状态')
            for target in [source, source+1]:
                if target == len(self.lanes):
                    continue
                if not self.polygons[source].covers(Point(a)) or not self.polygons[target].covers(Point(b)):
                    continue
                changed = source != target
                if changed:
                    end = self.lanes[source]['end_pose'][:2]
                    start = self.lanes[target]['start_pose'][:2]
                    if max(math.dist(p, q) for p in [a, b] for q in [end, start])+sag > self.endpoint_tolerance_m:
                        continue
                region = self.polygons[source] if not changed else unary_union([self.polygons[source], self.polygons[target]])
                if not region.covers(tube):
                    continue
                directional = [certify_arc_direction(self.lines[i], a, before.yaw_rad, turn, travel, max_depth=5)
                               for i in sorted({source, target})]
                if any(d['status'] not in {'certificate_forward', 'not_exercised_stationary'} for d in directional):
                    continue
                possible.add(target)
                witnesses.append(dict(from_stage=source, to_stage=target, source_token=self.tokens[source],
                    target_token=self.tokens[target], declared_connection_checked=changed,
                    rear_arc_tube_margin_m=sag+1e-8))
        return dict(legal=bool(possible), next_stages=sorted(possible), witnesses=witnesses,
                    reason=None if possible else 'no_continuous_forward_connected_lane_assignment')


def reconstruct_lane_assignment(records, final_stage=None):
    """回溯贯穿全程的同一见证；重叠区的多个候选区间不会重复增加过渡数。"""
    if not records:
        return dict(status='not_exercised', lane_stage_path=[], distinct_connections=0)
    possible = records[-1]['lane_transition']['next_stages']
    if not possible or (final_stage is not None and final_stage not in possible):
        return dict(status='unknown_no_complete_assignment', lane_stage_path=[], distinct_connections=None)
    path = [min(possible) if final_stage is None else final_stage]
    for record in reversed(records):
        witnesses = [w for w in record['lane_transition'].get('witnesses', []) if w['to_stage'] == path[-1]]
        if not witnesses:
            return dict(status='unknown_no_complete_assignment', lane_stage_path=[], distinct_connections=None)
        source = min(w['from_stage'] for w in witnesses)
        if not 0 <= path[-1]-source <= 1:
            raise ValueError('车道见证包含跳级或逆向连接')
        path.append(source)
    path.reverse()
    if path[0] != 0:
        return dict(status='unknown_missing_entry_assignment', lane_stage_path=[], distinct_connections=None)
    return dict(status='complete', lane_stage_path=path,
                distinct_connections=sum(a != b for a, b in zip(path, path[1:])))


def execute_corridor(start, path, goal_xy, goal_yaw, corridor, *, spec=VehicleSpec(), maximum_steps=900):
    """实际积分控制；到达只称参考位姿到达，绝不获得正确对象语义到达信用。"""
    states, records = [start], []
    active = corridor.initial_stages(start)
    if not active or not body_is_legal(start, corridor.road, corridor.obstacles, spec, padding_m=0):
        return dict(status='INVALID_INITIAL_BODY_OR_ENTRY_LANE', records=[], states=[asdict(start)],
                    reference_pose_reached=False, actual_semantic_arrival_credit=False)
    if len(path) < 2:
        return dict(status='NO_ROUTE', records=[], states=[asdict(start)],
                    reference_pose_reached=False, actual_semantic_arrival_credit=False)
    state = start
    held = 0.
    status = 'TIMEOUT'
    latched = False
    for index in range(maximum_steps):
        acceleration, steering, tracking = tracking_control(state, path, goal_xy, spec, cruise_speed_mps=1.)
        predicted = step_vehicle(state, acceleration, steering, spec)
        # 提前检查从预测姿态持续制动的全部有限步，保持拓扑状态按分支推进。
        brake, brake_stages = predicted, active.copy()
        first = corridor.validate_step(state, predicted, brake_stages, spec)
        brake_safe = first['legal']
        brake_stages = set(first['next_stages'])
        for _ in range(math.ceil(spec.max_speed_mps/spec.max_braking_mps2/spec.dt_s)+1):
            if not brake_safe or brake.speed_mps == 0:
                break
            nxt = step_vehicle(brake, -spec.max_braking_mps2, brake.steering_rad, spec)
            checked = corridor.validate_step(brake, nxt, brake_stages, spec)
            brake_safe = checked['legal']
            brake_stages = set(checked['next_stages'])
            brake = nxt
        latched |= not brake_safe
        if latched:
            acceleration, steering = -spec.max_braking_mps2, state.steering_rad
            predicted = step_vehicle(state, acceleration, steering, spec)
        checked = corridor.validate_step(state, predicted, active, spec)
        if not checked['legal']:
            # 保存实际执行过的危险结果，不能把拒绝后的状态跳回安全点。
            status = 'EXECUTION_GUARD_VIOLATION'
        active = set(checked['next_stages'])
        state = predicted
        states.append(state)
        yaw_error = abs(math.atan2(math.sin(state.yaw_rad-goal_yaw), math.cos(state.yaw_rad-goal_yaw)))
        inside = math.dist([state.x, state.y], goal_xy) <= .75 and yaw_error <= math.radians(12)
        final_lane = len(corridor.lanes)-1 in active
        stopped = state.speed_mps <= .1
        held = held+spec.dt_s if inside and final_lane and stopped and checked['legal'] else 0.
        records.append(dict(index=index, state=asdict(state), control=dict(acceleration_mps2=acceleration,
            steering_command_rad=steering), tracking=tracking, lane_transition=checked,
            safety_guard_latched=latched, reference_pose_inside=inside, final_lane=final_lane,
            reference_stopped=stopped, reference_held_s=held))
        if status == 'EXECUTION_GUARD_VIOLATION':
            break
        if held >= 2.-1e-8:
            status = 'REFERENCE_POSE_REACHED'
            break
        if latched and state.speed_mps == 0:
            status = 'SAFETY_GUARD_STOPPED'
            break
    assignment = reconstruct_lane_assignment(records, len(corridor.lanes)-1 if status == 'REFERENCE_POSE_REACHED' else None)
    return dict(status=status, records=records, states=[asdict(s) for s in states],
        reference_pose_reached=status == 'REFERENCE_POSE_REACHED', actual_semantic_arrival_credit=False,
        actual_model_calls=0, actual_path_length_m=sum(math.dist([a.x, a.y], [b.x, b.y]) for a, b in zip(states, states[1:])),
        connection_candidate_intervals=sum(any(w['declared_connection_checked'] for w in r['lane_transition'].get('witnesses', [])) for r in records),
        accepted_lane_stage_path=assignment['lane_stage_path'], accepted_distinct_connections=assignment['distinct_connections'],
        connection_assignment_status=assignment['status'],
        final_lane_indices=sorted(active), final_state=asdict(state), goal_complete=False)
