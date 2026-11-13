"""对保存的车辆积分圆弧审核有向车道方向，不调用规划器或其成功标志。"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

POSITION_TOLERANCE_M = 1e-6
TIME_TOLERANCE_S = 1e-7
DISTANCE_PADDING_M = 1e-8
COSINE_MARGIN = 1e-10
MOTION_TOLERANCE_M = 1e-12


class DirectedPolyline:
    """保持输入顺序的有向线段链；不以投影函数任取一个并列车道。"""

    def __init__(self, points):
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError('中心线必须包含至少两个有限二维点')
        self.starts = values[:-1]
        self.vectors = np.diff(values, axis=0)
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        if (self.lengths <= 1e-12).any():
            raise ValueError('中心线含退化线段，不能静默修改冻结拓扑')
        self.units = self.vectors/self.lengths[:, None]
        self.headings = np.arctan2(self.units[:, 1], self.units[:, 0])
        self.prefix = np.r_[0., np.cumsum(self.lengths)]

    def projection(self, point):
        offset = np.asarray(point)-self.starts
        fractions = np.clip(np.sum(offset*self.vectors, axis=1)/self.lengths**2, 0., 1.)
        displacement = offset-fractions[:, None]*self.vectors
        distances = np.linalg.norm(displacement, axis=1)
        return distances, self.prefix[:-1]+fractions*self.lengths

    def nearest(self, point):
        distances, progress = self.projection(point)
        indices = np.flatnonzero(distances <= float(distances.min())+DISTANCE_PADDING_M)
        return indices, progress[indices]

    def possible_nearest(self, point, radius):
        # 距离函数是1-Lipschitz。圆弧任意点距中点≤radius；若某段在其中
        # 成为最近段，则它在中点的距离至多为最近距离+2*radius。
        distances, _ = self.projection(point)
        return np.flatnonzero(distances <= float(distances.min())+2*radius+DISTANCE_PADDING_M)


def minimum_cosine(low, high, reference):
    """闭合航向区间与固定有向切线的最小点积，包括区间内的反向极值。"""
    low, high = min(low, high)-reference, max(low, high)-reference
    if math.ceil((low-math.pi)/(2*math.pi)) <= math.floor((high-math.pi)/(2*math.pi)):
        return -1.
    return min(math.cos(low), math.cos(high))


def arc_position(start_xy, yaw, turn, travel, fraction):
    angle = turn*fraction
    half = angle/2
    scale = travel*fraction*(math.sin(half)/half if abs(half) > 1e-12 else 1.)
    return np.asarray(start_xy)+scale*np.array([math.cos(yaw+half), math.sin(yaw+half)])


def certify_arc_direction(line, start_xy, yaw, turn, travel, *, max_depth=12):
    """保守覆盖整个圆弧的速度方向；未决不算安全，停留不算运动证据。"""
    values = [*start_xy, yaw, turn, travel]
    if not all(math.isfinite(v) for v in values) or travel < 0:
        raise ValueError('非法圆弧输入')
    if type(max_depth) is not int or not 0 <= max_depth <= 20:
        raise ValueError('非法细分预算')
    if travel <= MOTION_TOLERANCE_M:
        return dict(status='not_exercised_stationary', moving=False, deepest_subdivision=0,
                    terminal_spans=0, minimum_cosine_lower_bound=None)
    leaves = []

    def check(a, b, depth):
        middle = (a+b)/2
        point = arc_position(start_xy, yaw, turn, travel, middle)
        radius = travel*(b-a)/2+DISTANCE_PADDING_M
        possible = line.possible_nearest(point, radius)
        low = yaw+turn*a
        high = yaw+turn*b
        lower = min(minimum_cosine(low, high, line.headings[i]) for i in possible)
        entry = dict(fraction_start=a, fraction_end=b, depth=depth,
                     possible_nearest_segments=possible.tolist(), cosine_lower_bound=lower)
        if lower > COSINE_MARGIN:
            entry['status'] = 'certificate_forward'
            leaves.append(entry)
            return
        nearest, _ = line.nearest(point)
        actual = [math.cos(yaw+turn*middle-line.headings[i]) for i in nearest]
        # 最近段存在歧义时，只有所有并列段都反向才提供确定的违规见证。
        if max(actual) < -COSINE_MARGIN:
            entry.update(status='wrong_way_witness', witness_fraction=middle,
                         witness_xy=point.tolist(), nearest_segments=nearest.tolist(), nearest_cosines=actual)
            leaves.append(entry)
            return
        if depth == max_depth:
            entry['status'] = 'unknown_direction_bound'
            leaves.append(entry)
            return
        check(a, middle, depth+1)
        check(middle, b, depth+1)

    check(0., 1., 0)
    counts = Counter(e['status'] for e in leaves)
    status = ('wrong_way_witness' if counts['wrong_way_witness'] else
              'unknown_direction_bound' if counts['unknown_direction_bound'] else 'certificate_forward')
    return dict(status=status, moving=True, spans=leaves, terminal_spans=len(leaves),
                deepest_subdivision=max(e['depth'] for e in leaves),
                minimum_cosine_lower_bound=min(e['cosine_lower_bound'] for e in leaves))


def _finite_state(state):
    fields = ['x', 'y', 'yaw_rad', 'speed_mps', 'time_s']
    if not isinstance(state, dict) or any(type(state.get(k)) not in {int, float}
            or not math.isfinite(state[k]) for k in fields):
        raise ValueError('执行状态缺失或包含非有限数值')
    if state['speed_mps'] < 0:
        raise ValueError('冻结车辆模型不允许负速度')


def audit_execution_direction(initial_state, records, centerline, vehicle, *, max_depth=12):
    """基于实际初始观测和保存状态，不从seed重造初始姿态。"""
    _finite_state(initial_state)
    line = DirectedPolyline(centerline)
    if initial_state['time_s'] != 0:
        return dict(status='unknown_missing_initial_observation', passed=None, audited_intervals=0,
                    moving_intervals=0, intervals=[], multi_lane_connection_accuracy=None)
    dt = vehicle['dt_s']
    maximum_speed = vehicle['max_speed_mps']
    minimum_radius = vehicle['wheelbase_m']/math.tan(vehicle['max_steering_rad'])
    if not all(math.isfinite(v) and v > 0 for v in [dt, maximum_speed, minimum_radius]):
        raise ValueError('非法车辆参数')
    if maximum_speed*dt/minimum_radius >= math.pi/2:
        raise ValueError('单步转角可能有解包歧义，须先细化保存的状态')
    before = initial_state
    output = []
    for index, row in enumerate(records):
        after = row['state']
        _finite_state(after)
        entry = dict(index=index)
        elapsed = after['time_s']-before['time_s']
        travel = (before['speed_mps']+after['speed_mps'])/2*dt
        turn = math.atan2(math.sin(after['yaw_rad']-before['yaw_rad']), math.cos(after['yaw_rad']-before['yaw_rad']))
        start = [before['x'], before['y']]
        expected = arc_position(start, before['yaw_rad'], turn, travel, 1.)
        residual = math.dist(expected, [after['x'], after['y']])
        entry.update(travel_m=travel, heading_change_rad=turn, endpoint_arc_residual_m=residual,
                     dt_residual_s=abs(elapsed-dt))
        if (abs(elapsed-dt) > TIME_TOLERANCE_S or residual > POSITION_TOLERANCE_M
                or max(before['speed_mps'], after['speed_mps']) > maximum_speed+1e-8
                or abs(turn) > travel/minimum_radius+1e-7):
            entry.update(status='unknown_invalid_saved_arc', moving=travel > MOTION_TOLERANCE_M)
        else:
            checked = certify_arc_direction(line, start, before['yaw_rad'], turn, travel, max_depth=max_depth)
            # 独立端点进度诊断也保存全部并列投影的范围，不任取有利投影。
            _, previous_s = line.nearest(start)
            _, current_s = line.nearest([after['x'], after['y']])
            entry.update(checked, progress_before_min_m=float(min(previous_s)),
                progress_before_max_m=float(max(previous_s)), progress_after_min_m=float(min(current_s)),
                progress_after_max_m=float(max(current_s)),
                endpoint_progress_delta_lower_m=float(min(current_s)-max(previous_s)))
            if entry['status'] == 'certificate_forward' and entry['endpoint_progress_delta_lower_m'] < -POSITION_TOLERANCE_M:
                entry['status'] = 'unknown_projection_inconsistency'
        output.append(entry)
        before = after
    counts = Counter(e['status'] for e in output)
    moving = sum(e['moving'] for e in output)
    unknown = sum(n for name, n in counts.items() if name.startswith('unknown_'))
    status = ('wrong_way_witness' if counts['wrong_way_witness'] else
              'unknown_incomplete_direction_evidence' if unknown else
              'certificate_forward' if moving else 'not_exercised_stationary')
    return dict(status=status, passed=True if status == 'certificate_forward' else
                False if status == 'wrong_way_witness' else None,
                counts=dict(counts), audited_intervals=len(output), moving_intervals=moving, intervals=output,
                max_endpoint_arc_residual_m=max((e['endpoint_arc_residual_m'] for e in output), default=None),
                geometry='piecewise_constant_average_speed_and_steering_saved_model_arc',
                position_tolerance_m=POSITION_TOLERANCE_M, distance_padding_m=DISTANCE_PADDING_M,
                cosine_margin=COSINE_MARGIN, maximum_subdivision_depth=max_depth,
                lane_containment_checked_here=False, dynamic_safety_claim=False,
                multi_lane_connection_accuracy=None, connection_status='not_exercised_single_directed_lane',
                new_arrivals=0, goal_complete=False)
