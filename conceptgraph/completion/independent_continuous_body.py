"""从保存控制独立重算圆弧，以速度界包络审核连续矩形车身。

不调用项目原车辆积分、车身、扫掠或规划器；结论限于冻结静态场景。
"""
from __future__ import annotations

from collections import Counter
import cmath
import math

from shapely.geometry import Polygon

STATE_FIELDS = ('x', 'y', 'yaw_rad', 'speed_mps', 'steering_rad', 'time_s')
SPEC_FIELDS = ('wheelbase_m', 'front_from_rear_axle_m', 'rear_overhang_m', 'width_m',
               'max_speed_mps', 'max_acceleration_mps2', 'max_braking_mps2',
               'max_steering_rad', 'max_steering_rate_radps', 'dt_s')
TOLERANCES = dict(x=1e-6, y=1e-6, yaw_rad=1e-7, speed_mps=1e-8, steering_rad=1e-8, time_s=1e-7)
NUMERICAL_PADDING_M = 2e-6


def validate_spec(spec):
    if any(not math.isfinite(spec[k]) or spec[k] <= 0 for k in SPEC_FIELDS):
        raise ValueError('车辆参数必须有限且为正')
    if spec['max_steering_rad'] >= math.pi/2 or spec['wheelbase_m'] >= spec['front_from_rear_axle_m']:
        raise ValueError('车辆几何或转角参数无效')


def validate_state(state, spec):
    if any(not math.isfinite(state[k]) for k in STATE_FIELDS):
        raise ValueError('保存状态有非有限值')
    if not 0 <= state['speed_mps'] <= spec['max_speed_mps']+1e-8:
        raise ValueError('保存速度越界')
    if abs(state['steering_rad']) > spec['max_steering_rad']+1e-8:
        raise ValueError('保存转角越界')


def clip(value, low, high):
    return min(high, max(low, value))


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def pose_on_arc(before, velocity, omega, elapsed_s):
    """复数中点sinc积分，避免原实现的两个正弦相减。"""
    half_angle = omega*elapsed_s/2
    sinc = math.sin(half_angle)/half_angle if half_angle else 1.0
    displacement = velocity*elapsed_s*sinc*cmath.exp(1j*(before['yaw_rad']+half_angle))
    return dict(x=before['x']+displacement.real, y=before['y']+displacement.imag,
                yaw_rad=wrap(before['yaw_rad']+2*half_angle))


def replay_control(before, control, spec):
    validate_spec(spec)
    validate_state(before, spec)
    acceleration, target = control['acceleration_mps2'], control['steering_command_rad']
    if not math.isfinite(acceleration) or not math.isfinite(target):
        raise ValueError('控制命令有非有限值')
    dt = spec['dt_s']
    speed = clip(before['speed_mps']+clip(acceleration, -spec['max_braking_mps2'],
                  spec['max_acceleration_mps2'])*dt, 0, spec['max_speed_mps'])
    target = clip(target, -spec['max_steering_rad'], spec['max_steering_rad'])
    step = spec['max_steering_rate_radps']*dt
    steering = before['steering_rad']+clip(target-before['steering_rad'], -step, step)
    velocity = (before['speed_mps']+speed)/2
    omega = velocity*math.tan((before['steering_rad']+steering)/2)/spec['wheelbase_m']
    state = dict(**pose_on_arc(before, velocity, omega, dt), speed_mps=speed,
                 steering_rad=steering, time_s=before['time_s']+dt)
    return state, dict(velocity_mps=velocity, angular_velocity_radps=omega, dt_s=dt)


def rectangle(center_xy, yaw, front, rear, width):
    c, s = math.cos(yaw), math.sin(yaw)
    points = [(center_xy[0]+u*c-v*s, center_xy[1]+u*s+v*c)
              for u, v in [(-rear, -width/2), (front, -width/2), (front, width/2), (-rear, width/2)]]
    return Polygon(points)


def body(pose, spec):
    return rectangle((pose['x'], pose['y']), pose['yaw_rad'], spec['front_from_rear_axle_m'],
                     spec['rear_overhang_m'], spec['width_m'])


def enclosure(before, motion, spec, left, right):
    middle = (left+right)/2
    pose = pose_on_arc(before, motion['velocity_mps'], motion['angular_velocity_radps'], middle)
    radius = math.hypot(max(spec['front_from_rear_axle_m'], spec['rear_overhang_m']), spec['width_m']/2)
    speed_bound = motion['velocity_mps']+radius*abs(motion['angular_velocity_radps'])
    deviation = speed_bound*(right-left)/2+NUMERICAL_PADDING_M
    # 对凸矩形的miter外偏移包含圆盘Minkowski和，无圆弧buffer的内接多边形问题。
    polygon = body(pose, spec)
    return polygon.buffer(deviation, join_style=2), polygon, pose, deviation


def illegal(polygon, road, obstacles):
    return (not road.covers(polygon), [i for i, obstacle in enumerate(obstacles) if polygon.intersects(obstacle)])


def certify_interval(before, after, motion, spec, road, obstacles, max_depth=12):
    """包络只能证安全；实际时刻车身才提供违例见证，未决不记安全。"""
    if type(max_depth) is not int or not 0 <= max_depth <= 16:
        raise ValueError('二分深度必须是0到16的整数')
    evaluations, deepest, unresolved = 0, 0, []
    for fraction, pose in [(0., before), (1., after)]:
        polygon = body(pose, spec)
        outside, touched = illegal(polygon, road, obstacles)
        if outside or touched:
            return dict(status='witness_violation', envelope_evaluations=0, deepest_subdivision=0,
                        witness=dict(interval_fraction=fraction, pose={k: pose[k] for k in ('x', 'y', 'yaw_rad')},
                                     vertices_xy=list(polygon.exterior.coords), outside_road=outside,
                                     obstacle_indices=touched, source='saved_endpoint'))
    pending = [(0., motion['dt_s'], 0)]
    while pending:
        left, right, depth = pending.pop()
        evaluations += 1
        deepest = max(deepest, depth)
        envelope, polygon, pose, deviation = enclosure(before, motion, spec, left, right)
        outside, touched = illegal(envelope, road, obstacles)
        if not outside and not touched:
            continue
        outside_body, touched_body = illegal(polygon, road, obstacles)
        if outside_body or touched_body:
            return dict(status='witness_violation', envelope_evaluations=evaluations, deepest_subdivision=deepest,
                        witness=dict(interval_fraction=(left+right)/(2*motion['dt_s']), pose=pose,
                                     vertices_xy=list(polygon.exterior.coords), outside_road=outside_body,
                                     obstacle_indices=touched_body, source='recomputed_arc_midpoint'))
        if depth == max_depth or motion['velocity_mps'] == 0:
            unresolved.append(dict(left_s=left, right_s=right, deviation_bound_m=deviation,
                                   envelope_outside_road=outside, envelope_obstacle_indices=touched))
            continue
        middle = (left+right)/2
        pending.extend([(middle, right, depth+1), (left, middle, depth+1)])
    return dict(status='unresolved' if unresolved else 'certificate_safe', envelope_evaluations=evaluations,
                deepest_subdivision=deepest, unresolved_leaf_count=len(unresolved), unresolved_examples=unresolved[:8])


def audit_trajectory(initial_state, records, spec, road, obstacles, *, max_depth=12):
    validate_spec(spec)
    validate_state(initial_state, spec)
    if not road.is_valid or road.is_empty or any(not o.is_valid or o.is_empty for o in obstacles):
        raise ValueError('冻结几何无效，不修补或删除')
    before = initial_state
    counts, intervals = Counter(), []
    maxima = dict.fromkeys(STATE_FIELDS, 0.)
    geometry_cache = {}
    for index, record in enumerate(records):
        row = dict(index=index, start_time_s=before.get('time_s') if isinstance(before, dict) else None)
        try:
            after = record['state']
            validate_state(after, spec)
            predicted, motion = replay_control(before, record['control'], spec)
            residual = {k: abs(wrap(after[k]-predicted[k])) if k == 'yaw_rad'
                        else abs(after[k]-predicted[k]) for k in STATE_FIELDS}
            for key, value in residual.items():
                maxima[key] = max(maxima[key], value)
            row['state_residuals'] = residual
            invalid = [key for key, value in residual.items() if value > TOLERANCES[key]]
            if invalid:
                row.update(status='invalid_control_replay', mismatched_fields=invalid)
            else:
                key = tuple(before[k] for k in ('x', 'y', 'yaw_rad'))+tuple(after[k] for k in ('x', 'y', 'yaw_rad'))+(
                    motion['velocity_mps'], motion['angular_velocity_radps'], motion['dt_s'])
                if key in geometry_cache:
                    result = geometry_cache[key]
                    counts['geometry_cache_hits'] += 1
                else:
                    result = certify_interval(before, after, motion, spec, road, obstacles, max_depth)
                    geometry_cache[key] = result
                    counts['actual_envelope_evaluations'] += result['envelope_evaluations']
                row.update(result)
        except (ValueError, TypeError, KeyError) as exc:
            row.update(status='invalid_record', error=str(exc))
        counts[row['status']] += 1
        intervals.append(row)
        before = record.get('state', {}) if isinstance(record, dict) else {}
    if not records:
        state_illegal = illegal(body(initial_state, spec), road, obstacles)
        status = 'no_executed_intervals'
        initial_witness = dict(outside_road=state_illegal[0], obstacle_indices=state_illegal[1])
    else:
        status = ('invalid_control_replay' if counts['invalid_record']+counts['invalid_control_replay'] else
                  'witness_violation' if counts['witness_violation'] else
                  'unresolved' if counts['unresolved'] else 'certificate_safe')
        initial_witness = None
    return dict(status=status, passed=status == 'certificate_safe', counts=dict(counts),
                intervals=intervals, audited_intervals=len(records), max_state_residuals=maxima,
                initial_only_geometry=initial_witness, numerical_padding_m=NUMERICAL_PADDING_M,
                tolerances=TOLERANCES.copy(), maximum_subdivision_depth=max_depth,
                new_execution=False, arrival_credit=0, dynamic_safety_claim=False, goal_complete=False)
