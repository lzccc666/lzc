import math

import pytest
from shapely.geometry import box

from conceptgraph.completion.independent_continuous_body import (
    audit_trajectory, body, enclosure, pose_on_arc, replay_control,
)


def specification():
    return dict(wheelbase_m=.8, front_from_rear_axle_m=1., rear_overhang_m=.2, width_m=.4,
                max_speed_mps=8., max_acceleration_mps2=1., max_braking_mps2=2.,
                max_steering_rad=math.radians(60), max_steering_rate_radps=1., dt_s=1.)


def state(speed=4., steering=0.):
    return dict(x=0., y=0., yaw_rad=0., speed_mps=speed, steering_rad=steering, time_s=0.)


def test_straight_saved_command_and_between_endpoint_collision():
    spec = specification()
    initial = state()
    final = dict(x=4., y=0., yaw_rad=0., speed_mps=4., steering_rad=0., time_s=1.)
    obstacle = box(2., -.1, 2.1, .1)
    assert not body(initial, spec).intersects(obstacle) and not body(final, spec).intersects(obstacle)
    records = [dict(state=final, control=dict(acceleration_mps2=0., steering_command_rad=0.))]
    audit = audit_trajectory(initial, records, spec, box(-10, -10, 10, 10), [obstacle])
    assert audit['status'] == 'witness_violation' and audit['audited_intervals'] == 1
    assert 0 < audit['intervals'][0]['witness']['interval_fraction'] < 1


def test_clear_road_gets_certificate_and_clipping_has_expected_values():
    spec = specification()
    initial = state(speed=0.)
    command = dict(acceleration_mps2=100., steering_command_rad=100.)
    final, _ = replay_control(initial, command, spec)
    assert final['speed_mps'] == 1. and final['steering_rad'] == 1.
    audit = audit_trajectory(initial, [dict(state=final, control=command)], spec, box(-20, -20, 20, 20), [])
    assert audit['passed'] and audit['arrival_credit'] == 0 and not audit['new_execution']


def test_arc_endpoint_matches_known_quarter_circle():
    spec = specification()
    spec['dt_s'] = math.pi
    initial = state(speed=1., steering=math.atan(.4))  # r=2m
    final, _ = replay_control(initial, dict(acceleration_mps2=0., steering_command_rad=initial['steering_rad']), spec)
    assert (final['x'], final['y'], final['yaw_rad']) == pytest.approx((2., 2., math.pi/2))


def test_lipschitz_enclosure_contains_dense_turning_rectangles():
    spec = specification()
    spec['rear_overhang_m'] = 1.4  # 最远角点可能在后方
    initial = state(speed=2., steering=.5)
    _, motion = replay_control(initial, dict(acceleration_mps2=0., steering_command_rad=.5), spec)
    envelope, _, _, _ = enclosure(initial, motion, spec, .1, .9)
    for k in range(101):
        pose = pose_on_arc(initial, motion['velocity_mps'], motion['angular_velocity_radps'], .1+.8*k/100)
        assert envelope.covers(body(pose, spec))


def test_coarse_enclosure_intersection_is_not_claimed_collision():
    spec = specification()
    initial = state(speed=4.)
    final = dict(x=4., y=0., yaw_rad=0., speed_mps=4., steering_rad=0., time_s=1.)
    record = dict(state=final, control=dict(acceleration_mps2=0., steering_command_rad=0.))
    road = box(-.3, -.3, 5.1, .3)
    audit = audit_trajectory(initial, [record], spec, road, [], max_depth=0)
    assert audit['status'] == 'unresolved' and not audit['passed']
    refined = audit_trajectory(initial, [record], spec, road, [], max_depth=12)
    assert refined['passed'] and refined['intervals'][0]['deepest_subdivision'] > 0


def test_stationary_boundary_numerical_uncertainty_not_safe():
    spec = specification()
    initial = state(speed=0.)
    final = dict(initial, time_s=1.)
    audit = audit_trajectory(initial, [dict(state=final, control=dict(acceleration_mps2=0., steering_command_rad=0.))],
                             spec, box(-.2, -.2, 1., .2), [])
    assert audit['status'] == 'unresolved' and not audit['passed']


@pytest.mark.parametrize('field', ['x', 'yaw_rad', 'time_s', 'speed_mps', 'steering_rad'])
def test_tampered_state_does_not_get_geometry_certificate(field):
    spec = specification()
    initial = state(speed=0.)
    final = dict(initial, time_s=1.)
    final[field] += .01
    audit = audit_trajectory(initial, [dict(state=final, control=dict(acceleration_mps2=0., steering_command_rad=0.))],
                             spec, box(-20, -20, 20, 20), [])
    assert audit['status'] == 'invalid_control_replay' and field in audit['intervals'][0]['mismatched_fields']


def test_invalid_records_and_missing_execution_preserve_unknown():
    spec = specification()
    initial = state(speed=0.)
    records = [dict(state=dict(initial, time_s=1.), control=dict(acceleration_mps2=float('nan'), steering_command_rad=0.)),
               dict(state=dict(initial, time_s=2.)), None]
    audit = audit_trajectory(initial, records, spec, box(-20, -20, 20, 20), [])
    assert audit['status'] == 'invalid_control_replay' and len(audit['intervals']) == 3
    assert audit['counts']['invalid_record'] == 3
    empty = audit_trajectory(initial, [], spec, box(-20, -20, 20, 20), [])
    assert empty['status'] == 'no_executed_intervals' and not empty['passed']
