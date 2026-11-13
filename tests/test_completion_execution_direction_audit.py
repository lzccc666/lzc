"""方向证书的逆向、歧义、隐藏区间与缺证据回归；不运行模型。"""
import math

import numpy as np
import pytest

from conceptgraph.completion.execution_direction_audit import (
    DirectedPolyline, arc_position, audit_execution_direction, certify_arc_direction, minimum_cosine,
)


VEHICLE = dict(dt_s=.1, max_speed_mps=2., wheelbase_m=2.7, max_steering_rad=math.radians(32))


def state(x=1., y=0., yaw=0., speed=1., time=0.):
    return dict(x=x, y=y, yaw_rad=yaw, speed_mps=speed, time_s=time)


def test_forward_and_wrong_way_use_actual_motion():
    line = DirectedPolyline([[0, 0], [10, 0]])
    assert certify_arc_direction(line, [1, 0], 0., 0., .1)['status'] == 'certificate_forward'
    wrong = certify_arc_direction(line, [1, 0], math.pi, 0., .1)
    assert wrong['status'] == 'wrong_way_witness'
    assert all(c < 0 for c in wrong['spans'][0]['nearest_cosines'])


def test_analytic_heading_bound_detects_reversal_hidden_from_three_samples():
    line = DirectedPolyline([[-10, 0], [10, 0]])
    assert all(math.cos(4*math.pi*t) > .99 for t in [0, .5, 1])
    report = certify_arc_direction(line, [0, 0], 0., 4*math.pi, 1.)
    assert report['status'] == 'wrong_way_witness' and report['deepest_subdivision'] >= 1


def test_far_opposite_segment_does_not_change_local_lane_direction():
    line = DirectedPolyline([[0, 0], [10, 0], [10, 10], [0, 10]])
    report = certify_arc_direction(line, [1, 0], 0., 0., .1)
    assert report['status'] == 'certificate_forward'
    assert report['spans'][0]['possible_nearest_segments'] == [0]


def test_equally_near_opposite_lanes_remain_unknown():
    line = DirectedPolyline([[0, -1], [10, -1], [10, 1], [0, 1]])
    report = certify_arc_direction(line, [5, 0], 0., 0., .1, max_depth=2)
    assert report['status'] == 'unknown_direction_bound'
    assert report['deepest_subdivision'] == 2


def test_stationary_reversed_heading_does_not_earn_moving_safety_credit():
    before = state(yaw=math.pi, speed=0)
    after = dict(before, time_s=.1)
    report = audit_execution_direction(before, [dict(state=after)], [[0, 0], [10, 0]], VEHICLE)
    assert report['status'] == 'not_exercised_stationary' and report['passed'] is None
    assert report['moving_intervals'] == 0 and report['multi_lane_connection_accuracy'] is None


def test_projection_preserves_all_ties_at_vertices():
    line = DirectedPolyline([[0, 0], [1, 0], [1, 1]])
    indices, progress = line.nearest([1, 0])
    assert list(indices) == [0, 1] and list(progress) == [1., 1.]


def test_missing_initial_and_position_jump_are_unknown():
    report = audit_execution_direction(state(time=.1), [], [[0, 0], [10, 0]], VEHICLE)
    assert report['passed'] is None and report['audited_intervals'] == 0
    report = audit_execution_direction(state(), [dict(state=state(x=3, time=.1))], [[0, 0], [10, 0]], VEHICLE)
    assert report['counts'] == {'unknown_invalid_saved_arc': 1} and report['passed'] is None


def test_heading_wrap_does_not_manufacture_a_full_turn():
    before = state(yaw=math.pi-.001)
    xy = arc_position([1, 0], before['yaw_rad'], .002, .1, 1.)
    after = state(x=float(xy[0]), y=float(xy[1]), yaw=-math.pi+.001, time=.1)
    report = audit_execution_direction(before, [dict(state=after)], [[10, 0], [-10, 0]], VEHICLE)
    assert report['passed'] is True


def test_certified_small_arcs_have_forward_dense_velocity_diagnostics():
    rng = np.random.default_rng(5723)
    line = DirectedPolyline([[-20, 0], [20, 0]])
    for _ in range(60):
        yaw, turn = rng.uniform(-1, 1), rng.uniform(-.5, .5)
        report = certify_arc_direction(line, [0, 0], yaw, turn, .1)
        if report['status'] == 'certificate_forward':
            assert min(math.cos(yaw+turn*t) for t in np.linspace(0, 1, 201)) > 0


@pytest.mark.parametrize('points', [[], [[0, 0]], [[0, 0], [0, 0]], [[0, 0], [math.nan, 1]]])
def test_invalid_centerlines_are_not_silently_repaired(points):
    with pytest.raises(ValueError):
        DirectedPolyline(points)


def test_minimum_cosine_includes_critical_point_and_reversed_bounds():
    assert minimum_cosine(3., 3.3, 0.) == -1.
    assert minimum_cosine(.4, -.4, 0.) == pytest.approx(math.cos(.4))
