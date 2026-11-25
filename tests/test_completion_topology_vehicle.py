import math

import pytest
from shapely.geometry import box

from conceptgraph.completion.topology_vehicle import DirectedVehicleCorridor, reconstruct_lane_assignment
from conceptgraph.navigation.vehicle_geometry import VehicleState


def lane(token, start, end):
    return dict(token=token, polygon=[[start, -2], [end, -2], [end, 2], [start, 2]],
        centerline=[[start, 0, 0], [end, 0, 0]], start_pose=[start, 0, 0], end_pose=[end, 0, 0])


def corridor():
    return DirectedVehicleCorridor([lane('a', -10, 10), lane('b', 9.95, 30)],
                                    {'a': ['b']}, box(-10, -2, 30, 2), [])


def test_near_endpoint_forward_transition_has_explicit_token_witness():
    c = corridor()
    before = VehicleState(9.9, 0, 0, 1)
    after = VehicleState(10, 0, 0, 1, time_s=.1)
    result = c.validate_step(before, after, {0})
    assert result['legal'] and 1 in result['next_stages']
    assert any(w['source_token'] == 'a' and w['target_token'] == 'b' for w in result['witnesses'])


def test_disconnected_crossing_and_missing_endpoints_cannot_authorize_a_chain():
    with pytest.raises(ValueError, match='未声明'):
        DirectedVehicleCorridor([lane('a', -10, 10), lane('b', 10, 30)], {}, box(-10, -2, 30, 2), [])
    second = lane('b', 10, 30)
    second['start_pose'] = None
    with pytest.raises(ValueError, match='端点'):
        DirectedVehicleCorridor([lane('a', -10, 10), second], {'a': ['b']}, box(-10, -2, 30, 2), [])


def test_polygon_overlap_far_from_connection_does_not_allow_switch():
    a, b = lane('a', -10, 10), lane('b', -10, 30)
    b['start_pose'] = [10, 0, 0]
    c = DirectedVehicleCorridor([a, b], {'a': ['b']}, box(-10, -2, 30, 2), [])
    result = c.validate_step(VehicleState(0, 0, 0, 1), VehicleState(.1, 0, 0, 1, time_s=.1), {0})
    assert result['legal'] and result['next_stages'] == [0]


def test_reversed_motion_and_outside_body_are_rejected():
    c = corridor()
    result = c.validate_step(VehicleState(0, 0, math.pi, 1), VehicleState(-.1, 0, math.pi, 1, time_s=.1), {0})
    assert not result['legal']
    result = c.validate_step(VehicleState(0, 1.9, 0, 1), VehicleState(.1, 1.9, 0, 1, time_s=.1), {0})
    assert not result['legal'] and 'body_sweep' in result['reason']


def test_later_lane_cannot_be_used_as_initial_entry_identity():
    c = corridor()
    assert not c.initial_stages(VehicleState(20, 0, 0))
    assert not c.validate_step(VehicleState(0, 0, 0), VehicleState(0, 0, 0, time_s=.1), set())['legal']


def test_overlap_candidates_count_as_one_connection_in_complete_assignment():
    def row(witnesses):
        return dict(lane_transition=dict(next_stages=sorted({b for _, b in witnesses}),
            witnesses=[dict(from_stage=a, to_stage=b) for a, b in witnesses]))
    records = [row([(0, 0), (0, 1)]) for _ in range(20)]+[row([(1, 1), (1, 2)])]
    assignment = reconstruct_lane_assignment(records, final_stage=2)
    assert assignment['status'] == 'complete'
    assert len(assignment['lane_stage_path']) == 22 and assignment['distinct_connections'] == 2
    assert sum(a != b for a, b in zip(assignment['lane_stage_path'], assignment['lane_stage_path'][1:])) == 2


def test_assignment_cannot_be_recovered_from_disconnected_witnesses():
    records = [dict(lane_transition=dict(next_stages=[2], witnesses=[dict(from_stage=1, to_stage=2)]))]
    result = reconstruct_lane_assignment(records, final_stage=2)
    assert result['status'] == 'unknown_missing_entry_assignment' and result['distinct_connections'] is None
