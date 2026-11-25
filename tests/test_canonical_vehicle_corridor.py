import copy

import pytest
from shapely.geometry import box

from conceptgraph.completion.canonical_vehicle_corridor import canonical_lane, canonical_vehicle_corridor
from conceptgraph.completion.topology_vehicle import DirectedVehicleCorridor
from conceptgraph.navigation.vehicle_geometry import VehicleState


def lane(token='a', points=None, start=-10, end=10):
    return dict(token=token, centerline=points if points is not None else [[start, 0, 0], [end, 0, 0]],
                polygon=list(box(start, -2, end, 2).exterior.coords), start_pose=[start, 0, 0], end_pose=[end, 0, 0])


def test_exact_adjacent_duplicate_is_explicitly_removed_without_mutating_input():
    original = lane(points=[[-10, 0, 0], [0, 0, 0], [0, 0, .1], [10, 0, 0]])
    saved = copy.deepcopy(original)
    result, receipt = canonical_lane(original)
    assert original == saved and result['centerline'] == [[-10, 0, 0], [0, 0, 0], [10, 0, 0]]
    assert receipt['removed_original_indices'] == [2] and receipt['geometric_curve_equal']
    assert receipt['original_length_m'] == receipt['canonical_length_m'] == 20
    assert all(result[key] == original[key] for key in ['token', 'polygon', 'start_pose', 'end_pose'])


def test_nonconsecutive_return_is_preserved_as_a_real_directed_curve():
    original = lane(points=[[0, 0], [1, 0], [0, 0], [2, 0]])
    result, receipt = canonical_lane(original)
    assert result['centerline'] == original['centerline'] and receipt['removed_exact_zero_segments'] == 0


def test_nonzero_short_segment_is_kept_and_strict_direction_validation_still_rejects_it():
    a = lane(points=[[-10, 0], [0, 0], [1e-12, 0], [10, 0]])
    b = lane('b', start=9.9, end=30)
    result, receipt = canonical_lane(a)
    assert result['centerline'] == a['centerline'] and receipt['nonzero_short_segments_removed'] == 0
    with pytest.raises(ValueError, match='退化'):
        canonical_vehicle_corridor([a, b], {'a': ['b']}, box(-10, -2, 30, 2), [])


@pytest.mark.parametrize('points', [[[0, 0], [0, 0]], [[0, 0], [float('nan'), 0]]])
def test_undefined_direction_is_never_repaired_by_inventing_an_edge(points):
    with pytest.raises(ValueError, match='中心线'):
        canonical_lane(lane(points=points))


def test_explicit_canonicalization_enables_same_declared_connection_without_relaxing_its_gate():
    a, b = lane(points=[[-10, 0], [10, 0], [10, 0]]), lane('b', start=9.9, end=30)
    with pytest.raises(ValueError, match='退化'):
        DirectedVehicleCorridor([a, b], {'a': ['b']}, box(-10, -2, 30, 2), [])
    corridor, receipts = canonical_vehicle_corridor([a, b], {'a': ['b']}, box(-10, -2, 30, 2), [])
    checked = corridor.validate_step(VehicleState(9.95, 0, 0, 1), VehicleState(10.05, 0, 0, 1, time_s=.1), {0})
    assert checked['legal'] and 1 in checked['next_stages'] and receipts[0]['removed_exact_zero_segments'] == 1
    with pytest.raises(ValueError, match='未声明'):
        canonical_vehicle_corridor([a, b], {}, box(-10, -2, 30, 2), [])
