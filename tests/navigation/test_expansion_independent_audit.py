from types import SimpleNamespace

import numpy as np

from conceptgraph.scripts.lane_expansion_independent_audit import IndependentLaneAudit, continuous_collisions, nearest_heading_breaks


def test_external_collision_oracle_detects_closed_cell_corner_contact():
    occupancy = np.zeros((4, 4), dtype=np.uint8)
    occupancy[0, 1] = 1
    assert (0, 1) in continuous_collisions([[0, 0], [1, 1]], occupancy)
    assert not continuous_collisions([[2, 0], [2, 3]], occupancy)


def test_exact_nearest_heading_partition_matches_dense_distance_oracle():
    rng = np.random.default_rng(4)
    centers = rng.normal(size=(40, 3))
    a, b = np.array([-2.0, 0.1]), np.array([2.0, 0.7])
    breaks = sorted([0.0, *nearest_heading_breaks(a, b, centers), 1.0])
    for left, right in zip(breaks[:-1], breaks[1:]):
        parameters = left+(right-left)*np.linspace(0.001, 0.999, 25)
        points = a+parameters[:, None]*(b-a)
        nearest = np.argmin(np.sum((points[:, None, :]-centers[None, :, :2])**2, axis=2), axis=1)
        assert len(set(nearest)) == 1


def _topology(gap=0.0, transitions=True):
    geometries = {}
    for token, lo, hi in [('a', 0.0, 1.0), ('b', 1.0+gap, 2.0)]:
        geometries[token] = SimpleNamespace(polygon=np.array([[lo, 0], [hi, 0], [hi, 1], [lo, 1]]), centerline=np.array([[lo, 0.5, 0], [hi, 0.5, 0]]), start_pose=(lo, 0.5, 0), end_pose=(hi, 0.5, 0))
    return SimpleNamespace(geometries=geometries, transitions={'a': ['b']} if transitions else {}, diagnostics={})


def test_external_lane_oracle_requires_direction_and_directed_transition():
    points = [[0.2, 0.5], [1.8, 0.5]]
    assert IndependentLaneAudit(_topology()).validate(points)['valid']
    assert not IndependentLaneAudit(_topology(transitions=False)).validate(points)['valid']
    assert not IndependentLaneAudit(_topology()).validate(points[::-1])['valid']


def test_external_lane_oracle_detects_subcell_polygon_gap():
    result = IndependentLaneAudit(_topology(gap=0.01)).validate([[0.2, 0.5], [1.8, 0.5]])
    assert not result['valid'] and result['reason'] == 'polygon_segment_gap'
