from __future__ import annotations

import json

import numpy as np
import pytest

from conceptgraph.navigation.lane_constraints import (
    LaneGeometry,
    LaneTopology,
    lane_astar_plan,
    load_lane_topology,
    project_to_legal_lane,
    validate_lane_path,
)
from conceptgraph.navigation.planner import OccupancyGrid2D
from conceptgraph.scripts.build_lane_constrained_bev import (
    _sample_arcline,
    build_lane_map,
)


def _grid():
    return OccupancyGrid2D(
        np.zeros((5, 9), dtype=np.uint8), 1.0, np.zeros(2), np.ones((5, 9))
    )


def _topology(connected=True):
    ids = np.full((5, 9), -1, dtype=np.int32)
    ids[2, :5], ids[2, 5:] = 0, 1
    headings = np.where(ids >= 0, 0.0, np.nan)
    geometries = {}
    for token, lo, hi in (("a", 0, 4.4), ("b", 3.6, 8)):
        polygon = np.array([[lo, 1.6], [hi, 1.6], [hi, 2.4], [lo, 2.4]])
        center = np.column_stack(
            [np.linspace(lo, hi, 20), np.full(20, 2), np.zeros(20)]
        )
        geometries[token] = LaneGeometry(
            token, "lane", polygon, center, 0.0, (lo, 2, 0), (hi, 2, 0)
        )
    return LaneTopology(
        ids,
        headings,
        np.zeros_like(headings),
        ["a", "b"],
        ["lane", "lane"],
        {"a": ("b",) if connected else (), "b": ()},
        geometries,
    )


def test_overlapping_lane_memberships_and_legal_forward_transition():
    topology = _topology()
    assert topology.lanes_at((2, 4)) == [0, 1]
    result = lane_astar_plan(
        _grid(), topology, np.array([0.0, 2.0]), np.array([8.0, 2.0])
    )
    assert result.failure_code is None
    assert result.validation["valid"]
    assert result.validation["lane_trace"] == ["a", "b"]


def test_reverse_path_and_opposite_vehicle_heading_rejected():
    topology = _topology()
    result = validate_lane_path([(2, 8), (2, 0)], topology, _grid().occupancy)
    assert not result["valid"] and result["reverse_count"] > 0
    result = lane_astar_plan(
        _grid(),
        topology,
        np.array([0.0, 2.0]),
        np.array([8.0, 2.0]),
        start_heading_xy=np.array([-1.0, 0.0]),
    )
    assert result.failure_code == "NO_LEGAL_LANE_PATH"


def test_disconnected_overlapping_lanes_cannot_be_crossed():
    topology = _topology(connected=False)
    validation = validate_lane_path([(2, 0), (2, 8)], topology)
    assert not validation["valid"] and validation["illegal_transitions"]
    assert (
        lane_astar_plan(
            _grid(), topology, np.array([0.0, 2.0]), np.array([8.0, 2.0])
        ).path
        is None
    )


def test_projection_bounds_and_all_blocked():
    grid, topology = _grid(), _topology()
    result = project_to_legal_lane(
        grid, topology, np.array([2.2, 3.0]), max_distance_m=2.0
    )
    assert result is not None and result[0][0] == 2 and result[2] > 1
    assert (
        project_to_legal_lane(grid, topology, np.array([2.0, 10.0]), max_distance_m=2.0)
        is None
    )
    grid.occupancy[topology.lane_mask] = 1
    assert (
        lane_astar_plan(
            grid, topology, np.array([0.0, 2.0]), np.array([8.0, 2.0])
        ).failure_code
        == "NO_LEGAL_LANE_START"
    )


def test_continuous_collision_between_waypoints_and_diagonal_corner():
    grid, topology = _grid(), _topology()
    grid.occupancy[2, 3] = 1
    result = validate_lane_path([(2, 0), (2, 8)], topology, grid.occupancy)
    assert not result["valid"] and [2, 3] in result["collision_cells"]
    from conceptgraph.navigation.lane_geometry import continuous_collision_cells

    occupancy = np.zeros((3, 3), dtype=np.uint8)
    occupancy[0, 1] = 1
    assert [0, 1] in continuous_collision_cells(
        np.array([0, 0]), np.array([1, 1]), occupancy
    )


def test_continuous_polygon_gap_detected_even_with_legal_waypoints():
    topology = _topology()
    a = topology.geometries["a"]
    # A concave notch crosses the line between two legal endpoints.
    polygon = np.array(
        [
            [0, 1.6],
            [4.4, 1.6],
            [4.4, 2.4],
            [2.6, 2.4],
            [2.6, 1.9],
            [2.4, 1.9],
            [2.4, 2.4],
            [0, 2.4],
        ]
    )
    topology.geometries["a"] = LaneGeometry(
        a.token, a.lane_type, polygon, a.centerline, a.heading, a.start_pose, a.end_pose
    )
    result = validate_lane_path([(2, 0), (2, 3)], topology)
    assert not result["valid"] and result["out_of_lane_length_m"] > 0


def test_empty_singleton_missing_heading_and_frame_mismatch():
    topology = _topology()
    assert not validate_lane_path([], topology)["valid"]
    assert validate_lane_path([(2, 0)], topology)["valid"]
    topology._trees.clear()
    topology.headings[:] = np.nan
    assert validate_lane_path([(2, 0)], topology)["missing_heading_steps"]
    grid = _grid()
    grid.origin_xy = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="coordinate frames"):
        project_to_legal_lane(grid, topology, np.array([0.0, 2.0]))


def test_invalid_lane_id_token_geometry_and_missing_map(tmp_path):
    topology = _topology()
    path = tmp_path / "lane.npz"
    meta = tmp_path / "lane_topology.json"
    ids = topology.lane_id_grid.copy()
    ids[0, 0] = 5
    np.savez(
        path,
        lane_id_grid=ids,
        lane_heading_grid=topology.lane_heading_grid,
        centerline_distance_grid=topology.centerline_distance_grid,
    )
    meta.write_text(
        json.dumps(
            {
                "lane_tokens": ["a", "b"],
                "lane_types": ["lane", "lane"],
                "geometries": {},
            }
        )
    )
    with pytest.raises(ValueError, match="unknown lane index"):
        load_lane_topology(path, meta)
    with pytest.raises(FileNotFoundError):
        load_lane_topology(tmp_path / "missing.npz")
    with pytest.raises(ValueError, match="unknown lane token"):
        LaneTopology(
            topology.lane_id_grid,
            topology.lane_heading_grid,
            topology.centerline_distance_grid,
            ["a", "b"],
            ["lane", "lane"],
            {"a": ("unknown",)},
        )


def test_arcline_and_polygon_build_preserve_heading_and_connectivity(tmp_path):
    arcline = {
        "start_pose": [0, 2, 0],
        "end_pose": [4, 2, 0],
        "shape": "LSL",
        "radius": 1,
        "segment_length": [0, 4, 0],
    }
    samples = _sample_arcline(arcline, 0.2)
    assert np.allclose(samples[:, 1], 2) and np.allclose(samples[:, 2], 0)
    nodes = [
        {"token": str(i), "x": x, "y": y}
        for i, (x, y) in enumerate([(0, 1), (4, 1), (4, 3), (0, 3)])
    ]
    mapping = {
        "node": nodes,
        "polygon": [{"token": "p", "exterior_node_tokens": [str(i) for i in range(4)]}],
        "lane": [{"token": "a", "polygon_token": "p"}],
        "arcline_path_3": {"a": [arcline]},
        "connectivity": {"a": {"incoming": [], "outgoing": ["unknown"]}},
    }
    arrays, metadata = build_lane_map(mapping, np.array([[0, 2], [4, 2]]), 0.5, 2)
    assert metadata["geometries"]["a"]["start_pose"] == [0.0, 2.0, 0.0]
    assert metadata["transitions"] == {"a": []}
    np.savez_compressed(tmp_path / "lane.npz", **arrays)
    (tmp_path / "lane_topology.json").write_text(json.dumps(metadata))
    restored = load_lane_topology(tmp_path / "lane.npz")
    assert restored.lane_tokens == ["a"]
    assert np.array_equal(restored.lane_mask, arrays["lane_id_grid"] >= 0)
    assert metadata["diagnostics"]["invalid_connectivity"]
    assert np.isfinite(arrays["lane_heading_grid"][arrays["lane_id_grid"] >= 0]).all()


def test_static_edge_cache_never_caches_dynamic_obstacle_decision():
    grid, topology = _grid(), _topology()
    start, goal = np.array([0.0, 2.0]), np.array([8.0, 2.0])
    assert lane_astar_plan(grid, topology, start, goal).path is not None
    grid.occupancy[2, 4] = 1
    assert (
        lane_astar_plan(grid, topology, start, goal, max_projection_distance_m=0.1).path
        is None
    )


def test_ordinary_astar_and_smoothing_cannot_cut_obstacle_corner():
    from conceptgraph.navigation.planner import (
        astar_plan_with_stats,
        path_is_collision_free,
        smooth_grid_path,
        _supercover_cells,
    )

    grid = _grid()
    grid.occupancy[0, 1] = 1
    grid.occupancy[1, 0] = 1
    assert not path_is_collision_free(grid, [(0, 0), (1, 1)])
    assert astar_plan_with_stats(grid, (0, 0), (1, 1)) is None
    grid.occupancy[1, 0] = 0
    planned = astar_plan_with_stats(grid, (0, 0), (1, 1))
    assert planned is not None and len(planned.path) == 3
    smoothed = smooth_grid_path(grid, planned.path)
    assert path_is_collision_free(grid, smoothed)
    assert (0, 1) in set(_supercover_cells((0, 0), (1, 1)))
    assert set(_supercover_cells((0, 0), (2, 4))) == set(
        _supercover_cells((2, 4), (0, 0))
    )
