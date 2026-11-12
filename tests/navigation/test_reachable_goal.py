from __future__ import annotations

import json

import numpy as np

from conceptgraph.navigation.io import load_scene_graph_nodes
from conceptgraph.navigation.pipeline import SemanticNavigationPipeline
from conceptgraph.navigation.target_selector import TargetSelectionConfig, target_selection_result_to_dict

from .conftest import make_candidate, make_config, make_grid, make_node, make_selector


def test_goals_must_be_connected_to_start() -> None:
    grid = make_grid(shape=(24, 24), occupied=[(slice(None), 11)])
    node = make_node(1, center=(18.0, 12.0), extent=(2.0, 2.0))
    config = make_config(max_fallback_distance_m={"default": 1.0})

    result = make_selector(grid, {1: node}, start_rc=(4, 4), config=config).select([make_candidate(1)])

    assert not result.success
    assert result.failure_code == "ALL_CANDIDATES_REJECTED"
    assert not any(goal.path_found for goal in result.evaluated_candidates[0].goal_candidates)
    assert all(
        goal.rejection_reason in {"fallback_distance_exceeded", "duplicate_goal"}
        for goal in result.evaluated_candidates[0].goal_candidates
    )


def test_clearance_threshold_rejects_all_goals() -> None:
    grid = make_grid(shape=(20, 20), occupied=[(slice(None), 0)])
    node = make_node(1, center=(10.0, 10.0), extent=(2.0, 2.0))
    config = make_config(min_clearance_m={"default": 100.0})

    result = make_selector(grid, {1: node}, config=config).select([make_candidate(1)])

    assert not result.success
    assert {goal.rejection_reason for goal in result.evaluated_candidates[0].goal_candidates} == {"clearance_insufficient"}


def test_fallback_distance_limit_is_hard_even_when_global_fallback_enabled() -> None:
    grid = make_grid(shape=(12, 12))
    node = make_node(1, center=(30.0, 30.0), extent=(2.0, 2.0))
    config = make_config(
        max_fallback_distance_m={"default": 0.5},
        allow_global_fallback=True,
    )

    result = make_selector(grid, {1: node}, config=config).select([make_candidate(1)])

    assert not result.success
    assert {goal.rejection_reason for goal in result.evaluated_candidates[0].goal_candidates} == {"fallback_distance_exceeded"}


def test_missing_and_degenerate_bbox_have_explicit_diagnostics() -> None:
    grid = make_grid()
    missing = make_node(1)
    missing.bbox_center = np.asarray([], dtype=float)
    missing.bbox_extent = np.asarray([], dtype=float)
    degenerate = make_node(2, extent=(0.0, 2.0))

    result = make_selector(grid, {1: missing, 2: degenerate}).select(
        [make_candidate(1), make_candidate(2)]
    )

    assert not result.success
    assert [candidate.rejection_reason for candidate in result.evaluated_candidates] == [
        "missing_bbox",
        "degenerate_bbox",
    ]


def test_bbox_fields_absent_in_loaded_metadata_are_reported_missing() -> None:
    grid = make_grid()
    node = make_node(1)
    node.metadata = {"id": 1, "object_tag": "car"}
    node.bbox_center = np.zeros(3, dtype=float)
    node.bbox_extent = np.zeros(3, dtype=float)

    result = make_selector(grid, {1: node}).select([make_candidate(1)])

    assert result.success is False
    assert result.evaluated_candidates[0].rejection_reason == "missing_bbox"


def test_scene_graph_loader_reads_dictionary_bbox(tmp_path) -> None:
    path = tmp_path / "scene_graph.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": 7,
                    "object_tag": "car",
                    "bbox": {"center": [3.0, 4.0, 1.0], "extent": [2.0, 1.0, 1.5]},
                }
            ]
        ),
        encoding="utf-8",
    )

    nodes = load_scene_graph_nodes(path)

    assert nodes[7].bbox_center.tolist() == [3.0, 4.0, 1.0]
    assert nodes[7].bbox_extent.tolist() == [2.0, 1.0, 1.5]


def test_all_failures_are_structured_and_json_ready() -> None:
    grid = make_grid(shape=(10, 10))
    node = make_node(1, center=(30.0, 30.0), extent=(2.0, 2.0))
    selector = make_selector(
        grid,
        {1: node},
        config=TargetSelectionConfig(mode="first_reachable", top_k=1),
        legacy_goal_search_radius_m=1.0,
        legacy_allow_goal_fallback=False,
    )

    result = selector.select([make_candidate(1)])

    assert result.success is False
    assert result.selected_candidate is None
    assert result.failure_code == "ALL_CANDIDATES_REJECTED"
    assert result.failure_reason
    assert isinstance(result.timing["target_selection_s"], float)
    reparsed = json.loads(json.dumps(target_selection_result_to_dict(result)))
    assert reparsed["success"] is False
    assert reparsed["failure_code"] == "ALL_CANDIDATES_REJECTED"


def test_empty_scene_graph_and_no_drivable_area_are_structured(pipeline_cfg) -> None:
    class EmptyPipeline(SemanticNavigationPipeline):
        def _load_nodes(self):
            return {}

        def _load_edges(self):
            return []

    empty = EmptyPipeline(pipeline_cfg).select("前往小汽车附近", np.asarray([1.0, 1.0, 0.0]))
    assert empty.failure_code == "EMPTY_SCENE_GRAPH"

    class BlockedPipeline(SemanticNavigationPipeline):
        def _load_nodes(self):
            return {1: make_node(1, center=(5.0, 5.0), extent=(2.0, 2.0))}

        def _load_edges(self):
            return []

        def _get_occupancy_grid(self, nodes):
            return make_grid(shape=(12, 12), occupied=[(slice(None), slice(None))])

    blocked = BlockedPipeline(pipeline_cfg).select("前往小汽车附近", np.asarray([1.0, 1.0, 0.0]))
    assert blocked.success is False
    assert blocked.failure_code == "NO_FREE_START_CELL"
