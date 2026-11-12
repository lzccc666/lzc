from __future__ import annotations

import json

import numpy as np
from omegaconf import OmegaConf

from conceptgraph.navigation.target_selector import (
    TargetSelectionConfig,
    candidate_target_to_dict,
)

from .conftest import make_candidate, make_config, make_grid, make_node, make_selector


def test_bbox_boundary_safety_ring_generates_deduplicated_free_goals() -> None:
    grid = make_grid(occupied=[(slice(14, 19), slice(14, 19))])
    node = make_node(1)
    result = make_selector(grid, {1: node}).select([make_candidate(1)])

    assert result.success
    goals = [goal for goal in result.selected_candidate.goal_candidates if goal.path_found]
    assert len(goals) >= 4
    assert len({goal.goal_rc for goal in goals}) == len(goals)
    for goal in goals:
        assert grid.in_bounds(goal.goal_rc)
        assert grid.is_free(goal.goal_rc)
        assert goal.connected_to_start
        assert goal.distance_to_object_m >= 2.0 - np.sqrt(2.0) / 2.0 - 1e-9


def test_joint_utility_prefers_better_path_over_semantic_tie() -> None:
    grid = make_grid(shape=(36, 36))
    nodes = {
        1: make_node(1, center=(29.0, 29.0), extent=(2.0, 2.0)),
        2: make_node(2, center=(9.0, 9.0), extent=(2.0, 2.0)),
    }
    candidates = [make_candidate(1, score=1.0), make_candidate(2, score=1.0)]
    weights = {
        "semantic_score": 0.2,
        "relation_score": 0.0,
        "path_cost": 0.8,
        "clearance_score": 0.0,
        "fallback_penalty": 0.0,
        "dynamic_risk": 0.0,
    }
    config = make_config(utility_weights=weights)

    result = make_selector(grid, nodes, config=config).select(candidates)

    assert result.success
    assert result.selected_candidate.node_id == 2
    assert all(candidate.goal_candidates for candidate in result.evaluated_candidates)
    assert all(
        goal.path_found or goal.rejection_reason
        for candidate in result.evaluated_candidates
        for goal in candidate.goal_candidates
    )


def test_dynamic_risk_penalty_changes_selection() -> None:
    grid = make_grid(shape=(24, 24))
    nodes = {
        1: make_node(1, center=(12.0, 12.0), extent=(2.0, 2.0), tag="pedestrian", motion_state="dynamic"),
        2: make_node(2, center=(12.0, 12.0), extent=(2.0, 2.0), tag="traffic sign"),
    }
    candidates = [make_candidate(1, score=1.0), make_candidate(2, score=0.98)]
    weights = {
        "semantic_score": 0.2,
        "relation_score": 0.0,
        "path_cost": 0.0,
        "clearance_score": 0.0,
        "fallback_penalty": 0.0,
        "dynamic_risk": 0.8,
    }
    config = make_config(
        utility_weights=weights,
        dynamic_class_penalties={"default": 0.0, "pedestrian": 0.7, "dynamic_motion": 0.3},
    )

    result = make_selector(grid, nodes, config=config).select(candidates)

    assert result.success
    assert result.selected_candidate.node_id == 2
    assert result.evaluated_candidates[0].dynamic_risk == 1.0
    assert result.evaluated_candidates[0].selection_components["weighted"]["dynamic_risk"] < 0.0


def test_first_reachable_stops_after_first_success() -> None:
    grid = make_grid(shape=(20, 20))
    nodes = {
        1: make_node(1, center=(40.0, 40.0), extent=(2.0, 2.0)),
        2: make_node(2, center=(8.0, 8.0), extent=(2.0, 2.0)),
        3: make_node(3, center=(12.0, 12.0), extent=(2.0, 2.0)),
    }
    config = TargetSelectionConfig(mode="first_reachable", top_k=3)
    selector = make_selector(
        grid,
        nodes,
        config=config,
        legacy_max_candidates=3,
        legacy_goal_search_radius_m=2.0,
        legacy_allow_goal_fallback=False,
    )

    result = selector.select([make_candidate(1), make_candidate(2), make_candidate(3)])

    assert result.success
    assert result.mode == "first_reachable"
    assert result.selected_candidate.node_id == 2
    assert result.selected_candidate.semantic_rank == 2
    assert [candidate.node_id for candidate in result.evaluated_candidates] == [1, 2]


def test_identical_inputs_are_deterministic() -> None:
    grid = make_grid(shape=(28, 28), occupied=[(slice(12, 16), slice(12, 16))])
    nodes = {
        2: make_node(2, center=(14.0, 14.0)),
        1: make_node(1, center=(20.0, 8.0)),
    }

    def run_once() -> str:
        candidates = [make_candidate(1, score=0.9), make_candidate(2, score=0.9)]
        result = make_selector(grid, nodes).select(candidates)
        payload = {
            "success": result.success,
            "selected": candidate_target_to_dict(result.selected_candidate),
            "evaluated": [candidate_target_to_dict(candidate) for candidate in result.evaluated_candidates],
        }
        return json.dumps(payload, sort_keys=True)

    assert run_once() == run_once()


def test_new_and_legacy_configs_load() -> None:
    new_cfg = OmegaConf.load("conceptgraph/hydra_configs/semantic_navigation.yaml")
    parsed_new = TargetSelectionConfig.from_pipeline_config(new_cfg)
    parsed_legacy = TargetSelectionConfig.from_pipeline_config(
        OmegaConf.create({"max_candidates": 7})
    )

    assert parsed_new.mode == "joint_utility"
    assert parsed_new.top_k == 5
    assert parsed_new.class_approach_distance_m["vehicle"] == 2.0
    assert parsed_legacy.mode == "first_reachable"
    assert parsed_legacy.top_k == 7

    scalar_cfg = TargetSelectionConfig.from_pipeline_config(
        OmegaConf.create({"target_selection": {"min_clearance_m": 1.25}})
    )
    assert scalar_cfg.min_clearance_m["default"] == 1.25
