from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from conceptgraph.navigation.planner import OccupancyGrid2D, free_component
from conceptgraph.navigation.target_selector import DEFAULT_UTILITY_WEIGHTS, TargetSelectionConfig, TargetSelector
from conceptgraph.navigation.types import CandidateTarget, SceneGraphNode


def make_grid(
    shape: tuple[int, int] = (32, 32),
    occupied: list[tuple[slice | int, slice | int]] | None = None,
    resolution: float = 1.0,
) -> OccupancyGrid2D:
    occupancy = np.zeros(shape, dtype=np.uint8)
    for rows, cols in occupied or []:
        occupancy[rows, cols] = 1
    clearance = distance_transform_edt(1 - occupancy) * resolution
    return OccupancyGrid2D(
        occupancy=occupancy,
        resolution=resolution,
        origin_xy=np.zeros(2, dtype=float),
        clearance_m=clearance,
    )


def make_node(
    node_id: int,
    center: tuple[float, float] = (16.0, 16.0),
    extent: tuple[float, float] = (4.0, 4.0),
    tag: str = "car",
    **kwargs,
) -> SceneGraphNode:
    return SceneGraphNode(
        node_id=node_id,
        object_tag=tag,
        possible_tags=[tag],
        caption=tag,
        bbox_center=np.asarray([center[0], center[1], 0.0], dtype=float),
        bbox_extent=np.asarray([extent[0], extent[1], 1.0], dtype=float),
        **kwargs,
    )


def make_candidate(node_id: int, score: float = 1.0, relation_score: float = 1.0) -> CandidateTarget:
    return CandidateTarget(
        node_id=node_id,
        score=score,
        text_score=score,
        relation_score=relation_score,
        distance_prior=0.0,
    )


def make_config(**overrides) -> TargetSelectionConfig:
    values = {
        "mode": "joint_utility",
        "top_k": 5,
        "goal_candidates_per_object": {"default": 8},
        "class_approach_distance_m": {"default": 2.0},
        "min_clearance_m": {"default": 0.5},
        "max_fallback_distance_m": {"default": 2.0},
        "allow_global_fallback": False,
        "utility_weights": dict(DEFAULT_UTILITY_WEIGHTS),
        "dynamic_class_penalties": {"default": 0.0, "dynamic_motion": 0.35, "short_term": 0.2},
        "deterministic_tie_break": "semantic_rank_node_id_goal_rc",
    }
    values.update(overrides)
    return TargetSelectionConfig(**values)


def make_selector(
    grid: OccupancyGrid2D,
    nodes: dict[int, SceneGraphNode],
    start_rc: tuple[int, int] = (2, 2),
    config: TargetSelectionConfig | None = None,
    **kwargs,
) -> TargetSelector:
    return TargetSelector(
        grid=grid,
        nodes=nodes,
        start_rc=start_rc,
        start_component=free_component(grid, start_rc),
        config=config or make_config(),
        path_smoothing_iterations=1,
        **kwargs,
    )


@pytest.fixture
def pipeline_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        default_near_distance_m=8.0,
        lambda_text=0.55,
        lambda_relation=0.30,
        lambda_distance_prior=0.25,
        start_search_radius_m=3.0,
        goal_search_radius_m=5.0,
        allow_goal_fallback=False,
        semantic_cost_weight=0.0,
        clearance_cost_weight=0.0,
        path_smoothing_iterations=1,
        max_candidates=3,
        target_selection=SimpleNamespace(mode="joint_utility", top_k=3),
    )
