from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SceneGraphNode:
    node_id: int
    object_tag: str
    possible_tags: list[str]
    caption: str
    bbox_center: np.ndarray
    bbox_extent: np.ndarray
    motion_state: str = "unknown"
    is_potential_dynamic: bool = False
    velocity: float = 0.0
    is_short_term: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneGraphEdge:
    source: int
    target: int
    relation: str


@dataclass
class ParsedNavigationQuery:
    raw_text: str
    target_text: str
    anchor_text: str | None
    relation: str | None
    prefer_nearest: bool = False
    prefer_front: bool = False
    target_categories: list[str] = field(default_factory=list)
    target_colors: list[str] = field(default_factory=list)
    anchor_colors: list[str] = field(default_factory=list)
    display_text: str = ""
    query_mode: str = "generic"
    near_distance: float | None = None
    min_target_distance_m: float | None = None
    max_target_distance_m: float | None = None


@dataclass
class CandidateTarget:
    node_id: int
    score: float
    text_score: float
    relation_score: float
    distance_prior: float
    reachable: bool = False
    target_point: np.ndarray | None = None
    target_grid: tuple[int, int] | None = None
    debug: dict[str, Any] = field(default_factory=dict)
    reachability_reason: str | None = None
    reachable_rank: int | None = None


@dataclass
class NavigationPlan:
    query: ParsedNavigationQuery
    selected_node_id: int
    selected_object_tag: str
    selected_target_point: np.ndarray
    path_world: np.ndarray
    path_grid: list[tuple[int, int]]
    path_cost: float
    candidate_targets: list[CandidateTarget]
    occupancy_origin_xy: np.ndarray
    occupancy_resolution: float
    occupancy_shape: tuple[int, int]
    path_world_raw: np.ndarray | None = None
    top1_semantic_node_id: int | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    fallback_distance_m: float = 0.0
    planning_time_s: float = 0.0
    debug: dict[str, Any] = field(default_factory=dict)
