from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from conceptgraph.scenegraph.relations import SceneGraphEdge as SceneGraphEdge


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
    excluded_categories: list[str] = field(default_factory=list)
    excluded_colors: list[str] = field(default_factory=list)
    requires_static: bool = False


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
    semantic_rank: int | None = None
    goal_candidates: list[ReachableGoalCandidate] = field(default_factory=list)
    path_cost: float | None = None
    path_length_m: float | None = None
    min_clearance_m: float | None = None
    fallback_distance_m: float | None = None
    selection_utility: float | None = None
    rejection_reason: str | None = None
    dynamic_risk: float = 0.0
    selection_components: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReachableGoalCandidate:
    goal_xy: np.ndarray | None
    goal_rc: tuple[int, int] | None
    distance_to_object_m: float | None
    clearance_m: float | None
    connected_to_start: bool
    path_found: bool = False
    path_cost: float | None = None
    path_length_m: float | None = None
    fallback_distance_m: float | None = None
    fallback_reason: str | None = None
    rejection_reason: str | None = None
    desired_goal_xy: np.ndarray | None = None
    desired_goal_rc: tuple[int, int] | None = None
    path_grid: list[tuple[int, int]] = field(default_factory=list)
    path_grid_raw: list[tuple[int, int]] = field(default_factory=list)
    min_path_clearance_m: float | None = None
    expanded_nodes: int = 0
    selection_utility: float | None = None
    scoring_components: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetSelectionResult:
    success: bool
    mode: str
    selected_candidate: CandidateTarget | None
    evaluated_candidates: list[CandidateTarget]
    failure_code: str | None
    failure_reason: str | None
    scoring_weights: dict[str, float]
    timing: dict[str, float]
    plan: NavigationPlan | None = None
    debug: dict[str, Any] = field(default_factory=dict)


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
    target_selection_mode: str = "first_reachable"
    selection_utility: float | None = None
    scoring_weights: dict[str, float] = field(default_factory=dict)
    selection_timing: dict[str, float] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OnlineNavigationFailure:
    code: str
    message: str
    recoverable: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "recoverable": bool(self.recoverable),
            "details": _online_json_safe(self.details),
        }


@dataclass
class OnlineNavigationUpdate:
    frame_idx: int
    state_before: str
    state_after: str
    transition_reason: str
    candidate_ids: list[int]
    selected_node_id: int | None
    target_confidence: float
    path_world: np.ndarray
    replanned: bool
    replan_reason: str | None
    timings: dict[str, float]
    qwen_audit: dict[str, Any]
    failure: OnlineNavigationFailure | None
    state_trace: list[str] = field(default_factory=list)
    candidate_scores: dict[int, float] = field(default_factory=dict)
    replan_audit: list[dict[str, Any]] = field(default_factory=list)
    relation_update: dict[str, Any] = field(default_factory=dict)
    occupancy_audit: dict[str, Any] = field(default_factory=dict)
    input_version: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_full_path: bool = True) -> dict[str, Any]:
        path = np.asarray(self.path_world, dtype=float)
        return {
            "frame_idx": int(self.frame_idx),
            "state_before": self.state_before,
            "state_after": self.state_after,
            "transition_reason": self.transition_reason,
            "state_trace": list(self.state_trace),
            "candidate_ids": [int(item) for item in self.candidate_ids],
            "candidate_scores": {
                str(key): float(value) for key, value in sorted(self.candidate_scores.items())
            },
            "selected_node_id": (
                None if self.selected_node_id is None else int(self.selected_node_id)
            ),
            "target_confidence": float(self.target_confidence),
            "path_world": path.tolist() if include_full_path else [],
            "path_point_count": int(len(path)),
            "replanned": bool(self.replanned),
            "replan_reason": self.replan_reason,
            "replan_audit": _online_json_safe(self.replan_audit),
            "timings": {key: float(value) for key, value in sorted(self.timings.items())},
            "qwen_audit": _online_json_safe(self.qwen_audit),
            "relation_update": _online_json_safe(self.relation_update),
            "occupancy_audit": _online_json_safe(self.occupancy_audit),
            "input_version": _online_json_safe(self.input_version),
            "failure": None if self.failure is None else self.failure.to_dict(),
        }


def _online_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _online_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_online_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
