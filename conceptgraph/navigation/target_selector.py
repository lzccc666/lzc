from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from conceptgraph.navigation.planner import (
    FreeComponent,
    OccupancyGrid2D,
    ReachableGoalResult,
    astar_plan_with_stats,
    compute_clearance_bias_grid,
    compute_semantic_bias_grid,
    evaluate_reachable_goal_near_point_from_start,
    minimum_path_clearance_m,
    path_is_collision_free,
    path_length_m,
    smooth_grid_path,
)
from conceptgraph.navigation.lane_constraints import (
    LaneTopology,
    lane_astar_plan,
    project_to_legal_lane,
    validate_lane_path,
)
from conceptgraph.navigation.types import (
    CandidateTarget,
    ReachableGoalCandidate,
    SceneGraphNode,
    TargetSelectionResult,
)


DEFAULT_UTILITY_WEIGHTS = {
    "semantic_score": 0.35,
    "relation_score": 0.15,
    "path_cost": 0.25,
    "clearance_score": 0.10,
    "fallback_penalty": 0.10,
    "dynamic_risk": 0.05,
}


@dataclass(frozen=True)
class TargetSelectionConfig:
    mode: str = "first_reachable"
    top_k: int = 5
    goal_candidates_per_object: Mapping[str, int] = field(default_factory=lambda: {"default": 12})
    class_approach_distance_m: Mapping[str, float] = field(default_factory=lambda: {"default": 2.0})
    min_clearance_m: Mapping[str, float] = field(default_factory=lambda: {"default": 0.5})
    max_fallback_distance_m: Mapping[str, float] = field(default_factory=lambda: {"default": 3.0})
    allow_global_fallback: bool = False
    utility_weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_UTILITY_WEIGHTS))
    dynamic_class_penalties: Mapping[str, float] = field(default_factory=lambda: {"default": 0.0})
    deterministic_tie_break: str = "semantic_rank_node_id_goal_rc"

    @classmethod
    def from_pipeline_config(cls, cfg: Any) -> TargetSelectionConfig:
        raw = _mapping(getattr(cfg, "target_selection", None))
        if not raw:
            return cls(mode="first_reachable", top_k=int(getattr(cfg, "max_candidates", 8)))
        weights = dict(DEFAULT_UTILITY_WEIGHTS)
        weights.update(_float_mapping(raw.get("utility_weights")))
        return cls(
            mode=str(raw.get("mode", "first_reachable")),
            top_k=max(1, int(raw.get("top_k", getattr(cfg, "max_candidates", 8)))),
            goal_candidates_per_object=_value_mapping(raw.get("goal_candidates_per_object"), int, 12),
            class_approach_distance_m=_value_mapping(raw.get("class_approach_distance_m"), float, 2.0),
            min_clearance_m=_value_mapping(raw.get("min_clearance_m"), float, 0.5),
            max_fallback_distance_m=_value_mapping(raw.get("max_fallback_distance_m"), float, 3.0),
            allow_global_fallback=bool(raw.get("allow_global_fallback", False)),
            utility_weights=weights,
            dynamic_class_penalties=_value_mapping(raw.get("dynamic_class_penalties"), float, 0.0),
            deterministic_tie_break=str(raw.get("deterministic_tie_break", "semantic_rank_node_id_goal_rc")),
        )


class TargetSelector:
    def __init__(
        self,
        grid: OccupancyGrid2D,
        nodes: Mapping[int, SceneGraphNode],
        start_rc: tuple[int, int],
        start_component: FreeComponent | set[tuple[int, int]],
        config: TargetSelectionConfig,
        semantic_cost_weight: float = 0.0,
        clearance_cost_weight: float = 0.0,
        path_smoothing_iterations: int = 2,
        legacy_max_candidates: int | None = None,
        legacy_goal_search_radius_m: float = 25.0,
        legacy_allow_goal_fallback: bool = True,
        lane_topology: LaneTopology | None = None,
        lane_projection_max_distance_m: float = 8.0,
        lane_centerline_weight: float = 0.25,
        lane_direction_tolerance_deg: float = 75.0,
        start_heading_xy: np.ndarray | None = None,
        vehicle_xy: np.ndarray | None = None,
    ) -> None:
        self.grid = grid
        self.nodes = nodes
        self.start_rc = (int(start_rc[0]), int(start_rc[1]))
        self.start_component = start_component
        self.config = config
        self.semantic_cost_weight = float(semantic_cost_weight)
        self.clearance_cost_weight = float(clearance_cost_weight)
        self.path_smoothing_iterations = int(path_smoothing_iterations)
        self.legacy_max_candidates = int(legacy_max_candidates or config.top_k)
        self.legacy_goal_search_radius_m = float(legacy_goal_search_radius_m)
        self.legacy_allow_goal_fallback = bool(legacy_allow_goal_fallback)
        self.vehicle_xy = self.grid.grid_to_world(self.start_rc) if vehicle_xy is None else np.asarray(vehicle_xy, dtype=float)
        self.lane_topology = lane_topology
        self.lane_projection_max_distance_m = float(lane_projection_max_distance_m)
        self.lane_centerline_weight = float(lane_centerline_weight)
        self.lane_direction_tolerance_deg = float(lane_direction_tolerance_deg)
        self.start_heading_xy = None if start_heading_xy is None else np.asarray(start_heading_xy, dtype=float)

    @property
    def lane_enabled(self) -> bool:
        return self.lane_topology is not None

    def select(self, candidates: list[CandidateTarget]) -> TargetSelectionResult:
        started = time.perf_counter()
        for rank, candidate in enumerate(candidates, start=1):
            candidate.semantic_rank = rank

        if not candidates:
            return self._failure(
                mode=self.config.mode,
                candidates=[],
                code="NO_SEMANTIC_CANDIDATES",
                reason="Semantic ranking produced no target candidates.",
                started=started,
            )
        if self.config.mode == "first_reachable":
            return self._select_first_reachable(candidates, started)
        if self.config.mode in {"joint_utility", "lane_constrained"}:
            return self._select_joint_utility(candidates, started)
        return self._failure(
            mode=self.config.mode,
            candidates=[],
            code="INVALID_SELECTION_MODE",
            reason=f"Unsupported target selection mode: {self.config.mode}",
            started=started,
        )

    def _select_first_reachable(
        self,
        candidates: list[CandidateTarget],
        started: float,
    ) -> TargetSelectionResult:
        evaluated: list[CandidateTarget] = []
        selected: CandidateTarget | None = None
        for candidate in candidates[: self.legacy_max_candidates]:
            evaluated.append(candidate)
            node = self.nodes.get(candidate.node_id)
            if node is None:
                candidate.rejection_reason = "missing_scene_graph_node"
                candidate.reachability_reason = candidate.rejection_reason
                continue
            target_xy = np.asarray(node.bbox_center, dtype=float)[:2]
            target_rc = self.grid.world_to_grid(target_xy)
            candidate.debug.update(
                {
                    "target_grid_raw": list(target_rc),
                    "target_in_map": self.grid.in_bounds(target_rc),
                    "target_free": self.grid.is_free(target_rc),
                    "target_generation": "legacy_bbox_center",
                }
            )
            if self.lane_enabled:
                projected = project_to_legal_lane(
                    self.grid,
                    self.lane_topology,
                    target_xy,
                    max_distance_m=self.lane_projection_max_distance_m,
                )
                reachable = None if projected is None else _lane_projection_as_reachable(projected)
            else:
                reachable = evaluate_reachable_goal_near_point_from_start(
                    self.grid,
                    target_xy,
                    self.start_rc,
                    self.legacy_goal_search_radius_m,
                    allow_global_fallback=self.legacy_allow_goal_fallback,
                    component=self.start_component,
                )
            if reachable is None:
                reason = "no_free_goal_nearby"
                if not self.grid.in_bounds(target_rc):
                    reason = "out_of_map"
                elif not self.grid.is_free(target_rc):
                    reason = "occupied_target"
                candidate.rejection_reason = reason
                candidate.reachability_reason = reason
                continue

            goal = ReachableGoalCandidate(
                goal_xy=reachable.goal_xy,
                goal_rc=reachable.goal_rc,
                desired_goal_xy=target_xy,
                desired_goal_rc=target_rc,
                distance_to_object_m=float(np.linalg.norm(reachable.goal_xy - target_xy)),
                clearance_m=self._clearance(reachable.goal_rc),
                connected_to_start=bool(reachable.start_connected),
                fallback_distance_m=float(reachable.fallback_distance_m),
                fallback_reason=reachable.reason,
            )
            self._plan_goal(goal)
            candidate.goal_candidates.append(goal)
            if not goal.path_found:
                candidate.rejection_reason = goal.rejection_reason or "astar_failed"
                candidate.reachability_reason = candidate.rejection_reason
                continue

            self._apply_goal(candidate, node, goal)
            candidate.reachable_rank = int(candidate.semantic_rank or 0)
            candidate.debug.update(
                {
                    "reachability": "reachable",
                    "reachability_reason": reachable.reason,
                    "fallback_used": bool(reachable.fallback_used),
                    "fallback_distance_m": float(reachable.fallback_distance_m),
                    "selected_goal_xy": reachable.goal_xy.tolist(),
                }
            )
            self._score_goals([candidate])
            selected = candidate
            break

        if selected is None:
            return self._failure(
                mode="first_reachable",
                candidates=evaluated,
                code="NO_LEGAL_LANE_PATH" if self.lane_enabled else "ALL_CANDIDATES_REJECTED",
                reason="No semantic candidate produced a valid reachable path.",
                started=started,
            )
        return TargetSelectionResult(
            success=True,
            mode="first_reachable",
            selected_candidate=selected,
            evaluated_candidates=evaluated,
            failure_code=None,
            failure_reason=None,
            scoring_weights=self._weights(),
            timing={"target_selection_s": float(time.perf_counter() - started)},
            debug={"stopped_after_semantic_rank": selected.semantic_rank},
        )

    def _select_joint_utility(
        self,
        candidates: list[CandidateTarget],
        started: float,
    ) -> TargetSelectionResult:
        evaluated = candidates[: self.config.top_k]
        for candidate in evaluated:
            node = self.nodes.get(candidate.node_id)
            if node is None:
                candidate.rejection_reason = "missing_scene_graph_node"
                candidate.reachability_reason = candidate.rejection_reason
                continue
            candidate.dynamic_risk = self._dynamic_risk(node)
            goals, object_rejection, generation_debug = self._generate_goal_candidates(node)
            candidate.goal_candidates = goals
            candidate.debug["goal_generation"] = generation_debug
            if object_rejection is not None:
                candidate.rejection_reason = object_rejection
                candidate.reachability_reason = object_rejection
                continue
            for goal in candidate.goal_candidates:
                if goal.rejection_reason is None:
                    self._plan_goal(goal)

        self._score_goals(evaluated)
        feasible = [candidate for candidate in evaluated if candidate.reachable]
        if not feasible:
            for candidate in evaluated:
                if candidate.rejection_reason is None:
                    candidate.rejection_reason = self._candidate_rejection(candidate)
                    candidate.reachability_reason = candidate.rejection_reason
            return self._failure(
                mode=self.config.mode,
                candidates=evaluated,
                code="NO_LEGAL_LANE_PATH" if self.lane_enabled else "ALL_CANDIDATES_REJECTED",
                reason="Every Top-K semantic candidate failed goal or path validation.",
                started=started,
            )

        selected = min(feasible, key=self._candidate_selection_key)
        selected.reachable_rank = 1 + sorted(
            feasible,
            key=lambda item: (int(item.semantic_rank or 10**9), int(item.node_id)),
        ).index(selected)
        return TargetSelectionResult(
            success=True,
            mode=self.config.mode,
            selected_candidate=selected,
            evaluated_candidates=evaluated,
            failure_code=None,
            failure_reason=None,
            scoring_weights=self._weights(),
            timing={"target_selection_s": float(time.perf_counter() - started)},
            debug={
                "top_k": len(evaluated),
                "feasible_candidate_count": len(feasible),
                "deterministic_tie_break": self.config.deterministic_tie_break,
            },
        )

    def _generate_goal_candidates(
        self,
        node: SceneGraphNode,
    ) -> tuple[list[ReachableGoalCandidate], str | None, dict[str, Any]]:
        center = np.asarray(node.bbox_center, dtype=float)
        extent = np.asarray(node.bbox_extent, dtype=float)
        debug: dict[str, Any] = {"object_tag": node.object_tag}
        metadata = node.metadata if isinstance(node.metadata, dict) else {}
        bbox_payload = metadata.get("bbox")
        bbox_np_present = metadata.get("bbox_np") is not None
        center_present = metadata.get("bbox_center") is not None or bbox_np_present
        extent_present = metadata.get("bbox_extent") is not None or bbox_np_present
        if isinstance(bbox_payload, dict):
            center_present = center_present or bbox_payload.get("center") is not None
            extent_present = extent_present or bbox_payload.get("extent") is not None
        elif bbox_payload is not None:
            center_present = center_present or getattr(bbox_payload, "center", None) is not None
            extent_present = extent_present or getattr(bbox_payload, "extent", None) is not None
        if metadata and not (center_present and extent_present):
            return [], "missing_bbox", {**debug, "bbox_diagnostic": "bbox_fields_absent"}
        if center.size < 2 or extent.size < 2 or not np.all(np.isfinite(center[:2])):
            return [], "missing_bbox", {**debug, "bbox_diagnostic": "missing_or_nonfinite_center"}
        if not np.all(np.isfinite(extent[:2])) or np.any(extent[:2] <= 0.0):
            return [], "degenerate_bbox", {**debug, "bbox_diagnostic": "nonpositive_or_nonfinite_extent"}

        tag = _canonical_class(node.object_tag)
        count = max(1, int(_class_value(self.config.goal_candidates_per_object, tag, 12)))
        approach = max(0.0, float(_class_value(self.config.class_approach_distance_m, tag, 2.0)))
        min_clearance = max(0.0, float(_class_value(self.config.min_clearance_m, tag, 0.5)))
        max_fallback = max(0.0, float(_class_value(self.config.max_fallback_distance_m, tag, 3.0)))
        debug.update(
            {
                "candidate_count": count,
                "approach_distance_m": approach,
                "min_clearance_m": min_clearance,
                "max_fallback_distance_m": max_fallback,
                "allow_global_fallback": self.config.allow_global_fallback,
            }
        )

        desired_points = _bbox_ring_points(center[:2], extent[:2], approach, count)
        if self.lane_enabled:
            return self._generate_lane_goal_candidates(node, desired_points, debug)
        component_cells = self._component_cells()
        if component_cells.size == 0:
            return [], "no_start_connected_free_cells", debug
        component_world = np.column_stack(
            (
                self.grid.origin_xy[0] + component_cells[:, 1] * self.grid.resolution,
                self.grid.origin_xy[1] + component_cells[:, 0] * self.grid.resolution,
            )
        )
        clearances = np.asarray([self._clearance((int(r), int(c))) for r, c in component_cells], dtype=float)
        object_distances = _points_to_bbox_distance(component_world, center[:2], extent[:2])
        safety_tolerance = self.grid.resolution * math.sqrt(2.0) * 0.5 + 1e-9
        safe_component = (clearances + 1e-9 >= min_clearance) & (
            object_distances + safety_tolerance >= approach
        )

        goals: list[ReachableGoalCandidate] = []
        used_goal_cells: set[tuple[int, int]] = set()
        for desired_xy in desired_points:
            desired_rc = self.grid.world_to_grid(desired_xy)
            direct_reason = self._direct_rejection(desired_rc, min_clearance, center[:2], extent[:2], approach)
            deltas = component_world - desired_xy.reshape(1, 2)
            fallback_distances = np.linalg.norm(deltas, axis=1)
            within_fallback = fallback_distances <= max_fallback + 1e-9
            eligible = safe_component & within_fallback
            if not bool(np.any(eligible)):
                reason = self._bounded_fallback_rejection(
                    direct_reason=direct_reason,
                    safe_component=safe_component,
                    clearances=clearances,
                    min_clearance=min_clearance,
                    object_distances=object_distances,
                    approach=approach,
                    fallback_distances=fallback_distances,
                    max_fallback=max_fallback,
                )
                goals.append(
                    ReachableGoalCandidate(
                        goal_xy=None,
                        goal_rc=None,
                        desired_goal_xy=desired_xy,
                        desired_goal_rc=desired_rc,
                        distance_to_object_m=None,
                        clearance_m=None,
                        connected_to_start=False,
                        fallback_reason=direct_reason,
                        rejection_reason=reason,
                        debug={"nearest_connected_fallback_distance_m": float(np.min(fallback_distances))},
                    )
                )
                continue

            eligible_indices = np.flatnonzero(eligible)
            best_index = min(
                eligible_indices.tolist(),
                key=lambda idx: (
                    round(float(fallback_distances[idx]), 12),
                    -round(float(clearances[idx]), 12),
                    int(component_cells[idx, 0]),
                    int(component_cells[idx, 1]),
                ),
            )
            goal_rc = (int(component_cells[best_index, 0]), int(component_cells[best_index, 1]))
            goal_xy = component_world[best_index]
            fallback_distance = float(fallback_distances[best_index])
            if goal_rc in used_goal_cells:
                goals.append(
                    ReachableGoalCandidate(
                        goal_xy=goal_xy,
                        goal_rc=goal_rc,
                        desired_goal_xy=desired_xy,
                        desired_goal_rc=desired_rc,
                        distance_to_object_m=float(object_distances[best_index]),
                        clearance_m=float(clearances[best_index]),
                        connected_to_start=True,
                        fallback_distance_m=fallback_distance,
                        fallback_reason=direct_reason,
                        rejection_reason="duplicate_goal",
                    )
                )
                continue
            used_goal_cells.add(goal_rc)
            goals.append(
                ReachableGoalCandidate(
                    goal_xy=np.asarray(goal_xy, dtype=float),
                    goal_rc=goal_rc,
                    desired_goal_xy=np.asarray(desired_xy, dtype=float),
                    desired_goal_rc=desired_rc,
                    distance_to_object_m=float(object_distances[best_index]),
                    clearance_m=float(clearances[best_index]),
                    connected_to_start=True,
                    fallback_distance_m=fallback_distance,
                    fallback_reason=direct_reason or ("grid_quantization" if fallback_distance > 1e-9 else None),
                )
            )
        return goals, None, debug

    def _plan_goal(self, goal: ReachableGoalCandidate) -> None:
        if goal.goal_rc is None:
            goal.rejection_reason = goal.rejection_reason or "missing_goal_cell"
            return
        if self.lane_enabled:
            self._plan_lane_goal(goal)
            return
        semantic_bias = compute_semantic_bias_grid(
            self.grid,
            goal.goal_rc,
            self.semantic_cost_weight,
        )
        clearance_bias = compute_clearance_bias_grid(self.grid, self.clearance_cost_weight)
        planned = astar_plan_with_stats(
            self.grid,
            self.start_rc,
            goal.goal_rc,
            semantic_bias=semantic_bias + clearance_bias,
            clearance_weight=self.clearance_cost_weight,
        )
        if planned is None:
            goal.rejection_reason = "astar_failed"
            return
        path_grid = smooth_grid_path(
            self.grid,
            planned.path,
            chaikin_iterations=self.path_smoothing_iterations,
        )
        if not path_is_collision_free(self.grid, path_grid):
            goal.rejection_reason = "smoothed_path_collision"
            return
        goal.path_found = True
        goal.path_cost = float(planned.cost)
        goal.path_grid_raw = list(planned.path)
        goal.path_grid = list(path_grid)
        goal.path_length_m = path_length_m(self.grid, path_grid)
        goal.min_path_clearance_m = minimum_path_clearance_m(self.grid, path_grid)
        goal.expanded_nodes = int(planned.expanded_nodes)

    def _generate_lane_goal_candidates(
        self,
        node: SceneGraphNode,
        desired_points: list[np.ndarray],
        debug: dict[str, Any],
    ) -> tuple[list[ReachableGoalCandidate], str | None, dict[str, Any]]:
        assert self.lane_topology is not None
        goals: list[ReachableGoalCandidate] = []
        used: set[tuple[int, int]] = set()
        for desired_xy in desired_points:
            desired_rc = self.grid.world_to_grid(desired_xy)
            projection = project_to_legal_lane(
                self.grid,
                self.lane_topology,
                desired_xy,
                max_distance_m=self.lane_projection_max_distance_m,
            )
            if projection is None:
                goals.append(ReachableGoalCandidate(
                    goal_xy=None,
                    goal_rc=None,
                    desired_goal_xy=np.asarray(desired_xy, dtype=float),
                    desired_goal_rc=desired_rc,
                    distance_to_object_m=None,
                    clearance_m=None,
                    connected_to_start=False,
                    fallback_reason="lane_projection_failed",
                    rejection_reason="NO_LEGAL_LANE_GOAL",
                ))
                continue
            goal_rc, goal_xy, distance = projection
            if goal_rc in used:
                goals.append(ReachableGoalCandidate(
                    goal_xy=goal_xy,
                    goal_rc=goal_rc,
                    desired_goal_xy=np.asarray(desired_xy, dtype=float),
                    desired_goal_rc=desired_rc,
                    distance_to_object_m=float(_point_to_bbox_distance(goal_xy, np.asarray(node.bbox_center)[:2], np.asarray(node.bbox_extent)[:2])),
                    clearance_m=self._clearance(goal_rc),
                    connected_to_start=False,
                    fallback_distance_m=float(distance),
                    fallback_reason="lane_projection",
                    rejection_reason="duplicate_goal",
                ))
                continue
            object_distance = float(_point_to_bbox_distance(goal_xy, np.asarray(node.bbox_center)[:2], np.asarray(node.bbox_extent)[:2]))
            if self._clearance(goal_rc) < float(debug["min_clearance_m"]) or object_distance < float(debug["approach_distance_m"]) - 1e-6:
                goals.append(ReachableGoalCandidate(
                    goal_xy=goal_xy, goal_rc=goal_rc, desired_goal_xy=np.asarray(desired_xy),
                    desired_goal_rc=desired_rc, distance_to_object_m=object_distance,
                    clearance_m=self._clearance(goal_rc), connected_to_start=False,
                    rejection_reason="lane_projection_unsafe_clearance",
                ))
                continue
            used.add(goal_rc)
            goals.append(ReachableGoalCandidate(
                goal_xy=np.asarray(goal_xy, dtype=float),
                goal_rc=goal_rc,
                desired_goal_xy=np.asarray(desired_xy, dtype=float),
                desired_goal_rc=desired_rc,
                distance_to_object_m=float(_point_to_bbox_distance(goal_xy, np.asarray(node.bbox_center)[:2], np.asarray(node.bbox_extent)[:2])),
                clearance_m=self._clearance(goal_rc),
                connected_to_start=True,
                fallback_distance_m=float(distance),
                fallback_reason="lane_projection" if distance > 1e-9 else None,
                debug={"lane_projection_distance_m": float(distance)},
            ))
        debug.update({"lane_projection": True, "candidate_count": len(goals)})
        return goals, None, debug

    def _plan_lane_goal(self, goal: ReachableGoalCandidate) -> None:
        assert self.lane_topology is not None
        if goal.goal_xy is None:
            goal.rejection_reason = goal.rejection_reason or "NO_LEGAL_LANE_GOAL"
            return
        start_xy = self.vehicle_xy
        planned = lane_astar_plan(
            self.grid,
            self.lane_topology,
            start_xy,
            goal.goal_xy,
            start_heading_xy=self.start_heading_xy,
            max_projection_distance_m=self.lane_projection_max_distance_m,
            centerline_weight=self.lane_centerline_weight,
            direction_tolerance_deg=self.lane_direction_tolerance_deg,
        )
        goal.debug.update({
            "lane_constrained": True,
            "lane_failure_code": planned.failure_code,
            "lane_failure_reason": planned.failure_reason,
            "start_projection_distance_m": planned.start_projection_distance_m,
            "goal_projection_distance_m": goal.fallback_distance_m,
        })
        if planned.path is None:
            goal.rejection_reason = planned.failure_code or "NO_LEGAL_LANE_PATH"
            return
        raw_path = list(planned.path)
        smoothed = smooth_grid_path(self.grid, raw_path, chaikin_iterations=self.path_smoothing_iterations)
        smoothed_validation = validate_lane_path(
            smoothed,
            self.lane_topology,
            self.grid.occupancy,
            start_heading_xy=self.start_heading_xy,
            direction_tolerance_deg=self.lane_direction_tolerance_deg,
        )
        if smoothed_validation["valid"] and path_is_collision_free(self.grid, smoothed):
            path_grid = smoothed
        else:
            goal.debug["smoothing_rejected"] = True
            goal.debug["smoothing_validation"] = smoothed_validation
            replanned = lane_astar_plan(
                self.grid, self.lane_topology, start_xy, goal.goal_xy,
                start_heading_xy=self.start_heading_xy,
                max_projection_distance_m=self.lane_projection_max_distance_m,
                centerline_weight=self.lane_centerline_weight,
                direction_tolerance_deg=self.lane_direction_tolerance_deg,
            )
            goal.debug["replanned_without_smoothing"] = True
            if replanned.path is None:
                goal.rejection_reason = "NO_LEGAL_LANE_PATH"
                return
            planned = replanned
            raw_path = list(replanned.path)
            path_grid = raw_path
        raw_validation = validate_lane_path(
            raw_path,
            self.lane_topology,
            self.grid.occupancy,
            start_heading_xy=self.start_heading_xy,
            direction_tolerance_deg=self.lane_direction_tolerance_deg,
        )
        if not raw_validation["valid"] or not path_is_collision_free(self.grid, path_grid):
            goal.rejection_reason = "NO_LEGAL_LANE_PATH"
            goal.debug["raw_validation"] = raw_validation
            return
        goal.path_found = True
        goal.path_cost = float(planned.cost or 0.0)
        goal.path_grid_raw = raw_path
        goal.path_grid = list(path_grid)
        goal.path_length_m = path_length_m(self.grid, path_grid)
        goal.min_path_clearance_m = minimum_path_clearance_m(self.grid, path_grid)
        goal.expanded_nodes = int(planned.expanded_nodes)
        goal.connected_to_start = True
        goal.debug["lane_validation"] = validate_lane_path(
            path_grid,
            self.lane_topology,
            self.grid.occupancy,
            start_heading_xy=self.start_heading_xy,
            direction_tolerance_deg=self.lane_direction_tolerance_deg,
        )

    def _score_goals(self, candidates: list[CandidateTarget]) -> None:
        entries = [
            (candidate, goal)
            for candidate in candidates
            for goal in candidate.goal_candidates
            if goal.path_found and goal.path_cost is not None
        ]
        if not entries:
            for candidate in candidates:
                if candidate.selection_components:
                    continue
                candidate.selection_components = _empty_scoring_components(candidate)
            return

        max_semantic = max(max(float(candidate.score) for candidate, _goal in entries), 1e-9)
        max_path_cost = max(max(float(goal.path_cost or 0.0) for _candidate, goal in entries), 1e-9)
        finite_clearances = [
            float(goal.min_path_clearance_m)
            for _candidate, goal in entries
            if goal.min_path_clearance_m is not None and math.isfinite(goal.min_path_clearance_m)
        ]
        max_clearance = max(max(finite_clearances, default=0.0), 1e-9)
        weights = self._weights()

        for candidate, goal in entries:
            node = self.nodes.get(candidate.node_id)
            dynamic_risk = self._dynamic_risk(node) if node is not None else 1.0
            candidate.dynamic_risk = dynamic_risk
            tag = _canonical_class(node.object_tag if node is not None else "default")
            max_fallback = max(
                float(_class_value(self.config.max_fallback_distance_m, tag, 3.0)),
                self.grid.resolution,
            )
            raw = {
                "semantic_score": float(candidate.score),
                "relation_score": float(candidate.relation_score),
                "path_cost": float(goal.path_cost or 0.0),
                "clearance_score": _finite_or_none(goal.min_path_clearance_m),
                "fallback_penalty": float(goal.fallback_distance_m or 0.0),
                "dynamic_risk": float(dynamic_risk),
            }
            normalized = {
                "semantic_score": min(max(raw["semantic_score"] / max_semantic, 0.0), 1.0),
                "relation_score": min(max(raw["relation_score"], 0.0), 1.0),
                "path_cost": min(max(raw["path_cost"] / max_path_cost, 0.0), 1.0),
                "clearance_score": 1.0 if raw["clearance_score"] is None else min(max(raw["clearance_score"] / max_clearance, 0.0), 1.0),
                "fallback_penalty": min(max(raw["fallback_penalty"] / max_fallback, 0.0), 1.0),
                "dynamic_risk": min(max(raw["dynamic_risk"], 0.0), 1.0),
            }
            weighted = {
                "semantic_score": weights["semantic_score"] * normalized["semantic_score"],
                "relation_score": weights["relation_score"] * normalized["relation_score"],
                "path_cost": -weights["path_cost"] * normalized["path_cost"],
                "clearance_score": weights["clearance_score"] * normalized["clearance_score"],
                "fallback_penalty": -weights["fallback_penalty"] * normalized["fallback_penalty"],
                "dynamic_risk": -weights["dynamic_risk"] * normalized["dynamic_risk"],
            }
            goal.selection_utility = float(sum(weighted.values()))
            goal.scoring_components = {
                "raw": raw,
                "normalized": normalized,
                "weighted": weighted,
            }

        for candidate in candidates:
            feasible_goals = [goal for goal in candidate.goal_candidates if goal.path_found]
            if not feasible_goals:
                candidate.rejection_reason = candidate.rejection_reason or self._candidate_rejection(candidate)
                candidate.reachability_reason = candidate.rejection_reason
                candidate.selection_components = _empty_scoring_components(candidate)
                continue
            selected_goal = min(feasible_goals, key=self._goal_selection_key)
            node = self.nodes[candidate.node_id]
            self._apply_goal(candidate, node, selected_goal)
            candidate.selection_utility = float(selected_goal.selection_utility or 0.0)
            candidate.selection_components = selected_goal.scoring_components

    def _apply_goal(
        self,
        candidate: CandidateTarget,
        node: SceneGraphNode,
        goal: ReachableGoalCandidate,
    ) -> None:
        candidate.reachable = True
        candidate.rejection_reason = None
        candidate.reachability_reason = "reachable"
        candidate.target_grid = goal.goal_rc
        z = float(node.bbox_center[2]) if np.asarray(node.bbox_center).size > 2 else 0.0
        candidate.target_point = np.asarray([goal.goal_xy[0], goal.goal_xy[1], z], dtype=float)
        candidate.path_cost = goal.path_cost
        candidate.path_length_m = goal.path_length_m
        candidate.min_clearance_m = goal.min_path_clearance_m
        candidate.fallback_distance_m = goal.fallback_distance_m
        if self.lane_enabled:
            candidate.debug["lane_planning"] = goal.debug
        candidate.debug.update(
            {
                "selected_goal_xy": goal.goal_xy.tolist(),
                "selected_goal_rc": list(goal.goal_rc),
                "fallback_reason": goal.fallback_reason,
                "fallback_distance_m": goal.fallback_distance_m,
                "astar_expanded_nodes": goal.expanded_nodes,
                "raw_astar_waypoints": len(goal.path_grid_raw),
                "smoothed_waypoints": len(goal.path_grid),
                "continuous_collision_check": True,
            }
        )

    def _component_cells(self) -> np.ndarray:
        if isinstance(self.start_component, FreeComponent):
            return np.asarray(self.start_component.cells, dtype=np.int32)
        if not self.start_component:
            return np.empty((0, 2), dtype=np.int32)
        return np.asarray(sorted(self.start_component), dtype=np.int32)

    def _clearance(self, rc: tuple[int, int]) -> float:
        if self.grid.clearance_m is None:
            return float("inf")
        return float(self.grid.clearance_m[rc])

    def _direct_rejection(
        self,
        rc: tuple[int, int],
        min_clearance: float,
        center_xy: np.ndarray,
        extent_xy: np.ndarray,
        approach: float,
    ) -> str | None:
        if not self.grid.in_bounds(rc):
            return "out_of_bounds"
        if not self.grid.is_free(rc):
            return "occupied_or_inflated"
        if not _component_contains(self.start_component, rc):
            return "not_connected_to_start"
        if self._clearance(rc) + 1e-9 < min_clearance:
            return "clearance_insufficient"
        distance = _point_to_bbox_distance(self.grid.grid_to_world(rc), center_xy, extent_xy)
        tolerance = self.grid.resolution * math.sqrt(2.0) * 0.5 + 1e-9
        if distance + tolerance < approach:
            return "safety_distance_insufficient"
        return None

    @staticmethod
    def _bounded_fallback_rejection(
        direct_reason: str | None,
        safe_component: np.ndarray,
        clearances: np.ndarray,
        min_clearance: float,
        object_distances: np.ndarray,
        approach: float,
        fallback_distances: np.ndarray,
        max_fallback: float,
    ) -> str:
        if not bool(np.any(clearances + 1e-9 >= min_clearance)):
            return "clearance_insufficient"
        if not bool(np.any(object_distances + 1e-9 >= approach)):
            return "safety_distance_insufficient"
        if bool(np.any(safe_component)) and float(np.min(fallback_distances[safe_component])) > max_fallback + 1e-9:
            return "fallback_distance_exceeded"
        return direct_reason or "no_valid_bounded_fallback"

    def _dynamic_risk(self, node: SceneGraphNode) -> float:
        penalties = self.config.dynamic_class_penalties
        tag = _canonical_class(node.object_tag)
        risk = float(_class_value(penalties, tag, 0.0))
        if node.motion_state.lower() == "dynamic" or node.is_potential_dynamic:
            risk += float(_class_value(penalties, "dynamic_motion", 0.35))
        if node.is_short_term:
            risk += float(_class_value(penalties, "short_term", 0.20))
        return min(max(risk, 0.0), 1.0)

    def _weights(self) -> dict[str, float]:
        return {key: float(self.config.utility_weights.get(key, default)) for key, default in DEFAULT_UTILITY_WEIGHTS.items()}

    @staticmethod
    def _candidate_rejection(candidate: CandidateTarget) -> str:
        reasons = [goal.rejection_reason for goal in candidate.goal_candidates if goal.rejection_reason]
        if not reasons:
            return "no_reachable_goal_candidates"
        if all(reason == "astar_failed" for reason in reasons):
            return "all_goal_paths_failed"
        if all(reason == "duplicate_goal" for reason in reasons):
            return "all_goal_candidates_duplicate"
        return "no_reachable_goal_candidates"

    @staticmethod
    def _goal_selection_key(goal: ReachableGoalCandidate) -> tuple[Any, ...]:
        rc = goal.goal_rc or (10**9, 10**9)
        return (
            -round(float(goal.selection_utility or 0.0), 12),
            round(float(goal.path_cost if goal.path_cost is not None else float("inf")), 12),
            -round(float(goal.min_path_clearance_m or 0.0), 12),
            round(float(goal.fallback_distance_m or 0.0), 12),
            int(rc[0]),
            int(rc[1]),
        )

    @staticmethod
    def _candidate_selection_key(candidate: CandidateTarget) -> tuple[Any, ...]:
        rc = candidate.target_grid or (10**9, 10**9)
        return (
            -round(float(candidate.selection_utility or 0.0), 12),
            int(candidate.semantic_rank or 10**9),
            int(candidate.node_id),
            int(rc[0]),
            int(rc[1]),
        )

    def _failure(
        self,
        mode: str,
        candidates: list[CandidateTarget],
        code: str,
        reason: str,
        started: float,
    ) -> TargetSelectionResult:
        return TargetSelectionResult(
            success=False,
            mode=mode,
            selected_candidate=None,
            evaluated_candidates=candidates,
            failure_code=code,
            failure_reason=reason,
            scoring_weights=self._weights(),
            timing={"target_selection_s": float(time.perf_counter() - started)},
        )


def _lane_projection_as_reachable(
    projection: tuple[tuple[int, int], np.ndarray, float],
) -> ReachableGoalResult:
    goal_rc, goal_xy, distance = projection
    return ReachableGoalResult(
        goal_xy=np.asarray(goal_xy, dtype=float),
        goal_rc=goal_rc,
        reason="lane_projection" if distance > 1e-9 else "lane_direct",
        fallback_used=distance > 1e-9,
        fallback_distance_m=float(distance),
        target_in_map=True,
        target_free=True,
        start_connected=True,
    )


def candidate_target_to_dict(candidate: CandidateTarget | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "node_id": int(candidate.node_id),
        "semantic_rank": candidate.semantic_rank,
        "score": float(candidate.score),
        "text_score": float(candidate.text_score),
        "relation_score": float(candidate.relation_score),
        "distance_prior": float(candidate.distance_prior),
        "reachable": bool(candidate.reachable),
        "target_point": _array_or_none(candidate.target_point),
        "target_grid": _tuple_or_none(candidate.target_grid),
        "path_cost": _finite_or_none(candidate.path_cost),
        "path_length_m": _finite_or_none(candidate.path_length_m),
        "min_clearance_m": _finite_or_none(candidate.min_clearance_m),
        "fallback_distance_m": _finite_or_none(candidate.fallback_distance_m),
        "dynamic_risk": float(candidate.dynamic_risk),
        "selection_utility": _finite_or_none(candidate.selection_utility),
        "selection_components": candidate.selection_components,
        "rejection_reason": candidate.rejection_reason,
        "reachability_reason": candidate.reachability_reason,
        "reachable_rank": candidate.reachable_rank,
        "goal_candidates": [reachable_goal_to_dict(goal) for goal in candidate.goal_candidates],
        "debug": candidate.debug,
    }


def reachable_goal_to_dict(goal: ReachableGoalCandidate) -> dict[str, Any]:
    return {
        "goal_xy": _array_or_none(goal.goal_xy),
        "goal_rc": _tuple_or_none(goal.goal_rc),
        "desired_goal_xy": _array_or_none(goal.desired_goal_xy),
        "desired_goal_rc": _tuple_or_none(goal.desired_goal_rc),
        "distance_to_object_m": _finite_or_none(goal.distance_to_object_m),
        "clearance_m": _finite_or_none(goal.clearance_m),
        "connected_to_start": bool(goal.connected_to_start),
        "path_found": bool(goal.path_found),
        "path_cost": _finite_or_none(goal.path_cost),
        "path_length_m": _finite_or_none(goal.path_length_m),
        "min_path_clearance_m": _finite_or_none(goal.min_path_clearance_m),
        "fallback_distance_m": _finite_or_none(goal.fallback_distance_m),
        "fallback_reason": goal.fallback_reason,
        "rejection_reason": goal.rejection_reason,
        "selection_utility": _finite_or_none(goal.selection_utility),
        "scoring_components": goal.scoring_components,
        "expanded_nodes": int(goal.expanded_nodes),
        "path_grid": [list(rc) for rc in goal.path_grid],
        "path_grid_raw": [list(rc) for rc in goal.path_grid_raw],
        "debug": goal.debug,
    }


def target_selection_result_to_dict(result: TargetSelectionResult) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "mode": result.mode,
        "selected_candidate": candidate_target_to_dict(result.selected_candidate),
        "evaluated_candidates": [candidate_target_to_dict(candidate) for candidate in result.evaluated_candidates],
        "failure_code": result.failure_code,
        "failure_reason": result.failure_reason,
        "scoring_weights": result.scoring_weights,
        "timing": result.timing,
        "debug": result.debug,
    }


def _bbox_ring_points(
    center_xy: np.ndarray,
    extent_xy: np.ndarray,
    approach_distance_m: float,
    count: int,
) -> list[np.ndarray]:
    half_extent = np.asarray(extent_xy, dtype=float)[:2] * 0.5
    points: list[np.ndarray] = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        direction = np.asarray([math.cos(angle), math.sin(angle)], dtype=float)
        scales = [
            half_extent[axis] / abs(direction[axis])
            for axis in range(2)
            if abs(direction[axis]) > 1e-12
        ]
        boundary_scale = min(scales)
        boundary = np.asarray(center_xy, dtype=float)[:2] + boundary_scale * direction
        points.append(boundary + approach_distance_m * direction)
    return points


def _point_to_bbox_distance(point_xy: np.ndarray, center_xy: np.ndarray, extent_xy: np.ndarray) -> float:
    return float(_points_to_bbox_distance(np.asarray(point_xy, dtype=float).reshape(1, 2), center_xy, extent_xy)[0])


def _points_to_bbox_distance(points_xy: np.ndarray, center_xy: np.ndarray, extent_xy: np.ndarray) -> np.ndarray:
    half_extent = np.asarray(extent_xy, dtype=float)[:2] * 0.5
    outside = np.maximum(np.abs(np.asarray(points_xy, dtype=float) - np.asarray(center_xy, dtype=float)[:2]) - half_extent, 0.0)
    return np.linalg.norm(outside, axis=1)


def _component_contains(component: FreeComponent | set[tuple[int, int]], rc: tuple[int, int]) -> bool:
    if isinstance(component, FreeComponent):
        return component.contains(rc)
    return rc in component


def _canonical_class(tag: str) -> str:
    normalized = " ".join(str(tag).strip().lower().replace("_", " ").split())
    return normalized or "default"


def _class_value(values: Mapping[str, Any], tag: str, default: Any) -> Any:
    if tag in values:
        return values[tag]
    if tag in {"car", "truck", "bus", "van"} and "vehicle" in values:
        return values["vehicle"]
    return values.get("default", values.get("_default", default))


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    try:
        return {str(key): value[key] for key in value.keys()}
    except (AttributeError, TypeError):
        return {}


def _value_mapping(value: Any, converter, default: Any) -> dict[str, Any]:
    mapping = _mapping(value)
    if not mapping:
        resolved = default if value is None else value
        return {"default": converter(resolved)}
    return {str(key): converter(item) for key, item in mapping.items()}


def _float_mapping(value: Any) -> dict[str, float]:
    return {key: float(item) for key, item in _mapping(value).items()}


def _empty_scoring_components(candidate: CandidateTarget) -> dict[str, Any]:
    raw = {
        "semantic_score": float(candidate.score),
        "relation_score": float(candidate.relation_score),
        "path_cost": None,
        "clearance_score": None,
        "fallback_penalty": None,
        "dynamic_risk": float(candidate.dynamic_risk),
    }
    return {
        "raw": raw,
        "normalized": {key: None for key in raw},
        "weighted": {key: None for key in raw},
    }


def _array_or_none(value: np.ndarray | None) -> list[float] | None:
    return None if value is None else np.asarray(value).tolist()


def _tuple_or_none(value: tuple[int, int] | None) -> list[int] | None:
    return None if value is None else [int(value[0]), int(value[1])]


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)
