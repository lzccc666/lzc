from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
import numpy as np

from conceptgraph.navigation.io import load_scene_graph_edges, load_scene_graph_nodes, load_scene_map
from conceptgraph.navigation.lane_constraints import load_lane_topology, project_to_legal_lane
from conceptgraph.navigation.planner import (
    find_reachable_goal_near_point,
    free_component,
    load_occupancy_grid,
    path_grid_to_world,
)
from conceptgraph.navigation.semantic import (
    apply_qwen_query_parse,
    parse_navigation_query,
    rank_candidate_targets,
)
from conceptgraph.navigation.target_selector import (
    TargetSelectionConfig,
    TargetSelector,
    candidate_target_to_dict,
    target_selection_result_to_dict,
)
from conceptgraph.navigation.types import NavigationPlan, TargetSelectionResult

CHINESE_FONT = FontProperties(fname="/root/autodl-tmp/lizichong/checkpoint/fonts/NotoSansCJKsc-Regular.otf")


def _configure_chinese_font() -> None:
    font_path = Path("/root/autodl-tmp/lizichong/checkpoint/fonts/NotoSansCJKsc-Regular.otf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        global CHINESE_FONT
        CHINESE_FONT = FontProperties(fname=str(font_path))
        plt.rcParams["font.family"] = [CHINESE_FONT.get_name()]
    plt.rcParams["axes.unicode_minus"] = False


_configure_chinese_font()


class SemanticNavigationPipeline:
    def __init__(self, cfg, semantic_delegate=None):
        self.cfg = cfg
        self._semantic_delegate = semantic_delegate
        self._semantic_delegate_initialized = semantic_delegate is not None
        self._last_query_audit = None
        self._viz_occupancy_grid = None
        self._nodes = None
        self._edges = None
        self._occupancy_grid = None
        self._lane_topology = None

    def set_in_memory_inputs(self, *, nodes, edges, occupancy_grid) -> None:
        """Inject a live navigation snapshot and bypass all final-graph file loads."""
        self._nodes = dict(nodes)
        self._edges = list(edges)
        self._occupancy_grid = occupancy_grid
        self._viz_occupancy_grid = occupancy_grid

    def select(
        self,
        query_text: str,
        vehicle_pose: np.ndarray,
        *,
        parsed_query=None,
        target_node_id: int | None = None,
        run_semantic_delegate: bool = True,
    ) -> TargetSelectionResult:
        run_start = time.perf_counter()
        vehicle_pose = np.asarray(vehicle_pose, dtype=float)
        vehicle_xy = vehicle_pose[:2]
        if parsed_query is None:
            parsed_query, query_audit = self._parse_query_once(query_text)
        else:
            query_audit = None
        self._last_query_audit = query_audit

        try:
            nodes = self._load_nodes()
            edges = self._load_edges()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return self._failure_result(
                mode=self._selection_config().mode,
                code="INPUT_LOAD_FAILED",
                reason=f"Failed to load navigation inputs: {exc}",
                started=run_start,
            )
        if not nodes:
            return self._failure_result(
                mode=self._selection_config().mode,
                code="EMPTY_SCENE_GRAPH",
                reason="Scene graph contains no target nodes.",
                started=run_start,
            )
        try:
            occupancy_grid = self._get_occupancy_grid(nodes)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return self._failure_result(
                mode=self._selection_config().mode,
                code="OCCUPANCY_GRID_FAILED",
                reason=f"Failed to build or load occupancy grid: {exc}",
                started=run_start,
            )

        if vehicle_pose.shape[0] >= 5:
            parsed_query.vehicle_heading_xy = np.asarray(vehicle_pose[3:5], dtype=float)
        candidates = rank_candidate_targets(
            parsed_query=parsed_query,
            nodes=nodes,
            edges=edges,
            vehicle_xy=vehicle_xy,
            lambda_text=self.cfg.lambda_text,
            lambda_relation=self.cfg.lambda_relation,
            lambda_distance=self.cfg.lambda_distance_prior,
            minimum_edge_confidence=float(
                getattr(self.cfg, "relation_edge_minimum_confidence", 0.35)
            ),
            relation_coordinate_frame=str(
                getattr(self.cfg, "relation_coordinate_frame", "world")
            ),
            relation_coordinate_frame_fallback=str(
                getattr(self.cfg, "relation_coordinate_frame_fallback", "world")
            ),
            relation_trajectory=self._load_relation_trajectory(),
            evidence_policy=getattr(self.cfg, "target_evidence_policy", None),
        )
        if not candidates:
            return self._failure_result(
                mode=self._selection_config().mode,
                code="NO_SEMANTIC_CANDIDATES",
                reason=f"No semantic target found for query: {query_text}",
                started=run_start,
            )
        if target_node_id is not None:
            candidates = [
                candidate for candidate in candidates if candidate.node_id == int(target_node_id)
            ]
            if not candidates:
                return self._failure_result(
                    mode=self._selection_config().mode,
                    code="LOCKED_TARGET_NOT_CANDIDATE",
                    reason=f"Locked target {int(target_node_id)} is not a semantic candidate.",
                    started=run_start,
                )
        top1_semantic_node_id = int(candidates[0].node_id)

        selection_config = self._selection_config()
        lane_topology = None
        if selection_config.mode == "lane_constrained":
            try:
                lane_topology = self._get_lane_topology()
                from conceptgraph.navigation.lane_constraints import check_lane_grid_alignment
                check_lane_grid_alignment(occupancy_grid, lane_topology)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return self._failure_result(
                    mode=selection_config.mode,
                    code="NO_LANE_MAP",
                    reason=f"Lane-constrained planning requires a valid lane map: {exc}",
                    started=run_start,
                    candidates=candidates,
                )
            start_projection = project_to_legal_lane(
                occupancy_grid,
                lane_topology,
                vehicle_xy,
                max_distance_m=float(getattr(self.cfg, "lane_projection_max_distance_m", 8.0)),
            )
            if start_projection is None:
                return self._failure_result(
                    mode=selection_config.mode,
                    code="NO_LEGAL_LANE_START",
                    reason="Vehicle pose cannot be projected to a free lane cell within the configured error bound.",
                    started=run_start,
                    candidates=candidates,
                )
            start_rc = start_projection[0]
            start_component = free_component(occupancy_grid, start_rc)
        else:
            start_reachable = find_reachable_goal_near_point(
                occupancy_grid,
                vehicle_xy,
                self.cfg.start_search_radius_m,
            )
            if start_reachable is None:
                return self._failure_result(
                    mode=selection_config.mode,
                    code="NO_FREE_START_CELL",
                    reason="Current vehicle pose has no free cell within the configured start search radius.",
                    started=run_start,
                    candidates=candidates,
                )
            _, start_rc = start_reachable
            start_component = free_component(occupancy_grid, start_rc)
            if not start_component:
                return self._failure_result(
                    mode=selection_config.mode,
                    code="EMPTY_START_COMPONENT",
                    reason="Current vehicle pose has no connected free component in the occupancy grid.",
                    started=run_start,
                    candidates=candidates,
                )
        selector = TargetSelector(
            grid=occupancy_grid,
            nodes=nodes,
            start_rc=start_rc,
            start_component=start_component,
            config=selection_config,
            semantic_cost_weight=float(getattr(self.cfg, "semantic_cost_weight", 0.0)),
            clearance_cost_weight=float(getattr(self.cfg, "clearance_cost_weight", 0.0)),
            path_smoothing_iterations=int(getattr(self.cfg, "path_smoothing_iterations", 2)),
            legacy_max_candidates=int(getattr(self.cfg, "max_candidates", selection_config.top_k)),
            legacy_goal_search_radius_m=float(getattr(self.cfg, "goal_search_radius_m", 25.0)),
            legacy_allow_goal_fallback=bool(getattr(self.cfg, "allow_goal_fallback", True)),
            lane_topology=lane_topology,
            vehicle_xy=vehicle_xy,
            lane_projection_max_distance_m=float(getattr(self.cfg, "lane_projection_max_distance_m", 8.0)),
            lane_centerline_weight=float(getattr(self.cfg, "lane_centerline_weight", 0.25)),
            lane_direction_tolerance_deg=float(getattr(self.cfg, "lane_direction_tolerance_deg", 75.0)),
            start_heading_xy=vehicle_pose[3:5] if vehicle_pose.shape[0] >= 5 else None,
        )
        result = selector.select(candidates)
        result.timing["total_s"] = float(time.perf_counter() - run_start)
        if query_audit is not None:
            result.debug["qwen_query_parse"] = query_audit
        rerank_audit = (
            self._rerank_verified_candidates(query_text, result, edges)
            if run_semantic_delegate
            else None
        )
        if rerank_audit is not None:
            result.debug["qwen_candidate_rerank"] = rerank_audit
        if not result.success or result.selected_candidate is None:
            return result

        selected = result.selected_candidate
        selected_goal = next(
            (
                goal
                for goal in selected.goal_candidates
                if goal.path_found and goal.goal_rc == selected.target_grid
            ),
            None,
        )
        if selected_goal is None:
            return self._failure_result(
                mode=result.mode,
                code="INTERNAL_SELECTED_PATH_MISSING",
                reason="Selected target has no corresponding validated path.",
                started=run_start,
                candidates=result.evaluated_candidates,
                scoring_weights=result.scoring_weights,
            )

        node = nodes[selected.node_id]
        z = float(vehicle_pose[2] if vehicle_pose.shape[0] > 2 else 0.0)
        path_world = path_grid_to_world(occupancy_grid, selected_goal.path_grid, z=z)
        path_world_raw = path_grid_to_world(occupancy_grid, selected_goal.path_grid_raw, z=z)
        fallback_reason = selected_goal.fallback_reason
        fallback_used = bool(selected.debug.get("fallback_used", False))
        if result.mode == "joint_utility":
            fallback_used = fallback_reason not in {None, "grid_quantization"}
        planning_time_s = float(time.perf_counter() - run_start)
        plan = NavigationPlan(
            query=parsed_query,
            selected_node_id=node.node_id,
            selected_object_tag=node.object_tag,
            selected_target_point=selected.target_point,
            path_world=path_world,
            path_grid=selected_goal.path_grid,
            path_cost=float(selected_goal.path_cost),
            candidate_targets=candidates,
            occupancy_origin_xy=occupancy_grid.origin_xy,
            occupancy_resolution=float(occupancy_grid.resolution),
            occupancy_shape=occupancy_grid.shape,
            path_world_raw=path_world_raw,
            top1_semantic_node_id=top1_semantic_node_id,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            fallback_distance_m=float(selected_goal.fallback_distance_m or 0.0),
            planning_time_s=planning_time_s,
            target_selection_mode=result.mode,
            selection_utility=selected.selection_utility,
            scoring_weights=result.scoring_weights,
            selection_timing=result.timing,
            debug={
                "selected_candidate_score": selected.score,
                "selected_candidate_debug": selected.debug,
                "planning_time_s": planning_time_s,
                "top1_semantic_node_id": top1_semantic_node_id,
                "selected_reachable_rank": selected.reachable_rank,
                "astar_expanded_nodes": selected_goal.expanded_nodes,
                "raw_astar_waypoints": len(selected_goal.path_grid_raw),
                "smoothed_waypoints": len(selected_goal.path_grid),
                "path_smoothing_iterations": int(getattr(self.cfg, "path_smoothing_iterations", 2)),
                "continuous_collision_check": True,
                "start_grid": list(start_rc),
                "goal_grid": list(selected_goal.goal_rc),
                "occupancy_free_ratio": float((occupancy_grid.occupancy == 0).mean()),
                "occupancy_source": occupancy_grid.source,
                "occupancy_fallback_used": occupancy_grid.fallback_used,
                "occupancy_metadata": occupancy_grid.metadata or {},
                "obstacle_profile": str(getattr(self.cfg, "obstacle_profile", "thesis")),
                "robot_radius_m": float(getattr(self.cfg, "robot_radius_m", 0.8)),
                "object_bbox_obstacle_scale": float(getattr(self.cfg, "object_bbox_obstacle_scale", 0.18)),
                "max_object_obstacle_radius_m": float(getattr(self.cfg, "max_object_obstacle_radius_m", 1.2)),
                "goal_search_radius_m": float(getattr(self.cfg, "goal_search_radius_m", 20.0)),
                "candidate_failure_reasons": [
                    {
                        "node_id": int(candidate.node_id),
                        "reason": candidate.rejection_reason or candidate.reachability_reason,
                        "score": float(candidate.score),
                    }
                    for candidate in result.evaluated_candidates
                ],
                "relation_query": self._relation_query_diagnostics(selected),
                "qwen_query_parse": query_audit,
                "qwen_candidate_rerank": rerank_audit,
            },
        )
        result.plan = plan
        return result

    def run(self, query_text: str, vehicle_pose: np.ndarray) -> NavigationPlan:
        """Legacy success API; use select() when business failures must stay structured."""
        result = self.select(query_text, vehicle_pose)
        if not result.success or result.plan is None:
            raise RuntimeError(f"{result.failure_code}: {result.failure_reason}")
        return result.plan

    def save_selection_result(
        self,
        result: TargetSelectionResult,
        output_dir: str | Path,
        save_plan: bool = True,
        save_visualizations: bool = False,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "target_selection_result.json"
        payload = target_selection_result_to_dict(result)
        if result.success and result.plan is not None and save_plan:
            plan_path = self.save_plan(result.plan, output_dir, save_visualizations=save_visualizations)
            payload["plan_path"] = str(plan_path)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        return output_path

    def save_plan(
        self,
        plan: NavigationPlan,
        output_dir: str | Path,
        save_visualizations: bool = True,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "semantic_navigation_plan.json"
        payload = {
            "query": {
                "raw_text": plan.query.raw_text,
                "target_text": plan.query.target_text,
                "anchor_text": plan.query.anchor_text,
                "relation": plan.query.relation,
                "prefer_nearest": plan.query.prefer_nearest,
                "prefer_front": plan.query.prefer_front,
                "target_categories": plan.query.target_categories,
                "target_colors": plan.query.target_colors,
                "anchor_colors": plan.query.anchor_colors,
                "display_text": plan.query.display_text or plan.query.raw_text,
                "query_mode": plan.query.query_mode,
                "resolved_instruction": self._resolved_instruction(plan.query),
            },
            "selected_target": {
                "node_id": plan.selected_node_id,
                "object_tag": plan.selected_object_tag,
                "target_point": plan.selected_target_point.tolist(),
                "top1_semantic_node_id": plan.top1_semantic_node_id,
                "top1_reachable_same_as_selected": plan.top1_semantic_node_id == plan.selected_node_id,
                "fallback_used": plan.fallback_used,
                "fallback_reason": plan.fallback_reason,
                "fallback_distance_m": plan.fallback_distance_m,
                "semantic_rank": next(
                    (candidate.semantic_rank for candidate in plan.candidate_targets if candidate.node_id == plan.selected_node_id),
                    None,
                ),
                "selection_utility": plan.selection_utility,
            },
            "path": {
                "world_xyz": plan.path_world.tolist(),
                "world_xyz_raw_astar": [] if plan.path_world_raw is None else plan.path_world_raw.tolist(),
                "grid_rc": [list(item) for item in plan.path_grid],
                "path_cost": plan.path_cost,
            },
            "occupancy_grid": {
                "origin_xy": plan.occupancy_origin_xy.tolist(),
                "resolution": plan.occupancy_resolution,
                "shape": list(plan.occupancy_shape),
            },
            "target_selection": {
                "success": True,
                "mode": plan.target_selection_mode,
                "scoring_weights": plan.scoring_weights,
                "timing": plan.selection_timing,
            },
            "candidates": [candidate_target_to_dict(candidate) for candidate in plan.candidate_targets],
            "debug": plan.debug,
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        if save_visualizations:
            self.save_plan_visualization(plan, output_dir)
            self.save_astar_visualization(plan, output_dir)
            self.save_candidate_visualization(plan, output_dir)
            self.save_reachable_goal_visualization(plan, output_dir)
        return output_path

    def _selection_config(self) -> TargetSelectionConfig:
        return TargetSelectionConfig.from_pipeline_config(self.cfg)

    def _failure_result(
        self,
        mode: str,
        code: str,
        reason: str,
        started: float,
        candidates=None,
        scoring_weights=None,
    ) -> TargetSelectionResult:
        weights = scoring_weights or TargetSelectionConfig.from_pipeline_config(self.cfg).utility_weights
        result = TargetSelectionResult(
            success=False,
            mode=mode,
            selected_candidate=None,
            evaluated_candidates=list(candidates or []),
            failure_code=code,
            failure_reason=reason,
            scoring_weights={key: float(value) for key, value in weights.items()},
            timing={"total_s": float(time.perf_counter() - started)},
        )
        if self._last_query_audit is not None:
            result.debug["qwen_query_parse"] = self._last_query_audit
        return result

    def _parse_query_once(self, query_text: str):
        default_distance = float(self.cfg.default_near_distance_m)
        rule_query = parse_navigation_query(query_text, default_distance)
        delegate = self._get_semantic_delegate()
        if delegate is None:
            return rule_query, None
        try:
            audit = delegate.parse_query(
                query_text,
                default_near_distance_m=default_distance,
                rule_result=rule_query,
            )
        except Exception as exc:  # noqa: BLE001 - injected delegates cannot break navigation
            audit = {
                "task": "query_parse",
                "used_qwen": False,
                "fallback_used": True,
                "fallback_reason": f"delegate exception: {type(exc).__name__}: {str(exc)[:240]}",
                "failure_code": "DELEGATE_EXCEPTION",
                "normalized_output": None,
            }
            return rule_query, audit
        if not audit.get("used_qwen", False):
            return rule_query, audit
        return apply_qwen_query_parse(
            rule_query,
            audit.get("normalized_output"),
            default_distance,
        ), audit

    def _rerank_verified_candidates(self, query_text, result, edges):
        delegate = self._get_semantic_delegate()
        if delegate is None or not result.success or result.selected_candidate is None:
            return None
        verified = [
            candidate
            for candidate in result.evaluated_candidates
            if candidate.reachable
            and any(goal.path_found and goal.rejection_reason is None for goal in candidate.goal_candidates)
        ]
        if not verified:
            return None
        selected_id = int(result.selected_candidate.node_id)
        verified.sort(
            key=lambda candidate: (
                0 if candidate.node_id == selected_id else 1,
                -float(
                    candidate.selection_utility
                    if candidate.selection_utility is not None
                    else float("-inf")
                ),
                int(candidate.semantic_rank or 10**9),
                int(candidate.node_id),
            )
        )
        payload = [
            {
                "node_id": int(candidate.node_id),
                "semantic_rank": candidate.semantic_rank,
                "stage2_scores": {
                    "combined_semantic": float(candidate.score),
                    "text": float(candidate.text_score),
                    "relation": float(candidate.relation_score),
                    "distance_prior": float(candidate.distance_prior),
                    "joint_utility": candidate.selection_utility,
                    "path_cost": candidate.path_cost,
                    "path_length_m": candidate.path_length_m,
                    "min_clearance_m": candidate.min_clearance_m,
                },
                "path_validated": True,
            }
            for candidate in verified
        ]
        ids = {item["node_id"] for item in payload}
        edge_payload = [
            edge.to_dict()
            for edge in edges
            if int(edge.source) in ids or int(edge.target) in ids
        ]
        try:
            audit = delegate.rerank_candidates(query_text, payload, edge_payload)
        except Exception as exc:  # noqa: BLE001 - reranking is advisory
            return {
                "used_qwen": False,
                "fallback_used": True,
                "failure_code": "DELEGATE_EXCEPTION",
                "fallback_reason": f"delegate exception: {type(exc).__name__}: {str(exc)[:240]}",
                "verified_candidate_ids": [item["node_id"] for item in payload],
                "authoritative_selected_node_id": selected_id,
                "selection_applied": False,
            }
        recommendation = audit.get("selected_node_id")
        audit["verified_candidate_ids"] = [item["node_id"] for item in payload]
        audit["authoritative_selected_node_id"] = selected_id
        audit["recommendation_changed"] = recommendation != selected_id
        audit["selection_applied"] = False
        audit["selection_policy"] = "stage2_joint_utility_and_validated_path_are_authoritative"
        return audit

    def _get_semantic_delegate(self):
        if self._semantic_delegate_initialized:
            return self._semantic_delegate
        self._semantic_delegate_initialized = True
        qwen_cfg = getattr(self.cfg, "qwen_semantic", None)
        if qwen_cfg is None:
            self._semantic_delegate = None
            return None
        from conceptgraph.vlm.qwen_semantic_delegate import QwenDelegateConfig, QwenSemanticDelegate

        fields = QwenDelegateConfig.__dataclass_fields__
        values = {
            name: getattr(qwen_cfg, name)
            for name in fields
            if hasattr(qwen_cfg, name) and getattr(qwen_cfg, name) is not None
        }
        self._semantic_delegate = QwenSemanticDelegate(QwenDelegateConfig(**values))
        return self._semantic_delegate

    def _resolved_instruction(self, query) -> str:
        return query.display_text or query.raw_text

    def save_plan_visualization(self, plan: NavigationPlan, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "semantic_navigation_plan.png"

        path_xy = plan.path_world[:, :2] if len(plan.path_world) > 0 else np.empty((0, 2))
        raw_path = plan.path_world_raw if plan.path_world_raw is not None else plan.path_world
        raw_path_xy = raw_path[:, :2] if len(raw_path) > 0 else np.empty((0, 2))
        candidates_xy = np.asarray(
            [
                cand.target_point[:2]
                for cand in plan.candidate_targets
                if cand.target_point is not None
            ],
            dtype=float,
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        background_handles = self._draw_occupancy_background(ax)
        display_text = plan.query.display_text or plan.query.raw_text
        ax.set_title(f"语义导航：{display_text}", fontproperties=CHINESE_FONT)
        ax.set_xlabel("x (米)", fontproperties=CHINESE_FONT)
        ax.set_ylabel("y (米)", fontproperties=CHINESE_FONT)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")

        if candidates_xy.size > 0:
            ax.scatter(candidates_xy[:, 0], candidates_xy[:, 1], s=30, c="#8a8a8a", label="候选目标")
        if path_xy.size > 0:
            if raw_path_xy.size > 0:
                ax.plot(raw_path_xy[:, 0], raw_path_xy[:, 1], "--", c="#94a3b8", linewidth=1.3, label="A*原始折线")
            ax.plot(path_xy[:, 0], path_xy[:, 1], c="#1f77b4", linewidth=2.4, label="平滑轨迹")
            ax.scatter(path_xy[0, 0], path_xy[0, 1], marker="o", s=70, c="#2ca02c", label="起点")
            ax.scatter(path_xy[-1, 0], path_xy[-1, 1], marker="*", s=160, c="#d62728", label="选中目标")

        self._legend_with_background(ax, background_handles)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return output_path

    def save_astar_visualization(self, plan: NavigationPlan, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_path = output_dir / "astar_path.png"
        path_xy = plan.path_world[:, :2] if len(plan.path_world) > 0 else np.empty((0, 2))
        raw_path = plan.path_world_raw if plan.path_world_raw is not None else plan.path_world
        raw_path_xy = raw_path[:, :2] if len(raw_path) > 0 else np.empty((0, 2))

        fig, ax = plt.subplots(figsize=(8, 8))
        background_handles = self._draw_occupancy_background(ax)
        display_text = plan.query.display_text or plan.query.raw_text
        ax.set_title(f"A*路径：{display_text}", fontproperties=CHINESE_FONT)
        ax.set_xlabel("x (米)", fontproperties=CHINESE_FONT)
        ax.set_ylabel("y (米)", fontproperties=CHINESE_FONT)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")

        if path_xy.size > 0:
            if raw_path_xy.size > 0:
                ax.plot(raw_path_xy[:, 0], raw_path_xy[:, 1], "--", c="#94a3b8", linewidth=1.4, label="A*原始折线")
            ax.plot(path_xy[:, 0], path_xy[:, 1], c="#1f77b4", linewidth=2.5, label="平滑轨迹")
            ax.scatter(path_xy[0, 0], path_xy[0, 1], marker="o", s=70, c="#2ca02c", label="起点")
            ax.scatter(path_xy[-1, 0], path_xy[-1, 1], marker="*", s=160, c="#d62728", label="终点")
        self._legend_with_background(ax, background_handles)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return output_path

    def save_candidate_visualization(self, plan: NavigationPlan, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_path = output_dir / "navigation_candidates.png"
        fig, ax = plt.subplots(figsize=(8, 8))
        background_handles = self._draw_occupancy_background(ax)
        display_text = plan.query.display_text or plan.query.raw_text
        ax.set_title(f"候选目标：{display_text}", fontproperties=CHINESE_FONT)
        ax.set_xlabel("x (米)", fontproperties=CHINESE_FONT)
        ax.set_ylabel("y (米)", fontproperties=CHINESE_FONT)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        for rank, cand in enumerate(plan.candidate_targets[:10], start=1):
            raw_xy = cand.debug.get("selected_goal_xy")
            if cand.target_point is not None:
                xy = cand.target_point[:2]
            elif raw_xy is not None:
                xy = np.asarray(raw_xy, dtype=float)
            else:
                continue
            marker = "*" if cand.node_id == plan.selected_node_id else ("o" if cand.reachable else "x")
            color = "#dc2626" if cand.node_id == plan.selected_node_id else ("#2563eb" if cand.reachable else "#ef4444")
            ax.scatter(xy[0], xy[1], s=150 if marker == "*" else 70, marker=marker, c=color)
            ax.text(xy[0], xy[1], f" {rank}:{cand.node_id} {cand.score:.2f}", fontsize=8, fontproperties=CHINESE_FONT)
        path_xy = plan.path_world[:, :2] if len(plan.path_world) > 0 else np.empty((0, 2))
        raw_path = plan.path_world_raw if plan.path_world_raw is not None else plan.path_world
        raw_path_xy = raw_path[:, :2] if len(raw_path) > 0 else np.empty((0, 2))
        if path_xy.size > 0:
            if raw_path_xy.size > 0:
                ax.plot(raw_path_xy[:, 0], raw_path_xy[:, 1], "--", c="#94a3b8", linewidth=1.0, alpha=0.75)
            ax.plot(path_xy[:, 0], path_xy[:, 1], c="#111827", linewidth=1.8, alpha=0.85)
        self._legend_with_background(ax, background_handles)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return output_path

    def save_reachable_goal_visualization(self, plan: NavigationPlan, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_path = output_dir / "reachable_goal_search.png"
        fig, ax = plt.subplots(figsize=(8, 8))
        background_handles = self._draw_occupancy_background(ax)
        display_text = plan.query.display_text or plan.query.raw_text
        ax.set_title(f"可达点搜索：{display_text}", fontproperties=CHINESE_FONT)
        ax.set_xlabel("x (米)", fontproperties=CHINESE_FONT)
        ax.set_ylabel("y (米)", fontproperties=CHINESE_FONT)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        selected = next((c for c in plan.candidate_targets if c.node_id == plan.selected_node_id), None)
        if selected is not None:
            target_raw = selected.debug.get("target_grid_raw")
            if target_raw is not None:
                raw_rc = tuple(int(v) for v in target_raw)
                raw_xy = plan.occupancy_origin_xy + np.asarray([raw_rc[1], raw_rc[0]], dtype=float) * plan.occupancy_resolution
                ax.scatter(raw_xy[0], raw_xy[1], marker="x", s=90, c="#ef4444", label="原始目标")
                ax.scatter(plan.selected_target_point[0], plan.selected_target_point[1], marker="*", s=170, c="#16a34a", label="回退/可达目标")
                if plan.fallback_used:
                    ax.plot(
                        [raw_xy[0], plan.selected_target_point[0]],
                        [raw_xy[1], plan.selected_target_point[1]],
                        "--",
                        c="#f59e0b",
                        linewidth=1.8,
                        label="回退距离",
                    )
        path_xy = plan.path_world[:, :2] if len(plan.path_world) > 0 else np.empty((0, 2))
        raw_path = plan.path_world_raw if plan.path_world_raw is not None else plan.path_world
        raw_path_xy = raw_path[:, :2] if len(raw_path) > 0 else np.empty((0, 2))
        if path_xy.size > 0:
            if raw_path_xy.size > 0:
                ax.plot(raw_path_xy[:, 0], raw_path_xy[:, 1], "--", c="#94a3b8", linewidth=1.2, label="A*原始折线")
            ax.plot(path_xy[:, 0], path_xy[:, 1], c="#2563eb", linewidth=2.0, label="平滑轨迹")
            ax.scatter(path_xy[0, 0], path_xy[0, 1], marker="o", s=70, c="#111827", label="起点")
        self._legend_with_background(ax, background_handles)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return output_path

    def _draw_occupancy_background(self, ax) -> list[Patch]:
        try:
            grid = self._get_viz_occupancy_grid()
        except Exception:
            return []
        occupancy = np.asarray(grid.occupancy, dtype=np.uint8)
        extent = [
            grid.origin_xy[0],
            grid.origin_xy[0] + (occupancy.shape[1] - 1) * grid.resolution,
            grid.origin_xy[1],
            grid.origin_xy[1] + (occupancy.shape[0] - 1) * grid.resolution,
        ]
        free = np.ma.masked_where(occupancy != 0, np.ones_like(occupancy, dtype=float))
        occupied = np.ma.masked_where(occupancy == 0, np.ones_like(occupancy, dtype=float))
        ax.imshow(
            free,
            origin="lower",
            extent=extent,
            cmap=ListedColormap(["#d1d5db"]),
            alpha=0.42,
            interpolation="nearest",
            zorder=0,
        )
        ax.imshow(
            occupied,
            origin="lower",
            extent=extent,
            cmap=ListedColormap(["#ef4444"]),
            alpha=0.12,
            interpolation="nearest",
            zorder=0,
        )
        return [
            Patch(facecolor="#d1d5db", edgecolor="none", alpha=0.42, label="可行驶区域"),
            Patch(facecolor="#ef4444", edgecolor="none", alpha=0.12, label="障碍/不可通行"),
        ]

    def _get_viz_occupancy_grid(self):
        if self._viz_occupancy_grid is None:
            nodes = self._load_nodes()
            self._viz_occupancy_grid = load_occupancy_grid(
                self.cfg.bev_map_path,
            self.cfg.grid_resolution,
            _build_lightweight_scene_map_from_nodes(nodes),
            self.cfg.grid_padding_m,
            self.cfg.robot_radius_m,
            object_bbox_scale=float(getattr(self.cfg, "object_bbox_obstacle_scale", 0.18)),
            max_object_radius_m=float(getattr(self.cfg, "max_object_obstacle_radius_m", 1.2)),
            ignore_dynamic_obstacles=bool(getattr(self.cfg, "ignore_dynamic_obstacles", False)),
            obstacle_profile=str(getattr(self.cfg, "obstacle_profile", "thesis")),
        )
        return self._viz_occupancy_grid

    def _load_nodes(self):
        if self._nodes is None:
            self._nodes = load_scene_graph_nodes(
                self.cfg.scene_graph_json_path,
                self.cfg.annotated_scenegraph_json_path,
            )
        return self._nodes

    def _load_edges(self):
        if self._edges is None:
            self._edges = load_scene_graph_edges(self.cfg.scene_graph_edge_path)
        return self._edges

    def _load_relation_trajectory(self):
        path = getattr(self.cfg, "relation_trajectory_path", None)
        if path in {None, ""} or not Path(str(path)).exists():
            return None
        poses = np.loadtxt(path)
        if poses.ndim == 1:
            poses = poses.reshape(1, -1)
        trajectory = []
        for row in poses:
            if row.size >= 16:
                trajectory.append(row.reshape(4, 4)[:2, 3])
            elif row.size >= 2:
                trajectory.append(row[:2])
        return None if not trajectory else np.asarray(trajectory, dtype=float)

    def _relation_query_diagnostics(self, selected) -> dict:
        relation_debug = selected.debug.get("relation_debug", {})
        matches = relation_debug.get("matches", [])
        best = matches[0] if matches else {}
        chain = best.get("relation_chain", [])
        return {
            "relation_chain": chain,
            "edge_source": best.get("edge_source"),
            "fallback_reason": best.get("fallback_reason", []),
            "geometry_fallback_count": int(
                best.get("edge_source") == "geometry_fallback"
                and float(best.get("geometry_relation_score", 0.0)) > 0.0
            ),
            "minimum_edge_confidence": float(
                getattr(self.cfg, "relation_edge_minimum_confidence", 0.35)
            ),
        }

    def _get_occupancy_grid(self, nodes):
        if self._occupancy_grid is None:
            scene_map = _build_lightweight_scene_map_from_nodes(nodes)
            if not scene_map:
                scene_map = load_scene_map(self.cfg.scene_map_path)
            self._occupancy_grid = load_occupancy_grid(
                self.cfg.bev_map_path,
                self.cfg.grid_resolution,
                scene_map,
                self.cfg.grid_padding_m,
                self.cfg.robot_radius_m,
                object_bbox_scale=float(getattr(self.cfg, "object_bbox_obstacle_scale", 0.18)),
                max_object_radius_m=float(getattr(self.cfg, "max_object_obstacle_radius_m", 1.2)),
                ignore_dynamic_obstacles=bool(getattr(self.cfg, "ignore_dynamic_obstacles", False)),
                obstacle_profile=str(getattr(self.cfg, "obstacle_profile", "thesis")),
            )
        return self._occupancy_grid

    def _get_lane_topology(self):
        if self._lane_topology is None:
            lane_map_path = getattr(self.cfg, "lane_map_path", None)
            topology_path = getattr(self.cfg, "lane_topology_path", None)
            if lane_map_path in {None, ""}:
                raise FileNotFoundError("lane_map_path is not configured")
            self._lane_topology = load_lane_topology(str(lane_map_path), None if topology_path in {None, ""} else str(topology_path))
        return self._lane_topology

    def _legend_with_background(self, ax, background_handles: list[Patch]) -> None:
        handles, labels = ax.get_legend_handles_labels()
        merged_handles = []
        merged_labels = []
        for handle, label in list(zip(background_handles, [h.get_label() for h in background_handles])) + list(zip(handles, labels)):
            if label in merged_labels:
                continue
            merged_handles.append(handle)
            merged_labels.append(label)
        if merged_handles:
            ax.legend(merged_handles, merged_labels, loc="best", prop=CHINESE_FONT)


def _build_lightweight_scene_map_from_nodes(nodes) -> list[dict]:
    scene_map = []
    for node in nodes.values():
        center = np.asarray(node.bbox_center, dtype=float)
        extent = np.asarray(node.bbox_extent, dtype=float)
        if center.size < 2 or extent.size < 2:
            continue
        if not np.all(np.isfinite(center[:2])) or not np.all(np.isfinite(extent[:2])):
            continue
        if float(np.max(extent[:2])) <= 0:
            continue
        scene_map.append(
            {
                "bbox": {
                    "center": center.tolist(),
                    "extent": np.maximum(extent, 0.1).tolist(),
                },
                "object_tag": node.object_tag,
                "class_name": [node.object_tag],
                "motion_state": node.motion_state,
                "is_short_term": node.is_short_term,
            }
        )
    return scene_map
