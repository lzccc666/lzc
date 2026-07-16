from __future__ import annotations

import json
import os
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
from conceptgraph.navigation.planner import (
    astar_plan,
    astar_plan_with_stats,
    compute_clearance_bias_grid,
    compute_semantic_bias_grid,
    evaluate_reachable_goal_near_point_from_start,
    find_reachable_goal_near_point,
    find_reachable_goal_near_point_from_start,
    free_component,
    load_occupancy_grid,
    path_grid_to_world,
    smooth_grid_path,
)
from conceptgraph.navigation.semantic import parse_navigation_query, rank_candidate_targets
from conceptgraph.navigation.types import NavigationPlan

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
    def __init__(self, cfg):
        self.cfg = cfg
        self._viz_occupancy_grid = None
        self._nodes = None
        self._edges = None
        self._occupancy_grid = None

    def run(self, query_text: str, vehicle_pose: np.ndarray) -> NavigationPlan:
        run_start = time.perf_counter()
        vehicle_pose = np.asarray(vehicle_pose, dtype=float)
        vehicle_xy = vehicle_pose[:2]

        nodes = self._load_nodes()
        edges = self._load_edges()
        occupancy_grid = self._get_occupancy_grid(nodes)

        parsed_query = parse_navigation_query(query_text, self.cfg.default_near_distance_m)
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
        )
        if not candidates:
            raise RuntimeError(f"No semantic target found for query: {query_text}")
        top1_semantic_node_id = int(candidates[0].node_id)

        start_rc = occupancy_grid.world_to_grid(vehicle_xy)
        start_reachable = find_reachable_goal_near_point(
            occupancy_grid,
            vehicle_xy,
            self.cfg.start_search_radius_m,
        )
        if start_reachable is None:
            raise RuntimeError("Current vehicle pose is not connected to a free cell in the occupancy grid.")
        _, start_rc = start_reachable
        start_component = free_component(occupancy_grid, start_rc)
        if not start_component:
            raise RuntimeError("Current vehicle pose has no connected free component in the occupancy grid.")

        best_plan = None
        last_failure = "no candidate evaluated"
        for rank, candidate in enumerate(candidates[: self.cfg.max_candidates], start=1):
            node = nodes[candidate.node_id]
            target_rc = occupancy_grid.world_to_grid(node.bbox_center[:2])
            candidate.debug["target_grid_raw"] = list(target_rc)
            candidate.debug["target_in_map"] = occupancy_grid.in_bounds(target_rc)
            candidate.debug["target_free"] = occupancy_grid.is_free(target_rc)
            reachable = evaluate_reachable_goal_near_point_from_start(
                occupancy_grid,
                node.bbox_center[:2],
                start_rc,
                self.cfg.goal_search_radius_m,
                allow_global_fallback=bool(getattr(self.cfg, "allow_goal_fallback", True)),
                component=start_component,
            )
            if reachable is None:
                reason = "no_free_goal_nearby"
                if not occupancy_grid.in_bounds(target_rc):
                    reason = "out_of_map"
                elif not occupancy_grid.is_free(target_rc):
                    reason = "occupied_target"
                candidate.debug["reachability"] = reason
                candidate.reachability_reason = reason
                last_failure = f"candidate {candidate.node_id} failed reachability: {reason}"
                continue

            goal_xy, goal_rc = reachable.goal_xy, reachable.goal_rc
            candidate.debug.update(
                {
                    "reachability": "reachable",
                    "reachability_reason": reachable.reason,
                    "fallback_used": reachable.fallback_used,
                    "fallback_distance_m": reachable.fallback_distance_m,
                    "selected_goal_xy": goal_xy.tolist(),
                }
            )
            semantic_bias = compute_semantic_bias_grid(
                occupancy_grid,
                goal_rc,
                self.cfg.semantic_cost_weight,
            )
            clearance_bias = compute_clearance_bias_grid(occupancy_grid, self.cfg.clearance_cost_weight)
            planned = astar_plan_with_stats(
                occupancy_grid,
                start_rc,
                goal_rc,
                semantic_bias=semantic_bias + clearance_bias,
                clearance_weight=self.cfg.clearance_cost_weight,
            )
            if planned is None:
                candidate.debug["reachability"] = "astar_failed"
                candidate.reachability_reason = "astar_failed"
                last_failure = f"candidate {candidate.node_id} path planning failed"
                continue

            path_grid_raw, path_cost = planned.path, planned.cost
            smooth_iterations = int(getattr(self.cfg, "path_smoothing_iterations", 2))
            path_grid = smooth_grid_path(occupancy_grid, path_grid_raw, chaikin_iterations=smooth_iterations)
            candidate.reachable = True
            candidate.target_point = np.asarray([goal_xy[0], goal_xy[1], node.bbox_center[2]], dtype=float)
            candidate.target_grid = goal_rc
            candidate.reachable_rank = rank

            path_world = path_grid_to_world(occupancy_grid, path_grid, z=float(vehicle_pose[2] if vehicle_pose.shape[0] > 2 else 0.0))
            path_world_raw = path_grid_to_world(occupancy_grid, path_grid_raw, z=float(vehicle_pose[2] if vehicle_pose.shape[0] > 2 else 0.0))
            best_plan = NavigationPlan(
                query=parsed_query,
                selected_node_id=node.node_id,
                selected_object_tag=node.object_tag,
                selected_target_point=candidate.target_point,
                path_world=path_world,
                path_grid=path_grid,
                path_cost=float(path_cost),
                candidate_targets=candidates,
                occupancy_origin_xy=occupancy_grid.origin_xy,
                occupancy_resolution=float(occupancy_grid.resolution),
                occupancy_shape=occupancy_grid.shape,
                path_world_raw=path_world_raw,
                top1_semantic_node_id=top1_semantic_node_id,
                fallback_used=bool(reachable.fallback_used),
                fallback_reason=reachable.reason,
                fallback_distance_m=float(reachable.fallback_distance_m),
                planning_time_s=float(time.perf_counter() - run_start),
                debug={
                    "selected_candidate_score": candidate.score,
                    "selected_candidate_debug": candidate.debug,
                    "planning_time_s": float(time.perf_counter() - run_start),
                    "top1_semantic_node_id": top1_semantic_node_id,
                    "selected_reachable_rank": rank,
                    "astar_expanded_nodes": planned.expanded_nodes,
                    "raw_astar_waypoints": len(path_grid_raw),
                    "smoothed_waypoints": len(path_grid),
                    "path_smoothing_iterations": smooth_iterations,
                    "start_grid": list(start_rc),
                    "goal_grid": list(goal_rc),
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
                            "node_id": int(c.node_id),
                            "reason": c.reachability_reason or c.debug.get("reachability"),
                            "score": float(c.score),
                        }
                        for c in candidates[: self.cfg.max_candidates]
                    ],
                },
            )
            break

        if best_plan is None:
            raise RuntimeError(f"Semantic target found but all path plans failed: {last_failure}")
        return best_plan

    def save_plan(self, plan: NavigationPlan, output_dir: str | Path) -> Path:
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
            "candidates": [
                {
                    "node_id": cand.node_id,
                    "score": cand.score,
                    "text_score": cand.text_score,
                    "relation_score": cand.relation_score,
                    "distance_prior": cand.distance_prior,
                    "reachable": cand.reachable,
                    "target_point": None if cand.target_point is None else cand.target_point.tolist(),
                    "target_grid": None if cand.target_grid is None else list(cand.target_grid),
                    "reachability_reason": cand.reachability_reason,
                    "reachable_rank": cand.reachable_rank,
                    "debug": cand.debug,
                }
                for cand in plan.candidate_targets
            ],
            "debug": plan.debug,
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self.save_plan_visualization(plan, output_dir)
        self.save_astar_visualization(plan, output_dir)
        self.save_candidate_visualization(plan, output_dir)
        self.save_reachable_goal_visualization(plan, output_dir)
        return output_path

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
