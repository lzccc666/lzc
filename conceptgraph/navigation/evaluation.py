from __future__ import annotations

import csv
import gzip
import json
import pickle as pkl
import shutil
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


def save_scene_evaluation(scene_dir: str | Path, exp_dir: str | Path, navigation_dir: str | Path | None = None) -> Path:
    scene_dir = Path(scene_dir)
    exp_dir = Path(exp_dir)
    navigation_dir = Path(navigation_dir) if navigation_dir is not None else exp_dir / "navigation_examples"
    eval_dir = exp_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    scene_metrics = compute_scenegraph_metrics(scene_dir, exp_dir)
    navigation_cases = collect_navigation_cases(scene_dir, exp_dir, navigation_dir)
    retrieval_metrics = [case["target_retrieval"] for case in navigation_cases]
    planning_metrics = [case["path_planning"] for case in navigation_cases]
    reachability_metrics = [case["reachability"] for case in navigation_cases]
    failures = [case["failure"] for case in navigation_cases if case.get("failure")]

    _write_json(eval_dir / "scene_metrics.json", scene_metrics)
    _write_json(eval_dir / "navigation_cases.json", navigation_cases)
    _write_json(eval_dir / "target_retrieval_metrics.json", retrieval_metrics)
    _write_json(eval_dir / "path_planning_metrics.json", planning_metrics)
    _write_json(eval_dir / "reachability_metrics.json", reachability_metrics)
    _write_json(eval_dir / "scenegraph_metrics.json", scene_metrics)
    _write_json(eval_dir / "failures.json", failures)
    _write_summary_csv(eval_dir / "summary.csv", scene_metrics, navigation_cases)
    export_visualization_pipeline(scene_dir, exp_dir, navigation_dir)
    return eval_dir


def compute_scenegraph_metrics(scene_dir: Path, exp_dir: Path) -> dict[str, Any]:
    scene_graph = _load_json(exp_dir / "scene_graph.json", default=[])
    edges = _load_pickle(exp_dir / "cfslam_scenegraph_edges.pkl", default=[])
    objects = _load_map_objects(exp_dir / "map" / "scene_map_cfslam_pruned.pkl.gz")
    detections_total = _count_detections_total(exp_dir / "detections")
    if detections_total == 0:
        detections_total = _count_detections_total(_find_detection_dir(scene_dir))
    poses = _load_poses(scene_dir / "poses.txt")
    obs_counts = [int(obj.get("num_detections", len(obj.get("image_idx", [])) or 0)) for obj in objects]
    dynamic_nodes = [node for node in scene_graph if node.get("motion_state") == "dynamic" or node.get("is_short_term")]
    static_nodes = [node for node in scene_graph if node.get("motion_state") == "static" and not node.get("is_short_term")]
    unknown_motion_nodes = [
        node
        for node in scene_graph
        if str(node.get("motion_state", "unknown")).lower() not in {"static", "dynamic"} and not node.get("is_short_term")
    ]
    invalid_nodes = [node for node in scene_graph if str(node.get("object_tag", "")).lower() in {"invalid", "fail", ""}]
    return {
        "scene_id": scene_dir.name,
        "num_frames": int(len(poses)),
        "num_detections_total": int(detections_total if detections_total > 0 else sum(obs_counts)),
        "num_objects_before_filter": int(len(objects)),
        "num_objects_after_filter": int(len(objects)),
        "num_scenegraph_nodes": int(len(scene_graph)),
        "num_scenegraph_edges": int(len(edges)),
        "avg_edges_per_node": float((2 * len(edges) / len(scene_graph)) if scene_graph else 0.0),
        "num_static_nodes": int(len(static_nodes)),
        "num_dynamic_nodes": int(len(dynamic_nodes)),
        "num_unknown_motion_nodes": int(len(unknown_motion_nodes)),
        "num_invalid_nodes": int(len(invalid_nodes)),
        "duplicate_merge_count": None,
        "object_observation_count_mean": float(mean(obs_counts)) if obs_counts else 0.0,
        "object_observation_count_median": float(median(obs_counts)) if obs_counts else 0.0,
    }


def collect_navigation_cases(scene_dir: Path, exp_dir: Path, navigation_dir: Path) -> list[dict[str, Any]]:
    cases = []
    if not navigation_dir.exists():
        return cases
    for plan_path in sorted(navigation_dir.glob("*/semantic_navigation_plan.json")):
        plan = _load_json(plan_path, default={})
        cases.append(_case_from_plan(scene_dir.name, plan_path, plan))
    return cases


def _case_from_plan(scene_id: str, plan_path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    query = plan.get("query", {})
    selected = plan.get("selected_target", {})
    path = plan.get("path", {})
    candidates = plan.get("candidates", [])
    debug = plan.get("debug", {})
    query_display_text = query.get("display_text") or query.get("raw_text", "")
    reachable_candidates = [cand for cand in candidates if cand.get("reachable")]
    top1 = candidates[0] if candidates else {}
    selected_node_id = selected.get("node_id")
    selected_candidate = next((cand for cand in candidates if cand.get("node_id") == selected_node_id), {})
    world_path = np.asarray(path.get("world_xyz", []), dtype=float)
    path_length = _path_length(world_path)
    clearances = _candidate_clearance_values(selected_candidate)
    fallback_used = bool(selected.get("fallback_used", False))

    retrieval = {
        "scene_id": scene_id,
        "query_text": query_display_text,
        "raw_query_text": query.get("raw_text", ""),
        "parsed_target_text": query.get("target_text", ""),
        "parsed_relation": query.get("relation"),
        "target_categories": query.get("target_categories", []),
        "num_candidates": int(len(candidates)),
        "top1_semantic_node_id": top1.get("node_id"),
        "top1_semantic_score": float(top1.get("score", 0.0) or 0.0),
        "selected_reachable_node_id": selected_node_id,
        "selected_reachable_score": float(selected_candidate.get("score", 0.0) or 0.0),
        "semantic_score": float(selected_candidate.get("score", top1.get("score", 0.0)) or 0.0),
        "top1_reachable_same_as_selected": bool(selected.get("top1_reachable_same_as_selected", False)),
        "topk_reachable_rank": selected_candidate.get("reachable_rank"),
        "relation_score": float(selected_candidate.get("relation_score", 0.0) or 0.0),
        "distance_prior": float(selected_candidate.get("distance_prior", 0.0) or 0.0),
        "target_retrieval_success": debug.get("target_validation", {}).get("valid"),
        "target_validation": debug.get("target_validation"),
    }
    reachability = {
        "candidate_in_map_ratio": _ratio(candidates, lambda c: bool(c.get("debug", {}).get("target_in_map", False))),
        "candidate_reachable_ratio": _ratio(candidates, lambda c: bool(c.get("reachable", False))),
        "occupancy_free_ratio": float(debug.get("occupancy_free_ratio", 0.0) or 0.0),
        "occupancy_source": debug.get("occupancy_source"),
        "occupancy_fallback_used": bool(debug.get("occupancy_fallback_used", False)),
        "obstacle_profile": debug.get("obstacle_profile"),
        "robot_radius_m": debug.get("robot_radius_m"),
        "object_bbox_obstacle_scale": debug.get("object_bbox_obstacle_scale"),
        "max_object_obstacle_radius_m": debug.get("max_object_obstacle_radius_m"),
        "selected_target_in_map": bool(selected_candidate.get("debug", {}).get("target_in_map", False)),
        "selected_target_reachable": bool(selected_candidate.get("reachable", False)),
        "fallback_used": fallback_used,
        "fallback_reason": selected.get("fallback_reason"),
        "fallback_distance_m": float(selected.get("fallback_distance_m", 0.0) or 0.0),
        "goal_search_radius_m": None,
        "start_connected": True,
    }
    planning = {
        "scene_id": scene_id,
        "query_text": query_display_text,
        "planning_success": bool(len(world_path) > 0),
        "path_length_m": float(path_length),
        "path_cost": float(path.get("path_cost", 0.0) or 0.0),
        "num_waypoints": int(len(world_path)),
        "planning_time_s": float(debug.get("planning_time_s", 0.0) or 0.0),
        "raw_astar_waypoints": int(debug.get("raw_astar_waypoints", len(path.get("world_xyz_raw_astar", [])) or 0) or 0),
        "smoothed_waypoints": int(debug.get("smoothed_waypoints", len(world_path)) or 0),
        "min_clearance_m": float(min(clearances)) if clearances else None,
        "mean_clearance_m": float(mean(clearances)) if clearances else None,
        "astar_expanded_nodes": debug.get("astar_expanded_nodes"),
        "start_xy": world_path[0, :2].tolist() if len(world_path) else None,
        "goal_xy": world_path[-1, :2].tolist() if len(world_path) else None,
        "selected_target_xy": selected.get("target_point", [None, None])[:2],
    }
    failure = None
    if not planning["planning_success"] or not reachable_candidates:
        failure = {
            "query_text": query.get("raw_text", ""),
            "scene_id": scene_id,
            "failed_stage": "navigation",
            "failure_reason": debug.get("last_failure", "no_reachable_candidate"),
            "candidate_debug": [cand.get("debug", {}) for cand in candidates[:10]],
            "suggested_fix": _suggest_fix(candidates),
        }
    return {
        "scene_id": scene_id,
        "case_dir": str(plan_path.parent),
        "target_retrieval": retrieval,
        "reachability": reachability,
        "path_planning": planning,
        "failure": failure,
    }


def _load_map_objects(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with gzip.open(path, "rb") as f:
        payload = pkl.load(f)
    if isinstance(payload, dict) and "objects" in payload:
        return payload["objects"]
    if isinstance(payload, list):
        return payload
    return []


def _count_detection_files(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob("*.pkl.gz")))


def _count_detections_total(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for det_path in sorted(path.glob("*.pkl.gz")):
        try:
            with gzip.open(det_path, "rb") as f:
                payload = pkl.load(f)
            if isinstance(payload, dict):
                for key in ("xyxy", "mask", "class_id", "confidence"):
                    if key in payload:
                        total += len(payload[key])
                        break
        except Exception:
            continue
    return total


def _find_detection_dir(scene_dir: Path) -> Path:
    candidates = sorted(scene_dir.glob("exp_*/detections"))
    for path in candidates:
        if list(path.glob("*.pkl.gz")):
            return path
    return scene_dir / "detections"


def export_visualization_pipeline(scene_dir: Path, exp_dir: Path, navigation_dir: Path) -> Path:
    out = exp_dir / "visualization_pipeline"
    if out.exists():
        shutil.rmtree(out)
    folders = [
        "01_input_rgb",
        "02_projected_depth",
        "03_detections",
        "04_masks",
        "05_object_pointcloud_or_bev",
        "06_scenegraph",
        "07_navigation_candidates",
        "08_reachable_goal_search",
        "09_astar_path",
    ]
    for folder in folders:
        (out / folder).mkdir(parents=True, exist_ok=True)

    _copy_some(scene_dir / "color", out / "01_input_rgb", patterns=("*.jpg", "*.png"), limit=12)
    _copy_some(scene_dir / "depth", out / "02_projected_depth", patterns=("*.png",), limit=12)
    detection_vis_dirs = list(scene_dir.glob("exp_*/vis")) + list(scene_dir.glob("exp_*/detections_vis")) + list(scene_dir.glob("exp_*/visualizations"))
    for vis_dir in detection_vis_dirs:
        _copy_some(vis_dir, out / "03_detections", patterns=("*.png", "*.jpg"), limit=12)
    _copy_some(exp_dir / "cfslam_captions_qwen_debug", out / "04_masks", patterns=("*.png", "*.jpg"), limit=None)

    scenegraph_vis = exp_dir / "scenegraph_vis"
    for name in ("scenegraph_bev.png", "scenegraph_network.png", "scenegraph_3d.png", "scenegraph_3d_interactive.html"):
        _copy_file(scenegraph_vis / name, out / "06_scenegraph" / name)
    for name in ("scenegraph_bev.png", "scenegraph_3d.png"):
        _copy_file(scenegraph_vis / name, out / "05_object_pointcloud_or_bev" / name)

    for case_dir in sorted(navigation_dir.glob("*")):
        if not case_dir.is_dir():
            continue
        prefix = case_dir.name
        _copy_file(case_dir / "navigation_candidates.png", out / "07_navigation_candidates" / f"{prefix}_navigation_candidates.png")
        _copy_file(case_dir / "reachable_goal_search.png", out / "08_reachable_goal_search" / f"{prefix}_reachable_goal_search.png")
        _copy_file(case_dir / "astar_path.png", out / "09_astar_path" / f"{prefix}_astar_path.png")
        _copy_file(case_dir / "semantic_navigation_plan.png", out / "09_astar_path" / f"{prefix}_semantic_navigation_plan.png")
    return out


def _copy_some(src: Path, dst: Path, patterns: tuple[str, ...], limit: int | None) -> None:
    if not src.exists():
        return
    copied = 0
    for pattern in patterns:
        for path in sorted(src.glob(pattern)):
            if limit is not None and copied >= limit:
                return
            _copy_file(path, dst / path.name)
            copied += 1


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return
    shutil.copy2(src, dst)


def _load_poses(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 16))
    poses = np.loadtxt(path)
    return poses.reshape(1, -1) if poses.ndim == 1 else poses


def _path_length(points: np.ndarray) -> float:
    if points.ndim != 2 or len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())


def _candidate_clearance_values(candidate: dict[str, Any]) -> list[float]:
    value = candidate.get("debug", {}).get("clearance_m")
    if value is None:
        return []
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(value)]


def _ratio(items: list[dict[str, Any]], predicate) -> float:
    if not items:
        return 0.0
    return float(sum(1 for item in items if predicate(item)) / len(items))


def _suggest_fix(candidates: list[dict[str, Any]]) -> str:
    reasons = [str(c.get("reachability_reason") or c.get("debug", {}).get("reachability", "")) for c in candidates]
    if any("out_of_map" in reason for reason in reasons):
        return "use_fallback_map_or_expand_bev_bounds"
    if any("occupied" in reason for reason in reasons):
        return "enlarge_goal_radius"
    if any("astar_failed" in reason for reason in reasons):
        return "increase_clearance_or_relax_obstacles"
    return "inspect_query_or_scenegraph"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pickle(path: Path, default):
    if not path.exists():
        return default
    with open(path, "rb") as f:
        return pkl.load(f)


def _write_summary_csv(path: Path, scene_metrics: dict[str, Any], navigation_cases: list[dict[str, Any]]) -> None:
    rows = []
    for case in navigation_cases:
        row = {"scene_id": scene_metrics.get("scene_id")}
        row.update(case.get("target_retrieval", {}))
        row.update(case.get("reachability", {}))
        row.update(case.get("path_planning", {}))
        rows.append(row)
    if not rows:
        rows = [scene_metrics]
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
