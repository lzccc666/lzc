from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from conceptgraph.navigation import SemanticNavigationPipeline
from conceptgraph.navigation.evaluation import save_scene_evaluation
from conceptgraph.navigation.io import load_scene_graph_nodes
from conceptgraph.navigation.semantic import (
    CLASS_COMPATIBILITY,
    is_node_compatible_with_query,
    parse_navigation_query,
)


VEHICLE_TAGS = {"vehicle", "car", "truck", "bus", "motorcycle", "bicycle", "van"}
STATIC_TAGS = {"traffic sign", "traffic light", "road", "sidewalk", "building", "tree", "fence", "pole"}
COLOR_ZH = {
    "white": "白色",
    "black": "黑色",
    "blue": "蓝色",
    "red": "红色",
    "gray": "灰色",
    "silver": "银色",
    "green": "绿色",
    "yellow": "黄色",
    "orange": "橙色",
}
TAG_ZH = {
    "car": "小汽车",
    "truck": "卡车",
    "bus": "公交车",
    "motorcycle": "摩托车",
    "bicycle": "自行车",
    "pedestrian": "行人",
    "traffic sign": "交通标志",
    "traffic light": "交通信号灯",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run several semantic navigation examples for one nuScenes scene.")
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--qwen-exp-dir", required=True)
    parser.add_argument("--baseline-exp-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--keep-failures-debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir)
    qwen_exp_dir = Path(args.qwen_exp_dir)
    baseline_exp_dir = Path(args.baseline_exp_dir)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failure_debug_dir = out_dir.parent / "navigation_failures_debug"
    if failure_debug_dir.exists():
        shutil.rmtree(failure_debug_dir)

    poses = load_poses(scene_dir / "poses.txt")
    nodes = load_scene_graph_nodes(qwen_exp_dir / "scene_graph.json", None)
    edges = load_edges(qwen_exp_dir / "cfslam_scenegraph_edges.pkl")
    examples = build_examples(poses, nodes, edges)
    timing_rows = []
    summary = []
    failures = []
    success_index = 1

    for index, example in enumerate(examples, start=1):
        temp_case_dir = out_dir / f"_tmp_{index:02d}_{example['name']}"
        if temp_case_dir.exists():
            shutil.rmtree(temp_case_dir)
        temp_case_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        try:
            plan, selected_pose, pipeline = run_best_long_path(
                qwen_exp_dir=qwen_exp_dir,
                baseline_exp_dir=baseline_exp_dir,
                case_dir=temp_case_dir,
                query=example["query"],
                candidate_poses=example["candidate_poses"],
                min_waypoints=example.get("min_waypoints", 15),
                min_path_m=example.get("min_path_m", 4.0),
                max_path_m=example.get("max_path_m", 35.0),
            )
            validation = validate_plan_target(plan, nodes)
            if not validation["valid"]:
                raise RuntimeError(f"semantic_target_mismatch: {validation}")
            plan.debug["target_validation"] = validation
            case_dir = out_dir / f"{success_index:02d}_{example['name']}"
            if case_dir.exists():
                shutil.rmtree(case_dir)
            temp_case_dir.rename(case_dir)
            plan_path = pipeline.save_plan(plan, case_dir)
            elapsed = time.perf_counter() - start
            timing_rows.append(
                {
                    "name": example["name"],
                    "query": example["query"],
                    "display_query": plan.query.display_text or plan.query.raw_text,
                    "seconds": elapsed,
                    "selected_node_id": int(plan.selected_node_id),
                    "selected_object_tag": plan.selected_object_tag,
                    "target_validation": validation,
                    "path_waypoints": int(len(plan.path_world)),
                    "path_length_m": path_length_m(plan.path_world),
                    "path_cost": float(plan.path_cost),
                    "fallback_used": bool(plan.fallback_used),
                    "fallback_reason": plan.fallback_reason,
                    "fallback_distance_m": float(plan.fallback_distance_m),
                    "top1_semantic_node_id": plan.top1_semantic_node_id,
                    "top1_reachable_same_as_selected": plan.top1_semantic_node_id == plan.selected_node_id,
                    "vehicle_pose": selected_pose,
                }
            )
            summary.append({**timing_rows[-1], "plan_path": str(plan_path)})
            success_index += 1
            if args.visualize:
                run_scenegraph_overlay(
                    qwen_exp_dir=qwen_exp_dir,
                    baseline_exp_dir=baseline_exp_dir,
                    scene_dir=scene_dir,
                    case_dir=case_dir,
                    plan_path=plan_path,
                    vehicle_pose=selected_pose,
                )
        except Exception as exc:
            failure = {
                "name": example["name"],
                "query": example["query"],
                "failed_stage": "navigation",
                "failure_reason": str(exc),
                "case_dir": str(failure_debug_dir / f"{index:02d}_{example['name']}"),
                "suggested_fix": "inspect_query_or_reachability",
            }
            failures.append(failure)
            if args.keep_failures_debug:
                debug_case_dir = failure_debug_dir / f"{index:02d}_{example['name']}"
                debug_case_dir.mkdir(parents=True, exist_ok=True)
                (debug_case_dir / "failure.json").write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8")
            if temp_case_dir.exists():
                shutil.rmtree(temp_case_dir)

    (out_dir / "navigation_examples_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "navigation_timing.json").write_text(
        json.dumps(timing_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.keep_failures_debug:
        failure_debug_dir.mkdir(parents=True, exist_ok=True)
        (failure_debug_dir / "failures.json").write_text(
            json.dumps(failures, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    save_scene_evaluation(scene_dir=scene_dir, exp_dir=qwen_exp_dir, navigation_dir=out_dir)


def load_poses(path: Path) -> np.ndarray:
    poses = np.loadtxt(path)
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    centers = []
    for row in poses:
        if row.size < 16:
            continue
        mat = row.reshape(4, 4)
        centers.append(mat[:3, 3])
    if not centers:
        raise RuntimeError(f"No valid poses found in {path}")
    return np.asarray(centers, dtype=float)


def build_examples(poses: np.ndarray, nodes: dict, edges: list[tuple[int, int, str]] | None = None) -> list[dict]:
    indices = np.linspace(0, len(poses) - 1, num=min(10, len(poses)), dtype=int)
    starts = [pose_with_heading(poses, int(idx)) for idx in indices]
    tags = {str(node.object_tag).lower() for node in nodes.values()}
    has_vehicle = bool(tags & VEHICLE_TAGS)
    has_static = bool(tags & STATIC_TAGS)
    queries = []
    if has_vehicle:
        queries.append(("最近车辆", "前往最近的车辆附近", 4.0, 28.0, 8))
        queries.append(("前方车辆", "前往前方的车辆附近", 4.0, 32.0, 8))
    if "car" in tags:
        queries.append(("小汽车", "前往小汽车附近", 5.0, 35.0, 10))
    if "truck" in tags:
        queries.append(("卡车", "前往卡车附近", 5.0, 35.0, 8))
    if "bus" in tags:
        queries.append(("公交车", "前往公交车附近", 5.0, 35.0, 8))
    if "motorcycle" in tags:
        queries.append(("摩托车", "前往摩托车附近", 4.0, 32.0, 7))
    if "traffic sign" in tags:
        queries.append(("交通标志", "前往交通标志附近", 4.0, 35.0, 8))
    if "pedestrian" in tags:
        queries.append(("行人", "前往行人附近", 3.0, 30.0, 6))
    if has_static:
        queries.append(("最近静态目标", "前往最近的静态目标附近", 3.0, 30.0, 6))
    queries.extend(build_color_examples(nodes))
    queries.extend(build_relation_examples(nodes, edges or []))
    examples = []
    for name, query, min_path_m, max_path_m, min_waypoints in queries:
        examples.append(
            {
                "name": name,
                "query": query,
                "candidate_poses": starts,
                "min_path_m": min_path_m,
                "max_path_m": max_path_m,
                "min_waypoints": min_waypoints,
            }
        )
    return examples


def build_color_examples(nodes: dict) -> list[tuple[str, str, float, float, int]]:
    examples = []
    seen = set()

    def colors_for_node(node) -> list[str]:
        metadata = node.metadata or {}
        attrs = metadata.get("attributes", {}) if isinstance(metadata, dict) else {}
        return [str(c).lower() for c in attrs.get("colors", []) if str(c).lower() in COLOR_ZH]

    for node in nodes.values():
        tag = str(node.object_tag).lower()
        if tag not in {"car", "truck", "bus", "motorcycle"}:
            continue
        colors = colors_for_node(node)
        if not colors:
            continue
        label = f"{COLOR_ZH[colors[0]]}{TAG_ZH.get(tag, tag)}"
        query = f"前往{label}附近"
        if query in seen:
            continue
        seen.add(query)
        examples.append((label, query, 4.0, 35.0, 8))
        if len(examples) >= 4:
            break
    return examples


def build_relation_examples(nodes: dict, edges: list[tuple[int, int, str]]) -> list[tuple[str, str, float, float, int]]:
    relation_queries = []
    edge_pairs = [(int(src), int(dst)) for src, dst, rel in edges if str(rel).lower() == "near"]

    def node_tag(node_id: int) -> str:
        return str(nodes[node_id].object_tag).lower()

    def colors_for(node_id: int) -> list[str]:
        metadata = nodes[node_id].metadata or {}
        attrs = metadata.get("attributes", {}) if isinstance(metadata, dict) else {}
        return [str(c).lower() for c in attrs.get("colors", []) if str(c).lower() in COLOR_ZH]

    def label_for(node_id: int, include_color: bool = True) -> str:
        tag = node_tag(node_id)
        color = colors_for(node_id)[0] if include_color and colors_for(node_id) else ""
        return f"{COLOR_ZH.get(color, '')}{TAG_ZH.get(tag, tag)}"

    def add_query(name: str, query: str, min_m: float = 4.0, max_m: float = 35.0, min_wp: int = 8) -> None:
        if query not in [item[1] for item in relation_queries]:
            relation_queries.append((name, query, min_m, max_m, min_wp))

    for src, dst in edge_pairs:
        tag_src, tag_dst = node_tag(src), node_tag(dst)
        src_colors, dst_colors = colors_for(src), colors_for(dst)
        if tag_src == "pedestrian" and tag_dst in VEHICLE_TAGS and dst_colors:
            anchor = label_for(dst)
            add_query(f"靠近{anchor}的行人", f"前往{anchor}旁边的行人附近", 3.0, 32.0, 6)
        if tag_dst == "pedestrian" and tag_src in VEHICLE_TAGS and src_colors:
            anchor = label_for(src)
            add_query(f"靠近{anchor}的行人", f"前往{anchor}旁边的行人附近", 3.0, 32.0, 6)
        if tag_src == "car" and tag_dst == "car" and src_colors:
            anchor = label_for(src)
            add_query(f"停在{anchor}旁边的车辆", f"前往停在{anchor}旁边的车辆附近", 4.0, 35.0, 8)
        if tag_dst == "car" and tag_src == "car" and dst_colors:
            anchor = label_for(dst)
            add_query(f"停在{anchor}旁边的车辆", f"前往停在{anchor}旁边的车辆附近", 4.0, 35.0, 8)
        if tag_src == "traffic sign" and tag_dst in VEHICLE_TAGS:
            add_query("靠近交通标志的车辆", "前往靠近交通标志的车辆附近", 4.0, 35.0, 8)
            add_query("交通标志旁边的车辆", "前往交通标志旁边的车辆附近", 4.0, 35.0, 8)
        if tag_dst == "traffic sign" and tag_src in VEHICLE_TAGS:
            add_query("靠近交通标志的车辆", "前往靠近交通标志的车辆附近", 4.0, 35.0, 8)
            add_query("交通标志旁边的车辆", "前往交通标志旁边的车辆附近", 4.0, 35.0, 8)
        if tag_src == "motorcycle" and tag_dst == "car" and dst_colors:
            anchor = label_for(dst)
            add_query(f"{anchor}旁边的摩托车", f"前往{anchor}旁边的摩托车附近", 3.0, 32.0, 7)
        if tag_dst == "motorcycle" and tag_src == "car" and src_colors:
            anchor = label_for(src)
            add_query(f"{anchor}旁边的摩托车", f"前往{anchor}旁边的摩托车附近", 3.0, 32.0, 7)
        if tag_src in VEHICLE_TAGS and tag_dst in VEHICLE_TAGS and colors_for(src):
            anchor = label_for(src)
            add_query(f"停在{anchor}旁边的车辆", f"前往停在{anchor}旁边的车辆附近", 4.0, 35.0, 8)
        if len(relation_queries) >= 8:
            break
    return relation_queries[:8]


def load_edges(path: Path) -> list[tuple[int, int, str]]:
    if not path.exists():
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def pose_with_heading(poses: np.ndarray, index: int) -> list[float]:
    center = np.asarray(poses[index], dtype=float)
    if len(poses) <= 1:
        heading = np.asarray([1.0, 0.0], dtype=float)
    elif index < len(poses) - 1:
        heading = poses[index + 1, :2] - poses[index, :2]
    else:
        heading = poses[index, :2] - poses[index - 1, :2]
    norm = float(np.linalg.norm(heading))
    if norm < 1e-6:
        heading = np.asarray([1.0, 0.0], dtype=float)
    else:
        heading = heading / norm
    return center[:3].tolist() + heading[:2].tolist()


def run_best_long_path(
    qwen_exp_dir: Path,
    baseline_exp_dir: Path,
    case_dir: Path,
    query: str,
    candidate_poses: list[list[float]],
    min_waypoints: int,
    min_path_m: float,
    max_path_m: float,
):
    best = None
    failures = []
    parsed_query = parse_navigation_query(query, default_near_distance=10.0)
    shared_cfg = make_cfg(
        qwen_exp_dir=qwen_exp_dir,
        baseline_exp_dir=baseline_exp_dir,
        output_dir=case_dir,
        query_text=query,
        vehicle_pose=candidate_poses[0],
    )
    pipeline = SemanticNavigationPipeline(shared_cfg)
    for pose in candidate_poses:
        pipeline.cfg.vehicle_pose = pose
        try:
            plan = pipeline.run(query, np.asarray(pose, dtype=float))
        except Exception as exc:
            failures.append(str(exc))
            continue
        length = int(len(plan.path_world))
        path_m = path_length_m(plan.path_world)
        if not (min_path_m <= path_m <= max_path_m):
            failures.append(f"path_length_out_of_range={path_m:.2f}m")
            continue
        if best is None:
            best = (plan, pose, pipeline)
        elif plan_quality_key(plan, parsed_query) > plan_quality_key(best[0], parsed_query):
            best = (plan, pose, pipeline)
        if length < min_waypoints:
            failures.append(f"too_few_waypoints={length}")
    if best is None:
        raise RuntimeError(f"No navigation plan succeeded for query {query}: {failures[:3]}")
    return best[0], best[1], best[2]


def plan_quality_key(plan, parsed_query) -> tuple[float, float, float]:
    selected = next((cand for cand in plan.candidate_targets if cand.node_id == plan.selected_node_id), None)
    distance = float("inf")
    if selected is not None:
        distance = float(selected.debug.get("distance_to_vehicle", float("inf")))
    path_m = path_length_m(plan.path_world)
    if parsed_query.prefer_nearest or parsed_query.prefer_front:
        return (-distance, -abs(path_m - 12.0), float(plan.candidate_targets[0].score if plan.candidate_targets else 0.0))
    return (-abs(path_m - 18.0), -distance, float(plan.candidate_targets[0].score if plan.candidate_targets else 0.0))


def path_length_m(path_world: np.ndarray) -> float:
    path_world = np.asarray(path_world, dtype=float)
    if path_world.ndim != 2 or len(path_world) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path_world[:, :2], axis=0), axis=1).sum())


def validate_plan_target(plan, nodes: dict) -> dict:
    parsed = plan.query
    node = nodes.get(int(plan.selected_node_id))
    if node is None:
        return {"valid": False, "reason": "missing_selected_node", "selected_node_id": int(plan.selected_node_id)}
    valid, reason = is_node_compatible_with_query(parsed, node)
    target_categories = list(parsed.target_categories)
    allowed = sorted({item for category in target_categories for item in CLASS_COMPATIBILITY.get(category, {category})})
    return {
        "valid": bool(valid),
        "reason": reason,
        "selected_node_id": int(plan.selected_node_id),
        "selected_object_tag": node.object_tag,
        "target_categories": target_categories,
        "allowed_tags": allowed,
    }


def make_cfg(
    qwen_exp_dir: Path,
    baseline_exp_dir: Path,
    output_dir: Path,
    query_text: str,
    vehicle_pose: list[float],
):
    return OmegaConf.create(
        {
            "scene_map_path": str(qwen_exp_dir / "map" / "scene_map_cfslam_pruned.pkl.gz"),
            "scene_graph_json_path": str(qwen_exp_dir / "scene_graph.json"),
            "scene_graph_edge_path": str(qwen_exp_dir / "cfslam_scenegraph_edges.pkl"),
            "annotated_scenegraph_json_path": None,
            "bev_map_path": str(baseline_exp_dir / "navigation" / "bev_map.npz"),
            "output_dir": str(output_dir),
            "query_text": query_text,
            "vehicle_pose": vehicle_pose,
            "grid_resolution": 0.5,
            "grid_padding_m": 20.0,
            "obstacle_profile": "thesis",
            "robot_radius_m": 0.8,
            "object_bbox_obstacle_scale": 0.18,
            "max_object_obstacle_radius_m": 1.2,
            "ignore_dynamic_obstacles": False,
            "default_near_distance_m": 10.0,
            "start_search_radius_m": 16.0,
            "goal_search_radius_m": 14.0,
            "allow_goal_fallback": True,
            "semantic_cost_weight": 1.0,
            "clearance_cost_weight": 0.2,
            "path_smoothing_iterations": 2,
            "lambda_text": 0.65,
            "lambda_relation": 0.2,
            "lambda_distance_prior": 0.15,
            "max_candidates": 15,
        }
    )


def run_scenegraph_overlay(
    qwen_exp_dir: Path,
    baseline_exp_dir: Path,
    scene_dir: Path,
    case_dir: Path,
    plan_path: Path,
    vehicle_pose: list[float],
) -> None:
    cmd = [
        "python",
        "conceptgraph/scripts/visualize_scenegraph.py",
        "--scene-graph",
        str(qwen_exp_dir / "scene_graph.json"),
        "--edges",
        str(qwen_exp_dir / "cfslam_scenegraph_edges.pkl"),
        "--out-dir",
        str(case_dir / "scenegraph_overlay"),
        "--vehicle-pose",
        ",".join(str(v) for v in vehicle_pose[:3]),
        "--navigation-plan",
        str(plan_path),
        "--trajectory-poses",
        str(scene_dir / "poses.txt"),
        "--bev-map",
        str(baseline_exp_dir / "navigation" / "bev_map.npz"),
    ]
    subprocess.run(cmd, check=True)




if __name__ == "__main__":
    main()
