from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from conceptgraph.navigation import SemanticNavigationPipeline
from conceptgraph.scenegraph.incremental_graph import IncrementalSceneGraph
from conceptgraph.scenegraph.relations import (
    SceneGraphEdge,
    build_relations,
    edge_relation_for_query,
    save_legacy_edge_pickle,
    save_relation_result,
    save_structured_edges,
)
from conceptgraph.scripts.run_nuscenes_navigation_examples import make_cfg
from conceptgraph.scripts.run_stage2_offline_navigation import (
    _reference_navigation_case,
    _resolve_scene_inputs,
)


DEFAULT_PROJECT_ROOT = Path("/root/autodl-tmp/lizichong")
DYNAMIC_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行阶段3真实离线关系图实验。")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--scene-count", type=int, default=2)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--skip-visualizations", action="store_true")
    parser.add_argument("--skip-navigation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else project_root / "outputs/project_v2/scenegraph_relations"
    )
    full_root = output_root / "stage3_full_rebuild"
    incremental_root = output_root / "stage3_incremental"
    visualization_root = output_root / "stage3_visualizations"
    for path in (full_root, incremental_root, visualization_root):
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = project_root / "outputs/project_v2/subset/selected_scenes.json"
    requested = args.scenes or _selected_scene_names(manifest_path)
    relation_config = OmegaConf.to_container(
        OmegaConf.load(project_root / "conceptgraph/hydra_configs/scenegraph_relations.yaml"),
        resolve=True,
    )
    relation_config = {
        key: value
        for key, value in relation_config.items()
        if key not in {"defaults", "visualization"}
    }
    navigation_target_config = OmegaConf.load(
        project_root / "conceptgraph/hydra_configs/semantic_navigation.yaml"
    ).target_selection

    comparisons: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    required_count = max(1, int(args.scene_count))
    for scene_name in requested:
        inputs, diagnostic = _resolve_stage3_inputs(project_root, scene_name)
        if inputs is None:
            skipped.append({"scene_name": scene_name, "diagnostic": diagnostic})
            continue
        try:
            row = run_scene(
                project_root=project_root,
                scene_name=scene_name,
                inputs=inputs,
                relation_config=relation_config,
                navigation_target_config=navigation_target_config,
                full_dir=full_root / scene_name,
                incremental_dir=incremental_root / scene_name,
                visualization_dir=visualization_root / scene_name,
                run_visualizations=not args.skip_visualizations,
                run_navigation=not args.skip_navigation,
            )
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as exc:
            skipped.append(
                {
                    "scene_name": scene_name,
                    "diagnostic": {
                        "success": False,
                        "failure_code": "SCENE_RUN_FAILED",
                        "failure_reason": str(exc),
                    },
                }
            )
            continue
        comparisons.append(row)
        if len(comparisons) >= required_count:
            break

    payload = {
        "schema_version": 1,
        "success": len(comparisons) >= required_count,
        "mode": "stage3_offline_relations",
        "selected_scenes_manifest": str(manifest_path.resolve()),
        "requested_scene_count": required_count,
        "completed_scene_count": len(comparisons),
        "scenes": comparisons,
        "skipped_scenes": skipped,
        "failure_code": None if len(comparisons) >= required_count else "INSUFFICIENT_COMPLETE_SCENES",
        "failure_reason": (
            None
            if len(comparisons) >= required_count
            else f"仅完成 {len(comparisons)} 个场景，要求 {required_count} 个。"
        ),
        "config": relation_config,
    }
    payload["deterministic_content_sha256"] = deterministic_payload_hash(payload)
    comparison_path = output_root / "stage3_comparison.json"
    comparison_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(comparison_path.resolve())
    if not payload["success"]:
        raise RuntimeError(payload["failure_reason"])


def run_scene(
    *,
    project_root: Path,
    scene_name: str,
    inputs: dict[str, Path],
    relation_config: dict[str, Any],
    navigation_target_config: Any,
    full_dir: Path,
    incremental_dir: Path,
    visualization_dir: Path,
    run_visualizations: bool,
    run_navigation: bool,
) -> dict[str, Any]:
    for path in (full_dir, incremental_dir, visualization_dir):
        path.mkdir(parents=True, exist_ok=True)
    scene_nodes = json.loads(inputs["scene_graph"].read_text(encoding="utf-8"))
    trajectory = load_trajectory_xy(inputs["poses"])
    ego_position = trajectory[-1] if len(trajectory) else None
    ego_heading = trajectory_heading(trajectory)

    full_started = time.perf_counter()
    full_result = build_relations(
        scene_nodes,
        relation_config,
        frame_index=max(0, len(trajectory) - 1),
        ego_heading=ego_heading,
        trajectory=trajectory,
        ego_position=ego_position,
        mode="full_rebuild",
    )
    full_elapsed = float(time.perf_counter() - full_started)
    if not full_result.success:
        raise RuntimeError(f"{full_result.failure_code}: {full_result.failure_reason}")
    full_result.timing["offline_runner_s"] = full_elapsed
    full_result_path = save_relation_result(full_result, full_dir / "relation_build_result.json")
    structured_json = save_structured_edges(full_result.edges, full_dir / "structured_edges.json")
    structured_pickle = save_structured_edges(full_result.edges, full_dir / "structured_edges.pkl")
    legacy_output = save_legacy_edge_pickle(full_result.edges, full_dir / "cfslam_scenegraph_edges.pkl")
    (full_dir / "resolved_config.json").write_text(
        json.dumps(relation_config, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    legacy_stats = legacy_edge_statistics(inputs["legacy_edges"], len(scene_nodes))
    full_stats = structured_edge_statistics(full_result.edges, len(scene_nodes))
    incremental_summary = replay_incremental(
        snapshot_dir=inputs["snapshot_dir"],
        poses=trajectory,
        relation_config=relation_config,
        output_dir=incremental_dir,
        comparison_plot_path=visualization_dir / "incremental_full_comparison.png",
    )

    reference, reference_diagnostic = _reference_navigation_case(inputs["qwen_exp_dir"])
    if reference is None:
        reference = {
            "instruction": "car near traffic sign",
            "vehicle_pose": _fallback_vehicle_pose(trajectory),
            "source_plan": None,
        }
    navigation_queries: list[dict[str, Any]] = []
    highlight_plan: Path | None = None
    if run_navigation:
        query_specs = relation_navigation_queries(full_result.edges, scene_nodes, count=2)
        navigation_queries, highlight_plan = run_navigation_queries(
            inputs=inputs,
            query_specs=query_specs,
            reference=reference,
            target_config=navigation_target_config,
            structured_edge_path=structured_json,
            output_dir=full_dir / "navigation_queries",
            trajectory_path=inputs["poses"],
            minimum_confidence=float(relation_config["minimum_confidence"]),
        )

    visualization_paths: dict[str, str] = {}
    if run_visualizations:
        visualization_paths = render_visualizations(
            project_root=project_root,
            inputs=inputs,
            structured_edge_path=structured_json,
            output_dir=visualization_dir,
            reference=reference,
            navigation_plan=highlight_plan,
            minimum_confidence=float(relation_config["minimum_confidence"]),
        )
        visualization_paths["incremental_full_comparison"] = str(
            (visualization_dir / "incremental_full_comparison.png").resolve()
        )

    fallback_count = sum(
        int(query.get("structured", {}).get("geometry_fallback_count", 0))
        for query in navigation_queries
    )
    row = {
        "scene_name": scene_name,
        "frame_count": incremental_summary["frame_count"],
        "node_count": len(scene_nodes),
        "legacy": legacy_stats,
        "structured_full_rebuild": {
            **full_stats,
            "conflict_statistics": full_result.conflict_statistics,
            "pruning_statistics": full_result.pruning_statistics,
            "relation_build_time_s": full_elapsed,
            "result_path": str(full_result_path.resolve()),
            "structured_json_path": str(structured_json.resolve()),
            "structured_pickle_path": str(structured_pickle.resolve()),
            "compatible_legacy_pickle_path": str(legacy_output.resolve()),
        },
        "incremental": incremental_summary,
        "navigation_queries": navigation_queries,
        "navigation_selection_changed": any(
            query.get("selection_changed", False) for query in navigation_queries
        ),
        "navigation_relation_chains": [
            query.get("structured", {}).get("relation_chain", []) for query in navigation_queries
        ],
        "geometry_fallback_count": fallback_count,
        "visualization_files": visualization_paths,
        "reference_navigation_diagnostic": reference_diagnostic,
    }
    row["deterministic_content_sha256"] = deterministic_payload_hash(row)
    return row


def replay_incremental(
    *,
    snapshot_dir: Path,
    poses: np.ndarray,
    relation_config: dict[str, Any],
    output_dir: Path,
    comparison_plot_path: Path,
) -> dict[str, Any]:
    incremental_config = json.loads(json.dumps(relation_config))
    # A changing ego/local basis is a global dependency. The incremental replay
    # therefore uses the fixed world basis so unchanged node neighborhoods remain reusable.
    incremental_config["coordinate_frame"] = "world"
    incremental_config["coordinate_frame_fallback"] = "world"
    frame_paths = sorted(
        path for path in snapshot_dir.glob("*.pkl.gz") if path.stem.split(".")[0].isdigit()
    )
    meta = load_gzip_pickle(snapshot_dir / "meta.pkl.gz") if (snapshot_dir / "meta.pkl.gz").exists() else {}
    class_names = [str(item) for item in meta.get("class_names", [])]
    graph = IncrementalSceneGraph(incremental_config)
    object_registry: dict[tuple[Any, ...], int] = {}
    previous_centers: dict[int, np.ndarray] = {}
    updates = []
    comparisons = []
    total_recomputed = 0
    total_reused = 0
    total_expired = 0

    for sequence_index, frame_path in enumerate(frame_paths):
        payload = load_gzip_pickle(frame_path)
        frame_index = int(frame_path.name.split(".")[0])
        snapshot_nodes, current_centers = snapshot_relation_nodes(
            payload.get("objects", []),
            class_names,
            object_registry,
            previous_centers,
            frame_index,
        )
        pose_index = min(max(frame_index, 0), max(0, len(poses) - 1))
        local_trajectory = poses[: pose_index + 1] if len(poses) else np.empty((0, 2))
        heading = trajectory_heading(local_trajectory)
        ego_position = local_trajectory[-1] if len(local_trajectory) else None
        result = graph.update_snapshot(
            snapshot_nodes,
            frame_index=frame_index,
            ego_heading=heading,
            trajectory=local_trajectory,
            ego_position=ego_position,
        )
        full = build_relations(
            snapshot_nodes,
            incremental_config,
            frame_index=frame_index,
            ego_heading=heading,
            trajectory=local_trajectory,
            ego_position=ego_position,
            mode="same_frame_full_rebuild",
        )
        comparison = compare_edge_sets(result.edges, full.edges)
        comparison["frame_index"] = frame_index
        comparison["incremental_edge_count"] = len(result.edges)
        comparison["full_edge_count"] = len(full.edges)
        comparisons.append(comparison)
        update_payload = {
            "frame_index": frame_index,
            "source_snapshot": str(frame_path.resolve()),
            "node_count": len(snapshot_nodes),
            "edge_count": result.edge_count,
            "success": result.success,
            "update_statistics": result.update_statistics,
            "timing": result.timing,
            "comparison": comparison,
        }
        updates.append(update_payload)
        total_recomputed += int(result.update_statistics.get("recomputed_edge_count", 0))
        total_reused += int(result.update_statistics.get("reused_edge_count", 0))
        total_expired += int(result.update_statistics.get("expired_dynamic_edge_count", 0))
        previous_centers = current_centers

    final_result = graph.export_snapshot()
    save_relation_result(final_result, output_dir / "relation_snapshot.json")
    save_structured_edges(final_result.edges, output_dir / "structured_edges.json")
    (output_dir / "frame_updates.json").write_text(
        json.dumps(updates, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "frame_comparisons.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    plot_incremental_comparison(comparisons, comparison_plot_path)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(incremental_config, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    agreement_values = [float(item["active_edge_agreement"]) for item in comparisons]
    summary = {
        "success": True,
        "mode": "snapshot_replay",
        "snapshot_directory": str(snapshot_dir.resolve()),
        "frame_count": len(frame_paths),
        "first_frame": None if not frame_paths else int(frame_paths[0].name.split(".")[0]),
        "last_frame": None if not frame_paths else int(frame_paths[-1].name.split(".")[0]),
        "final_node_count": final_result.node_count,
        "final_edge_count": final_result.edge_count,
        "total_recomputed_edge_count": total_recomputed,
        "total_reused_edge_count": total_reused,
        "expired_dynamic_edge_count": total_expired,
        "mean_active_edge_agreement": float(np.mean(agreement_values)) if agreement_values else 1.0,
        "minimum_active_edge_agreement": min(agreement_values, default=1.0),
        "final_active_edge_agreement": agreement_values[-1] if agreement_values else 1.0,
        "comparison_policy": "TTL stale dynamic edges are excluded from active-edge agreement",
        "coordinate_frame": "world",
        "coordinate_frame_reason": "固定世界坐标避免车辆航向变化导致全图方向事实失效",
        "update_log_path": str((output_dir / "frame_updates.json").resolve()),
        "comparison_path": str((output_dir / "frame_comparisons.json").resolve()),
        "final_snapshot_path": str((output_dir / "relation_snapshot.json").resolve()),
    }
    summary["deterministic_content_sha256"] = deterministic_payload_hash(summary)
    (output_dir / "incremental_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def run_navigation_queries(
    *,
    inputs: dict[str, Path],
    query_specs: list[dict[str, Any]],
    reference: dict[str, Any],
    target_config: Any,
    structured_edge_path: Path,
    output_dir: Path,
    trajectory_path: Path,
    minimum_confidence: float,
) -> tuple[list[dict[str, Any]], Path | None]:
    summaries = []
    highlight_plan = None
    for index, spec in enumerate(query_specs, start=1):
        query_dir = output_dir / f"query_{index:02d}"
        legacy = run_navigation_query(
            inputs=inputs,
            edge_path=inputs["legacy_edges"],
            query_text=spec["query"],
            reference=reference,
            target_config=target_config,
            output_dir=query_dir / "legacy",
            trajectory_path=trajectory_path,
            minimum_confidence=minimum_confidence,
        )
        structured = run_navigation_query(
            inputs=inputs,
            edge_path=structured_edge_path,
            query_text=spec["query"],
            reference=reference,
            target_config=target_config,
            output_dir=query_dir / "structured",
            trajectory_path=trajectory_path,
            minimum_confidence=minimum_confidence,
        )
        if highlight_plan is None and structured.get("plan_path"):
            highlight_plan = Path(structured["plan_path"])
        summaries.append(
            {
                **spec,
                "legacy": legacy,
                "structured": structured,
                "selection_changed": legacy.get("selected_node_id") != structured.get("selected_node_id"),
            }
        )
    return summaries, highlight_plan


def run_navigation_query(
    *,
    inputs: dict[str, Path],
    edge_path: Path,
    query_text: str,
    reference: dict[str, Any],
    target_config: Any,
    output_dir: Path,
    trajectory_path: Path,
    minimum_confidence: float,
) -> dict[str, Any]:
    cfg = make_cfg(
        qwen_exp_dir=inputs["qwen_exp_dir"],
        baseline_exp_dir=inputs["baseline_exp_dir"],
        output_dir=output_dir,
        query_text=query_text,
        vehicle_pose=reference["vehicle_pose"],
    )
    cfg.scene_graph_edge_path = str(edge_path)
    cfg.target_selection = OmegaConf.create(OmegaConf.to_container(target_config, resolve=True))
    cfg.target_selection.mode = "joint_utility"
    cfg.relation_edge_minimum_confidence = minimum_confidence
    cfg.relation_coordinate_frame = "local_trajectory"
    cfg.relation_coordinate_frame_fallback = "world"
    cfg.relation_trajectory_path = str(trajectory_path)
    pipeline = SemanticNavigationPipeline(cfg)
    result = pipeline.select(query_text, np.asarray(reference["vehicle_pose"], dtype=float))
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = pipeline.save_selection_result(
        result,
        output_dir,
        save_plan=True,
        save_visualizations=False,
    )
    selected = result.selected_candidate
    relation_query = {}
    plan_path = None
    if result.plan is not None:
        relation_query = result.plan.debug.get("relation_query", {})
        candidate_plan = output_dir / "semantic_navigation_plan.json"
        plan_path = str(candidate_plan.resolve()) if candidate_plan.exists() else None
    return {
        "success": bool(result.success),
        "failure_code": result.failure_code,
        "failure_reason": result.failure_reason,
        "selected_node_id": None if selected is None else int(selected.node_id),
        "semantic_rank": None if selected is None else selected.semantic_rank,
        "relation_score": None if selected is None else float(selected.relation_score),
        "relation_chain": relation_query.get("relation_chain", []),
        "edge_source": relation_query.get("edge_source"),
        "fallback_reason": relation_query.get("fallback_reason", []),
        "geometry_fallback_count": int(relation_query.get("geometry_fallback_count", 0)),
        "result_path": str(result_path.resolve()),
        "plan_path": plan_path,
    }


def relation_navigation_queries(
    edges: Iterable[SceneGraphEdge],
    scene_nodes: list[dict[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    nodes_by_id = {int(node["id"]): node for node in scene_nodes}
    tags = {
        node_id: navigation_node_phrase(node)
        for node_id, node in nodes_by_id.items()
    }
    candidates = []
    for edge in edges:
        source_tag = tags.get(edge.source, "object")
        target_tag = tags.get(edge.target, "object")
        diversity = int(source_tag != target_tag)
        candidates.append((diversity, edge.confidence, edge, source_tag, target_tag, False))
        reverse = edge_relation_for_query(edge, edge.target, edge.source)
        if reverse is not None:
            candidates.append((diversity, edge.confidence, edge, target_tag, source_tag, True))
    candidates.sort(
        key=lambda item: (
            -int(item[2].directed),
            -item[0],
            -round(float(item[1]), 12),
            item[2].relation,
            item[2].source,
            item[2].target,
            item[5],
        )
    )
    queries = []
    seen = set()
    for _diversity, _confidence, edge, source_tag, target_tag, reverse_view in candidates:
        source_id, target_id = (
            (edge.target, edge.source) if reverse_view else (edge.source, edge.target)
        )
        relation = edge_relation_for_query(edge, source_id, target_id)
        query = f"{source_tag} {relation.replace('_', ' ')} {target_tag}"
        if query in seen:
            continue
        seen.add(query)
        queries.append(
            {
                "query": query,
                "expected_relation": relation,
                "expected_source_node_id": source_id,
                "expected_target_node_id": target_id,
                "stored_fact": edge.to_dict(),
            }
        )
        if len(queries) >= count:
            break
    while len(queries) < count:
        queries.append(
            {
                "query": "car near traffic sign",
                "expected_relation": "near",
                "expected_source_node_id": None,
                "expected_target_node_id": None,
                "stored_fact": None,
            }
        )
    return queries


def navigation_node_phrase(node: dict[str, Any]) -> str:
    tag = str(node.get("object_tag", "object"))
    attributes = node.get("attributes", {})
    colors = attributes.get("colors", []) if isinstance(attributes, dict) else []
    color = str(colors[0]).strip().lower() if colors else ""
    return f"{color} {tag}".strip()


def render_visualizations(
    *,
    project_root: Path,
    inputs: dict[str, Path],
    structured_edge_path: Path,
    output_dir: Path,
    reference: dict[str, Any],
    navigation_plan: Path | None,
    minimum_confidence: float,
) -> dict[str, str]:
    vehicle_pose = ",".join(str(value) for value in reference["vehicle_pose"][:3])
    base = [
        "--scene-graph",
        str(inputs["scene_graph"]),
        "--edges",
        str(structured_edge_path),
        "--out-dir",
        str(output_dir),
        "--vehicle-pose",
        vehicle_pose,
        "--minimum-confidence",
        str(minimum_confidence),
    ]
    command_2d = [
        sys.executable,
        "-m",
        "conceptgraph.scripts.visualize_scenegraph",
        *base,
        "--trajectory-poses",
        str(inputs["poses"]),
        "--bev-map",
        str(inputs["bev_map"]),
    ]
    if navigation_plan is not None:
        command_2d.extend(["--navigation-plan", str(navigation_plan)])
    subprocess.run(command_2d, cwd=project_root, check=True)
    subprocess.run(
        [sys.executable, "-m", "conceptgraph.scripts.visualize_scenegraph_3d", *base],
        cwd=project_root,
        check=True,
    )
    interactive_command = [
        sys.executable,
        "-m",
        "conceptgraph.scripts.visualize_scenegraph_3d_interactive",
        *base,
    ]
    if navigation_plan is not None:
        interactive_command.extend(["--navigation-plan", str(navigation_plan)])
    subprocess.run(interactive_command, cwd=project_root, check=True)
    paths = {
        "bev_static_png": output_dir / "scenegraph_bev.png",
        "network_static_png": output_dir / "scenegraph_network.png",
        "static_3d_png": output_dir / "scenegraph_3d.png",
        "interactive_html": output_dir / "scenegraph_3d_interactive.html",
        "navigation_relation_chain_png": output_dir / "navigation_relation_chain.png",
    }
    return {
        key: str(path.resolve())
        for key, path in paths.items()
        if path.exists()
    }


def snapshot_relation_nodes(
    objects: list[dict[str, Any]],
    class_names: list[str],
    registry: dict[tuple[Any, ...], int],
    previous_centers: dict[int, np.ndarray],
    frame_index: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, np.ndarray]]:
    nodes: dict[int, dict[str, Any]] = {}
    current_centers: dict[int, np.ndarray] = {}
    for object_index, obj in enumerate(objects):
        bbox_points = np.asarray(obj.get("bbox_np"), dtype=float)
        if bbox_points.ndim != 2 or bbox_points.shape[0] == 0 or bbox_points.shape[1] < 2:
            continue
        center = np.mean(bbox_points[:, :3], axis=0)
        extent = np.ptp(bbox_points[:, :3], axis=0)
        key = snapshot_object_key(obj, object_index)
        if key not in registry:
            registry[key] = len(registry)
        node_id = registry[key]
        class_id = dominant_class_id(obj.get("class_id", []))
        tag = class_names[class_id] if 0 <= class_id < len(class_names) else "object"
        displacement = 0.0
        if node_id in previous_centers:
            displacement = float(np.linalg.norm(center[:2] - previous_centers[node_id][:2]))
        dynamic = tag in DYNAMIC_CLASSES and displacement > 0.75
        nodes[node_id] = {
            "id": node_id,
            "bbox_center": center.tolist(),
            "dynamic": dynamic,
            "observation_count": 1,
            "first_seen_frame": frame_index,
            "last_seen_frame": frame_index,
            "metadata": {
                "object_tag": tag,
                "bbox_extent": extent.tolist(),
                "snapshot_object_key": list(key),
                "frame_displacement_m": displacement,
            },
        }
        current_centers[node_id] = center
    return dict(sorted(nodes.items())), current_centers


def snapshot_object_key(obj: dict[str, Any], object_index: int) -> tuple[Any, ...]:
    color = np.asarray(obj.get("inst_color", []), dtype=float).reshape(-1)
    if color.size >= 3 and np.all(np.isfinite(color[:3])):
        return ("inst_color",) + tuple(round(float(value), 6) for value in color[:3])
    return ("object_index", int(object_index))


def dominant_class_id(values: Any) -> int:
    array = np.asarray(values, dtype=int).reshape(-1)
    if array.size == 0:
        return -1
    counts = Counter(int(value) for value in array)
    return min(counts, key=lambda item: (-counts[item], item))


def compare_edge_sets(
    incremental_edges: Iterable[SceneGraphEdge],
    full_edges: Iterable[SceneGraphEdge],
) -> dict[str, Any]:
    incremental_active = {
        (edge.source, edge.target, edge.relation)
        for edge in incremental_edges
        if edge.metadata.get("stale_since_frame") is None
    }
    full_keys = {(edge.source, edge.target, edge.relation) for edge in full_edges}
    union = incremental_active | full_keys
    intersection = incremental_active & full_keys
    agreement = 1.0 if not union else len(intersection) / len(union)
    return {
        "active_edge_agreement": float(agreement),
        "matching_edge_count": len(intersection),
        "union_edge_count": len(union),
        "incremental_only": [list(item) for item in sorted(incremental_active - full_keys)],
        "full_only": [list(item) for item in sorted(full_keys - incremental_active)],
        "ttl_stale_edge_count": sum(
            edge.metadata.get("stale_since_frame") is not None for edge in incremental_edges
        ),
    }


def plot_incremental_comparison(comparisons: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [item["frame_index"] for item in comparisons]
    agreement = [item["active_edge_agreement"] for item in comparisons]
    incremental_counts = [item["incremental_edge_count"] for item in comparisons]
    full_counts = [item["full_edge_count"] for item in comparisons]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(frames, agreement, color="#0072B2", linewidth=2.0)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Active-edge agreement")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(frames, incremental_counts, color="#D55E00", label="Incremental")
    axes[1].plot(frames, full_counts, color="#009E73", linestyle="--", label="Full rebuild")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Edge count")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")
    fig.suptitle("Stage 3 incremental vs full relation graph")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def legacy_edge_statistics(path: Path, node_count: int) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    counts = Counter(str(item[2]) for item in payload)
    degree = Counter()
    for source, target, _relation in payload:
        degree[int(source)] += 1
        degree[int(target)] += 1
    return {
        "edge_count": len(payload),
        "relation_counts": dict(sorted(counts.items())),
        "average_degree": (sum(degree.values()) / max(node_count, 1)),
        "maximum_degree": max(degree.values(), default=0),
        "source_path": str(path.resolve()),
    }


def structured_edge_statistics(edges: Iterable[SceneGraphEdge], node_count: int) -> dict[str, Any]:
    edges = list(edges)
    counts = Counter(edge.relation for edge in edges)
    degree = Counter()
    for edge in edges:
        degree[edge.source] += 1
        degree[edge.target] += 1
    return {
        "edge_count": len(edges),
        "relation_counts": dict(sorted(counts.items())),
        "symmetric_edge_count": sum(not edge.directed for edge in edges),
        "directed_edge_count": sum(edge.directed for edge in edges),
        "average_degree": (sum(degree.values()) / max(node_count, 1)),
        "maximum_degree": max(degree.values(), default=0),
    }


def load_trajectory_xy(path: Path) -> np.ndarray:
    poses = np.loadtxt(path)
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    centers = []
    for row in poses:
        if row.size >= 16:
            centers.append(row.reshape(4, 4)[:2, 3])
        elif row.size >= 2:
            centers.append(row[:2])
    return np.asarray(centers, dtype=float)


def trajectory_heading(trajectory: np.ndarray) -> np.ndarray | None:
    if len(trajectory) < 2:
        return None
    for index in range(len(trajectory) - 1, 0, -1):
        delta = trajectory[index] - trajectory[index - 1]
        norm = float(np.linalg.norm(delta))
        if norm > 1e-9:
            return delta / norm
    return None


def load_gzip_pickle(path: Path) -> Any:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def deterministic_payload_hash(payload: Any) -> str:
    canonical = json.dumps(
        strip_nondeterministic(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def strip_nondeterministic(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): strip_nondeterministic(item)
            for key, item in value.items()
            if key not in {"timing", "deterministic_content_sha256"}
            and not str(key).endswith("_time_s")
            and not str(key).endswith("_runtime_s")
        }
    if isinstance(value, list):
        return [strip_nondeterministic(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _selected_scene_names(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(item["scene_name"]) for item in payload.get("scenes", [])]


def _resolve_stage3_inputs(
    project_root: Path,
    scene_name: str,
) -> tuple[dict[str, Path] | None, dict[str, Any]]:
    stage2_inputs, stage2_diagnostic = _resolve_scene_inputs(project_root, scene_name)
    if stage2_inputs is None:
        return None, {
            "success": False,
            "failure_code": "MISSING_STAGE2_INPUTS",
            "failure_reason": stage2_diagnostic,
        }
    scene_dir = stage2_inputs["scene_dir"]
    qwen_exp_dir = stage2_inputs["qwen_exp_dir"]
    baseline_exp_dir = stage2_inputs["baseline_exp_dir"]
    snapshot_dir = select_snapshot_directory(scene_dir / "objects_all_frames")
    required = {
        "scene_graph": qwen_exp_dir / "scene_graph.json",
        "legacy_edges": qwen_exp_dir / "cfslam_scenegraph_edges.pkl",
        "object_map": qwen_exp_dir / "map/scene_map_cfslam_pruned.pkl.gz",
        "poses": scene_dir / "poses.txt",
        "bev_map": baseline_exp_dir / "navigation/bev_map.npz",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if snapshot_dir is None:
        missing.append("objects_all_frames")
    if missing:
        return None, {
            "success": False,
            "failure_code": "MISSING_STAGE3_INPUTS",
            "failure_reason": "missing: " + ", ".join(missing),
        }
    return {
        **required,
        "scene_dir": scene_dir,
        "qwen_exp_dir": qwen_exp_dir,
        "baseline_exp_dir": baseline_exp_dir,
        "snapshot_dir": snapshot_dir,
    }, {"success": True}


def select_snapshot_directory(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        frame_count = sum(path.name.split(".")[0].isdigit() for path in directory.glob("*.pkl.gz"))
        preference = int("tracking_gate" in directory.name)
        candidates.append((preference, frame_count, directory.name, directory))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return candidates[0][3]


def _fallback_vehicle_pose(trajectory: np.ndarray) -> list[float]:
    if not len(trajectory):
        return [0.0, 0.0, 0.0, 1.0, 0.0]
    heading = trajectory_heading(trajectory)
    if heading is None:
        heading = np.asarray([1.0, 0.0], dtype=float)
    return [float(trajectory[0, 0]), float(trajectory[0, 1]), 0.0, float(heading[0]), float(heading[1])]


if __name__ == "__main__":
    main()
