from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from conceptgraph.navigation import SemanticNavigationPipeline
from conceptgraph.navigation.target_selector import target_selection_result_to_dict
from conceptgraph.scripts.run_nuscenes_navigation_examples import make_cfg


DEFAULT_PROJECT_ROOT = Path("/root/autodl-tmp/lizichong")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase-2 target selection on existing offline navigation inputs.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--scene-count", type=int, default=2)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else project_root / "outputs/project_v2/offline_navigation"
    )
    selected_manifest = project_root / "outputs/project_v2/subset/selected_scenes.json"
    base_target_config = OmegaConf.load(
        project_root / "conceptgraph/hydra_configs/semantic_navigation.yaml"
    ).target_selection
    requested_scenes = args.scenes or _selected_scene_names(selected_manifest)

    output_root.mkdir(parents=True, exist_ok=True)
    first_root = output_root / "stage2_first_reachable"
    joint_root = output_root / "stage2_joint_utility"
    first_root.mkdir(parents=True, exist_ok=True)
    joint_root.mkdir(parents=True, exist_ok=True)

    comparisons: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for scene_name in requested_scenes:
        inputs, diagnostic = _resolve_scene_inputs(project_root, scene_name)
        if inputs is None:
            skipped.append({"scene_name": scene_name, "reason": diagnostic})
            continue
        reference, diagnostic = _reference_navigation_case(inputs["qwen_exp_dir"])
        if reference is None:
            skipped.append({"scene_name": scene_name, "reason": diagnostic})
            continue

        first = _run_mode(
            mode="first_reachable",
            inputs=inputs,
            reference=reference,
            target_config=base_target_config,
            output_dir=first_root / scene_name,
        )
        joint = _run_mode(
            mode="joint_utility",
            inputs=inputs,
            reference=reference,
            target_config=base_target_config,
            output_dir=joint_root / scene_name,
        )
        comparisons.append(_comparison_row(scene_name, reference, first, joint))
        if len(comparisons) >= max(1, args.scene_count):
            break

    payload = {
        "schema_version": 1,
        "input_policy": "existing scene graph, object map, BEV map, poses, and navigation result only",
        "selected_scenes_manifest": str(selected_manifest),
        "requested_scene_count": max(1, args.scene_count),
        "completed_scene_count": len(comparisons),
        "scenes": comparisons,
        "skipped_scenes": skipped,
    }
    comparison_path = output_root / "stage2_comparison.json"
    comparison_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if len(comparisons) < max(1, args.scene_count):
        raise RuntimeError(
            f"Only {len(comparisons)} scenes had complete offline inputs; "
            f"required {max(1, args.scene_count)}. See {comparison_path}."
        )
    print(comparison_path)


def _selected_scene_names(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(scene["scene_name"]) for scene in payload.get("scenes", [])]


def _resolve_scene_inputs(project_root: Path, scene_name: str) -> tuple[dict[str, Path] | None, str]:
    scene_dir = project_root / "dataset/nuscenes" / scene_name
    qwen_exp_dir = scene_dir / "exp_qwen2vl_semantics_v1"
    baseline_exp_dir = scene_dir / "exp_streamlined_yolo_outdoor_objects_v1"
    required = {
        "scene directory": scene_dir,
        "scene graph": qwen_exp_dir / "scene_graph.json",
        "object map": qwen_exp_dir / "map/scene_map_cfslam_pruned.pkl.gz",
        "BEV occupancy map": baseline_exp_dir / "navigation/bev_map.npz",
        "vehicle poses": scene_dir / "poses.txt",
    }
    missing = [label for label, path in required.items() if not path.exists()]
    if missing:
        return None, "missing offline inputs: " + ", ".join(missing)
    return {
        "scene_dir": scene_dir,
        "qwen_exp_dir": qwen_exp_dir,
        "baseline_exp_dir": baseline_exp_dir,
    }, "complete"


def _reference_navigation_case(qwen_exp_dir: Path) -> tuple[dict[str, Any] | None, str]:
    plan_paths = sorted(qwen_exp_dir.glob("navigation_examples/*/semantic_navigation_plan.json"))
    preferred: list[tuple[Path, dict[str, Any]]] = []
    fallback: list[tuple[Path, dict[str, Any]]] = []
    for path in plan_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        raw_path = payload.get("path", {}).get("world_xyz_raw_astar") or payload.get("path", {}).get("world_xyz")
        if not raw_path:
            continue
        entry = (path, payload)
        categories = payload.get("query", {}).get("target_categories", [])
        if categories == ["car"] and not payload.get("query", {}).get("target_colors"):
            preferred.append(entry)
        else:
            fallback.append(entry)
    choices = preferred or fallback
    if not choices:
        return None, "no parseable existing navigation plan with a path"
    path, payload = choices[0]
    raw_path = payload.get("path", {}).get("world_xyz_raw_astar") or payload["path"]["world_xyz"]
    start = np.asarray(raw_path[0], dtype=float)
    heading = np.asarray([1.0, 0.0], dtype=float)
    if len(raw_path) > 1:
        heading = np.asarray(raw_path[1], dtype=float)[:2] - start[:2]
        norm = float(np.linalg.norm(heading))
        heading = heading / norm if norm > 1e-9 else np.asarray([1.0, 0.0], dtype=float)
    return {
        "source_plan": str(path),
        "instruction": str(payload.get("query", {}).get("raw_text", "前往小汽车附近")),
        "vehicle_pose": start[:3].tolist() + heading[:2].tolist(),
    }, "complete"


def _run_mode(
    mode: str,
    inputs: dict[str, Path],
    reference: dict[str, Any],
    target_config,
    output_dir: Path,
) -> dict[str, Any]:
    cfg = make_cfg(
        qwen_exp_dir=inputs["qwen_exp_dir"],
        baseline_exp_dir=inputs["baseline_exp_dir"],
        output_dir=output_dir,
        query_text=reference["instruction"],
        vehicle_pose=reference["vehicle_pose"],
    )
    cfg.target_selection = OmegaConf.create(OmegaConf.to_container(target_config, resolve=True))
    cfg.target_selection.mode = mode
    pipeline = SemanticNavigationPipeline(cfg)
    started = time.perf_counter()
    result = pipeline.select(reference["instruction"], np.asarray(reference["vehicle_pose"], dtype=float))
    elapsed = float(time.perf_counter() - started)
    result.timing["offline_runner_s"] = elapsed
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.save_selection_result(result, output_dir, save_plan=True, save_visualizations=False)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(OmegaConf.to_container(cfg, resolve=True), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "result": result,
        "payload": target_selection_result_to_dict(result),
        "elapsed_s": elapsed,
        "output_dir": str(output_dir),
    }


def _comparison_row(
    scene_name: str,
    reference: dict[str, Any],
    first: dict[str, Any],
    joint: dict[str, Any],
) -> dict[str, Any]:
    first_summary = _mode_summary(first)
    joint_summary = _mode_summary(joint)
    selection_changed = _selection_identity(first_summary) != _selection_identity(joint_summary)
    if not first_summary["success"] or not joint_summary["success"]:
        reason = "至少一种模式失败；详见结构化 failure_code 与 rejection_statistics"
    elif selection_changed:
        reason = "联合模式在Top-K内综合路径代价、clearance、fallback和动态风险后改变了对象或目标点"
    else:
        reason = "联合模式评估了Top-K，但最高效用目标与first_reachable一致"
    return {
        "scene_name": scene_name,
        "instruction": reference["instruction"],
        "reference_navigation_plan": reference["source_plan"],
        "vehicle_pose": reference["vehicle_pose"],
        "first_reachable": first_summary,
        "joint_utility": joint_summary,
        "selection_changed": selection_changed,
        "change_reason": reason,
    }


def _mode_summary(run: dict[str, Any]) -> dict[str, Any]:
    result = run["result"]
    selected = result.selected_candidate
    rejection_statistics = Counter()
    for candidate in result.evaluated_candidates:
        if candidate.rejection_reason:
            rejection_statistics[candidate.rejection_reason] += 1
        for goal in candidate.goal_candidates:
            if goal.rejection_reason:
                rejection_statistics[goal.rejection_reason] += 1
    return {
        "success": bool(result.success),
        "mode": result.mode,
        "failure_code": result.failure_code,
        "failure_reason": result.failure_reason,
        "selected_object": None
        if selected is None
        else {"node_id": int(selected.node_id), "object_tag": selected.debug.get("object_tag")},
        "selected_goal_xy": None if selected is None or selected.target_point is None else selected.target_point[:2].tolist(),
        "selected_goal_rc": None if selected is None or selected.target_grid is None else list(selected.target_grid),
        "semantic_rank": None if selected is None else selected.semantic_rank,
        "path_length_m": None if selected is None else selected.path_length_m,
        "path_cost": None if selected is None else selected.path_cost,
        "min_clearance_m": None if selected is None else _finite_or_none(selected.min_clearance_m),
        "fallback_distance_m": None if selected is None else selected.fallback_distance_m,
        "dynamic_risk": None if selected is None else selected.dynamic_risk,
        "selection_utility": None if selected is None else selected.selection_utility,
        "selection_components": None if selected is None else selected.selection_components,
        "rejection_statistics": dict(sorted(rejection_statistics.items())),
        "evaluated_candidate_count": len(result.evaluated_candidates),
        "runtime_s": run["elapsed_s"],
        "scoring_weights": result.scoring_weights,
        "output_dir": run["output_dir"],
    }


def _selection_identity(summary: dict[str, Any]) -> tuple[Any, Any]:
    selected = summary.get("selected_object") or {}
    return selected.get("node_id"), summary.get("selected_goal_rc")


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


if __name__ == "__main__":
    main()
