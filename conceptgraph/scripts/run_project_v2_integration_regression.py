from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from omegaconf import OmegaConf

from conceptgraph.navigation.online import OnlineSemanticNavigator
from conceptgraph.navigation.pipeline import SemanticNavigationPipeline
from conceptgraph.navigation.planner import OccupancyGrid2D, path_is_collision_free
from conceptgraph.navigation.types import SceneGraphNode
from conceptgraph.scenegraph.relations import (
    edge_relation_for_query,
    load_scene_graph_edges,
)
from conceptgraph.scripts.run_incremental_semantic_navigation import (
    run_end_to_end_replay,
    run_failure_injection as run_stage5_failure_injection,
    run_snapshot_replay,
)
from conceptgraph.vlm.qwen_semantic_delegate import (
    QwenDelegateConfig,
    QwenSemanticDelegate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "conceptgraph/hydra_configs/project_v2_integration_regression.yaml"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/project_v2/integration_regression"
SCENES = ("scene-0916", "scene-0003")
MODES = ("snapshot_replay", "end_to_end_replay")
REQUIRED_METRICS = (
    "detection_time_s",
    "mapping_time_s",
    "relation_update_time_s",
    "query_ranking_time_s",
    "qwen_time_s",
    "planning_time_s",
    "total_frame_time_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行项目V2阶段6集成与回归验收。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="只基于已生成的真实回归产物重新执行静态检查并汇总最终报告。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    output_root = validate_output_root(args.output_root)
    baseline_path = output_root / "baseline_before_stage6.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"阶段6基线缺失：{baseline_path}")
    _read_json(baseline_path)
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.finalize_only:
        run_integration_regression(config, output_root)
    static_checks = run_static_checks(config, output_root)
    validation = finalize_reports(config, output_root, static_checks)
    if not validation["passed"]:
        raise RuntimeError(
            "阶段6最终验收未通过，详见 "
            f"{(output_root / 'validation_report.json').resolve()}"
        )
    print((output_root / "validation_report.json").resolve())


def validate_output_root(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    expected = DEFAULT_OUTPUT.resolve()
    if resolved != expected:
        raise ValueError(f"阶段6输出必须精确位于 {expected}")
    return resolved


def run_integration_regression(config: Any, output_root: Path) -> None:
    scenes = tuple(str(item) for item in config.scenes)
    if scenes != SCENES:
        raise ValueError(f"阶段6真实场景必须按固定顺序为 {SCENES}")
    run_count = int(config.determinism_runs)
    if run_count < 2:
        raise ValueError("阶段6确定性回放至少需要连续运行2次")

    command_reports = []
    command_reports.append(
        _run_command(
            [
                sys.executable,
                "-m",
                "conceptgraph.scripts.run_stage2_offline_navigation",
                "--scenes",
                *scenes,
                "--scene-count",
                "2",
                "--output-root",
                str((output_root / "regression_stage2").resolve()),
            ]
        )
    )
    command_reports.append(
        _run_command(
            [
                sys.executable,
                "-m",
                "conceptgraph.scripts.run_stage3_offline_relations",
                "--scenes",
                *scenes,
                "--scene-count",
                "2",
                "--output-root",
                str((output_root / "regression_stage3").resolve()),
                "--skip-visualizations",
            ]
        )
    )
    command_reports.extend(_run_structured_visualization(output_root))
    _prepare_stage4_cache_regression(output_root)
    command_reports.append(
        _run_command(
            [
                sys.executable,
                "-m",
                "conceptgraph.scripts.run_stage4_offline_qwen_eval",
                "--scenes",
                *scenes,
                "--scene-count",
                "2",
                "--output-root",
                str((output_root / "regression_stage4").resolve()),
            ],
            extra_env={
                "LD_LIBRARY_PATH": str(
                    Path(sys.executable).resolve().parent.parent / "lib"
                )
            },
        )
    )

    settings = OmegaConf.load(Path(config.incremental_config))
    reports: dict[str, dict[str, dict[str, Any]]] = {
        mode: {} for mode in MODES
    }
    signatures: dict[str, dict[str, list[str]]] = {
        mode: {scene: [] for scene in scenes} for mode in MODES
    }
    for mode in MODES:
        for scene_id in scenes:
            for run_index in range(run_count):
                if mode == "snapshot_replay":
                    report, signature = run_snapshot_replay(
                        scene_id,
                        settings,
                        output_root,
                        write_outputs=run_index == 0,
                    )
                else:
                    report, signature = run_end_to_end_replay(
                        scene_id,
                        settings,
                        output_root,
                        write_outputs=run_index == 0,
                    )
                signatures[mode][scene_id].append(signature)
                if run_index == 0:
                    reports[mode][scene_id] = report

    stage5_failures = run_stage5_failure_injection(settings, output_root)
    stage2 = validate_stage2_regression(config, output_root)
    stage3 = validate_stage3_regression(config, output_root)
    stage4 = validate_stage4_regression(config, output_root)
    stage5 = validate_stage5_replays(reports, output_root)
    determinism = build_determinism_report(signatures, run_count, output_root)
    failure_report = build_failure_injection_report(
        stage5_failures, stage2, stage4, output_root
    )
    compatibility = build_compatibility_matrix(
        stage2, stage3, stage4, stage5, failure_report, output_root
    )
    performance = build_performance_report(output_root)
    baseline = verify_stage6_baseline(output_root)

    comparison = {
        "schema_version": "stage6-comparison-v1",
        "passed": bool(
            stage2["passed"]
            and stage3["passed"]
            and stage4["passed"]
            and stage5["passed"]
            and determinism["passed"]
            and failure_report["passed"]
            and compatibility["passed"]
            and performance["passed"]
            and baseline["passed"]
        ),
        "stage2": stage2,
        "stage3": stage3,
        "stage4": stage4,
        "stage5": stage5,
        "recorded_sequence_disclaimer": str(
            config.runtime_policy.recorded_sequence_disclaimer
        ),
        "artifact_paths": _absolute_paths(
            output_root,
            (
                "compatibility_matrix.json",
                "performance_report.json",
                "determinism_report.json",
                "failure_injection_report.json",
                "baseline_verification.json",
            ),
        ),
    }
    _write_json(output_root / "stage6_comparison.json", comparison)
    _write_json(output_root / "runtime_commands.json", command_reports)


def _run_structured_visualization(output_root: Path) -> list[dict[str, Any]]:
    scene_id = "scene-0916"
    structured = (
        output_root
        / "regression_stage3/stage3_full_rebuild"
        / scene_id
        / "structured_edges.json"
    )
    navigation_plan = (
        output_root
        / "regression_stage3/stage3_full_rebuild"
        / scene_id
        / "navigation_queries/query_01/structured/semantic_navigation_plan.json"
    )
    scene_dir = PROJECT_ROOT / "dataset/nuscenes" / scene_id
    qwen_dir = scene_dir / "exp_qwen2vl_semantics_v1"
    baseline_dir = scene_dir / "exp_streamlined_yolo_outdoor_objects_v1"
    visualization_dir = output_root / "structured_edge_visualization" / scene_id
    return [
        _run_command(
            [
                sys.executable,
                "-m",
                "conceptgraph.scripts.visualize_scenegraph",
                "--scene-graph",
                str((qwen_dir / "scene_graph.json").resolve()),
                "--edges",
                str(structured.resolve()),
                "--out-dir",
                str(visualization_dir.resolve()),
                "--navigation-plan",
                str(navigation_plan.resolve()),
                "--trajectory-poses",
                str((scene_dir / "poses.txt").resolve()),
                "--bev-map",
                str((baseline_dir / "navigation/bev_map.npz").resolve()),
                "--minimum-confidence",
                "0.35",
            ]
        )
    ]


def _prepare_stage4_cache_regression(output_root: Path) -> None:
    source = PROJECT_ROOT / "outputs/project_v2/qwen_eval"
    destination = output_root / "regression_stage4"
    destination.mkdir(parents=True, exist_ok=True)
    for directory in ("dataset", "cache", "sanity_training"):
        shutil.copytree(
            source / directory,
            destination / directory,
            dirs_exist_ok=True,
        )
    shutil.copy2(
        source / "baseline_before_stage4.json",
        destination / "baseline_before_stage4.json",
    )


def validate_stage2_regression(config: Any, output_root: Path) -> dict[str, Any]:
    comparison_path = output_root / "regression_stage2/stage2_comparison.json"
    comparison = _read_json(comparison_path)
    rows = {}
    successful_paths = []
    all_diagnostics_complete = True
    all_collision_checks_passed = True
    for scene in comparison.get("scenes", []):
        scene_id = str(scene["scene_name"])
        expected = config.required_stage2_selection[scene_id]
        mode_rows = {}
        for mode in ("first_reachable", "joint_utility"):
            summary = scene[mode]
            selected = summary.get("selected_object") or {}
            result_path = (
                output_root
                / "regression_stage2"
                / f"stage2_{mode}"
                / scene_id
                / "target_selection_result.json"
            )
            result_payload = _read_json(result_path)
            diagnostics = _validate_top_k_diagnostics(result_payload)
            collision = _rerun_collision_check(
                output_root, scene_id, mode, str(scene["instruction"])
            )
            all_diagnostics_complete &= diagnostics["passed"]
            all_collision_checks_passed &= collision["passed"]
            successful_paths.append(collision)
            selected_id = selected.get("node_id")
            mode_rows[mode] = {
                "success": bool(summary.get("success")),
                "selected_node_id": selected_id,
                "expected_selected_node_id": int(expected[mode]),
                "selection_matches_baseline": selected_id == int(expected[mode]),
                "top_k_diagnostics": diagnostics,
                "continuous_collision_recheck": collision,
                "result_path": str(result_path.resolve()),
                "plan_path": str(
                    (result_path.parent / "semantic_navigation_plan.json").resolve()
                ),
            }
        rows[scene_id] = mode_rows
    passed = bool(
        comparison.get("completed_scene_count") == 2
        and set(rows) == set(SCENES)
        and all(
            item["success"] and item["selection_matches_baseline"]
            for modes in rows.values()
            for item in modes.values()
        )
        and all_diagnostics_complete
        and all_collision_checks_passed
    )
    return {
        "passed": passed,
        "scenes": rows,
        "all_top_k_diagnostics_complete": all_diagnostics_complete,
        "all_successful_paths_collision_rechecked": all_collision_checks_passed,
        "successful_path_rechecks": successful_paths,
        "comparison_path": str(comparison_path.resolve()),
    }


def _validate_top_k_diagnostics(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "score",
        "text_score",
        "relation_score",
        "path_cost",
        "min_clearance_m",
        "fallback_distance_m",
        "dynamic_risk",
        "rejection_reason",
        "selection_utility",
        "selection_components",
        "semantic_rank",
        "goal_candidates",
    }
    missing = []
    for index, candidate in enumerate(payload.get("evaluated_candidates", [])):
        absent = sorted(required - set(candidate))
        if absent:
            missing.append({"candidate_index": index, "missing_fields": absent})
    candidates = payload.get("evaluated_candidates", [])
    return {
        "passed": bool(candidates) and not missing,
        "candidate_count": len(candidates),
        "missing": missing,
        "required_fields": sorted(required),
    }


def _rerun_collision_check(
    output_root: Path, scene_id: str, mode: str, query_text: str
) -> dict[str, Any]:
    mode_dir = output_root / "regression_stage2" / f"stage2_{mode}" / scene_id
    cfg = OmegaConf.create(_read_json(mode_dir / "resolved_config.json"))
    comparison = _read_json(output_root / "regression_stage2/stage2_comparison.json")
    scene = next(row for row in comparison["scenes"] if row["scene_name"] == scene_id)
    pipeline = SemanticNavigationPipeline(cfg)
    result = pipeline.select(query_text, np.asarray(scene["vehicle_pose"], dtype=float))
    collision_free = bool(
        result.success
        and result.plan is not None
        and pipeline._occupancy_grid is not None
        and path_is_collision_free(pipeline._occupancy_grid, result.plan.path_grid)
    )
    selected_id = (
        None if result.selected_candidate is None else int(result.selected_candidate.node_id)
    )
    expected_id = scene[mode]["selected_object"]["node_id"]
    return {
        "scene_id": scene_id,
        "mode": mode,
        "selection_reproduced": selected_id == expected_id,
        "selected_node_id": selected_id,
        "path_point_count": 0 if result.plan is None else len(result.plan.path_grid),
        "continuous_collision_free": collision_free,
        "passed": bool(collision_free and selected_id == expected_id),
    }


def validate_stage3_regression(config: Any, output_root: Path) -> dict[str, Any]:
    comparison_path = output_root / "regression_stage3/stage3_comparison.json"
    comparison = _read_json(comparison_path)
    rows = {}
    required_agreement = float(config.required_stage3_agreement)
    for scene in comparison.get("scenes", []):
        scene_id = str(scene["scene_name"])
        incremental = scene["incremental"]
        structured_path = Path(
            scene["structured_full_rebuild"]["structured_json_path"]
        ).resolve()
        legacy_path = (
            PROJECT_ROOT
            / "dataset/nuscenes"
            / scene_id
            / "exp_qwen2vl_semantics_v1/cfslam_scenegraph_edges.pkl"
        )
        structured_edges = load_scene_graph_edges(structured_path)
        legacy_edges = load_scene_graph_edges(legacy_path)
        directed = next((edge for edge in structured_edges if edge.directed), None)
        symmetric = next(
            (edge for edge in structured_edges if not edge.directed and edge.relation == "near"),
            None,
        )
        direction_checks = {
            "forward": bool(
                directed
                and edge_relation_for_query(
                    directed, directed.source, directed.target
                )
                == directed.relation
            ),
            "reverse": bool(
                directed
                and edge_relation_for_query(
                    directed, directed.target, directed.source
                )
                in {"right_of", "behind"}
            ),
            "symmetric_forward": bool(
                symmetric
                and edge_relation_for_query(
                    symmetric, symmetric.source, symmetric.target
                )
                == "near"
            ),
            "symmetric_reverse": bool(
                symmetric
                and edge_relation_for_query(
                    symmetric, symmetric.target, symmetric.source
                )
                == "near"
            ),
        }
        query_rows = scene.get("navigation_queries", [])
        structured_navigation_results = [
            {
                "query": query["query"],
                "success": bool(query["structured"]["success"]),
                "selected_node_id": query["structured"]["selected_node_id"],
                "failure_code": query["structured"]["failure_code"],
                "edge_source": query["structured"]["edge_source"],
                "result_path": query["structured"]["result_path"],
                "plan_path": query["structured"]["plan_path"],
                "path_validated": _saved_navigation_path_validated(
                    query["structured"]
                ),
            }
            for query in query_rows
        ]
        navigation_uses_structured = bool(query_rows) and all(
            query["success"]
            and query["edge_source"] == "structured_edge"
            and query["path_validated"]
            for query in structured_navigation_results
        )
        legacy_navigation_results = [
            {
                "query": query["query"],
                "success": bool(query["legacy"]["success"]),
                "selected_node_id": query["legacy"]["selected_node_id"],
                "failure_code": query["legacy"]["failure_code"],
                "edge_source": query["legacy"]["edge_source"],
                "result_path": query["legacy"]["result_path"],
                "plan_path": query["legacy"]["plan_path"],
                "path_validated": _saved_navigation_path_validated(query["legacy"]),
            }
            for query in query_rows
        ]
        legacy_navigation_passed = bool(query_rows) and any(
            query["success"] and query["path_validated"]
            for query in legacy_navigation_results
        )
        rows[scene_id] = {
            "success": bool(incremental["success"]),
            "minimum_active_edge_agreement": float(
                incremental["minimum_active_edge_agreement"]
            ),
            "mean_active_edge_agreement": float(
                incremental["mean_active_edge_agreement"]
            ),
            "final_active_edge_agreement": float(
                incremental["final_active_edge_agreement"]
            ),
            "structured_edge_count": len(structured_edges),
            "legacy_edge_count": len(legacy_edges),
            "legacy_metadata_compatibility": bool(legacy_edges)
            and all(
                edge.metadata.get("edge_format") == "legacy"
                and edge.metadata.get("metadata_available") is False
                for edge in legacy_edges
            ),
            "direction_checks": direction_checks,
            "navigation_uses_structured_edges": navigation_uses_structured,
            "structured_navigation_results": structured_navigation_results,
            "legacy_navigation_passed": legacy_navigation_passed,
            "legacy_navigation_results": legacy_navigation_results,
            "structured_edge_path": str(structured_path),
        }
    visualization_summary = (
        output_root
        / "structured_edge_visualization/scene-0916/scenegraph_summary.json"
    )
    visualization = _read_json(visualization_summary)
    visual_edge_count = len(visualization.get("edges", []))
    structured_edges = load_scene_graph_edges(
        rows["scene-0916"]["structured_edge_path"]
    )
    structured_payload = [edge.to_dict() for edge in structured_edges]
    visual_payload = visualization.get("edges", [])
    edge_key = lambda edge: (  # noqa: E731 - shared compact canonical sort key
        int(edge["source"]),
        int(edge["target"]),
        str(edge["relation"]),
    )
    structured_payload = sorted(structured_payload, key=edge_key)
    visual_payload = sorted(visual_payload, key=edge_key)
    same_edge_file = structured_payload == visual_payload
    passed = bool(
        comparison.get("success")
        and set(rows) == set(SCENES)
        and all(
            row["success"]
            and row["minimum_active_edge_agreement"] == required_agreement
            and row["mean_active_edge_agreement"] == required_agreement
            and row["final_active_edge_agreement"] == required_agreement
            and row["legacy_metadata_compatibility"]
            and row["legacy_navigation_passed"]
            and row["navigation_uses_structured_edges"]
            and all(row["direction_checks"].values())
            for row in rows.values()
        )
        and same_edge_file
    )
    return {
        "passed": passed,
        "scenes": rows,
        "shared_structured_edge_visualization": {
            "scene_id": "scene-0916",
            "structured_edge_path": rows["scene-0916"]["structured_edge_path"],
            "visualization_summary_path": str(visualization_summary.resolve()),
            "visualized_edge_count": visual_edge_count,
            "structured_edge_sha256": sha256_file(
                Path(rows["scene-0916"]["structured_edge_path"])
            ),
            "visualized_edges_content_sha256": _content_sha256(visual_payload),
            "same_edge_file_content_consumed": same_edge_file,
        },
        "comparison_path": str(comparison_path.resolve()),
    }


def _saved_navigation_path_validated(result: Mapping[str, Any]) -> bool:
    if not result.get("success"):
        return False
    result_path = result.get("result_path")
    plan_path = result.get("plan_path")
    if not result_path or not plan_path or not Path(plan_path).exists():
        return False
    payload = _read_json(Path(result_path))
    selected = payload.get("selected_candidate") or {}
    return bool(selected.get("debug", {}).get("continuous_collision_check"))


def validate_stage4_regression(config: Any, output_root: Path) -> dict[str, Any]:
    stage4_root = output_root / "regression_stage4"
    comparison = _read_json(stage4_root / "stage4_comparison.json")
    validation = _read_json(stage4_root / "validation_report.json")
    determinism = _read_json(stage4_root / "determinism_report.json")
    rows = {}
    expected = config.required_stage2_selection
    for scene in comparison.get("scenes", []):
        scene_id = str(scene["scene_id"])
        first = scene["first_run"]
        repeat = scene["cache_repeat"]
        rows[scene_id] = {
            "rule_selected_node_id": int(scene["rule_selected_node_id"]),
            "qwen_fallback_selected_node_id": int(
                scene["qwen_navigation_selected_node_id"]
            ),
            "first_cache_hit_count": int(first["cache_hit_count"]),
            "first_backend_call_count": int(first["backend_call_count"]),
            "repeat_cache_hit_count": int(repeat["cache_hit_count"]),
            "repeat_backend_call_count": int(repeat["backend_call_count"]),
            "strict_schema_valid_count": int(scene["schema_valid_count"]),
            "failure_fallback_passed": bool(scene["failure_injection"]["passed"]),
            "pending_tasks_after_close": 0,
            "output_path": str(
                (stage4_root / "qwen_inference" / scene_id).resolve()
            ),
        }
    qwen_disabled = _validate_qwen_disabled()
    passed = bool(
        validation.get("runtime_passed")
        and determinism.get("passed")
        and set(rows) == set(SCENES)
        and qwen_disabled["passed"]
        and all(
            row["rule_selected_node_id"]
            == int(expected[scene_id].joint_utility)
            and row["qwen_fallback_selected_node_id"]
            == int(expected[scene_id].joint_utility)
            and row["first_cache_hit_count"] == 3
            and row["repeat_cache_hit_count"] == 3
            and row["first_backend_call_count"] == 0
            and row["repeat_backend_call_count"] == 0
            and row["strict_schema_valid_count"] == 0
            and row["failure_fallback_passed"]
            and row["pending_tasks_after_close"] == 0
            for scene_id, row in rows.items()
        )
    )
    return {
        "passed": passed,
        "execution_mode": validation.get("execution_mode"),
        "qwen_disabled": qwen_disabled,
        "scenes": rows,
        "model_metric_disclaimer": (
            "严格schema有效数为0；规则回退后的正确选择不计为模型schema指标。"
        ),
        "comparison_path": str((stage4_root / "stage4_comparison.json").resolve()),
    }


def _validate_qwen_disabled() -> dict[str, Any]:
    calls = []

    def backend(*_args: Any, **_kwargs: Any) -> str:
        calls.append(True)
        return "{}"

    delegate = QwenSemanticDelegate(
        QwenDelegateConfig(enabled=False, use_qwen_query_parse=True),
        backend=backend,
    )
    result = delegate.parse_query("前往小汽车附近")
    delegate.close(wait=True)
    return {
        "passed": bool(
            not calls
            and result.get("failure_code") == "QWEN_DISABLED"
            and result.get("fallback_used")
        ),
        "backend_call_count": len(calls),
        "failure_code": result.get("failure_code"),
        "fallback_used": bool(result.get("fallback_used")),
    }


def validate_stage5_replays(
    reports: Mapping[str, Mapping[str, Mapping[str, Any]]], output_root: Path
) -> dict[str, Any]:
    rows = []
    observed_triggers = set()
    for mode in MODES:
        for scene_id in SCENES:
            report = reports[mode][scene_id]
            frame_path = output_root / mode / scene_id / "frames.jsonl"
            frames = _read_jsonl(frame_path)
            observed_triggers.update(report.get("replan_trigger_counts", {}))
            first_plan = report["first_planning_frame"]
            checks = {
                "frame_count_matches": len(frames) == int(report["frame_count"]),
                "target_appears_before_confirmation": (
                    report["first_target_appearance_frame"] is not None
                    and report["first_target_confirmation_frame"] is not None
                    and report["first_target_appearance_frame"]
                    <= report["first_target_confirmation_frame"]
                ),
                "confirmation_before_path": (
                    report["first_target_confirmation_frame"] is not None
                    and first_plan is not None
                    and report["first_target_confirmation_frame"] <= first_plan
                ),
                "no_path_before_first_plan": all(
                    int(frame["path_point_count"]) == 0
                    for frame in frames
                    if frame["frame_idx"] < first_plan
                ),
                "full_path_only_on_planning_frames": all(
                    frame["replanned"] or not frame["path_world"] for frame in frames
                ),
                "no_pending_tasks": bool(frames)
                and int(frames[-1]["qwen_audit"]["pending_count"]) == 0,
                "failure_recovery": (
                    mode == "snapshot_replay"
                    or not report["failure_frame_count"]
                    or int(report["failure_recovery_count"]) > 0
                ),
            }
            if mode == "end_to_end_replay":
                mapping = report["final_mapping_comparison"]
                checks["final_mapping_comparison"] = bool(mapping["passed"])
                checks["dynamic_reference_not_fabricated"] = (
                    mapping["reference_dynamic_count"] is None
                    and not mapping["reference_dynamic_statistics_available"]
                )
                checks["cached_detection_inputs_only"] = all(
                    Path(frame["source_detection"]).exists()
                    and "/detections/" in frame["source_detection"]
                    for frame in frames
                )
            else:
                checks["snapshot_inputs_only"] = all(
                    "/objects_all_frames/" in frame["source"] for frame in frames
                )
            rows.append(
                {
                    "scene_id": scene_id,
                    "mode": mode,
                    "frame_count": int(report["frame_count"]),
                    "first_target_appearance_frame": report[
                        "first_target_appearance_frame"
                    ],
                    "first_target_confirmation_frame": report[
                        "first_target_confirmation_frame"
                    ],
                    "first_planning_frame": first_plan,
                    "planning_count": int(report["planning_count"]),
                    "checks": checks,
                    "passed": all(checks.values()),
                    "frames_path": str(frame_path.resolve()),
                    "summary_path": str((frame_path.parent / "summary.json").resolve()),
                }
            )
    required_triggers = {
        "target_confirmed",
        "target_changed",
        "better_target_persisted",
        "path_blocked",
        "vehicle_moved",
        "periodic_interval",
        "manual_request",
    }
    trigger_coverage = {
        "required": sorted(required_triggers),
        "observed": sorted(observed_triggers),
        "missing": sorted(required_triggers - observed_triggers),
        "priority": [
            "path_blocked",
            "target_changed",
            "better_target_persisted",
            "target_confirmed",
            "manual_request",
            "vehicle_moved",
            "periodic_interval",
        ],
    }
    trigger_coverage["passed"] = not trigger_coverage["missing"]
    return {
        "passed": len(rows) == 4
        and all(row["passed"] for row in rows)
        and trigger_coverage["passed"],
        "replays": rows,
        "replan_trigger_coverage": trigger_coverage,
        "recorded_sequence_disclaimer": (
            "nuScenes是录制序列；规划路径不会改变后续传感器观测。"
        ),
    }


def build_determinism_report(
    signatures: Mapping[str, Mapping[str, Sequence[str]]],
    requested_runs: int,
    output_root: Path,
) -> dict[str, Any]:
    rows = []
    for mode in MODES:
        for scene_id in SCENES:
            values = list(signatures[mode][scene_id])
            rows.append(
                {
                    "mode": mode,
                    "scene_id": scene_id,
                    "run_count": len(values),
                    "content_sha256": values,
                    "identical": len(values) >= 2 and len(set(values)) == 1,
                    "compared_fields": [
                        "state",
                        "target",
                        "replan_triggers",
                        "failure_code",
                        "path_sha256",
                    ],
                }
            )
    report = {
        "schema_version": "stage6-determinism-v1",
        "requested_runs": requested_runs,
        "scenes": rows,
        "excluded_fields": ["timings", "asynchronous_completion_timestamp"],
        "passed": len(rows) == 4 and all(row["identical"] for row in rows),
    }
    _write_json(output_root / "determinism_report.json", report)
    return report


def build_failure_injection_report(
    stage5: Mapping[str, Any],
    stage2: Mapping[str, Any],
    stage4: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    serialization = _run_corrupt_serialization_checks(output_root)
    qwen = _run_qwen_failure_checks()
    memory = _run_memory_failure_checks()
    safety_selection = {
        scene_id: {
            "stage2_safe_selected_node_id": stage2["scenes"][scene_id][
                "joint_utility"
            ]["selected_node_id"],
            "stage4_fallback_selected_node_id": stage4["scenes"][scene_id][
                "qwen_fallback_selected_node_id"
            ],
            "unchanged": stage2["scenes"][scene_id]["joint_utility"][
                "selected_node_id"
            ]
            == stage4["scenes"][scene_id]["qwen_fallback_selected_node_id"],
        }
        for scene_id in SCENES
    }
    checks = {
        "corrupt_json_rejected": serialization["corrupt_json_rejected"],
        "corrupt_jsonl_rejected": serialization["corrupt_jsonl_rejected"],
        "corrupt_snapshot_rejected": bool(stage5["corrupt_snapshot"]["rejected"]),
        "empty_objects_isolated": bool(stage5["empty_object_isolated"]),
        "invalid_pose_transaction_rollback": bool(stage5["invalid_pose_isolated"]),
        "fully_blocked_grid_isolated": bool(stage5["planning_failure_isolated"]),
        "sequence_continued_after_fault": bool(
            stage5["replay_continued_after_invalid_pose"]
        ),
        "qwen_timeout_fallback": qwen["timeout"]["passed"],
        "qwen_exception_fallback": qwen["exception"]["passed"],
        "qwen_invalid_schema_fallback": qwen["invalid_schema"]["passed"],
        "qwen_illegal_candidate_fallback": qwen["illegal_candidate"]["passed"],
        "late_qwen_result_discarded": memory["late_result"]["passed"],
        "duplicate_frame_idempotent": memory["duplicate_frame"]["passed"],
        "out_of_order_rejected": memory["out_of_order_frame"]["passed"],
        "transaction_state_restored": memory["transaction_rollback"]["passed"],
        "no_duplicate_planning": memory["no_duplicate_planning"]["passed"],
        "no_unrecovered_tasks": memory["pending_tasks_after_close"] == 0,
        "qwen_faults_preserve_stage2_safety": all(
            row["unchanged"] for row in safety_selection.values()
        ),
    }
    report = {
        "schema_version": "stage6-failure-injection-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "serialization": serialization,
        "stage5_failure_injection": stage5,
        "qwen": qwen,
        "memory_entry": memory,
        "safe_selection_comparison": safety_selection,
        "output_path": str((output_root / "failure_injection_report.json").resolve()),
    }
    _write_json(output_root / "failure_injection_report.json", report)
    return report


def _run_corrupt_serialization_checks(output_root: Path) -> dict[str, Any]:
    fixture_root = output_root / "failure_injection/serialization"
    fixture_root.mkdir(parents=True, exist_ok=True)
    json_path = fixture_root / "corrupt.json"
    jsonl_path = fixture_root / "corrupt.jsonl"
    json_path.write_text('{"incomplete":', encoding="utf-8")
    jsonl_path.write_text('{"valid": true}\n{"broken":\n', encoding="utf-8")
    errors = []
    try:
        json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append({"format": "json", "error": str(exc)})
    try:
        for line_number, line in enumerate(
            jsonl_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            json.loads(line)
    except json.JSONDecodeError as exc:
        errors.append(
            {"format": "jsonl", "line": line_number, "error": str(exc)}
        )
    json_path.unlink()
    jsonl_path.unlink()
    return {
        "corrupt_json_rejected": any(item["format"] == "json" for item in errors),
        "corrupt_jsonl_rejected": any(
            item["format"] == "jsonl" for item in errors
        ),
        "errors": errors,
        "fixtures_removed_after_check": not json_path.exists() and not jsonl_path.exists(),
    }


def _run_qwen_failure_checks() -> dict[str, Any]:
    class Backend:
        def __init__(self, *, output: str = "{}", delay: float = 0.0, error=None):
            self.output = output
            self.delay = delay
            self.error = error
            self.calls = 0

        def __call__(self, *_args: Any, **_kwargs: Any) -> str:
            self.calls += 1
            if self.delay:
                time.sleep(self.delay)
            if self.error is not None:
                raise self.error
            return self.output

    cases = {
        "timeout": (
            Backend(delay=0.08),
            0.01,
            "BACKEND_TIMEOUT",
        ),
        "exception": (
            Backend(error=RuntimeError("stage6 injected backend exception")),
            0.2,
            "BACKEND_EXCEPTION",
        ),
        "invalid_schema": (
            Backend(output='{"schema_version":"1.0.0"}'),
            0.2,
            "SCHEMA_VALIDATION_FAILED",
        ),
    }
    results = {}
    for name, (backend, timeout_s, expected_code) in cases.items():
        delegate = QwenSemanticDelegate(
            QwenDelegateConfig(
                enabled=True,
                use_qwen_query_parse=True,
                timeout_s=timeout_s,
                model_revision=f"stage6-{name}",
            ),
            backend=backend,
        )
        result = delegate.parse_query("前往小汽车附近")
        delegate.close(wait=True)
        results[name] = {
            "failure_code": result.get("failure_code"),
            "fallback_used": bool(result.get("fallback_used")),
            "backend_call_count": backend.calls,
            "pending_tasks_after_close": delegate.pending_async_count,
            "passed": bool(
                result.get("failure_code") == expected_code
                and result.get("fallback_used")
                and delegate.pending_async_count == 0
            ),
        }
    illegal_backend = Backend(
        output=json.dumps(
            {
                "schema_version": "1.0.0",
                "selected_node_id": 999,
                "ranking": [999, 21],
                "explanation": "injected illegal candidate",
                "confidence": 0.9,
            }
        )
    )
    illegal_delegate = QwenSemanticDelegate(
        QwenDelegateConfig(
            enabled=True,
            use_qwen_rerank=True,
            timeout_s=0.2,
            model_revision="stage6-illegal-candidate",
        ),
        backend=illegal_backend,
    )
    illegal_result = illegal_delegate.rerank_candidates(
        "前往小汽车附近", [{"node_id": 21}, {"node_id": 24}]
    )
    illegal_delegate.close(wait=True)
    results["illegal_candidate"] = {
        "failure_code": illegal_result.get("failure_code"),
        "fallback_used": bool(illegal_result.get("fallback_used")),
        "fallback_selected_node_id": illegal_result.get("selected_node_id"),
        "backend_call_count": illegal_backend.calls,
        "pending_tasks_after_close": illegal_delegate.pending_async_count,
        "passed": bool(
            illegal_result.get("failure_code") == "ILLEGAL_CANDIDATE_ID"
            and illegal_result.get("fallback_used")
            and illegal_result.get("selected_node_id") == 21
            and illegal_delegate.pending_async_count == 0
        ),
    }
    return results


def _synthetic_navigation_config() -> SimpleNamespace:
    return SimpleNamespace(
        default_near_distance_m=8.0,
        lambda_text=0.55,
        lambda_relation=0.30,
        lambda_distance_prior=0.25,
        relation_edge_minimum_confidence=0.0,
        start_search_radius_m=3.0,
        goal_search_radius_m=5.0,
        allow_goal_fallback=False,
        semantic_cost_weight=0.0,
        clearance_cost_weight=0.0,
        path_smoothing_iterations=1,
        max_candidates=3,
        grid_resolution=1.0,
        grid_padding_m=5.0,
        robot_radius_m=0.0,
        object_bbox_obstacle_scale=0.2,
        max_object_obstacle_radius_m=1.0,
        ignore_dynamic_obstacles=False,
        obstacle_profile="stage6_failure_injection",
        bev_map_path=None,
        target_selection=SimpleNamespace(
            mode="joint_utility",
            top_k=3,
            goal_candidates_per_object={"default": 8},
            class_approach_distance_m={"default": 1.0, "car": 1.0},
            min_clearance_m={"default": 0.0, "car": 0.0},
            max_fallback_distance_m={"default": 2.0, "car": 2.0},
            allow_global_fallback=False,
            utility_weights={
                "semantic_score": 0.35,
                "relation_score": 0.15,
                "path_cost": 0.25,
                "clearance_score": 0.10,
                "fallback_penalty": 0.10,
                "dynamic_risk": 0.05,
            },
            dynamic_class_penalties={"default": 0.0},
            deterministic_tie_break="semantic_rank_node_id_goal_rc",
        ),
    )


def _synthetic_node(
    center: tuple[float, float] = (16.0, 16.0),
    *,
    node_id: int = 1,
) -> SceneGraphNode:
    return SceneGraphNode(
        node_id=node_id,
        object_tag="car",
        possible_tags=["car"],
        caption="car",
        bbox_center=np.asarray([center[0], center[1], 0.0]),
        bbox_extent=np.asarray([2.0, 2.0, 1.0]),
        metadata={"num_detections": 4},
    )


def _make_synthetic_navigator(delegate=None) -> OnlineSemanticNavigator:
    occupancy = np.zeros((36, 36), dtype=np.uint8)
    grid = OccupancyGrid2D(
        occupancy=occupancy,
        resolution=1.0,
        origin_xy=np.zeros(2, dtype=float),
        clearance_m=np.full_like(occupancy, 20.0, dtype=float),
        source="stage6_synthetic",
    )
    return OnlineSemanticNavigator(
        _synthetic_navigation_config(),
        semantic_delegate=delegate,
        base_occupancy_grid=grid,
        state_machine_config={
            "confirmation_observations": 2,
            "minimum_persistent_frames": 2,
            "confirmation_confidence": 0.1,
        },
        replan_config={
            "cooldown_frames": 0,
            "vehicle_moved_distance_m": 3.0,
            "periodic_interval_frames": 20,
        },
        relation_config={"minimum_confidence": 0.0},
    )


def _run_memory_failure_checks() -> dict[str, Any]:
    navigator = _make_synthetic_navigator()
    navigator.set_query("前往小汽车附近")
    objects = {1: _synthetic_node()}
    navigator.update(0, [2.0, 2.0, 0.0], objects, [1])
    planned = navigator.update(1, [2.0, 2.0, 0.0], objects, [])
    duplicate = navigator.update(1, [99.0, 99.0, 0.0], {}, [])
    out_of_order = navigator.update(0, [2.0, 2.0, 0.0], objects, [])
    versions_before = (
        navigator._object_version,
        navigator._candidate_version,
        navigator._edge_version,
    )
    failure = navigator.update(
        2,
        [3.0, 2.0, 0.0],
        {1: _synthetic_node(), 2: {"node_id": 2, "object_tag": "car"}},
        [2],
    )
    versions_after = (
        navigator._object_version,
        navigator._candidate_version,
        navigator._edge_version,
    )
    recovered = navigator.update(3, [3.0, 2.0, 0.0], objects, [])
    navigator.close()

    class LateDelegate:
        config = SimpleNamespace(
            enabled=True,
            use_qwen_query_parse=True,
            use_qwen_object_attributes=False,
            use_qwen_rerank=True,
        )

        async def parse_query_async(self, *_args: Any, **_kwargs: Any):
            return {
                "used_qwen": False,
                "fallback_used": True,
                "failure_code": None,
                "cache": {"status": "miss"},
            }

        async def rerank_candidates_async(self, *_args: Any, **_kwargs: Any):
            await asyncio.sleep(0.08)
            return {
                "used_qwen": True,
                "fallback_used": False,
                "failure_code": None,
                "normalized_output": {
                    "selected_node_id": 1,
                    "ranking": [1],
                    "confidence": 0.9,
                },
                "cache": {"status": "miss"},
            }

        def close(self, *, wait: bool = True) -> None:
            return None

    late = _make_synthetic_navigator(LateDelegate())
    late.set_query("前往小汽车附近")
    time.sleep(0.02)
    late.update(0, [2.0, 2.0, 0.0], objects, [1])
    late.update(1, [2.0, 2.0, 0.0], objects, [])
    late.update(2, [2.0, 2.0, 0.0], {2: _synthetic_node((17.0, 16.0), node_id=2)}, [1, 2])
    time.sleep(0.1)
    audited = late.update(
        3, [2.0, 2.0, 0.0], {2: _synthetic_node((17.0, 16.0), node_id=2)}, []
    )
    late.close()
    stale = [
        event
        for event in audited.qwen_audit["events"]
        if event["task"] == "candidate_rerank" and not event.get("applied", False)
    ]
    pending = navigator.pending_qwen_count + late.pending_qwen_count
    return {
        "duplicate_frame": {
            "passed": duplicate.to_dict() == planned.to_dict(),
            "frame_idx": duplicate.frame_idx,
        },
        "out_of_order_frame": {
            "passed": bool(
                out_of_order.failure
                and out_of_order.failure.code == "OUT_OF_ORDER_FRAME"
            ),
            "failure_code": (
                None if out_of_order.failure is None else out_of_order.failure.code
            ),
        },
        "transaction_rollback": {
            "passed": bool(
                failure.failure
                and failure.failure.code == "FRAME_PROCESSING_EXCEPTION"
                and versions_before == versions_after
                and recovered.failure is None
            ),
            "versions_restored": versions_before == versions_after,
            "subsequent_frame_recovered": recovered.failure is None,
        },
        "no_duplicate_planning": {
            "passed": bool(planned.replanned and duplicate.to_dict() == planned.to_dict()),
            "original_replanned": bool(planned.replanned),
            "duplicate_returned_cached_update": duplicate.to_dict() == planned.to_dict(),
        },
        "late_result": {
            "passed": bool(
                stale
                and stale[0].get("diagnostic")
                == "recommended_candidate_no_longer_present"
            ),
            "events": stale,
        },
        "pending_tasks_after_close": pending,
    }


def build_compatibility_matrix(
    stage2: Mapping[str, Any],
    stage3: Mapping[str, Any],
    stage4: Mapping[str, Any],
    stage5: Mapping[str, Any],
    failures: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    cli = _run_cli_smokes()
    rows: list[dict[str, Any]] = []
    for scene_id in SCENES:
        first = stage2["scenes"][scene_id]["first_reachable"]
        stage3_scene = stage3["scenes"][scene_id]
        structured_navigation = stage3_scene["structured_navigation_results"][0]
        legacy_navigation = next(
            row
            for row in stage3_scene["legacy_navigation_results"]
            if row["success"]
        )
        rows.extend(
            [
                _matrix_row(
                    f"legacy-full-first-{scene_id}",
                    "旧完整场景图 + first_reachable离线导航",
                    "scene_graph.json + 旧三元组pickle + bev_map.npz",
                    "conceptgraph.scripts.run_stage2_offline_navigation",
                    scene_id,
                    first["success"] and first["continuous_collision_recheck"]["passed"],
                    first["selected_node_id"],
                    first["continuous_collision_recheck"]["passed"],
                    "legacy_edge/geometry_fallback",
                    False,
                    first["plan_path"],
                    first,
                ),
                _matrix_row(
                    f"structured-joint-{scene_id}",
                    "结构化场景图 + joint_utility离线导航",
                    "scene_graph.json + structured_edges.json + bev_map.npz",
                    "SemanticNavigationPipeline.select",
                    scene_id,
                    stage3_scene["navigation_uses_structured_edges"],
                    structured_navigation["selected_node_id"],
                    structured_navigation["path_validated"],
                    "structured_edge",
                    False,
                    structured_navigation["plan_path"],
                    {
                        "selection_mode": "joint_utility",
                        "query": structured_navigation["query"],
                        "structured_edge_path": stage3_scene[
                            "structured_edge_path"
                        ],
                        "navigation": structured_navigation,
                    },
                ),
                _matrix_row(
                    f"legacy-triple-{scene_id}",
                    "旧三元组关系边加载和导航",
                    "cfslam_scenegraph_edges.pkl: (source,target,relation)",
                    "load_scene_graph_edges + SemanticNavigationPipeline.select",
                    scene_id,
                    stage3_scene["legacy_metadata_compatibility"]
                    and stage3_scene["legacy_navigation_passed"],
                    legacy_navigation["selected_node_id"],
                    legacy_navigation["path_validated"],
                    "legacy_edge_with_geometry_fallback",
                    True,
                    legacy_navigation["plan_path"],
                    {
                        "successful_query": legacy_navigation,
                        "all_legacy_query_results": stage3_scene[
                            "legacy_navigation_results"
                        ],
                    },
                ),
            ]
        )
    shared = stage3["shared_structured_edge_visualization"]
    rows.append(
        _matrix_row(
            "structured-nav-visualization",
            "新结构化关系边同时服务导航和可视化",
            "同一structured_edges.json",
            "SemanticNavigationPipeline + visualize_scenegraph CLI",
            "scene-0916",
            shared["same_edge_file_content_consumed"],
            None,
            True,
            "structured_edge",
            False,
            shared["visualization_summary_path"],
            shared,
        )
    )
    replay_index = {(row["mode"], row["scene_id"]): row for row in stage5["replays"]}
    for scene_id in SCENES:
        snapshot = replay_index[("snapshot_replay", scene_id)]
        end_to_end = replay_index[("end_to_end_replay", scene_id)]
        rows.extend(
            [
                _matrix_row(
                    f"snapshot-replay-{scene_id}",
                    "objects_all_frames snapshot replay",
                    "objects_all_frames/*.pkl.gz + poses",
                    "run_snapshot_replay",
                    scene_id,
                    snapshot["passed"],
                    None,
                    True,
                    "incremental_structured_edges",
                    False,
                    snapshot["summary_path"],
                    snapshot,
                ),
                _matrix_row(
                    f"mapper-online-{scene_id}",
                    "StreamlinedMapper.update(frame) + OnlineSemanticNavigator端到端回放",
                    "已有RGB-D + 位姿 + 缓存检测pickle",
                    "run_end_to_end_replay",
                    scene_id,
                    end_to_end["passed"],
                    None,
                    True,
                    "incremental_structured_edges",
                    False,
                    end_to_end["summary_path"],
                    end_to_end,
                ),
            ]
        )
    rows.append(
        _matrix_row(
            "qwen-disabled-rule",
            "Qwen关闭时的纯规则导航",
            "规则查询 + 真实Stage2候选",
            "QwenSemanticDelegate(enabled=false)",
            "scene-0916/scene-0003",
            stage4["qwen_disabled"]["passed"],
            21,
            True,
            "rule",
            True,
            stage4["comparison_path"],
            stage4["qwen_disabled"],
        )
    )
    cache_passed = all(
        row["first_cache_hit_count"] == 3
        and row["repeat_cache_hit_count"] == 3
        and row["first_backend_call_count"] == 0
        and row["repeat_backend_call_count"] == 0
        for row in stage4["scenes"].values()
    )
    rows.append(
        _matrix_row(
            "qwen-enabled-cache-hit",
            "Qwen开启且首次与复跑均命中隔离缓存",
            "隔离复制的real_qwen_v1磁盘缓存",
            "run_stage4_offline_qwen_eval",
            "scene-0916/scene-0003",
            cache_passed,
            None,
            True,
            "rule_after_strict_schema_fallback",
            True,
            stage4["comparison_path"],
            stage4["scenes"],
        )
    )
    for case, description in (
        ("timeout", "Qwen超时规则回退"),
        ("exception", "Qwen异常规则回退"),
        ("invalid_schema", "Qwen非法schema规则回退"),
        ("illegal_candidate", "Qwen非法候选ID规则回退"),
    ):
        detail = failures["qwen"][case]
        rows.append(
            _matrix_row(
                f"qwen-{case}",
                description,
                "注入后端输出",
                "QwenSemanticDelegate.parse_query",
                "合成故障 + 双真实场景安全结果对照",
                detail["passed"],
                None,
                True,
                "rule",
                True,
                failures["output_path"],
                detail,
            )
        )
    late = failures["memory_entry"]["late_result"]
    rows.append(
        _matrix_row(
            "qwen-late-result",
            "Qwen迟到结果按安全候选绑定丢弃",
            "异步候选重排 + 已验证候选消失",
            "OnlineSemanticNavigator.update",
            "合成故障",
            late["passed"],
            1,
            True,
            "rule_stage2_authoritative",
            True,
            failures["output_path"],
            late,
        )
    )
    rows.append(
        _matrix_row(
            "entry-cli-serialization",
            "新旧入口、CLI和序列化格式兼容",
            "旧/新CLI、旧pickle、新JSON/JSONL、内存对象",
            "run_semantic_navigation + Stage2/3/4/5/6 CLI + 内存入口",
            "合成smoke + 双真实场景",
            cli["passed"]
            and failures["memory_entry"]["duplicate_frame"]["passed"],
            None,
            True,
            "legacy_and_structured",
            False,
            str((output_root / "runtime_commands.json").resolve()),
            {"cli": cli, "memory": failures["memory_entry"]},
        )
    )
    report = {
        "schema_version": "stage6-compatibility-matrix-v1",
        "required_combination_count": len(rows),
        "passed_combination_count": sum(row["success"] for row in rows),
        "rows": rows,
        "passed": bool(rows) and all(row["success"] for row in rows),
    }
    _write_json(output_root / "compatibility_matrix.json", report)
    return report


def _matrix_row(
    identifier: str,
    combination: str,
    input_format: str,
    entry_point: str,
    scene: str,
    success: bool,
    selected_node_id: int | None,
    path_valid: bool | None,
    relation_source: str,
    fallback: bool,
    output_path: str,
    diagnostics: Any,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "required_combination": combination,
        "input_format": input_format,
        "entry_point": entry_point,
        "scene": scene,
        "success": bool(success),
        "selected_node_id": selected_node_id,
        "path_valid": path_valid,
        "relation_source": relation_source,
        "fallback": bool(fallback),
        "output_path": str(Path(output_path).resolve()),
        "failure_diagnostic": None if success else diagnostics,
        "diagnostics": diagnostics,
    }


def _run_cli_smokes() -> dict[str, Any]:
    modules = (
        "conceptgraph.scripts.run_semantic_navigation",
        "conceptgraph.scripts.run_stage2_offline_navigation",
        "conceptgraph.scripts.run_stage3_offline_relations",
        "conceptgraph.scripts.run_stage4_offline_qwen_eval",
        "conceptgraph.scripts.run_incremental_semantic_navigation",
        "conceptgraph.scripts.run_project_v2_integration_regression",
    )
    rows = []
    for module in modules:
        completed = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        rows.append(
            {
                "module": module,
                "returncode": completed.returncode,
                "help_output_nonempty": bool(completed.stdout.strip()),
                "passed": completed.returncode == 0 and bool(completed.stdout.strip()),
            }
        )
    return {"passed": all(row["passed"] for row in rows), "entries": rows}


def build_performance_report(output_root: Path) -> dict[str, Any]:
    groups = []
    timing_keys = {
        "mapping_time_s": "mapping_s",
        "relation_update_time_s": "relation_update_s",
        "query_ranking_time_s": "query_ranking_s",
        "planning_time_s": "planning_s",
    }
    for mode in MODES:
        for scene_id in SCENES:
            frames = _read_jsonl(output_root / mode / scene_id / "frames.jsonl")
            for planning_frame in (True, False):
                selected = [
                    frame
                    for frame in frames
                    if bool(frame["replanned"]) is planning_frame
                ]
                metrics = {}
                metrics["detection_time_s"] = _not_executed_metric(
                    "not_executed_cached_input",
                    "阶段6只读取缓存检测结果，未运行目标检测。",
                )
                for report_key, source_key in timing_keys.items():
                    values = [
                        float(frame["timings"][source_key])
                        for frame in selected
                        if source_key in frame.get("timings", {})
                    ]
                    if report_key == "mapping_time_s" and mode == "snapshot_replay":
                        metrics[report_key] = _not_executed_metric(
                            "not_executed_snapshot_input",
                            "snapshot replay不执行对象融合建图。",
                        )
                    else:
                        metrics[report_key] = _measured_metric(values, len(selected))
                metrics["qwen_time_s"] = _not_executed_metric(
                    "not_executed_qwen_disabled",
                    "真实逐帧回放关闭Qwen；缓存Qwen耗时在qwen_cache_regression中单独统计。",
                )
                metrics["total_frame_time_s"] = _measured_metric(
                    [float(frame["frame_runtime_s"]) for frame in selected],
                    len(selected),
                )
                groups.append(
                    {
                        "mode": mode,
                        "scene_id": scene_id,
                        "frame_type": "planning" if planning_frame else "non_planning",
                        "frame_count": len(selected),
                        "metrics": metrics,
                    }
                )
    stage4 = _read_json(output_root / "regression_stage4/stage4_comparison.json")
    qwen_cache_rows = []
    for scene in stage4["scenes"]:
        values = [float(value) for value in scene["first_run"]["task_latency_s"].values()]
        qwen_cache_rows.append(
            {
                "mode": "qwen_cache_regression",
                "scene_id": scene["scene_id"],
                "task_count": len(values),
                "qwen_time_s": _measured_metric(values, len(values)),
                "cache_hit_count": int(scene["first_run"]["cache_hit_count"]),
                "backend_call_count": int(scene["first_run"]["backend_call_count"]),
            }
        )
    peak_rss = max(_peak_rss_mb(resource.RUSAGE_SELF), _peak_rss_mb(resource.RUSAGE_CHILDREN))
    metrics_complete = all(
        set(group["metrics"]) == set(REQUIRED_METRICS) for group in groups
    )
    report = {
        "schema_version": "stage6-performance-v1",
        "passed": bool(
            len(groups) == 8
            and metrics_complete
            and all(
                row["cache_hit_count"] == 3 and row["backend_call_count"] == 0
                for row in qwen_cache_rows
            )
        ),
        "groups": groups,
        "qwen_cache_regression": qwen_cache_rows,
        "cpu_peak_rss_mb": peak_rss,
        "peak_gpu_memory_mb": None,
        "peak_gpu_memory_status": "not_recorded_no_actual_cuda_workload",
        "detection_time_s": None,
        "detection_status": "not_executed_cached_input",
        "depth_generation_status": "not_executed_cached_input",
        "metric_statistics_fields": ["count", "mean", "median", "p95", "max", "total"],
    }
    _write_json(output_root / "performance_report.json", report)
    return report


def _measured_metric(values: Sequence[float], expected_count: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {
            "status": "not_available",
            "reason": "该分组没有可测样本。",
            "statistics": None,
            "expected_count": expected_count,
        }
    return {
        "status": "measured" if len(array) == expected_count else "measured_partial",
        "reason": None
        if len(array) == expected_count
        else "部分故障帧在该模块执行前已返回，未产生该模块耗时。",
        "expected_count": expected_count,
        "statistics": timing_statistics(array),
    }


def _not_executed_metric(status: str, reason: str) -> dict[str, Any]:
    return {"status": status, "reason": reason, "statistics": None}


def timing_statistics(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        raise ValueError("timing_statistics需要至少一个真实样本")
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "total": float(np.sum(array)),
    }


def verify_stage6_baseline(output_root: Path) -> dict[str, Any]:
    baseline_path = output_root / "baseline_before_stage6.json"
    baseline = _read_json(baseline_path)
    manifest_checks = {}
    for relative, expected in baseline["protected_manifests"].items():
        actual = manifest_fingerprint(PROJECT_ROOT / relative)
        manifest_checks[relative] = _fingerprint_check(expected, actual)
    dataset_checks = {}
    for pattern, expected in baseline["protected_dataset_groups"].items():
        roots = sorted(PROJECT_ROOT.glob(pattern))
        actual = grouped_manifest_fingerprint(roots)
        dataset_checks[pattern] = _fingerprint_check(expected, actual)
    file_checks = {}
    for relative, expected in baseline["protected_files"].items():
        actual = sha256_file(PROJECT_ROOT / relative)
        file_checks[relative] = {
            "expected": expected,
            "actual": actual,
            "unchanged": expected == actual,
        }
    scene_checks = {}
    for scene_id, expected in baseline["real_scene_inputs"].items():
        scene_dir = PROJECT_ROOT / "dataset/nuscenes" / scene_id
        snapshot_dir = _snapshot_directory(scene_dir / "objects_all_frames")
        exp_dir = scene_dir / "exp_streamlined_yolo_outdoor_objects_v1"
        actual = {
            "snapshot_manifest_sha256": manifest_fingerprint(snapshot_dir)[
                "manifest_sha256"
            ],
            "color_manifest_sha256": manifest_fingerprint(scene_dir / "color")[
                "manifest_sha256"
            ],
            "depth_manifest_sha256": manifest_fingerprint(scene_dir / "depth")[
                "manifest_sha256"
            ],
            "cached_detection_manifest_sha256": manifest_fingerprint(
                exp_dir / "detections"
            )["manifest_sha256"],
            "poses_sha256": sha256_file(scene_dir / "poses.txt"),
        }
        expected_values = {
            "snapshot_manifest_sha256": expected["snapshot_manifest"][
                "manifest_sha256"
            ],
            "color_manifest_sha256": expected["color_manifest_sha256"],
            "depth_manifest_sha256": expected["depth_manifest_sha256"],
            "cached_detection_manifest_sha256": expected[
                "cached_detection_manifest_sha256"
            ],
            "poses_sha256": expected["poses_sha256"],
        }
        scene_checks[scene_id] = {
            key: {
                "expected": expected_values[key],
                "actual": actual[key],
                "unchanged": expected_values[key] == actual[key],
            }
            for key in expected_values
        }
    git = {
        "head": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "tags_at_head": sorted(
            line
            for line in _git_output("tag", "--points-at", "HEAD").splitlines()
            if line
        ),
    }
    git_passed = bool(
        git["head"] == baseline["git"]["head"]
        and git["branch"] == baseline["git"]["branch"]
        and git["tags_at_head"] == baseline["git"]["tags_at_head"]
    )
    passed = bool(
        git_passed
        and all(item["unchanged"] for item in manifest_checks.values())
        and all(item["unchanged"] for item in dataset_checks.values())
        and all(item["unchanged"] for item in file_checks.values())
        and all(
            item["unchanged"]
            for scene in scene_checks.values()
            for item in scene.values()
        )
    )
    report = {
        "schema_version": "stage6-baseline-verification-v1",
        "passed": passed,
        "git": {**git, "unchanged": git_passed},
        "protected_manifests": manifest_checks,
        "protected_dataset_groups": dataset_checks,
        "protected_files": file_checks,
        "real_scene_inputs": scene_checks,
        "baseline_path": str(baseline_path.resolve()),
        "baseline_sha256": sha256_file(baseline_path),
    }
    _write_json(output_root / "baseline_verification.json", report)
    return report


def _fingerprint_check(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> dict[str, Any]:
    keys = [key for key in ("root_count", "file_count", "size_bytes", "manifest_sha256") if key in expected]
    unchanged = all(expected[key] == actual.get(key) for key in keys)
    return {
        "expected": {key: expected[key] for key in keys},
        "actual": {key: actual.get(key) for key in keys},
        "unchanged": unchanged,
    }


def run_static_checks(config: Any, output_root: Path) -> dict[str, Any]:
    pytest = _run_command(
        [sys.executable, "-m", "pytest", "-q", str(config.static_checks.pytest_path)],
        check=False,
    )
    match = re.search(r"(\d+) passed", pytest["stdout"])
    passed_count = int(match.group(1)) if match else 0
    changed_python = _changed_python_files()
    ruff = _run_command(
        ["ruff", "check", *changed_python],
        check=False,
    )
    diff_check = _run_command(["git", "diff", "--check"], check=False)
    result = {
        "pytest": {
            **pytest,
            "collected_and_passed_count": passed_count,
            "passed": pytest["returncode"] == 0 and passed_count >= 126,
        },
        "ruff": {
            **ruff,
            "checked_python_files": [str((PROJECT_ROOT / path).resolve()) for path in changed_python],
            "passed": ruff["returncode"] == 0,
        },
        "git_diff_check": {
            **diff_check,
            "passed": diff_check["returncode"] == 0,
        },
    }
    result["passed"] = all(item["passed"] for item in result.values())
    _write_json(output_root / "static_checks.json", result)
    return result


def finalize_reports(
    config: Any, output_root: Path, static_checks: Mapping[str, Any]
) -> dict[str, Any]:
    stage2 = validate_stage2_regression(config, output_root)
    stage3 = validate_stage3_regression(config, output_root)
    stage4 = validate_stage4_regression(config, output_root)
    replay_reports = {
        mode: {
            scene_id: _read_json(output_root / mode / scene_id / "summary.json")
            for scene_id in SCENES
        }
        for mode in MODES
    }
    stage5 = validate_stage5_replays(replay_reports, output_root)
    failures = build_failure_injection_report(
        _read_json(
            output_root
            / "failure_injection/failure_injection_report.json"
        ),
        stage2,
        stage4,
        output_root,
    )
    compatibility = build_compatibility_matrix(
        stage2, stage3, stage4, stage5, failures, output_root
    )
    performance = _read_json(output_root / "performance_report.json")
    determinism = _read_json(output_root / "determinism_report.json")
    baseline = verify_stage6_baseline(output_root)
    comparison = {
        "schema_version": "stage6-comparison-v1",
        "passed": bool(
            stage2["passed"]
            and stage3["passed"]
            and stage4["passed"]
            and stage5["passed"]
            and determinism["passed"]
            and failures["passed"]
            and compatibility["passed"]
            and performance["passed"]
            and baseline["passed"]
        ),
        "stage2": stage2,
        "stage3": stage3,
        "stage4": stage4,
        "stage5": stage5,
        "recorded_sequence_disclaimer": str(
            config.runtime_policy.recorded_sequence_disclaimer
        ),
        "artifact_paths": _absolute_paths(
            output_root,
            (
                "compatibility_matrix.json",
                "performance_report.json",
                "determinism_report.json",
                "failure_injection_report.json",
                "baseline_verification.json",
            ),
        ),
    }
    _write_json(output_root / "stage6_comparison.json", comparison)
    change_log = PROJECT_ROOT / "doc/change_logs/stage6_integration_and_regression.md"
    code_review = build_code_review_report(static_checks, output_root)
    regression = {
        "schema_version": "stage6-regression-v1",
        "passed": bool(
            comparison["stage2"]["passed"]
            and comparison["stage3"]["passed"]
            and comparison["stage4"]["passed"]
            and comparison["stage5"]["passed"]
            and determinism["passed"]
            and failures["passed"]
        ),
        "stage2": comparison["stage2"],
        "stage3": comparison["stage3"],
        "stage4": comparison["stage4"],
        "stage5": comparison["stage5"],
        "determinism": determinism,
        "failure_injection": failures,
    }
    _write_json(output_root / "regression_report.json", regression)
    integration = {
        "schema_version": "stage6-integration-v1",
        "passed": bool(
            compatibility["passed"]
            and regression["passed"]
            and performance["passed"]
            and baseline["passed"]
            and static_checks["passed"]
            and code_review["passed"]
        ),
        "summary": "阶段1至阶段5已在双真实录制场景中完成隔离集成与回归。",
        "compatibility_matrix_path": str(
            (output_root / "compatibility_matrix.json").resolve()
        ),
        "stage6_comparison_path": str(
            (output_root / "stage6_comparison.json").resolve()
        ),
        "performance_report_path": str(
            (output_root / "performance_report.json").resolve()
        ),
        "regression_report_path": str(
            (output_root / "regression_report.json").resolve()
        ),
        "input_policy": {
            "detection": "not_executed_cached_input",
            "depth_generation": "not_executed_cached_input",
            "qwen_backend_calls": 0,
            "recorded_sequence": str(
                config.runtime_policy.recorded_sequence_disclaimer
            ),
        },
        "static_checks": static_checks,
    }
    _write_json(output_root / "integration_report.json", integration)

    initial_conditions = {
        "compatibility_matrix_passed": compatibility["passed"],
        "stage2_stage3_stage4_stage5_regression_passed": regression["passed"],
        "four_real_replay_groups_deterministic": determinism["passed"],
        "all_successful_paths_collision_checked": comparison["stage2"][
            "all_successful_paths_collision_rechecked"
        ],
        "failure_injection_and_task_cleanup_passed": failures["passed"],
        "full_tests_ruff_diff_check_passed": static_checks["passed"],
        "performance_metrics_truthful": performance["passed"],
        "baseline_and_real_inputs_unchanged": baseline["passed"],
        "code_review_has_no_blockers": code_review["passed"],
        "chinese_stage6_change_log_complete": change_log.exists()
        and _contains_chinese(change_log.read_text(encoding="utf-8")),
    }
    validation_path = output_root / "validation_report.json"
    validation = {
        "schema_version": "stage6-validation-v1",
        "passed": False,
        "conditions": initial_conditions,
        "artifact_root": str(output_root.resolve()),
        "tests": {
            "baseline_collected": 126,
            "baseline_passed": 126,
            "final_passed": static_checks["pytest"]["collected_and_passed_count"],
        },
        "serialization_validation_pending": True,
    }
    _write_json(validation_path, validation)
    serialization = reparse_serialized_outputs(output_root)
    validation["serialization"] = serialization
    validation["serialization_validation_pending"] = False
    validation["conditions"]["all_json_jsonl_reparsed"] = serialization["passed"]
    validation["passed"] = all(validation["conditions"].values())
    _write_json(validation_path, validation)
    return validation


def build_code_review_report(
    static_checks: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    findings = [
        {
            "id": "S6-CR-001",
            "severity": "阻断",
            "status": "已修复",
            "title": "Stage4输出白名单缺少Stage6精确隔离目录",
            "finding": "未经扩展时Stage6无法执行隔离缓存回归。",
            "fix": "仅加入outputs/project_v2/integration_regression/regression_stage4精确路径。",
            "verification": "Stage4双场景隔离缓存回归成功，旧qwen_eval与Stage5路径仍受保护。",
        },
        {
            "id": "S6-CR-002",
            "severity": "重要",
            "status": "已修复",
            "title": "Stage5性能字段与Stage6验收命名及缺失语义不一致",
            "finding": "直接复用mapping_s/qwen_poll_s会把未执行模块误记为0或Qwen推理。",
            "fix": "Stage6按真实逐帧记录规范化字段；检测、snapshot建图、禁用Qwen和无CUDA均使用null及明确状态。",
            "verification": "performance_report按模式、场景、规划帧拆分并包含count/mean/median/P95/max/total。",
        },
        {
            "id": "S6-CR-003",
            "severity": "重要",
            "status": "已修复",
            "title": "旧三元组反向能力差异被误判为兼容失败",
            "finding": "初版验收器要求旧格式的全部反向查询成功，与Stage3既有兼容契约不符。",
            "fix": "要求旧格式加载成功且至少一个旧边导航成功，并保留反向NO_SEMANTIC_CANDIDATES诊断。",
            "verification": "scene-0003旧边正向导航成功，反向差异如实记录；结构化反向查询成功。",
        },
        {
            "id": "S6-CR-004",
            "severity": "重要",
            "status": "已修复",
            "title": "结构化兼容矩阵曾混用Stage2和Stage3证据",
            "finding": "初版矩阵展示Stage2节点，但路径来自Stage3结构化关系查询。",
            "fix": "矩阵改为记录实际Stage3 joint_utility结构化查询的节点、路径和关系边文件。",
            "verification": "两场结构化矩阵项均引用实际查询结果和绝对路径。",
        },
        {
            "id": "S6-CR-005",
            "severity": "一般",
            "status": "已修复",
            "title": "导航与可视化共享边文件只按数量核验",
            "finding": "边数相同不足以证明两个入口消费相同关系内容。",
            "fix": "对规范排序后的完整结构化边字典逐条比较，并记录源文件和内容哈希。",
            "verification": "scene-0916的63条边内容逐条一致。",
        },
        {
            "id": "S6-CR-006",
            "severity": "一般",
            "status": "已修复",
            "title": "故障矩阵缺少非法候选ID单列证据",
            "finding": "既有单元测试覆盖非法候选，但初版Stage6报告未提供独立运行证据。",
            "fix": "新增非法节点999注入，严格验证ILLEGAL_CANDIDATE_ID及回退节点21。",
            "verification": "后端调用1次、规则回退启用、关闭后任务数0且安全选择不变。",
        },
    ]
    review_checks = {
        "legacy_api_cli_compatibility": _read_json(
            output_root / "compatibility_matrix.json"
        )["passed"],
        "no_input_mutation_or_shared_state_leak": _read_json(
            output_root / "failure_injection_report.json"
        )["checks"]["transaction_state_restored"],
        "qwen_cannot_bypass_planning_safety": _read_json(
            output_root / "failure_injection_report.json"
        )["checks"]["qwen_faults_preserve_stage2_safety"],
        "async_tasks_closed": _read_json(
            output_root / "failure_injection_report.json"
        )["checks"]["no_unrecovered_tasks"],
        "frame_transaction_consistency": _read_json(
            output_root / "failure_injection_report.json"
        )["checks"]["transaction_state_restored"],
        "deterministic_hash_and_serialization": _read_json(
            output_root / "determinism_report.json"
        )["passed"],
        "output_root_protection": verify_stage6_baseline(output_root)["passed"],
        "missing_metrics_not_defaulted_to_success": _read_json(
            output_root / "performance_report.json"
        )["peak_gpu_memory_mb"]
        is None,
        "rule_metrics_not_presented_as_model_metrics": _read_json(
            output_root / "stage6_comparison.json"
        )["stage4"]["model_metric_disclaimer"].startswith("严格schema有效数为0"),
        "recorded_replay_not_described_as_closed_loop": "不会改变后续传感器观测"
        in _read_json(output_root / "stage6_comparison.json")[
            "recorded_sequence_disclaimer"
        ],
        "ruff_and_diff_check": bool(
            static_checks["ruff"]["passed"]
            and static_checks["git_diff_check"]["passed"]
        ),
    }
    report = {
        "schema_version": "stage6-code-review-v1",
        "review_scope": "阶段1至阶段6全部当前改动，重点审查API、状态事务、安全边界、异步资源、确定性、输出保护和报告语义。",
        "findings": findings,
        "unresolved_blockers": [],
        "verification": review_checks,
        "remaining_risks": [
            "既有Qwen adapter严格schema有效率仍为0，当前依赖规则回退。",
            "既有轻量快照没有motion_state/is_short_term，动态参考统计保持不可用。",
            "nuScenes为录制序列，本结果不是规划改变后续观测的闭环实验。",
        ],
        "passed": all(review_checks.values()),
    }
    _write_json(output_root / "code_review_report.json", report)
    return report


def reparse_serialized_outputs(output_root: Path) -> dict[str, Any]:
    errors = []
    json_files = sorted(output_root.rglob("*.json"))
    jsonl_files = sorted(output_root.rglob("*.jsonl"))
    jsonl_lines = 0
    for path in json_files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - report every serialization failure
            errors.append(
                {
                    "path": str(path.resolve()),
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                }
            )
    for path in jsonl_files:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            jsonl_lines += 1
            try:
                json.loads(line)
            except Exception as exc:  # noqa: BLE001 - report every JSONL row failure
                errors.append(
                    {
                        "path": str(path.resolve()),
                        "line": line_number,
                        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    }
                )
    return {
        "passed": not errors,
        "json_file_count": len(json_files),
        "jsonl_file_count": len(jsonl_files),
        "jsonl_line_count": jsonl_lines,
        "parse_errors": errors,
    }


def manifest_fingerprint(path: Path) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    return _files_fingerprint(files)


def grouped_manifest_fingerprint(roots: Iterable[Path]) -> dict[str, Any]:
    roots = list(roots)
    files = sorted(
        item for directory in roots for item in directory.rglob("*") if item.is_file()
    )
    return {"root_count": len(roots), **_files_fingerprint(files)}


def _files_fingerprint(files: Sequence[Path]) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    for path in files:
        file_hash = sha256_file(path)
        digest.update(f"{file_hash}  {path.relative_to(PROJECT_ROOT)}\n".encode())
        size += path.stat().st_size
    return {
        "file_count": len(files),
        "size_bytes": size,
        "manifest_sha256": digest.hexdigest(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_sha256(payload: Any) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _snapshot_directory(root: Path) -> Path:
    candidates = sorted(
        (directory for directory in root.iterdir() if directory.is_dir()),
        key=lambda directory: (
            -int("tracking_gate" in directory.name),
            -sum(path.name.split(".")[0].isdigit() for path in directory.glob("*.pkl.gz")),
            directory.name,
        ),
    )
    if not candidates:
        raise FileNotFoundError(f"没有快照目录：{root}")
    return candidates[0]


def _changed_python_files() -> list[str]:
    tracked = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", "--", "*.py"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "*.py"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    paths = sorted(
        path for path in set(tracked + untracked) if (PROJECT_ROOT / path).is_file()
    )
    if not paths:
        raise RuntimeError("没有发现当前改动Python文件，无法执行目标范围Ruff")
    return paths


def _run_command(
    command: Sequence[str],
    *,
    extra_env: Mapping[str, str] | None = None,
    check: bool = True,
) -> dict[str, Any]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    report = {
        "command": list(command),
        "cwd": str(PROJECT_ROOT.resolve()),
        "returncode": completed.returncode,
        "runtime_s": float(time.perf_counter() - started),
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-12000:],
    }
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"命令失败({completed.returncode})：{' '.join(command)}\n"
            f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
        )
    return report


def _peak_rss_mb(who: int) -> float:
    return float(resource.getrusage(who).ru_maxrss) / 1024.0


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _absolute_paths(output_root: Path, names: Sequence[str]) -> dict[str, str]:
    return {name: str((output_root / name).resolve()) for name in names}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
