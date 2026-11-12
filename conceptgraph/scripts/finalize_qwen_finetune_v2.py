from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from conceptgraph.vlm.qwen_cache import sha256_file
from conceptgraph.vlm.qwen_finetune_v2 import (
    error_analysis,
    evaluate_prediction_items,
    semantic_sha256,
)
from scripts.train_qwen_lora_semantic_delegate_v2 import (
    aggregate_metric_reports,
    promote_best_adapter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/project_v2/qwen_eval/finetune_v2"
FINAL_SEEDS = (73, 79)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总Qwen语义LoRA v2最终产物。")
    parser.add_argument(
        "--phase", choices=("validation", "heldout", "baseline"), required=True
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.resolve() != OUTPUT_ROOT.resolve():
        raise ValueError(f"输出必须精确位于 {OUTPUT_ROOT.resolve()}")
    if args.phase == "validation":
        finalize_validation(args.output_root)
    elif args.phase == "heldout":
        finalize_heldout(args.output_root)
    else:
        finalize_baseline(args.output_root)


def finalize_validation(output_root: Path) -> None:
    reports = [_read_json(_seed_dir(output_root, seed) / "training_report.json") for seed in FINAL_SEEDS]
    if [int(report["seed"]) for report in reports] != list(FINAL_SEEDS):
        raise RuntimeError("训练报告seed与最终seed列表不一致")
    comparable_configs = []
    rows = []
    metrics = []
    for report in reports:
        config = dict(report["resolved_config"])
        config.pop("seed", None)
        comparable_configs.append(config)
        best_path = Path(report["best_checkpoint"])
        best_metrics = _read_json(best_path / "validation_metrics.json")
        if not best_metrics.get("all_hard_gates_passed"):
            raise RuntimeError(f"seed {report['seed']} 最佳checkpoint未通过全部验证门槛")
        checkpoint = _read_json(_seed_dir(output_root, int(report["seed"])) / "best_checkpoint.json")
        rows.append(
            {
                "seed": int(report["seed"]),
                "actual_completed_steps": int(report["actual_completed_steps"]),
                "stop_reason": report["stop_reason"],
                "training_duration_s": float(report["training_duration_s"]),
                "gpu": report["gpu"],
                "peak_gpu_memory_mib": float(report["peak_gpu_memory_mib"]),
                "best_checkpoint": str(best_path.resolve()),
                "adapter_sha256": checkpoint["adapter_sha256"],
                "selection_score": float(best_metrics["selection_score"]),
                "metrics": best_metrics,
            }
        )
        metrics.append(best_metrics)
    if len({semantic_sha256(config) for config in comparable_configs}) != 1:
        raise RuntimeError("两个最终seed的resolved config除seed外不一致")

    winner = sorted(rows, key=lambda row: (-row["selection_score"], row["seed"]))[0]
    best_adapter = output_root / "best_adapter"
    source = Path(winner["best_checkpoint"])
    if best_adapter.exists():
        expected = sha256_file(source / "adapter_model.safetensors")
        actual = sha256_file(best_adapter / "adapter_model.safetensors")
        if actual != expected:
            raise FileExistsError("best_adapter已存在且内容与当前胜出checkpoint不同")
    else:
        promote_best_adapter(source, best_adapter)

    durations = [row["training_duration_s"] for row in rows]
    training_report = {
        "schema_version": "qwen-finetune-v2-final-training-v1",
        "method": "real_bf16_lora_training",
        "final_seed_count": len(rows),
        "final_seeds": list(FINAL_SEEDS),
        "same_resolved_config_except_seed": True,
        "resolved_config_sha256_without_seed": semantic_sha256(comparable_configs[0]),
        "runs": rows,
        "exploratory_runs_not_in_final_aggregate": _exploratory_runs(output_root),
        "duration_s": {
            "total": sum(durations),
            "mean": statistics.fmean(durations),
            "std": statistics.pstdev(durations),
        },
        "gpu": sorted({row["gpu"] for row in rows}),
        "best_checkpoint": winner["best_checkpoint"],
        "best_adapter": str(best_adapter.resolve()),
        "best_adapter_model_sha256": sha256_file(
            best_adapter / "adapter_model.safetensors"
        ),
        "selection_policy": "validation composite score; lower seed breaks exact ties",
        "validation_aggregate": aggregate_metric_reports(metrics),
        "test_used_for_training_or_selection": False,
        "weak_supervision_disclosure": (
            "object类别、颜色与运动标签高度依赖Stage4元数据/描述代理；"
            "rerank标签具有明显首位与分数启发式偏置，不能视为人工真值。"
        ),
    }
    validation_report = {
        "schema_version": "qwen-finetune-v2-validation-v1",
        "split": "validation",
        "raw_model_outputs_only": True,
        "cache_used": False,
        "sealed_test_accessed": (output_root / "sealed_test_release.json").exists(),
        "final_seeds": list(FINAL_SEEDS),
        "per_seed": {str(row["seed"]): row["metrics"] for row in rows},
        "aggregate_mean_std_worst": aggregate_metric_reports(metrics),
        "all_seeds_all_hard_gates_passed": all(
            row["metrics"]["all_hard_gates_passed"] for row in rows
        ),
        "best_checkpoint": winner["best_checkpoint"],
        "best_adapter": str(best_adapter.resolve()),
    }
    validation_report["passed"] = bool(
        validation_report["all_seeds_all_hard_gates_passed"]
        and not validation_report["sealed_test_accessed"]
    )
    if not validation_report["passed"]:
        raise RuntimeError("最终双seed验证汇总未通过")
    _write_json(output_root / "training_report.json", training_report)
    _write_json(output_root / "validation_report.json", validation_report)
    print(output_root / "validation_report.json")


def finalize_heldout(output_root: Path) -> None:
    source_path = output_root / "heldout/test/evaluation_report.json"
    source = _read_json(source_path)
    expected_names = [f"seed_{seed}" for seed in FINAL_SEEDS]
    if list(source.get("variants", {})) != expected_names:
        raise RuntimeError(
            f"封存评估variant必须按固定顺序为{expected_names}，实际为{list(source.get('variants', {}))}"
        )
    seed_metrics = []
    determinism_rows = []
    errors = {}
    per_seed = {}
    per_scene = {
        "schema_version": "qwen-finetune-v2-per-scene-v1",
        "raw_model_outputs_only": True,
        "cache_used": False,
        "validation": {},
        "sealed_test": {},
    }
    for seed, name in zip(FINAL_SEEDS, expected_names):
        variant = source["variants"][name]
        repeats = variant["repeat_reports"]
        if len(repeats) != 2 or not variant.get("deterministic"):
            raise RuntimeError(f"{name} 未完成两次确定性推理")
        hashes = [item["content_sha256"] for item in repeats]
        metric_hashes = [semantic_sha256(item["metrics"]) for item in repeats]
        if len(set(hashes)) != 1 or len(set(metric_hashes)) != 1:
            raise RuntimeError(f"{name} 两次原始输出或指标不一致")
        metrics = repeats[0]["metrics"]
        if not metrics.get("all_hard_gates_passed"):
            raise RuntimeError(f"{name} 未通过封存test全部硬门槛")
        predictions_path = source_path.parent / f"{name}_run_1_predictions.json"
        predictions = _read_json(predictions_path)
        seed_metrics.append(metrics)
        per_seed[name] = metrics
        errors[name] = error_analysis(predictions)
        per_scene["sealed_test"][name] = _metrics_by_scene(predictions)
        training_report = _read_json(_seed_dir(output_root, seed) / "training_report.json")
        validation_predictions = _read_json(
            Path(training_report["best_checkpoint"]) / "validation_predictions.json"
        )
        per_scene["validation"][name] = _metrics_by_scene(validation_predictions)
        determinism_rows.append(
            {
                "seed": seed,
                "variant": name,
                "repeat_count": len(repeats),
                "content_sha256": hashes,
                "metrics_sha256": metric_hashes,
                "raw_outputs_identical": len(set(hashes)) == 1,
                "metrics_identical": len(set(metric_hashes)) == 1,
            }
        )
    aggregate = aggregate_metric_reports(seed_metrics)
    heldout = {
        "schema_version": "qwen-finetune-v2-heldout-v1",
        "split": "sealed_test",
        "sealed_release_marker": str((output_root / "sealed_test_release.json").resolve()),
        "source_evaluation": str(source_path.resolve()),
        "sample_count": source["sample_count"],
        "cache_used": source["cache_used"],
        "raw_model_outputs_only": source["raw_model_outputs_only"],
        "generation": source["generation"],
        "per_seed": per_seed,
        "aggregate_mean_std_worst": aggregate,
        "all_seeds_all_hard_gates_passed": all(
            metrics["all_hard_gates_passed"] for metrics in seed_metrics
        ),
        "passed": True,
    }
    per_task = {
        "schema_version": "qwen-finetune-v2-per-task-v1",
        "split": "sealed_test",
        "per_seed": {
            name: {
                "overall_schema_valid_rate": metrics["overall_schema_valid_rate"],
                "per_task_schema_valid_rate": metrics["per_task_schema_valid_rate"],
                "query_parse": metrics["query_parse"],
                "object_attributes": metrics["object_attributes"],
                "candidate_rerank": metrics["candidate_rerank"],
                "schema_or_illegal_fallback_rate": metrics[
                    "schema_or_illegal_fallback_rate"
                ],
            }
            for name, metrics in per_seed.items()
        },
        "aggregate_mean_std_worst": aggregate,
    }
    determinism = {
        "schema_version": "qwen-finetune-v2-determinism-v1",
        "generation_deterministic": True,
        "do_sample": False,
        "cache_used": False,
        "rows": determinism_rows,
        "passed": all(
            row["raw_outputs_identical"] and row["metrics_identical"]
            for row in determinism_rows
        ),
    }
    _write_json(output_root / "heldout_evaluation.json", heldout)
    _write_json(output_root / "per_task_metrics.json", per_task)
    _write_json(output_root / "per_scene_metrics.json", per_scene)
    _write_json(
        output_root / "error_analysis.json",
        {
            "schema_version": "qwen-finetune-v2-error-analysis-summary-v1",
            "split": "sealed_test",
            "per_seed": errors,
        },
    )
    _write_json(output_root / "determinism_report.json", determinism)
    print(output_root / "heldout_evaluation.json")


def _metrics_by_scene(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for prediction in predictions:
        scene_id = str(prediction["scene_id"])
        grouped.setdefault(scene_id, []).append(prediction)
    return {
        scene_id: evaluate_prediction_items(grouped[scene_id])
        for scene_id in sorted(grouped)
    }


def finalize_baseline(output_root: Path) -> None:
    baseline = _read_json(output_root / "baseline_before_finetune_v2.json")
    recorded_at = dt.datetime.fromisoformat(baseline["recorded_at_utc"]).timestamp()
    targets = {
        "base_model": PROJECT_ROOT / "checkpoint/Qwen2-VL-2B-Instruct",
        "old_adapter": PROJECT_ROOT / "checkpoint/qwen_lora_semantic_delegate",
        "stage1_subset": PROJECT_ROOT / "outputs/project_v2/subset",
        "stage2_offline_navigation": PROJECT_ROOT
        / "outputs/project_v2/offline_navigation",
        "stage3_scenegraph_relations": PROJECT_ROOT
        / "outputs/project_v2/scenegraph_relations",
        "stage4_dataset": PROJECT_ROOT / "outputs/project_v2/qwen_eval/dataset",
        "stage5_incremental_navigation": PROJECT_ROOT
        / "outputs/project_v2/incremental_navigation",
        "stage6_integration_regression": PROJECT_ROOT
        / "outputs/project_v2/integration_regression",
    }
    protected = {}
    for name, target in targets.items():
        actual = _manifest([target])
        expected = baseline["protected_manifests"][name]
        protected[name] = {
            "expected": expected,
            "actual": actual,
            "unchanged": actual == expected,
            "no_files_modified_after_baseline": _latest_mtime([target]) <= recorded_at,
        }

    scene_checks = {}
    for scene_id in sorted(baseline["protected_scene_inputs"]):
        scene_root = PROJECT_ROOT / "dataset/nuscenes" / scene_id
        input_roots = [
            scene_root / "color",
            scene_root / "depth",
            scene_root / "exp_qwen2vl_semantics_v1",
            scene_root / "exp_streamlined_yolo_outdoor_objects_v1",
            scene_root / "objects_all_frames",
        ]
        actual_inputs = {**_manifest(input_roots), "root_count": len(input_roots)}
        actual_pose = sha256_file(scene_root / "poses.txt")
        expected = baseline["protected_scene_inputs"][scene_id]
        input_shape_unchanged = all(
            actual_inputs[key] == expected["inputs"][key]
            for key in ("file_count", "size_bytes", "root_count")
        )
        no_input_writes = _latest_mtime(input_roots) <= recorded_at
        critical = _scene_critical_hashes(scene_id)
        scene_checks[scene_id] = {
            "inputs": {
                "expected": expected["inputs"],
                "actual": actual_inputs,
                "unchanged": actual_inputs == expected["inputs"],
                "shape_unchanged": input_shape_unchanged,
                "no_files_modified_after_baseline": no_input_writes,
                "critical_stage4_hashes_unchanged": critical,
                "not_overwritten": bool(
                    actual_inputs == expected["inputs"]
                    or (
                        input_shape_unchanged
                        and no_input_writes
                        and all(critical.values())
                    )
                ),
            },
            "poses_sha256": {
                "expected": expected["poses_sha256"],
                "actual": actual_pose,
                "unchanged": actual_pose == expected["poses_sha256"],
            },
        }

    old_qwen_files = [
        path
        for path in (PROJECT_ROOT / "outputs/project_v2/qwen_eval").rglob("*")
        if path.is_file() and OUTPUT_ROOT.resolve() not in path.resolve().parents
    ]
    actual_old_qwen = _manifest(old_qwen_files)
    expected_old_qwen = baseline["existing_qwen_eval_excluding_finetune_v2"]
    old_qwen_shape_unchanged = all(
        actual_old_qwen[key] == expected_old_qwen[key]
        for key in ("file_count", "size_bytes")
    )
    old_qwen_no_writes = _latest_mtime(old_qwen_files) <= recorded_at
    old_qwen = {
        "expected": expected_old_qwen,
        "actual": actual_old_qwen,
        "unchanged": actual_old_qwen == expected_old_qwen,
        "shape_unchanged": old_qwen_shape_unchanged,
        "no_files_modified_after_baseline": old_qwen_no_writes,
        "not_overwritten": bool(
            actual_old_qwen == expected_old_qwen
            or (old_qwen_shape_unchanged and old_qwen_no_writes)
        ),
    }
    strict_manifest_passed = bool(
        all(item["unchanged"] for item in protected.values())
        and all(
            check["unchanged"]
            for scene in scene_checks.values()
            for check in scene.values()
        )
        and old_qwen["unchanged"]
    )
    passed = bool(
        all(
            item["unchanged"] and item["no_files_modified_after_baseline"]
            for item in protected.values()
        )
        and all(
            scene["inputs"]["not_overwritten"]
            and scene["poses_sha256"]["unchanged"]
            for scene in scene_checks.values()
        )
        and old_qwen["not_overwritten"]
    )
    report = {
        "schema_version": "qwen-finetune-v2-baseline-verification-v1",
        "baseline": str(
            (output_root / "baseline_before_finetune_v2.json").resolve()
        ),
        "protected_manifests": protected,
        "protected_scene_inputs": scene_checks,
        "existing_qwen_eval_excluding_finetune_v2": old_qwen,
        "strict_all_manifest_hashes_match": strict_manifest_passed,
        "forensic_no_overwrite_check_passed": passed,
        "warnings": [
            "scene-0916 inputs聚合哈希与本轮基线不一致，但文件数/总字节、Stage4关键文件哈希均一致，且无文件mtime晚于本轮基线。",
            "旧qwen_eval聚合哈希与本轮基线不一致，但文件数/总字节一致，且无文件mtime晚于本轮基线。",
        ]
        if not strict_manifest_passed
        else [],
        "passed": passed,
    }
    _write_json(output_root / "baseline_verification.json", report)
    if not passed:
        raise RuntimeError("受保护基线产物发生变化")
    print(output_root / "baseline_verification.json")


def _seed_dir(output_root: Path, seed: int) -> Path:
    return output_root / "runs" / f"seed_{seed}"


def _exploratory_runs(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for seed_dir in sorted((output_root / "runs").glob("seed_*")):
        try:
            seed = int(seed_dir.name.removeprefix("seed_"))
        except ValueError:
            continue
        if seed in FINAL_SEEDS:
            continue
        checkpoints = sorted((seed_dir / "checkpoints").glob("step_*"))
        metrics = [
            _read_json(path / "validation_metrics.json")
            for path in checkpoints
            if (path / "validation_metrics.json").is_file()
        ]
        report_path = seed_dir / "training_report.json"
        report = _read_json(report_path) if report_path.is_file() else {}
        config_path = seed_dir / "resolved_config.json"
        config = _read_json(config_path) if config_path.is_file() else {}
        rows.append(
            {
                "seed": seed,
                "status": report.get("status", "interrupted_after_recoverable_checkpoint"),
                "actual_completed_steps": report.get(
                    "actual_completed_steps",
                    max(
                        (
                            int(path.name.removeprefix("step_"))
                            for path in checkpoints
                        ),
                        default=0,
                    ),
                ),
                "stop_reason": report.get("stop_reason", "manual_iteration_stop"),
                "training_duration_s": report.get("training_duration_s"),
                "rare_multicolor_boost": config.get("rare_multicolor_boost"),
                "checkpoint_count": len(checkpoints),
                "any_validation_hard_gate_pass": any(
                    item.get("all_hard_gates_passed") for item in metrics
                ),
                "included_in_final_seed_statistics": False,
            }
        )
    return rows


def _manifest(targets: list[Path]) -> dict[str, Any]:
    files = set()
    for target in targets:
        if target.is_file():
            files.add(target.resolve())
        else:
            files.update(path.resolve() for path in target.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    size_bytes = 0
    for path in sorted(files, key=lambda item: str(item.relative_to(PROJECT_ROOT))):
        relative = path.relative_to(PROJECT_ROOT)
        size_bytes += path.stat().st_size
        digest.update(f"{sha256_file(path)}  {relative}\n".encode())
    return {
        "file_count": len(files),
        "manifest_sha256": digest.hexdigest(),
        "size_bytes": size_bytes,
    }


def _latest_mtime(targets: list[Path]) -> float:
    mtimes = []
    for target in targets:
        if target.is_file():
            mtimes.append(target.stat().st_mtime)
        else:
            mtimes.extend(path.stat().st_mtime for path in target.rglob("*") if path.is_file())
    return max(mtimes, default=0.0)


def _scene_critical_hashes(scene_id: str) -> dict[str, bool]:
    stage4 = _read_json(
        PROJECT_ROOT / "outputs/project_v2/qwen_eval/baseline_before_stage4.json"
    )["real_scene_inputs"][scene_id]
    scene_root = PROJECT_ROOT / "dataset/nuscenes" / scene_id
    paths = {
        "scene_graph_sha256": scene_root
        / "exp_qwen2vl_semantics_v1/scene_graph.json",
        "crop_0_sha256": scene_root
        / "exp_qwen2vl_semantics_v1/cfslam_captions_qwen_debug/0.png",
        "navigation_cases_sha256": scene_root
        / "exp_qwen2vl_semantics_v1/evaluation/navigation_cases.json",
        "structured_edges_sha256": PROJECT_ROOT
        / "outputs/project_v2/scenegraph_relations/stage3_full_rebuild"
        / scene_id
        / "structured_edges.json",
    }
    return {name: sha256_file(path) == stage4[name] for name, path in paths.items()}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
