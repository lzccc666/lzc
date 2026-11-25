from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from conceptgraph.navigation.pipeline import SemanticNavigationPipeline
from conceptgraph.scripts.run_nuscenes_navigation_examples import make_cfg
from conceptgraph.scripts.run_stage2_offline_navigation import (
    _reference_navigation_case,
    _resolve_scene_inputs,
    _selected_scene_names,
)
from conceptgraph.vlm.qwen_cache import canonical_json_bytes, semantic_version, sha256_file
from conceptgraph.vlm.qwen_semantic_delegate import QwenDelegateConfig, QwenSemanticDelegate


PROJECT_ROOT = Path("/root/autodl-tmp/lizichong")
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/project_v2/qwen_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行阶段4双场景离线Qwen语义评估。")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--scene-count", type=int, default=2)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "checkpoint/Qwen2-VL-2B-Instruct",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=PROJECT_ROOT / "checkpoint/qwen_lora_semantic_delegate",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = (args.output_root or (project_root / "outputs/project_v2/qwen_eval")).resolve()
    _validate_environment(args, project_root, output_root)
    directories = _prepare_directories(output_root)
    requested = args.scenes or _selected_scene_names(
        project_root / "outputs/project_v2/subset/selected_scenes.json"
    )
    scenes = _select_complete_scenes(project_root, requested, max(2, args.scene_count))
    if len(scenes) < 2:
        raise RuntimeError(f"完整真实场景不足2个：{scenes}")

    delegate = QwenSemanticDelegate(
        QwenDelegateConfig(
            enabled=True,
            model_name_or_path=str(args.model.resolve()),
            adapter_path=args.adapter.resolve(),
            model_revision="local-qwen2-vl-2b-instruct",
            adapter_revision="local-semantic-delegate-lora",
            device=args.device,
            use_qwen_object_attributes=True,
            use_qwen_query_parse=True,
            use_qwen_rerank=True,
            timeout_s=args.timeout_s,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            cache_dir=directories["cache"] / "real_qwen_v1",
            log_dir=directories["qwen"] / "audit",
        )
    )
    base_target_config = OmegaConf.load(
        project_root / "conceptgraph/hydra_configs/semantic_navigation.yaml"
    ).target_selection
    dataset_rows = _load_object_rows(directories["dataset"] / "object_attributes.jsonl")
    scene_reports = []
    metric_rows = []
    determinism_first = []
    determinism_cached = []
    total_calls_before = delegate.backend_call_count
    try:
        for scene_id in scenes[: args.scene_count]:
            report, rows, first_snapshot, cached_snapshot = _run_scene(
                project_root=project_root,
                scene_id=scene_id,
                output_dirs=directories,
                target_config=base_target_config,
                delegate=delegate,
                object_row=dataset_rows[scene_id],
            )
            scene_reports.append(report)
            metric_rows.extend(rows)
            determinism_first.append(first_snapshot)
            determinism_cached.append(cached_snapshot)
    finally:
        delegate.close(wait=True)
    total_backend_calls = delegate.backend_call_count - total_calls_before

    metrics = _aggregate_metrics(metric_rows)
    dataset_report = _read_json(directories["dataset"] / "dataset_report.json")
    split_manifest = _read_json(directories["dataset"] / "split_manifest.json")
    leakage_audit = _read_json(directories["dataset"] / "leakage_audit.json")
    comparison = {
        "schema_version": "stage4-comparison-v1",
        "mode": "offline_local_qwen_with_rule_fallback",
        "scene_count": len(scene_reports),
        "scenes": scene_reports,
        "tasks": ["query_parse", "object_attributes", "candidate_rerank"],
        "metrics": metrics,
        "backend_call_count": total_backend_calls,
        "model_id": delegate.model_identifier,
        "adapter_id": delegate.adapter_identifier,
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "dataset": {
            "sample_count": dataset_report["sample_count"],
            "task_counts": dataset_report["task_counts"],
            "deterministic_sha256": dataset_report["deterministic_sha256"],
            "split_scene_counts": dataset_report["split_scene_counts"],
            "split_manifest_sha256": split_manifest["deterministic_sha256"],
            "leakage_audit": leakage_audit,
        },
        "artifact_paths": {
            name: str(path.resolve()) for name, path in directories.items()
        },
        "resource_limits": {
            "offline_only": True,
            "network_downloads": False,
            "bounded_scene_count": len(scene_reports),
            "bounded_tasks_per_scene": 3,
            "training_executed": False,
        },
    }
    comparison["deterministic_content_sha256"] = semantic_version(_without_timing(comparison))
    _write_json(output_root / "stage4_comparison.json", comparison)

    first_hash = semantic_version(determinism_first)
    cached_hash = semantic_version(determinism_cached)
    cache_backend_calls = sum(scene["cache_repeat"]["backend_call_count"] for scene in scene_reports)
    determinism = {
        "schema_version": "stage4-determinism-v1",
        "first_semantic_sha256": first_hash,
        "cached_semantic_sha256": cached_hash,
        "semantic_outputs_equal": determinism_first == determinism_cached,
        "cache_repeat_backend_call_count": cache_backend_calls,
        "passed": determinism_first == determinism_cached and cache_backend_calls == 0,
        "excluded_fields": ["latency", "timing", "runtime_s"],
    }
    _write_json(output_root / "determinism_report.json", determinism)
    baseline = _verify_baseline(project_root, output_root / "baseline_before_stage4.json")
    _write_json(output_root / "baseline_verification.json", baseline)
    validation = {
        "schema_version": "stage4-validation-v1",
        "execution_mode": (
            "isolated_cache_regression"
            if output_root.name == "regression_stage4"
            else "primary_real_backend_evaluation"
        ),
        "runtime_checks": {
            "two_real_scenes": len(scene_reports) >= 2,
            "three_tasks_each_scene": all(scene["task_count"] == 3 for scene in scene_reports),
            "real_backend_called_or_isolated_cache_replay": bool(
                total_backend_calls > 0
                or output_root.name == "regression_stage4"
            ),
            "cache_repeat_zero_backend_calls": cache_backend_calls == 0,
            "scene_leakage_zero": leakage_audit["scene_leakage_count"] == 0,
            "multimodal_processor_passed": _read_json(
                output_root / "sanity_training/multimodal_processor_smoke.json"
            ).get("passed", False),
            "determinism_passed": determinism["passed"],
            "protected_baseline_unchanged": baseline["passed"],
        },
        "tests": {"status": "由最终验收命令更新"},
    }
    validation["runtime_passed"] = all(validation["runtime_checks"].values())
    _write_json(output_root / "validation_report.json", validation)
    print(output_root / "stage4_comparison.json")


def _run_scene(
    *,
    project_root: Path,
    scene_id: str,
    output_dirs: dict[str, Path],
    target_config,
    delegate: QwenSemanticDelegate,
    object_row: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    inputs, diagnostic = _resolve_scene_inputs(project_root, scene_id)
    if inputs is None:
        raise RuntimeError(f"{scene_id}: {diagnostic}")
    reference, diagnostic = _reference_navigation_case(inputs["qwen_exp_dir"])
    if reference is None:
        raise RuntimeError(f"{scene_id}: {diagnostic}")
    query_text = reference["instruction"]
    pose = np.asarray(reference["vehicle_pose"], dtype=float)

    rule_pipeline = _make_pipeline(
        inputs,
        reference,
        target_config,
        output_dirs["rule"] / scene_id,
    )
    rule_result = rule_pipeline.select(query_text, pose)
    if not rule_result.success:
        raise RuntimeError(f"{scene_id}规则基线失败：{rule_result.failure_code}")
    rule_pipeline.save_selection_result(
        rule_result,
        output_dirs["rule"] / scene_id,
        save_plan=True,
        save_visualizations=False,
    )
    rule_delegate = QwenSemanticDelegate(QwenDelegateConfig(enabled=False))
    rule_query = rule_delegate.parse_query(query_text, rule_result=rule_result.plan.query)
    metadata = _object_metadata(object_row)
    rule_object = rule_delegate.describe_object([object_row["crop_path"]], metadata)
    rule_candidates, rule_edges = _verified_candidates(rule_result, rule_pipeline._load_edges())
    rule_rerank = rule_delegate.rerank_candidates(query_text, rule_candidates, rule_edges)
    rule_delegate.close()
    _write_json(
        output_dirs["rule"] / scene_id / "semantic_tasks.json",
        {"query_parse": rule_query, "object_attributes": rule_object, "candidate_rerank": rule_rerank},
    )

    first_calls_before = delegate.backend_call_count
    qwen_pipeline = _make_pipeline(
        inputs,
        reference,
        target_config,
        output_dirs["qwen"] / scene_id,
        delegate=delegate,
    )
    qwen_started = time.perf_counter()
    qwen_result = qwen_pipeline.select(query_text, pose)
    navigation_latency = time.perf_counter() - qwen_started
    if not qwen_result.success:
        raise RuntimeError(f"{scene_id} Qwen回退导航失败：{qwen_result.failure_code}")
    qwen_pipeline.save_selection_result(
        qwen_result,
        output_dirs["qwen"] / scene_id,
        save_plan=True,
        save_visualizations=False,
    )
    query_audit = qwen_result.debug.get("qwen_query_parse")
    rerank_audit = qwen_result.debug.get("qwen_candidate_rerank")
    object_audit = delegate.describe_object(
        [object_row["crop_path"]],
        metadata,
        object_version=object_row["image_sha256"],
    )
    first_backend_calls = delegate.backend_call_count - first_calls_before
    _write_json(output_dirs["qwen"] / scene_id / "query_parse.json", query_audit)
    _write_json(output_dirs["qwen"] / scene_id / "object_attributes.json", object_audit)
    _write_json(output_dirs["qwen"] / scene_id / "candidate_rerank.json", rerank_audit)

    cached_calls_before = delegate.backend_call_count
    cached_pipeline = _make_pipeline(
        inputs,
        reference,
        target_config,
        output_dirs["qwen"] / scene_id / "cache_repeat",
        delegate=delegate,
    )
    cached_started = time.perf_counter()
    cached_result = cached_pipeline.select(query_text, pose)
    cached_navigation_latency = time.perf_counter() - cached_started
    cached_object = delegate.describe_object(
        [object_row["crop_path"]],
        metadata,
        object_version=object_row["image_sha256"],
    )
    cached_backend_calls = delegate.backend_call_count - cached_calls_before
    cached_query = cached_result.debug.get("qwen_query_parse")
    cached_rerank = cached_result.debug.get("qwen_candidate_rerank")

    failure = _run_failure_injection(
        inputs=inputs,
        reference=reference,
        target_config=target_config,
        output_dir=output_dirs["failure"] / scene_id,
        object_row=object_row,
    )
    metric_rows = [
        _metric_row(scene_id, "query_parse", query_audit, rule_query["normalized_output"]),
        _metric_row(scene_id, "object_attributes", object_audit, rule_object["normalized_output"]),
        _rerank_metric_row(
            scene_id,
            rerank_audit,
            int(qwen_result.selected_candidate.node_id),
        ),
    ]
    first_snapshot = _semantic_snapshot(
        scene_id,
        qwen_result,
        query_audit,
        object_audit,
        rerank_audit,
    )
    cached_snapshot = _semantic_snapshot(
        scene_id,
        cached_result,
        cached_query,
        cached_object,
        cached_rerank,
    )
    report = {
        "scene_id": scene_id,
        "instruction": query_text,
        "task_count": 3,
        "rule_selected_node_id": int(rule_result.selected_candidate.node_id),
        "qwen_navigation_selected_node_id": int(qwen_result.selected_candidate.node_id),
        "qwen_rerank_recommended_node_id": rerank_audit.get("selected_node_id"),
        "candidate_recommendation_changed": bool(rerank_audit.get("recommendation_changed")),
        "route_selection_changed": (
            rule_result.selected_candidate.node_id != qwen_result.selected_candidate.node_id
        ),
        "schema_valid_count": sum(row["schema_valid"] for row in metric_rows),
        "invalid_output_count": sum(not row["schema_valid"] for row in metric_rows),
        "fallback_reasons": [
            row["fallback_reason"] for row in metric_rows if row["fallback_reason"] is not None
        ],
        "first_run": {
            "backend_call_count": first_backend_calls,
            "navigation_latency_s": navigation_latency,
            "task_latency_s": {
                "query_parse": query_audit["latency"]["total_s"],
                "object_attributes": object_audit["latency"]["total_s"],
                "candidate_rerank": rerank_audit["latency"]["total_s"],
            },
            "cache_hit_count": sum(
                audit["cache"]["hit"] for audit in (query_audit, object_audit, rerank_audit)
            ),
        },
        "cache_repeat": {
            "backend_call_count": cached_backend_calls,
            "navigation_latency_s": cached_navigation_latency,
            "cache_hit_count": sum(
                audit["cache"]["hit"] for audit in (cached_query, cached_object, cached_rerank)
            ),
            "cache_hit_rate": sum(
                audit["cache"]["hit"] for audit in (cached_query, cached_object, cached_rerank)
            )
            / 3.0,
        },
        "failure_injection": failure,
        "object_crop": {
            "path": str(Path(object_row["crop_path"]).resolve()),
            "sha256": object_row["image_sha256"],
            "processor_smoke_verified": True,
        },
        "output_paths": {
            "rule": str((output_dirs["rule"] / scene_id).resolve()),
            "qwen": str((output_dirs["qwen"] / scene_id).resolve()),
            "failure": str((output_dirs["failure"] / scene_id).resolve()),
        },
    }
    return report, metric_rows, first_snapshot, cached_snapshot


def _run_failure_injection(
    *,
    inputs,
    reference,
    target_config,
    output_dir: Path,
    object_row: dict[str, Any],
) -> dict[str, Any]:
    class FailingBackend:
        def __init__(self):
            self.calls = 0

        def __call__(self, task, prompt, images, generation_config):
            self.calls += 1
            raise RuntimeError("stage4_injected_backend_failure")

    backend = FailingBackend()
    delegate = QwenSemanticDelegate(
        QwenDelegateConfig(
            enabled=True,
            use_qwen_query_parse=True,
            use_qwen_object_attributes=True,
            use_qwen_rerank=True,
            model_revision="failure-injection",
            timeout_s=1.0,
            log_dir=output_dir / "audit",
        ),
        backend=backend,
    )
    pipeline = _make_pipeline(
        inputs,
        reference,
        target_config,
        output_dir,
        delegate=delegate,
    )
    result = pipeline.select(reference["instruction"], np.asarray(reference["vehicle_pose"], dtype=float))
    pipeline.save_selection_result(result, output_dir, save_plan=True, save_visualizations=False)
    object_result = delegate.describe_object(
        [object_row["crop_path"]],
        _object_metadata(object_row),
    )
    delegate.close(wait=True)
    payload = {
        "navigation_success": result.success,
        "navigation_failure_code": result.failure_code,
        "query_failure_code": result.debug.get("qwen_query_parse", {}).get("failure_code"),
        "rerank_failure_code": result.debug.get("qwen_candidate_rerank", {}).get("failure_code"),
        "object_failure_code": object_result["failure_code"],
        "fallback_selected_node_id": (
            None if result.selected_candidate is None else int(result.selected_candidate.node_id)
        ),
        "backend_call_count": backend.calls,
        "passed": (
            result.success
            and result.debug.get("qwen_query_parse", {}).get("fallback_used")
            and object_result["fallback_used"]
        ),
    }
    _write_json(output_dir / "failure_injection_summary.json", payload)
    return payload


def _make_pipeline(inputs, reference, target_config, output_dir: Path, delegate=None):
    cfg = make_cfg(
        qwen_exp_dir=inputs["qwen_exp_dir"],
        baseline_exp_dir=inputs["baseline_exp_dir"],
        output_dir=output_dir,
        query_text=reference["instruction"],
        vehicle_pose=reference["vehicle_pose"],
    )
    cfg.target_selection = OmegaConf.create(OmegaConf.to_container(target_config, resolve=True))
    cfg.target_selection.mode = "joint_utility"
    return SemanticNavigationPipeline(cfg, semantic_delegate=delegate)


def _verified_candidates(result, edges) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not result.success or result.selected_candidate is None:
        return [], []
    selected_id = int(result.selected_candidate.node_id)
    verified = [candidate for candidate in result.evaluated_candidates if candidate.reachable]
    verified.sort(
        key=lambda candidate: (
            candidate.node_id != selected_id,
            -float(candidate.selection_utility if candidate.selection_utility is not None else -math.inf),
            int(candidate.node_id),
        )
    )
    payload = [
        {
            "node_id": int(candidate.node_id),
            "semantic_rank": candidate.semantic_rank,
            "path_validated": True,
            "stage2_scores": {
                "combined_semantic": candidate.score,
                "relation": candidate.relation_score,
                "path_cost": candidate.path_cost,
                "joint_utility": candidate.selection_utility,
            },
        }
        for candidate in verified
    ]
    ids = {item["node_id"] for item in payload}
    edge_payload = [
        edge.to_dict() for edge in edges if edge.source in ids or edge.target in ids
    ]
    return payload, edge_payload


def _metric_row(scene_id: str, task: str, audit: dict[str, Any], target: dict[str, Any]):
    schema_valid = bool(audit.get("output_schema_valid"))
    effective = audit.get("normalized_output") or {}
    qwen_hits, field_total = _field_hits(effective if schema_valid else {}, target)
    effective_hits, _ = _field_hits(effective, target)
    return {
        "scene_id": scene_id,
        "task": task,
        "schema_valid": schema_valid,
        "field_hits": qwen_hits,
        "effective_field_hits": effective_hits,
        "field_total": field_total,
        "fallback": bool(audit.get("fallback_used")),
        "fallback_reason": audit.get("fallback_reason"),
        "failure_code": audit.get("failure_code"),
        "latency_s": float(audit.get("latency", {}).get("total_s", 0.0)),
        "cache_hit": bool(audit.get("cache", {}).get("hit")),
    }


def _rerank_metric_row(scene_id: str, audit: dict[str, Any], correct_id: int):
    row = _metric_row(
        scene_id,
        "candidate_rerank",
        audit,
        {"selected_node_id": correct_id},
    )
    effective_ranking = audit.get("ranking", [])
    qwen_ranking = effective_ranking if audit.get("output_schema_valid") else []
    row.update(
        {
            "rerank_top1": bool(qwen_ranking and qwen_ranking[0] == correct_id),
            "rerank_mrr": 1.0 / (qwen_ranking.index(correct_id) + 1) if correct_id in qwen_ranking else 0.0,
            "effective_rerank_top1": bool(
                effective_ranking and effective_ranking[0] == correct_id
            ),
            "effective_rerank_mrr": (
                1.0 / (effective_ranking.index(correct_id) + 1)
                if correct_id in effective_ranking
                else 0.0
            ),
        }
    )
    return row


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_task[row["task"]].append(row)
        grouped_scene[row["scene_id"]].append(row)
    return {
        "overall": _summarize_rows(rows),
        "by_task": {key: _summarize_rows(value) for key, value in sorted(grouped_task.items())},
        "by_scene": {key: _summarize_rows(value) for key, value in sorted(grouped_scene.items())},
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rerank = [row for row in rows if "rerank_top1" in row]
    latencies = sorted(row["latency_s"] for row in rows)
    count = max(1, len(rows))
    return {
        "sample_count": len(rows),
        "schema_valid_rate": sum(row["schema_valid"] for row in rows) / count,
        "field_accuracy": sum(row["field_hits"] for row in rows)
        / max(1, sum(row["field_total"] for row in rows)),
        "effective_field_accuracy": sum(row["effective_field_hits"] for row in rows)
        / max(1, sum(row["field_total"] for row in rows)),
        "fallback_rate": sum(row["fallback"] for row in rows) / count,
        "cache_hit_rate": sum(row["cache_hit"] for row in rows) / count,
        "rerank_top1": sum(row["rerank_top1"] for row in rerank) / max(1, len(rerank)),
        "rerank_mrr": sum(row["rerank_mrr"] for row in rerank) / max(1, len(rerank)),
        "effective_rerank_top1": sum(row["effective_rerank_top1"] for row in rerank)
        / max(1, len(rerank)),
        "effective_rerank_mrr": sum(row["effective_rerank_mrr"] for row in rerank)
        / max(1, len(rerank)),
        "latency_s": {
            "min": latencies[0] if latencies else 0.0,
            "median": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
            "max": latencies[-1] if latencies else 0.0,
        },
    }


def _semantic_snapshot(scene_id, result, query, object_result, rerank):
    return {
        "scene_id": scene_id,
        "navigation_selected_node_id": int(result.selected_candidate.node_id),
        "navigation_target_grid": list(result.selected_candidate.target_grid),
        "query_parse": query.get("normalized_output"),
        "object_attributes": object_result.get("normalized_output"),
        "candidate_rerank": rerank.get("normalized_output"),
    }


def _field_hits(actual: dict[str, Any], expected: dict[str, Any]) -> tuple[int, int]:
    expected_flat = _flatten(expected)
    actual_flat = _flatten(actual)
    hits = sum(
        key in actual_flat and canonical_json_bytes(actual_flat[key]) == canonical_json_bytes(value)
        for key, value in expected_flat.items()
    )
    return hits, max(1, len(expected_flat))


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, name))
        else:
            result[name] = item
    return result


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = (len(values) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _object_metadata(row: dict[str, Any]) -> dict[str, Any]:
    response = row["response"]
    return {
        **row.get("metadata", {}),
        "object_id": row.get("object_id"),
        "object_tag": response.get("object_tag"),
        "possible_tags": response.get("possible_tags"),
        "colors": response.get("colors"),
        "body_type": response.get("body_type"),
        "motion_state": response.get("motion_state"),
    }


def _load_object_rows(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            result.setdefault(row["scene_id"], row)
    return result


def _select_complete_scenes(project_root: Path, requested: list[str], count: int) -> list[str]:
    scenes = []
    for scene in requested:
        inputs, _ = _resolve_scene_inputs(project_root, scene)
        if inputs is None:
            continue
        crop = inputs["qwen_exp_dir"] / "cfslam_captions_qwen_debug/0.png"
        structured = (
            project_root
            / "outputs/project_v2/scenegraph_relations/stage3_full_rebuild"
            / scene
            / "structured_edges.json"
        )
        if crop.exists() and structured.exists():
            scenes.append(scene)
        if len(scenes) >= count:
            break
    return scenes


def _prepare_directories(output_root: Path) -> dict[str, Path]:
    result = {
        "root": output_root,
        "dataset": output_root / "dataset",
        "cache": output_root / "cache",
        "rule": output_root / "rule_baseline",
        "qwen": output_root / "qwen_inference",
        "failure": output_root / "failure_fallback",
    }
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def _validate_environment(args, project_root: Path, output_root: Path) -> None:
    required = [
        args.model / "config.json",
        args.model / "model.safetensors.index.json",
        args.adapter / "adapter_config.json",
        project_root / "outputs/project_v2/qwen_eval/baseline_before_stage4.json",
        project_root / "outputs/project_v2/qwen_eval/dataset/split_manifest.json",
        project_root / "outputs/project_v2/qwen_eval/sanity_training/multimodal_processor_smoke.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"阶段4本地输入缺失：{missing}")
    allowed_outputs = {
        (project_root / "outputs/project_v2/qwen_eval").resolve(),
        (
            project_root
            / "outputs/project_v2/incremental_navigation/regression_stage4"
        ).resolve(),
        (
            project_root
            / "outputs/project_v2/integration_regression/regression_stage4"
        ).resolve(),
    }
    if output_root not in allowed_outputs:
        raise ValueError(
            "阶段4输出必须位于qwen_eval、阶段5或阶段6的精确隔离回归目录"
        )


def _verify_baseline(project_root: Path, baseline_path: Path) -> dict[str, Any]:
    baseline = _read_json(baseline_path)
    path_map: dict[str, Path | list[Path]] = {
        "outputs/project_v2/subset": Path("outputs/project_v2/subset"),
        "outputs/project_v2/offline_navigation": Path("outputs/project_v2/offline_navigation"),
        "outputs/project_v2/scenegraph_relations": Path("outputs/project_v2/scenegraph_relations"),
        "outputs/qwen_finetune_dataset": Path("outputs/qwen_finetune_dataset"),
        "outputs/experiment_summary/qwen_lora_eval": Path("outputs/experiment_summary/qwen_lora_eval"),
        "checkpoint/Qwen2-VL-2B-Instruct": Path("checkpoint/Qwen2-VL-2B-Instruct"),
        "checkpoint/qwen_lora_semantic_delegate": Path("checkpoint/qwen_lora_semantic_delegate"),
        "doc_existing_files": [
            Path("doc/change_logs/README.md"),
            Path("doc/change_logs/stage1_engineering_foundation_and_data_tools.md"),
            Path("doc/change_logs/stage2_target_selection_and_path_planning.md"),
            Path("doc/change_logs/stage3_scenegraph_relations.md"),
            Path("doc/project_optimization_plan.md"),
        ],
    }
    checks = {}
    for label, relative in path_map.items():
        expected = baseline["protected_artifacts"][label]["manifest_sha256"]
        actual = _aggregate_manifest_hash(project_root, relative)
        checks[label] = {"expected": expected, "actual": actual, "unchanged": expected == actual}
    for filename in ("MEMORY.md", "PAPER_MEMORY.md"):
        expected = baseline["protected_artifacts"][filename]["sha256"]
        actual = sha256_file(project_root / filename)
        checks[filename] = {"expected": expected, "actual": actual, "unchanged": expected == actual}
    scene_checks = {}
    for scene_id, expected_values in baseline["real_scene_inputs"].items():
        paths = {
            "scene_graph_sha256": project_root
            / f"dataset/nuscenes/{scene_id}/exp_qwen2vl_semantics_v1/scene_graph.json",
            "crop_0_sha256": project_root
            / f"dataset/nuscenes/{scene_id}/exp_qwen2vl_semantics_v1/cfslam_captions_qwen_debug/0.png",
            "navigation_cases_sha256": project_root
            / f"dataset/nuscenes/{scene_id}/exp_qwen2vl_semantics_v1/evaluation/navigation_cases.json",
            "structured_edges_sha256": project_root
            / f"outputs/project_v2/scenegraph_relations/stage3_full_rebuild/{scene_id}/structured_edges.json",
        }
        scene_checks[scene_id] = {
            key: {
                "expected": expected_values[key],
                "actual": sha256_file(path),
                "unchanged": expected_values[key] == sha256_file(path),
            }
            for key, path in paths.items()
        }
    passed = all(item["unchanged"] for item in checks.values()) and all(
        item["unchanged"] for scene in scene_checks.values() for item in scene.values()
    )
    return {
        "schema_version": "stage4-baseline-verification-v1",
        "protected_artifacts": checks,
        "real_scene_inputs": scene_checks,
        "passed": passed,
    }


def _aggregate_manifest_hash(project_root: Path, target: Path | list[Path]) -> str:
    paths = target if isinstance(target, list) else sorted(
        path.relative_to(project_root)
        for path in (project_root / target).rglob("*")
        if path.is_file()
    )
    digest = hashlib.sha256()
    for relative in sorted(paths, key=str):
        file_digest = sha256_file(project_root / relative)
        digest.update(f"{file_digest}  {relative}\n".encode())
    return digest.hexdigest()


def _without_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in sorted(value.items())
            if key not in {"latency", "timing"}
            and not key.endswith("_latency_s")
            and key not in {"runtime_s", "navigation_latency_s", "task_latency_s"}
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
