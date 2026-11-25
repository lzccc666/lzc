from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from conceptgraph.navigation.planner import path_is_collision_free
from conceptgraph.scripts.run_stage2_offline_navigation import (
    _reference_navigation_case,
    _resolve_scene_inputs,
)
from conceptgraph.scripts.run_stage4_offline_qwen_eval import (
    _load_object_rows,
    _make_pipeline,
    _object_metadata,
)
from conceptgraph.vlm.qwen_cache import semantic_version, sha256_file
from conceptgraph.vlm.qwen_semantic_delegate import QwenDelegateConfig, QwenSemanticDelegate
from conceptgraph.vlm.qwen_vl_chat import QwenVLChat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/project_v2/qwen_eval/finetune_v2"
SCENES = ("scene-0916", "scene-0003")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行新版Qwen LoRA双真实场景无缓存下游回归。"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "checkpoint/Qwen2-VL-2B-Instruct",
    )
    parser.add_argument("--adapter", type=Path, default=OUTPUT_ROOT / "best_adapter")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root != OUTPUT_ROOT.resolve():
        raise ValueError(f"输出必须精确位于 {OUTPUT_ROOT.resolve()}")
    for required in (
        args.model / "config.json",
        args.model / "model.safetensors.index.json",
        args.adapter / "adapter_config.json",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    run_root = output_root / "downstream"
    run_root.mkdir(parents=True, exist_ok=True)
    backend = QwenVLChat(
        args.model.resolve(),
        args.device,
        adapter_path=args.adapter.resolve(),
        torch_dtype=torch.bfloat16,
        max_image_side=448,
    )
    delegate = QwenSemanticDelegate(
        QwenDelegateConfig(
            enabled=True,
            model_name_or_path=args.model.resolve(),
            adapter_path=args.adapter.resolve(),
            model_revision="local-qwen2-vl-2b-instruct",
            adapter_revision=sha256_file(args.adapter / "adapter_model.safetensors"),
            device=args.device,
            use_qwen_object_attributes=True,
            use_qwen_query_parse=True,
            use_qwen_rerank=True,
            timeout_s=args.timeout_s,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            cache_dir=None,
            log_dir=run_root / "audit",
            prompt_profile="finetune_v2",
        ),
        backend=backend,
    )
    target_config = OmegaConf.load(
        PROJECT_ROOT / "conceptgraph/hydra_configs/semantic_navigation.yaml"
    ).target_selection
    object_rows = _load_object_rows(
        PROJECT_ROOT / "outputs/project_v2/qwen_eval/dataset/object_attributes.jsonl"
    )
    manifest = _read_json(
        PROJECT_ROOT / "outputs/project_v2/qwen_eval/dataset/split_manifest.json"
    )
    rows = []
    started = time.perf_counter()
    try:
        for scene_id in SCENES:
            rows.append(
                _run_scene(
                    scene_id,
                    delegate,
                    target_config,
                    object_rows[scene_id],
                    run_root,
                    manifest,
                )
            )
    finally:
        delegate.close(wait=True)

    report = {
        "schema_version": "qwen-finetune-v2-downstream-v1",
        "mode": "real_backend_no_cache_system_regression",
        "model": str(args.model.resolve()),
        "adapter": str(args.adapter.resolve()),
        "adapter_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
        "prompt_profile": "finetune_v2",
        "inference_dtype": "bfloat16",
        "max_image_side": 448,
        "cache_enabled": False,
        "scenes": rows,
        "backend_call_count": delegate.backend_call_count,
        "runtime_s": time.perf_counter() - started,
        "checks": {
            "exact_scenes": tuple(row["scene_id"] for row in rows) == SCENES,
            "three_raw_calls_per_scene": delegate.backend_call_count == 3 * len(SCENES),
            "all_raw_schema_valid": all(row["all_raw_schema_valid"] for row in rows),
            "all_candidate_ids_legal": all(row["candidate_ids_legal"] for row in rows),
            "no_fallback_used": all(row["fallback_count"] == 0 for row in rows),
            "all_paths_continuous_collision_free": all(
                row["continuous_collision_free"] for row in rows
            ),
            "no_cache": all(row["cache_hit_count"] == 0 for row in rows),
        },
        "generalization_claim": False,
        "disclosure": (
            "scene-0916和scene-0003仅用于系统回归；若位于训练划分，"
            "本报告不将其计作封存集泛化精度。"
        ),
    }
    report["passed"] = all(report["checks"].values())
    report["deterministic_content_sha256"] = semantic_version(_without_runtime(report))
    _write_json(output_root / "downstream_regression.json", report)
    if not report["passed"]:
        raise RuntimeError("双场景下游回归未通过")
    print(output_root / "downstream_regression.json")


def _run_scene(
    scene_id: str,
    delegate: QwenSemanticDelegate,
    target_config: Any,
    object_row: dict[str, Any],
    run_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    inputs, diagnostic = _resolve_scene_inputs(PROJECT_ROOT, scene_id)
    if inputs is None:
        raise RuntimeError(f"{scene_id}: {diagnostic}")
    reference, diagnostic = _reference_navigation_case(inputs["qwen_exp_dir"])
    if reference is None:
        raise RuntimeError(f"{scene_id}: {diagnostic}")
    pipeline = _make_pipeline(
        inputs,
        reference,
        target_config,
        run_root / scene_id,
        delegate=delegate,
    )
    calls_before = delegate.backend_call_count
    result = pipeline.select(
        reference["instruction"], np.asarray(reference["vehicle_pose"], dtype=float)
    )
    if not result.success or result.plan is None:
        raise RuntimeError(f"{scene_id}: navigation failed: {result.failure_code}")
    pipeline.save_selection_result(
        result, run_root / scene_id, save_plan=True, save_visualizations=False
    )
    object_audit = delegate.describe_object(
        [object_row["crop_path"]],
        _object_metadata(object_row),
        object_version=object_row["image_sha256"],
    )
    audits = {
        "query_parse": result.debug["qwen_query_parse"],
        "object_attributes": object_audit,
        "candidate_rerank": result.debug["qwen_candidate_rerank"],
    }
    for task, audit in audits.items():
        _write_json(run_root / scene_id / f"{task}.json", audit)
    rerank = audits["candidate_rerank"]
    ranking = rerank.get("normalized_output", {}).get("ranking", [])
    verified_ids = [int(value) for value in rerank.get("verified_candidate_ids", [])]
    candidate_ids_legal = bool(ranking) and sorted(ranking) == sorted(verified_ids)
    collision_free = bool(
        pipeline._occupancy_grid is not None
        and path_is_collision_free(pipeline._occupancy_grid, result.plan.path_grid)
    )
    return {
        "scene_id": scene_id,
        "dataset_split": manifest["scene_to_split"].get(scene_id),
        "system_regression_only": True,
        "instruction": reference["instruction"],
        "selected_node_id": int(result.selected_candidate.node_id),
        "qwen_recommended_node_id": rerank.get("selected_node_id"),
        "verified_candidate_ids": verified_ids,
        "raw_ranking": ranking,
        "raw_outputs": {task: audit.get("raw_output") for task, audit in audits.items()},
        "schema_valid": {
            task: bool(audit.get("output_schema_valid")) for task, audit in audits.items()
        },
        "all_raw_schema_valid": all(
            audit.get("output_schema_valid") for audit in audits.values()
        ),
        "candidate_ids_legal": candidate_ids_legal,
        "fallback_count": sum(bool(audit.get("fallback_used")) for audit in audits.values()),
        "cache_hit_count": sum(
            bool(audit.get("cache", {}).get("hit")) for audit in audits.values()
        ),
        "backend_call_count": delegate.backend_call_count - calls_before,
        "path_point_count": len(result.plan.path_grid),
        "continuous_collision_free": collision_free,
        "result_path": str((run_root / scene_id / "target_selection_result.json").resolve()),
        "plan_path": str((run_root / scene_id / "semantic_navigation_plan.json").resolve()),
    }


def _without_runtime(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_runtime(item)
            for key, item in sorted(value.items())
            if key != "runtime_s"
        }
    if isinstance(value, list):
        return [_without_runtime(item) for item in value]
    return value


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
