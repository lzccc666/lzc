from __future__ import annotations

import argparse
import time
from pathlib import Path

from omegaconf import OmegaConf

from conceptgraph.scripts.run_incremental_semantic_navigation import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    run_end_to_end_replay,
    write_json,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/project_v2/qwen_online_closed_loop"
REQUIRED_SCENES = ("scene-0916", "scene-0003")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root != DEFAULT_OUTPUT.resolve():
        raise ValueError(f"Qwen online output must be isolated at {DEFAULT_OUTPUT.resolve()}")
    final_report = output_root / "qwen_online_closed_loop_report.json"
    if final_report.exists() and not args.overwrite:
        raise FileExistsError(f"report already exists: {final_report}; pass --overwrite to rerun")

    settings = OmegaConf.load(args.config)
    settings.online.qwen_enabled = True
    settings.qwen_online.enabled = True
    runtime = validate_runtime(settings)
    output_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    scene_reports = {}
    signatures = {}
    for scene_id in REQUIRED_SCENES:
        report, signature = run_end_to_end_replay(
            scene_id,
            settings,
            output_root,
            write_outputs=True,
        )
        scene_reports[scene_id] = report
        signatures[scene_id] = signature

    qwen_totals: dict[str, int] = {}
    for report in scene_reports.values():
        for key, value in report.get("qwen", {}).items():
            qwen_totals[key] = qwen_totals.get(key, 0) + int(value)
    pending = sum(
        int(report.get("qwen_final_audit", {}).get("pending_count", 0))
        for report in scene_reports.values()
    )
    result = {
        "schema_version": "qwen-online-closed-loop-v1",
        "passed": bool(
            len(scene_reports) == len(REQUIRED_SCENES)
            and pending == 0
            and qwen_totals.get("query_parse_applied", 0) > 0
            and qwen_totals.get("object_attributes_applied", 0) > 0
            and qwen_totals.get("candidate_rerank_applied", 0) > 0
        ),
        "scenes": scene_reports,
        "deterministic_content_sha256": signatures,
        "qwen_totals": dict(sorted(qwen_totals.items())),
        "pending_qwen_count": pending,
        "runtime": runtime,
        "runtime_s": time.perf_counter() - started,
        "safety_contract": {
            "object_geometry_authority": "Stage2/Stage3 geometry and occupancy",
            "rerank_input": "A-star reachable and collision-validated candidates only",
            "target_switch": "state-machine persistence followed by fresh Stage2 replanning",
            "qwen_direct_path_write": False,
        },
        "output_root": str(output_root),
    }
    write_json(final_report, result)
    if not result["passed"]:
        raise RuntimeError(f"Qwen closed-loop acceptance failed; inspect {final_report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="微调Qwen双场景在线语义导航闭环回放")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_runtime(settings) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Qwen online closed loop requires at least one visible CUDA GPU")
    model_path = Path(str(settings.qwen_online.model_name_or_path)).resolve()
    adapter_value = settings.qwen_online.adapter_path
    adapter_path = None if adapter_value in {None, "", "None"} else Path(str(adapter_value)).resolve()
    required = [
        model_path / "config.json",
        model_path / "model.safetensors.index.json",
    ]
    import json
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        required.extend(model_path / name for name in sorted(set(index["weight_map"].values())))
    if adapter_path is not None:
        required.extend([adapter_path / "adapter_config.json", adapter_path / "adapter_model.safetensors"])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Qwen artifacts: {missing}")
    requested_dtype = str(settings.qwen_online.torch_dtype)
    dtype_fallback = None
    if requested_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        settings.qwen_online.torch_dtype = "float16"
        dtype_fallback = "float16"
    return {
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "requested_torch_dtype": requested_dtype,
        "resolved_torch_dtype": str(settings.qwen_online.torch_dtype),
        "dtype_fallback": dtype_fallback,
        "model_path": str(model_path),
        "adapter_path": None if adapter_path is None else str(adapter_path),
        "gpu_memory_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "prompt_profile": str(settings.qwen_online.prompt_profile),
        "max_image_side": int(settings.qwen_online.max_image_side),
    }


if __name__ == "__main__":
    main()
