"""CPU fault injection on real scene inputs; these are NOT Qwen model metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from conceptgraph.navigation.pipeline import SemanticNavigationPipeline
from conceptgraph.scripts.run_incremental_semantic_navigation import (
    PROJECT_ROOT,
    write_json,
)
from conceptgraph.scripts.run_lane_qwen_experiment import (
    EXPERIMENT_ROOT,
    SCENES,
    _pipeline_config,
)
from conceptgraph.vlm.qwen_semantic_delegate import (
    QwenDelegateConfig,
    QwenSemanticDelegate,
)


def path_signature(result):
    path = [] if result.plan is None else [list(cell) for cell in result.plan.path_grid]
    return hashlib.sha256(json.dumps(path, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-run", type=Path, required=True)
    parser.add_argument("--run-id", default="v2")
    args = parser.parse_args()
    import re

    if re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id) is None:
        raise ValueError("invalid run id")
    output = EXPERIMENT_ROOT / "failure_injection_cpu" / args.run_id
    if output.exists():
        raise FileExistsError(f"fault injection output exists: {output}")
    output.mkdir(parents=True)
    scenes = {}
    for scene in SCENES:
        directory = args.static_run / "static_comparison" / scene / "lane_constrained"
        saved = json.loads((directory / "target_selection_result.json").read_text())
        candidate = saved["selected_candidate"] or saved["evaluated_candidates"][0]
        target_id = int(candidate["node_id"])
        cfg = _pipeline_config(scene, lane=True)
        cfg.bev_map_path = str((directory / "input_bev.npz").resolve())
        pipeline = SemanticNavigationPipeline(cfg)
        matrix = np.loadtxt(
            PROJECT_ROOT / "dataset/nuscenes" / scene / "poses.txt"
        ).reshape(-1, 4, 4)[0]
        pose = np.r_[matrix[:3, 3], matrix[:2, 2]]
        baseline = pipeline.select(
            "前往小汽车附近",
            pose,
            target_node_id=target_id,
            run_semantic_delegate=False,
        )
        signature = path_signature(baseline)
        pipeline.save_selection_result(
            baseline, output / scene / "qwen_off", save_visualizations=False
        )
        cases = {}
        for failure in ("exception", "timeout", "invalid_schema", "empty_output"):

            def backend(task, prompt, images, generation_config, _failure=failure):
                if _failure == "exception":
                    raise RuntimeError("injected fault; no model is loaded")
                if _failure == "timeout":
                    time.sleep(0.06)
                    return "{}"
                return (
                    ""
                    if _failure == "empty_output"
                    else '{"unexpected":"corrupt schema"}'
                )

            delegate = QwenSemanticDelegate(
                QwenDelegateConfig(
                    enabled=True,
                    use_qwen_query_parse=True,
                    use_qwen_rerank=True,
                    timeout_s=0.01,
                    model_revision="synthetic-fault-injection-only",
                    log_dir=output / scene / failure / "audit",
                ),
                backend=backend,
            )
            try:
                pipeline._semantic_delegate = delegate
                pipeline._semantic_delegate_initialized = True
                actual = pipeline.select(
                    "前往小汽车附近", pose, target_node_id=target_id
                )
                audit = actual.debug.get("qwen_query_parse", {})
                # Failure returns can occur before the query audit is attached to result.debug.
                audit = audit or pipeline._last_query_audit or {}
                unchanged = (
                    actual.success == baseline.success
                    and actual.failure_code == baseline.failure_code
                    and path_signature(actual) == signature
                )
                cases[failure] = {
                    "passed": unchanged and bool(audit.get("fallback_used")),
                    "same_safety_outcome": unchanged,
                    "fallback_reason": audit.get("fallback_reason"),
                    "failure_code": audit.get("failure_code"),
                    "injected_backend_call_count": delegate.backend_call_count,
                    "path_sha256": path_signature(actual),
                    "model_loaded": False,
                }
                pipeline.save_selection_result(
                    actual, output / scene / failure, save_visualizations=False
                )
            finally:
                delegate.close()
            print(scene, failure, cases[failure]["passed"], flush=True)
        scenes[scene] = {
            "baseline_success": baseline.success,
            "baseline_failure_code": baseline.failure_code,
            "locked_target_id": target_id,
            "baseline_path_sha256": signature,
            "cases": cases,
        }
    report = {
        "passed": all(
            case["passed"]
            for scene in scenes.values()
            for case in scene["cases"].values()
        ),
        "scope": "真实场景输入 + 注入故障后端；验证规则回退和车道安全，不代表真实Qwen推理或模型准确率",
        "model_loaded": False,
        "artifact_root": str(output.resolve()),
        "real_qwen_backend_call_count": 0,
        "scenes": scenes,
    }
    write_json(EXPERIMENT_ROOT / "failure_injection_report.json", report)


if __name__ == "__main__":
    main()
