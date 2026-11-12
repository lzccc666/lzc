from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/project_v2/qwen_eval/finetune_v2"


def main() -> None:
    commands = {
        "failure_fallbacks": [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/vlm/test_qwen_fallback.py",
            "tests/vlm/test_qwen_navigation_integration.py::test_illegal_qwen_rerank_id_falls_back_without_bypassing_joint_utility",
            "tests/navigation/test_online_state_machine.py::test_late_rerank_with_changed_object_version_is_discarded",
            "tests/integration/test_project_v2_integration.py::test_qwen_timeout_exception_and_schema_faults_all_fallback",
        ],
        "stage4_stage5_stage6_related": [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/vlm",
            "tests/navigation",
            "tests/integration",
        ],
        "full_pytest": [sys.executable, "-m", "pytest", "-q", "tests"],
        "targeted_ruff": [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "conceptgraph/vlm/qwen_finetune_v2.py",
            "conceptgraph/vlm/qwen_semantic_delegate.py",
            "conceptgraph/vlm/qwen_vl_chat.py",
            "conceptgraph/scripts/finalize_qwen_finetune_v2.py",
            "conceptgraph/scripts/run_qwen_finetune_v2_downstream_regression.py",
            "conceptgraph/scripts/run_qwen_finetune_v2_qa.py",
            "scripts/train_qwen_lora_semantic_delegate_v2.py",
            "tests/vlm/test_qwen_finetune_v2.py",
        ],
        "git_diff_check": ["git", "diff", "--check"],
    }
    command_reports = {name: _run(command) for name, command in commands.items()}
    old_reports = {
        "stage4": _read_json(
            PROJECT_ROOT / "outputs/project_v2/qwen_eval/validation_report.json"
        ),
        "stage5": _read_json(
            PROJECT_ROOT
            / "outputs/project_v2/incremental_navigation/validation_report.json"
        ),
        "stage6": _read_json(
            PROJECT_ROOT
            / "outputs/project_v2/integration_regression/validation_report.json"
        ),
    }
    existing_regressions = {
        "stage4_runtime_passed": bool(old_reports["stage4"].get("runtime_passed")),
        "stage5_passed": bool(old_reports["stage5"].get("passed")),
        "stage6_passed": bool(old_reports["stage6"].get("passed")),
        "stage5_failure_injection_passed": bool(
            old_reports["stage5"].get("failure_injection_passed")
        ),
    }
    required = {
        name: OUTPUT_ROOT / name
        for name in (
            "training_report.json",
            "heldout_evaluation.json",
            "per_task_metrics.json",
            "per_scene_metrics.json",
            "error_analysis.json",
            "determinism_report.json",
            "downstream_regression.json",
            "baseline_verification.json",
            "validation_report.json",
        )
    }
    artifact_checks = {}
    for name, path in required.items():
        payload = _read_json(path)
        artifact_checks[name] = {
            "exists": True,
            "json_reparsed": True,
            "passed_field": payload.get("passed"),
        }
    best_adapter = OUTPUT_ROOT / "best_adapter"
    artifact_checks["best_adapter"] = {
        "exists": best_adapter.is_dir(),
        "adapter_config_exists": (best_adapter / "adapter_config.json").is_file(),
        "adapter_model_exists": (best_adapter / "adapter_model.safetensors").is_file(),
    }
    passed = bool(
        all(report["passed"] for report in command_reports.values())
        and all(existing_regressions.values())
        and all(all(value is not False for value in check.values()) for check in artifact_checks.values())
        and _read_json(OUTPUT_ROOT / "heldout_evaluation.json")["passed"]
        and _read_json(OUTPUT_ROOT / "downstream_regression.json")["passed"]
        and _read_json(OUTPUT_ROOT / "baseline_verification.json")["passed"]
    )
    report = {
        "schema_version": "qwen-finetune-v2-qa-v1",
        "commands": command_reports,
        "existing_stage_reports_read_only": existing_regressions,
        "artifact_checks": artifact_checks,
        "protected_stage_outputs_rewritten": False,
        "passed": passed,
    }
    _write_json(OUTPUT_ROOT / "qa_report.json", report)
    validation_path = OUTPUT_ROOT / "validation_report.json"
    validation = _read_json(validation_path)
    validation["final_qa"] = report
    validation["baseline_verification_passed"] = _read_json(
        OUTPUT_ROOT / "baseline_verification.json"
    )["passed"]
    validation["heldout_passed"] = _read_json(
        OUTPUT_ROOT / "heldout_evaluation.json"
    )["passed"]
    validation["downstream_regression_passed"] = _read_json(
        OUTPUT_ROOT / "downstream_regression.json"
    )["passed"]
    validation["passed"] = bool(validation.get("passed") and passed)
    _write_json(validation_path, validation)
    if not passed:
        raise RuntimeError("Qwen finetune v2最终QA未通过")
    print(OUTPUT_ROOT / "qa_report.json")


def _run(command: list[str]) -> dict[str, Any]:
    env = dict(os.environ)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "output": completed.stdout[-20000:],
    }


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
