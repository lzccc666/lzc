"""Finalize CPU acceptance using stored frames, live grids and actual check results."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from conceptgraph.navigation.lane_constraints import (
    load_lane_topology,
    validate_lane_path,
)
from conceptgraph.navigation.lane_geometry import continuous_collision_cells
from conceptgraph.scripts.lane_experiment_metrics import replay_metrics
from conceptgraph.scripts.run_incremental_semantic_navigation import (
    PROJECT_ROOT,
    deterministic_replay_signature,
    write_json,
)
from conceptgraph.scripts.run_lane_qwen_experiment import (
    EXPERIMENT_ROOT,
    LANE_ROOT,
    SCENES,
    baseline_verify,
    verify_static_artifacts,
)


def check_replay(run):
    topologies = {
        scene: load_lane_topology(LANE_ROOT / scene / "lane_constrained_bev.npz")
        for scene in SCENES
    }
    checks = []
    metrics = {}
    determinism = {}
    for condition in ("drivable_qwen_off", "lane_qwen_off"):
        metrics[condition] = {}
        hashes = {}
        for repeat in ("first", "determinism_repeat"):
            hashes[repeat] = {}
            for scene in SCENES:
                directory = (
                    run / "online" / condition / repeat / "end_to_end_replay" / scene
                )
                rows = [
                    json.loads(line)
                    for line in (directory / "frames.jsonl").read_text().splitlines()
                    if line.strip()
                ]
                summary = json.loads((directory / "summary.json").read_text())
                signature = deterministic_replay_signature(rows)
                hashes[repeat][scene] = signature
                if (
                    signature != summary["deterministic_content_sha256"]
                    or summary["source_failures"]
                ):
                    raise ValueError(f"replay signature or source failure: {directory}")
                expected_mode = (
                    "lane_constrained"
                    if condition == "lane_qwen_off"
                    else "joint_utility"
                )
                if (
                    summary["planning_configuration"]["target_selection"]["mode"]
                    != expected_mode
                ):
                    raise ValueError("condition used the wrong planning mode")
                if summary.get("qwen_backend_call_count", 0) or any(
                    summary.get("qwen", {}).values()
                ):
                    raise ValueError(
                        "CPU Qwen-off condition unexpectedly invoked semantic jobs"
                    )
                topology = topologies[scene]
                successful = 0
                for row in rows:
                    audit = row["experiment_audit"]
                    cells = np.asarray(audit["path_grid"], dtype=int).reshape(-1, 2)
                    occupancy_path = audit["occupancy_artifact"]
                    if not occupancy_path:
                        continue
                    with np.load(occupancy_path) as data:
                        if cells.size:
                            world = np.asarray(audit["path_world"])
                            expected = data["origin_xy"] + cells[:, ::-1] * float(
                                data["resolution"]
                            )
                            if not np.allclose(
                                world[:, :2], expected, rtol=0, atol=1e-6
                            ):
                                raise ValueError("path/grid frame mismatch")
                            start_index = (
                                0
                                if row["replanned"]
                                else audit["remaining_path_start_index"]
                            )
                            remaining = cells[start_index:]
                            collisions = set()
                            for a, b in zip(remaining[:-1], remaining[1:]):
                                collisions.update(
                                    map(
                                        tuple,
                                        continuous_collision_cells(
                                            a, b, data["occupancy"]
                                        ),
                                    )
                                )
                            if len(remaining) == 1:
                                collisions.update(
                                    map(
                                        tuple,
                                        continuous_collision_cells(
                                            remaining[0],
                                            remaining[0],
                                            data["occupancy"],
                                        ),
                                    )
                                )
                            if collisions:
                                raise ValueError(
                                    f"collision in accepted path: {directory}, frame {row['frame_idx']}"
                                )
                            if condition == "lane_qwen_off":
                                validation = validate_lane_path(
                                    (
                                        (world[:, :2] - topology.origin_xy)
                                        / topology.resolution
                                    )[:, ::-1],
                                    topology,
                                )
                                if not validation["valid"]:
                                    raise ValueError(
                                        f"illegal lane path: {directory}, frame {row['frame_idx']}"
                                    )
                    if row["replanned"] and len(cells) and row["failure"] is None:
                        successful += 1
                checks.append(
                    {
                        "condition": condition,
                        "repeat": repeat,
                        "scene": scene,
                        "frames": len(rows),
                        "successful_plans": successful,
                        "passed": True,
                        "signature": signature,
                    }
                )
                if repeat == "first":
                    metrics[condition][scene] = replay_metrics(
                        directory / "frames.jsonl"
                    )
        determinism[condition] = {
            "first": hashes["first"],
            "repeat": hashes["determinism_repeat"],
            "passed": hashes["first"] == hashes["determinism_repeat"],
        }
    return checks, metrics, determinism


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reuse-baseline-verification", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if not run.is_relative_to(EXPERIMENT_ROOT.resolve()):
        raise ValueError("run must be within isolated experiment output")
    root = EXPERIMENT_ROOT
    checks, metrics, determinism = check_replay(run)
    static = verify_static_artifacts(run, SCENES)
    # The test process records real assertion failures in JUnit; no hard-coded success.
    suites = ET.parse(root / "full_tests.xml").getroot()
    cases = suites.findall(".//testcase")
    failed = [
        case.get("name")
        for case in cases
        if any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
    ]
    if failed or not cases:
        raise ValueError(f"tests not passing: {failed}")
    files = sorted(
        set(
            subprocess.check_output(
                ["git", "diff", "--name-only", "--", "*.py"], text=True
            ).splitlines()
            + subprocess.check_output(
                ["git", "ls-files", "--others", "--exclude-standard", "--", "*.py"],
                text=True,
            ).splitlines()
        )
    )
    lint = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *files],
        capture_output=True,
        text=True,
    )
    diff = subprocess.run(["git", "diff", "--check"], capture_output=True, text=True)
    write_json(
        root / "verification_checks.json",
        {
            "pytest": {
                "passed": not failed,
                "test_count": len(cases),
                "junit": str(root / "full_tests.xml"),
            },
            "ruff": {
                "passed": lint.returncode == 0,
                "python_files": len(files),
                "output": lint.stdout + lint.stderr,
            },
            "git_diff_check": {
                "passed": diff.returncode == 0,
                "output": diff.stdout + diff.stderr,
            },
        },
    )
    if lint.returncode or diff.returncode:
        raise ValueError("lint or diff check failed")
    if args.reuse_baseline_verification:
        baseline_path = root / "baseline_verification.json"
        baseline = json.loads(baseline_path.read_text())
        newest_condition = max(
            path.stat().st_mtime
            for path in (run / "online").rglob("condition_report.json")
        )
        if (
            not baseline.get("full_sha256_rehash")
            or not baseline.get("protected_directory_membership_checked")
            or baseline_path.stat().st_mtime < newest_condition
        ):
            raise ValueError(
                "baseline verification must follow the completed replay conditions"
            )
    else:
        baseline = baseline_verify(root)
    failure = json.loads((root / "failure_injection_report.json").read_text())
    write_json(root / "lane_constraint_report.json", metrics)
    write_json(root / "determinism_report.json", determinism)
    write_json(
        root / "independent_path_verification.json",
        {
            "passed": True,
            "checks": checks,
            "scope": "all accepted replan paths and all remaining reused paths, checked against each frame's live occupancy",
        },
    )
    write_json(
        root / "performance_report.json",
        {
            "conditions": metrics,
            "unit": "one scene is one recorded query replay; plan/frame distributions are separate",
            "timings_include": "frame_runtime_s covers mapping/navigation; independent audit and NPZ writing are excluded from frame timing and included in each scene runtime_s",
        },
    )
    existing = json.loads((root / "ablation_report.json").read_text())
    existing["static_four_map_conditions"] = static
    existing["gpu_work_deferred_by_user"] = True
    write_json(root / "ablation_report.json", existing)
    todo = {
        "status": "待办：用户明确推迟需要GPU的部分",
        "model": str(PROJECT_ROOT / "checkpoint/Qwen2-VL-2B-Instruct"),
        "adapter": str(
            PROJECT_ROOT / "outputs/project_v2/qwen_eval/finetune_v2/best_adapter"
        ),
        "tasks": [
            "CUDA实际加载本地Qwen，记录设备/dtype/显存与模型及adapter标识",
            "base/adapter × 普通/车道 × 两场景首次隔离缓存实际后端调用",
            "复跑缓存命中，新增backend调用为0并比较结构化内容哈希",
            "核验query_parse、object_attributes、candidate_rerank的实际应用计数与schema/fallback/耗时",
            "消费真实GPU结果绘制候选重排前后和Qwen开关/缓存对照",
        ],
        "metric_limit": "逐帧对象检索Top-1/MRR缺少独立目标真值，当前记null；需要独立标注或明确的基准，不能用规则排序自证正确率",
    }
    write_json(root / "gpu_todo.json", todo)
    git = {
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip(),
        "status": subprocess.check_output(["git", "status", "--short"], text=True),
        "committed_or_pushed_by_this_task": False,
    }
    write_json(root / "git_state_after_cpu.json", git)
    manifest = []
    for relative in files:
        path = PROJECT_ROOT / relative
        manifest.append(
            {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    write_json(root / "code_manifest_after_cpu.json", manifest)
    visualization_path = (
        root / "visualizations" / run.name / "visualization_manifest.json"
    )
    visualization = json.loads(visualization_path.read_text())
    from PIL import Image

    png_files = sorted(visualization_path.parent.rglob("*.png"))
    if len(png_files) < 7:
        raise ValueError("missing visualizations")
    for path in png_files:
        with Image.open(path) as image:
            image.verify()
    report = {
        "cpu_scope_passed": baseline["passed"]
        and failure["passed"]
        and all(v["passed"] for v in determinism.values()),
        "full_gpu_goal_complete": False,
        "gpu_work_deferred_by_user": True,
        "run_dir": str(run),
        "test_count": len(cases),
        "independent_path_checks": checks,
        "baseline_sha256_passed": baseline["passed"],
        "determinism_passed": all(v["passed"] for v in determinism.values()),
        "failure_injection_passed": failure["passed"],
        "cpu_artifacts_complete": True,
        "visualization_png_count": len(png_files),
        "visualization_manifest": str(visualization_path),
        "visualization_run_matches": Path(visualization["run_dir"]).resolve() == run,
        "real_qwen_gpu_pending": True,
        "retrieval_ground_truth_pending": True,
        "qwen_real_backend_call_count": 0,
    }
    write_json(root / "validation_report.json", report)
    bad = []
    counts = {"json": 0, "jsonl": 0, "jsonl_rows": 0}

    def invalid_constant(value):
        raise ValueError(f"nonstandard JSON constant {value}")

    for path in sorted(root.rglob("*")):
        try:
            if path.suffix == ".json":
                json.loads(path.read_text(), parse_constant=invalid_constant)
                counts["json"] += 1
            elif path.suffix == ".jsonl":
                for line in path.read_text().splitlines():
                    if line.strip():
                        json.loads(line, parse_constant=invalid_constant)
                        counts["jsonl_rows"] += 1
                counts["jsonl"] += 1
        except (ValueError, UnicodeError) as exc:
            bad.append({"path": str(path), "error": str(exc)})
    write_json(
        root / "json_parse_report.json",
        {"passed": not bad, "counts": counts, "errors": bad, "strict_json": True},
    )
    report["json_parse_passed"] = not bad
    report["json_parse_counts"] = counts
    report["cpu_scope_passed"] = (
        report["cpu_scope_passed"] and not bad and report["visualization_run_matches"]
    )
    write_json(root / "validation_report.json", report)
    if bad or not report["cpu_scope_passed"]:
        raise ValueError("CPU acceptance failed")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
