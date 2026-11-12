"""Execute manifest-selected query replays with isolated artifacts and caches."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from omegaconf import OmegaConf

from conceptgraph.nuscenes.expansion_benchmark import content_sha256
from conceptgraph.nuscenes.benchmark_fingerprints import (
    implementation_manifest,
    scene_input_manifest,
)
from conceptgraph.scripts.lane_experiment_metrics import (
    qwen_audit_metrics,
    replay_metrics,
)
from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT, PROJECT_ROOT
from conceptgraph.scripts.run_incremental_semantic_navigation import (
    DEFAULT_CONFIG,
    run_end_to_end_replay,
    write_json,
)
from conceptgraph.scripts.run_qwen_online_closed_loop import validate_runtime


def jobs_from_manifest(root, phase):
    selection = json.loads((root / "scene_selection_manifest.json").read_text())
    protocol = json.loads((root / "benchmark_protocol.json").read_text())
    queries = [
        json.loads(line)
        for line in (root / "benchmark_queries.jsonl").read_text().splitlines()
    ]
    jobs = []
    for query in queries:
        if not query["online_replay_selected"]:
            continue
        if (
            phase == "gpu"
            and query["role"] != "sealed_test"
            and query["scene"] not in {"scene-0916", "scene-0003"}
        ):
            continue
        for lane in (False, True):
            for variant in ("off",) if phase == "cpu" else ("base", "adapter"):
                jobs.append(
                    {
                        "scene": query["scene"],
                        "query_id": query["query_id"],
                        "query_text": query["query_text"],
                        "role": query["role"],
                        "variant": variant,
                        "lane": lane,
                        "selection_sha256": selection["sha256"],
                        "protocol_sha256": protocol["sha256"],
                    }
                )
    return jobs


def require_test_release(root, job):
    if job["role"] != "sealed_test":
        return
    path = root / "test_release.json"
    if not path.exists():
        raise ValueError(
            "sealed test not released: input/crop leakage and GT association audit required"
        )
    release = json.loads(path.read_text())
    if release.get("protocol_sha256") != job["protocol_sha256"] or not release.get(
        "approved_for_frozen_evaluation"
    ):
        raise ValueError("test release missing or inconsistent with frozen protocol")
    if implementation_manifest(PROJECT_ROOT)["sha256"] != release.get(
        "implementation_sha256"
    ):
        raise ValueError(
            "method implementation changed after test release; create a disclosed new protocol version"
        )


def run_job(root, directory, job, *, qwen_backend=None):
    require_test_release(root, job)
    material = {
        "job": job,
        "implementation": implementation_manifest(PROJECT_ROOT),
        "inputs": scene_input_manifest(root, job["scene"]),
    }
    expected = content_sha256(material)
    complete = directory / "complete.json"
    if complete.exists():
        previous = json.loads(complete.read_text())
        if previous["input_sha256"] != expected:
            raise ValueError("completed job identity changed")
        return previous
    if directory.exists():
        raise FileExistsError(f"partial job retained: {directory}; choose a new run-id")
    directory.mkdir(parents=True)
    write_json(directory / "job.json", job)
    write_json(directory / "input_manifest.json", material)
    settings = OmegaConf.load(DEFAULT_CONFIG)
    values = {
        "dataset_root": str(root / "converted"),
        "lane_map_root": str(root / "lane_maps"),
        "lane_constrained": job["lane"],
        "experiment_audit": True,
        "export_benchmark_nodes": True,
        "mapping_device": "cpu",
        "lane_projection_max_distance_m": 8.0,
        "lane_centerline_weight": 0.25,
        "lane_direction_tolerance_deg": 75.0,
    }
    for key, value in values.items():
        OmegaConf.update(settings, key, value, force_add=True)
    settings.query_text = job["query_text"]
    enabled = job["variant"] != "off"
    settings.online.qwen_enabled = enabled
    OmegaConf.update(settings, "online.pose_forward_axis", 2, force_add=True)
    settings.online.relation_coordinate_frame = "ego_heading"
    settings.qwen_online.enabled = enabled
    settings.qwen_online.cache_dir = str(directory / "cache")
    settings.qwen_online.frame_period_s = 0.0
    OmegaConf.update(
        settings, "qwen_online.deterministic_frame_barrier", enabled, force_add=True
    )
    if job["variant"] == "base":
        settings.qwen_online.adapter_path = None
        settings.qwen_online.adapter_revision = "base-model-no-adapter"
    runtime = validate_runtime(settings) if enabled else {"qwen_enabled": False}
    phases = {}
    for repeat in ("first", "cache_repeat"):
        settings.qwen_online.log_dir = str(directory / repeat / "qwen_audit")
        report, signature = run_end_to_end_replay(
            job["scene"],
            settings,
            directory / repeat,
            write_outputs=True,
            qwen_backend=qwen_backend,
        )
        frames = (
            directory / repeat / "end_to_end_replay" / job["scene"] / "frames.jsonl"
        )
        phases[repeat] = {
            "summary": report,
            "signature": signature,
            "metrics": replay_metrics(frames),
            "qwen_audit": qwen_audit_metrics(
                directory / repeat / "qwen_audit",
                int(report.get("qwen_backend_call_count", 0)),
            ),
        }
        write_json(directory / f"{repeat}_report.json", phases[repeat])
    result = {
        "input_sha256": expected,
        "job": job,
        "runtime": runtime,
        "first": phases["first"],
        "cache_repeat": phases["cache_repeat"],
        "determinism_passed": phases["first"]["signature"]
        == phases["cache_repeat"]["signature"],
        "cache_zero_new_backend_calls": phases["cache_repeat"]["summary"].get(
            "qwen_backend_call_count", 0
        )
        == 0,
        "source_frames_passed": all(
            not v["summary"]["source_failures"] for v in phases.values()
        ),
    }
    result["execution_passed"] = (
        result["determinism_passed"]
        and result["cache_zero_new_backend_calls"]
        and result["source_frames_passed"]
    )
    write_json(complete, result)
    print(
        json.dumps({"job": job, "execution_passed": result["execution_passed"]}),
        flush=True,
    )
    if not result["execution_passed"]:
        raise RuntimeError(f"cache, source, or determinism check failed: {directory}")
    return result


def main():
    parser = argparse.ArgumentParser(description="多场景、查询、Qwen条件隔离回放")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--phase", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--role",
        choices=("development", "legacy_regression", "sealed_test"),
        action="append",
    )
    parser.add_argument("--scene", action="append")
    parser.add_argument("--worker-job", type=Path)
    parser.add_argument("--persistent-backend", action="store_true")
    parser.add_argument(
        "--variant", choices=("off", "base", "adapter"), action="append"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if (
        not root.is_relative_to(DEFAULT_ROOT)
        or re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id) is None
    ):
        raise ValueError("invalid output root/run-id")
    run = root / "experiments" / args.run_id
    if args.worker_job:
        job = json.loads(args.worker_job.read_text())
        run_job(
            root,
            run
            / job["query_id"]
            / f"{job['variant']}_{'lane' if job['lane'] else 'drivable'}",
            job,
        )
        return
    jobs = jobs_from_manifest(root, args.phase)
    jobs = [
        job
        for job in jobs
        if (not args.role or job["role"] in args.role)
        and (not args.scene or job["scene"] in args.scene)
        and (not args.variant or job["variant"] in args.variant)
    ]
    if args.persistent_backend and args.phase != "gpu":
        raise ValueError("persistent backend only applies to GPU conditions")
    if args.persistent_backend:
        jobs.sort(
            key=lambda job: (job["variant"], job["scene"], job["query_id"], job["lane"])
        )
    if not jobs or args.workers < 1 or (args.phase == "gpu" and args.workers != 1):
        raise ValueError(
            "empty jobs, invalid workers, or GPU jobs requested concurrently"
        )
    run.mkdir(parents=True, exist_ok=True)
    plan = {
        "phase": args.phase,
        "workers": args.workers,
        "jobs": jobs,
        "sha256": content_sha256(jobs),
    }
    plan_path = run / "execution_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("existing execution plan changed")
    write_json(plan_path, plan)
    jobs_dir = run / "job_inputs"
    jobs_dir.mkdir(exist_ok=True)
    if args.persistent_backend:
        import gc
        import torch
        from conceptgraph.vlm.qwen_vl_chat import QwenVLChat

        backend, variant = None, None
        statuses = []
        for job in jobs:
            directory = (
                run
                / job["query_id"]
                / f"{job['variant']}_{'lane' if job['lane'] else 'drivable'}"
            )
            if backend is None or variant != job["variant"]:
                backend = None
                gc.collect()
                torch.cuda.empty_cache()
                settings = OmegaConf.load(DEFAULT_CONFIG)
                started = __import__("time").perf_counter()
                backend = QwenVLChat(
                    str(settings.qwen_online.model_name_or_path),
                    adapter_path=str(settings.qwen_online.adapter_path)
                    if job["variant"] == "adapter"
                    else None,
                    torch_dtype=getattr(torch, str(settings.qwen_online.torch_dtype)),
                    max_image_side=int(settings.qwen_online.max_image_side),
                )
                variant = job["variant"]
                write_json(
                    run / f"{variant}_model_initialization.json",
                    {
                        "elapsed_s": __import__("time").perf_counter() - started,
                        "backend_reused_across_jobs": True,
                        "cache_isolation_per_job": True,
                    },
                )
            run_job(root, directory, job, qwen_backend=backend)
            statuses.append({"job": job, "returncode": 0})
            write_json(
                run / "execution_status.json",
                {
                    "jobs": statuses,
                    "all_processes_succeeded": len(statuses) == len(jobs),
                },
            )
        return

    def execute(pair):
        i, job = pair
        path = jobs_dir / f"{i:05}.json"
        write_json(path, job)
        with (jobs_dir / f"{i:05}.log").open("a") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    __name__.replace(
                        "__main__", "conceptgraph.scripts.run_lane_expansion_benchmark"
                    ),
                    "--root",
                    str(root),
                    "--run-id",
                    args.run_id,
                    "--worker-job",
                    str(path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=3600,
            )
        return {
            "job": job,
            "returncode": result.returncode,
            "log": str(path.with_suffix(".log")),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(execute, enumerate(jobs)))
    write_json(
        run / "execution_status.json",
        {
            "jobs": results,
            "all_processes_succeeded": all(r["returncode"] == 0 for r in results),
        },
    )
    if any(r["returncode"] for r in results):
        raise RuntimeError(
            "query batch contains failed jobs; inspect execution_status.json"
        )


if __name__ == "__main__":
    main()
