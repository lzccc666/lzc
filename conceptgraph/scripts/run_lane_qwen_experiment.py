"""Run isolated, evidence-backed lane and real-Qwen experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from conceptgraph.navigation.lane_constraints import (
    load_lane_topology,
    validate_lane_path,
)
from conceptgraph.navigation.lane_geometry import continuous_collision_cells
from conceptgraph.navigation.pipeline import SemanticNavigationPipeline
from conceptgraph.scripts.lane_experiment_metrics import (
    qwen_audit_metrics,
    replay_metrics,
)
from conceptgraph.scripts.run_incremental_semantic_navigation import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    run_end_to_end_replay,
    write_json,
)
from conceptgraph.scripts.run_qwen_online_closed_loop import validate_runtime

SCENES = ("scene-0916", "scene-0003")
EXPERIMENT_ROOT = PROJECT_ROOT / "outputs/project_v2/lane_qwen_experiment"
LANE_ROOT = EXPERIMENT_ROOT / "lane_maps"


def _settings(*, lane, qwen, cache_dir, log_dir, adapter=True):
    settings = OmegaConf.load(DEFAULT_CONFIG)
    for key, value in {
        "lane_constrained": lane,
        "lane_map_root": str(LANE_ROOT),
        "experiment_audit": True,
        "lane_projection_max_distance_m": 8.0,
        "lane_centerline_weight": 0.25,
        "lane_direction_tolerance_deg": 75.0,
    }.items():
        OmegaConf.update(settings, key, value, force_add=True)
    settings.online.qwen_enabled = qwen
    settings.qwen_online.enabled = qwen
    settings.qwen_online.cache_dir = str(cache_dir)
    settings.qwen_online.log_dir = str(log_dir)
    OmegaConf.update(
        settings, "qwen_online.deterministic_frame_barrier", bool(qwen), force_add=True
    )
    settings.qwen_online.frame_period_s = 0.0
    if not adapter:
        settings.qwen_online.adapter_path = None
        settings.qwen_online.adapter_revision = "base-model-no-adapter"
    return settings


def runtime_diagnostics():
    import torch

    model = PROJECT_ROOT / "checkpoint/Qwen2-VL-2B-Instruct"
    adapter = PROJECT_ROOT / "outputs/project_v2/qwen_eval/finetune_v2/best_adapter"
    index = model / "model.safetensors.index.json"
    shards = (
        sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if index.exists()
        else []
    )
    required = [
        model / "config.json",
        index,
        adapter / "adapter_config.json",
        adapter / "adapter_model.safetensors",
        *[model / name for name in shards],
    ]
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device_nodes": [str(p) for p in Path("/dev").glob("nvidia*")],
        "gpu_names": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        "model_path": str(model),
        "adapter_path": str(adapter),
        "missing_artifacts": [str(p) for p in required if not p.is_file()],
        "weight_shards": shards,
        "real_inference_completed": False,
        "backend_call_count": 0,
    }


def _run_replay_condition(root, *, lane, qwen, scenes, repeat="first", adapter=True):
    reports, signatures, metrics, audit_counts = {}, {}, {}, {}
    for scene in scenes:
        settings = _settings(
            lane=lane,
            qwen=qwen,
            cache_dir=root / "cache" / scene,
            log_dir=root / repeat / "qwen_audit" / scene,
            adapter=adapter,
        )
        if qwen:
            validate_runtime(settings)
        print(f"START {root.name}/{repeat}/{scene}", flush=True)
        report, signature = run_end_to_end_replay(
            scene, settings, root / repeat, write_outputs=True
        )
        reports[scene], signatures[scene] = report, signature
        metrics[scene] = replay_metrics(
            root / repeat / "end_to_end_replay" / scene / "frames.jsonl"
        )
        audit_counts[scene] = qwen_audit_metrics(
            root / repeat / "qwen_audit" / scene,
            int(report.get("qwen_backend_call_count", 0)),
        )
        audit_counts[scene]["applied"] = {
            task: int(report.get("qwen", {}).get(task + "_applied", 0))
            for task in ("query_parse", "object_attributes", "candidate_rerank")
        }
        print(
            f"DONE {scene}: {metrics[scene]['successful_plans']}/{metrics[scene]['planning_attempts']} plans",
            flush=True,
        )
    result = {
        "lane_constrained": lane,
        "qwen_enabled": qwen,
        "adapter_enabled": adapter if qwen else False,
        "repeat": repeat,
        "scenes": reports,
        "signatures": signatures,
        "metrics": metrics,
        "qwen_audit": audit_counts,
        "backend_call_count": sum(
            v["backend_call_count"] for v in audit_counts.values()
        ),
    }
    write_json(root / repeat / "condition_report.json", result)
    return result


def _pipeline_config(scene, *, lane):
    cfg = OmegaConf.load(
        PROJECT_ROOT / "conceptgraph/hydra_configs/semantic_navigation.yaml"
    )
    exp = PROJECT_ROOT / "dataset/nuscenes" / scene / "exp_qwen2vl_semantics_v1"
    cfg.scene_id = scene
    cfg.scene_graph_json_path = str(exp / "scene_graph.json")
    cfg.annotated_scenegraph_json_path = str(exp / "annotated_scenegraph.json")
    cfg.scene_graph_edge_path = str(exp / "cfslam_scenegraph_edges.pkl")
    cfg.scene_map_path = str(exp / "map/scene_map_cfslam_pruned.pkl.gz")
    cfg.bev_map_path = str(LANE_ROOT / scene / "lane_constrained_bev.npz")
    if lane:
        cfg.lane_map_path = cfg.bev_map_path
        cfg.lane_topology_path = str(LANE_ROOT / scene / "lane_topology.json")
        cfg.target_selection.mode = "lane_constrained"
    return cfg


def run_static_scene_comparison(root, scenes):
    comparisons = {}
    for scene in scenes:
        poses = np.loadtxt(
            PROJECT_ROOT / "dataset/nuscenes" / scene / "poses.txt"
        ).reshape(-1, 4, 4)
        pose = np.r_[poses[0, :3, 3], poses[0, :2, 2]]
        topology = load_lane_topology(LANE_ROOT / scene / "lane_constrained_bev.npz")
        with np.load(LANE_ROOT / scene / "lane_constrained_bev.npz") as loaded:
            base = {
                key: loaded[key] for key in ("occupancy", "origin_xy", "resolution")
            }
        comparisons[scene] = {}
        for mode in ("free_space", "drivable_area", "lane_area", "lane_constrained"):
            directory = root / "static_comparison" / scene / mode
            directory.mkdir(parents=True, exist_ok=True)
            arrays = dict(base)
            if mode == "free_space":
                arrays["occupancy"] = np.zeros_like(base["occupancy"])
            elif mode == "lane_area":
                arrays["occupancy"] = np.maximum(base["occupancy"], ~topology.lane_mask)
            np.savez_compressed(directory / "input_bev.npz", **arrays)
            cfg = _pipeline_config(scene, lane=mode == "lane_constrained")
            cfg.bev_map_path = str((directory / "input_bev.npz").resolve())
            pipeline = SemanticNavigationPipeline(cfg)
            result = pipeline.select(
                "前往小汽车附近", pose, run_semantic_delegate=False
            )
            pipeline.save_selection_result(result, directory, save_visualizations=False)
            grid = pipeline._occupancy_grid
            np.savez_compressed(
                directory / "live_grid.npz",
                occupancy=grid.occupancy,
                origin_xy=grid.origin_xy,
                resolution=grid.resolution,
            )
            validation = None
            if result.plan is not None:
                rc = (
                    (result.plan.path_world[:, :2] - topology.origin_xy)
                    / topology.resolution
                )[:, ::-1]
                validation = validate_lane_path(rc, topology)
            comparisons[scene][mode] = {
                "success": result.success,
                "failure_code": result.failure_code,
                "selected_node_id": None
                if result.selected_candidate is None
                else result.selected_candidate.node_id,
                "lane_validation": validation,
                "output_dir": str(directory.resolve()),
            }
            print(f"STATIC {scene}/{mode}: {result.success}", flush=True)
    return comparisons


def verify_static_artifacts(source, scenes):
    """Resume a completed static phase only after rechecking inputs and full paths."""
    source = source.resolve()
    if not source.is_relative_to(EXPERIMENT_ROOT.resolve()):
        raise ValueError(
            "static source must be within the isolated experiment directory"
        )
    comparisons = {}
    for scene in scenes:
        topology = load_lane_topology(LANE_ROOT / scene / "lane_constrained_bev.npz")
        with np.load(LANE_ROOT / scene / "lane_constrained_bev.npz") as data:
            ordinary = data["occupancy"]
        comparisons[scene] = {}
        for mode in ("free_space", "drivable_area", "lane_area", "lane_constrained"):
            directory = source / "static_comparison" / scene / mode
            result = json.loads(
                (directory / "target_selection_result.json").read_text()
            )
            expected = (
                np.zeros_like(ordinary)
                if mode == "free_space"
                else np.maximum(ordinary, ~topology.lane_mask)
                if mode == "lane_area"
                else ordinary
            )
            with np.load(directory / "input_bev.npz") as data:
                if (
                    not np.array_equal(data["occupancy"], expected)
                    or not np.allclose(
                        data["origin_xy"], topology.origin_xy, rtol=0, atol=1e-6
                    )
                    or float(data["resolution"]) != topology.resolution
                ):
                    raise ValueError(f"static input changed: {directory}")
            validation = None
            collisions = set()
            if result["success"]:
                plan = json.loads(
                    (directory / "semantic_navigation_plan.json").read_text()
                )
                with np.load(directory / "live_grid.npz") as data:
                    cells = np.asarray(plan["path"]["grid_rc"])
                    for a, b in zip(cells[:-1], cells[1:]):
                        collisions.update(
                            map(
                                tuple,
                                continuous_collision_cells(a, b, data["occupancy"]),
                            )
                        )
                    if len(cells) == 1:
                        collisions.update(
                            map(
                                tuple,
                                continuous_collision_cells(
                                    cells[0], cells[0], data["occupancy"]
                                ),
                            )
                        )
                world = np.asarray(plan["path"]["world_xyz"])
                validation = validate_lane_path(
                    ((world[:, :2] - topology.origin_xy) / topology.resolution)[
                        :, ::-1
                    ],
                    topology,
                )
                if collisions or (
                    mode == "lane_constrained" and not validation["valid"]
                ):
                    raise ValueError(f"static safety verification failed: {directory}")
            comparisons[scene][mode] = {
                "success": result["success"],
                "failure_code": result["failure_code"],
                "selected_node_id": None
                if result["selected_candidate"] is None
                else result["selected_candidate"]["node_id"],
                "lane_validation": validation,
                "collision_cells": [list(cell) for cell in sorted(collisions)],
                "output_dir": str(directory),
                "resumed_from_completed_static_phase": True,
                "selection_sha256": hashlib.sha256(
                    (directory / "target_selection_result.json").read_bytes()
                ).hexdigest(),
            }
    return comparisons


def baseline_verify(output_root: Path) -> dict[str, Any]:
    baseline = json.loads(
        (output_root / "baseline_before_lane_qwen.json").read_text(encoding="utf-8")
    )
    mismatches = []
    checked = 0
    for entry in baseline.get("protected", {}).values():
        for file_entry in entry.get("files", []):
            file_path = PROJECT_ROOT / file_entry["path"]
            if not file_path.exists():
                mismatches.append({"path": file_entry["path"], "reason": "missing"})
                continue
            checked += 1
            digest = hashlib.sha256()
            with file_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != file_entry["sha256"]:
                mismatches.append(
                    {"path": file_entry["path"], "reason": "sha256_changed"}
                )
    unexpected = []
    allowed_new_docs = {
        "doc/change_logs/stage7_lane_constrained_navigation_and_qwen_online.md"
    }
    for protected_root, entry in baseline.get("protected", {}).items():
        directory = PROJECT_ROOT / protected_root
        if directory.is_dir():
            expected = {item["path"] for item in entry.get("files", [])}
            current = {
                str(path.relative_to(PROJECT_ROOT))
                for path in directory.rglob("*")
                if path.is_file()
            }
            unexpected.extend(sorted(current - expected - allowed_new_docs))
    mismatches.extend(
        {"path": path, "reason": "unexpected_file_in_protected_namespace"}
        for path in unexpected
    )
    report = {
        "schema_version": "lane-qwen-baseline-verification-v1",
        "passed": not mismatches,
        "checked_files": checked,
        "mismatches": mismatches,
        "new_output_root_excluded": True,
        "verification_method": "streaming_full_SHA256_comparison",
        "full_sha256_rehash": True,
        "protected_directory_membership_checked": True,
    }
    write_json(output_root / "baseline_verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description="HD Map车道规划与真实Qwen回放对照")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--run-id", default="strict_v2")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument(
        "--static-artifacts-from",
        type=Path,
        help="验证并复用本会话已完成的静态阶段产物",
    )
    parser.add_argument("--scene", action="append", choices=SCENES, dest="scenes")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output != EXPERIMENT_ROOT.resolve():
        raise ValueError("实验只允许写入隔离的 lane_qwen_experiment 目录")
    import re

    if re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id) is None:
        raise ValueError(
            "run-id must contain only letters, numbers, underscore or hyphen"
        )
    run = output / "runs" / args.run_id
    if run.exists():
        raise FileExistsError(f"运行目录已存在，请使用新的 --run-id: {run}")
    run.mkdir(parents=True)
    started = time.perf_counter()
    scenes = tuple(args.scenes or SCENES)
    runtime = runtime_diagnostics()
    write_json(output / "qwen_runtime_report.json", runtime)
    static = (
        verify_static_artifacts(args.static_artifacts_from, scenes)
        if args.static_artifacts_from
        else run_static_scene_comparison(run, scenes)
    )
    write_json(run / "static_phase_report.json", static)
    conditions, determinism = {}, {}
    for name, lane in (("drivable_qwen_off", False), ("lane_qwen_off", True)):
        first = _run_replay_condition(
            run / "online" / name, lane=lane, qwen=False, scenes=scenes
        )
        repeat = _run_replay_condition(
            run / "online" / name,
            lane=lane,
            qwen=False,
            scenes=scenes,
            repeat="determinism_repeat",
        )
        conditions[name] = first
        determinism[name] = {
            "first": first["signatures"],
            "repeat": repeat["signatures"],
            "passed": first["signatures"] == repeat["signatures"],
        }
    qwen_ready = (
        runtime["cuda_available"]
        and runtime["cuda_device_count"] > 0
        and not runtime["missing_artifacts"]
        and not args.skip_qwen
    )
    if qwen_ready:
        for adapter in (False, True):
            for lane in (False, True):
                name = f"{'lane' if lane else 'drivable'}_qwen_{'adapter' if adapter else 'base'}"
                first = _run_replay_condition(
                    run / "online" / name,
                    lane=lane,
                    qwen=True,
                    scenes=scenes,
                    adapter=adapter,
                )
                repeat = _run_replay_condition(
                    run / "online" / name,
                    lane=lane,
                    qwen=True,
                    scenes=scenes,
                    repeat="cache_repeat",
                    adapter=adapter,
                )
                conditions[name] = first
                determinism[name] = {
                    "first": first["signatures"],
                    "repeat": repeat["signatures"],
                    "passed": first["signatures"] == repeat["signatures"],
                    "first_backend_count": first["backend_call_count"],
                    "repeat_backend_count": repeat["backend_call_count"],
                    "cache_verified": first["backend_call_count"] > 0
                    and repeat["backend_call_count"] == 0
                    and all(
                        v["cache_hit_count"] > 0 for v in repeat["qwen_audit"].values()
                    ),
                }
        runtime["backend_call_count"] = sum(
            c["backend_call_count"] for c in conditions.values()
        )
        runtime["successful_backend_response_count"] = sum(
            audit["successful_backend_response_count"]
            for c in conditions.values()
            for audit in c["qwen_audit"].values()
        )
        runtime["real_inference_completed"] = (
            runtime["successful_backend_response_count"] > 0
        )
        runtime["all_qwen_conditions_passed"] = all(
            all(
                audit["successful_backend_response_count"] > 0
                and all(count > 0 for count in audit["applied"].values())
                for audit in c["qwen_audit"].values()
            )
            and determinism[name]["passed"]
            and determinism[name]["cache_verified"]
            for name, c in conditions.items()
            if c["qwen_enabled"]
        )
    else:
        runtime["failure_code"] = (
            "GPU_WORK_DEFERRED_BY_USER"
            if args.skip_qwen
            else "CUDA_UNAVAILABLE"
            if not runtime["cuda_available"]
            else "MISSING_MODEL_ARTIFACTS"
        )
        runtime["pending_conditions"] = [
            "Qwen base/adapter × drivable/lane × first/cache_repeat × both scenes"
        ]
    write_json(output / "qwen_runtime_report.json", runtime)
    write_json(output / "scene_reports.json", conditions)
    write_json(
        output / "ablation_report.json",
        {
            "run_dir": str(run),
            "static_four_map_conditions": static,
            "online_conditions": list(conditions),
            "qwen_pending": not runtime.get("all_qwen_conditions_passed", False),
        },
    )
    write_json(output / "determinism_report.json", determinism)
    write_json(
        output / "lane_constraint_report.json",
        {name: c["metrics"] for name, c in conditions.items()},
    )
    write_json(
        output / "performance_report.json",
        {
            "run_time_s": time.perf_counter() - started,
            "conditions": {name: c["metrics"] for name, c in conditions.items()},
        },
    )
    write_json(
        output / "qwen_online_report.json",
        {
            "real_backend_completed": runtime["real_inference_completed"],
            "runtime": runtime,
            "conditions": {name: c["qwen_audit"] for name, c in conditions.items()},
        },
    )
    maps = {
        scene: json.loads((LANE_ROOT / scene / "lane_topology.json").read_text())[
            "diagnostics"
        ]
        for scene in scenes
    }
    write_json(output / "lane_map_reports.json", maps)
    baseline = baseline_verify(output)
    write_json(
        output / "validation_report.json",
        {
            "goal_complete": False,
            "code_and_cpu_experiments_completed": len(scenes) == 2,
            "gpu_work_deferred_by_user": bool(args.skip_qwen),
            "real_qwen_gpu_pending": not runtime["real_inference_completed"],
            "baseline_sha256_passed": baseline["passed"],
            "determinism_passed": all(v["passed"] for v in determinism.values()),
            "run_dir": str(run),
            "tests_and_lint": "see verification_checks.json; not inferred from experiment completion",
        },
    )
    print(
        json.dumps(
            {
                "run": str(run),
                "qwen_real": runtime["real_inference_completed"],
                "baseline_passed": baseline["passed"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
