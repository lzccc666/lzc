"""有界阶段计时；CPU试跑与唯一CUDA进程的正式测量严格分开。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import functools
import gzip
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import threading
import time

import numpy as np

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


def summarize(samples):
    """失败不计为成功吞吐；原样保留失败数和冷启动分母。"""
    groups = defaultdict(list)
    for row in samples:
        duration = row["duration_s"]
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be positive finite seconds")
        groups[(row["stage"], row["phase"])].append(row)
    result = {}
    for (stage, phase), rows in sorted(groups.items()):
        durations = [r["duration_s"] for r in rows]
        result[stage + "/" + phase] = dict(
            calls=len(rows),
            failures=sum(not r["succeeded"] for r in rows),
            median_s=float(np.median(durations)),
            p95_s=float(np.percentile(durations, 95)),
            summed_s=sum(durations),
            successful_calls_per_measured_s=sum(r["succeeded"] for r in rows)
            / sum(durations),
        )
    return result


class Recorder:
    def __init__(self, destination):
        self.destination = destination
        self.rows = []
        self.lock = threading.Lock()
        self.context = {}

    @contextmanager
    def measure(self, stage, *, phase="warm", synchronize=None, **metadata):
        if synchronize is not None:
            synchronize()
        start = time.perf_counter_ns()
        succeeded = False
        try:
            yield
            if synchronize is not None:
                synchronize()
            succeeded = True
        finally:
            row = dict(
                stage=stage,
                phase=phase,
                duration_s=(time.perf_counter_ns() - start) / 1e9,
                succeeded=succeeded,
                **self.context,
                **metadata,
            )
            with self.lock:
                self.rows.append(row)
                with (self.destination / "timings.jsonl").open("a") as handle:
                    handle.write(
                        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                    )

    @contextmanager
    def patch(self, owner, name, stage):
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with self.measure(stage):
                return original(*args, **kwargs)

        setattr(owner, name, wrapped)
        try:
            yield
        finally:
            setattr(owner, name, original)


def cuda_processes():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return sorted(
        {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}
    )


class ExclusiveCudaMonitor:
    """持续记录计算进程；任何其它PID或监测失败均取消独占信用。"""

    def __init__(self, destination, *, interval_s=0.5):
        self.destination = destination
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.events = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        try:
            pids = cuda_processes()
            event = dict(
                monotonic_s=time.monotonic(),
                pids=pids,
                other_pids=[p for p in pids if p != os.getpid()],
                error=None,
            )
        except Exception as exc:
            event = dict(
                monotonic_s=time.monotonic(), pids=[], other_pids=[], error=repr(exc)
            )
        self.events.append(event)
        with (self.destination / "CUDA_process_monitor.jsonl").open("a") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")

    def _run(self):
        while not self.stop_event.wait(self.interval_s):
            self.sample()

    def __enter__(self):
        self.sample()
        if self.events[-1]["other_pids"] or self.events[-1]["error"]:
            raise RuntimeError(
                "GPU is occupied or cannot be monitored; do not start model"
            )
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=20)
        self.sample()
        if self.thread.is_alive() or any(
            e["other_pids"] or e["error"] for e in self.events
        ):
            raise RuntimeError(
                "CUDA exclusivity was not established; keep contaminated samples"
            )


class PredictionProxy:
    def __init__(self, model, recorder, stage, synchronize):
        self.model, self.recorder, self.stage = model, recorder, stage
        self.synchronize = synchronize
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def predict(self, *args, **kwargs):
        phase = "cold_first_call" if self.calls == 0 else "warm"
        self.calls += 1
        with self.recorder.measure(
            self.stage, phase=phase, synchronize=self.synchronize
        ):
            return self.model.predict(*args, **kwargs)

    def set_classes(self, *args, **kwargs):
        with self.recorder.measure(
            "YOLO_vocabulary_setup", phase="cold_setup", synchronize=self.synchronize
        ):
            return self.model.set_classes(*args, **kwargs)


def frontend(plan, destination, recorder, device):
    import torch
    from omegaconf import OmegaConf
    from conceptgraph.scripts import streamlined_detections as frontend_module

    sync = torch.cuda.synchronize if device.startswith("cuda") else None
    cfg = OmegaConf.create(dict(device=device))
    with recorder.measure(
        "frontend_model_load", phase="cold_process", synchronize=sync
    ):
        models = frontend_module.load_detection_models(cfg)
    models = (
        PredictionProxy(models[0], recorder, "YOLO_detection", sync),
        PredictionProxy(models[1], recorder, "SAM_segmentation", sync),
        *models[2:],
    )
    original = frontend_module.compute_clip_features_batched
    calls = 0

    def clip(*args, **kwargs):
        nonlocal calls
        phase = "cold_first_call" if calls == 0 else "warm"
        calls += 1
        with recorder.measure(
            "CLIP_preprocess_image_text", phase=phase, synchronize=sync
        ):
            return original(*args, **kwargs)

    frontend_module.compute_clip_features_batched = clip
    results = []
    try:
        for index, item in enumerate(plan["frontend_inputs"]):
            assert digest(item["rgb"]) == item["rgb_sha256"]
            recorder.context = dict(input_id=item["input_id"], scene=item["scene"])
            with recorder.measure(
                "frontend_total",
                phase="cold_first_frame" if index == 0 else "warm",
                synchronize=sync,
            ):
                gobs, _ = frontend_module.detect_rgb_file(
                    cfg, models, item["rgb"], plan["classes"]
                )
            path = destination / f"{index:03d}_detections.pkl.gz"
            with gzip.open(path, "xb") as handle:
                pickle.dump(gobs, handle)
            results.append(
                dict(
                    **item,
                    detections=len(gobs["xyxy"]),
                    output_sha256=digest(path),
                    source_RGB_provenance=gobs["frame_provenance"],
                )
            )
    finally:
        frontend_module.compute_clip_features_batched = original
    dump(
        destination / "frontend_inputs_outputs.json",
        dict(records=results, actual_frontend_calls=len(results)),
    )
    return dict(actual_frontend_calls=len(results), model_calls_reused=False)


def semantic(plan, destination, recorder, variant):
    import torch
    from conceptgraph.completion.semantic_evaluation import invoke
    from conceptgraph.vlm.qwen_semantic_delegate import (
        QwenDelegateConfig,
        QwenSemanticDelegate,
    )
    from conceptgraph.vlm.qwen_vl_chat import QwenVLChat

    adapter = plan["semantic_variants"][variant]
    with recorder.measure(
        "Qwen_model_load", phase="cold_process", synchronize=torch.cuda.synchronize
    ):
        backend = QwenVLChat(
            str(PROJECT / "checkpoint/Qwen2-VL-2B-Instruct"),
            "cuda:0",
            adapter_path=adapter["adapter"],
            torch_dtype=torch.bfloat16,
            max_image_side=448,
        )
    delegate = QwenSemanticDelegate(
        QwenDelegateConfig(
            enabled=True,
            model_name_or_path=str(PROJECT / "checkpoint/Qwen2-VL-2B-Instruct"),
            adapter_path=adapter["adapter"],
            model_revision=plan["base_revision"],
            adapter_revision=adapter["weights_sha256"],
            device="cuda:0",
            prompt_profile="completion_v3",
            use_qwen_query_parse=True,
            use_qwen_object_attributes=True,
            use_qwen_rerank=True,
            max_new_tokens=256,
            max_images=1,
            torch_dtype="bfloat16",
            max_image_side=448,
            timeout_s=60,
            cache_dir=destination / "cache",
            log_dir=destination / "audit",
        ),
        backend=backend,
    )
    originals = []
    seen = set()
    try:
        with (destination / "uncached_responses.jsonl").open("x") as handle:
            for row in plan["semantic_inputs"]:
                if row.get("crop_path"):
                    assert digest(row["crop_path"]) == row["crop_sha256"]
                task = row["task"]
                recorder.context = dict(
                    sample_id=row["sample_id"], task=task, variant=variant
                )
                before = delegate.backend_call_count
                with recorder.measure(
                    "Qwen_" + task,
                    phase="cold_first_task" if task not in seen else "warm_uncached",
                    synchronize=torch.cuda.synchronize,
                ):
                    output = invoke(delegate, row)
                assert (
                    not output["cache"]["hit"]
                    and delegate.backend_call_count - before == 1
                )
                seen.add(task)
                originals.append(output)
                handle.write(
                    json.dumps(
                        dict(sample_id=row["sample_id"], envelope=output),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
        first_calls = delegate.backend_call_count
        with (destination / "cached_responses.jsonl").open("x") as handle:
            for row, original in zip(plan["semantic_inputs"], originals):
                recorder.context = dict(
                    sample_id=row["sample_id"], task=row["task"], variant=variant
                )
                with recorder.measure("Qwen_" + row["task"], phase="cache_hit"):
                    output = invoke(delegate, row)
                assert (
                    output["cache"]["hit"]
                    and output["raw_output"] == original["raw_output"]
                )
                assert output["normalized_output"] == original["normalized_output"]
                handle.write(
                    json.dumps(
                        dict(sample_id=row["sample_id"], envelope=output),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
        assert (
            delegate.backend_call_count == first_calls == len(plan["semantic_inputs"])
        )
    finally:
        delegate.close()
    return dict(
        actual_Qwen_calls=first_calls,
        replay_added_calls=0,
        cache_replayed_requests=len(originals),
        schema_valid=sum(r["output_schema_valid"] for r in originals),
        fallback_count=sum(r["fallback_used"] for r in originals),
        accuracy_credit=False,
    )


def recorded(plan, destination, recorder):
    from contextlib import ExitStack
    from PIL import Image
    from conceptgraph.completion import interactive_navigation as navigation
    from conceptgraph.completion import navigation_semantics as semantic_navigation
    from conceptgraph.completion.kinematic_benchmark import SPEC
    from conceptgraph.navigation.vehicle_geometry import VehicleState, step_vehicle

    case = Path(plan["recorded_case"])
    bridge = semantic_navigation.NavigationSemanticBridge(
        plan["recorded_query"],
        destination,
        None,
        tasks=(),
        maximum_candidates=3,
        candidate_plan_budget_s=10.0,
    )
    agent = navigation.PerceptionVehicleAgent(
        plan["recorded_query"],
        plan["recorded_prior"],
        plan["classes"],
        destination,
        semantic_bridge=bridge,
    )
    with ExitStack() as stack:
        for owner, name, stage in [
            (type(agent.mapper), "update", "mapping_update"),
            (type(agent.graph), "update_snapshot", "incremental_relations"),
            (navigation, "rank_candidate_targets", "rule_candidate_ranking"),
            (navigation, "hybrid_astar", "Hybrid_Astar_planning"),
            (semantic_navigation, "hybrid_astar", "Hybrid_Astar_planning"),
            (type(bridge), "select", "verified_candidate_chain"),
            (navigation, "lidar_obstacles", "observed_LiDAR_obstacles"),
        ]:
            stack.enter_context(recorder.patch(owner, name, stage))
        for item in plan["recorded_frames"]:
            index = item["frame_idx"]
            sensor_path = case / "sensor_evidence" / f"{index:06d}.json"
            assert digest(sensor_path) == item["sensor_sha256"]
            sensor = json.loads(sensor_path.read_text())
            color = Path(sensor["rgb"])
            det = case / "detections" / f"{index:06d}.pkl.gz"
            assert (
                digest(color) == sensor["rgb_sha256"]
                and digest(sensor["depth"]) == sensor["depth_sha256"]
            )
            assert digest(det) == item["detections_sha256"]
            rgb = np.asarray(Image.open(color).convert("RGB"))
            depth = np.asarray(Image.open(sensor["depth"]), dtype=float) / 1000
            with gzip.open(det) as handle:
                gobs = pickle.load(handle)
            state = VehicleState(**sensor["state"])
            recorder.context = dict(
                frame_idx=index, source="recorded_interactive_baseline"
            )
            with recorder.measure("perception_mapping_planning_total"):
                record = agent.perceive(
                    index,
                    state,
                    rgb,
                    depth,
                    np.asarray(sensor["camera_pose_cv"]),
                    color,
                    gobs,
                )
            with recorder.measure("tracking_control"):
                agent.control(state)
            dump(
                destination / f"frame_{index:06d}.json",
                json.loads(
                    json.dumps(record, default=navigation.json_safe, allow_nan=False)
                ),
            )
    execution = case / "execution.jsonl"
    assert digest(execution) == plan["recorded_execution_sha256"]
    before = VehicleState(
        **json.loads((case / "sensor_evidence/000000.json").read_text())["state"]
    )
    states = 0
    for index, row in enumerate(map(json.loads, execution.open())):
        recorder.context = dict(
            control_index=index, source="recorded_control_recompute"
        )
        command = row["control"]
        with recorder.measure("bicycle_state_update"):
            after = step_vehicle(
                before,
                command["acceleration_mps2"],
                command["steering_command_rad"],
                SPEC,
            )
        assert all(
            abs(value - row["state"][key]) < 1e-10
            for key, value in asdict(after).items()
        )
        before = after
        states += 1
    return dict(
        recorded_perception_calls=len(plan["recorded_frames"]),
        recomputed_saved_control_states=states,
        new_frontend_calls=0,
        new_closed_loop_executions=0,
        arrival_credit=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--mode", choices=["frontend", "semantic", "recorded"], required=True
    )
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    parser.add_argument("--variant", choices=["base", "old", "new"], default="new")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("invalid run id")
    plan = json.loads(args.plan.read_text())
    destination = OUTPUT / "performance" / args.run_id
    destination.mkdir(exist_ok=False)
    recorder = Recorder(destination)
    dump(
        destination / "release.json",
        dict(
            arguments={**vars(args), "plan": str(args.plan)},
            plan_sha256=digest(args.plan),
            source_sha256=digest(__file__),
            started_utc=datetime.now(timezone.utc).isoformat(),
            pid=os.getpid(),
            cold_process_does_not_mean_dropped_OS_page_cache=True,
            independent_test=False,
            goal_complete=False,
        ),
    )
    monitor = None
    try:
        import torch

        if args.device == "cpu" and torch.cuda.is_available():
            raise RuntimeError(
                "CPU profiling requires CUDA_VISIBLE_DEVICES empty before process start"
            )
        if args.mode == "semantic" and args.device != "cuda:0":
            raise ValueError("formal semantic profiling requires actual CUDA")
        if args.device.startswith("cuda"):
            monitor = ExclusiveCudaMonitor(destination)
            monitor.__enter__()
            torch.cuda.reset_peak_memory_stats()
        if args.mode == "frontend":
            actual = frontend(plan, destination, recorder, args.device)
        elif args.mode == "semantic":
            actual = semantic(plan, destination, recorder, args.variant)
        else:
            actual = recorded(plan, destination, recorder)
        if monitor:
            monitor.__exit__(None, None, None)
        report = dict(
            mode=args.mode,
            device=args.device,
            variant=args.variant,
            stages=summarize(recorder.rows),
            actual=actual,
            peak_torch_GPU_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20
            if monitor
            else None,
            peak_torch_GPU_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20
            if monitor
            else None,
            unique_CUDA_process_verified=monitor is not None,
            samples_are_stage_invocations_not_independent_scenes=True,
            nested_stage_times_not_additive=True,
            GPU_sync_added_at_frontend_and_semantic_boundaries=monitor is not None,
            CPU_shared_host_no_exclusive_performance_credit=monitor is None,
            external_GPU_activity_between_half_second_samples_not_excluded=monitor
            is not None,
            cold_OS_page_cache_not_guaranteed=True,
            end_to_end_throughput_credit=False,
            independent_test=False,
            goal_complete=False,
        )
        dump(destination / "report.json", report)
        print(
            json.dumps(
                dict(
                    run_id=args.run_id,
                    mode=args.mode,
                    actual=actual,
                    timing_rows=len(recorder.rows),
                    goal_complete=False,
                ),
                ensure_ascii=False,
            ),
            flush=True,
        )
    except BaseException as exc:
        if monitor is not None and monitor.thread.is_alive():
            monitor.stop_event.set()
            monitor.thread.join(timeout=20)
        dump(
            destination / "interrupted.json",
            dict(error=repr(exc), samples=len(recorder.rows), goal_complete=False),
        )
        raise


if __name__ == "__main__":
    main()
