"""独立子进程的真实时钟传感器故障；复用冻结导航链，不向推理提供对象GT。"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import copy
import gzip
import json
import math
from pathlib import Path
import pickle
import shutil
import sys
import time

import numpy as np


CONDITIONS = (
    "baseline",
    "sensor_blackout_4s",
    "lidar_dropout_80pct",
    "depth_noise_02m",
    "pose_noise_025m_2deg",
    "detection_dropout_30pct",
)
DETECTION_FIELDS = (
    "xyxy",
    "confidence",
    "class_id",
    "mask",
    "image_crops",
    "image_feats",
    "text_feats",
)


def retain_detections(gobs, keep):
    """只筛同一个真实检测的七个对齐字段；类别表、图像来源与输入不变。"""
    keep = np.asarray(keep)
    count = len(gobs["xyxy"])
    if keep.dtype != bool or keep.shape != (count,):
        raise ValueError(
            "detection mask must be boolean with the actual detection count"
        )
    result = copy.deepcopy(gobs)
    for key in DETECTION_FIELDS:
        value = result.get(key)
        if value is not None:
            if len(value) != count:
                raise ValueError("misaligned actual detection field: " + key)
            result[key] = (
                [value[i] for i in np.flatnonzero(keep)]
                if isinstance(value, list)
                else value[keep]
            )
    return result


@dataclass
class FaultPolicy:
    condition: str
    seed: int
    onset_distance_m: float = 0.5
    minimum_speed_mps: float = 0.2
    duration_s: float = 6.0

    def __post_init__(self):
        if self.condition not in CONDITIONS:
            raise ValueError("unknown sensor fault condition")
        if any(
            not math.isfinite(v) or v <= 0
            for v in [self.onset_distance_m, self.minimum_speed_mps, self.duration_s]
        ):
            raise ValueError("fault trigger and duration must be finite and positive")
        self.initial_xy = None
        self.onset_wall_s = None
        self.onset_frame = None
        self.onset_speed_mps = None
        self.onset_distance_actual_m = None
        self.last_wall_s = None
        self.last_index = None
        self.active = False
        self.active_frames = []
        self.last_frame_active = False
        self._rng = None

    def begin_frame(self, index, xy, speed_mps, wall_s):
        xy = np.asarray(xy, dtype=float)
        if (
            xy.shape != (2,)
            or not np.isfinite(xy).all()
            or not math.isfinite(wall_s)
            or not math.isfinite(speed_mps)
            or speed_mps < 0
            or index < 0
            or self.last_index is not None
            and index <= self.last_index
            or self.last_wall_s is not None
            and wall_s < self.last_wall_s
        ):
            raise ValueError("invalid or nonmonotonic physical observation")
        if self.initial_xy is None:
            self.initial_xy = xy.copy()
        distance = float(np.linalg.norm(xy - self.initial_xy))
        if (
            self.onset_wall_s is None
            and distance >= self.onset_distance_m
            and speed_mps >= self.minimum_speed_mps
        ):
            self.onset_wall_s, self.onset_frame = wall_s, index
            self.onset_speed_mps, self.onset_distance_actual_m = speed_mps, distance
        self.active = (
            self.condition != "baseline"
            and self.onset_wall_s is not None
            and wall_s - self.onset_wall_s < self.duration_s
        )
        if self.active:
            self.active_frames.append(index)
        self.last_index, self.last_wall_s = index, wall_s
        self.last_frame_active = self.active
        self._rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        return self.active

    def transform_sensors(self, depth, pose):
        depth, pose = np.asarray(depth), np.asarray(pose)
        if (
            self.last_index is None
            or depth.ndim != 2
            or not np.isfinite(depth).all()
            or np.any(depth < 0)
            or pose.shape != (4, 4)
            or not np.isfinite(pose).all()
        ):
            raise ValueError("fault input must be a valid current observation")
        result, measured_pose = depth.astype(float).copy(), pose.astype(float).copy()
        if self.active:
            valid = result > 0
            if self.condition == "lidar_dropout_80pct":
                result[valid & (self._rng.random(result.shape) >= 0.2)] = 0
            elif self.condition == "depth_noise_02m":
                result[valid] = np.maximum(
                    0.001, result[valid] + self._rng.normal(0, 0.2, int(valid.sum()))
                )
            elif self.condition == "pose_noise_025m_2deg":
                measured_pose[:2, 3] += self._rng.normal(0, 0.25, 2)
                yaw = self._rng.normal(0, math.radians(2))
                c, s = math.cos(yaw), math.sin(yaw)
                measured_pose[:3, :3] = (
                    np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ pose[:3, :3]
                )
        # 输入以实际保存PNG的毫米量化值为准，不让文件与模型输入出现两套深度。
        millimeters = np.clip(np.rint(result * 1000), 0, 65535).astype(np.uint16)
        return millimeters, measured_pose

    def transform_detections(self, gobs):
        if self.last_index is None:
            raise ValueError("sensor frame must precede its detector output")
        count = len(gobs["xyxy"])
        keep = (
            self._rng.random(count) < 0.7
            if self.active and self.condition == "detection_dropout_30pct"
            else np.ones(count, dtype=bool)
        )
        return retain_detections(gobs, keep), np.flatnonzero(keep).tolist()

    def summary(self):
        return dict(
            condition=self.condition,
            seed=self.seed,
            configured_duration_s=self.duration_s,
            configured_onset_distance_m=self.onset_distance_m,
            configured_minimum_speed_mps=self.minimum_speed_mps,
            onset_frame=self.onset_frame,
            onset_wall_s=self.onset_wall_s,
            onset_speed_mps=self.onset_speed_mps,
            onset_distance_m=self.onset_distance_actual_m,
            active_frames=self.active_frames,
            actual_fault_exposed=bool(self.active_frames),
            trigger_used_only_current_measured_ego_motion=True,
            unknown_exposure_not_success=True,
        )


@contextmanager
def installed_faults(policy, frontend_device="cuda:0"):
    """仅在单个子进程替换导入入口；退出恢复原函数，不写冻结源码或其他进程。"""
    from PIL import Image
    from conceptgraph.completion import interactive_environment as environment
    from conceptgraph.completion import realtime_navigation as navigation
    from conceptgraph.scripts import streamlined_detections as frontend
    from conceptgraph.completion.stage9 import digest, dump

    original_environment = environment.InteractiveRoadEnvironment
    original_vehicle = navigation.WallClockVehicle
    original_detect, original_load = (
        frontend.detect_rgb_file,
        frontend.load_detection_models,
    )
    state = {"dest": None, "blackout_performed": False}

    def event(record):
        with (state["dest"] / "fault_events.jsonl").open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    class FaultVehicle(original_vehicle):
        def __init__(self, state, path, **kwargs):
            super().__init__(state, path, **kwargs)
            dump(
                path.parent / "fault_driver_clock.json",
                dict(
                    started_monotonic_s=self.started,
                    sensor_ttl_s=self.sensor_ttl_s,
                    independent_control_thread_unchanged=True,
                ),
            )

    class FaultEnvironment(original_environment):
        def observe(self, dest, index):
            state["dest"] = dest
            active = policy.begin_frame(
                index,
                [self.state.x, self.state.y],
                self.state.speed_mps,
                time.monotonic(),
            )
            if (
                active
                and policy.condition == "sensor_blackout_4s"
                and not state["blackout_performed"]
            ):
                event(
                    dict(
                        event="blackout_begin",
                        frame_idx=index,
                        wall_s=time.monotonic(),
                        physical_speed_mps=self.state.speed_mps,
                        physical_time_s=self.state.time_s,
                    )
                )
                # 感知线程真实阻塞，独立控制线程照常运行；不补造传感器时间。
                time.sleep(4.0)
                state["blackout_performed"] = True
                event(
                    dict(event="blackout_end", frame_idx=index, wall_s=time.monotonic())
                )
            rgb, depth, pose, color, raw = super().observe(
                dest / "unmodified_sensors", index
            )
            effective_depth, effective_pose = policy.transform_sensors(depth, pose)
            colorpath, depthpath = (
                dest / "color" / f"{index:06d}.jpg",
                dest / "depth" / f"{index:06d}.png",
            )
            colorpath.parent.mkdir(exist_ok=True)
            depthpath.parent.mkdir(exist_ok=True)
            shutil.copy2(color, colorpath)
            Image.fromarray(effective_depth).save(depthpath)
            evidence = dict(
                raw,
                rgb=str(colorpath),
                rgb_sha256=digest(colorpath),
                depth=str(depthpath),
                depth_sha256=digest(depthpath),
                lidar_pixels=int(np.count_nonzero(effective_depth)),
                camera_pose_cv=effective_pose.tolist(),
                unmodified_sensor_evidence=str(
                    dest / "unmodified_sensors/sensor_evidence" / f"{index:06d}.json"
                ),
                fault_condition=policy.condition,
                fault_active=active,
                observation_started_monotonic_s=policy.last_wall_s,
                physical_pose_not_modified=True,
                model_observation_pose_may_be_perturbed=True,
            )
            dump(dest / "sensor_evidence" / f"{index:06d}.json", evidence)
            event(
                dict(
                    event="sensor",
                    frame_idx=index,
                    wall_s=time.monotonic(),
                    active=active,
                    input_depth_pixels=raw["lidar_pixels"],
                    retained_depth_pixels=evidence["lidar_pixels"],
                    position_delta_m=(effective_pose[:3, 3] - pose[:3, 3]).tolist(),
                )
            )
            return (
                rgb,
                effective_depth.astype(float) / 1000,
                effective_pose,
                colorpath,
                evidence,
            )

    def detect(cfg, models, color, classes):
        gobs, vis = original_detect(cfg, models, color, classes)
        raw_path = (
            state["dest"] / "unmodified_detections" / f"{policy.last_index:06d}.pkl.gz"
        )
        raw_path.parent.mkdir(exist_ok=True)
        with gzip.open(raw_path, "xb") as handle:
            pickle.dump(gobs, handle)
        effective, keep = policy.transform_detections(gobs)
        event(
            dict(
                event="detector",
                frame_idx=policy.last_index,
                wall_s=time.monotonic(),
                active=policy.last_frame_active,
                input_detections=len(gobs["xyxy"]),
                retained_detections=len(keep),
                retained_indices=keep,
                unmodified_detection_sha256=digest(raw_path),
                visualization_scope="unmodified detector overlay; saved gobs is the effective input",
            )
        )
        # 原可视图表示实际未删减前端；有效检测以perception/已保存gobs和保留索引为准。
        # 旧建图器将额外gobs字段当作逐检测数组；故障元数据只写独立事件日志。
        return effective, vis

    def load(cfg):
        cfg.device = frontend_device
        return original_load(cfg)

    environment.InteractiveRoadEnvironment = FaultEnvironment
    navigation.WallClockVehicle = FaultVehicle
    frontend.detect_rgb_file, frontend.load_detection_models = detect, load
    try:
        yield state
    finally:
        environment.InteractiveRoadEnvironment = original_environment
        navigation.WallClockVehicle = original_vehicle
        frontend.detect_rgb_file, frontend.load_detection_models = (
            original_detect,
            original_load,
        )
        if state["dest"] is not None:
            dump(
                state["dest"] / "fault_exposure.json",
                dict(
                    **policy.summary(),
                    actual_blocking_delay_performed=state["blackout_performed"],
                    frontend_device=frontend_device,
                    source_module_sha256=digest(__file__),
                    arrival_credit=False,
                    goal_complete=False,
                ),
            )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fault", choices=CONDITIONS, required=True)
    parser.add_argument("--fault-seed", type=int, default=17)
    parser.add_argument("--fault-duration-s", type=float, default=6.0)
    parser.add_argument(
        "--frontend-device", choices=["cuda:0", "cpu"], default="cuda:0"
    )
    args, navigation = parser.parse_known_args()
    if "--task" not in navigation or navigation.count("--task") != 1:
        raise ValueError("one fault policy per independent single-task process")
    if (
        "--seed" not in navigation
        or int(navigation[navigation.index("--seed") + 1]) != args.fault_seed
    ):
        raise ValueError("fault seed must equal the frozen physical perturbation seed")
    policy = FaultPolicy(args.fault, args.fault_seed, duration_s=args.fault_duration_s)
    from conceptgraph.completion import realtime_navigation
    from conceptgraph.completion.realtime_batch import pipeline_sources
    from conceptgraph.completion.stage9 import OUTPUT, PROJECT, dump, digest

    if "--run-id" not in navigation:
        raise ValueError("a unique run-id is required")
    run_id = navigation[navigation.index("--run-id") + 1]
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run-id must be a single directory name")
    protocol = OUTPUT / "faults" / run_id
    protocol.mkdir(exist_ok=False)
    sources = sorted(set(pipeline_sources() + [Path(__file__).resolve()]))
    dump(
        protocol / "release.json",
        dict(
            fault_arguments=vars(args),
            navigation_arguments=navigation,
            source_sha256={str(p.relative_to(PROJECT)): digest(p) for p in sources},
            synthetic_task_release_sha256=digest(
                OUTPUT / "interactive_simulation/synthetic_tasks_v1.json"
            ),
            original_frozen_sources_not_written=True,
            inference_has_no_object_GT=True,
            future_observation_used=False,
            independent_test=False,
            performance_credit=False,
            goal_complete=False,
        ),
    )

    old = sys.argv
    try:
        sys.argv = [old[0]] + navigation
        with installed_faults(policy, args.frontend_device):
            realtime_navigation.main()
        dump(
            protocol / "complete.json",
            dict(
                exposure=policy.summary(),
                navigation_root=str(OUTPUT / "interactive_simulation" / run_id),
                goal_complete=False,
            ),
        )
    except BaseException as exc:
        dump(protocol / "interrupted.json", dict(error=repr(exc), goal_complete=False))
        raise
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
