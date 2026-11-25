"""有界重放同一前端的运动关联试验；固定全范围分母和三米评价。"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path
import pickle
import shutil
import time
import traceback

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from conceptgraph.completion.fault_robustness import mapping_config
from conceptgraph.completion.quality import object_frame_metrics
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, available_development, digest, predict
from conceptgraph.completion.temporal_identity_evaluation import identity_counts, prepare_sequence
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.evidence_mapper import EvidenceAwareMapper
from conceptgraph.slam.motion_association_mapper import MotionAssociationMapper
from conceptgraph.slam.streamlined_mapping import MappingFrame
from run_mapping_iteration import dump, ratios


SETTINGS = {"baseline": {}, "motion_gate_25": {"motion_association_gate_m": 2.5},
            "motion_gate_40": {"motion_association_gate_m": 4.0}}


def summarize(records, method):
    counts = {key: sum(row["metrics"][method][key] for row in records)
              for key in ("true_positive", "truth", "predictions", "duplicate_nodes")}
    counts["center_error_sum_m"] = sum(
        m["distance_m"] for row in records for m in row["metrics"][method]["matches"])
    counts.update(identity_counts(prepare_sequence(records, method), "all_gated")["counts"])
    return dict(counts=counts, metrics=ratios(counts))


def paired_log_intervals(reports):
    logs = sorted({row["log"] for row in reports})
    grouped = {method: np.array([
        [sum(row["conditions"][method]["counts"][key] for row in reports if row["log"] == log)
         for key in ("true_positive", "truth", "predictions", "IDTP", "IDFP", "IDFN")]
        for log in logs]) for method in SETTINGS}
    draws = np.random.default_rng(20261120).integers(0, len(logs), (2000, len(logs)))
    sampled = {method: array[draws].sum(axis=1) for method, array in grouped.items()}
    def metrics(array):
        return {"recall": array[:, 0] / np.maximum(array[:, 1], 1),
                "F1": 2 * array[:, 0] / np.maximum(array[:, 1] + array[:, 2], 1),
                "IDF1": 2 * array[:, 3] / np.maximum(2 * array[:, 3] + array[:, 4] + array[:, 5], 1)}
    base = metrics(sampled["baseline"])
    return {method: {key: np.quantile(value - base[key], [.025, .975]).tolist()
                     for key, value in metrics(sampled[method]).items()}
            for method in SETTINGS if method != "baseline"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scene-count", type=int)
    parser.add_argument("--maximum-frames-per-scene", type=int)
    parser.add_argument("--wall-budget-s", type=float, default=900)
    args = parser.parse_args()
    if args.scene_count is not None and args.scene_count <= 0:
        parser.error("scene-count必须为正")
    if args.maximum_frames_per_scene is not None and args.maximum_frames_per_scene <= 0:
        parser.error("maximum-frames-per-scene必须为正")
    started = time.monotonic()
    scenes = available_development()[:args.scene_count]
    destination = experiment_directory("nuscenes", "validation", args.run_id, create=True)
    reference = PROJECT / "outputs/results/nuscenes/validation/iterations/vehicle_frontend_full_20261118_v1"
    inputs = {s["scene"]: OUTPUT / "mapping_evaluation/causal_mapping_full_001" / s["scene"] / "frames.jsonl"
              for s in scenes}
    sources = [Path(__file__).resolve(), PROJECT / "scripts/run_mapping_iteration.py",
               PROJECT / "conceptgraph/slam/motion_association_mapper.py",
               PROJECT / "conceptgraph/slam/evidence_mapper.py",
               PROJECT / "conceptgraph/slam/streamlined_mapping.py",
               PROJECT / "conceptgraph/slam/mapping.py", PROJECT / "conceptgraph/slam/utils.py",
               PROJECT / "conceptgraph/completion/association.py", PROJECT / "conceptgraph/completion/depth.py",
               PROJECT / "conceptgraph/completion/quality.py", PROJECT / "conceptgraph/completion/stage9.py",
               PROJECT / "conceptgraph/completion/temporal_identity_evaluation.py"]
    hashes = {str(p): digest(p) for p in sources}
    for source in sources:
        target = destination / "source_snapshot" / source.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    base_config = mapping_config()
    base_config.merge_with(dict(mask_conf_threshold=.1, strong_confidence=.2, weak_confirmation_observations=2))
    dump(destination / "release.json", dict(
        scenes=scenes, configurations=SETTINGS, base_config=OmegaConf.to_container(base_config),
        source_sha256=hashes, input_sha256={str(p): digest(p) for p in inputs.values()},
        reference_frames_sha256={s["scene"]: digest(reference / s["scene"] / "frames.jsonl.gz") for s in scenes},
        random_seed=17, bootstrap_seed=20261120, bootstrap_replicates=2000,
        threshold_m=3.0, wall_budget_s=args.wall_budget_s, maximum_new_gib=1, minimum_free_gib=20,
        maximum_frames_per_scene=args.maximum_frames_per_scene,
        hypothesis="同前端利用最近历史位置与真实时间差限制误配，不使用未来帧或官方答案",
        test_accessed=False, actual_GPU_calls=0, inference_GT_access=False,
        selection_policy="预先固定两种关联半径；不按成功挑场景；全量优先看F1和IDF1，召回单列"))
    reports = []
    try:
        for scene in scenes:
            name = scene["scene"]
            rows = [json.loads(line) for line in inputs[name].open()][:args.maximum_frames_per_scene]
            with gzip.open(reference / name / "frames.jsonl.gz", "rt") as stream:
                cached = {r["frame_idx"]: r for r in map(json.loads, stream)}
            mappers = {}
            for method, settings in SETTINGS.items():
                cfg = copy.deepcopy(base_config)
                cfg.merge_with(settings)
                mapper_class = EvidenceAwareMapper if method == "baseline" else MotionAssociationMapper
                mappers[method] = mapper_class(cfg, CLASSES)
            records = []
            case = destination / name
            case.mkdir()
            with gzip.open(case / "frames.jsonl.gz", "wt") as stream:
                for row in rows:
                    if time.monotonic() - started > args.wall_budget_s:
                        raise TimeoutError("达到运行预算；保留实际前缀")
                    if shutil.disk_usage(destination).free < 20 * 1024**3:
                        raise OSError("空闲空间低于20GiB")
                    idx = row["frame_idx"]
                    path = Path(cached[idx]["detections"])
                    if digest(path) != cached[idx]["detection_sha256"]:
                        raise ValueError("实际检测缓存哈希发生变化")
                    with gzip.open(path, "rb") as handle:
                        detections = pickle.load(handle)
                    image = np.asarray(Image.open(row["rgb"]).convert("RGB"))
                    depth = np.asarray(Image.open(row["depth"]), dtype=float) / 1000
                    pose, intrinsic = np.asarray(row["camera_pose"]), np.asarray(row["intrinsic"])
                    predictions, timings = {}, {}
                    for method, mapper in mappers.items():
                        np.random.seed(17 + idx)
                        then = time.monotonic()
                        mapper.update(MappingFrame(idx, image, depth, intrinsic, pose,
                                                   copy.deepcopy(detections), row["rgb"],
                                                   timestamp_s=row["camera_timestamp_s"]))
                        predictions[method] = predict(mapper.confirmed_objects(), CLASSES, pose=pose, intrinsic=intrinsic)
                        timings[method] = time.monotonic() - then
                    truth = row["evaluation_only_official"]
                    metrics = {m: object_frame_metrics(p, truth) for m, p in predictions.items()}
                    expected = cached[idx]["metrics"]["frontend_v2_evidence"]
                    for key in ("true_positive", "truth", "predictions", "duplicate_nodes"):
                        if metrics["baseline"][key] != expected[key]:
                            raise ValueError(f"历史基线逐帧未复现：{name}/{idx}/{key}")
                    record = dict(scene=name, frame_idx=idx, camera_timestamp_s=row["camera_timestamp_s"],
                                  predictions=predictions, metrics=metrics, evaluation_only_official=truth,
                                  timings_s=timings, gt_in_inference=False,
                                  input=dict(rgb=row["rgb"], depth=row["depth"], detections=str(path),
                                             detection_sha256=cached[idx]["detection_sha256"]))
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    records.append(record)
            report = dict(scene=name, log=scene["log_token"], frames=len(records),
                          conditions={m: summarize(records, m) for m in SETTINGS},
                          records_sha256=digest(case / "frames.jsonl.gz"))
            dump(case / "report.json", report)
            reports.append(report)
            print(json.dumps(dict(scene=name, frames=len(records), metrics={m: r["metrics"] for m, r in report["conditions"].items()})), flush=True)
        totals = {}
        for method in SETTINGS:
            counts = {key: sum(r["conditions"][method]["counts"][key] for r in reports)
                      for key in reports[0]["conditions"][method]["counts"]}
            totals[method] = dict(counts=counts, metrics=ratios(counts))
        if any(digest(Path(p)) != sha for p, sha in hashes.items()):
            raise ValueError("执行期间源码改变")
        dump(destination / "report.json", dict(scenes=len(reports), logs=len({r["log"] for r in reports}),
             frames=sum(r["frames"] for r in reports), conditions=totals,
             paired_log_ci95_delta=paired_log_intervals(reports),
             baseline_reproduced_every_frame=True, test_accessed=False, threshold_m=3.0,
             wall_s=time.monotonic() - started, actual_arrival_evaluated=False,
             full_validation_scope=args.scene_count is None and args.maximum_frames_per_scene is None))
        print(json.dumps(totals), flush=True)
    except BaseException as exc:
        dump(destination / "interrupted.json", dict(error=repr(exc), traceback=traceback.format_exc(),
                                                   completed_scenes=len(reports)))
        raise


if __name__ == "__main__":
    main()
