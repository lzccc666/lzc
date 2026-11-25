"""实际运行新版GPU感知和逐帧建图，输出到新目录并保留原始失败证据。"""
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
import torch

from conceptgraph.completion.fault_robustness import mapping_config
from conceptgraph.completion.quality import object_frame_metrics
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, available_development, digest, predict
from conceptgraph.completion.temporal_identity_evaluation import identity_counts, prepare_sequence
from conceptgraph.completion.vehicle_frontend import CLASSES, PROMPTS, configure_vehicle_prompts, detect_vehicle_frame
from conceptgraph.project_paths import experiment_directory
from conceptgraph.scripts.streamlined_detections import load_detection_models
from conceptgraph.slam.evidence_mapper import EvidenceAwareMapper
from conceptgraph.slam.streamlined_mapping import MappingFrame, StreamlinedMapper
from run_mapping_iteration import ratios


def dump(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--wall-budget-s", type=float, default=1800)
    args = parser.parse_args()
    scenes = [s for s in available_development() if not args.scene or s["scene"] in args.scene]
    if not scenes or (args.scene and set(args.scene) != {s["scene"] for s in scenes}):
        raise ValueError("只在验证／调试场景迭代")
    dest = experiment_directory("nuscenes", "validation", args.run_id, create=True)
    methods = ["frontend_v2", "frontend_v2_evidence"]
    begun = time.monotonic()
    sources = [Path(__file__).resolve(), PROJECT / "conceptgraph/completion/vehicle_frontend.py",
               PROJECT / "conceptgraph/slam/evidence_mapper.py", PROJECT / "scripts/run_mapping_iteration.py",
               PROJECT / "conceptgraph/scripts/streamlined_detections.py", PROJECT / "conceptgraph/slam/utils.py",
               PROJECT / "conceptgraph/slam/streamlined_mapping.py", PROJECT / "conceptgraph/slam/mapping.py"]
    frozen = {str(p): digest(p) for p in sources}
    snapshot = dest / "source_snapshot"
    for path in sources:
        target = snapshot / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    configs = {k: mapping_config() for k in methods}
    configs["frontend_v2_evidence"].merge_with(dict(mask_conf_threshold=0.1, strong_confidence=0.2,
                                                  weak_confirmation_observations=2))
    dump(dest / "release.json", dict(
        scenes=scenes, methods=methods, detector_prompts=PROMPTS, canonical_classes=CLASSES,
        image_size=960, detector_confidence=0.1, mapping_configs={k: OmegaConf.to_container(v) for k, v in configs.items()},
        sources_sha256=frozen, input_root="dataset/processed/nuscenes/validation/converted",
        weights_sha256={name: digest(PROJECT / "checkpoint" / name)
                        for name in ["yolov8l-world.pt", "mobile_sam.pt"]},
        evaluation_threshold_m=3.0, seed=17, wall_budget_s=args.wall_budget_s,
        minimum_free_gib=20, new_storage_budget_gib=4, inference_GT_access=False,
        test_accessed=False, note="改变感知类别提示与分辨率；不与同前端建图消融混称；运行耗时单列"))
    reports = []
    try:
        models = load_detection_models(type("Cfg", (), dict(device="cuda:0"))())
        configure_vehicle_prompts(models)
        total_bytes = 0
        for scene in scenes:
            source = OUTPUT / "mapping_evaluation/causal_mapping_full_001" / scene["scene"] / "frames.jsonl"
            rows = [json.loads(line) for line in source.open()]
            mappers = {k: (EvidenceAwareMapper if k.endswith("evidence") else StreamlinedMapper)(copy.deepcopy(v), CLASSES)
                       for k, v in configs.items()}
            case = dest / scene["scene"]
            (case / "detections").mkdir(parents=True)
            records = []
            with gzip.open(case / "frames.jsonl.gz", "wt") as output:
                for row in rows:
                    if time.monotonic() - begun > args.wall_budget_s:
                        raise TimeoutError("达到GPU与建图迭代预算，保留结果")
                    if shutil.disk_usage(dest).free < 20 * 1024**3 or total_bytes > 4 * 1024**3:
                        raise OSError("达到磁盘保护线或事前4GiB配额")
                    idx = row["frame_idx"]
                    start = time.monotonic()
                    detections = detect_vehicle_frame(models, row["rgb"])
                    torch.cuda.synchronize()
                    frontend_s = time.monotonic() - start
                    detection_path = case / "detections" / f"{idx:06d}.pkl.gz"
                    with gzip.open(detection_path, "wb") as stream:
                        pickle.dump(detections, stream)
                    total_bytes += detection_path.stat().st_size
                    image = np.asarray(Image.open(row["rgb"]).convert("RGB"))
                    depth = np.asarray(Image.open(row["depth"]), dtype=float) / 1000.0
                    pose, intrinsic = np.asarray(row["camera_pose"]), np.asarray(row["intrinsic"])
                    predictions, times = {}, {}
                    for method, mapper in mappers.items():
                        np.random.seed(17 + idx)
                        start = time.monotonic()
                        mapper.update(MappingFrame(idx, image, depth, intrinsic, pose, copy.deepcopy(detections),
                                                   row["rgb"], timestamp_s=row["camera_timestamp_s"]))
                        objects = mapper.confirmed_objects() if method.endswith("evidence") else mapper.objects
                        predictions[method] = predict(objects, CLASSES, pose=pose, intrinsic=intrinsic)
                        times[method] = time.monotonic() - start
                    truth = row["evaluation_only_official"]
                    full = dict(scene=scene["scene"], frame_idx=idx, camera_timestamp_s=row["camera_timestamp_s"],
                                rgb=row["rgb"], depth=row["depth"], detections=str(detection_path),
                                detection_sha256=digest(detection_path), predictions=predictions,
                                metrics={k: object_frame_metrics(v, truth) for k, v in predictions.items()},
                                evaluation_only_official=truth, gt_in_inference=False,
                                frontend_s=frontend_s, mapping_s=times)
                    output.write(json.dumps(full, ensure_ascii=False, allow_nan=False) + "\n")
                    output.flush()
                    records.append(full)
            condition_reports = {}
            for method in methods:
                counts = {key: sum(r["metrics"][method][key] for r in records)
                          for key in ["true_positive", "truth", "predictions", "duplicate_nodes"]}
                counts["center_error_sum_m"] = sum(m["distance_m"] for r in records for m in r["metrics"][method]["matches"])
                counts.update(identity_counts(prepare_sequence(records, method), "all_gated")["counts"])
                condition_reports[method] = dict(counts=counts, metrics=ratios(counts))
            report = dict(scene=scene["scene"], log=scene["log_token"], frames=len(records), conditions=condition_reports,
                          frames_sha256=digest(case / "frames.jsonl.gz"),
                          frontend_seconds=sum(r["frontend_s"] for r in records))
            dump(case / "report.json", report)
            reports.append(report)
            print(json.dumps(dict(scene=scene["scene"], metrics={k: v["metrics"] for k, v in condition_reports.items()},
                                  total_sensor_bytes=total_bytes)), flush=True)
        totals = {}
        for method in methods:
            counts = {k: sum(r["conditions"][method]["counts"][k] for r in reports)
                      for k in reports[0]["conditions"][method]["counts"]}
            totals[method] = dict(counts=counts, metrics=ratios(counts))
        if any(digest(Path(p)) != h for p, h in frozen.items()):
            raise ValueError("源码发生变化，保留结果但不封为完整运行")
        dump(dest / "report.json", dict(scenes=len(scenes), frames_each=sum(r["frames"] for r in reports),
                                        conditions=totals, raw_detection_bytes=total_bytes,
                                        actual_GPU_frontend_calls=sum(r["frames"] for r in reports),
                                        wall_s=time.monotonic() - begun, test_accessed=False,
                                        actual_arrival_evaluated=False, GT_in_inference=False))
        print(json.dumps(totals, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        dump(dest / "interrupted.json", dict(error=repr(exc), traceback=traceback.format_exc(), completed_scenes=len(reports)))
        raise


if __name__ == "__main__":
    main()
