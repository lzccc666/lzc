"""新目录下有界运行同输入建图迭代，保留逐帧预测与全分母指标。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gzip
import json
from pathlib import Path
import pickle
import resource
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
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.evidence_mapper import EvidenceAwareMapper
from conceptgraph.slam.streamlined_mapping import MappingFrame, StreamlinedMapper


SETTINGS = {
    "baseline": {"mask_conf_threshold": 0.2},
    "low_confidence": {"mask_conf_threshold": 0.1},
    "evidence_v2": {"mask_conf_threshold": 0.1, "strong_confidence": 0.2,
                    "weak_confirmation_observations": 2},
}


def dump(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def ratios(counts):
    tp = counts["true_positive"]
    precision = tp / counts["predictions"] if counts["predictions"] else 0.0
    recall = tp / counts["truth"] if counts["truth"] else 0.0
    return dict(recall=recall, precision=precision,
                F1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                IDF1=2 * counts["IDTP"] / (2 * counts["IDTP"] + counts["IDFP"] + counts["IDFN"])
                if counts["IDTP"] else 0.0,
                center_error_m=counts["center_error_sum_m"] / tp if tp else None,
                duplicate_per_GT=counts["duplicate_nodes"] / counts["truth"] if counts["truth"] else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--conditions", nargs="+", choices=SETTINGS, default=list(SETTINGS))
    parser.add_argument("--wall-budget-s", type=float, default=1800)
    args = parser.parse_args()
    scenes = [s for s in available_development() if not args.scene or s["scene"] in args.scene]
    if not scenes or (args.scene and set(args.scene) != {s["scene"] for s in scenes}):
        raise ValueError("仅允许已有验证／调试场景，不使用测试集选方法")
    dest = experiment_directory("nuscenes", "validation", args.run_id, create=True)
    source_root = OUTPUT / "mapping_evaluation/causal_mapping_full_001"
    inputs = [source_root / s["scene"] / "frames.jsonl" for s in scenes]
    configs = {}
    for condition in args.conditions:
        cfg = mapping_config()
        cfg.merge_with(SETTINGS[condition])
        configs[condition] = cfg
    sources = [Path(__file__).resolve(), PROJECT / "conceptgraph/project_paths.py",
               PROJECT / "conceptgraph/slam/evidence_mapper.py",
               PROJECT / "conceptgraph/slam/streamlined_mapping.py",
               PROJECT / "conceptgraph/slam/utils.py", PROJECT / "conceptgraph/slam/mapping.py"]
    frozen = {str(p): digest(p) for p in sources}
    (dest / "source_snapshot").mkdir()
    for path in sources:
        target = dest / "source_snapshot" / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    dump(dest / "release.json", dict(
        scenes=scenes, conditions=args.conditions, config={k: OmegaConf.to_container(v) for k, v in configs.items()},
        sources_sha256=frozen, inputs_sha256={str(p): digest(p) for p in inputs},
        threshold_m=3.0, seed=17, wall_budget_s=args.wall_budget_s, minimum_free_gib=20,
        planned_maximum_new_bytes=2 * 1024**3, actual_new_GPU_calls=0,
        inference_GT_access=False, same_frontend=True, test_accessed=False,
        scope="真实感知输入逐帧重建；验证集方法迭代，不是新独立测试或导航到达"))
    begun = time.monotonic()
    reports = []
    try:
        for scene, path in zip(scenes, inputs):
            sources_rows = [json.loads(line) for line in path.open()]
            class_path = Path(sources_rows[0]["detections"]).parents[2] / "gsa_classes_streamlined_yolo.json"
            classes = json.loads(class_path.read_text())
            mappers = {k: (EvidenceAwareMapper if k == "evidence_v2" else StreamlinedMapper)(copy.deepcopy(cfg), classes)
                       for k, cfg in configs.items()}
            records = {k: [] for k in mappers}
            output = dest / scene["scene"]
            output.mkdir()
            with gzip.open(output / "frames.jsonl.gz", "wt") as handle:
                for row in sources_rows:
                    if time.monotonic() - begun > args.wall_budget_s:
                        raise TimeoutError("达到事前运行预算，保留已完成记录")
                    if shutil.disk_usage(dest).free < 20 * 1024**3:
                        raise OSError("空闲磁盘低于20GiB")
                    index = row["frame_idx"]
                    image = np.asarray(Image.open(row["rgb"]).convert("RGB"))
                    depth = np.asarray(Image.open(row["depth"]), dtype=float) / 1000.0
                    pose, intrinsic = np.asarray(row["camera_pose"]), np.asarray(row["intrinsic"])
                    with gzip.open(row["detections"], "rb") as stream:
                        detections = pickle.load(stream)
                    predictions, times = {}, {}
                    for condition, mapper in mappers.items():
                        np.random.seed(17 + index)
                        start = time.monotonic()
                        mapper.update(MappingFrame(index, image, depth, intrinsic, pose,
                                                   copy.deepcopy(detections), row["rgb"],
                                                   timestamp_s=row["camera_timestamp_s"]))
                        objects = mapper.confirmed_objects() if condition == "evidence_v2" else mapper.objects
                        predictions[condition] = predict(objects, classes, pose=pose, intrinsic=intrinsic)
                        times[condition] = time.monotonic() - start
                    # 各方法推理完毕才读取监督答案；推理函数从不接收这些字段。
                    truth = row["evaluation_only_official"]
                    metrics = {k: object_frame_metrics(v, truth) for k, v in predictions.items()}
                    full = dict(scene=scene["scene"], frame_idx=index,
                                camera_timestamp_s=row["camera_timestamp_s"], rgb=row["rgb"],
                                depth=row["depth"], detections=row["detections"],
                                predictions=predictions, metrics=metrics, evaluation_only_official=truth,
                                elapsed_s=times, gt_in_inference=False)
                    handle.write(json.dumps(full, ensure_ascii=False, allow_nan=False) + "\n")
                    for condition in mappers:
                        records[condition].append(full)
            scene_report = {}
            for condition, saved in records.items():
                counts = {key: sum(r["metrics"][condition][key] for r in saved)
                          for key in ["true_positive", "truth", "predictions", "duplicate_nodes"]}
                counts["center_error_sum_m"] = sum(
                    m["distance_m"] for r in saved for m in r["metrics"][condition]["matches"])
                counts.update(identity_counts(prepare_sequence(saved, condition), "all_gated")["counts"])
                scene_report[condition] = dict(counts=counts, metrics=ratios(counts),
                                               mapping_seconds=sum(r["elapsed_s"][condition] for r in saved))
            report = dict(scene=scene["scene"], log=scene["log_token"], frames=len(sources_rows),
                          conditions=scene_report, records_sha256=digest(output / "frames.jsonl.gz"))
            dump(output / "report.json", report)
            reports.append(report)
            print(json.dumps(dict(scene=scene["scene"], metrics={k: v["metrics"] for k, v in scene_report.items()})), flush=True)
        totals = {}
        for condition in args.conditions:
            counts = defaultdict(float)
            for report in reports:
                for key, value in report["conditions"][condition]["counts"].items():
                    counts[key] += value
            totals[condition] = dict(counts=dict(counts), metrics=ratios(counts))
        reference = json.loads((PROJECT / "outputs/results/nuscenes/validation/statistics/report.json").read_text())
        all_scenes = len(scenes) == reference["scenes"]
        baseline_reproduced = None
        if all_scenes and "baseline" in totals:
            baseline_reproduced = int(totals["baseline"]["counts"]["true_positive"]) == int(reference["conditions"]["full"]["counts"]["TP"])
        if any(digest(Path(p)) != h for p, h in frozen.items()):
            raise ValueError("运行期间源码发生变化，不能封为完整记录")
        dump(dest / "report.json", dict(
            scenes=len(scenes), logs=len({s["log_token"] for s in scenes}),
            frames_each=sum(r["frames"] for r in reports), conditions=totals,
            baseline_TP_reproduced=baseline_reproduced, test_accessed=False,
            source_fingerprints_preserved=True, wall_s=time.monotonic() - begun,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            actual_new_GPU_calls=0, actual_arrival_evaluated=False))
        print(json.dumps(totals, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        dump(dest / "interrupted.json", dict(error=repr(exc), traceback=traceback.format_exc(),
                                             completed_scenes=len(reports)))
        raise


if __name__ == "__main__":
    main()
