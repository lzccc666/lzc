"""完整验证范围内重放已有三维观测；轻量原型，不重新执行点云融合。"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import time

import numpy as np

from conceptgraph.completion.quality import object_frame_metrics
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, available_development, digest, dump
from conceptgraph.completion.temporal_identity_evaluation import identity_counts, prepare_sequence
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.observation_tracker import CausalObservationTracker, ObservationTrackingConfig


def ratios(c):
    return dict(recall=c["true_positive"] / c["truth"],
                precision=c["true_positive"] / max(1, c["predictions"]),
                F1=2 * c["true_positive"] / max(1, c["truth"] + c["predictions"]),
                IDF1=2 * c["IDTP"] / max(1, 2 * c["IDTP"] + c["IDFP"] + c["IDFN"]))


def summarize(records, method):
    c = {k: sum(r["metrics"][method][k] for r in records)
         for k in ("true_positive", "truth", "predictions", "duplicate_nodes")}
    c.update(identity_counts(prepare_sequence(records, method), "all_gated")["counts"])
    return dict(counts=c, metrics=ratios(c))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    dest = experiment_directory("nuscenes", "validation", args.run_id, create=True)
    reference = PROJECT / "outputs/results/nuscenes/validation/iterations/current_geometry_full_20261120_v2"
    scenes = available_development()
    configs = {"track_25": ObservationTrackingConfig(gate_m=2.5), "track_40": ObservationTrackingConfig(gate_m=4.0)}
    sources = [Path(__file__).resolve(), PROJECT / "conceptgraph/slam/observation_tracker.py",
               PROJECT / "conceptgraph/completion/association.py", PROJECT / "conceptgraph/completion/quality.py",
               PROJECT / "conceptgraph/completion/temporal_identity_evaluation.py"]
    hashes = {str(p): digest(p) for p in sources}
    inputs = {s["scene"]: reference / s["scene"] / "frames.jsonl.gz" for s in scenes}
    for source in sources:
        target = dest / "source_snapshot" / source.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    dump(dest / "release.json", dict(scenes=scenes, source_sha256=hashes,
        input_sha256={str(p): digest(p) for p in inputs.values()}, settings={k: vars(v) for k, v in configs.items()},
        seed=17, threshold_m=3.0, wall_budget_s=180, maximum_new_gib=.1, minimum_free_gib=20,
        input="已保存single_frame_low_conf三维观测；无图像/外观特征/检测置信度输入",
        test_accessed=False, pointcloud_refusion=False, actual_arrival_evaluated=False))
    reports = []
    for scene in scenes:
        name = scene["scene"]
        trackers = {k: CausalObservationTracker(v) for k, v in configs.items()}
        source_path = OUTPUT / "mapping_evaluation/causal_mapping_full_001" / name / "frames.jsonl"
        with source_path.open() as stream:
            poses = {r["frame_idx"]: (np.asarray(r["camera_pose"]), np.asarray(r["intrinsic"])) for r in map(json.loads, stream)}
        case = dest / name
        case.mkdir()
        records = []
        with gzip.open(inputs[name], "rt") as stream, gzip.open(case / "frames.jsonl.gz", "wt") as out:
            for row in map(json.loads, stream):
                if time.monotonic() - started > 180 or shutil.disk_usage(dest).free < 20 * 1024**3:
                    raise RuntimeError("达到时间或空间预算；保留前缀，不生成完整汇总")
                observations = row["predictions"]["single_frame_low_conf"]
                predictions = {"map_baseline": row["predictions"]["baseline"], "single_frame": observations}
                pose, intrinsic = poses[row["frame_idx"]]
                for method, tracker in trackers.items():
                    values = tracker.update(observations, row["camera_timestamp_s"])
                    kept = []
                    for obj in values:
                        camera = pose[:3, :3].T @ (np.asarray(obj["center"]) - pose[:3, 3])
                        uv = intrinsic @ camera
                        if 1 < camera[2] < 65 and 0 <= uv[0] / uv[2] < 1600 and 0 <= uv[1] / uv[2] < 900:
                            kept.append(obj)
                    predictions[method] = kept
                truth = row["evaluation_only_official"]
                metrics = {k: object_frame_metrics(v, truth) for k, v in predictions.items()}
                for key in ("true_positive", "truth", "predictions", "duplicate_nodes"):
                    assert metrics["map_baseline"][key] == row["metrics"]["baseline"][key]
                record = dict(scene=name, frame_idx=row["frame_idx"], camera_timestamp_s=row["camera_timestamp_s"], predictions=predictions,
                              metrics=metrics, evaluation_only_official=truth, gt_in_inference=False)
                out.write(json.dumps(record, allow_nan=False) + "\n")
                records.append(record)
        report = dict(scene=name, log=scene["log_token"], frames=len(records),
                      conditions={m: summarize(records, m) for m in records[0]["predictions"]})
        dump(case / "report.json", report)
        reports.append(report)
    totals = {}
    for method in reports[0]["conditions"]:
        c = {key: sum(r["conditions"][method]["counts"][key] for r in reports)
             for key in reports[0]["conditions"][method]["counts"]}
        totals[method] = dict(counts=c, metrics=ratios(c))
    assert all(digest(Path(p)) == sha for p, sha in hashes.items())
    dump(dest / "report.json", dict(scenes=len(reports), frames=sum(r["frames"] for r in reports),
        conditions=totals, wall_s=time.monotonic() - started, threshold_m=3.0,
        full_validation_scope=True, test_accessed=False, pointcloud_refusion=False,
        baseline_counts_reproduced_every_frame=True, actual_arrival_evaluated=False))
    print(json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()
