"""按采集日志配对重采样，保留全范围召回、F1与身份质量的取舍。"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil

import numpy as np

from conceptgraph.completion.stage9 import PROJECT, digest, dump
from conceptgraph.project_paths import experiment_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    source = experiment_directory("nuscenes", "validation", args.input_run)
    dest = experiment_directory("nuscenes", "validation", args.run_id, create=True)
    summary = json.loads((source / "report.json").read_text())
    if summary["scenes"] != 38 or summary["frames"] != 1508:
        raise ValueError("不是完整冻结验证范围")
    scenes = [json.loads(p.read_text()) for p in sorted(source.glob("scene-*/report.json"))]
    methods = list(summary["conditions"])
    logs = sorted({s["log"] for s in scenes})
    draws = np.random.default_rng(20261120).integers(0, len(logs), (2000, len(logs)))
    samples = {}
    for method in methods:
        c = np.array([[sum(s["conditions"][method]["counts"][k] for s in scenes if s["log"] == log)
                       for k in ("true_positive", "truth", "predictions", "IDTP", "IDFP", "IDFN")]
                      for log in logs])[draws].sum(axis=1)
        samples[method] = dict(recall=c[:, 0] / c[:, 1], F1=2 * c[:, 0] / (c[:, 1] + c[:, 2]),
                               IDF1=2 * c[:, 3] / (2 * c[:, 3] + c[:, 4] + c[:, 5]))
    intervals = {method: {k: (100 * np.quantile(v - samples["map_baseline"][k], [.025, .975])).tolist()
                          for k, v in samples[method].items()} for method in methods if method != "map_baseline"}
    groups = {}
    decomposition = dict(both_correct=0, observation_only=0, map_only=0, neither_correct=0)
    for path in sorted(source.glob("scene-*/frames.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            for row in map(json.loads, stream):
                a = {m["truth_index"] for m in row["metrics"]["single_frame"]["matches"]}
                b = {m["truth_index"] for m in row["metrics"]["map_baseline"]["matches"]}
                decomposition["both_correct"] += len(a & b)
                decomposition["observation_only"] += len(a - b)
                decomposition["map_only"] += len(b - a)
                decomposition["neither_correct"] += len(row["evaluation_only_official"]) - len(a | b)
                for method in methods:
                    matches = {m["truth_index"] for m in row["metrics"][method]["matches"]}
                    for i, obj in enumerate(row["evaluation_only_official"]):
                        group = groups.setdefault((method, obj["category"]), dict(truth=0, matched=0))
                        group["truth"] += 1
                        group["matched"] += i in matches
    with (dest / "category_recall.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "category", "truth", "matched", "recall"])
        for (method, category), group in sorted(groups.items()):
            writer.writerow([method, category, group["truth"], group["matched"], group["matched"] / group["truth"]])
    snapshot = dest / "source_snapshot/scripts"
    snapshot.mkdir(parents=True)
    shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
    dump(dest / "report.json", dict(input_run=str(source), input_report_sha256=digest(source / "report.json"),
        script_sha256=digest(Path(__file__)), scenes=len(scenes), logs=len(logs), frames=1508,
        threshold_m=3.0, seed=20261120, bootstrap_replicates=2000,
        full_scope_metrics=summary["conditions"], paired_log_ci95_delta_percentage_points=intervals,
        observation_map_diagnostic=decomposition,
        diagnostic_caveat="单帧3D观测与地图的可配对性对照；不是纯关联因果损失或可达到的理论上界",
        promote_default=False, reason="召回增加但身份质量下降；未证明整体改进",
        pointcloud_refusion=False, test_accessed=False, actual_arrival_evaluated=False,
        protected_reference_sha256={str(p): digest(p) for p in [
            PROJECT / "MEMORY.md", PROJECT / "PAPER_MEMORY.md",
            PROJECT / "outputs/results/nuscenes/validation/iterations/vehicle_frontend_full_20261118_v1/report.json",
            PROJECT / "outputs/results/nuscenes/validation/iterations/current_geometry_full_20261120_v2/report.json"]}))
    print(json.dumps(dict(intervals=intervals, decomposition=decomposition)), flush=True)


if __name__ == "__main__":
    main()
