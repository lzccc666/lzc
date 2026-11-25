"""区分逐帧召回与整段独立对象覆盖；仅补充指标，不修改原评分。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from conceptgraph.completion.stage9 import digest
from conceptgraph.project_paths import RESULTS_ROOT, experiment_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = RESULTS_ROOT / "nuscenes/validation/iterations" / args.input_run
    summary = json.loads((root / "report.json").read_text())
    methods = list(summary["conditions"])
    paths = sorted(root.glob("scene-*/frames.jsonl.gz"))
    if len(paths) != summary["scenes"]:
        raise ValueError("逐场景记录不完整")
    objects = {m: defaultdict(lambda: dict(visible=0, matched=0, scene_frames=defaultdict(list))) for m in methods}
    for path in paths:
        with gzip.open(path, "rt") as stream:
            for frame in map(json.loads, stream):
                for method in methods:
                    associations = {m["truth_index"]: frame["predictions"][method][m["prediction_index"]]["node_id"]
                                    for m in frame["metrics"][method]["matches"]}
                    for ti, obj in enumerate(frame["evaluation_only_official"]):
                        record = objects[method][obj["instance_token"]]
                        record["category"] = obj["category"]
                        record["visible"] += 1
                        record["matched"] += ti in associations
                        record["scene_frames"][frame["scene"]].append((frame["frame_idx"], associations.get(ti)))
    output_rows, results = [], {}
    for method, values in objects.items():
        counts = Counter()
        recalls = []
        for instance, record in values.items():
            best = 0
            for frames in record["scene_frames"].values():
                last_frame, last_node, streak = -2, None, 0
                for frame, node in sorted(frames):
                    if node is None:
                        streak = 0
                    elif frame == last_frame + 1 and node == last_node:
                        streak += 1
                    else:
                        streak = 1
                    best = max(best, streak)
                    last_frame, last_node = frame, node
            counts["objects"] += 1
            counts["ever_recovered"] += record["matched"] > 0
            counts["same_identity_three_consecutive_frames"] += best >= 3
            counts["GT_observations"] += record["visible"]
            counts["matched_observations"] += record["matched"]
            recalls.append(record["matched"] / record["visible"])
            output_rows.append(dict(method=method, instance_token=instance, category=record["category"],
                                    scenes=len(record["scene_frames"]), visible_frames=record["visible"],
                                    matched_frames=record["matched"], best_same_identity_streak=best,
                                    fraction_correct_frames=recalls[-1]))
        results[method] = dict(counts=counts,
                               frame_recall=counts["matched_observations"] / counts["GT_observations"],
                               unique_object_ever_coverage=counts["ever_recovered"] / counts["objects"],
                               unique_object_three_frame_coverage=counts["same_identity_three_consecutive_frames"] / counts["objects"],
                               object_balanced_frame_recall=float(np.mean(recalls)))
        expected = summary["conditions"][method]["counts"]
        if counts["GT_observations"] != expected["truth"] or counts["matched_observations"] != expected["true_positive"]:
            raise ValueError("补充指标和原逐帧总数不一致")
    dest = experiment_directory("nuscenes", "validation", args.run_id, create=True)
    report = dict(source=str(root), scenes=len(paths), conditions=results,
                  definitions={
                      "frame_recall": "每帧正确图节点数累计/每帧GT对象数累计，原主指标不变",
                      "unique_object_ever_coverage": "至少一次正确匹配的官方独立instance数/出现过的全部官方instance数",
                      "unique_object_three_frame_coverage": "至少连续3帧以同一图节点正确对应的独立instance比例",
                      "object_balanced_frame_recall": "先算每个独立instance正确帧比例，再对instance等权平均"},
                  sources_sha256={str(p): digest(p) for p in paths},
                  source_sha256=digest(Path(__file__)), original_metric_changed=False,
                  limitations=["曾经覆盖不代表每帧可用，更不代表正确到达",
                               "独立身份仍来自官方类别/三米XY关联，非人工逐像素核验",
                               "新补充指标在发现问题后增加；不能冒充此前预注册主指标"],
                  test_accessed=False)
    (dest / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (dest / "per_object.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
