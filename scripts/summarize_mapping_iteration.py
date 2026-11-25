"""汇总完整38场景迭代；按采集log配对统计，不用小样本替代全量。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json

import numpy as np

from conceptgraph.completion.stage9 import OUTPUT, digest
from conceptgraph.project_paths import RESULTS_ROOT, experiment_directory


def metric_vector(c):
    tp, gt, pred, error, idtp, idfp, idfn, duplicate = np.moveaxis(c, -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack([tp / gt, tp / pred, 2 * tp / (gt + pred),
                         2 * idtp / (2 * idtp + idfp + idfn), error / tp, duplicate / gt], axis=-1)


NAMES = ["recall", "precision", "F1", "IDF1", "center_error_m", "duplicate_per_GT"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--confidence-run", default="confidence_full_20261118_v1")
    parser.add_argument("--frontend-run", default="vehicle_frontend_full_20261118_v1")
    args = parser.parse_args()
    root = RESULTS_ROOT / "nuscenes/validation/iterations"
    roots = [root / args.confidence_run, root / args.frontend_run]
    if not all((p / "report.json").exists() for p in roots):
        raise RuntimeError("两组全量报告均完成后才允许汇总")
    summary = [json.loads((p / "report.json").read_text()) for p in roots]
    if any(a["scenes"] != 38 or a["frames_each"] != 1508 for a in summary):
        raise ValueError("本轮全量应为同38场景1508帧")
    if summary[0]["baseline_TP_reproduced"] is not True:
        raise ValueError("基线没有复现，停止主比较")
    per_scene = [{p.parent.name: json.loads(p.read_text()) for p in root.glob("scene-*/report.json")} for root in roots]
    if set(per_scene[0]) != set(per_scene[1]):
        raise ValueError("两组场景不一致")
    methods = ["baseline", "low_confidence", "evidence_v2", "frontend_v2", "frontend_v2_evidence"]
    logs = sorted({r["log"] for r in per_scene[0].values()})
    counts = np.zeros((len(logs), len(methods), 8))
    scene_rows = []
    for cases in per_scene:
        for name, report in sorted(cases.items()):
            for method, values in report["conditions"].items():
                c = values["counts"]
                counts[logs.index(report["log"]), methods.index(method)] += [
                    c["true_positive"], c["truth"], c["predictions"], c["center_error_sum_m"],
                    c["IDTP"], c["IDFP"], c["IDFN"], c["duplicate_nodes"]]
                scene_rows.append(dict(scene=name, log=report["log"], method=method, **values["metrics"]))
    rng = np.random.default_rng(17)
    samples = metric_vector(counts[rng.integers(0, len(logs), (4000, len(logs)))].sum(axis=1))
    point = metric_vector(counts.sum(axis=0))
    estimates, csv_rows = {}, []
    for mi, method in enumerate(methods):
        estimates[method] = {}
        for ni, metric in enumerate(NAMES):
            differences = samples[:, mi, ni] - samples[:, 0, ni]
            finite = samples[:, mi, ni][np.isfinite(samples[:, mi, ni])]
            delta_finite = differences[np.isfinite(differences)]
            entry = dict(value=float(point[mi, ni]),
                         ci95=np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                         delta_vs_baseline=float(point[mi, ni] - point[0, ni]),
                         delta_ci95=np.quantile(delta_finite, [.025, .975]).tolist() if len(delta_finite) else None)
            estimates[method][metric] = entry
            csv_rows.append(dict(method=method, metric=metric, value=entry["value"],
                                 ci95_low=entry["ci95"][0] if entry["ci95"] else None,
                                 ci95_high=entry["ci95"][1] if entry["ci95"] else None,
                                 delta_vs_baseline=entry["delta_vs_baseline"],
                                 delta_ci95_low=entry["delta_ci95"][0] if entry["delta_ci95"] else None,
                                 delta_ci95_high=entry["delta_ci95"][1] if entry["delta_ci95"] else None))
    # 附加分组覆盖全体GT；只按同一距离/类别条件分层，不挑成功帧。
    groups = defaultdict(Counter)
    for ri, cases in enumerate(per_scene):
        for name in sorted(cases):
            source_path = OUTPUT / "mapping_evaluation/causal_mapping_full_001" / name / "frames.jsonl"
            poses = {r["frame_idx"]: np.asarray(r["camera_pose"]) for r in map(json.loads, source_path.open())}
            with gzip.open(roots[ri] / name / "frames.jsonl.gz", "rt") as stream:
                for frame in map(json.loads, stream):
                    for method in frame["predictions"]:
                        matched = {m["truth_index"] for m in frame["metrics"][method]["matches"]}
                        for ti, obj in enumerate(frame["evaluation_only_official"]):
                            distance = np.linalg.norm(np.asarray(obj["center"][:2]) - poses[frame["frame_idx"]][:2, 3])
                            labels = ["category:" + obj["category"],
                                      "range:" + ("0_20m" if distance < 20 else "20_40m" if distance < 40 else "40m_plus")]
                            if obj["category"] == "car" and distance < 20:
                                labels.append("near20m_car")
                            for label in labels:
                                groups[(method, label)]["GT"] += 1
                                groups[(method, label)]["matched"] += ti in matched
    group_rows = [dict(method=method, group=group, truth=c["GT"], matched=c["matched"],
                       recall=c["matched"] / c["GT"]) for (method, group), c in sorted(groups.items())]
    dest = experiment_directory("nuscenes", "validation", args.run_id, create=True)
    report = dict(scenes=38, logs=len(logs), frames_each=1508, conditions=estimates,
                  bootstrap=dict(unit="采集log", paired=True, repetitions=4000, seed=17),
                  sources_sha256={str(p / "report.json"): digest(p / "report.json") for p in roots},
                  baseline_TP_reproduced=True, test_accessed=False,
                  limitations=["本轮只在验证组优化，未声称独立新测试提升",
                               "frontend_v2联合改变类别提示、类别集合和分辨率，未将收益单独归因于某一项",
                               "近距离与类别分组是附加诊断，不能替代全范围主表",
                               "输出匹配比例不是单独类别分类准确率，未匹配包含关联不确定性",
                               "不是完整静态环境地图、车辆闭环或实际到达评价"])
    (dest / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for name, rows in [("metrics.csv", csv_rows), ("per_scene.csv", scene_rows), ("groups.csv", group_rows)]:
        with (dest / name).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k: {m: round(v["value"], 6) for m, v in a.items()} for k, a in estimates.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
