"""汇总同38验证场景的RAM/DINO前端比较，保留推荐门限和共享门限全部结果。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json

import numpy as np

from conceptgraph.completion.ram_dino_frontend import canonical_category
from conceptgraph.completion.stage9 import OUTPUT, digest
from conceptgraph.project_paths import RESULTS_ROOT, experiment_directory
from summarize_mapping_iteration import metric_vector, NAMES


def save_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dino-run", default="ram_dino_full_20261118_v3")
    parser.add_argument("--ram-run", default="ram_dino_full_20261118_v3")
    args = parser.parse_args()
    root = RESULTS_ROOT / "nuscenes/validation/iterations"
    assignments = [(root / "confidence_full_20261118_v1", ["baseline"]),
                   (root / "vehicle_frontend_full_20261118_v1", ["frontend_v2_evidence"]),
                   (root / args.dino_run, ["dino_fixed_native", "dino_fixed_shared"]),
                   (root / args.ram_run, ["ram_dino_native", "ram_dino_shared"])]
    assignments = [(base, [m for m in chosen if m in json.loads((base / "report.json").read_text())["conditions"]])
                   for base, chosen in assignments]
    if any(not chosen for _, chosen in assignments):
        raise ValueError("比较方法缺失")
    methods = sum([v for _, v in assignments], [])
    scene_sets, reports = [], {}
    for base, chosen in assignments:
        report = json.loads((base / "report.json").read_text())
        if report["scenes"] != 38 or report["frames_each"] != 1508:
            raise ValueError("不允许把小批结果混入完整主比较")
        scenes = {p.parent.name: json.loads(p.read_text()) for p in base.glob("scene-*/report.json")}
        scene_sets.append(set(scenes))
        for method in chosen:
            reports[method] = scenes
    if not all(s == scene_sets[0] for s in scene_sets):
        raise ValueError("场景清单不一致")
    logs = sorted({r["log"] for r in reports["baseline"].values()})
    counts = np.zeros((len(logs), len(methods), 8))
    scene_rows = []
    for method, cases in reports.items():
        for name, report in sorted(cases.items()):
            values = report["conditions"][method]
            c = values["counts"]
            if sum(r["conditions"][method]["counts"]["truth"] for r in cases.values()) != 5723:
                raise ValueError("标准答案分母变化")
            counts[logs.index(report["log"]), methods.index(method)] += [
                c["true_positive"], c["truth"], c["predictions"], c["center_error_sum_m"],
                c["IDTP"], c["IDFP"], c["IDFN"], c["duplicate_nodes"]]
            scene_rows.append(dict(scene=name, log=report["log"], method=method, **values["metrics"]))
    rng = np.random.default_rng(17)
    samples = metric_vector(counts[rng.integers(0, len(logs), (4000, len(logs)))].sum(axis=1))
    point = metric_vector(counts.sum(axis=0))
    reference = methods.index("frontend_v2_evidence")
    estimates, metric_rows = {}, []
    for mi, method in enumerate(methods):
        estimates[method] = {}
        for ni, metric in enumerate(NAMES):
            difference = samples[:, mi, ni] - samples[:, reference, ni]
            finite = samples[:, mi, ni][np.isfinite(samples[:, mi, ni])]
            delta_finite = difference[np.isfinite(difference)]
            entry = dict(value=float(point[mi, ni]), ci95=np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                         delta_vs_current_yolo=float(point[mi, ni] - point[reference, ni]),
                         delta_ci95=np.quantile(delta_finite, [.025, .975]).tolist() if len(delta_finite) else None)
            estimates[method][metric] = entry
            metric_rows.append(dict(method=method, metric=metric, value=entry["value"],
                                    delta_vs_current_yolo=entry["delta_vs_current_yolo"],
                                    ci95_low=entry["ci95"][0] if entry["ci95"] else None,
                                    ci95_high=entry["ci95"][1] if entry["ci95"] else None,
                                    delta_ci95_low=entry["delta_ci95"][0] if entry["delta_ci95"] else None,
                                    delta_ci95_high=entry["delta_ci95"][1] if entry["delta_ci95"] else None))
    groups, timings, prompt_audit = defaultdict(Counter), defaultdict(list), Counter()
    frame_pairs = defaultdict(set)
    for base, chosen in assignments:
        for name in sorted(scene_sets[0]):
            source = OUTPUT / "mapping_evaluation/causal_mapping_full_001" / name / "frames.jsonl"
            poses = {r["frame_idx"]: np.asarray(r["camera_pose"]) for r in map(json.loads, source.open())}
            with gzip.open(base / name / "frames.jsonl.gz", "rt") as stream:
                for frame in map(json.loads, stream):
                    for method in chosen:
                        frame_pairs[method].add((name, frame["frame_idx"]))
                        matched = {m["truth_index"] for m in frame["metrics"][method]["matches"]}
                        for ti, obj in enumerate(frame["evaluation_only_official"]):
                            distance = np.linalg.norm(np.asarray(obj["center"][:2]) - poses[frame["frame_idx"]][:2, 3])
                            for label in ["category:" + obj["category"],
                                          "range:" + ("0_20m" if distance < 20 else "20_40m" if distance < 40 else "40m_plus")]:
                                groups[(method, label)]["GT"] += 1
                                groups[(method, label)]["matched"] += ti in matched
                    if base.name == "vehicle_frontend_full_20261118_v1":
                        timings["yolo_six_960"].append(frame["frontend_s"])
                    elif any(m.startswith(("ram_dino_", "dino_fixed_")) for m in chosen):
                        for mode, value in frame["frontend_s"].items():
                            if any(m.startswith(mode + "_") for m in chosen):
                                timings[mode].append(value)
    if any(pairs != frame_pairs["baseline"] for pairs in frame_pairs.values()):
        raise ValueError("参与评价的具体帧不一致")
    # 逐帧输出中的自动类别完整保留；这里只读取模型生成的诊断，不改变推理结果。
    import pickle
    for path in sorted((root / args.ram_run).glob("scene-*/frames.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            for frame in map(json.loads, stream):
                dp = frame["detections"]["ram_dino"]
                with gzip.open(dp, "rb") as det_stream:
                    d = pickle.load(det_stream)["frame_provenance"]["diagnostics"]
                prompt_audit["frames"] += 1
                prompt_audit["tokens_exceed_256"] += d["prompt_token_count"] > d["max_text_len"]
                prompt_audit["unknown_or_out_of_scope_boxes"] += d["raw_unknown_or_out_of_scope_count"]
                available = {canonical_category(c) for c in d["detector_classes"]}
                for obj in frame["evaluation_only_official"]:
                    prompt_audit["GT_observations"] += 1
                    prompt_audit["GT_category_present_in_RAM_prompt"] += obj["category"] in available
    group_rows = [dict(method=method, group=group, truth=c["GT"], matched=c["matched"], recall=c["matched"] / c["GT"])
                  for (method, group), c in sorted(groups.items())]
    timing_rows = [dict(frontend=mode, frames=len(values), mean_s=float(np.mean(values)),
                        median_s=float(np.median(values)), p95_s=float(np.quantile(values, .95)),
                        equivalent_serial_fps=1 / float(np.mean(values))) for mode, values in timings.items()]
    if any(r["frames"] != 1508 for r in timing_rows):
        raise ValueError("耗时样本数量应与实际帧数一致")
    dest = experiment_directory("nuscenes", "validation", args.run_id, create=True)
    report = dict(scenes=38, frames_each=1508, GT_observations=5723, logs=len(logs), conditions=estimates,
                  bootstrap=dict(unit="采集log", repetitions=4000, paired=True, seed=17),
                  timing=timing_rows, RAM_prompt_diagnostic=prompt_audit,
                  sources_sha256={str(base / "report.json"): digest(base / "report.json") for base, _ in assignments},
                  test_accessed=False, actual_arrival_evaluated=False, full_ConceptGraphs_reproduction=False,
                  limitations=["比较的是前端适配，不是原论文完整方法复现或官方排行榜",
                               "RAM自动标签未偷偷补充固定六类，词义只按预定同义词表映射；不确定类别保留未知",
                               "DINO使用官方800/1333预处理，YOLO使用960；不同输入大小作为速度质量权衡同时披露",
                               "DINO框分数与YOLO置信度不完全同标定；主表完整跑推荐0.25，共享0.1完整二维对照另附，三维前缀不能冒充全量",
                               "所有方法仅评价六类交通参与者，未评价完整道路环境或导航到达",
                               "耗时为本机实际串行前端记录，含读图和分割特征；不是无I/O独立实时闭环FPS",
                               "RAM类别是否出现在提示中的统计是类别名诊断，不是逐对象检测正确率"])
    (dest / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    save_csv(dest / "metrics.csv", metric_rows)
    save_csv(dest / "per_scene.csv", scene_rows)
    save_csv(dest / "groups.csv", group_rows)
    save_csv(dest / "frontend_timing.csv", timing_rows)
    labels = dict(baseline="原方法YOLO", frontend_v2_evidence="当前改进YOLO",
                  dino_fixed_native="固定六类DINO，推荐0.25", dino_fixed_shared="固定六类DINO，共享0.1",
                  ram_dino_native="RAM自动类别+DINO，推荐0.25", ram_dino_shared="RAM自动类别+DINO，共享0.1")
    lines = ["# RAM / Grounding DINO真实连续建图对照", "",
             "相同38个nuScenes验证场景、1508帧、5723次标准对象出现；相同MobileSAM、CLIP、真实距离和位姿；相同三米同类别一对一评价。未使用测试场景。", "",
             "| 方法 | 建图召回↑ | 输出匹配比例↑ | F1↑ | 跨帧身份IDF1↑ | 匹配对象平均水平位置误差↓ |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in methods:
        v = estimates[method]
        lines.append("| " + labels[method] + " | " + " | ".join(f"{100 * v[k]['value']:.2f}%" for k in NAMES[:4])
                     + f" | {v['center_error_m']['value']:.3f}米 |")
    lines += ["", "输出匹配比例包括位置、类别和关联是否符合本次评价，并非纯分类准确率。共有六类交通参与者，非完整道路环境或实际到达评价。", "",
              "## 与当前YOLO比较的不确定性", "", "以11个采集日志为单位配对重采样4000次，不把相邻帧视为独立样本。"]
    for method in methods[2:]:
        v = estimates[method]["recall"]
        lo, hi = v["delta_ci95"]
        lines.append(f"- {labels[method]}：召回变化{100*v['delta_vs_current_yolo']:+.2f}个百分点，95%区间[{100*lo:+.2f}, {100*hi:+.2f}]。")
    lines += ["", "## 真实完整前端耗时", "", "包含读图、检测、分割和CLIP特征，不含后续建图。模型输入不同：YOLO为960；DINO为官方800短边／1333最长边。此表不能直接解释成车辆实时闭环FPS。", "",
              "| 前端 | 帧数 | 平均每帧秒 | 中位数秒 | 95%分位秒 |", "|---|---:|---:|---:|---:|"]
    for row in timing_rows:
        lines.append(f"| {row['frontend']} | {row['frames']} | {row['mean_s']:.3f} | {row['median_s']:.3f} | {row['p95_s']:.3f} |")
    lines += ["", "## 自动标签诊断", "", f"RAM提示超过256 token的帧数：{prompt_audit['tokens_exceed_256']}/1508。",
              f"标准对象的类别名出现在RAM提示中的次数：{prompt_audit['GT_category_present_in_RAM_prompt']}/{prompt_audit['GT_observations']}。这是类别名可用性，不是逐对象检测成功率。", "",
              "RAM没有被偷偷补入固定六类；固定类别DINO是独立对照。car door、bus stop等部件／场所不当作车辆，van/SUV按nuScenes定义归入car；混合短语按模型对完整提示词的分数选择，不按第一个词强行归类。", "",
              "## 使用边界和文件", "",
              "- 主表采用完整运行的推荐0.25配置；共享0.1的完整154帧二维探查和部分三维记录另存。不能把前缀混入完整三维主表。不同检测器分数不完全同标定。",
              "- 这是RAM/Grounding DINO前端接入同一建图系统的比较，不是ConceptGraphs原论文完整复现。",
              "- metrics.csv：指标与区间；per_scene.csv：每场景结果；groups.csv：类别和距离；frontend_timing.csv：耗时。",
              "- 逐帧检测、分割、模型提示和三维地图在原始run目录；模型权重、输入清单、源码快照和哈希在release.json。",
              "- 实验只支持对象地图结论，没有新的Qwen训练、导航到达或车辆闭环成绩。", ""]
    (dest / "说明.md").write_text("\n".join(lines))
    print(json.dumps({m: {k: round(v["value"], 6) for k, v in values.items()} for m, values in estimates.items()},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
