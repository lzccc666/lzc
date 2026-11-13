"""独立报告实际类别与正GT覆盖，避免三个窗口被当作三个正场景。"""

from collections import Counter
import gzip
import json
import pickle
from conceptgraph.completion.stage9 import OUTPUT, digest, dump

ROOT = OUTPUT / "statistics/kitti_tracking_full_001"
DEST = ROOT / "scope_audit_002"
DEST.mkdir(exist_ok=False)
agent = json.loads(
    (
        OUTPUT / "data_manifest/kitti_tracking_center90_002/agent_manifest.json"
    ).read_text()
)
truth = json.loads(
    (
        OUTPUT / "data_manifest/kitti_tracking_center90_002/evaluation_only/truth.json"
    ).read_text()
)
front = json.loads(
    (
        OUTPUT / "mapping_evaluation/kitti_tracking_frontend_full_001/summary.json"
    ).read_text()
)
raw = {(r["sequence"], r["original_frame_id"]): r["objects"] for r in truth["frames"]}
byseq = []
for seq in ["0000", "0005", "0013"]:
    frames = [json.loads(x) for x in (ROOT / f"{seq}_frames.jsonl").open()]
    labels = Counter()
    primary = set()
    rawinstances = set()
    gt = 0
    allboxes = Counter()
    for f in frames:
        for o in raw[seq, f["original_frame_id"]]:
            labels[o["category"]] += 1
            if o["track_id"] >= 0:
                rawinstances.add(o["instance_token"])
        for o in f["evaluation_only_official"]:
            gt += 1
            primary.add(o["instance_token"])
    for r in front["frames"]:
        if r["sequence"] != seq:
            continue
        with gzip.open(r["detections"], "rb") as h:
            g = pickle.load(h)
        allboxes.update(g["classes"][int(i)] for i in g["class_id"])
    byseq.append(
        dict(
            sequence=seq,
            frames=len(frames),
            raw_label_categories=dict(labels),
            valid_primary_GT_observations=gt,
            valid_primary_unique_instances=len(primary),
            valid_primary_instances=sorted(primary),
            all_raw_tracked_instances=len(rawinstances),
            actual_frontend_2D_categories=dict(allboxes),
            object_counts={
                m: sum(len(f["predictions"][m]) for f in frames)
                for m in frames[0]["predictions"]
            },
        )
    )
assert sum(r["valid_primary_GT_observations"] for r in byseq) == 74
assert sum(r["valid_primary_GT_observations"] > 0 for r in byseq) == 1
report = json.loads((ROOT / "report.json").read_text())
limits = [
    "全部74次主类别GT观测只出现在0005；0000和0013仍完整保留，不重选有利样本",
    "三个固定短窗口仅支持一个序列内主类别召回诊断，不能报告三独立正场景泛化提升",
    "序列bootstrap有2845/4000次含正GT，召回与匹配误差差值区间退化；退化不代表高置信跨场景改善",
    "全门限IDF1包含其他两序列预测，受DontCare/Van等未支持标签影响；不能解读成完整官方追踪榜单",
    "不存在官方列表间隔返回机会时保持率unknown，两个方法匹配间隔事件不等于两个官方遮挡恢复实验",
    "类别映射不改成将Van强行加入主要car指标；原始全部标签与检测保留供另立敏感性版本",
]
dump(
    DEST / "report.json",
    dict(
        passed=True,
        by_sequence=byseq,
        valid_primary_GT_observations=74,
        positive_GT_sequences=1,
        total_sequences=3,
        labels_read_only_after_frontend=True,
        selected_windows_unchanged=True,
        uncertainty_limits=limits,
        source_sha256=digest(__file__),
        inference_unchanged=True,
        goal_complete=False,
    ),
)
failed = json.loads((ROOT / "independent_audit_001/interrupted.json").read_text())
passed = json.loads((ROOT / "independent_audit_002/report.json").read_text())
assert passed["passed"]
dump(
    ROOT / "audit_erratum_002.json",
    dict(
        failed="independent_audit_001/interrupted.json",
        original_failure=failed,
        replacement="independent_audit_002/report.json",
        change="只把独立浮点区间比较从逐位相等改成绝对1e-12且rtol0；整数计数与有效重采样数仍精确相等",
        maximum_interval_difference_rounded_femtounits=passed["checks"][
            "floating_interval_max_abs_difference_femtounits"
        ],
        model_calls_added=0,
        results_rewritten=False,
        primary_report_unchanged_sha256=digest(ROOT / "report.json"),
        goal_complete=False,
    ),
)
print(
    json.dumps(
        dict(
            passed=True,
            positive_GT_sequences=1,
            primary_instances=byseq[1]["valid_primary_unique_instances"],
        )
    )
)
