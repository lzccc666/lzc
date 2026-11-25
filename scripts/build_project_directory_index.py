"""从当前真实目录生成中文导航索引，不改数据、标签或实验结果。"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from organize_project_layout import AUDIT, PROJECT, PUBLIC, links


GUIDE = PROJECT / "doc/project_guide"


def write_csv(name, rows, fields):
    with (GUIDE / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def relative(path):
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def category(path):
    name = str(path)
    if name.startswith("conceptgraph/completion"):
        return "实验实现与独立评价"
    if name.startswith("conceptgraph/slam"):
        return "对象建图与融合"
    if name.startswith("conceptgraph/navigation"):
        return "语义检索、车道、车辆规划与状态"
    if name.startswith("conceptgraph/scenegraph"):
        return "对象关系图"
    if name.startswith("conceptgraph/vlm"):
        return "Qwen与视觉语言接口"
    if name.startswith("conceptgraph/hydra_configs"):
        return "配置"
    if name.startswith("conceptgraph/nuscenes"):
        return "nuScenes输入与标准答案工具"
    if name.startswith("conceptgraph/scripts"):
        return "项目命令与感知预处理"
    if name.startswith("scripts/"):
        return "训练与整理命令"
    if name.startswith("tests/"):
        return "自动检查"
    return "基础工具或其他项目模块"


def directory_index():
    descriptions = {
        "conceptgraph": "业务代码", "scripts": "命令与冻结脚本入口", "tests": "自动检查",
        "dataset": "输入数据与监督答案", "checkpoint": "模型权重",
        "outputs": "运行输出与历史证据", "doc": "项目说明",
        "thesis_materials": "原稿与后续写作资料入口",
        "Grounded-Segment-Anything": "第三方感知依赖，原位保留",
        "gradslam": "第三方依赖，原位保留", "chamferdist": "第三方依赖，原位保留",
        "nature-skills": "已有工具资料", "assets": "静态资源", "build": "已有构建产物",
        "conceptgraph.egg-info": "已有Python包安装元数据",
        "figure": "历史图片目录", "draw_figure_scripts": "历史绘图目录",
    }
    rows = []
    for path in sorted(PROJECT.iterdir()):
        if path.name.startswith(".") or not path.is_dir():
            continue
        rows.append(dict(分类=descriptions.get(path.name, "已有目录，保留"),
                         入口=relative(path), 实体位置=str(path.resolve()),
                         是否链接=path.is_symlink(), 是否存在=path.exists()))
    for path_name, target in links().items():
        path = PROJECT / path_name
        if path_name.startswith("dataset/"):
            role = "输入、标注或划分入口"
        elif path_name.startswith("outputs/"):
            role = "实验结果或报告入口"
        else:
            role = "代码、模型或论文资料入口"
        rows.append(dict(分类=role, 入口=path_name, 实体位置=str(target.resolve()),
                         是否链接=path.is_symlink(), 是否存在=path.exists()))
    plan = json.loads((AUDIT / "plan.json").read_text())
    for item in plan["moves"]:
        new = Path(item["new"])
        rows.append(dict(分类="迁移后的实际输入目录", 入口=relative(new),
                         实体位置=str(new.resolve()), 是否链接=new.is_symlink(), 是否存在=new.exists()))
    detection = PROJECT / "outputs/results/nuscenes/validation/detections"
    rows.append(dict(分类="验证组逐帧检测结果入口", 入口=relative(detection),
                     实体位置=str(detection.resolve()), 是否链接=True, 是否存在=detection.exists()))
    write_csv("目录分类索引.csv", rows, ["分类", "入口", "实体位置", "是否链接", "是否存在"])


def source_index():
    rows = []
    extensions = {".py", ".sh", ".yaml", ".yml"}
    for root in [PROJECT / name for name in ("conceptgraph", "scripts", "tests")]:
        for parent, directories, names in os.walk(root, followlinks=False):
            directories[:] = sorted(d for d in directories if not d.startswith(".") and d != "__pycache__")
            for name in sorted(names):
                path = Path(parent) / name
                if path.suffix in extensions:
                    rows.append(dict(模块=category(path.relative_to(PROJECT)),
                                     代码位置=relative(path), 阅读用途="现有项目实现或命令"))
    for path in sorted((PUBLIC / "source").glob("*.py")):
        rows.append(dict(模块="冻结的后期实验驱动脚本", 代码位置=relative(path),
                         阅读用途="归档入口 scripts/experiment_archive/public_20261115/" + path.name))
    write_csv("源码文件索引.csv", rows, ["模块", "代码位置", "阅读用途"])
    return len(rows)


def split_index():
    root = PROJECT / "dataset/qwen/real_graph_v003"
    rows = []
    summaries = {}
    roles = {"train": "训练集", "validation": "训练验证集", "development_holdout": "验证／调试子集（已查看）"}
    for split, role in roles.items():
        path = root / f"{split}.jsonl"
        records = [json.loads(line) for line in path.open() if line.strip()]
        summaries[split] = dict(role=role, path=relative(path), samples=len(records),
                                unique_frames=len({(a["scene_id"], a["frame_idx"]) for a in records}),
                                scenes=len({a["scene_id"] for a in records}),
                                logs=len({a["log_token"] for a in records}),
                                null_selection_labels=sum(a["response"]["selected_node_id"] is None for a in records),
                                manually_reviewed=sum(bool(a.get("manual_reviewed")) for a in records))
        for a in records:
            rows.append(dict(用途=role, 源文件=relative(path), 样本编号=a["sample_id"],
                             场景=a["scene_id"], 帧=a["frame_idx"], 采集日志=a["log_token"],
                             任务类型=a["query_kind"], 指令=a["query_text"],
                             官方存在目标=a["official_positive"], 正确目标在候选中=a["candidate_target_coverage"],
                             训练标准回答是否不选=a["response"]["selected_node_id"] is None,
                             人工审核=a.get("manual_reviewed", False)))
    write_csv("Qwen样本划分索引.csv", rows, list(rows[0]))
    registry = dict(
        naming_policy="训练=更新参数；验证调试=分析与选方法；测试=固定后评价；不改历史split字段",
        qwen_real_graph_v003=summaries,
        mapping_scene_groups={
            "validation": dict(role="验证／调试场景组，不代表全部用于Qwen训练", scenes=38, logs=11,
                               frames=1508, inputs="dataset/processed/nuscenes/validation/converted",
                               evidence="outputs/results/nuscenes/validation/statistics/report.json"),
            "test": dict(role="冻结后开启的测试场景组，当前已使用", scenes=30, logs=14, frames=1196,
                         inputs="dataset/processed/nuscenes/test/converted",
                         evidence="outputs/results/nuscenes/test/evaluation/report.json")},
        label_changes=False, experiment_rerun=False,
        note="指令模板改写不增加独立画面；official_graph数据是辅助诊断；不把看过的数据重新宣称未见")
    (GUIDE / "数据划分清单.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    return summaries


def frame_examples():
    rows = []
    for group in ["validation", "test"]:
        base = PROJECT / f"dataset/processed/nuscenes/{group}/converted"
        scene = sorted(p for p in base.iterdir() if p.is_dir())[0]
        for part in ["color", "depth"]:
            for path in sorted((scene / part).iterdir())[:1]:
                rows.append(dict(组别="nuScenes " + group, 内容="输入：" + part,
                                 实际路径=relative(path), 实体位置=str(path.resolve())))
        alignment = scene / "frame_alignment.json"
        rows.append(dict(组别="nuScenes " + group, 内容="输入时间与来源", 实际路径=relative(alignment),
                         实体位置=str(alignment.resolve())))
    candidates = [
        ("nuScenes validation", "检测分割", PROJECT / "outputs/results/nuscenes/validation/detections", ".pkl.gz"),
        ("nuScenes validation", "逐帧对象图", PROJECT / "outputs/results/nuscenes/validation/mapping", "frames.jsonl"),
        ("nuScenes test", "检测分割", PROJECT / "outputs/results/nuscenes/test/perception", ".pkl.gz"),
        ("nuScenes test", "逐帧对象图", PROJECT / "outputs/results/nuscenes/test/mapping/full", "frames.jsonl.gz"),
        ("KITTI", "检测分割", PROJECT / "outputs/results/kitti/perception", ".pkl.gz"),
        ("KITTI", "逐帧对象图", PROJECT / "outputs/results/kitti/mapping", "frames.jsonl"),
    ]
    for group, kind, root, suffix in candidates:
        selected = None
        for parent, dirs, names in os.walk(root, followlinks=True):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not d.startswith("."))
            matches = sorted(n for n in names if n.endswith(suffix))
            if matches:
                selected = Path(parent) / matches[0]
                break
        if selected is None:
            raise FileNotFoundError(f"未找到真实结果：{root} / {suffix}")
        rows.append(dict(组别=group, 内容=kind, 实际路径=relative(selected), 实体位置=str(selected.resolve())))
    write_csv("逐帧实际路径示例.csv", rows, ["组别", "内容", "实际路径", "实体位置"])
    return len(rows)


def main():
    GUIDE.mkdir(parents=True, exist_ok=True)
    directory_index()
    count = source_index()
    splits = split_index()
    examples = frame_examples()
    print(json.dumps(dict(source_index_rows=count, frame_examples=examples, qwen_splits=splits),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
