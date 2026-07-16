from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Qwen semantic delegate finetune datasets from nuScenes experiments.")
    parser.add_argument("--minimal-root", type=Path, default=Path("/root/autodl-tmp/lizichong/dataset/nuscenes"))
    parser.add_argument("--exp-suffix", default="qwen2vl_semantics_v1")
    parser.add_argument("--scene-list", nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path, default=Path("/root/autodl-tmp/lizichong/outputs/qwen_finetune_dataset"))
    parser.add_argument("--summary-dir", type=Path, default=Path("/root/autodl-tmp/lizichong/outputs/experiment_summary"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    scenes = args.scene_list or [p.name for p in sorted(args.minimal_root.glob("scene-*")) if p.is_dir()]

    object_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    navigation_rows: list[dict[str, Any]] = []
    rerank_rows: list[dict[str, Any]] = []
    class_counter: Counter[str] = Counter()
    color_counter: Counter[str] = Counter()
    relation_counter: Counter[str] = Counter()
    scene_counter: Counter[str] = Counter()

    for scene_id in scenes:
        exp_dir = args.minimal_root / scene_id / f"exp_{args.exp_suffix}"
        nodes = load_json(exp_dir / "scene_graph.json", [])
        edges = load_json(exp_dir / "cfslam_scenegraph_edge_metadata.json", [])
        nav_cases = load_json(exp_dir / "evaluation" / "navigation_cases.json", [])
        node_lookup = {int(node.get("id", -1)): node for node in nodes}

        for node in nodes:
            tag = str(node.get("object_tag", "unknown"))
            attrs = node.get("attributes", {}) if isinstance(node.get("attributes", {}), dict) else {}
            colors = attrs.get("colors", []) or []
            class_counter[tag] += 1
            for color in colors:
                color_counter[str(color)] += 1
            scene_counter[scene_id] += 1
            sample = {
                "image": None,
                "crop_path": first_debug_crop(exp_dir, int(node.get("id", -1))),
                "mask_path": None,
                "bbox": {"center": node.get("bbox_center"), "extent": node.get("bbox_extent")},
                "scene_id": scene_id,
                "object_id": node.get("id"),
                "detection_label": tag,
                "qwen_or_rule_caption": node.get("caption", ""),
                "object_tag": tag,
                "color": colors[0] if colors else None,
                "body_type": first_or_none(attrs.get("body_types", [])),
                "motion_state": node.get("motion_state", "unknown"),
                "center_history_summary": summarize_center_history(node.get("center_history", [])),
            }
            sample["instruction"] = "请根据目标裁剪图和元数据识别对象类别、颜色、运动状态，并输出结构化 JSON。"
            sample["response"] = {
                "summary": sample["qwen_or_rule_caption"],
                "object_tag": tag,
                "color": sample["color"],
                "body_type": sample["body_type"],
                "motion_state": sample["motion_state"],
            }
            object_rows.append(sample)

        for edge in edges:
            src = int(edge.get("source", -1))
            dst = int(edge.get("target", -1))
            src_node = node_lookup.get(src, {})
            dst_node = node_lookup.get(dst, {})
            relation = str(edge.get("relation", "near"))
            relation_counter[relation] += 1
            sample = {
                "scene_id": scene_id,
                "source_node_id": src,
                "target_node_id": dst,
                "source_tag": src_node.get("object_tag"),
                "target_tag": dst_node.get("object_tag"),
                "source_color": first_color(src_node),
                "target_color": first_color(dst_node),
                "relative_position": relation,
                "distance_m": edge.get("distance_m"),
                "relation_label": relation,
                "relation_type": edge.get("relation_type", "stable"),
            }
            sample["instruction"] = "请根据两个对象的类别、颜色、位置和距离判断空间关系，并输出 relation/confidence/explanation。"
            sample["response"] = {
                "relation": relation,
                "confidence": edge.get("confidence", 1.0),
                "relation_type": sample["relation_type"],
            }
            relation_rows.append(sample)

        for case in nav_cases:
            retrieval = case.get("target_retrieval", {})
            planning = case.get("path_planning", {})
            query = str(retrieval.get("query_text", ""))
            selected_id = retrieval.get("selected_reachable_node_id")
            selected = node_lookup.get(int(selected_id), {}) if selected_id is not None else {}
            success = bool(planning.get("planning_success") and retrieval.get("target_validation", {}).get("valid", True))
            nav_sample = {
                "scene_id": scene_id,
                "query_text": query,
                "structured_parse": {
                    "target_categories": retrieval.get("target_categories", []),
                    "parsed_relation": retrieval.get("parsed_relation"),
                },
                "selected_node_id": selected_id,
                "selected_object_tag": selected.get("object_tag"),
                "selected_target_xy": planning.get("selected_target_xy"),
                "success": success,
                "reason": None if success else case.get("failure", {}).get("failure_reason"),
                "instruction": "请将中文语义导航指令解析为目标类别、颜色、锚点和空间关系。",
            }
            nav_sample["response"] = nav_sample["structured_parse"]
            navigation_rows.append(nav_sample)
            rerank_rows.append(
                {
                    "query_text": query,
                    "scene_id": scene_id,
                    "candidate_nodes": [],
                    "selected_node_id": selected_id,
                    "candidate_scores": {
                        "semantic_score": retrieval.get("semantic_score"),
                        "relation_score": retrieval.get("relation_score"),
                        "distance_prior": retrieval.get("distance_prior"),
                    },
                    "target_validation": retrieval.get("target_validation"),
                    "instruction": "请根据 query 和候选节点排序，选择语义正确且可达的目标节点。",
                    "response": {"selected_node_id": selected_id, "success": success},
                }
            )

    write_jsonl(args.out_dir / "object_caption_dataset.jsonl", object_rows)
    write_jsonl(args.out_dir / "relation_dataset.jsonl", relation_rows)
    write_jsonl(args.out_dir / "navigation_query_dataset.jsonl", navigation_rows)
    write_jsonl(args.out_dir / "candidate_rerank_dataset.jsonl", rerank_rows)
    stats = {
        "num_object_samples": len(object_rows),
        "num_relation_samples": len(relation_rows),
        "num_navigation_samples": len(navigation_rows),
        "num_rerank_samples": len(rerank_rows),
        "class_distribution": dict(class_counter),
        "color_distribution": dict(color_counter),
        "relation_distribution": dict(relation_counter),
        "scene_distribution": dict(scene_counter),
    }
    (args.out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    write_reports(args.summary_dir, args.out_dir, stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_debug_crop(exp_dir: Path, object_id: int) -> str | None:
    path = exp_dir / "cfslam_captions_qwen_debug" / f"{object_id}.png"
    return str(path) if path.exists() else None


def first_or_none(values):
    return values[0] if isinstance(values, list) and values else None


def first_color(node: dict) -> str | None:
    attrs = node.get("attributes", {}) if isinstance(node.get("attributes", {}), dict) else {}
    return first_or_none(attrs.get("colors", []))


def summarize_center_history(history) -> dict[str, Any]:
    if not isinstance(history, list) or not history:
        return {"num_centers": 0}
    return {"num_centers": len(history), "first": history[0], "last": history[-1]}


def write_reports(summary_dir: Path, dataset_dir: Path, stats: dict[str, Any]) -> None:
    (summary_dir / "qwen_usage_report.md").write_text(
        "\n".join(
            [
                "# Qwen 使用说明",
                "",
                "- 对象 crop caption：通过 `QwenSemanticDelegate.describe_object` 支持，默认规则回退，启用后记录 prompt/raw response/parsed response。",
                "- object_tag/possible_tags/颜色属性：当前主流程以检测标签、Qwen/规则响应和颜色统计融合为主。",
                "- 关系推理：scenegraph 生成默认使用几何强关联，Qwen delegate 可作为可选 relation inference。",
                "- 导航 query 解析与候选重排序：当前规则逻辑稳定运行，Qwen delegate 提供可选解析和 rerank 接口。",
                "- 在线开销：Qwen2-VL 会占用显存并增加单目标/单 query 延迟；论文实验建议离线批处理语义增强，在线导航保留规则 fallback。",
                f"- 微调数据集目录：`{dataset_dir}`。",
            ]
        ),
        encoding="utf-8",
    )
    (summary_dir / "qwen_finetune_feasibility.md").write_text(
        "\n".join(
            [
                "# Qwen 微调可行性评估",
                "",
                "- 当前推荐：优先使用 Qwen2-VL-2B-Instruct 或 Qwen2.5-VL-3B 做 LoRA/QLoRA，不建议在 RTX 4090 24GB 上全量微调 7B VLM。",
                "- 数据量评估：当前自动构建数据集适合 sanity tuning 和格式验证；若要显著提升效果，需要人工校验颜色、关系和导航目标标签。",
                "- 显存估算：2B/3B LoRA 通常可在 24GB 内小 batch 训练；7B 建议 4bit QLoRA、梯度检查点和很小 batch。",
                "- 预期收益：更稳定的中文 query 解析、颜色/关系描述和候选重排序；对几何可达性和 A* 本身收益有限。",
                "- 风险：自动标签噪声会放大错误，特别是颜色、摩托车/车辆细分类和关系边。",
                "- 建议：先不强行训练，先用本数据集进行小规模 LoRA sanity run，并人工抽查 100 条样本后再正式训练。",
                "",
                f"数据统计：`{json.dumps(stats, ensure_ascii=False)}`",
            ]
        ),
        encoding="utf-8",
    )
    (summary_dir / "qwen_lora_training_plan.md").write_text(
        "\n".join(
            [
                "# Qwen LoRA 训练计划",
                "",
                "1. 人工抽查并清洗 `outputs/qwen_finetune_dataset/*.jsonl`。",
                "2. 使用 2B/3B VL 模型做 LoRA sanity run，max_steps 先设 20-50。",
                "3. 仅确认 loss 下降和输出 JSON 格式稳定，不把 sanity 权重直接作为论文主结果。",
                "4. 正式训练前补充人工标注或规则校验，避免错误语义污染模型。",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
