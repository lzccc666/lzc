from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from conceptgraph.navigation.io import load_scene_graph_edges, load_scene_graph_nodes
from conceptgraph.navigation.semantic import parse_navigation_query, rank_candidate_targets
from conceptgraph.vlm.qwen_cache import semantic_version, sha256_file
from conceptgraph.vlm.schemas import SCHEMA_VERSION, normalize_category, normalize_color


DEFAULT_ROOT = Path("/root/autodl-tmp/lizichong")
TASK_FILES = {
    "object_attributes": "object_attributes.jsonl",
    "query_parse": "query_parse.jsonl",
    "candidate_rerank": "candidate_rerank.jsonl",
}


@dataclass(frozen=True)
class DatasetBuildConfig:
    project_root: Path = DEFAULT_ROOT
    minimal_root: Path = DEFAULT_ROOT / "dataset/nuscenes"
    exp_suffix: str = "qwen2vl_semantics_v1"
    out_dir: Path = DEFAULT_ROOT / "outputs/project_v2/qwen_eval/dataset"
    scene_list: tuple[str, ...] = ()
    tasks: tuple[str, ...] = tuple(TASK_FILES)
    split_seed: int = 41
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    max_candidates: int = 5
    task_weights: tuple[tuple[str, float], ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建阶段4 Qwen多模态语义数据集。")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--minimal-root", type=Path, default=None)
    parser.add_argument("--exp-suffix", default="qwen2vl_semantics_v1")
    parser.add_argument("--scene-list", nargs="*", default=[])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_ROOT / "outputs/project_v2/qwen_eval/dataset",
        help="旧CLI参数仍可用；为保护旧数据集，默认已迁移到阶段4目录。",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help="兼容旧CLI；报告现在统一写入out-dir，若指定则额外复制中文摘要。",
    )
    parser.add_argument("--tasks", nargs="*", choices=sorted(TASK_FILES), default=list(TASK_FILES))
    parser.add_argument("--split-seed", type=int, default=41)
    parser.add_argument("--split-ratios", nargs=3, type=float, default=(0.7, 0.15, 0.15))
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument(
        "--task-weight",
        action="append",
        default=[],
        metavar="TASK=WEIGHT",
        help="可重复指定训练任务采样权重。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    minimal_root = (args.minimal_root or (project_root / "dataset/nuscenes")).resolve()
    config = DatasetBuildConfig(
        project_root=project_root,
        minimal_root=minimal_root,
        exp_suffix=args.exp_suffix,
        out_dir=args.out_dir.resolve(),
        scene_list=tuple(args.scene_list),
        tasks=tuple(args.tasks),
        split_seed=args.split_seed,
        split_ratios=tuple(args.split_ratios),
        max_candidates=max(1, args.max_candidates),
        task_weights=tuple(sorted(_parse_task_weights(args.task_weight).items())),
    )
    report = build_dataset(config)
    if args.summary_dir is not None:
        args.summary_dir.mkdir(parents=True, exist_ok=True)
        summary = config.out_dir / "dataset_report.md"
        (args.summary_dir / "stage4_qwen_dataset_report.md").write_text(
            summary.read_text(encoding="utf-8"), encoding="utf-8"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


def build_dataset(config: DatasetBuildConfig) -> dict[str, Any]:
    _validate_config(config)
    requested_scenes = list(config.scene_list) or [
        path.name
        for path in sorted(config.minimal_root.glob("scene-*"))
        if (path / f"exp_{config.exp_suffix}/scene_graph.json").exists()
    ]
    scenes = sorted(set(requested_scenes))
    scene_splits = deterministic_scene_split(scenes, config.split_seed, config.split_ratios)
    rows_by_task: dict[str, list[dict[str, Any]]] = {task: [] for task in config.tasks}
    crop_issues: list[dict[str, Any]] = []
    scene_diagnostics: list[dict[str, Any]] = []

    for scene_id in scenes:
        built, diagnostic = _build_scene_rows(config, scene_id, scene_splits[scene_id])
        scene_diagnostics.append(diagnostic)
        crop_issues.extend(diagnostic.pop("crop_issues"))
        for task in config.tasks:
            rows_by_task[task].extend(built.get(task, []))

    config.out_dir.mkdir(parents=True, exist_ok=True)
    for task, rows in rows_by_task.items():
        rows.sort(key=lambda row: row["sample_id"])
        write_jsonl(config.out_dir / TASK_FILES[task], rows)
    combined = sorted(
        (row for rows in rows_by_task.values() for row in rows),
        key=lambda row: (row["task"], row["sample_id"]),
    )
    write_jsonl(config.out_dir / "all_tasks.jsonl", combined)

    task_weights = _resolved_task_weights(config.tasks, dict(config.task_weights))
    manifest = _build_split_manifest(
        rows_by_task,
        scene_splits,
        seed=config.split_seed,
        ratios=config.split_ratios,
        task_weights=task_weights,
    )
    _write_json(config.out_dir / "split_manifest.json", manifest)
    leakage = audit_scene_leakage(rows_by_task, scene_splits)
    _write_json(config.out_dir / "leakage_audit.json", leakage)
    crop_report = {
        "schema_version": "stage4-crop-audit-v1",
        "issue_count": len(crop_issues),
        "missing_count": sum(item["status"] == "missing" for item in crop_issues),
        "corrupt_count": sum(item["status"] == "corrupt" for item in crop_issues),
        "issues": crop_issues,
    }
    _write_json(config.out_dir / "crop_audit.json", crop_report)
    distributions = _dataset_distributions(rows_by_task)
    _write_json(config.out_dir / "dataset_distributions.json", distributions)

    content_material = {
        "manifest": {key: value for key, value in manifest.items() if key != "deterministic_sha256"},
        "rows": rows_by_task,
        "crop_report": crop_report,
        "leakage": leakage,
    }
    deterministic_hash = semantic_version(content_material)
    report = {
        "schema_version": "stage4-dataset-report-v1",
        "output_dir": str(config.out_dir),
        "scenes": scenes,
        "scene_count": len(scenes),
        "sample_count": len(combined),
        "task_counts": {task: len(rows) for task, rows in rows_by_task.items()},
        "split_scene_counts": {
            split: sum(value == split for value in scene_splits.values())
            for split in ("train", "validation", "test")
        },
        "scene_diagnostics": scene_diagnostics,
        "crop_audit": crop_report,
        "leakage_audit": leakage,
        "task_sampling_weights": task_weights,
        "label_policy": "自动规则和既有导航结果仅作为弱监督标签，不视为人工语义真值。",
        "deterministic_sha256": deterministic_hash,
    }
    _write_json(config.out_dir / "dataset_report.json", report)
    _write_markdown_report(config.out_dir / "dataset_report.md", report, distributions)
    return report


def deterministic_scene_split(
    scenes: list[str] | tuple[str, ...],
    seed: int,
    ratios: tuple[float, float, float],
) -> dict[str, str]:
    unique = sorted(set(scenes), key=lambda scene: (semantic_version([seed, scene]), scene))
    counts = _split_counts(len(unique), ratios)
    result = {}
    offset = 0
    for split, count in zip(("train", "validation", "test"), counts):
        for scene in unique[offset : offset + count]:
            result[scene] = split
        offset += count
    return result


def audit_scene_leakage(
    rows_by_task: dict[str, list[dict[str, Any]]],
    scene_splits: dict[str, str],
) -> dict[str, Any]:
    observed: dict[str, set[str]] = defaultdict(set)
    for rows in rows_by_task.values():
        for row in rows:
            observed[row["scene_id"]].add(row["split"])
    leaked = {scene: sorted(splits) for scene, splits in observed.items() if len(splits) > 1}
    mismatches = [
        {"scene_id": scene, "expected": scene_splits.get(scene), "observed": sorted(splits)}
        for scene, splits in sorted(observed.items())
        if splits != {scene_splits.get(scene)}
    ]
    return {
        "schema_version": "stage4-leakage-audit-v1",
        "scene_leakage_count": len(leaked),
        "leaked_scenes": leaked,
        "assignment_mismatch_count": len(mismatches),
        "assignment_mismatches": mismatches,
        "passed": not leaked and not mismatches,
    }


def _build_scene_rows(
    config: DatasetBuildConfig,
    scene_id: str,
    split: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    exp_dir = config.minimal_root / scene_id / f"exp_{config.exp_suffix}"
    node_path = exp_dir / "scene_graph.json"
    nodes_json = load_json(node_path, [])
    navigation_cases = load_json(exp_dir / "evaluation/navigation_cases.json", [])
    node_lookup = {int(node.get("id", -1)): node for node in nodes_json}
    rows: dict[str, list[dict[str, Any]]] = {task: [] for task in config.tasks}
    crop_issues: list[dict[str, Any]] = []

    if "object_attributes" in rows:
        for node_id, node in sorted(node_lookup.items()):
            crop_path = exp_dir / "cfslam_captions_qwen_debug" / f"{node_id}.png"
            crop = validate_crop(crop_path)
            if not crop["valid"]:
                crop_issues.append(
                    {
                        "scene_id": scene_id,
                        "object_id": node_id,
                        "crop_name": crop_path.name,
                        "status": crop["status"],
                        "diagnostic": crop["diagnostic"],
                    }
                )
                continue
            attributes = node.get("attributes", {}) if isinstance(node.get("attributes"), dict) else {}
            tag = _safe_category(node.get("object_tag"))
            possible_tags = _canonical_categories(node.get("possible_tags", []) or [tag]) or [tag]
            colors = _canonical_colors(attributes.get("colors", []))
            body_types = attributes.get("body_types", [])
            body_type = body_types[0] if isinstance(body_types, list) and body_types else None
            if body_type not in {"bus", "hatchback", "motorcycle", "sedan", "suv", "truck", "unknown", "van"}:
                body_type = None
            response = {
                "schema_version": SCHEMA_VERSION,
                "summary": str(node.get("caption") or f"{tag} object")[:512],
                "object_tag": tag,
                "possible_tags": possible_tags[:8],
                "colors": colors[:4],
                "body_type": body_type,
                "motion_state": _motion_state(node.get("motion_state")),
                "confidence": 0.7,
            }
            rows["object_attributes"].append(
                {
                    **_common_row(
                        task="object_attributes",
                        scene_id=scene_id,
                        local_id=f"object-{node_id:04d}",
                        split=split,
                        source="existing_qwen_scene_graph_and_real_crop",
                        label_source="automatic_weak_supervision",
                        image_sha256=crop["sha256"],
                    ),
                    "crop_path": str(crop_path.resolve()),
                    "image_width": crop["width"],
                    "image_height": crop["height"],
                    "object_id": node_id,
                    "metadata": {
                        "bbox_center": node.get("bbox_center"),
                        "bbox_extent": node.get("bbox_extent"),
                        "detection_label": node.get("object_tag"),
                        "caption": node.get("caption"),
                        "motion_state": node.get("motion_state", "unknown"),
                    },
                    "instruction": "请根据真实对象裁剪图和元数据输出严格对象属性JSON。",
                    "response": response,
                }
            )

    scene_nodes = load_scene_graph_nodes(node_path) if node_path.exists() else {}
    edges, edge_source = _load_best_edges(config.project_root, scene_id, exp_dir)
    edge_dicts = [edge.to_dict() for edge in edges]
    for case_index, case in enumerate(navigation_cases):
        retrieval = case.get("target_retrieval", {})
        query_text = str(retrieval.get("query_text") or "").strip()
        if not query_text:
            continue
        parsed = parse_navigation_query(query_text, 8.0)
        selected_id = retrieval.get("selected_reachable_node_id")
        query_response = {
            "schema_version": SCHEMA_VERSION,
            "target_text": parsed.target_text,
            "target_categories": _canonical_categories(parsed.target_categories),
            "target_colors": _canonical_colors(parsed.target_colors),
            "anchor_text": parsed.anchor_text,
            "anchor_categories": [],
            "anchor_colors": _canonical_colors(parsed.anchor_colors),
            "relation": parsed.relation,
            "prefer_nearest": parsed.prefer_nearest,
            "prefer_front": parsed.prefer_front,
            "confidence": 0.7,
        }
        if "query_parse" in rows:
            rows["query_parse"].append(
                {
                    **_common_row(
                        task="query_parse",
                        scene_id=scene_id,
                        local_id=f"query-{case_index:04d}",
                        split=split,
                        source="existing_navigation_case",
                        label_source="automatic_rule_weak_supervision",
                        image_sha256=None,
                    ),
                    "query_text": query_text,
                    "instruction": "请将中文语义导航指令解析为严格JSON。",
                    "response": query_response,
                }
            )
        if "candidate_rerank" in rows and selected_id is not None and int(selected_id) in scene_nodes:
            candidate_row = _build_rerank_row(
                scene_id=scene_id,
                split=split,
                case_index=case_index,
                case=case,
                parsed=parsed,
                nodes=scene_nodes,
                edges=edges,
                edge_dicts=edge_dicts,
                edge_source=edge_source,
                max_candidates=config.max_candidates,
            )
            if candidate_row is not None:
                rows["candidate_rerank"].append(candidate_row)

    diagnostic = {
        "scene_id": scene_id,
        "split": split,
        "node_count": len(nodes_json),
        "navigation_case_count": len(navigation_cases),
        "edge_count": len(edges),
        "edge_source": edge_source,
        "task_counts": {task: len(task_rows) for task, task_rows in rows.items()},
        "crop_issues": crop_issues,
    }
    return rows, diagnostic


def _build_rerank_row(
    *,
    scene_id: str,
    split: str,
    case_index: int,
    case: dict[str, Any],
    parsed,
    nodes,
    edges,
    edge_dicts: list[dict[str, Any]],
    edge_source: str,
    max_candidates: int,
) -> dict[str, Any] | None:
    retrieval = case.get("target_retrieval", {})
    planning = case.get("path_planning", {})
    selected_id = int(retrieval["selected_reachable_node_id"])
    start_xy = np.asarray(planning.get("start_xy", [0.0, 0.0]), dtype=float)[:2]
    selected_xy = np.asarray(planning.get("selected_target_xy", start_xy), dtype=float)[:2]
    heading = selected_xy - start_xy
    if float(np.linalg.norm(heading)) > 1e-9:
        parsed.vehicle_heading_xy = heading
    candidates = rank_candidate_targets(
        parsed,
        nodes,
        edges,
        start_xy,
        lambda_text=0.55,
        lambda_relation=0.30,
        lambda_distance=0.25,
        minimum_edge_confidence=0.35,
        relation_coordinate_frame="world",
        relation_coordinate_frame_fallback="world",
    )
    top = candidates[:max_candidates]
    if selected_id not in {candidate.node_id for candidate in top}:
        selected_candidate = next((item for item in candidates if item.node_id == selected_id), None)
        if selected_candidate is None:
            return None
        top = ([selected_candidate] + top)[:max_candidates]
    candidate_ids = [int(candidate.node_id) for candidate in top]
    candidate_nodes = []
    for candidate in top:
        node = nodes[candidate.node_id]
        candidate_nodes.append(
            {
                "node_id": int(candidate.node_id),
                "object_tag": node.object_tag,
                "possible_tags": node.possible_tags,
                "caption": node.caption,
                "bbox_center": node.bbox_center.tolist(),
                "motion_state": node.motion_state,
                "stage2_scores": {
                    "combined_semantic": float(candidate.score),
                    "text": float(candidate.text_score),
                    "relation": float(candidate.relation_score),
                    "distance_prior": float(candidate.distance_prior),
                },
            }
        )
    candidate_set = set(candidate_ids)
    relation_metadata = [
        edge
        for edge in edge_dicts
        if int(edge["source"]) in candidate_set or int(edge["target"]) in candidate_set
    ]
    label_ranking = [selected_id] + [node_id for node_id in candidate_ids if node_id != selected_id]
    return {
        **_common_row(
            task="candidate_rerank",
            scene_id=scene_id,
            local_id=f"rerank-{case_index:04d}",
            split=split,
            source="stage2_candidate_scores_with_stage3_relation_metadata",
            label_source="existing_navigation_result_weak_supervision",
            image_sha256=None,
        ),
        "query_text": str(retrieval.get("query_text")),
        "candidate_nodes": candidate_nodes,
        "candidate_ids": candidate_ids,
        "relation_metadata": relation_metadata,
        "relation_metadata_source": edge_source,
        "correct_target_node_id": selected_id,
        "target_validation": retrieval.get("target_validation"),
        "instruction": "请只在给定候选集合内重排并输出严格JSON。",
        "response": {
            "schema_version": SCHEMA_VERSION,
            "selected_node_id": selected_id,
            "ranking": label_ranking,
            "explanation": "existing_validated_navigation_target_first",
            "confidence": 0.7,
        },
    }


def validate_crop(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"valid": False, "status": "missing", "diagnostic": "crop file does not exist"}
    try:
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            width, height = rgb.size
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return {
            "valid": False,
            "status": "corrupt",
            "diagnostic": f"{type(exc).__name__}: {str(exc)[:200]}",
        }
    if width <= 0 or height <= 0:
        return {"valid": False, "status": "corrupt", "diagnostic": "image has zero extent"}
    return {
        "valid": True,
        "status": "valid",
        "diagnostic": None,
        "width": width,
        "height": height,
        "sha256": sha256_file(path),
    }


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_best_edges(project_root: Path, scene_id: str, exp_dir: Path):
    structured = (
        project_root
        / "outputs/project_v2/scenegraph_relations/stage3_full_rebuild"
        / scene_id
        / "structured_edges.json"
    )
    if structured.exists():
        return load_scene_graph_edges(structured), "stage3_structured_edges"
    legacy = exp_dir / "cfslam_scenegraph_edges.pkl"
    if legacy.exists():
        return load_scene_graph_edges(legacy), "legacy_edges_normalized_by_stage3_loader"
    return [], "missing_edges"


def _common_row(
    *,
    task: str,
    scene_id: str,
    local_id: str,
    split: str,
    source: str,
    label_source: str,
    image_sha256: str | None,
) -> dict[str, Any]:
    return {
        "dataset_schema_version": "stage4-qwen-dataset-v1",
        "task": task,
        "scene_id": scene_id,
        "sample_id": f"{scene_id}:{task}:{local_id}",
        "split": split,
        "source": source,
        "image_sha256": image_sha256,
        "label_source": label_source,
    }


def _build_split_manifest(
    rows_by_task: dict[str, list[dict[str, Any]]],
    scene_splits: dict[str, str],
    *,
    seed: int,
    ratios: tuple[float, float, float],
    task_weights: dict[str, float],
) -> dict[str, Any]:
    split_scenes = {
        split: sorted(scene for scene, assigned in scene_splits.items() if assigned == split)
        for split in ("train", "validation", "test")
    }
    samples = {
        task: {
            split: [row["sample_id"] for row in rows if row["split"] == split]
            for split in ("train", "validation", "test")
        }
        for task, rows in rows_by_task.items()
    }
    manifest = {
        "schema_version": "stage4-split-manifest-v1",
        "algorithm": "sha256(seed, scene_id)排序后按确定性计数切分",
        "seed": seed,
        "requested_ratios": {
            "train": ratios[0],
            "validation": ratios[1],
            "test": ratios[2],
        },
        "scene_to_split": dict(sorted(scene_splits.items())),
        "scenes": split_scenes,
        "samples": samples,
        "task_sampling_weights": task_weights,
    }
    manifest["deterministic_sha256"] = semantic_version(manifest)
    return manifest


def _dataset_distributions(rows_by_task: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    scene_counts = Counter()
    split_counts = Counter()
    category_counts = Counter()
    task_scene_counts: dict[str, Counter] = defaultdict(Counter)
    for task, rows in rows_by_task.items():
        for row in rows:
            scene_counts[row["scene_id"]] += 1
            split_counts[row["split"]] += 1
            task_scene_counts[task][row["scene_id"]] += 1
            response = row.get("response", {})
            if response.get("object_tag"):
                category_counts[response["object_tag"]] += 1
            for category in response.get("target_categories", []):
                category_counts[category] += 1
    return {
        "task": {task: len(rows) for task, rows in rows_by_task.items()},
        "scene": dict(sorted(scene_counts.items())),
        "split": dict(sorted(split_counts.items())),
        "category": dict(sorted(category_counts.items())),
        "task_scene": {
            task: dict(sorted(counter.items())) for task, counter in sorted(task_scene_counts.items())
        },
    }


def _write_markdown_report(path: Path, report: dict[str, Any], distributions: dict[str, Any]) -> None:
    lines = [
        "# 阶段4 Qwen数据集审计",
        "",
        f"- 场景数：{report['scene_count']}",
        f"- 样本数：{report['sample_count']}",
        f"- 任务分布：`{json.dumps(report['task_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- split场景数：`{json.dumps(report['split_scene_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- scene泄漏数：{report['leakage_audit']['scene_leakage_count']}",
        f"- 缺失crop：{report['crop_audit']['missing_count']}",
        f"- 损坏crop：{report['crop_audit']['corrupt_count']}",
        f"- 确定性哈希：`{report['deterministic_sha256']}`",
        "",
        "所有对象属性样本均引用真实存在、可解码且已计算内容哈希的crop；不合格crop不会进入训练集。",
        "candidate_rerank样本包含真实节点、阶段2语义分数、阶段3结构化关系元数据和既有已验证目标。",
        "",
        "## 标签限制",
        "",
        "对象类别、颜色、查询解析和正确目标均来自已有自动管线，只能作为弱监督标签。",
        "这些标签可能包含颜色误判、细分类混淆、关系噪声和既有导航选择偏差，不等价于人工真值。",
        "",
        "## 分布",
        "",
        f"`{json.dumps(distributions, ensure_ascii=False, sort_keys=True)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _split_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if total == 0:
        return 0, 0, 0
    raw = [total * ratio / sum(ratios) for ratio in ratios]
    counts = [int(value) for value in raw]
    for index in sorted(range(3), key=lambda item: (-(raw[item] - counts[item]), item)):
        if sum(counts) >= total:
            break
        counts[index] += 1
    if total >= 3:
        for index in range(3):
            if counts[index] == 0:
                donor = max(range(3), key=lambda item: counts[item])
                counts[donor] -= 1
                counts[index] += 1
    return counts[0], counts[1], counts[2]


def _resolved_task_weights(tasks: tuple[str, ...], supplied: dict[str, float]) -> dict[str, float]:
    weights = {task: float(supplied.get(task, 1.0)) for task in tasks}
    if any(value <= 0 for value in weights.values()):
        raise ValueError("task weights must be positive")
    total = sum(weights.values())
    return {task: value / total for task, value in sorted(weights.items())}


def _parse_task_weights(values: list[str]) -> dict[str, float]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid task weight: {value}; expected TASK=WEIGHT")
        task, raw_weight = value.split("=", 1)
        if task not in TASK_FILES:
            raise ValueError(f"unknown task in weight: {task}")
        result[task] = float(raw_weight)
    return result


def _validate_config(config: DatasetBuildConfig) -> None:
    unknown = set(config.tasks) - set(TASK_FILES)
    if unknown:
        raise ValueError(f"unsupported tasks: {sorted(unknown)}")
    if len(config.split_ratios) != 3 or any(value < 0 for value in config.split_ratios):
        raise ValueError("split_ratios must contain three non-negative values")
    if sum(config.split_ratios) <= 0:
        raise ValueError("split ratios must have a positive sum")
    legacy = (config.project_root / "outputs/qwen_finetune_dataset").resolve()
    output = config.out_dir.resolve()
    if output == legacy or legacy in output.parents:
        raise ValueError("拒绝覆盖旧Qwen数据集；请将--out-dir指向outputs/project_v2/qwen_eval/dataset")


def _safe_category(value: Any) -> str:
    try:
        return normalize_category(str(value or "unknown"))
    except (TypeError, ValueError):
        return "unknown"


def _canonical_categories(values: list[Any]) -> list[str]:
    result = []
    for value in values:
        category = _safe_category(value)
        if category not in result:
            result.append(category)
    return result


def _canonical_colors(values: list[Any]) -> list[str]:
    result = []
    for value in values:
        try:
            color = normalize_color(value)
        except (TypeError, ValueError):
            continue
        if color not in result:
            result.append(color)
    return result


def _motion_state(value: Any) -> str:
    normalized = str(value or "unknown").lower()
    return normalized if normalized in {"static", "dynamic", "unknown"} else "unknown"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
