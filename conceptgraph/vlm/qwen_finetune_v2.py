from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from conceptgraph.vlm.qwen_cache import sha256_file
from conceptgraph.vlm.schemas import (
    CANONICAL_BODY_TYPES,
    CANONICAL_CATEGORIES,
    CANONICAL_COLORS,
    CANONICAL_MOTION_STATES,
    CANONICAL_RELATIONS,
    SCHEMA_VERSION,
    extract_json_object,
    parse_and_validate_output,
)


TASK_FILES = {
    "object_attributes": "object_attributes.jsonl",
    "query_parse": "query_parse.jsonl",
    "candidate_rerank": "candidate_rerank.jsonl",
}
QUERY_SEMANTIC_FIELDS = (
    "target_text",
    "target_categories",
    "target_colors",
    "anchor_text",
    "anchor_categories",
    "anchor_colors",
    "relation",
    "prefer_nearest",
    "prefer_front",
)
HARD_GATES = {
    "overall_schema_valid_rate": 0.95,
    "per_task_schema_valid_rate": 0.90,
    "query_macro_f1": 0.90,
    "query_semantic_exact_match": 0.80,
    "object_category_macro_f1": 0.85,
    "object_color_macro_f1": 0.80,
    "object_motion_macro_f1": 0.85,
    "rerank_top1": 0.85,
    "rerank_mrr": 0.90,
    "rerank_candidate_legality": 1.0,
    "schema_or_illegal_fallback_rate_max": 0.05,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def caption_color_candidates(caption: Any) -> list[str]:
    normalized = normalize_text(caption)
    matches = {
        color: list(
            re.finditer(rf"(?<![a-z]){re.escape(color)}(?![a-z])", normalized)
        )
        for color in CANONICAL_COLORS - {"unknown"}
    }
    counts = {color: len(items) for color, items in matches.items()}
    maximum = max(counts.values(), default=0)
    candidates = [
        color for color, count in counts.items() if count == maximum and count > 0
    ]
    return sorted(candidates, key=lambda color: (matches[color][0].start(), color))


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    scores = candidate.get("stage2_scores", {})
    return {
        "node_id": int(candidate["node_id"]),
        "object_tag": candidate.get("object_tag"),
        "possible_tags": candidate.get("possible_tags", []),
        "caption": candidate.get("caption"),
        "motion_state": candidate.get("motion_state"),
        "stage2_scores": {
            key: round(float(scores[key]), 6)
            for key in ("text", "relation", "distance_prior", "combined_semantic")
            if key in scores and isinstance(scores[key], (int, float))
        },
    }


def compact_relations(
    relations: Iterable[dict[str, Any]],
    candidate_ids: Iterable[int],
    *,
    limit: int = 24,
) -> list[dict[str, Any]]:
    allowed = {int(item) for item in candidate_ids}
    compact = []
    for edge in relations:
        source = edge.get("source")
        target = edge.get("target")
        if source not in allowed and target not in allowed:
            continue
        compact.append(
            {
                "source": source,
                "target": target,
                "relation": edge.get("relation"),
                "directed": edge.get("directed"),
                "confidence": round(float(edge.get("confidence", 0.0)), 6),
                "distance_m": (
                    round(float(edge["distance_m"]), 4)
                    if isinstance(edge.get("distance_m"), (int, float))
                    else None
                ),
            }
        )
    compact.sort(
        key=lambda item: (
            -item["confidence"],
            str(item["relation"]),
            int(item["source"]),
            int(item["target"]),
        )
    )
    return compact[:limit]


def build_prompt(task: str, row: dict[str, Any]) -> str:
    if task == "query_parse":
        contract = {
            "schema_version": SCHEMA_VERSION,
            "target_text": "目标短语",
            "target_categories": ["car"],
            "target_colors": ["gray"],
            "anchor_text": None,
            "anchor_categories": [],
            "anchor_colors": [],
            "relation": "near",
            "prefer_nearest": False,
            "prefer_front": False,
            "confidence": 0.9,
        }
        return (
            "任务：解析中文语义导航指令。只输出一个JSON对象，不要Markdown、解释或额外字段。\n"
            f"字段模板：{canonical_json(contract)}\n"
            f"类别只能来自：{canonical_json(sorted(CANONICAL_CATEGORIES))}\n"
            f"颜色只能来自：{canonical_json(sorted(CANONICAL_COLORS))}\n"
            f"关系只能为null或：{canonical_json(sorted(CANONICAL_RELATIONS))}\n"
            f"指令：{str(row.get('query_text', '')).strip()}"
        )
    if task == "object_attributes":
        contract = {
            "schema_version": SCHEMA_VERSION,
            "summary": "gray car",
            "object_tag": "car",
            "possible_tags": ["car"],
            "colors": ["gray"],
            "body_type": "sedan",
            "motion_state": "static",
            "confidence": 0.9,
        }
        metadata = {
            key: row.get("metadata", {}).get(key)
            for key in ("caption", "detection_label", "motion_state", "bbox_extent")
            if key in row.get("metadata", {})
        }
        metadata["colors_required_exact"] = caption_color_candidates(
            metadata.get("caption")
        )
        return (
            "任务：结合对象裁剪图和元数据输出对象属性。只输出一个JSON对象，不要Markdown、解释或额外字段。\n"
            f"字段模板：{canonical_json(contract)}\n"
            f"object_tag/possible_tags只能来自：{canonical_json(sorted(CANONICAL_CATEGORIES))}\n"
            f"colors只能来自：{canonical_json(sorted(CANONICAL_COLORS))}\n"
            f"body_type只能为null或：{canonical_json(sorted(CANONICAL_BODY_TYPES))}\n"
            f"motion_state只能来自：{canonical_json(sorted(CANONICAL_MOTION_STATES))}\n"
            "colors字段必须逐项复制元数据colors_required_exact，不能遗漏、增加或改变顺序。"
            "例如gray/red必须输出[\"gray\",\"red\"]。\n"
            f"元数据：{canonical_json(metadata)}"
        )
    if task == "candidate_rerank":
        candidate_ids = [int(item) for item in row.get("candidate_ids", [])]
        payload = {
            "query_text": row.get("query_text"),
            "candidate_nodes": [compact_candidate(item) for item in row.get("candidate_nodes", [])],
            "candidate_relations": compact_relations(
                row.get("relation_metadata", []), candidate_ids
            ),
        }
        contract = {
            "schema_version": SCHEMA_VERSION,
            "selected_node_id": candidate_ids[0] if candidate_ids else 0,
            "ranking": candidate_ids,
            "explanation": "semantic ranking",
            "confidence": 0.9,
        }
        return (
            "任务：按指令语义重排已经通过可达性与碰撞检查的候选。只输出一个JSON对象，不要Markdown、解释或额外字段。\n"
            "ranking必须包含全部候选ID、不得重复或新增ID，selected_node_id必须等于ranking首项。\n"
            f"字段模板：{canonical_json(contract)}\n"
            f"输入：{canonical_json(payload)}"
        )
    raise ValueError(f"unsupported task: {task}")


def load_dataset_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for task, filename in TASK_FILES.items():
        path = dataset_dir / filename
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                if row.get("task") != task:
                    raise ValueError(f"{path}:{line_number} task mismatch")
                rows.append(row)
    return sorted(rows, key=lambda row: (row["task"], row["sample_id"]))


def _dhash(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(gray.getdata())
    bits = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            bits = (bits << 1) | int(left > right)
    return bits


def _input_signature(row: dict[str, Any]) -> str:
    task = row["task"]
    if task == "query_parse":
        value = {"query_text": normalize_text(row.get("query_text"))}
    elif task == "object_attributes":
        value = {"image_sha256": row.get("image_sha256")}
    else:
        value = {
            "query_text": normalize_text(row.get("query_text")),
            "candidate_tags": [
                normalize_text(item.get("object_tag")) for item in row.get("candidate_nodes", [])
            ],
        }
    return semantic_sha256(value)


def _cross_split_groups(rows: list[dict[str, Any]], key_fn) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    result = []
    for signature, items in groups.items():
        splits = sorted({item["split"] for item in items})
        if len(splits) < 2:
            continue
        result.append(
            {
                "signature": signature,
                "splits": splits,
                "sample_count": len(items),
                "sample_ids_sha256": semantic_sha256(sorted(item["sample_id"] for item in items)),
            }
        )
    return sorted(result, key=lambda item: item["signature"])


def audit_dataset(dataset_dir: Path) -> dict[str, Any]:
    rows = load_dataset_rows(dataset_dir)
    manifest = json.loads((dataset_dir / "split_manifest.json").read_text(encoding="utf-8"))
    expected = {
        sample_id: (task, split)
        for task, split_map in manifest["samples"].items()
        for split, sample_ids in split_map.items()
        for sample_id in sample_ids
    }
    sample_ids = [row["sample_id"] for row in rows]
    duplicate_ids = sorted(item for item, count in Counter(sample_ids).items() if count > 1)
    manifest_mismatches = []
    task_split_counts: dict[str, Counter] = defaultdict(Counter)
    scene_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        task_split_counts[row["task"]][row["split"]] += 1
        scene_splits[row["scene_id"]].add(row["split"])
        if expected.get(row["sample_id"]) != (row["task"], row["split"]):
            manifest_mismatches.append(row["sample_id"])

    crop_missing = []
    crop_hash_mismatch = []
    crop_decode_errors = []
    image_rows = [row for row in rows if row["task"] == "object_attributes"]
    perceptual = []
    for row in image_rows:
        path = Path(row.get("crop_path", ""))
        if not path.is_file():
            crop_missing.append(row["sample_id"])
            continue
        if sha256_file(path) != row.get("image_sha256"):
            crop_hash_mismatch.append(row["sample_id"])
        try:
            perceptual.append((row, _dhash(path)))
        except OSError:
            crop_decode_errors.append(row["sample_id"])

    perceptual_pairs = Counter()
    for index, (left, left_hash) in enumerate(perceptual):
        for right, right_hash in perceptual[index + 1 :]:
            if left["split"] == right["split"]:
                continue
            distance = (left_hash ^ right_hash).bit_count()
            if distance <= 4:
                split_pair = "<->".join(sorted((left["split"], right["split"])))
                perceptual_pairs[split_pair] += 1

    object_rows = [row for row in rows if row["task"] == "object_attributes"]
    rerank_rows = [row for row in rows if row["task"] == "candidate_rerank"]
    object_tag_proxy = sum(
        normalize_text(row.get("metadata", {}).get("detection_label"))
        == normalize_text(row["response"].get("object_tag"))
        for row in object_rows
    )
    object_motion_proxy = sum(
        normalize_text(row.get("metadata", {}).get("motion_state"))
        == normalize_text(row["response"].get("motion_state"))
        for row in object_rows
    )
    object_color_proxy = sum(
        all(
            normalize_text(color) in normalize_text(row.get("metadata", {}).get("caption"))
            for color in row["response"].get("colors", [])
        )
        for row in object_rows
    )
    caption_color_candidate_proxy = sum(
        caption_color_candidates(row.get("metadata", {}).get("caption"))
        == row["response"].get("colors", [])
        for row in object_rows
    )
    rerank_first_proxy = sum(
        bool(row.get("candidate_ids"))
        and int(row["candidate_ids"][0]) == int(row["correct_target_node_id"])
        for row in rerank_rows
    )
    rerank_highest_score_proxy = 0
    for row in rerank_rows:
        candidates = row.get("candidate_nodes", [])
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda item: float(item.get("stage2_scores", {}).get("combined_semantic", -math.inf)),
        )
        rerank_highest_score_proxy += int(best.get("node_id") == row.get("correct_target_node_id"))

    distributions: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        if split == "test":
            distributions[split] = {
                "sample_count": len(split_rows),
                "labels": "sealed_until_final_evaluation",
                "label_digest": semantic_sha256([row["response"] for row in split_rows]),
            }
            continue
        distributions[split] = {
            "sample_count": len(split_rows),
            "object_tag": dict(
                sorted(
                    Counter(
                        row["response"]["object_tag"]
                        for row in split_rows
                        if row["task"] == "object_attributes"
                    ).items()
                )
            ),
            "object_color": dict(
                sorted(
                    Counter(
                        color
                        for row in split_rows
                        if row["task"] == "object_attributes"
                        for color in row["response"].get("colors", []) or ["<empty>"]
                    ).items()
                )
            ),
            "motion_state": dict(
                sorted(
                    Counter(
                        row["response"]["motion_state"]
                        for row in split_rows
                        if row["task"] == "object_attributes"
                    ).items()
                )
            ),
        }

    exact_input_cross_split = {
        task: _cross_split_groups(
            [row for row in rows if row["task"] == task], _input_signature
        )
        for task in TASK_FILES
    }
    report = {
        "schema_version": "qwen-finetune-v2-data-audit-v1",
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_manifest_sha256": sha256_file(dataset_dir / "split_manifest.json"),
        "all_tasks_sha256": sha256_file(dataset_dir / "all_tasks.jsonl"),
        "sample_count": len(rows),
        "sample_id_unique": not duplicate_ids,
        "duplicate_sample_ids": duplicate_ids,
        "manifest_mismatch_count": len(manifest_mismatches),
        "scene_leakage": {
            scene: sorted(splits) for scene, splits in sorted(scene_splits.items()) if len(splits) > 1
        },
        "task_split_counts": {
            task: dict(sorted(counts.items())) for task, counts in sorted(task_split_counts.items())
        },
        "crop_integrity": {
            "checked": len(image_rows),
            "missing": crop_missing,
            "hash_mismatch": crop_hash_mismatch,
            "decode_errors": crop_decode_errors,
        },
        "cross_split_near_duplicates": {
            "exact_input_group_counts": {
                task: len(groups) for task, groups in exact_input_cross_split.items()
            },
            "exact_input_groups": exact_input_cross_split,
            "perceptual_image_pairs_hamming_le_4": dict(sorted(perceptual_pairs.items())),
            "interpretation": (
                "scene隔离成立，但自动模板指令和候选语义模式存在跨scene重复；该评估衡量既有弱监督分布内能力，不等价于开放世界泛化。"
            ),
        },
        "label_proxy_leakage": {
            "object_tag_equals_detection_label": {
                "count": object_tag_proxy,
                "rate": object_tag_proxy / max(1, len(object_rows)),
            },
            "object_motion_equals_metadata": {
                "count": object_motion_proxy,
                "rate": object_motion_proxy / max(1, len(object_rows)),
            },
            "object_colors_lexical_in_caption": {
                "count": object_color_proxy,
                "rate": object_color_proxy / max(1, len(object_rows)),
            },
            "derived_caption_color_candidates_equal_label": {
                "count": caption_color_candidate_proxy,
                "rate": caption_color_candidate_proxy / max(1, len(object_rows)),
            },
            "rerank_correct_target_is_first_input": {
                "count": rerank_first_proxy,
                "rate": rerank_first_proxy / max(1, len(rerank_rows)),
            },
            "rerank_correct_target_has_highest_combined_score": {
                "count": rerank_highest_score_proxy,
                "rate": rerank_highest_score_proxy / max(1, len(rerank_rows)),
            },
            "interpretation": (
                "这些字段是现有系统在部署时可用的弱监督代理，不是人工真值；因此必须同时报告代理泄漏率和系统安全回归。"
            ),
        },
        "label_distributions": distributions,
        "passed_integrity_gates": (
            not duplicate_ids
            and not manifest_mismatches
            and not any(len(splits) > 1 for splits in scene_splits.values())
            and not crop_missing
            and not crop_hash_mismatch
            and not crop_decode_errors
        ),
    }
    report["deterministic_sha256"] = semantic_sha256(report)
    return report


def _safe_prediction(item: dict[str, Any]) -> dict[str, Any] | None:
    value = item.get("prediction_json")
    return value if isinstance(value, dict) else None


def _classification_macro_f1(targets: list[str], predictions: list[str | None]) -> float:
    labels = sorted(set(targets) | {item for item in predictions if item is not None})
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(target == label and prediction == label for target, prediction in zip(targets, predictions))
        fp = sum(target != label and prediction == label for target, prediction in zip(targets, predictions))
        fn = sum(target == label and prediction != label for target, prediction in zip(targets, predictions))
        scores.append(2 * tp / max(1, 2 * tp + fp + fn))
    return sum(scores) / len(scores)


def _multilabel_macro_f1(targets: list[set[str]], predictions: list[set[str]]) -> float:
    labels = sorted(set().union(*targets, *predictions)) if targets or predictions else []
    if not labels:
        return float(all(not item for item in targets) and all(not item for item in predictions))
    scores = []
    for label in labels:
        tp = sum(label in target and label in prediction for target, prediction in zip(targets, predictions))
        fp = sum(label not in target and label in prediction for target, prediction in zip(targets, predictions))
        fn = sum(label in target and label not in prediction for target, prediction in zip(targets, predictions))
        scores.append(2 * tp / max(1, 2 * tp + fp + fn))
    return sum(scores) / len(scores)


def _value_f1(target: Any, prediction: Any) -> float:
    if isinstance(target, list):
        target_set = {canonical_json(item) for item in target}
        prediction_set = {
            canonical_json(item) for item in prediction
        } if isinstance(prediction, list) else set()
        if not target_set and not prediction_set:
            return 1.0
        overlap = len(target_set & prediction_set)
        return 2 * overlap / max(1, len(target_set) + len(prediction_set))
    if isinstance(target, str) or target is None:
        return float(normalize_text(target) == normalize_text(prediction))
    return float(target == prediction)


def evaluate_prediction_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_task = {
        task: [item for item in items if item["task"] == task] for task in TASK_FILES
    }
    overall_schema = sum(bool(item.get("schema_valid")) for item in items) / max(1, len(items))
    per_task_schema = {
        task: sum(bool(item.get("schema_valid")) for item in task_items) / max(1, len(task_items))
        for task, task_items in by_task.items()
    }

    query_items = by_task["query_parse"]
    query_field_scores = {}
    for field in QUERY_SEMANTIC_FIELDS:
        query_field_scores[field] = sum(
            _value_f1(item["target_json"].get(field), (_safe_prediction(item) or {}).get(field))
            for item in query_items
        ) / max(1, len(query_items))
    query_macro_f1 = sum(query_field_scores.values()) / max(1, len(query_field_scores))
    query_exact = sum(
        all(
            _value_f1(item["target_json"].get(field), (_safe_prediction(item) or {}).get(field)) == 1.0
            for field in QUERY_SEMANTIC_FIELDS
        )
        for item in query_items
    ) / max(1, len(query_items))

    object_items = by_task["object_attributes"]
    object_targets = [item["target_json"] for item in object_items]
    object_predictions = [_safe_prediction(item) or {} for item in object_items]
    object_category_f1 = _classification_macro_f1(
        [item["object_tag"] for item in object_targets],
        [item.get("object_tag") for item in object_predictions],
    )
    object_color_f1 = _multilabel_macro_f1(
        [set(item.get("colors", [])) for item in object_targets],
        [set(item.get("colors", [])) for item in object_predictions],
    )
    object_motion_f1 = _classification_macro_f1(
        [item["motion_state"] for item in object_targets],
        [item.get("motion_state") for item in object_predictions],
    )

    rerank_items = by_task["candidate_rerank"]
    top1_values = []
    reciprocal_ranks = []
    legality_values = []
    for item in rerank_items:
        prediction = _safe_prediction(item) or {}
        ranking = prediction.get("ranking", []) if isinstance(prediction.get("ranking", []), list) else []
        allowed = [int(value) for value in item["row"].get("candidate_ids", [])]
        correct = int(item["row"]["correct_target_node_id"])
        top1_values.append(bool(ranking) and ranking[0] == correct)
        reciprocal_ranks.append(1.0 / (ranking.index(correct) + 1) if correct in ranking else 0.0)
        legality_values.append(
            bool(item.get("schema_valid"))
            and len(ranking) == len(allowed)
            and len(set(ranking)) == len(ranking)
            and set(ranking) == set(allowed)
        )

    fallback_rate = 1.0 - overall_schema
    metrics = {
        "schema_version": "qwen-finetune-v2-metrics-v1",
        "num_samples": len(items),
        "overall_schema_valid_rate": overall_schema,
        "per_task_schema_valid_rate": per_task_schema,
        "query_parse": {
            "num_samples": len(query_items),
            "semantic_field_f1": query_field_scores,
            "semantic_macro_f1": query_macro_f1,
            "semantic_exact_match": query_exact,
        },
        "object_attributes": {
            "num_samples": len(object_items),
            "category_macro_f1": object_category_f1,
            "color_macro_f1": object_color_f1,
            "motion_macro_f1": object_motion_f1,
        },
        "candidate_rerank": {
            "num_samples": len(rerank_items),
            "top1": sum(top1_values) / max(1, len(top1_values)),
            "mrr": sum(reciprocal_ranks) / max(1, len(reciprocal_ranks)),
            "candidate_legality_rate": sum(legality_values) / max(1, len(legality_values)),
        },
        "schema_or_illegal_fallback_rate": fallback_rate,
    }
    metrics["selection_score"] = selection_score(metrics)
    metrics["hard_gate_checks"] = hard_gate_checks(metrics)
    metrics["all_hard_gates_passed"] = all(metrics["hard_gate_checks"].values())
    return metrics


def selection_score(metrics: dict[str, Any]) -> float:
    values = [
        metrics["overall_schema_valid_rate"],
        min(metrics["per_task_schema_valid_rate"].values()),
        metrics["query_parse"]["semantic_macro_f1"],
        metrics["query_parse"]["semantic_exact_match"],
        metrics["object_attributes"]["category_macro_f1"],
        metrics["object_attributes"]["color_macro_f1"],
        metrics["object_attributes"]["motion_macro_f1"],
        metrics["candidate_rerank"]["top1"],
        metrics["candidate_rerank"]["mrr"],
        metrics["candidate_rerank"]["candidate_legality_rate"],
    ]
    return sum(values) / len(values)


def hard_gate_checks(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "overall_schema_valid_rate": metrics["overall_schema_valid_rate"]
        >= HARD_GATES["overall_schema_valid_rate"],
        "per_task_schema_valid_rate": min(metrics["per_task_schema_valid_rate"].values())
        >= HARD_GATES["per_task_schema_valid_rate"],
        "query_macro_f1": metrics["query_parse"]["semantic_macro_f1"]
        >= HARD_GATES["query_macro_f1"],
        "query_semantic_exact_match": metrics["query_parse"]["semantic_exact_match"]
        >= HARD_GATES["query_semantic_exact_match"],
        "object_category_macro_f1": metrics["object_attributes"]["category_macro_f1"]
        >= HARD_GATES["object_category_macro_f1"],
        "object_color_macro_f1": metrics["object_attributes"]["color_macro_f1"]
        >= HARD_GATES["object_color_macro_f1"],
        "object_motion_macro_f1": metrics["object_attributes"]["motion_macro_f1"]
        >= HARD_GATES["object_motion_macro_f1"],
        "rerank_top1": metrics["candidate_rerank"]["top1"] >= HARD_GATES["rerank_top1"],
        "rerank_mrr": metrics["candidate_rerank"]["mrr"] >= HARD_GATES["rerank_mrr"],
        "rerank_candidate_legality": metrics["candidate_rerank"]["candidate_legality_rate"]
        >= HARD_GATES["rerank_candidate_legality"],
        "schema_or_illegal_fallback_rate": metrics["schema_or_illegal_fallback_rate"]
        <= HARD_GATES["schema_or_illegal_fallback_rate_max"],
    }


def prediction_item(
    *,
    row: dict[str, Any],
    raw_output: str,
    latency_s: float,
) -> dict[str, Any]:
    validation = parse_and_validate_output(
        row["task"], raw_output, allowed_candidate_ids=row.get("candidate_ids")
    )
    parsed, _ = extract_json_object(raw_output)
    return {
        "task": row["task"],
        "scene_id": row["scene_id"],
        "sample_id": row["sample_id"],
        "split": row["split"],
        "raw_output": raw_output,
        "parsed_output": parsed,
        "prediction_json": validation.normalized_output,
        "target_json": row["response"],
        "schema_valid": validation.valid,
        "validation_errors": validation.validation_errors,
        "latency_s": latency_s,
        "row": {
            "candidate_ids": row.get("candidate_ids"),
            "correct_target_node_id": row.get("correct_target_node_id"),
        },
    }


def error_analysis(items: list[dict[str, Any]]) -> dict[str, Any]:
    schema_errors = Counter(
        error["code"]
        for item in items
        for error in item.get("validation_errors", [])
    )
    object_items = [item for item in items if item["task"] == "object_attributes"]
    category_confusion = Counter()
    motion_confusion = Counter()
    for item in object_items:
        prediction = _safe_prediction(item) or {}
        category_confusion[
            f"{item['target_json']['object_tag']} -> {prediction.get('object_tag', '<invalid>')}"
        ] += 1
        motion_confusion[
            f"{item['target_json']['motion_state']} -> {prediction.get('motion_state', '<invalid>')}"
        ] += 1
    failed = [
        {
            "sample_id": item["sample_id"],
            "task": item["task"],
            "scene_id": item["scene_id"],
            "schema_valid": item["schema_valid"],
            "validation_errors": item.get("validation_errors", []),
            "raw_output": item.get("raw_output"),
            "target_json": item.get("target_json"),
        }
        for item in items
        if not item.get("schema_valid")
        or semantic_sha256(item.get("prediction_json")) != semantic_sha256(item.get("target_json"))
    ]
    return {
        "schema_version": "qwen-finetune-v2-error-analysis-v1",
        "schema_error_counts": dict(sorted(schema_errors.items())),
        "object_category_confusion": dict(sorted(category_confusion.items())),
        "object_motion_confusion": dict(sorted(motion_confusion.items())),
        "failed_or_nonexact_count": len(failed),
        "failed_or_nonexact_samples": failed,
    }
