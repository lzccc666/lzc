from __future__ import annotations

import json

import pytest
from PIL import Image

from conceptgraph.vlm.qwen_finetune_v2 import (
    build_prompt,
    caption_color_candidates,
    compact_relations,
    evaluate_prediction_items,
    prediction_item,
)
from conceptgraph.vlm.qwen_semantic_delegate import QwenDelegateConfig, QwenSemanticDelegate
from conceptgraph.vlm.qwen_vl_chat import QwenVLChat
from scripts.train_qwen_lora_semantic_delegate_v2 import augment_color_copy_contracts


def _query_row() -> dict:
    response = {
        "schema_version": "1.0.0",
        "target_text": "最近的车辆",
        "target_categories": ["vehicle"],
        "target_colors": [],
        "anchor_text": None,
        "anchor_categories": [],
        "anchor_colors": [],
        "relation": "near",
        "prefer_nearest": True,
        "prefer_front": False,
        "confidence": 0.7,
    }
    return {
        "task": "query_parse",
        "scene_id": "scene-a",
        "sample_id": "query-1",
        "split": "validation",
        "query_text": "前往最近的车辆附近",
        "response": response,
    }


def _object_row() -> dict:
    response = {
        "schema_version": "1.0.0",
        "summary": "gray car",
        "object_tag": "car",
        "possible_tags": ["car"],
        "colors": ["gray"],
        "body_type": None,
        "motion_state": "static",
        "confidence": 0.7,
    }
    return {
        "task": "object_attributes",
        "scene_id": "scene-a",
        "sample_id": "object-1",
        "split": "validation",
        "metadata": {
            "caption": "gray car",
            "detection_label": "car",
            "motion_state": "static",
        },
        "response": response,
    }


def _rerank_row() -> dict:
    response = {
        "schema_version": "1.0.0",
        "selected_node_id": 2,
        "ranking": [2, 1],
        "explanation": "semantic ranking",
        "confidence": 0.7,
    }
    return {
        "task": "candidate_rerank",
        "scene_id": "scene-a",
        "sample_id": "rerank-1",
        "split": "validation",
        "query_text": "最近的车辆",
        "candidate_ids": [2, 1],
        "correct_target_node_id": 2,
        "candidate_nodes": [
            {"node_id": 2, "object_tag": "car", "stage2_scores": {"text": 0.9}},
            {"node_id": 1, "object_tag": "truck", "stage2_scores": {"text": 0.8}},
        ],
        "relation_metadata": [],
        "response": response,
    }


def test_compact_relations_keeps_only_candidate_incident_edges() -> None:
    relations = [
        {"source": 1, "target": 2, "relation": "near", "confidence": 0.5, "distance_m": 2},
        {"source": 8, "target": 9, "relation": "near", "confidence": 0.9, "distance_m": 1},
    ]
    compact = compact_relations(relations, [1, 2])
    assert compact == [
        {
            "source": 1,
            "target": 2,
            "relation": "near",
            "directed": None,
            "confidence": 0.5,
            "distance_m": 2.0,
        }
    ]


def test_caption_color_candidates_preserves_tied_slash_colors() -> None:
    assert caption_color_candidates("gray/red gray/red gray/red truck") == ["gray", "red"]
    assert caption_color_candidates("green/black green/black traffic sign") == [
        "green",
        "black",
    ]
    assert caption_color_candidates("black black gray truck") == ["black"]


def test_finetune_v2_delegate_profile_uses_training_prompt_and_no_cache(tmp_path) -> None:
    captured = {}

    def backend(task, prompt, images, generation_config):
        captured[task] = prompt
        return json.dumps(
            {
                "schema_version": "1.0.0",
                "target_text": "小汽车",
                "target_categories": ["car"],
                "target_colors": [],
                "anchor_text": None,
                "anchor_categories": [],
                "anchor_colors": [],
                "relation": "near",
                "prefer_nearest": False,
                "prefer_front": False,
                "confidence": 0.9,
            },
            ensure_ascii=False,
        )

    delegate = QwenSemanticDelegate(
        QwenDelegateConfig(
            enabled=True,
            use_qwen_query_parse=True,
            prompt_profile="finetune_v2",
            cache_dir=None,
            log_dir=tmp_path / "audit",
        ),
        backend=backend,
    )
    result = delegate.parse_query("前往小汽车附近")
    delegate.close()

    assert result["output_schema_valid"]
    assert result["cache"]["enabled"] is False
    assert captured["query_parse"] == build_prompt(
        "query_parse", {"query_text": "前往小汽车附近"}
    )


def test_color_copy_augmentation_covers_ordered_canonical_pairs() -> None:
    source = _object_row()
    source["response"]["colors"] = ["green", "black"]
    augmented = augment_color_copy_contracts([source])
    synthetic = [row for row in augmented if row.get("synthetic_color_copy")]

    assert len(synthetic) == 90
    red_gray = next(row for row in synthetic if row["response"]["colors"] == ["red", "gray"])
    assert caption_color_candidates(red_gray["metadata"]["caption"]) == ["red", "gray"]
    assert "colors_required_exact" in build_prompt("object_attributes", red_gray)


def test_chat_image_resize_matches_finetune_preprocessing() -> None:
    backend = object.__new__(QwenVLChat)
    backend.max_image_side = 448
    source = Image.new("RGB", (1920, 1056))

    messages = backend._messages("prompt", [source])
    resized = messages[0]["content"][0]["image"]

    assert resized.size == (448, 246)
    assert source.size == (1920, 1056)


@pytest.mark.parametrize('size',[(1,1),(5,21),(2,1000),(1200,3)])
def test_tiny_real_crop_is_padded_for_qwen_patch_size_without_changing_source(size):
    from conceptgraph.vlm.image_input import prepare_qwen_image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    source=Image.new('RGB',size,'red')
    prepared=prepare_qwen_image(source,448)
    assert min(prepared.size)>=28 and max(prepared.size)<=448
    assert (255,0,0) in {color for count,color in prepared.getcolors(maxcolors=448*448)}
    smart_resize(prepared.height,prepared.width)
    assert source.size==size


@pytest.mark.parametrize("row", [_query_row(), _object_row(), _rerank_row()])
def test_prompt_has_single_json_contract_and_no_schema_document(row: dict) -> None:
    prompt = build_prompt(row["task"], row)
    assert "schema_version" in prompt
    assert "只输出一个JSON对象" in prompt
    assert "JSON Schema" not in prompt


def test_metrics_count_schema_failure_without_rule_fallback_credit() -> None:
    query = _query_row()
    object_row = _object_row()
    rerank = _rerank_row()
    items = [
        prediction_item(
            row=query,
            raw_output=json.dumps(query["response"], ensure_ascii=False),
            latency_s=0.1,
        ),
        prediction_item(row=object_row, raw_output="not-json", latency_s=0.1),
        prediction_item(
            row=rerank,
            raw_output=json.dumps(
                {
                    **rerank["response"],
                    "selected_node_id": 999,
                    "ranking": [999, 2, 1],
                }
            ),
            latency_s=0.1,
        ),
    ]
    metrics = evaluate_prediction_items(items)
    assert metrics["overall_schema_valid_rate"] == pytest.approx(1 / 3)
    assert metrics["schema_or_illegal_fallback_rate"] == pytest.approx(2 / 3)
    assert metrics["candidate_rerank"]["candidate_legality_rate"] == 0.0
    assert metrics["candidate_rerank"]["top1"] == 0.0
    assert not metrics["all_hard_gates_passed"]


def test_perfect_raw_predictions_pass_all_metric_gates() -> None:
    rows = [_query_row(), _object_row(), _rerank_row()]
    items = [
        prediction_item(
            row=row,
            raw_output=json.dumps(row["response"], ensure_ascii=False),
            latency_s=0.1,
        )
        for row in rows
    ]
    metrics = evaluate_prediction_items(items)
    assert metrics["all_hard_gates_passed"]
    assert all(metrics["hard_gate_checks"].values())
