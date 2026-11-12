from __future__ import annotations

import json

import pytest

from conceptgraph.vlm.schemas import (
    SCHEMA_VERSION,
    extract_json_object,
    parse_and_validate_output,
    schema_contract,
)


def query_payload(**overrides):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "target_text": "车辆",
        "target_categories": ["vehicle"],
        "target_colors": ["gray"],
        "anchor_text": None,
        "anchor_categories": [],
        "anchor_colors": [],
        "relation": "near",
        "prefer_nearest": True,
        "prefer_front": False,
        "confidence": 0.9,
    }
    payload.update(overrides)
    return payload


def object_payload(**overrides):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "summary": "灰色小汽车",
        "object_tag": "car",
        "possible_tags": ["car", "vehicle"],
        "colors": ["gray"],
        "body_type": "sedan",
        "motion_state": "static",
        "confidence": 0.8,
    }
    payload.update(overrides)
    return payload


def rerank_payload(**overrides):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "selected_node_id": 3,
        "ranking": [3, 7],
        "explanation": "语义匹配",
        "confidence": 0.75,
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    ("task", "payload", "candidate_ids"),
    [
        ("query_parse", query_payload(), None),
        ("object_attributes", object_payload(), None),
        ("candidate_rerank", rerank_payload(), [3, 7]),
    ],
)
def test_three_task_schemas_accept_valid_payloads(task, payload, candidate_ids) -> None:
    result = parse_and_validate_output(task, json.dumps(payload), allowed_candidate_ids=candidate_ids)

    assert result.valid
    assert result.normalized_output["schema_version"] == SCHEMA_VERSION


def test_markdown_fence_and_surrounding_text_are_extracted() -> None:
    raw = f"模型结果如下：\n```json\n{json.dumps(query_payload(), ensure_ascii=False)}\n```\n结束。"

    result = parse_and_validate_output("query_parse", raw)

    assert result.valid
    assert result.normalized_output["target_categories"] == ["vehicle"]


def test_plain_surrounding_text_is_extracted_but_multiple_objects_are_rejected() -> None:
    single = f"结果：{json.dumps(object_payload(), ensure_ascii=False)}。"
    assert parse_and_validate_output("object_attributes", single).valid

    value, errors = extract_json_object('{"a": 1} 和 {"b": 2}')
    assert value is None
    assert errors[0]["code"] == "ambiguous_json"


@pytest.mark.parametrize(
    ("task", "payload", "error_fragment"),
    [
        ("query_parse", query_payload(extra_field="forbidden"), "extra_forbidden"),
        ("query_parse", query_payload(prefer_nearest=1), "normalization_error"),
        ("object_attributes", object_payload(confidence=1.1), "less_than_equal"),
        ("object_attributes", object_payload(possible_tags="car"), "normalization_error"),
        ("candidate_rerank", rerank_payload(ranking=[3, 3]), "value_error"),
        ("candidate_rerank", rerank_payload(selected_node_id=7), "value_error"),
    ],
)
def test_invalid_types_ranges_duplicates_and_extra_fields_are_rejected(
    task, payload, error_fragment
) -> None:
    result = parse_and_validate_output(task, payload, allowed_candidate_ids=[3, 7])

    assert not result.valid
    assert any(error_fragment in error["code"] for error in result.validation_errors)


def test_relations_colors_categories_and_boolean_words_are_normalized() -> None:
    payload = query_payload(
        target_categories=None,
        target_colors=None,
        relation="front",
        prefer_nearest="否",
        prefer_front="是",
    )
    payload.pop("target_categories")
    payload.pop("target_colors")
    payload["target_category"] = "汽车"
    payload["target_color"] = "灰色"

    result = parse_and_validate_output("query_parse", payload)

    assert result.valid
    assert result.normalized_output["target_categories"] == ["car"]
    assert result.normalized_output["target_colors"] == ["gray"]
    assert result.normalized_output["relation"] == "in_front_of"
    assert result.normalized_output["prefer_nearest"] is False
    assert result.normalized_output["prefer_front"] is True


def test_illegal_candidate_id_is_rejected_and_missing_verified_ids_are_appended() -> None:
    illegal = parse_and_validate_output(
        "candidate_rerank",
        rerank_payload(selected_node_id=99, ranking=[99, 3]),
        allowed_candidate_ids=[3, 7],
    )
    partial = parse_and_validate_output(
        "candidate_rerank",
        rerank_payload(ranking=[3]),
        allowed_candidate_ids=[3, 7],
    )

    assert illegal.validation_errors[0]["code"] == "illegal_candidate_id"
    assert partial.valid
    assert partial.normalized_output["ranking"] == [3, 7]


def test_schema_contracts_forbid_extra_fields_and_require_version() -> None:
    for task in ("query_parse", "object_attributes", "candidate_rerank"):
        contract = schema_contract(task)
        assert contract["additionalProperties"] is False
        assert "schema_version" in contract["required"]
