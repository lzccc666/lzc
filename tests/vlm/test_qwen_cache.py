from __future__ import annotations

import json
from pathlib import Path

import pytest

from conceptgraph.vlm.qwen_cache import QwenDiskCache, canonical_json_bytes, semantic_version


def make_key(cache: QwenDiskCache, **overrides) -> str:
    values = {
        "task": "query_parse",
        "schema_version": "1.0.0",
        "model_id": "model@v1",
        "adapter_id": "adapter@v1",
        "normalized_input": {"query_text": "前往车辆附近", "nested": {"b": 2, "a": 1}},
        "image_hashes": [],
        "object_version": None,
        "candidates": [{"node_id": 1}, {"node_id": 2}],
        "edge_version": "edges-v1",
    }
    values.update(overrides)
    return cache.build_key(**values)


def test_canonical_json_and_cache_key_are_order_independent_and_deterministic(tmp_path) -> None:
    cache = QwenDiskCache(tmp_path)
    first = make_key(cache)
    second = make_key(
        cache,
        normalized_input={"nested": {"a": 1, "b": 2}, "query_text": "前往车辆附近"},
    )

    assert first == second
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    assert semantic_version({"a": 1}) == semantic_version({"a": 1})


@pytest.mark.parametrize(
    "override",
    [
        {"schema_version": "2.0.0"},
        {"model_id": "model@v2"},
        {"adapter_id": "adapter@v2"},
        {"image_hashes": ["abc"]},
        {"object_version": "object-v2"},
        {"candidates": [{"node_id": 2}, {"node_id": 1}]},
        {"edge_version": "edges-v2"},
    ],
)
def test_all_required_version_dimensions_change_the_key(tmp_path, override) -> None:
    cache = QwenDiskCache(tmp_path)
    assert make_key(cache) != make_key(cache, **override)


def test_cache_hit_miss_and_atomic_write(tmp_path) -> None:
    cache = QwenDiskCache(tmp_path)
    key = make_key(cache)

    assert cache.get(key).status == "miss"
    path = cache.put(key, {"normalized_output": {"target": "car"}})
    lookup = cache.get(key)

    assert lookup.hit
    assert lookup.payload == {"normalized_output": {"target": "car"}}
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize(
    ("record", "status"),
    [
        ("not json", "corrupt"),
        (json.dumps({"cache_format_version": "old"}), "version_mismatch"),
        (json.dumps({"cache_format_version": "qwen-semantic-cache-v1"}), "incomplete"),
    ],
)
def test_corrupt_version_mismatch_and_incomplete_entries_are_diagnosed(
    tmp_path, record, status
) -> None:
    cache = QwenDiskCache(tmp_path)
    key = make_key(cache)
    path = cache.path_for_key(key)
    path.parent.mkdir(parents=True)
    path.write_text(record, encoding="utf-8")

    lookup = cache.get(key)

    assert not lookup.hit
    assert lookup.status == status
    assert lookup.diagnostic


def test_payload_checksum_corruption_is_detected(tmp_path) -> None:
    cache = QwenDiskCache(tmp_path)
    key = make_key(cache)
    path = cache.put(key, {"value": 1})
    record = json.loads(path.read_text(encoding="utf-8"))
    record["payload"]["value"] = 2
    path.write_text(json.dumps(record), encoding="utf-8")

    assert cache.get(key).status == "corrupt"


def test_non_finite_values_cannot_enter_cache_keys(tmp_path) -> None:
    cache = QwenDiskCache(tmp_path)

    with pytest.raises(ValueError, match="non-finite"):
        make_key(cache, normalized_input={"score": float("nan")})


def test_path_validation_rejects_non_sha256_keys(tmp_path) -> None:
    cache = QwenDiskCache(tmp_path)

    with pytest.raises(ValueError, match="SHA-256"):
        cache.path_for_key("../escape")

    assert isinstance(cache.path_for_key("a" * 64), Path)
