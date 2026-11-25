from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from PIL import Image

from conceptgraph.vlm.qwen_semantic_delegate import QwenDelegateConfig, QwenSemanticDelegate
from conceptgraph.vlm.schemas import SCHEMA_VERSION


def valid_query_output() -> str:
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "target_text": "车辆",
            "target_categories": ["vehicle"],
            "target_colors": [],
            "anchor_text": None,
            "anchor_categories": [],
            "anchor_colors": [],
            "relation": "near",
            "prefer_nearest": True,
            "prefer_front": False,
            "confidence": 0.9,
        },
        ensure_ascii=False,
    )


def valid_object_output(color: str = "gray") -> str:
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "summary": "车辆对象",
            "object_tag": "car",
            "possible_tags": ["car"],
            "colors": [color],
            "body_type": "sedan",
            "motion_state": "static",
            "confidence": 0.8,
        }
    )


class Backend:
    def __init__(self, output=valid_query_output(), *, delay=0.0, error=None):
        self.output = output
        self.delay = delay
        self.error = error
        self.calls = []

    def __call__(self, task, prompt, images, generation_config):
        self.calls.append(
            {
                "task": task,
                "image_count": len(images or []),
                "generation_config": generation_config,
            }
        )
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.output


def make_delegate(tmp_path: Path, backend, **overrides) -> QwenSemanticDelegate:
    values = {
        "enabled": True,
        "use_qwen_query_parse": True,
        "use_qwen_object_attributes": True,
        "use_qwen_rerank": True,
        "cache_dir": tmp_path / "cache",
        "model_revision": "test-model-v1",
        "adapter_revision": "test-adapter-v1",
        "timeout_s": 0.2,
    }
    values.update(overrides)
    return QwenSemanticDelegate(QwenDelegateConfig(**values), backend=backend)


def test_disabled_qwen_uses_rule_result_without_loading_backend(tmp_path) -> None:
    backend = Backend()
    delegate = QwenSemanticDelegate(
        QwenDelegateConfig(enabled=False, use_qwen_query_parse=True, cache_dir=tmp_path),
        backend=backend,
    )

    result = delegate.parse_query("前往最近的车辆附近")
    delegate.close()

    assert result["used_qwen"] is False
    assert result["failure_code"] == "QWEN_DISABLED"
    assert result["target_categories"] == ["vehicle"]
    assert backend.calls == []


@pytest.mark.parametrize(
    ("backend", "failure_code"),
    [
        (Backend(error=RuntimeError("boom")), "BACKEND_EXCEPTION"),
        (Backend(output=""), "EMPTY_OUTPUT"),
        (Backend(output='{"schema_version":"1.0.0"}'), "SCHEMA_VALIDATION_FAILED"),
    ],
)
def test_backend_exception_empty_and_schema_error_fall_back(tmp_path, backend, failure_code) -> None:
    delegate = make_delegate(tmp_path, backend)

    result = delegate.parse_query("前往最近的车辆附近")
    delegate.close()

    assert result["used_qwen"] is False
    assert result["fallback_used"] is True
    assert result["failure_code"] == failure_code
    assert result["target_categories"] == ["vehicle"]


def test_backend_exception_path_is_sanitized_from_audit(tmp_path) -> None:
    backend = Backend(error=RuntimeError(f"failed to load {tmp_path}/private/model.bin"))
    log_dir = tmp_path / "audit"
    delegate = make_delegate(tmp_path, backend, log_dir=log_dir)

    result = delegate.parse_query("前往车辆附近")
    delegate.close()
    audit_text = next(log_dir.glob("*.json")).read_text(encoding="utf-8")

    assert result["failure_code"] == "BACKEND_EXCEPTION"
    assert str(tmp_path) not in result["fallback_reason"]
    assert str(tmp_path) not in audit_text


def test_timeout_returns_quickly_and_falls_back(tmp_path) -> None:
    backend = Backend(delay=0.12)
    delegate = make_delegate(tmp_path, backend, timeout_s=0.02)

    started = time.perf_counter()
    result = delegate.parse_query("前往车辆附近")
    elapsed = time.perf_counter() - started
    delegate.close(wait=True)

    assert elapsed < 0.1
    assert result["failure_code"] == "BACKEND_TIMEOUT"
    assert result["fallback_used"]


def test_illegal_candidate_id_falls_back_to_verified_rule_order(tmp_path) -> None:
    backend = Backend(
        output=json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "selected_node_id": 99,
                "ranking": [99, 2],
                "explanation": "invalid",
                "confidence": 0.9,
            }
        )
    )
    delegate = make_delegate(tmp_path, backend)

    result = delegate.rerank_candidates("前往车辆附近", [{"node_id": 2}, {"node_id": 5}])
    delegate.close()

    assert result["failure_code"] == "ILLEGAL_CANDIDATE_ID"
    assert result["selected_node_id"] == 2
    assert result["ranking"] == [2, 5]


def test_cache_generation_budget_change_requires_new_backend_request(tmp_path) -> None:
    backend=Backend()
    first=make_delegate(tmp_path,backend,max_new_tokens=128)
    a=first.parse_query("前往最近的车辆附近")
    first.close()
    second=make_delegate(tmp_path,backend,max_new_tokens=256)
    b=second.parse_query("前往最近的车辆附近")
    c=second.parse_query("前往最近的车辆附近")
    second.close()
    assert a['cache']['key']!=b['cache']['key']
    assert len(backend.calls)==2 and c['cache']['hit']


def test_cache_hit_avoids_second_backend_call_and_corruption_recomputes(tmp_path) -> None:
    backend = Backend()
    delegate = make_delegate(tmp_path, backend)

    first = delegate.parse_query("前往最近的车辆附近")
    second = delegate.parse_query("前往最近的车辆附近")
    cache_path = delegate.cache.path_for_key(first["cache"]["key"])
    cache_path.write_text("broken", encoding="utf-8")
    third = delegate.parse_query("前往最近的车辆附近")
    delegate.close()

    assert first["cache"]["hit"] is False
    assert second["cache"]["hit"] is True
    assert third["cache"]["status"] == "corrupt_written"
    assert len(backend.calls) == 2


def test_model_version_and_image_content_change_invalidate_cache(tmp_path) -> None:
    crop = tmp_path / "crop.png"
    Image.new("RGB", (8, 8), "red").save(crop)
    backend = Backend(output=valid_object_output("red"))
    delegate = make_delegate(tmp_path, backend)
    metadata = {"object_tag": "car", "colors": ["red"], "motion_state": "static"}

    first = delegate.describe_object([str(crop)], metadata)
    Image.new("RGB", (8, 8), "blue").save(crop)
    backend.output = valid_object_output("blue")
    second = delegate.describe_object([str(crop)], metadata)
    delegate.close()

    other = make_delegate(tmp_path, Backend(), model_revision="test-model-v2")
    third = other.parse_query("前往最近的车辆附近")
    other.close()

    assert first["cache"]["key"] != second["cache"]["key"]
    assert len(backend.calls) == 2
    assert third["cache"]["hit"] is False


def test_generation_is_fixed_and_sampling_is_rejected(tmp_path) -> None:
    backend = Backend()
    delegate = make_delegate(tmp_path, backend)
    delegate.parse_query("前往车辆附近")
    delegate.close()

    assert backend.calls[0]["generation_config"]["do_sample"] is False
    with pytest.raises(ValueError, match="do_sample"):
        QwenDelegateConfig(do_sample=True)


def test_object_images_reach_backend_and_audit_omits_absolute_input_paths(tmp_path) -> None:
    crop = tmp_path / "crop.png"
    Image.new("RGB", (12, 10), "gray").save(crop)
    backend = Backend(output=valid_object_output())
    log_dir = tmp_path / "audit"
    delegate = make_delegate(tmp_path, backend, log_dir=log_dir)

    result = delegate.describe_object(
        [str(crop)],
        {"object_tag": "car", "colors": ["gray"], "motion_state": "static"},
    )
    delegate.close()
    audit_text = next(log_dir.glob("*.json")).read_text(encoding="utf-8")

    assert result["used_qwen"]
    assert backend.calls[0]["image_count"] == 1
    assert str(tmp_path) not in audit_text


def test_structured_stage3_relation_is_never_overridden(tmp_path) -> None:
    backend = Backend(output='{"relation":"right_of","confidence":1.0}')
    delegate = make_delegate(tmp_path, backend, use_qwen_relation=True)

    result = delegate.infer_relation(
        {"xy": [0, 0]},
        {"xy": [1, 0]},
        structured_fact={"relation": "left_of", "confidence": 0.91},
    )
    delegate.close()

    assert result["relation"] == "left_of"
    assert result["confidence"] == pytest.approx(0.91)
    assert backend.calls == []


def test_async_interface_does_not_block_event_loop_and_closes_cleanly(tmp_path) -> None:
    async def scenario():
        backend = Backend(delay=0.05)
        delegate = make_delegate(tmp_path, backend, timeout_s=0.2)
        task = asyncio.create_task(delegate.parse_query_async("前往车辆附近"))
        ticked = False
        await asyncio.sleep(0.005)
        ticked = True
        result = await task
        await delegate.aclose()
        return ticked, result, delegate.pending_async_count

    ticked, result, pending = asyncio.run(scenario())

    assert ticked
    assert result["used_qwen"]
    assert pending == 0


def test_async_cancellation_and_timeout_leave_no_tracked_tasks(tmp_path) -> None:
    async def scenario():
        delegate = make_delegate(tmp_path, Backend(delay=0.08), timeout_s=0.01)
        task = asyncio.create_task(delegate.parse_query_async("前往车辆附近"))
        await asyncio.sleep(0.002)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await delegate.aclose()
        return delegate.pending_async_count

    assert asyncio.run(scenario()) == 0
