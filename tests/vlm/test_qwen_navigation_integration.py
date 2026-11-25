from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from conceptgraph.navigation.pipeline import SemanticNavigationPipeline
from conceptgraph.scripts.build_qwen_finetune_dataset import (
    audit_scene_leakage,
    deterministic_scene_split,
)
from conceptgraph.vlm.qwen_semantic_delegate import QwenDelegateConfig, QwenSemanticDelegate
from conceptgraph.vlm.schemas import SCHEMA_VERSION
from scripts.train_qwen_lora_semantic_delegate import load_samples, prepare_training_inputs
from tests.navigation.conftest import make_grid, make_node


class FixturePipeline(SemanticNavigationPipeline):
    def __init__(self, cfg, nodes, delegate=None):
        super().__init__(cfg, semantic_delegate=delegate)
        self.fixture_nodes = nodes
        self.fixture_grid = make_grid(shape=(32, 32))

    def _load_nodes(self):
        return self.fixture_nodes

    def _load_edges(self):
        return []

    def _get_occupancy_grid(self, nodes):
        return self.fixture_grid


def navigation_cfg():
    return SimpleNamespace(
        default_near_distance_m=8.0,
        lambda_text=0.55,
        lambda_relation=0.30,
        lambda_distance_prior=0.25,
        start_search_radius_m=3.0,
        goal_search_radius_m=5.0,
        allow_goal_fallback=False,
        semantic_cost_weight=0.0,
        clearance_cost_weight=0.0,
        path_smoothing_iterations=1,
        max_candidates=3,
        target_selection=SimpleNamespace(mode="joint_utility", top_k=3),
    )


class CountingDisabledDelegate:
    def __init__(self):
        self.parse_calls = 0
        self.rerank_calls = 0
        self.rerank_inputs = []

    def parse_query(self, query_text, *, default_near_distance_m, rule_result):
        self.parse_calls += 1
        return {
            "used_qwen": False,
            "fallback_used": True,
            "failure_code": "QWEN_DISABLED",
            "normalized_output": None,
        }

    def rerank_candidates(self, query_text, candidates, edges):
        self.rerank_calls += 1
        self.rerank_inputs.append(candidates)
        ids = [candidate["node_id"] for candidate in candidates]
        return {
            "used_qwen": False,
            "fallback_used": True,
            "failure_code": "QWEN_DISABLED",
            "selected_node_id": ids[0],
            "ranking": ids,
            "normalized_output": {"selected_node_id": ids[0], "ranking": ids},
        }


def test_navigation_calls_query_parse_once_and_disabled_result_matches_rule_baseline() -> None:
    nodes = {
        1: make_node(1, center=(10.0, 10.0), tag="car"),
        2: make_node(2, center=(20.0, 20.0), tag="car"),
    }
    pose = np.asarray([2.0, 2.0, 0.0])
    baseline = FixturePipeline(navigation_cfg(), nodes).select("前往小汽车附近", pose)
    delegate = CountingDisabledDelegate()
    qwen_disabled = FixturePipeline(navigation_cfg(), nodes, delegate).select(
        "前往小汽车附近", pose
    )

    assert baseline.success and qwen_disabled.success
    assert delegate.parse_calls == 1
    assert baseline.selected_candidate.node_id == qwen_disabled.selected_candidate.node_id
    assert baseline.selected_candidate.target_grid == qwen_disabled.selected_candidate.target_grid
    assert baseline.selected_candidate.path_cost == qwen_disabled.selected_candidate.path_cost


def test_qwen_query_fields_convert_to_existing_parsed_navigation_query() -> None:
    nodes = {
        1: make_node(1, center=(10.0, 10.0), tag="car"),
        2: make_node(2, center=(14.0, 14.0), tag="truck"),
    }

    class SemanticDelegate(CountingDisabledDelegate):
        def parse_query(self, query_text, *, default_near_distance_m, rule_result):
            self.parse_calls += 1
            return {
                "used_qwen": True,
                "fallback_used": False,
                "normalized_output": {
                    "schema_version": SCHEMA_VERSION,
                    "target_text": "卡车",
                    "target_categories": ["truck"],
                    "target_colors": [],
                    "anchor_text": None,
                    "anchor_categories": [],
                    "anchor_colors": [],
                    "relation": "near",
                    "prefer_nearest": False,
                    "prefer_front": False,
                    "confidence": 0.9,
                },
            }

    delegate = SemanticDelegate()
    result = FixturePipeline(navigation_cfg(), nodes, delegate).select(
        "前往目标附近", np.asarray([2.0, 2.0, 0.0])
    )

    assert result.success
    assert delegate.parse_calls == 1
    assert result.plan.query.target_categories == ["truck"]
    assert result.selected_candidate.node_id == 2


def test_rerank_receives_only_path_validated_candidates_and_cannot_change_route() -> None:
    nodes = {
        1: make_node(1, center=(10.0, 10.0), tag="car"),
        2: make_node(2, center=(16.0, 16.0), tag="car"),
        3: make_node(3, center=(80.0, 80.0), tag="car"),
    }
    delegate = CountingDisabledDelegate()
    result = FixturePipeline(navigation_cfg(), nodes, delegate).select(
        "前往小汽车附近", np.asarray([2.0, 2.0, 0.0])
    )

    assert result.success
    authoritative = result.selected_candidate.node_id
    assert delegate.rerank_calls == 1
    assert all(candidate["path_validated"] for candidate in delegate.rerank_inputs[0])
    assert 3 not in {candidate["node_id"] for candidate in delegate.rerank_inputs[0]}
    assert result.selected_candidate.node_id == authoritative
    assert result.debug["qwen_candidate_rerank"]["selection_applied"] is False


def test_illegal_qwen_rerank_id_falls_back_without_bypassing_joint_utility(tmp_path) -> None:
    query_output = {
        "schema_version": SCHEMA_VERSION,
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
    }

    def backend(task, prompt, images, generation_config):
        if task == "query_parse":
            return json.dumps(query_output)
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "selected_node_id": 999,
                "ranking": [999],
                "explanation": "越界",
                "confidence": 1.0,
            }
        )

    delegate = QwenSemanticDelegate(
        QwenDelegateConfig(
            enabled=True,
            use_qwen_query_parse=True,
            use_qwen_rerank=True,
            model_revision="test",
            timeout_s=1.0,
            cache_dir=tmp_path / "cache",
        ),
        backend=backend,
    )
    nodes = {
        1: make_node(1, center=(10.0, 10.0), tag="car"),
        2: make_node(2, center=(16.0, 16.0), tag="car"),
    }
    result = FixturePipeline(navigation_cfg(), nodes, delegate).select(
        "前往小汽车附近", np.asarray([2.0, 2.0, 0.0])
    )
    delegate.close()

    audit = result.debug["qwen_candidate_rerank"]
    assert result.success
    assert audit["failure_code"] == "ILLEGAL_CANDIDATE_ID"
    assert audit["selected_node_id"] == result.selected_candidate.node_id
    assert audit["selection_applied"] is False


def test_scene_split_is_deterministic_and_has_no_leakage() -> None:
    scenes = [f"scene-{index:04d}" for index in range(11)]
    first = deterministic_scene_split(scenes, 41, (0.7, 0.15, 0.15))
    second = deterministic_scene_split(list(reversed(scenes)), 41, (0.7, 0.15, 0.15))
    rows = {
        "query_parse": [
            {"scene_id": scene, "split": split, "sample_id": scene}
            for scene, split in first.items()
        ],
        "object_attributes": [],
    }

    audit = audit_scene_leakage(rows, first)

    assert first == second
    assert set(first.values()) == {"train", "validation", "test"}
    assert audit["passed"]
    assert audit["scene_leakage_count"] == 0


@pytest.mark.skipif(
    not all((Path('outputs/project_v2/qwen_eval/dataset') / name).is_file()
            for name in ['candidate_rerank.jsonl', 'object_attributes.jsonl']),
    reason='需要服务器上的真实阶段4数据与crop；公开代码仓库不分发这些原始数据',
)
def test_stage4_dataset_has_real_candidates_and_real_decodable_crops() -> None:
    dataset_dir = Path("outputs/project_v2/qwen_eval/dataset")
    rerank = json.loads((dataset_dir / "candidate_rerank.jsonl").read_text(encoding="utf-8").splitlines()[0])
    object_row = json.loads((dataset_dir / "object_attributes.jsonl").read_text(encoding="utf-8").splitlines()[0])

    with Image.open(object_row["crop_path"]) as image:
        image.load()
        size = image.size

    assert rerank["candidate_nodes"]
    assert rerank["correct_target_node_id"] in rerank["candidate_ids"]
    assert all("stage2_scores" in candidate for candidate in rerank["candidate_nodes"])
    assert rerank["relation_metadata_source"] == "stage3_structured_edges"
    assert object_row["image_sha256"]
    assert size[0] > 0 and size[1] > 0


@pytest.mark.skipif(
    not (Path('outputs/project_v2/qwen_eval/dataset') / 'object_attributes.jsonl').is_file(),
    reason='需要服务器上的真实阶段4数据与crop；公开代码仓库不分发这些原始数据',
)
def test_real_crop_is_passed_to_multimodal_processor() -> None:
    dataset_dir = Path("outputs/project_v2/qwen_eval/dataset")
    row = json.loads((dataset_dir / "object_attributes.jsonl").read_text(encoding="utf-8").splitlines()[0])
    sample = {
        "task": "object_attributes",
        "scene_id": row["scene_id"],
        "sample_id": row["sample_id"],
        "prompt": "对象属性",
        "response": json.dumps(row["response"], ensure_ascii=False),
        "row": row,
    }

    class FakeProcessor:
        def __init__(self):
            self.image_calls = []

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "full answer" if len(messages) == 2 else "prompt"

        def __call__(self, *, text, images, **kwargs):
            self.image_calls.append(images)
            length = 12 if text[0] == "full answer" else 8
            result = {
                "input_ids": torch.arange(length).reshape(1, -1),
                "attention_mask": torch.ones((1, length), dtype=torch.long),
            }
            if images:
                assert isinstance(images[0], Image.Image)
                result["pixel_values"] = torch.ones((4, 3))
                result["image_grid_thw"] = torch.tensor([[1, 2, 2]])
            return result

    processor = FakeProcessor()
    tensors = prepare_training_inputs(processor, sample, max_length=128)

    assert len(processor.image_calls) == 2
    assert all(call and isinstance(call[0], Image.Image) for call in processor.image_calls)
    assert "pixel_values" in tensors and "image_grid_thw" in tensors
    assert int((tensors["labels"] != -100).sum()) == 4


def test_training_loader_rejects_samples_missing_from_manifest(tmp_path) -> None:
    row = {
        "task": "query_parse",
        "scene_id": "scene-0001",
        "sample_id": "scene-0001:query_parse:query-0000",
        "split": "train",
        "query_text": "前往车辆附近",
        "response": {
            "schema_version": SCHEMA_VERSION,
            "target_text": "车辆",
            "target_categories": ["vehicle"],
            "target_colors": [],
            "anchor_text": None,
            "anchor_categories": [],
            "anchor_colors": [],
            "relation": "near",
            "prefer_nearest": False,
            "prefer_front": False,
            "confidence": 0.7,
        },
    }
    (tmp_path / "query_parse.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "stage4-split-manifest-v1",
        "scene_to_split": {"scene-0001": "train"},
        "samples": {"query_parse": {"train": [], "validation": [], "test": []}},
    }

    with pytest.raises(RuntimeError, match="sample不在split manifest"):
        load_samples(
            tmp_path,
            ["query_parse"],
            split="train",
            manifest=manifest,
        )
