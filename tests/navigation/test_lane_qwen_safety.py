"""Injected backend failures test the real delegate/planner boundary, not GPU quality."""

from __future__ import annotations

import time

import numpy as np
import pytest
from omegaconf import OmegaConf

from conceptgraph.navigation.pipeline import SemanticNavigationPipeline
from conceptgraph.scripts.run_qwen_online_closed_loop import validate_runtime
from conceptgraph.vlm.qwen_semantic_delegate import (
    QwenDelegateConfig,
    QwenSemanticDelegate,
)
from tests.navigation.conftest import make_node
from tests.navigation.test_lane_constraints import _grid, _topology
from tests.vlm.test_qwen_navigation_integration import navigation_cfg


def pipeline(delegate=None, *, blocked=False):
    cfg = navigation_cfg()
    cfg.target_selection = {"mode": "lane_constrained", "top_k": 1}
    result = SemanticNavigationPipeline(cfg, semantic_delegate=delegate)
    grid = _grid()
    if blocked:
        grid.occupancy[:] = 1
    result.set_in_memory_inputs(
        nodes={1: make_node(1, center=(6.0, 5.0), extent=(1.0, 1.0))},
        edges=[],
        occupancy_grid=grid,
    )
    result._lane_topology = _topology()
    return result


@pytest.mark.parametrize(
    "failure", ["exception", "timeout", "invalid_schema", "illegal_candidate"]
)
def test_qwen_failure_cannot_change_lane_geometry_or_path(tmp_path, failure):
    def backend(task, prompt, images, generation_config):
        if failure == "exception":
            raise RuntimeError("injected backend failure")
        if failure == "timeout":
            time.sleep(0.04)
            return "{}"
        return (
            '{"selected_node_id":999,"ranking":[999]}'
            if failure == "illegal_candidate"
            else '{"unexpected":"corrupt"}'
        )

    delegate = QwenSemanticDelegate(
        QwenDelegateConfig(
            enabled=True,
            use_qwen_query_parse=True,
            use_qwen_rerank=True,
            timeout_s=0.005,
            cache_dir=tmp_path / "cache",
        ),
        backend=backend,
    )
    try:
        baseline = pipeline().select(
            "前往小汽车附近",
            np.array([0.0, 2.0, 0.0, 1.0, 0.0]),
            run_semantic_delegate=False,
        )
        actual = pipeline(delegate).select(
            "前往小汽车附近", np.array([0.0, 2.0, 0.0, 1.0, 0.0])
        )
        assert baseline.success and actual.success
        assert actual.plan.path_grid == baseline.plan.path_grid
        assert actual.selected_candidate.debug["lane_planning"]["lane_validation"][
            "valid"
        ]
        assert actual.debug["qwen_query_parse"]["fallback_used"]
        assert delegate.backend_call_count > 0
    finally:
        delegate.close()


def test_blocked_lanes_cannot_be_bypassed_by_qwen_candidate():
    class UnsafeDelegate:
        def parse_query(self, *args, **kwargs):
            return {
                "used_qwen": False,
                "fallback_used": True,
                "failure_code": "injected",
            }

        def rerank_candidates(self, *args, **kwargs):
            raise AssertionError("no verified path: rerank must not be called")

    result = pipeline(UnsafeDelegate(), blocked=True).select(
        "前往小汽车附近", np.array([0.0, 2.0, 0.0, 1.0, 0.0])
    )
    assert not result.success and result.plan is None
    assert result.failure_code == "NO_LEGAL_LANE_START"


def test_invalid_smoothing_is_rejected_and_lane_search_repeated(monkeypatch):
    import conceptgraph.navigation.target_selector as selector

    original = selector.lane_astar_plan
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(selector, "lane_astar_plan", counted)
    monkeypatch.setattr(
        selector, "smooth_grid_path", lambda *args, **kwargs: [(2, 0), (0, 4), (2, 7)]
    )
    result = pipeline().select(
        "前往小汽车附近",
        np.array([0.0, 2.0, 0.0, 1.0, 0.0]),
        run_semantic_delegate=False,
    )
    assert result.success and len(calls) >= 2
    audit = result.selected_candidate.debug["lane_planning"]
    assert audit["smoothing_rejected"] and audit["replanned_without_smoothing"]
    assert audit["lane_validation"]["valid"]


def test_missing_lane_map_fails_without_ordinary_fallback(tmp_path):
    candidate = pipeline()
    candidate._lane_topology = None
    candidate.cfg.lane_map_path = str(tmp_path / "missing.npz")
    result = candidate.select(
        "前往小汽车附近",
        np.array([0.0, 2.0, 0.0, 1.0, 0.0]),
        run_semantic_delegate=False,
    )
    assert not result.success and result.failure_code == "NO_LANE_MAP"


def test_gpu_unavailable_is_explicit_failure(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="visible CUDA GPU"):
        validate_runtime(OmegaConf.create({"qwen_online": {}}))


def test_cached_or_invalid_output_never_counts_as_fresh_gpu_success(tmp_path):
    import json
    from conceptgraph.scripts.lane_experiment_metrics import qwen_audit_metrics

    (tmp_path / "invalid.json").write_text(
        json.dumps(
            {
                "result": {
                    "used_qwen": False,
                    "output_schema_valid": False,
                    "fallback_used": True,
                    "failure_code": "SCHEMA_VALIDATION_FAILED",
                }
            }
        )
    )
    (tmp_path / "cached.json").write_text(
        json.dumps(
            {
                "result": {
                    "used_qwen": True,
                    "output_schema_valid": True,
                    "cache": {"status": "hit", "hit": True},
                }
            }
        )
    )
    metrics = qwen_audit_metrics(tmp_path, 1)
    assert metrics["backend_call_count"] == 1
    assert metrics["successful_backend_response_count"] == 0
    assert metrics["cache_hit_count"] == 1 and metrics["fallback_count"] == 1


def test_lane_camera_forward_axis_matches_mapping_and_hydra_config():
    from conceptgraph.navigation.online import OnlineSemanticNavigator

    cfg = pipeline().cfg
    navigator = OnlineSemanticNavigator(cfg, base_occupancy_grid=_grid())
    try:
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        assert np.allclose(navigator._normalize_vehicle_pose(camera_pose)[3:5], [1, 0])
        cfg.target_selection = {"mode": "joint_utility"}
        assert np.allclose(navigator._normalize_vehicle_pose(camera_pose)[3:5], [0, 1])
    finally:
        navigator.close()
