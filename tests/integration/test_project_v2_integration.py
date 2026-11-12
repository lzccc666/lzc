from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from conceptgraph.scenegraph.relations import (
    SceneGraphEdge,
    edge_relation_for_query,
    load_scene_graph_edges,
    save_structured_edges,
)
from conceptgraph.scripts.run_project_v2_integration_regression import (
    DEFAULT_OUTPUT,
    _matrix_row,
    _measured_metric,
    _run_memory_failure_checks,
    _run_qwen_failure_checks,
    _validate_top_k_diagnostics,
    reparse_serialized_outputs,
    timing_statistics,
    validate_output_root,
)


def test_stage6_output_root_is_exactly_isolated(tmp_path: Path) -> None:
    assert validate_output_root(DEFAULT_OUTPUT) == DEFAULT_OUTPUT.resolve()
    with pytest.raises(ValueError, match="精确位于"):
        validate_output_root(tmp_path)


def test_timing_statistics_and_partial_measurement_are_truthful() -> None:
    statistics = timing_statistics([0.1, 0.2, 0.3, 0.4])
    partial = _measured_metric([0.1, 0.2], expected_count=3)

    assert statistics["count"] == 4
    assert statistics["mean"] == pytest.approx(0.25)
    assert statistics["median"] == pytest.approx(0.25)
    assert statistics["p95"] == pytest.approx(0.385)
    assert statistics["max"] == pytest.approx(0.4)
    assert statistics["total"] == pytest.approx(1.0)
    assert partial["status"] == "measured_partial"
    assert partial["statistics"]["count"] == 2
    with pytest.raises(ValueError, match="真实样本"):
        timing_statistics([])


def test_top_k_diagnostics_require_every_acceptance_field() -> None:
    complete = {
        "score": 0.8,
        "text_score": 0.8,
        "relation_score": 0.2,
        "path_cost": 2.0,
        "min_clearance_m": 1.0,
        "fallback_distance_m": 0.1,
        "dynamic_risk": 0.0,
        "rejection_reason": None,
        "selection_utility": 0.6,
        "selection_components": {"raw": {}, "normalized": {}, "weighted": {}},
        "semantic_rank": 1,
        "goal_candidates": [],
    }
    assert _validate_top_k_diagnostics({"evaluated_candidates": [complete]})[
        "passed"
    ]
    incomplete = dict(complete)
    incomplete.pop("dynamic_risk")
    result = _validate_top_k_diagnostics({"evaluated_candidates": [incomplete]})
    assert not result["passed"]
    assert result["missing"][0]["missing_fields"] == ["dynamic_risk"]


def test_legacy_and_structured_relation_serialization_share_query_semantics(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "legacy.pkl"
    with legacy_path.open("wb") as handle:
        pickle.dump([(1, 2, "left_of"), (2, 3, "near")], handle)
    structured_path = tmp_path / "structured.json"
    save_structured_edges(
        [
            SceneGraphEdge(
                source=1,
                target=2,
                relation="left_of",
                directed=True,
                confidence=0.9,
            ),
            SceneGraphEdge(
                source=2,
                target=3,
                relation="near",
                directed=False,
                confidence=0.8,
            ),
        ],
        structured_path,
    )

    legacy = load_scene_graph_edges(legacy_path)
    structured = load_scene_graph_edges(structured_path)

    assert all(edge.metadata["edge_format"] == "legacy" for edge in legacy)
    directed = next(edge for edge in structured if edge.directed)
    symmetric = next(edge for edge in structured if not edge.directed)
    assert edge_relation_for_query(directed, 1, 2) == "left_of"
    assert edge_relation_for_query(directed, 2, 1) == "right_of"
    assert edge_relation_for_query(symmetric, 2, 3) == "near"
    assert edge_relation_for_query(symmetric, 3, 2) == "near"


def test_memory_entry_faults_preserve_transaction_and_drop_late_results() -> None:
    report = _run_memory_failure_checks()

    assert report["duplicate_frame"]["passed"]
    assert report["out_of_order_frame"]["passed"]
    assert report["transaction_rollback"]["passed"]
    assert report["no_duplicate_planning"]["passed"]
    assert report["late_result"]["passed"]
    assert report["pending_tasks_after_close"] == 0


def test_qwen_timeout_exception_and_schema_faults_all_fallback() -> None:
    report = _run_qwen_failure_checks()

    assert all(item["passed"] for item in report.values())
    assert report["timeout"]["failure_code"] == "BACKEND_TIMEOUT"
    assert report["exception"]["failure_code"] == "BACKEND_EXCEPTION"
    assert report["invalid_schema"]["failure_code"] == "SCHEMA_VALIDATION_FAILED"
    assert report["illegal_candidate"]["failure_code"] == "ILLEGAL_CANDIDATE_ID"
    assert report["illegal_candidate"]["fallback_selected_node_id"] == 21


def test_compatibility_row_keeps_diagnostics_and_absolute_output(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    row = _matrix_row(
        "case",
        "组合",
        "json",
        "memory",
        "synthetic",
        False,
        None,
        None,
        "structured_edge",
        True,
        str(output),
        {"reason": "injected"},
    )

    assert Path(row["output_path"]).is_absolute()
    assert row["failure_diagnostic"] == {"reason": "injected"}
    assert set(row) >= {
        "input_format",
        "entry_point",
        "scene",
        "selected_node_id",
        "path_valid",
        "relation_source",
        "fallback",
        "diagnostics",
    }


def test_final_serialization_reparse_counts_jsonl_lines(tmp_path: Path) -> None:
    (tmp_path / "report.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    (tmp_path / "frames.jsonl").write_text(
        '{"frame": 1}\n{"frame": 2}\n', encoding="utf-8"
    )

    result = reparse_serialized_outputs(tmp_path)

    assert result["passed"]
    assert result["json_file_count"] == 1
    assert result["jsonl_file_count"] == 1
    assert result["jsonl_line_count"] == 2
