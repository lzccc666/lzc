"""Aggregate measured replay and inference events; missing evidence stays null."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def distribution(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
        "max": max(values) if values else None,
        "min": min(values) if values else None,
    }


def replay_metrics(frames: Path):
    rows = [
        json.loads(line) for line in frames.read_text().splitlines() if line.strip()
    ]
    planned = [row for row in rows if row["replanned"]]
    successful = [
        row
        for row in planned
        if row.get("experiment_audit", {}).get("path_grid") and row["failure"] is None
    ]
    validations = [row["experiment_audit"]["lane_validation"] for row in successful]
    attempts = [
        row.get("experiment_audit", {}).get("planning_result") for row in planned
    ]
    attempts = [attempt for attempt in attempts if attempt is not None]
    goals = [
        goal
        for attempt in attempts
        if attempt.get("selected_candidate")
        for goal in attempt["selected_candidate"].get("goal_candidates", [])
        if goal["path_found"]
        and goal.get("goal_rc") == attempt["selected_candidate"].get("target_grid")
    ]
    projections = [
        goal.get("debug", {}).get(
            "goal_projection_distance_m", goal.get("fallback_distance_m")
        )
        for goal in goals
    ]
    failures = [row["failure"]["code"] for row in planned if row["failure"]]
    return {
        "frame_count": len(rows),
        "planning_attempts": len(planned),
        "successful_plans": len(successful),
        "path_success_rate": ratio(len(successful), len(planned)),
        "reachable_rate": ratio(len(successful), len(planned)),
        "target_confirmation_success_rate": float(
            any("LOCKED" in row.get("state_trace", []) for row in rows)
        ),
        "target_locked_frame_ratio": ratio(
            sum(row["selected_node_id"] is not None for row in rows), len(rows)
        ),
        "retrieval_top1": None,
        "retrieval_mrr": None,
        "retrieval_ground_truth_status": "录制回放没有独立逐帧目标真值，不能用规则排名作为模型正确率",
        "lane_adherence_rate": ratio(
            sum(v["valid"] for v in validations), len(validations)
        ),
        "out_of_lane_point_ratio": ratio(
            sum(len(v["invalid_cells"]) for v in validations),
            sum(v["sample_count"] for v in validations),
        ),
        "out_of_lane_length_m": sum(v["out_of_lane_length_m"] for v in validations),
        "direction_violation_sample_count": sum(
            v["direction_violation_count"] for v in validations
        ),
        "reverse_sample_count": sum(v["reverse_count"] for v in validations),
        "reverse_event_count": sum(v["reverse_event_count"] for v in validations),
        "illegal_transition_count": sum(
            len(v["illegal_transitions"]) for v in validations
        ),
        "collision_rate": ratio(
            sum(
                bool(row["experiment_audit"]["remaining_path_collision_cells"])
                for row in successful
            ),
            len(successful),
        ),
        "minimum_safety_distance_m": distribution(
            [row["experiment_audit"]["minimum_clearance_m"] for row in successful]
        ),
        "path_length_m": distribution(
            [row["experiment_audit"]["path_length_m"] for row in successful]
        ),
        "total_absolute_turn_rad": distribution(
            [row["experiment_audit"]["total_absolute_turn_rad"] for row in successful]
        ),
        "target_projection_distance_m": distribution(projections),
        "planning_time_s": distribution(
            [row["timings"].get("planning_s") for row in planned]
        ),
        "total_frame_time_s": distribution([row["frame_runtime_s"] for row in rows]),
        "failure_codes": {code: failures.count(code) for code in sorted(set(failures))},
        "source_artifact": str(frames.resolve()),
        "unit": "one plan per actual planning attempt; lane samples partition full segments",
    }


def qwen_audit_metrics(audit_root: Path, backend_call_count: int):
    envelopes = []
    for path in sorted(audit_root.rglob("*.json")):
        item = json.loads(path.read_text())
        envelopes.append(item.get("result", item))
    hits = sum(item.get("cache", {}).get("status") == "hit" for item in envelopes)
    fallbacks = [
        item.get("fallback_reason") or item.get("failure_code")
        for item in envelopes
        if item.get("fallback_used")
    ]
    return {
        "backend_call_count": backend_call_count,
        "successful_backend_response_count": sum(
            bool(item.get("used_qwen"))
            and bool(item.get("output_schema_valid"))
            and not item.get("cache", {}).get("hit", False)
            for item in envelopes
        ),
        "cache_hit_count": hits,
        "audited_call_count": len(envelopes),
        "cache_hit_rate": ratio(hits, len(envelopes)),
        "schema_valid_rate": ratio(
            sum(bool(item.get("output_schema_valid")) for item in envelopes),
            len(envelopes),
        ),
        "fallback_count": len(fallbacks),
        "fallback_rate": ratio(len(fallbacks), len(envelopes)),
        "fallback_reasons": {
            str(key): fallbacks.count(key) for key in sorted(set(fallbacks), key=str)
        },
        "qwen_time_s": distribution(
            [item.get("latency", {}).get("backend_s") for item in envelopes]
        ),
        "evidence": "delegate backend counter and per-call audit envelopes",
    }
