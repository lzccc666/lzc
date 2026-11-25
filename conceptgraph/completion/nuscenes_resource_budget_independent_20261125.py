"""Recount the fixed resource pilot without dropping failed GPU occupancy."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/results/nuscenes/validation/iterations/nuscenes_full_20261125_resource_v1"
DATA = ROOT / "dataset/processed/nuscenes_full_20261125_v1"
REPORT = ROOT / "outputs/reports/local_scope_audit_20261125"


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    summary_path = ROOT / "outputs/reports/nuscenes_full_20261125/GPU_resource_pilot_summary.json"
    summary = load(summary_path)
    assert summary["complete"] and summary["scenes"] == 8 and not summary["failed_scenes"]
    counts = Counter()
    stage = Counter()
    closed = []
    for number in range(1, 9):
        folder = RUN / f"scene-{number:04d}"
        record = load(folder / "complete.json")
        auxiliary = load(folder / "sharedscan_complete.json")
        assert record["complete"] and auxiliary["complete"]
        assert record["counts"]["frames"] == auxiliary["counts"]["frames"]
        counts.update(record["counts"])
        stage["source_IO_recorded_on_accepted_scene_s"] += record["source_seek_extract_s"]
        stage["frontend_wall_recorded_on_accepted_scene_s"] += record["GPU_frontend_wall_s"]
        stage["main_CPU_scene_wall_sum_s"] += record["CPU_scene_wall_s"]
        stage["aux_CPU_scene_wall_sum_s"] += auxiliary["CPU_wall_s"]
        closed.append(dict(scene=record["scene"], frames=record["counts"]["frames"],
            main_complete_SHA256=sha(folder / "complete.json"), auxiliary_complete_SHA256=sha(folder / "sharedscan_complete.json")))
    assert counts == Counter(summary["counts"])
    assert counts["frames"] == 318 and counts["actual_GPU_image_calls"] == 1908
    assert counts["resource_regression_extra_GPU_calls"] == 48
    assert counts["reused_failed_attempt_GPU_image_calls"] == 240
    assert counts["actual_GPU_image_calls"] == 6 * counts["frames"]
    attempts = []
    occupied = 0.0
    for acquired_path in sorted((RUN / "runtime_attempts").glob("*/GPU_acquired.json")):
        released_path = acquired_path.with_name("GPU_release.json")
        acquired, released = load(acquired_path), load(released_path)
        assert acquired["PID"] == released["PID"]
        assert abs(released["prior_GPU_occupied_s"] - occupied) < 1e-6
        utc_duration = (datetime.fromisoformat(released["released_utc"]) - datetime.fromisoformat(acquired["acquired_utc"])).total_seconds()
        assert abs(utc_duration - released["GPU_occupied_wall_s"]) < 0.01
        occupied += released["GPU_occupied_wall_s"]
        attempts.append(dict(attempt=acquired_path.parent.name, **released,
            acquired_SHA256=sha(acquired_path), released_SHA256=sha(released_path)))
    assert len(attempts) == 5
    assert abs(occupied - summary["total_GPU_occupied_s"]) < 0.01
    projected = occupied / counts["actual_GPU_image_calls"] * 204894 / 3600
    assert projected > 12 and projected < 32
    assert abs(projected - summary["projected_full_GPU_occupied_hours"]) < 1e-4
    # Scene0001's real inference occurred before a receipt writer failed.
    # Its accepted CPU-reuse receipt has timing placeholders, not measured zeros.
    first = load(RUN / "scene-0001/complete.json")
    assert first["CPU_preflight_reused"] and first["model_s"] == 0 and first["GPU_frontend_wall_s"] == 0
    old_io_path = ROOT / "outputs/reports/local_scope_audit_20261125/nuscenes_source_IO_comparison_independent_v1.json"
    overall = load(DATA / "overall_execution_clock_v1.json")
    assert overall["overall_start_UTC"] == "2026-11-25T11:02:00+00:00"
    result = dict(passed_measurement_recount=True, audit_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), summary_sha256=sha(summary_path),
        closed_scenes=closed, counts=dict(counts), method_frames=318 * 7,
        actual_primary_image_calls=1908, additional_source_regression_model_calls=48,
        total_physical_model_calls=1956, first_primary_240_already_included_not_repeated=True,
        all_attempts=attempts, total_actual_GPU_occupied_s=occupied,
        summary_pre_finally_vs_release_difference_s=occupied - summary["total_GPU_occupied_s"],
        projected_all850_actual_occupied_hours=projected,
        projection="linear conservative extrapolation of all cumulative pilot occupancy, including failures and IO/CPU wait; fixed retry costs are thus extrapolated too",
        old_12h_actual_occupancy_gate_passed=False,
        authorized_new_32h_numerical_projection_gate_passed=True,
        authorized_overall_wall_hours=48, authorized_nu_persistent_GiB=5.25,
        effective_runtime_budget_overlay_verified=False,
        runtime_gate_note="separate audit must verify the new effective32h/48h/5.25GiB runtime overlay before full-run approval",
        overall_start_not_reset=overall["overall_start_UTC"],
        stage_wall_sums=dict(stage), stage_wall_sums_not_additive_with_GPU_occupancy=True,
        first240_model_and_frontend_timing="unavailable, placeholders zero are not measurements; actual run0001 occupancy retained",
        first240_source_IO_evidence=str(old_io_path), independent_no_contention_steady_state_measurement=False,
        peak_aggregate_temp_bytes=summary["peak_total_temp_bytes"], temporary_512MiB_passed=summary["peak_total_temp_bytes"] <= 512 * 2**20,
        quality_metrics_not_evaluated=True, new_GPU_calls=0, elapsed_s=time.monotonic() - started)
    target = REPORT / "nuscenes_resource8_budget_independent_v1.json"
    with target.open("x") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ("passed_measurement_recount", "counts", "method_frames", "total_actual_GPU_occupied_s", "projected_all850_actual_occupied_hours", "old_12h_actual_occupancy_gate_passed", "authorized_new_32h_numerical_projection_gate_passed", "effective_runtime_budget_overlay_verified")}))


if __name__ == "__main__":
    main()
