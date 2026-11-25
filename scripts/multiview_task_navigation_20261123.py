"""Replay six once-issued task requests over frozen front/three-view maps, CPU only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from conceptgraph.navigation.task_replay_20261123 import (  # noqa: E402
    EpisodeSession, ObjectRoadProvider, PastPoseMotion, ego_pose_from_camera,
)
from conceptgraph.navigation.vehicle_geometry import VehicleSpec  # noqa: E402
from conceptgraph.project_paths import experiment_directory  # noqa: E402

MANIFEST = PROJECT / "dataset/processed/multiview_20261123_pilot_v1/manifest.json"
PLAN = MANIFEST.with_name("raw_sensor_plan.json")
CALIB = PROJECT / "dataset/nuscenes_trainval_raw/v1.0-trainval/calibrated_sensor.json"
MAP_RUN = PROJECT / "outputs/results/nuscenes/validation/iterations/multiview_20261123_graph_pilot_v1"
LANES = PROJECT / "outputs/project_v2/lane_qwen_expansion_v1/lane_maps"
CONDITIONS = ("A_front", "C_three_dedup")
REQUESTS = (("pull_over", "靠边停车"), ("park_nearest", "停到最近空闲车位"), ("follow_vehicle", "跟车行驶"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_predictions(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path, values):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "xt") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, values):
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    run = experiment_directory("nuscenes", "validation", args.run_id)
    registration = read(run / "registration.json")
    started = time.monotonic()
    if shutil.disk_usage(PROJECT).free < registration["minimum_free_bytes"]:
        raise OSError("磁盘低于20GiB")
    for path, expected in registration["input_sha256"].items():
        if sha(path) != expected:
            raise ValueError("登记输入已改变")
    args.input_dir.mkdir(parents=True, exist_ok=False)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest = read(MANIFEST)
    assert [s["scene"] for s in manifest["scenes"]] == ["scene-0012", "scene-0013"]
    plan = {(r["scene"], r["frame_idx"], r["channel"]): r for r in read(PLAN)["plans"]}
    calibration = {r["token"]: r for r in read(CALIB)}
    episodes = []
    for scene in manifest["scenes"]:
        assert len(scene["groups"]) == 20
        for action, instruction in REQUESTS:
            episodes.append({"episode_id": scene["scene"] + "__" + action, "scene": scene["scene"],
                             "log_token": scene["log_token"], "instruction": instruction, "action": action,
                             "issue_frame_idx": 0, "issue_time_s": scene["groups"][0]["reference_timestamp_s"],
                             "frame_indices": list(range(20)), "human_review_status": "pending",
                             "instruction_source": "user_authorized_protocol_request_not_human_ground_truth",
                             "ground_truth_target_or_region": None, "conditions": list(CONDITIONS)})
    dump(args.input_dir / "episodes.json", {"independent_protocol_requests": 6, "human_annotated_requests": 0,
                                           "rows": episodes, "new_training_samples": 0})
    vehicle = VehicleSpec()
    vehicle_diagnostic = {"source": str(PROJECT / "conceptgraph/navigation/vehicle_geometry.py"),
                          "engineering_configuration": asdict(vehicle), "length_m": vehicle.length_m,
                          "width_m": vehicle.width_m, "reference_origin": "rear_axle_center",
                          "nuScenes_actual_ego_dimensions": False, "ego_reference_alignment_verified": False,
                          "inserted_as_measured_task_graph_dimensions": False}
    dump(args.report_dir / "engineering_vehicle_configuration.json", vehicle_diagnostic)
    source_paths = [Path(__file__), PROJECT / "conceptgraph/navigation/task_replay_20261123.py",
                    PROJECT / "conceptgraph/navigation/semantic_tasks_20261123.py",
                    PROJECT / "conceptgraph/navigation/vehicle_geometry.py", args.input_dir / "episodes.json"]
    source_sha = {str(p): sha(p) for p in source_paths}
    dump(run / "frozen_configuration.json", {"registration_sha256": sha(run / "registration.json"),
                                           "additional_source_and_input_sha256": source_sha,
                                           "episode_requests": 6, "conditions": list(CONDITIONS),
                                           "expected_temporal_records": 240,
                                           "pose_estimator": "CAM_FRONT calibration inversion and one past pose difference; no extrapolation",
                                           "engineering_vehicle_configuration": vehicle_diagnostic,
                                           "new_model_calls": 0, "task_GT_read": False,
                                           "selection_or_control_success_metrics": "forbidden_without_human_task_GT_and_execution"})
    evidence = defaultdict(list)
    input_frames, frame_stats, gap_rows = [], [], []
    runtimes = []
    for scene in manifest["scenes"]:
        name = scene["scene"]
        prediction_path = MAP_RUN / name / "multiview_20261123_predictions.jsonl.gz"
        predictions = read_predictions(prediction_path)
        road_path = LANES / name / "lane_topology.json"
        provider = ObjectRoadProvider(read(road_path) if road_path.is_file() else None, str(road_path))
        motion = PastPoseMotion()
        sessions = {(e["episode_id"], condition): EpisodeSession(e, condition)
                    for e in episodes if e["scene"] == name for condition in CONDITIONS}
        assert len(predictions) == 20
        for group, frozen in zip(scene["groups"], predictions):
            begin = time.perf_counter()
            frame_idx = group["frame_idx"]
            now = group["reference_timestamp_s"]
            assert frozen["frame_idx"] == frame_idx and frozen["reference_timestamp_s"] == now
            front = group["views"]["CAM_FRONT"]
            record = plan[(name, frame_idx, "CAM_FRONT")]["camera"]
            assert record["token"] == front["camera_token"] and record["timestamp"] / 1e6 == front["camera_timestamp_s"]
            pose = ego_pose_from_camera(front["camera_pose"], calibration[record["calibrated_sensor_token"]])
            pose_source = f"{MANIFEST}#{name}/f{frame_idx}/CAM_FRONT + {CALIB}#{record['calibrated_sensor_token']}"
            ego, motion_audit = motion.update(front["camera_timestamp_s"], pose, now, pose_source)
            for condition in CONDITIONS:
                # Provider fields never contain official object labels/identities.
                graph, coverage = provider.build(frozen["predictions"][condition], ego, now,
                                                 f"{prediction_path}:line{frame_idx + 1}/{condition}", condition)
                graph_index = len(input_frames)
                input_frames.append({"scene": name, "frame_idx": frame_idx, "condition": condition,
                                     "time_s": now, "ego": asdict(graph.ego),
                                     "objects": [asdict(n) for n in graph.entities if n.kind != "lane"],
                                     "road_provider": {"path": str(road_path), "sha256": sha(road_path) if road_path.is_file() else None,
                                                       "kind": "known_static_road_prior_not_future_observation"},
                                     "motion_audit": motion_audit, "coverage": coverage,
                                     "GT_in_inference": False, "human_ground_truth": None})
                for (episode_id, method), session in sessions.items():
                    if method != condition:
                        continue
                    row = session.step(frame_idx, graph)
                    row.update(input_frame_row=graph_index + 1, input_frame_path=str(args.input_dir / "task_frames.jsonl.gz"),
                               source_prediction_path=str(prediction_path), source_prediction_line=frame_idx + 1,
                               pose_timestamp_s=front["camera_timestamp_s"], motion_audit=motion_audit,
                               coverage={k: v for k, v in coverage.items() if k != "lane_assignment_candidates"})
                    evidence[episode_id].append(row)
                    for gap in row["provider_gaps"]:
                        gap_rows.append({"episode_id": episode_id, "scene": name, "condition": condition,
                                         "frame_idx": frame_idx, **{k: gap[k] for k in ("field", "status", "needs")}})
                relation_counts = Counter(r["relation"] for r in coverage["relations"])
                frame_stats.append({"scene": name, "condition": condition, "frame_idx": frame_idx,
                                    "object_nodes": coverage["object_nodes"], "vehicle_nodes": coverage["vehicle_nodes"],
                                    "lane_nodes": coverage["lane_nodes"], "ego_unique_lane": coverage["ego_unique_lane"],
                                    "objects_unique_lane": coverage["objects_unique_lane"],
                                    "belongs_to_lane": relation_counts["belongs_to_lane"],
                                    "same_lane_ahead": relation_counts["same_lane_ahead"],
                                    "past_pose_motion_available": motion_audit["motion"] is not None})
            runtimes.append({"scene": name, "frame_idx": frame_idx, "adapter_and_six_session_steps_wall_s": time.perf_counter() - begin})
            if time.monotonic() - started > 1800 or shutil.disk_usage(PROJECT).free < 20 * 1024**3:
                raise RuntimeError("回放达到预算/磁盘停止线")
    write_jsonl(args.input_dir / "task_frames.jsonl.gz", input_frames)
    for episode_id, rows in evidence.items():
        assert len(rows) == 40 and len({(r["condition"], r["frame_idx"]) for r in rows}) == 40
        write_jsonl(run / (episode_id + ".jsonl.gz"), rows)
    write_csv(args.report_dir / "input_and_relation_coverage.csv", frame_stats)
    write_csv(args.report_dir / "per_step_provider_gaps.csv", gap_rows)
    gap_counts = Counter((r["condition"], r["field"], r["status"]) for r in gap_rows)
    write_csv(args.report_dir / "provider_gap_summary.csv", [{"condition": a, "field": b, "status": c, "step_count": n}
                                                           for (a, b, c), n in sorted(gap_counts.items())])
    dump(run / "runtime.json", {"groups": runtimes, "replay_wall_s": time.monotonic() - started,
                               "model_calls": 0, "planning_search_calls": 0, "vehicle_controls": 0})
    summary = {"independent_protocol_episodes": len(evidence), "scenes": 2, "logs": 1,
               "groups": 40, "conditions": list(CONDITIONS), "adapted_graph_frames": len(input_frames),
               "temporal_evidence_records": sum(map(len, evidence.values())), "human_annotated_episodes": 0,
               "instruction_issue_count_each_episode_condition": 1,
               "condition_input_summary": {c: {
                   "frames": sum(r["condition"] == c for r in frame_stats),
                   "frames_with_objects": sum(r["condition"] == c and r["object_nodes"] > 0 for r in frame_stats),
                   "object_node_observations": sum(r["object_nodes"] for r in frame_stats if r["condition"] == c),
                   "frames_with_road": sum(r["condition"] == c and r["lane_nodes"] > 0 for r in frame_stats),
                   "frames_with_past_pose_motion": sum(r["condition"] == c and r["past_pose_motion_available"] for r in frame_stats),
                   "same_lane_ahead_relations": sum(r["same_lane_ahead"] for r in frame_stats if r["condition"] == c)} for c in CONDITIONS},
               "decision_status_counts": dict(Counter(r["task_decision"]["status"] for rr in evidence.values() for r in rr)),
               "monitor_status_counts": dict(Counter(r["monitor"]["status"] for rr in evidence.values() for r in rr)),
               "episode_end": "replay_window_ended_not_task_completion",
               "correct_selection_rate": None, "parking_success_rate": None, "follow_success_rate": None,
               "task_GT_available": False, "actual_arrival": None, "execution_authorized": False,
               "new_training_examples": 0, "new_model_calls": 0, "GPU_calls": 0,
               "replay_wall_s": time.monotonic() - started}
    assert summary["independent_protocol_episodes"] == 6 and summary["temporal_evidence_records"] == 240
    dump(run / "summary.json", summary)
    dump(args.report_dir / "summary.json", summary)
    for path, expected in {**registration["input_sha256"], **source_sha}.items():
        if sha(path) != expected:
            raise ValueError("冻结源码或输入运行中改变")
    for path in source_paths[:-1]:
        destination = run / "source_snapshot" / path.relative_to(PROJECT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
