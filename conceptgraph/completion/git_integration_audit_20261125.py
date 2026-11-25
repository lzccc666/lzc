"""Bounded independent checks for the 2026-11-25 integration outputs."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
from itertools import combinations
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "outputs/reports/git_integration_20261125"


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_pixel_check(raw, view):
    """Reproject the declared real depth cluster without calling the new mapper."""
    with gzip.open(view["detections"], "rb") as stream:
        detections = pickle.load(stream)
    mask = np.asarray(detections["mask"][raw["original_detection_index"]], bool)
    depth = np.asarray(Image.open(view["depth"]), float) / 1000.0
    if mask.shape != depth.shape:
        raise AssertionError("audit requires original mask at declared depth resolution")
    quantiles = raw["depth_evidence"]["depth_quantiles_m"]
    valid = mask & np.isfinite(depth) & (depth > 1) & (depth < 65)
    selected = valid & (depth >= quantiles[0] - 1e-9) & (depth <= quantiles[-1] + 1e-9)
    y, x = np.nonzero(selected)
    z = depth[y, x]
    intrinsic = np.asarray(view["intrinsic"])
    pose = np.asarray(view["camera_pose"])
    local = np.column_stack(((x-intrinsic[0, 2])*z/intrinsic[0, 0],
                             (y-intrinsic[1, 2])*z/intrinsic[1, 1], z))
    world = local @ pose[:3, :3].T + pose[:3, 3]
    assert len(world) == raw["current_points"] == raw["depth_evidence"]["retained_pixels"]
    error = float(np.max(np.abs(np.median(world, axis=0)-raw["raw_center_world_m"])))
    assert error < 1e-8, (raw["key"], error)
    return dict(key=raw["key"], real_pixels=len(world), maximum_center_error_m=error,
                depth_sha256=sha(view["depth"]), detections_sha256=sha(view["detections"]))


def audit_nodes(run_id, scope):
    data = ROOT / "dataset/processed" / run_id
    manifest = json.loads((data / "manifest.json").read_text())
    run = ROOT / "outputs/results/nuscenes/validation/iterations" / run_id
    output = run / f"{scope}_frames.jsonl.gz"
    observed = list(rows(output))
    expected = {(r["scene"], r["frame_idx"]): r["predictions"] for r in rows(
        ROOT / "dataset/processed/relation_nav_unified_20261125_v1/method_frames.jsonl")
        if r["cohort"] == scope}
    assert len(observed) == manifest[f"{scope}_frames"] == len(expected)
    assert len({(r["scene"], r["frame_idx"]) for r in observed}) == len(observed)
    raw_index = {raw["key"]: raw for record in observed for raw in record["raw_observations"]}
    counts = {"frames": len(observed), "nodes": 0, "trace_links": 0,
              "known_motion": 0, "unknown_motion": 0, "static": 0, "dynamic": 0}
    samples = {}
    for frame in observed:
        assert frame["predictions"] == expected[frame["scene"], frame["frame_idx"]]
        assert frame["GT_in_method"] is False and frame["vehicle_safety_authorized"] is False
        assert len(frame["predictions"]) == len(frame["nodes"])
        states = {s["state_uid"]: s for s in frame["evidence_states"]}
        for raw in frame["raw_observations"]:
            sample_key = (raw["scene"], raw["source_key"][0])
            if raw["current_points"] and sample_key not in samples:
                samples[sample_key] = raw
        for index, (prediction, node) in enumerate(zip(frame["predictions"], frame["nodes"])):
            counts["nodes"] += 1
            assert node["original_index"] == index and node["node_id"] == prediction["node_id"]
            assert node["center_world_m"] == prediction["center"]
            assert node["body_heading_world_rad"] is None and node["body_heading_source"] == "unknown"
            assert node["vehicle_safety_authorized"] is False
            assert node["motion_history_cross_id_merged"] is False
            motion = node["motion_state"]
            assert motion in {"static", "dynamic", "unknown"}
            assert node["semantic_position_candidate"] == (motion == "static")
            state = states.get(node["measurement"]["state_uid"])
            trace = state["raw_trace"] if state else []
            assert node["measurement"]["observation_keys"] == [r["key"] for r in trace]
            for raw in trace:
                origin = raw_index[raw["key"]]
                for key in ["scene", "frame_idx", "timestamp_s", "raw_center_world_m", "current_points", "original_detection_index"]:
                    assert raw[key] == origin[key], (raw["key"], key)
                counts["trace_links"] += 1
            if motion == "unknown":
                counts["unknown_motion"] += 1
                assert node["motion_speed_xy_mps"] is None
                continue
            counts["known_motion"] += 1
            counts[motion] += 1
            times = [r["timestamp_s"] for r in trace]
            assert len(trace) >= 3 and len({(r["scene"], r["frame_idx"]) for r in trace}) == len(trace)
            assert all(a < b for a, b in zip(times, times[1:]))
            assert all(t <= frame["render_timestamp_s"] for t in times)
            assert times[-1]-times[0] >= 1-1e-8
            assert all(r["source_closed"] for r in trace)
            assert abs(trace[-1]["render_timestamp_s"]-frame["render_timestamp_s"]) < 1e-6
            assert trace[-1]["current_points"] >= 1
            assert trace[-1]["depth_evidence"].get("support_fraction", 0) >= .4
            variance = np.asarray(node["uncertainty"]["diag_m2"])
            assert np.isfinite(variance).all() and np.all(variance > 0) and max(variance[:2]) <= 4
            centers = np.asarray([r["raw_center_world_m"] for r in trace])
            velocities = [(centers[j, :2]-centers[i, :2])/(times[j]-times[i])
                          for i, j in combinations(range(len(times)), 2) if times[j]-times[i] >= .75]
            vector = np.median(velocities, axis=0)
            speed = float(np.linalg.norm(vector))
            assert abs(speed-node["motion_speed_xy_mps"]) < 1e-9
            assert motion == ("static" if speed <= .5 else "dynamic")
    if scope == "S2":
        views = {(s["scene"], g["frame_idx"], c): v for s in manifest["S2_manifest"]["scenes"]
                 for g in s["groups"] for c, v in g["views"].items()}
        chosen = list(samples.values())
    else:
        views = {(r["scene"], r["frame_idx"], "CAM_FRONT"): r for r in rows(manifest["R1_method_frames"])}
        chosen = list(samples.values())[:6]
    pixels = [raw_pixel_check(raw, views[raw["scene"], raw["frame_idx"], raw["source_key"][0]]) for raw in chosen]
    result = dict(passed=True, scope=scope, node_run=run_id, counts=counts,
                  output_sha256=sha(output), manifest_sha256=sha(data/"manifest.json"),
                  pixel_checks=pixels, independent_raw_pixel_checks=len(pixels),
                  pixel_sampling="first real observation per scene/camera; R1 first6 scenes; no outcome selection",
                  original_predictions_exact=True, causal_motion_known_only=True,
                  trace_identity_not_concatenated=True, GT_in_method=False,
                  geometry_pose_source="declared CAM_FRONT camera proxy, not measured rear axle pose",
                  claims_vehicle_path_or_closed_loop=False)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / f"node_independent_{scope}_v1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "pixel_checks"}, ensure_ascii=False))


def audit_relations():
    run = ROOT / "outputs/results/nuscenes/validation/iterations/relation_nav_unified_20261125_v1"
    source = ROOT / "dataset/processed/relation_nav_unified_20261125_v1/method_frames.jsonl"
    incoming = {(r["cohort"], r["scene"], r["frame_idx"]): r for r in rows(source)}
    counts = {"R1": 0, "S2": 0, "edges": 0, "legacy_fact_equal": 0}
    registries = {}
    seen = set()
    for record in rows(run / "relation_frames.jsonl.gz"):
        key = (record["cohort"], record["scene"], record["frame_idx"])
        assert key not in seen
        seen.add(key)
        frame = incoming[key]
        registry = registries.setdefault(key[:2], {})
        positions = {}
        for obj in frame["predictions"]:
            registry.setdefault(obj["node_id"], len(registry))
            positions[registry[obj["node_id"]]] = np.asarray(obj["center"][:2])
        forward = np.asarray(frame["heading"])
        forward = forward / np.linalg.norm(forward)
        left = np.array([-forward[1], forward[0]])
        canonical = lambda edges: sorted(json.dumps(e, sort_keys=True) for e in edges)
        assert canonical(record["full_edges"]) == canonical(record["incremental_edges"])
        assert record["all_current_fields_equal"] is True and record["GT_in_method"] is False
        assert record["nodes"] == len(positions)
        for edge in record["incremental_edges"]:
            delta = positions[edge["source"]] - positions[edge["target"]]
            distance = float(np.linalg.norm(delta))
            relation = edge["relation"]
            assert abs(distance-edge["distance_m"]) < 1e-8
            assert 0 < distance <= (6 if relation == "near" else 12)
            if relation in {"left_of", "right_of", "in_front_of", "behind"}:
                component = float(delta @ (left if relation in {"left_of", "right_of"} else forward))
                assert (component if relation in {"left_of", "in_front_of"} else -component) > 1
            else:
                assert relation == "near"
            metadata = edge["metadata"]
            assert metadata["current_fact_only"] is True
            assert metadata["historical_memory_used_as_current_fact"] is False
            assert metadata["computed_frame"] == key[2]
            assert metadata["timestamp_s"] == frame["timestamp_s"]
            assert metadata["reference_frame"] == "ego_heading"
            assert metadata["object_relative_supported"] is False
            assert np.allclose(metadata["forward_axis_world"], forward, atol=1e-12)
            assert np.allclose(metadata["left_axis_world"], left, atol=1e-12)
            counts["edges"] += 1
        counts[key[0]] += 1
        counts["legacy_fact_equal"] += bool(record["legacy_fact_equal"])
    assert seen == set(incoming)
    assert counts["R1"] == 1508 and counts["S2"] == 120
    result = dict(passed=True, counts=counts, all_current_payloads_equal=True,
                  physical_relation_signs_and_thresholds_recomputed=True,
                  current_frame_provenance_checked=True, near_distance_m=6,
                  directional_distance_m=12, component_dead_zone_m=1,
                  legacy_near8_results_unchanged=True,
                  geometry_pose_source="CAM_FRONT optical axis proxy, not object body heading",
                  input_sha256=sha(source), output_sha256=sha(run / "relation_frames.jsonl.gz"))
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "relation_independent_v1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False))


def audit_directions(run_id, scope):
    run = ROOT / "outputs/results/nuscenes/validation/iterations" / run_id
    base = run / f"{scope}_frames.jsonl.gz"
    extended = run / f"{scope}_task_nodes.jsonl.gz"
    fingerprint = sha(base)
    count = available = frames = 0
    old_rows = rows(base)
    for record in rows(extended):
        original = next(old_rows)
        frames += 1
        assert record["evidence_source_sha256"] == fingerprint
        for key in ["scene", "log", "frame_idx", "predictions", "task_predictions", "frame_source"]:
            assert record[key] == original[key]
        assert len(record["nodes"]) == len(original["nodes"])
        for node, old in zip(record["nodes"], original["nodes"]):
            assert {k: node[k] for k in old} == old
            count += 1
            valid = old["motion_state"] != "unknown"
            vector = node["raw_velocity_world_xy_mps"]
            available += vector is not None
            if vector is not None:
                speed = old["measurement"]["measured_raw_speed_mps"]
                if speed is not None:
                    assert abs(np.linalg.norm(vector)-speed) < 1e-8
            assert node["raw_velocity_valid_for_semantics"] == bool(valid and vector is not None)
            assert node["motion_velocity_world_xy_mps"] == (vector if valid else None)
            assert node["body_heading_available"] is False and node["body_heading_world_rad"] is None
            assert node["motion_heading_available"] == bool(old["motion_state"] == "dynamic" and old["motion_heading_world_rad"] is not None)
    assert next(old_rows, None) is None
    assert frames == (1508 if scope == "R1" else 120)
    result = dict(passed=True, scope=scope, frames=frames, nodes=count,
                  raw_velocity_available=available, prior_nodes_byte_content_preserved=True,
                  unknown_velocity_not_semantic=True, body_orientation_not_inferred=True,
                  original_sha256=sha(base), extended_sha256=sha(extended))
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / f"direction_independent_{scope}_v1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False))


def audit_motion_evaluation(run_id):
    from scipy.optimize import linear_sum_assignment

    run = ROOT / "outputs/results/nuscenes/validation/iterations" / run_id
    truth = {(r["cohort"], r["scene"], r["frame_idx"]): r["truth"] for r in rows(
        ROOT / "dataset/processed/relation_nav_unified_20261125_v1/evaluation_only/frames.jsonl")}
    evaluations = defaultdict(dict)
    for row in rows(run / "motion_evaluation_rows.jsonl.gz"):
        key = row["scope"], row["scene"], row["frame_idx"]
        assert row["instance_token"] not in evaluations[key]
        evaluations[key][row["instance_token"]] = row
    summary = json.loads((run/"motion_evaluation_summary.json").read_text())
    result = {}
    for scope in ["R1", "S2"]:
        counts = Counter()
        confusion = defaultdict(Counter)
        for frame in rows(run/f"{scope}_frames.jsonl.gz"):
            key = scope, frame["scene"], frame["frame_idx"]
            official = truth[key]
            counts["GT"] += len(official)
            assert len(evaluations[key]) == len(official)
            predictions = frame["predictions"]
            m, n = len(predictions), len(official)
            cost = np.full((m, n+m), 1e9)
            cost[:, n:] = 0
            for i, pred in enumerate(predictions):
                for j, obj in enumerate(official):
                    distance = np.linalg.norm(np.asarray(pred["center"][:2])-obj["center"][:2])
                    if pred["category"] == obj["category"] and np.isfinite(distance) and distance <= 3:
                        cost[i, j] = distance/4-(m+1)
            rr, cc = linear_sum_assignment(cost)
            matched = {j: i for i, j in zip(rr, cc) if j < n and cost[i, j] < 0}
            counts["matched_3m"] += len(matched)
            for j, obj in enumerate(official):
                row = evaluations[key][obj["instance_token"]]
                node = frame["nodes"][matched[j]] if j in matched else None
                assert row["matched_node_id"] == (node["node_id"] if node else None)
                state = node["motion_state"] if node else "missing_node"
                legacy = node["legacy_motion_state"] if node else "missing_node"
                assert row["predicted_motion"] == state and row["legacy_predicted_motion"] == legacy
                attributes = set(row["official_attributes"])
                labels = set()
                if attributes & {"vehicle.moving", "pedestrian.moving"}:
                    labels.add("dynamic")
                if attributes & {"vehicle.stopped", "vehicle.parked", "pedestrian.standing", "pedestrian.sitting_lying_down"}:
                    labels.add("static")
                attribute = next(iter(labels)) if len(labels) == 1 else "unknown"
                speed = row["past_displacement_speed_mps"]
                displacement = "unknown" if speed is None else "static" if speed <= .5 else "dynamic"
                assert row["attribute_motion"] == attribute and row["past_displacement_motion"] == displacement
                if speed is not None:
                    assert speed >= 0 and row["past_interval_s"] > 0
                for kind, label in [("attribute", attribute), ("past_displacement", displacement)]:
                    confusion[kind+":"+label][state] += 1
                    confusion["legacy_"+kind+":"+label][legacy] += 1
                    if label != "unknown":
                        counts[kind+":all_labeled"] += 1
                        counts[kind+":known_prediction"] += state in {"static", "dynamic"}
                        counts[kind+":correct_over_all_labeled"] += state == label
                        counts[kind+":unknown_or_missing"] += state not in {"static", "dynamic"}
        assert counts["GT"] == (5723 if scope == "R1" else 687)
        for name, value in counts.items():
            assert value == summary["scopes"][scope]["counts"][name], (scope, name, value)
        assert {k: dict(v) for k, v in confusion.items()} == summary["scopes"][scope]["confusion"]
        result[scope] = dict(counts)
    report = dict(passed=True, scopes=result, same_class_planar_3m_independent_assignment=True,
                  all_GT_denominators_retained=True, all_attribute_and_displacement_confusions_recounted=True,
                  raw_official_label_provenance_scope="frozen sources and evaluator dataflow read; row attribute mapping independently checked",
                  official_attributes_not_exact_instantaneous_velocity=True,
                  summary_sha256=sha(run/"motion_evaluation_summary.json"))
    (REPORT/"motion_evaluation_independent_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False))


def audit_navigation(node_run):
    from conceptgraph.completion import relation_nav_runtime_20261125 as runtime
    from conceptgraph.navigation.relation_nav_unified_20261125 import select_task
    from conceptgraph.scenegraph.relation_nav_unified_20261125 import full_current
    from conceptgraph.scenegraph.relations import RelationNode
    from scipy.optimize import linear_sum_assignment

    data = ROOT / "dataset/processed/relation_nav_unified_20261125_v1"
    run = ROOT / "outputs/results/nuscenes/validation/iterations/relation_nav_unified_20261125_v1"
    nodepath = ROOT / "outputs/results/nuscenes/validation/iterations" / node_run / "S2_frames.jsonl.gz"
    frames = {(r["scene"], r["frame_idx"]): r for r in rows(data/"method_frames.jsonl") if r["cohort"] == "S2"}
    unified = {(r["scene"], r["frame_idx"]): r for r in rows(nodepath)}
    labels = {r["task_id"]: r for r in rows(data/"evaluation_only/task_labels.jsonl")}
    truth = {(r["scene"], r["frame_idx"]): r["truth"] for r in rows(data/"evaluation_only/frames.jsonl") if r["cohort"] == "S2"}
    queries = {r["task_id"]: r for r in rows(data/"method_tasks.jsonl")}
    actual = {r["task_id"]: r for r in rows(run/"navigation_tasks.jsonl")}
    assert set(queries) == set(labels) == set(actual) and len(actual) == 1200
    assert set(frames) == set(unified) and len(frames) == 120
    registries = {}
    contexts = {}
    graph_hashes = []
    for key, frame in frames.items():
        nodeframe = unified[key]
        assert nodeframe["predictions"] == frame["predictions"]
        records = []
        for original in nodeframe["nodes"]:
            record = dict(original)
            record["legacy_prediction"] = frame["predictions"][record["original_index"]]
            assert record["node_id"] == record["legacy_prediction"]["node_id"]
            records.append(record)
        registry = registries.setdefault(key[0], {})
        nodes, eligible, indices = runtime.adapt_nodes(records, registry)
        relation_input = {i: RelationNode(i, n.bbox_center, dynamic=n.motion_state == "dynamic",
                         observation_count=n.metadata.get("num_detections", 1), last_seen_frame=key[1],
                         metadata=dict(timestamp_s=frame["timestamp_s"], source_node_id=n.metadata["frozen_object_id"], motion_state=n.motion_state))
                         for i, n in nodes.items()}
        edges = full_current(relation_input, frame_index=key[1], heading=frame["heading"], timestamp_s=frame["timestamp_s"]).edges
        payload = sorted(json.dumps(e.to_dict(), sort_keys=True) for e in edges)
        graph_hashes.append(dict(scene=key[0], frame_idx=key[1], nodes=len(nodes), edges=len(edges),
                                 navigation_graph_payload_sha256=hashlib.sha256(json.dumps(payload).encode()).hexdigest()))
        forward = np.asarray(frame["heading"])
        forward /= np.linalg.norm(forward)
        origin = np.asarray(frame["camera_pose"])[:2, 3]
        official = truth[key]
        m, n = len(frame["predictions"]), len(official)
        cost = np.full((m, n+m), 1e9)
        cost[:, n:] = 0
        for i, pred in enumerate(frame["predictions"]):
            for j, obj in enumerate(official):
                distance = np.linalg.norm(np.asarray(pred["center"][:2])-obj["center"][:2])
                if pred["category"] == obj["category"] and np.isfinite(distance) and distance <= 3:
                    cost[i, j] = distance/4-(m+1)
        rr, cc = linear_sum_assignment(cost)
        binding = {i: official[j]["instance_token"] for i, j in zip(rr, cc) if j < n and cost[i, j] < 0}
        contexts[key] = nodes, eligible, indices, edges, forward, origin, binding
    counts = Counter()
    for task_id, query in queries.items():
        row = actual[task_id]
        for key in query:
            assert row[key] == query[key]
        nodes, eligible, indices, edges, forward, origin, binding = contexts[query["scene"], query["frame_idx"]]
        result = select_task(query["query_text"], nodes, eligible, edges, heading=forward, origin=origin,
                             reference_frame=query["reference_frame"])
        assert result == row["inference"], task_id
        selected = result["selected_id"]
        if selected is not None:
            assert eligible[selected] and nodes[selected].motion_state == "static"
        label = labels[task_id]
        assessment = row["evaluation_only"]
        assert assessment["label_status"] == label["label_status"]
        assert assessment["positive"] == label["positive"]
        assert assessment["acceptable_instances"] == label["acceptable_instances"]
        identity = binding.get(indices.get(selected))
        assert assessment["selected_identity"] == identity
        scorable = label["label_status"] == "scorable"
        positive = label["positive"]
        correct = identity in label["acceptable_instances"] if scorable and positive else None
        assert assessment["selected_correct"] == correct
        counts.update(dict(tasks=1, scorable=int(scorable), positive=int(scorable and positive),
                           negative=int(scorable and not positive), pending=int(not scorable),
                           positive_selected_correct=int(bool(correct)),
                           negative_false_selection=int(scorable and not positive and selected is not None),
                           rejected=int(selected is None), pending_selected=int(not scorable and selected is not None),
                           engineering_subgoal_emitted=int(bool(row["engineering_region"] and row["engineering_region"]["status"] == "ENGINEERING_ROAD_SUBGOAL")),
                           vehicle_route_success=0, parking_success=0))
        region = row["engineering_region"]
        if region:
            assert region["vehicle_safety_authorized"] is False and region["route_found"] is False
            assert region["GT_region_available"] is False and region["parking_slot_claim"] is False
    summary = json.loads((run/"navigation_summary.json").read_text())
    assert dict(counts) == summary["counts"]
    assert sha(nodepath) == summary["node_input_sha256"]
    report = dict(passed=True, counts=dict(counts), tasks=1200, frames=120,
                  all_selections_replayed_from_unified_nodes_and_current_graph=True,
                  original_index_and_geometry_exact=True, official_labels_only_used_for_scoring=True,
                  no_negative_or_pending_dropped=True, near6_current_graph=True,
                  cached_regression_vs_navigation_graph="same builder/policy/geometry; navigation new motion metadata may change confidence; not identical archived payload claim",
                  engineering_region="8 outputs only; no independent region GT, path or parking credit",
                  task_scope="constructed static object approach, not audited natural language or parking",
                  node_input_sha256=sha(nodepath), navigation_output_sha256=sha(run/"navigation_tasks.jsonl"),
                  navigation_graphs=graph_hashes)
    (REPORT/"navigation_independent_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "navigation_graphs"}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-run", default="node_motion_unified_20261125_v2")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--node-scope", choices=["R1", "S2"])
    group.add_argument("--directions", choices=["R1", "S2"])
    group.add_argument("--relations", action="store_true")
    group.add_argument("--motion-evaluation", action="store_true")
    group.add_argument("--navigation", action="store_true")
    args = parser.parse_args()
    if args.relations:
        audit_relations()
    elif args.directions:
        audit_directions(args.node_run, args.directions)
    elif args.motion_evaluation:
        audit_motion_evaluation(args.node_run)
    elif args.navigation:
        audit_navigation(args.node_run)
    else:
        audit_nodes(args.node_run, args.node_scope)


if __name__ == "__main__":
    main()
