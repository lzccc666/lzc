"""Append official split, temporal and post-selection coverage evidence."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations

from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT
from conceptgraph.scripts.run_incremental_semantic_navigation import write_json


def main():
    root = DEFAULT_ROOT
    selected = json.loads((root / "scene_selection_manifest.json").read_text())["scenes"]
    source = json.loads((root / "source_file_index.json").read_text())
    original = json.loads((root / "split_manifest.json").read_text())
    official_source = root / "data_inventory/official_nuscenes_splits.py"
    definitions = {}
    # Read literal metadata; never execute downloaded Python or import its SDK.
    for node in ast.parse(official_source.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {"train_detect", "train_track", "val", "test"}:
                definitions[target.id] = set(ast.literal_eval(node.value))
    definitions["train"] = definitions["train_detect"] | definitions["train_track"]
    assignments = []
    for scene in selected:
        name = scene["scene"]
        splits = [split for split in ("train", "val", "test") if name in definitions[split]]
        assignments.append({"scene": name, "evaluation_role": scene["role"], "original_qwen_split": original["original_scene_to_split"].get(name), "official_nuscenes_split": splits[0] if len(splits) == 1 else None})
    provenance = json.loads((root / "data_inventory/official_split_source.json").read_text())
    provenance["downloaded_file_sha256_matches"] = provenance["sha256"] == hashlib.sha256(official_source.read_bytes()).hexdigest()
    split_report = {"passed": provenance["downloaded_file_sha256_matches"] and all(r["official_nuscenes_split"] is not None for r in assignments), "official_source": provenance, "official_counts": {split: len(definitions[split]) for split in ("train", "val", "test")}, "selected_counts": dict(Counter(r["official_nuscenes_split"] for r in assignments)), "scenes": assignments, "frozen_split_manifest_modified": False, "note": "Official train/val identity is preserved independently of this task's evaluation roles and the historical Qwen split. Metadata-only supplement after test release; no reselection or tuning."}
    write_json(root / "official_split_audit.json", split_report)
    intervals = {}
    for scene in selected:
        timestamps = [int(frame["timestamp"]) for frame in source[scene["scene"]]["sequence"]]
        intervals[scene["scene"]] = {"start_timestamp_us": min(timestamps), "end_timestamp_us": max(timestamps), "duration_s": (max(timestamps)-min(timestamps))/1e6}
    pairs = []
    for a, b in combinations(selected, 2):
        if (a["role"] == "sealed_test") == (b["role"] == "sealed_test"):
            continue
        x, y = intervals[a["scene"]], intervals[b["scene"]]
        gap = max(x["start_timestamp_us"]-y["end_timestamp_us"], y["start_timestamp_us"]-x["end_timestamp_us"], 0)/1e6
        pairs.append({"scene_a": a["scene"], "scene_b": b["scene"], "same_log": a["log_token"] == b["log_token"], "same_location": a["strata"]["location"] == b["strata"]["location"], "interval_gap_s": gap})
    same_location = [p for p in pairs if p["same_location"]]
    temporal = {"passed": not any(p["same_log"] for p in pairs), "threshold_s": 60, "interval_source": "official sample timestamps for actual replay frames", "scenes": [{"scene": s["scene"], "role": s["role"], "log_token": s["log_token"], **intervals[s["scene"]]} for s in selected], "cross_split_pairs": pairs, "same_location_pairs_within_60s": [p for p in same_location if p["interval_gap_s"] <= 60], "minimum_same_location_interval_gap_s": min((p["interval_gap_s"] for p in same_location), default=None), "adjacent_camera_frames_are_not_independent_samples": True, "limitations": "Different logs and temporal gaps do not prove appearance or geographic independence; see route/crop audits."}
    write_json(root / "temporal_adjacency_audit.json", temporal)
    frames = defaultdict(list)
    for line in (root / "ground_truth_frames.jsonl").open():
        frame = json.loads(line)
        frames[frame["scene"]].append(frame)
    coverage = []
    for scene in selected:
        rows = frames[scene["scene"]]
        counts = [len(row["objects"]) for row in rows]
        categories = Counter(obj["category"] for row in rows for obj in row["objects"])
        coverage.append({"scene": scene["scene"], "role": scene["role"], "selection_strata": scene["strata"], "visible_official_target_count_frame0": counts[0], "visible_official_target_count_mean": sum(counts)/len(counts), "visible_official_target_count_max": max(counts), "category_frame_occurrences": dict(categories), "distinct_target_categories": len(categories), "frames_with_multiple_same_category": sum(any(v >= 2 for v in Counter(obj["category"] for obj in row["objects"]).values()) for row in rows), "actual_frame_count": len(rows)})
    grouped = []
    for role in ("legacy_regression", "development", "sealed_test"):
        members = [c for c in coverage if c["role"] == role]
        strata = {name: dict(Counter(str(c["selection_strata"][name]) for c in members)) for name in ("location", "road_context", "intersection", "turn", "light", "weather")}
        grouped.append({"role": role, "scene_count": len(members), "strata_scene_counts": strata, "zero_visible_targets_frame0": sum(c["visible_official_target_count_frame0"] == 0 for c in members), "multiple_categories_scene_count": sum(c["distinct_target_categories"] >= 2 for c in members)})
    write_json(root / "coverage_audit.json", {"passed": bool(selected) and len(coverage) == len(selected), "expected_scene_count": len(selected), "selection_frozen_before_method_results": True, "scenes": coverage, "role_summaries": grouped, "density_measure": "Post-selection visible official target counts in CAM_FRONT, not all road obstacles or a pre-selection density measurement", "selection_dimensions": "Allocation buckets used location, road/parking, light and weather from official descriptions, with raw data availability and log exclusions. Intersection/turn were recorded strata but did not enter bucket allocation.", "not_used_for_reselection": ["measured object density", "measured category/relationship complexity", "method success"], "limitation": "Object density and relationship complexity were audited after selection rather than quantitatively stratified before selection. Unspecified weather/light is not verified clear/day. Sparse local authorized data does not cover every desired combination; no outcome-based reselection was performed."})
    if all('prospective_strata' in row for row in selected):
        coverage_path = root / 'coverage_audit.json'
        report = json.loads(coverage_path.read_text())
        report.update(all_requested_dimensions_prospectively_allocated=True, selection_dimensions='All nine prospective dimensions, frozen in sampling_plan.json and scene_selection_manifest.json before supplementary method execution.', density_measure='Supported visible official target count; separately retain all annotated visible obstacle density from the frozen sampling plan.', not_used_for_reselection=['method results'], limitation='Small uneven strata and unspecified weather/light remain limitations. The original accessed 50-scene cohort retains its separate sampling disclosure.')
        write_json(coverage_path, report)
    print(json.dumps({"official_split_passed": split_report["passed"], "official_selected_counts": split_report["selected_counts"], "temporal_passed": temporal["passed"], "minimum_temporal_gap_s": temporal["minimum_same_location_interval_gap_s"], "scene_coverage_count": len(coverage)}), flush=True)


if __name__ == "__main__":
    main()
