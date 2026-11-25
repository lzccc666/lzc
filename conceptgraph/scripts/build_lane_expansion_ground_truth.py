"""Author versioned queries and independent official-instance targets."""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from conceptgraph.nuscenes.benchmark_ground_truth import (
    CATEGORY_TEXT,
    accepted_targets,
    official_category,
)
from conceptgraph.nuscenes.expansion_benchmark import content_sha256, iter_records
from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description="从官方实例与几何标注生成独立基准真值")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    from pathlib import Path

    root = Path(args.root).resolve()
    if not root.is_relative_to(DEFAULT_ROOT):
        raise ValueError("GT outputs must stay in expansion root")
    destinations = [
        root / n
        for n in (
            "benchmark_protocol.json",
            "benchmark_queries.jsonl",
            "ground_truth_frames.jsonl",
            "ground_truth_manifest.json",
            "annotation_audit.json",
        )
    ]
    if any(p.exists() for p in destinations):
        raise FileExistsError(
            "GT version exists; refusing to overwrite or silently relabel"
        )
    selection = json.loads((root / "scene_selection_manifest.json").read_text())
    meta = PROJECT_ROOT / "dataset/nuscenes_trainval_raw/v1.0-trainval"
    categories = {r["token"]: r["name"] for r in iter_records(meta / "category.json")}
    instances = {
        r["token"]: categories[r["category_token"]]
        for r in iter_records(meta / "instance.json")
    }
    alignments = {}
    samples = {}
    for scene in selection["scenes"]:
        name = scene["scene"]
        alignment = json.loads(
            (root / "converted" / name / "frame_alignment.json").read_text()
        )
        poses = np.loadtxt(root / "converted" / name / "poses.txt").reshape(-1, 4, 4)
        alignments[name] = (alignment, poses)
        for row in alignment:
            samples[row["sample_token"]] = []
    for annotation in iter_records(meta / "sample_annotation.json"):
        if annotation["sample_token"] in samples:
            category = official_category(instances[annotation["instance_token"]])
            if category:
                samples[annotation["sample_token"]].append(
                    {
                        "annotation_token": annotation["token"],
                        "instance_token": annotation["instance_token"],
                        "category": category,
                        "center": annotation["translation"],
                        "size_wlh": annotation["size"],
                        "rotation_wxyz": annotation["rotation"],
                        "visibility_token": annotation["visibility_token"],
                        "num_lidar_pts": annotation["num_lidar_pts"],
                    }
                )
    templates = (
        [{"category": c, "nearest": False} for c in CATEGORY_TEXT]
        + [{"category": c, "nearest": True} for c in ("car", "truck", "pedestrian")]
        + [
            {
                "category": "car",
                "nearest": False,
                "anchor_category": "truck",
                "relation": "left_of",
            }
        ]
    )
    protocol = {
        "schema_version": "official-instance-query-v1",
        "selection_sha256": selection["sha256"],
        "templates": templates,
        "query_frame": 0,
        "online_query_indices": [0, 6, 9],
        "target_scope": "official annotated object center visible in current CAM_FRONT; visibility_token>=2; camera depth 1..65m",
        "nearest_tolerance_m": 0.5,
        "relation_frame": "camera +Z projected into world XY; left perpendicular",
        "relation_distance_max_m": 15.0,
        "relation_component_min_m": 1.0,
        "association_maximum_xy_distance_m": 3.0,
        "association_ambiguity_margin_m": 0.5,
        "human_reviewed": False,
        "methods_can_access_ground_truth": False,
        "test_evaluation_started": False,
        "planning_reachability_ground_truth": "pending independent topology/obstacle audit; not derived from tested planner",
    }
    frames, queries = [], []
    for scene in selection["scenes"]:
        name = scene["scene"]
        alignment, poses = alignments[name]
        first_objects = []
        for row, camera_world in zip(alignment, poses):
            intrinsic = np.asarray(row["intrinsic"])
            visible = []
            for obj in samples[row["sample_token"]]:
                camera = camera_world[:3, :3].T @ (
                    np.asarray(obj["center"]) - camera_world[:3, 3]
                )
                uv = intrinsic @ camera
                if (
                    1 < camera[2] < 65
                    and int(obj["visibility_token"]) >= 2
                    and 0 <= uv[0] / uv[2] < 1600
                    and 0 <= uv[1] / uv[2] < 900
                ):
                    visible.append(obj)
            visible.sort(key=lambda o: o["instance_token"])
            frames.append(
                {
                    "scene": name,
                    "frame_idx": row["frame_idx"],
                    "sample_token": row["sample_token"],
                    "objects": visible,
                }
            )
            if row["frame_idx"] == 0:
                first_objects = visible
        for i, template in enumerate(templates):
            category = template["category"]
            target = CATEGORY_TEXT[category]
            if template.get("anchor_category"):
                text = "前往当前画面中卡车左侧的小汽车附近"
            else:
                text = f"前往当前画面中{'最近的' if template['nearest'] else ''}{target}附近"
            tokens = accepted_targets(
                first_objects,
                category,
                poses[0, :2, 3],
                nearest=template["nearest"],
                anchor_category=template.get("anchor_category"),
                relation=template.get("relation"),
                forward_xy=poses[0, :2, 2],
            )
            queries.append(
                {
                    "query_id": f"{name}-q{i:02}",
                    "scene": name,
                    "role": scene["role"],
                    "frame_idx": 0,
                    "timestamp": alignment[0]["timestamp"],
                    "sample_token": alignment[0]["sample_token"],
                    "query_text": text,
                    "query_type": "relation"
                    if template.get("relation")
                    else "nearest"
                    if template["nearest"]
                    else "category",
                    "accepted_instance_tokens": tokens,
                    "no_matching_target": not tokens,
                    "online_replay_selected": i in protocol["online_query_indices"],
                    "ground_truth_source": "nuScenes official instance/category/translation/visibility and independent geometry",
                    "label_status": "official_derived_not_human_reviewed",
                    "reachable_label": None,
                    "unreachable_label_status": "pending_independent_audit",
                }
            )
    protocol["sha256"] = content_sha256(protocol)
    destinations[0].write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
    )
    for path, rows in ((destinations[1], queries), (destinations[2], frames)):
        with path.open("x") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
    manifest = {
        "protocol_sha256": protocol["sha256"],
        "queries_sha256": content_sha256(queries),
        "official_frames_sha256": content_sha256(frames),
        "query_count": len(queries),
        "scene_count": len(selection["scenes"]),
        "frame_count": len(frames),
        "source_tables": [
            "sample_annotation",
            "instance",
            "category",
            "sample_data",
            "calibrated_sensor",
            "ego_pose",
        ],
        "derivation_uses_qwen_or_rule_ranking_or_astar": False,
        "node_association_validation": "pending",
        "test_evaluation_started": False,
    }
    destinations[3].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    audit = {
        "label_status_counts": dict(Counter(q["label_status"] for q in queries)),
        "no_match_queries": sum(q["no_matching_target"] for q in queries),
        "test_query_count": sum(q["role"] == "sealed_test" for q in queries),
        "human_review_count": 0,
        "color_and_fine_attribute_truth": "not available; no fabricated labels",
        "near_route_and_crop_audit": "pending",
        "node_mapping_review": "pending",
        "unreachable_and_projection_difficulty_coverage": "pending independent proof",
        "retrieval_metrics_computed": False,
    }
    destinations[4].write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
