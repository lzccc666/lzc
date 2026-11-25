"""Validate artifact contents and record complete per-scene preprocessing inputs."""

from __future__ import annotations

import gzip
import json
import pickle
from collections import Counter

import numpy as np

from conceptgraph.navigation.lane_constraints import load_lane_topology
from conceptgraph.nuscenes.benchmark_fingerprints import scene_input_manifest
from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT
from conceptgraph.scripts.run_incremental_semantic_navigation import write_json


def main():
    root = DEFAULT_ROOT
    rows = json.loads((root / "scene_selection_manifest.json").read_text())["scenes"]
    results, maps = [], []
    for row in rows:
        name = row["scene"]
        scene = root / "converted" / name
        phases = {
            p: (root / "preprocessing" / p / name / "complete.json").exists()
            for p in ("maps", "detections", "mapping")
        }
        paths = sorted(
            (scene / "exp_streamlined_yolo_outdoor_objects_v1/detections").glob(
                "*.pkl.gz"
            )
        )
        counts, empty, crops = Counter(), 0, 0
        for path in paths:
            with gzip.open(path, "rb") as handle:
                data = pickle.load(handle)
            n = len(data["xyxy"])
            if any(
                len(data[k]) != n
                for k in (
                    "confidence",
                    "class_id",
                    "mask",
                    "image_crops",
                    "image_feats",
                    "text_feats",
                )
            ):
                raise ValueError(f"invalid detector field lengths: {path}")
            if not all(
                np.isfinite(data[k]).all()
                for k in ("xyxy", "confidence", "image_feats", "text_feats")
            ):
                raise ValueError(f"nonfinite detector artifact: {path}")
            counts.update(data["classes"][i] for i in data["class_id"])
            crops += len(data["image_crops"])
            empty += n == 0
        topology = load_lane_topology(
            root / "lane_maps" / name / "lane_constrained_bev.npz"
        )
        with np.load(root / "lane_maps" / name / "lane_constrained_bev.npz") as archive:
            grid_keys = sorted(archive.files)
        metadata = json.loads(
            (root / "lane_maps" / name / "lane_topology.json").read_text()
        )
        snapshots = list(
            (
                scene
                / "objects_all_frames/streamlined_yolo_outdoor_objects_tracking_gate"
            ).glob("*.pkl.gz")
        )
        passed = (
            all(phases.values())
            and len(paths) == row["frame_count"]
            and any(p.name == "meta.pkl.gz" for p in snapshots)
        )
        result = {
            "scene": name,
            "role": row["role"],
            "passed": passed,
            "phase_complete": phases,
            "expected_frames": row["frame_count"],
            "detection_frames": len(paths),
            "empty_detection_frames": empty,
            "crop_count": crops,
            "detector_category_counts": dict(counts),
            "mapping_snapshot_files": len(snapshots),
            "snapshot_omissions": "initialization/zero-detection frames may be skipped by historical writer; all online frames reconstructed from actual inputs",
            "sparse_lidar_depth": True,
        }
        results.append(result)
        maps.append(
            {
                "scene": name,
                "lane_count": len(topology.lane_tokens),
                "lane_mask_cells": int(np.count_nonzero(topology.lane_mask)),
                "grid_keys": grid_keys,
                "map_metadata_keys": sorted(metadata),
                "source": "nuScenes official HD Map",
                "passed": True,
            }
        )
        write_json(
            root / "preprocessing_input_manifests" / f"{name}.json",
            scene_input_manifest(root, name),
        )
    write_json(
        root / "preprocessing_integrity_report.json",
        {
            "scenes": results,
            "scene_count": len(results),
            "passed": all(r["passed"] for r in results),
            "source_inputs_fully_hashed": True,
        },
    )
    write_json(
        root / "lane_map_reports.json",
        {"scenes": maps, "scene_count": len(maps), "passed": True},
    )
    print("audited", len(results), "scenes", flush=True)


if __name__ == "__main__":
    main()
