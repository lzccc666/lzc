"""Audit actual converted frames, exact/perceptual duplicates and route overlap."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.fft import dctn
from scipy.spatial import cKDTree

from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT


def main():
    root = DEFAULT_ROOT
    scenes = json.loads((root / "scene_selection_manifest.json").read_text())["scenes"]
    positions, images, integrity = {}, [], []
    for scene in scenes:
        name = scene["scene"]
        directory = root / "converted" / name
        poses = np.loadtxt(directory / "poses.txt").reshape(-1, 4, 4)
        positions[name] = poses[:, :2, 3]
        expected = scene["frame_count"]
        color = sorted((directory / "color").glob("*.jpg"))
        depth = sorted((directory / "depth").glob("*.png"))
        dimensions = []
        nonzero = []
        for path in color:
            with Image.open(path) as image:
                dimensions.append(list(image.size))
                thumb = np.asarray(image.convert("L").resize((32, 32)), dtype=float)
            transform = dctn(thumb, norm="ortho")[:8, :8].ravel()
            perceptual = transform > np.median(transform[1:])
            images.append(
                {
                    "scene": name,
                    "role": scene["role"],
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "phash": np.packbits(perceptual).tobytes().hex(),
                }
            )
        for path in depth:
            with Image.open(path) as image:
                data = np.asarray(image)
                nonzero.append(float(np.count_nonzero(data) / data.size))
        valid = (
            len(color) == len(depth) == len(poses) == expected
            and np.isfinite(poses).all()
        )
        integrity.append(
            {
                "scene": name,
                "passed": bool(valid),
                "expected_frames": expected,
                "color_frames": len(color),
                "depth_frames": len(depth),
                "pose_frames": len(poses),
                "image_dimensions": sorted({tuple(x) for x in dimensions}),
                "depth_nonzero_fraction_mean": float(np.mean(nonzero))
                if nonzero
                else None,
                "depth_type": "sparse official LIDAR projection; not dense estimated depth",
            }
        )
    route_pairs = []
    for a in scenes:
        if a["role"] != "sealed_test":
            continue
        for b in scenes:
            if (
                b["role"] == "sealed_test"
                or a["strata"]["location"] != b["strata"]["location"]
            ):
                continue
            distances, _ = cKDTree(positions[b["scene"]]).query(positions[a["scene"]])
            if float(np.min(distances)) < 30:
                route_pairs.append(
                    {
                        "test_scene": a["scene"],
                        "other_scene": b["scene"],
                        "minimum_distance_m": float(np.min(distances)),
                        "test_frames_within_30m_fraction": float(
                            np.mean(distances < 30)
                        ),
                    }
                )
    hashes = defaultdict(list)
    for row in images:
        hashes[row["sha256"]].append(row)
    exact = [
        group
        for group in hashes.values()
        if {x["role"] for x in group} & {"sealed_test"}
        and any(x["role"] != "sealed_test" for x in group)
    ]
    heldout = [i for i in images if i["role"] == "sealed_test"]
    others = [i for i in images if i["role"] != "sealed_test"]
    near = []
    for a in heldout:
        bits_a = int(a["phash"], 16)
        matches = [(int(bits_a ^ int(b["phash"], 16)).bit_count(), b) for b in others]
        if matches:
            distance, b = min(matches, key=lambda pair: pair[0])
            if distance <= 6:
                near.append(
                    {
                        "test_image": a["path"],
                        "other_image": b["path"],
                        "phash_hamming_distance": distance,
                    }
                )
    report = {
        "passed": all(r["passed"] for r in integrity),
        "converted_scenes_checked": len(scenes),
        "frames_checked": len(images),
        "scenes": integrity,
        "input_image_manifest": images,
    }
    audit = {
        "same_log_isolation": "see leakage_audit.json",
        "cross_split_exact_image_groups": exact,
        "cross_split_near_image_pairs": near,
        "near_image_threshold_hamming": 6,
        "cross_split_near_routes": route_pairs,
        "near_route_threshold_m": 30,
        "near_image_human_reviewed": False,
        "object_crop_audit": "pending actual detector crops",
        "test_independence_claim": "different scene and acquisition log; geographic overlap explicitly disclosed",
        "test_release_gate": "not released; exact duplicate/near-image review and node association pending",
    }
    (root / "dataset_integrity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (root / "leakage_extended_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "converted_scenes": len(scenes),
                "frames": len(images),
                "passed": report["passed"],
                "exact_duplicate_groups": len(exact),
                "near_image_pairs": len(near),
                "near_route_pairs": len(route_pairs),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
