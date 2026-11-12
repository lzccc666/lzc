"""Exact pixel and perceptual cross-split audit of real detector/old Qwen crops."""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.fft import dctn

from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT, PROJECT_ROOT
from conceptgraph.scripts.run_incremental_semantic_navigation import write_json


def fingerprint(image, identity):
    rgb = image.convert("RGB")
    digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    tiny = np.asarray(rgb.convert("L").resize((32, 32)), dtype=float)
    transform = dctn(tiny, norm="ortho")[:8, :8].ravel()
    bits = np.packbits(transform > np.median(transform[1:])).tobytes().hex()
    return {**identity, "pixel_sha256": digest, "phash": bits, "size": list(rgb.size)}


def main():
    root = DEFAULT_ROOT
    destination = root / "crop_leakage_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    scenes = json.loads((root / "scene_selection_manifest.json").read_text())["scenes"]
    rows = []
    for scene in scenes:
        if not (
            root / "preprocessing/detections" / scene["scene"] / "complete.json"
        ).exists():
            raise ValueError("all detector inputs must finish before the crop gate")
        for path in sorted(
            (
                root
                / "converted"
                / scene["scene"]
                / "exp_streamlined_yolo_outdoor_objects_v1/detections"
            ).glob("*.pkl.gz")
        ):
            with gzip.open(path, "rb") as handle:
                payload = pickle.load(handle)
            for i, image in enumerate(payload["image_crops"]):
                rows.append(
                    fingerprint(
                        image,
                        {
                            "scene": scene["scene"],
                            "role": scene["role"],
                            "source": str(path),
                            "crop_index": i,
                        },
                    )
                )
    seen = set()
    for line in (
        PROJECT_ROOT / "outputs/project_v2/qwen_eval/dataset/object_attributes.jsonl"
    ).open():
        old = json.loads(line)
        if old["crop_path"] in seen:
            continue
        seen.add(old["crop_path"])
        with Image.open(old["crop_path"]) as image:
            rows.append(
                fingerprint(
                    image,
                    {
                        "scene": old["scene_id"],
                        "role": "old_qwen_dataset",
                        "source": old["crop_path"],
                        "original_split": old["split"],
                    },
                )
            )
    exact_groups = defaultdict(list)
    bands = defaultdict(set)
    others = [r for r in rows if r["role"] != "sealed_test"]
    for i, row in enumerate(others):
        value = int(row["phash"], 16)
        for j in range(8):
            bands[(j, (value >> (j * 8)) & 255)].add(i)
    near = []
    for row in rows:
        exact_groups[row["pixel_sha256"]].append(row)
        if row["role"] != "sealed_test":
            continue
        value = int(row["phash"], 16)
        candidates = set().union(
            *(bands[(j, (value >> (j * 8)) & 255)] for j in range(8))
        )
        # At <=6 changed bits, at least one of eight 8-bit bands is identical.
        matches = [
            (int(value ^ int(others[i]["phash"], 16)).bit_count(), i)
            for i in candidates
        ]
        if matches:
            distance, i = min(matches)
            if distance <= 6:
                near.append(
                    {"test": row, "other": others[i], "hamming_distance": distance}
                )
    exact = [
        group
        for group in exact_groups.values()
        if any(r["role"] == "sealed_test" for r in group)
        and any(r["role"] != "sealed_test" for r in group)
    ]
    write_json(root / "crop_fingerprint_manifest.json", {"crops": rows})
    write_json(
        destination,
        {
            "crop_count": len(rows),
            "old_qwen_crop_count": len(seen),
            "cross_split_exact_pixel_groups": exact,
            "near_crop_pairs_count": len(near),
            "near_crop_pairs": near,
            "phash_hamming_threshold": 6,
            "near_crop_does_not_establish_same_instance": True,
            "human_review_count": 0,
            "handling": "All pairs disclosed; no result-driven scene removal or label tuning. No geographic or appearance-independent generalization claim. Colors and fine attributes remain unknown.",
            "exact_duplicate_gate_passed": not exact,
        },
    )
    print(
        json.dumps(
            {
                "crop_count": len(rows),
                "exact_groups": len(exact),
                "near_pairs": len(near),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
