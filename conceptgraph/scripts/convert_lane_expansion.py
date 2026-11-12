"""Convert selected official keyframes into a new, source-aligned dataset root."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from conceptgraph.nuscenes.expansion_benchmark import iter_records
from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT, PROJECT_ROOT
from conceptgraph.scripts.preprocess_nuscenes_to_minimal import (
    build_transform,
    inverse_transform,
    load_lidar_points,
)


def nearest_depth_image(points, intrinsic, height, width, minimum=1.0, maximum=65.0):
    points = np.asarray(points, dtype=float)
    finite = np.isfinite(points).all(axis=1)
    valid = finite & (points[:, 2] > minimum) & (points[:, 2] < maximum)
    points = points[valid]
    depth = np.full((height, width), np.inf, dtype=float)
    if len(points):
        uv = (np.asarray(intrinsic) @ points.T).T
        uv = np.rint(uv[:, :2] / uv[:, 2, None]).astype(int)
        inside = (
            (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        )
        # A z-buffer must retain the nearest return when multiple points share a pixel.
        np.minimum.at(depth, (uv[inside, 1], uv[inside, 0]), points[inside, 2])
    depth[~np.isfinite(depth)] = 0
    return depth


def main():
    parser = argparse.ArgumentParser(
        description="批量转换冻结manifest中的真实相机和激光帧"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--scene", action="append")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(DEFAULT_ROOT):
        raise ValueError("conversion must be isolated under expansion root")
    sources = json.loads((root / "source_file_index.json").read_text())
    selected = sorted(args.scene or sources)
    if set(selected) - sources.keys():
        raise ValueError("scene not in frozen selection")
    meta = PROJECT_ROOT / "dataset/nuscenes_trainval_raw/v1.0-trainval"
    calibration = {r["token"]: r for r in iter_records(meta / "calibrated_sensor.json")}
    required_poses = {
        sensor["ego_pose_token"]
        for name in selected
        for frame in sources[name]["sequence"]
        for sensor in frame["sensors"].values()
    }
    poses = {
        r["token"]: r
        for r in iter_records(meta / "ego_pose.json")
        if r["token"] in required_poses
    }
    for name in selected:
        directory = root / "converted" / name
        if (directory / "conversion_complete.json").exists():
            print("EXISTS", name, flush=True)
            continue
        started = time.perf_counter()
        directory.mkdir(parents=True, exist_ok=True)
        for sub in ("color", "depth"):
            (directory / sub).mkdir(exist_ok=True)
        transforms, alignment = [], []
        for i, frame in enumerate(sources[name]["sequence"]):
            sensors = frame["sensors"]
            if set(sensors) != {"CAM_FRONT", "LIDAR_TOP"}:
                raise ValueError(f"missing keyframe metadata: {name}, {i}")
            paths = {}
            for channel, sd in sensors.items():
                source_path = sources[name]["source_files"][sd["filename"]]
                paths[channel] = (
                    Path(source_path) if source_path else root / "raw" / sd["filename"]
                )
                if not paths[channel].is_file():
                    raise FileNotFoundError(paths[channel])
            cam, lidar = (sensors[ch] for ch in ("CAM_FRONT", "LIDAR_TOP"))
            cc, lc = (calibration[s["calibrated_sensor_token"]] for s in (cam, lidar))
            cp, lp = (poses[s["ego_pose_token"]] for s in (cam, lidar))
            camera_world = build_transform(
                cp["translation"], cp["rotation"]
            ) @ build_transform(cc["translation"], cc["rotation"])
            lidar_world = build_transform(
                lp["translation"], lp["rotation"]
            ) @ build_transform(lc["translation"], lc["rotation"])
            lidar_camera = inverse_transform(camera_world) @ lidar_world
            points = load_lidar_points(paths["LIDAR_TOP"])
            camera_points = points @ lidar_camera[:3, :3].T + lidar_camera[:3, 3]
            image = imageio.imread(paths["CAM_FRONT"])
            depth = nearest_depth_image(
                camera_points, cc["camera_intrinsic"], *image.shape[:2]
            )
            color = directory / "color" / f"{i:06}.jpg"
            if not color.exists():
                os.link(paths["CAM_FRONT"], color)
            imageio.imwrite(
                directory / "depth" / f"{i:06}.png",
                np.rint(depth * 1000).astype(np.uint16),
            )
            transforms.append(camera_world.reshape(-1))
            alignment.append(
                {
                    "frame_idx": i,
                    "sample_token": frame["sample_token"],
                    "timestamp": frame["timestamp"],
                    "camera_source": str(paths["CAM_FRONT"]),
                    "lidar_source": str(paths["LIDAR_TOP"]),
                    "depth_source": "projected_official_lidar_sparse_zbuffer_nearest",
                    "intrinsic": cc["camera_intrinsic"],
                }
            )
        np.savetxt(directory / "poses.txt", transforms, fmt="%.10f")
        np.savetxt(directory / "intrinsics.txt", cc["camera_intrinsic"], fmt="%.10f")
        (directory / "frame_alignment.json").write_text(
            json.dumps(alignment, indent=2) + "\n"
        )
        complete = {
            "scene": name,
            "frames": len(alignment),
            "runtime_s": time.perf_counter() - started,
            "selection_sha256": json.loads(
                (root / "scene_selection_manifest.json").read_text()
            )["sha256"],
            "dense_depth_estimation_executed": False,
        }
        (directory / "conversion_complete.json").write_text(
            json.dumps(complete, indent=2) + "\n"
        )
        print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
