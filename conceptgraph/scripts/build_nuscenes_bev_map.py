from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath


SCENE_TO_LOCATION = {
    "scene-0003": "singapore-onenorth",
    "scene-0045": "singapore-onenorth",
    "scene-0128": "singapore-onenorth",
    "scene-0133": "singapore-onenorth",
    "scene-0168": "boston-seaport",
    "scene-0655": "boston-seaport",
    "scene-0916": "singapore-queenstown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a nuScenes drivable/carpark BEV occupancy map for one scene.")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--scene-name", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--nuscenes-meta-root",
        type=Path,
        default=Path("/root/autodl-tmp/lizichong/dataset/nuscenes"),
        help="Root containing v1.0-mini or v1.0-trainval metadata for automatic map location lookup.",
    )
    parser.add_argument("--version", default=None, help="nuScenes metadata version. Auto-detected when omitted.")
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--padding-m", type=float, default=20.0)
    return parser.parse_args()


def load_map_json(map_root: Path, location: str) -> dict:
    path = map_root / "expansion" / f"{location}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_location(scene_name: str, meta_root: Path, version: str | None) -> str | None:
    versions = [version] if version else ["v1.0-mini", "v1.0-trainval"]
    for item in versions:
        if item is None:
            continue
        scene_path = meta_root / item / "scene.json"
        log_path = meta_root / item / "log.json"
        if not scene_path.exists() or not log_path.exists():
            continue
        with open(scene_path, "r", encoding="utf-8") as f:
            scenes = json.load(f)
        with open(log_path, "r", encoding="utf-8") as f:
            logs = {record["token"]: record for record in json.load(f)}
        scene = next((record for record in scenes if record.get("name") == scene_name), None)
        if scene is None:
            continue
        log = logs.get(scene.get("log_token", ""))
        if log is not None and log.get("location"):
            return str(log["location"])
    return None


def load_pose_xy(scene_dir: Path) -> np.ndarray:
    poses = np.loadtxt(scene_dir / "poses.txt")
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    mats = poses.reshape(-1, 4, 4)
    return mats[:, :2, 3]


def polygon_from_token(map_json: dict, polygon_token: str, node_by_token: dict) -> np.ndarray | None:
    polygon_rec = next((item for item in map_json["polygon"] if item["token"] == polygon_token), None)
    if polygon_rec is None:
        return None
    coords = []
    for node_token in polygon_rec["exterior_node_tokens"]:
        node = node_by_token.get(node_token)
        if node is None:
            return None
        coords.append([float(node["x"]), float(node["y"])])
    if len(coords) < 3:
        return None
    return np.asarray(coords, dtype=float)


def collect_allowed_polygons(map_json: dict) -> list[np.ndarray]:
    node_by_token = {item["token"]: item for item in map_json["node"]}
    polygons: list[np.ndarray] = []

    for record in map_json.get("drivable_area", []):
        for token in record.get("polygon_tokens", []):
            poly = polygon_from_token(map_json, token, node_by_token)
            if poly is not None:
                polygons.append(poly)

    for layer_name in ("carpark_area", "lane", "lane_connector", "road_segment"):
        for record in map_json.get(layer_name, []):
            token = record.get("polygon_token")
            if not token:
                continue
            poly = polygon_from_token(map_json, token, node_by_token)
            if poly is not None:
                polygons.append(poly)
    return polygons


def rasterize_allowed_area(polygons: list[np.ndarray], min_xy: np.ndarray, max_xy: np.ndarray, resolution: float) -> np.ndarray:
    width = int(np.ceil((max_xy[0] - min_xy[0]) / resolution)) + 1
    height = int(np.ceil((max_xy[1] - min_xy[1]) / resolution)) + 1
    cols = min_xy[0] + np.arange(width) * resolution
    rows = min_xy[1] + np.arange(height) * resolution
    xs, ys = np.meshgrid(cols, rows)
    points = np.stack([xs.ravel(), ys.ravel()], axis=1)

    free_mask = np.zeros(points.shape[0], dtype=bool)
    for poly in polygons:
        path = MplPath(poly)
        free_mask |= path.contains_points(points)

    free_mask = free_mask.reshape(height, width)
    occupancy = (~free_mask).astype(np.uint8)
    return occupancy


def save_preview(occupancy: np.ndarray, origin_xy: np.ndarray, resolution: float, trajectory_xy: np.ndarray, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [
        origin_xy[0],
        origin_xy[0] + (occupancy.shape[1] - 1) * resolution,
        origin_xy[1],
        origin_xy[1] + (occupancy.shape[0] - 1) * resolution,
    ]
    ax.imshow(1 - occupancy, origin="lower", extent=extent, cmap="Greys", alpha=0.9)
    ax.plot(trajectory_xy[:, 0], trajectory_xy[:, 1], "--", color="#2563eb", linewidth=1.8, label="ego trajectory")
    ax.scatter(trajectory_xy[0, 0], trajectory_xy[0, 1], c="#16a34a", s=60, marker="o", label="start")
    ax.scatter(trajectory_xy[-1, 0], trajectory_xy[-1, 1], c="#dc2626", s=80, marker="X", label="end")
    ax.set_title("nuScenes drivable/carpark BEV")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    location = infer_location(args.scene_name, args.nuscenes_meta_root, args.version)
    if location is None:
        location = SCENE_TO_LOCATION.get(args.scene_name)
    if location is None:
        raise KeyError(f"No map location configured for {args.scene_name}")

    trajectory_xy = load_pose_xy(args.scene_dir)
    min_xy = trajectory_xy.min(axis=0) - args.padding_m
    max_xy = trajectory_xy.max(axis=0) + args.padding_m

    map_json = load_map_json(args.map_root, location)
    polygons = collect_allowed_polygons(map_json)
    occupancy = rasterize_allowed_area(polygons, min_xy, max_xy, args.resolution)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        occupancy=occupancy.astype(np.uint8),
        origin_xy=min_xy.astype(np.float32),
        resolution=np.float32(args.resolution),
        location=location,
        scene_name=args.scene_name,
    )
    save_preview(
        occupancy=occupancy,
        origin_xy=min_xy,
        resolution=args.resolution,
        trajectory_xy=trajectory_xy,
        out_png=args.output.with_suffix(".png"),
    )
    print(f"Saved BEV map to {args.output}")


if __name__ == "__main__":
    main()
