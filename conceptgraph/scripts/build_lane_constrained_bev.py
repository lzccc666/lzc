"""Build ordinary and strict lane-level BEV maps from a nuScenes expansion map."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from conceptgraph.navigation.lane_geometry import contains_points

from conceptgraph.scripts.build_nuscenes_bev_map import (
    SCENE_TO_LOCATION,
    collect_allowed_polygons,
    infer_location,
    load_map_json,
    load_pose_xy,
    rasterize_allowed_area,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建nuScenes车道约束BEV和HD Map拓扑")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--nuscenes-meta-root", type=Path, default=Path("dataset/nuscenes")
    )
    parser.add_argument("--version", default=None)
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--padding-m", type=float, default=20.0)
    return parser.parse_args()


def polygon_from_token(
    map_json: dict[str, Any], polygon_token: str, nodes: dict[str, dict[str, Any]]
) -> np.ndarray | None:
    record = next(
        (
            item
            for item in map_json.get("polygon", [])
            if item.get("token") == polygon_token
        ),
        None,
    )
    if record is None or record.get("holes"):
        # Fail closed for unsupported polygon holes rather than treating them as road.
        return None
    coords = []
    for node_token in record.get("exterior_node_tokens", []):
        node = nodes.get(node_token)
        if node is None:
            return None
        coords.append([float(node["x"]), float(node["y"])])
    if len(coords) < 3:
        return None
    return np.asarray(coords, dtype=float)


def _principal_value(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _segment_signs(shape: str) -> tuple[int, int, int]:
    if shape not in {"LRL", "LSL", "LSR", "RLR", "RSR", "RSL"}:
        raise ValueError(f"invalid arcline shape: {shape}")
    first = 1 if shape in {"LRL", "LSL", "LSR"} else -1
    if shape == "RLR":
        middle = 1
    elif shape == "LRL":
        middle = -1
    else:
        middle = 0
    last = 1 if shape in {"LRL", "LSL", "RSL"} else -1
    return first, middle, last


def _advance(
    pose: tuple[float, float, float], curvature: float, distance: float
) -> tuple[float, float, float]:
    x, y, heading = pose
    if abs(curvature) < 1e-9:
        return (
            x + math.cos(heading) * distance,
            y + math.sin(heading) * distance,
            heading,
        )
    new_heading = heading + curvature * distance
    return (
        x + (math.sin(new_heading) - math.sin(heading)) / curvature,
        y + (-math.cos(new_heading) + math.cos(heading)) / curvature,
        _principal_value(new_heading),
    )


def _sample_arcline(record: dict[str, Any], resolution: float) -> np.ndarray:
    start = tuple(float(v) for v in record["start_pose"][:3])
    lengths = [max(0.0, float(v)) for v in record.get("segment_length", [])]
    total = sum(lengths)
    if total <= 1e-8:
        end = tuple(float(v) for v in record.get("end_pose", start)[:3])
        return np.asarray([start, end], dtype=float)
    signs = _segment_signs(str(record.get("shape", "SSS")))
    radius = float(record.get("radius", 0.0))
    samples = max(2, int(math.ceil(total / max(resolution, 0.05))) + 1)
    distances = np.linspace(0.0, total, samples)
    result = []
    for distance in distances:
        pose = start
        remaining = float(distance)
        for index, length in enumerate(lengths[:3]):
            step = min(remaining, length)
            curvature = (
                0.0
                if signs[index] == 0 or abs(radius) < 1e-9
                else signs[index] / radius
            )
            pose = _advance(pose, curvature, step)
            remaining -= step
            if remaining <= 1e-9:
                break
        result.append(pose)
    result[-1] = tuple(float(v) for v in record.get("end_pose", result[-1])[:3])
    return np.asarray(result, dtype=float)


def _lane_records(
    map_json: dict[str, Any], min_xy: np.ndarray, max_xy: np.ndarray, resolution: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nodes = {item["token"]: item for item in map_json.get("node", [])}
    arcline = {
        str(key): value for key, value in map_json.get("arcline_path_3", {}).items()
    }
    connectivity = {
        str(key): value for key, value in map_json.get("connectivity", {}).items()
    }
    records = []
    diagnostics = {
        "missing_polygon": [],
        "missing_arcline": [],
        "missing_heading": [],
        "invalid_connectivity": [],
    }
    for lane_type in ("lane_connector", "lane"):
        for item in map_json.get(lane_type, []):
            token = str(item.get("token", ""))
            polygon = polygon_from_token(
                map_json, str(item.get("polygon_token", "")), nodes
            )
            if not token or polygon is None:
                diagnostics["missing_polygon"].append(token)
                continue
            if (
                polygon[:, 0].max() < min_xy[0]
                or polygon[:, 0].min() > max_xy[0]
                or polygon[:, 1].max() < min_xy[1]
                or polygon[:, 1].min() > max_xy[1]
            ):
                continue
            paths = arcline.get(token, [])
            if not paths:
                diagnostics["missing_arcline"].append(token)
                continue
            try:
                centerline = np.concatenate(
                    [_sample_arcline(path, resolution / 2.0) for path in paths], axis=0
                )
            except (KeyError, ValueError, TypeError, IndexError):
                diagnostics["missing_heading"].append(token)
                continue
            if len(centerline) < 2 or not np.all(np.isfinite(centerline)):
                diagnostics["missing_heading"].append(token)
                continue
            start_pose = tuple(
                float(v) for v in paths[0].get("start_pose", centerline[0])[:3]
            )
            end_pose = tuple(
                float(v) for v in paths[-1].get("end_pose", centerline[-1])[:3]
            )
            heading = float(start_pose[2])
            if not math.isfinite(heading):
                diagnostics["missing_heading"].append(token)
                continue
            conn = connectivity.get(token, {})
            incoming = tuple(
                str(value)
                for value in conn.get("incoming", [])
                if str(value) in arcline
            )
            outgoing = tuple(
                str(value)
                for value in conn.get("outgoing", [])
                if str(value) in arcline
            )
            diagnostics["invalid_connectivity"].extend(
                {"token": token, "direction": direction, "value": str(value)}
                for direction in ("incoming", "outgoing")
                for value in conn.get(direction, [])
                if str(value) not in arcline
            )
            records.append(
                {
                    "token": token,
                    "lane_type": lane_type,
                    "polygon": polygon,
                    "centerline": centerline,
                    "heading": heading,
                    "start_pose": start_pose,
                    "end_pose": end_pose,
                    "incoming": incoming,
                    "outgoing": outgoing,
                }
            )
    records.sort(key=lambda item: item["token"])
    return records, diagnostics


def build_lane_map(
    map_json: dict[str, Any],
    trajectory_xy: np.ndarray,
    resolution: float,
    padding_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if (
        resolution <= 0
        or not math.isfinite(resolution)
        or not np.isfinite(trajectory_xy).all()
    ):
        raise ValueError("invalid map resolution or trajectory")
    min_xy = trajectory_xy.min(axis=0) - float(padding_m)
    max_xy = trajectory_xy.max(axis=0) + float(padding_m)
    ordinary = rasterize_allowed_area(
        collect_allowed_polygons(map_json), min_xy, max_xy, resolution
    )
    records, diagnostics = _lane_records(map_json, min_xy, max_xy, resolution)
    height, width = ordinary.shape
    lane_id_grid = np.full((height, width), -1, dtype=np.int32)
    heading_grid = np.full((height, width), np.nan, dtype=np.float32)
    centerline_distance_grid = np.full((height, width), np.inf, dtype=np.float32)
    cols = min_xy[0] + np.arange(width) * resolution
    rows = min_xy[1] + np.arange(height) * resolution
    xs, ys = np.meshgrid(cols, rows)
    points = np.stack([xs.ravel(), ys.ravel()], axis=1)
    # Primary IDs support previews only. Planning reconstructs ALL polygon memberships.
    records = sorted(
        records, key=lambda item: (item["lane_type"] != "lane_connector", item["token"])
    )
    tokens = [item["token"] for item in records]
    types = [item["lane_type"] for item in records]
    for lane_index, item in enumerate(records):
        mask = contains_points(item["polygon"], points).reshape(height, width)
        available = mask & (lane_id_grid < 0)
        lane_id_grid[available] = lane_index
        if np.any(available):
            lane_points = points[available.ravel()]
            distances, nearest = cKDTree(item["centerline"][:, :2]).query(lane_points)
            heading_grid[available] = item["centerline"][nearest, 2]
            centerline_distance_grid[available] = distances
    local_tokens = set(tokens)
    diagnostics["boundary_connectivity"] = [
        {"token": item["token"], "direction": direction, "value": value}
        for item in records
        for direction in ("incoming", "outgoing")
        for value in item[direction]
        if value not in local_tokens
    ]
    for item in records:
        for direction in ("incoming", "outgoing"):
            item[direction] = tuple(
                value for value in item[direction] if value in local_tokens
            )
    transitions = {item["token"]: sorted(set(item["outgoing"])) for item in records}
    geometry_payload = {
        item["token"]: {
            "lane_type": item["lane_type"],
            "polygon": item["polygon"].tolist(),
            "centerline": item["centerline"].tolist(),
            "heading": item["heading"],
            "start_pose": list(item["start_pose"]),
            "end_pose": list(item["end_pose"]),
            "incoming": list(item["incoming"]),
            "outgoing": list(item["outgoing"]),
        }
        for item in records
    }
    diagnostics.update(
        {
            "lane_count": len(records),
            "lane_cell_count": int(np.sum(lane_id_grid >= 0)),
            "grid_overlap_unassigned": int(
                np.sum((lane_id_grid < 0) & (ordinary == 0))
            ),
            "transition_endpoint_tolerance_m": 2.0,
        }
    )
    arrays = {
        "occupancy": ordinary.astype(np.uint8),
        "lane_id_grid": lane_id_grid,
        "lane_heading_grid": heading_grid,
        "centerline_distance_grid": centerline_distance_grid,
        "origin_xy": min_xy.astype(np.float64),
        "resolution": np.float64(resolution),
    }
    metadata = {
        "lane_tokens": tokens,
        "lane_types": types,
        "transitions": transitions,
        "geometries": geometry_payload,
        "origin_xy": min_xy.tolist(),
        "max_xy": max_xy.tolist(),
        "resolution": float(resolution),
        "diagnostics": diagnostics,
    }
    return arrays, metadata


def save_preview(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    trajectory_xy: np.ndarray,
    output: Path,
) -> None:
    origin = arrays["origin_xy"]
    resolution = float(arrays["resolution"])
    fig, ax = plt.subplots(figsize=(10, 10))
    occupancy = arrays["occupancy"]
    extent = [
        origin[0],
        origin[0] + (occupancy.shape[1] - 1) * resolution,
        origin[1],
        origin[1] + (occupancy.shape[0] - 1) * resolution,
    ]
    ax.imshow(1 - occupancy, origin="lower", extent=extent, cmap="Greys", alpha=0.2)
    for token, item in metadata["geometries"].items():
        polygon = np.asarray(item["polygon"], dtype=float)
        centerline = np.asarray(item["centerline"], dtype=float)
        color = "#f59e0b" if item["lane_type"] == "lane_connector" else "#2563eb"
        ax.plot(
            *np.vstack([polygon, polygon[0]]).T, color=color, alpha=0.12, linewidth=0.5
        )
        ax.plot(
            centerline[:, 0], centerline[:, 1], color=color, alpha=0.65, linewidth=0.65
        )
        if len(centerline) > 1:
            middle = centerline[len(centerline) // 2]
            ax.arrow(
                middle[0],
                middle[1],
                math.cos(float(middle[2])) * 1.5,
                math.sin(float(middle[2])) * 1.5,
                color=color,
                width=0.03,
                head_width=0.35,
                length_includes_head=True,
            )
    ax.plot(
        trajectory_xy[:, 0],
        trajectory_xy[:, 1],
        "k--",
        linewidth=1.5,
        label="ego trajectory",
    )
    ax.set_title("Lane-constrained nuScenes HD Map")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    location = infer_location(
        args.scene_name, args.nuscenes_meta_root, args.version
    ) or SCENE_TO_LOCATION.get(args.scene_name)
    if location is None:
        raise KeyError(f"No map location configured for {args.scene_name}")
    trajectory = load_pose_xy(args.scene_dir)
    arrays, metadata = build_lane_map(
        load_map_json(args.map_root, location),
        trajectory,
        args.resolution,
        args.padding_m,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "lane_constrained_bev.npz",
        **arrays,
        location=location,
        scene_name=args.scene_name,
    )
    (args.output_dir / "lane_topology.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_preview(arrays, metadata, trajectory, args.output_dir / "lane_map_preview.png")
    print(
        json.dumps(
            {
                "scene_id": args.scene_name,
                "location": location,
                "diagnostics": metadata["diagnostics"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
