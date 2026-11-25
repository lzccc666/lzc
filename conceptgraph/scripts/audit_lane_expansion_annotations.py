"""Official geometry coverage and conservative node/instance association audit."""

from __future__ import annotations

import json
from collections import Counter, deque

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from conceptgraph.nuscenes.benchmark_ground_truth import associate_nodes
from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT, PROJECT_ROOT
from conceptgraph.scripts.run_incremental_semantic_navigation import write_json


def map_geometry(location):
    path = (
        PROJECT_ROOT / "dataset/nuscenes_map_expansion/expansion" / f"{location}.json"
    )
    data = json.loads(path.read_text())
    vertices = {v["token"]: (v["x"], v["y"]) for v in data["node"]}
    polygons = {}
    for p in data["polygon"]:
        exterior = [vertices[t] for t in p["exterior_node_tokens"]]
        holes = [[vertices[t] for t in h["node_tokens"]] for h in p.get("holes", [])]
        polygon = Polygon(exterior, holes)
        if polygon.is_valid:
            polygons[p["token"]] = polygon
    lanes = {
        r["token"]: polygons[r["polygon_token"]]
        for layer in ("lane", "lane_connector")
        for r in data.get(layer, [])
        if r["polygon_token"] in polygons
    }
    allowed = [
        polygons[t]
        for r in data.get("drivable_area", [])
        for t in r["polygon_tokens"]
        if t in polygons
    ]
    allowed += [
        polygons[r["polygon_token"]]
        for layer in ("carpark_area", "road_segment", "lane", "lane_connector")
        for r in data.get(layer, [])
        if r.get("polygon_token") in polygons
    ]
    return lanes, unary_union(allowed), data["connectivity"]


def nearby_lanes(point, lanes, limit):
    return {
        token for token, polygon in lanes.items() if polygon.distance(point) <= limit
    }


def reachable_tokens(starts, connectivity):
    visited, queue = set(starts), deque(starts)
    while queue:
        for token in connectivity.get(queue.popleft(), {}).get("outgoing", []):
            if token not in visited:
                visited.add(token)
                queue.append(token)
    return visited


def main():
    root = DEFAULT_ROOT
    output = root / "annotation_geometry_audit.json"
    if output.exists():
        raise FileExistsError(output)
    selection = json.loads((root / "scene_selection_manifest.json").read_text())[
        "scenes"
    ]
    frames = {
        r["scene"]: r
        for line in (root / "ground_truth_frames.jsonl").open()
        if (r := json.loads(line))["frame_idx"] == 0
    }
    queries = [json.loads(line) for line in (root / "benchmark_queries.jsonl").open()]
    maps, scenes, query_audits = {}, [], []
    for row in selection:
        scene, location = row["scene"], row["strata"]["location"]
        if location not in maps:
            maps[location] = map_geometry(location)
        lanes, allowed, connectivity = maps[location]
        pose = np.loadtxt(root / "converted" / scene / "poses.txt").reshape(-1, 4, 4)[0]
        start = Point(pose[:2, 3])
        starts = nearby_lanes(start, lanes, 8.0)
        reachable = reachable_tokens(starts, connectivity)
        records = []
        for obj in frames[scene]["objects"]:
            point = Point(obj["center"][:2])
            near = nearby_lanes(point, lanes, 8.0)
            records.append(
                {
                    "instance_token": obj["instance_token"],
                    "category": obj["category"],
                    "center": obj["center"],
                    "lane_polygon_distance_m": min(
                        (p.distance(point) for p in lanes.values()), default=None
                    ),
                    "allowed_area_distance_m": allowed.distance(point),
                    "allowed_area_boundary_distance_m": allowed.boundary.distance(
                        point
                    ),
                    "candidate_official_lane_tokens_within_8m": sorted(near),
                    "directed_connection_possible_optimistic": bool(near & reachable)
                    if starts
                    else False,
                }
            )
        nodes = json.loads((root / "frame0_inputs" / scene / "nodes.json").read_text())
        association = associate_nodes(nodes, frames[scene]["objects"])
        write_json(
            root / "annotations" / scene / "association_frame0.json", association
        )
        scenes.append(
            {
                "scene": scene,
                "role": row["role"],
                "official_visible_instances": len(records),
                "system_frame0_nodes": len(nodes),
                "association": association,
                "start_candidate_lane_count": len(starts),
                "objects": records,
            }
        )
        indexed = {r["instance_token"]: r for r in records}
        for query in (q for q in queries if q["scene"] == scene):
            targets = [indexed[t] for t in query["accepted_instance_tokens"]]
            outside = bool(targets) and all(
                r["lane_polygon_distance_m"] is None
                or r["lane_polygon_distance_m"] > 8.0
                for r in targets
            )
            disconnected = (
                bool(targets)
                and bool(starts)
                and all(
                    not r["directed_connection_possible_optimistic"] for r in targets
                )
            )
            query_audits.append(
                {
                    "query_id": query["query_id"],
                    "scene": scene,
                    "target_count": len(targets),
                    "map_boundary_or_projection_difficulty": any(
                        r["allowed_area_distance_m"] > 0
                        or r["allowed_area_boundary_distance_m"] < 2.0
                        or r["lane_polygon_distance_m"] > 8.0
                        for r in targets
                    ),
                    "center_projection_outside_8m": outside,
                    "center_topology_disconnected": disconnected,
                    "strict_lane_unreachable_proven": bool(targets) and not starts,
                    "proof": "start has no official lane polygon within frozen 8m bound; even optimistic membership is empty"
                    if targets and not starts
                    else None,
                    "ordinary_mode_reachability": "unknown; no planner-failure-derived truth",
                }
            )
    counts = Counter()
    for s in scenes:
        counts.update(
            official_visible_instances=s["official_visible_instances"],
            system_frame0_nodes=s["system_frame0_nodes"],
            matched_nodes=len(s["association"]["matches"]),
            ambiguous_nodes=len(s["association"]["ambiguous_nodes"]),
            unmatched_nodes=len(s["association"]["unmatched_nodes"]),
        )
    write_json(
        output,
        {
            "source": "independent official 3D annotations + full HD Map polygon geometry and directed connectivity BFS; no tested A* or semantic scores",
            "association_rule": {
                "category_equal": True,
                "maximum_xy_distance_m": 3.0,
                "ambiguity_margin_m": 0.5,
                "one_to_one": True,
            },
            "counts": dict(counts),
            "scenes": scenes,
            "queries": query_audits,
            "projection_difficulty_query_count": sum(
                q["map_boundary_or_projection_difficulty"] for q in query_audits
            ),
            "independently_proven_strict_unreachable_queries": sum(
                q["strict_lane_unreachable_proven"] for q in query_audits
            ),
            "human_review_count": 0,
            "color_fine_attributes": "unknown; excluded from field accuracy, never represented as human labels",
            "limitations": [
                "Association omissions remain retrieval failures, not dropped positives.",
                "Sparse lidar/masks may merge nearby objects; geometry association does not certify fine attributes.",
                "Topology proof is conditional on the frozen center-projection definition; positive reachability remains unknown.",
            ],
        },
    )
    print(dict(counts), flush=True)


if __name__ == "__main__":
    main()
