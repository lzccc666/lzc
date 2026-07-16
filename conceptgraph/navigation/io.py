from __future__ import annotations

import gzip
import json
import pickle as pkl
from pathlib import Path

import numpy as np

from conceptgraph.navigation.types import SceneGraphEdge, SceneGraphNode


def load_scene_map(scene_map_path: str | Path):
    with gzip.open(scene_map_path, "rb") as f:
        payload = pkl.load(f)
    if isinstance(payload, list):
        return payload
    from conceptgraph.slam.slam_classes import MapObjectList

    scene_map = MapObjectList()
    scene_map.load_serializable(payload)
    return scene_map


def load_scene_graph_nodes(
    scene_graph_json_path: str | Path,
    annotated_scenegraph_json_path: str | Path | None = None,
) -> dict[int, SceneGraphNode]:
    with open(scene_graph_json_path, "r") as f:
        scene_graph_json = json.load(f)

    annotations_by_id: dict[int, dict] = {}
    if annotated_scenegraph_json_path is not None and Path(annotated_scenegraph_json_path).exists():
        with open(annotated_scenegraph_json_path, "r") as f:
            annotations = json.load(f)
        annotations_by_id = {int(item["id"]): item for item in annotations if "id" in item}

    nodes: dict[int, SceneGraphNode] = {}
    for item in scene_graph_json:
        node_id = int(item["id"])
        metadata = dict(item)
        metadata["annotation"] = annotations_by_id.get(node_id, {})
        bbox_center = item.get("bbox_center")
        bbox_extent = item.get("bbox_extent")
        if bbox_center is None and "bbox" in item:
            bbox_center = getattr(item["bbox"], "center", [0.0, 0.0, 0.0])
        if bbox_extent is None and "bbox" in item:
            bbox_extent = getattr(item["bbox"], "extent", [0.0, 0.0, 0.0])
        nodes[node_id] = SceneGraphNode(
            node_id=node_id,
            object_tag=str(item.get("object_tag", "")),
            possible_tags=[str(tag) for tag in item.get("possible_tags", [])],
            caption=str(item.get("caption", "")),
            bbox_center=np.asarray(bbox_center if bbox_center is not None else [0.0, 0.0, 0.0], dtype=float),
            bbox_extent=np.asarray(bbox_extent if bbox_extent is not None else [0.0, 0.0, 0.0], dtype=float),
            motion_state=str(item.get("motion_state", metadata.get("metadata", {}).get("motion_state", "unknown"))),
            is_potential_dynamic=bool(item.get("is_potential_dynamic", metadata.get("metadata", {}).get("is_potential_dynamic", False))),
            velocity=float(item.get("velocity", metadata.get("metadata", {}).get("velocity", 0.0)) or 0.0),
            is_short_term=bool(item.get("is_short_term", metadata.get("metadata", {}).get("is_short_term", False))),
            metadata=metadata,
        )
    return nodes


def load_scene_graph_edges(edge_pkl_path: str | Path | None) -> list[SceneGraphEdge]:
    if edge_pkl_path is None or not Path(edge_pkl_path).exists():
        return []
    with open(edge_pkl_path, "rb") as f:
        edge_tuples = pkl.load(f)
    return [
        SceneGraphEdge(source=int(src), target=int(dst), relation=str(rel))
        for src, dst, rel in edge_tuples
    ]
