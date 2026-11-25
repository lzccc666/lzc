from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image

from conceptgraph.scenegraph.relations import (
    build_relations,
    save_legacy_edge_pickle,
    save_relation_result,
    save_structured_edges,
)


CHINESE_FONT = FontProperties()
COLOR_RGB = {
    "white": np.asarray([235, 235, 235], dtype=float),
    "black": np.asarray([25, 25, 25], dtype=float),
    "blue": np.asarray([55, 100, 190], dtype=float),
    "red": np.asarray([190, 55, 45], dtype=float),
    "gray": np.asarray([125, 125, 125], dtype=float),
    "silver": np.asarray([185, 185, 175], dtype=float),
    "green": np.asarray([60, 145, 75], dtype=float),
    "yellow": np.asarray([210, 185, 55], dtype=float),
    "orange": np.asarray([220, 125, 45], dtype=float),
}
POTENTIAL_DYNAMIC_TAGS = {"car", "truck", "bus", "motorcycle", "bicycle", "pedestrian", "vehicle", "van"}
STATIC_TAGS = {"traffic sign", "traffic light", "road", "sidewalk", "building", "tree", "fence", "pole", "parking area"}


def configure_chinese_font() -> None:
    font_path = Path("/root/autodl-tmp/lizichong/checkpoint/fonts/NotoSansCJKsc-Regular.otf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        global CHINESE_FONT
        CHINESE_FONT = FontProperties(fname=str(font_path))
        plt.rcParams["font.family"] = [CHINESE_FONT.get_name()]
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a lightweight scene_graph.json and strong geometric edges from a ConceptGraphs map.")
    parser.add_argument("--mapfile", type=Path, required=True)
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-edge-distance-m", type=float, default=8.0)
    parser.add_argument("--min-relation-confidence", type=float, default=0.35)
    parser.add_argument("--max-edges-per-node", type=int, default=4)
    parser.add_argument("--write-debug-crops", action="store_true")
    parser.add_argument("--max-detections-per-object", type=int, default=6)
    parser.add_argument("--coordinate-frame", choices=["world", "ego_heading", "local_trajectory"], default="world")
    parser.add_argument("--coordinate-frame-fallback", choices=["world", "ego_heading", "local_trajectory"], default="world")
    parser.add_argument("--ego-heading", default=None, help="Optional heading vector hx,hy")
    parser.add_argument("--trajectory-poses", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    configure_chinese_font()
    args = parse_args()
    objects = load_objects(args.mapfile)
    responses = load_responses(args.responses_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.responses_dir.mkdir(parents=True, exist_ok=True)
    captions_dir = args.out_dir / "cfslam_captions_qwen"
    debug_dir = args.out_dir / "cfslam_captions_qwen_debug"
    captions_dir.mkdir(parents=True, exist_ok=True)
    if args.write_debug_crops:
        debug_dir.mkdir(parents=True, exist_ok=True)
    pruned_objects = []
    scene_graph = []
    old_to_new = {}
    for old_idx, obj in enumerate(objects):
        response = responses.get(old_idx, default_response(obj, old_idx))
        tag = str(response.get("response", {}).get("object_tag", "")).strip().lower()
        if tag in {"", "invalid", "fail"}:
            continue
        inferred_colors = infer_object_colors(obj)
        if inferred_colors:
            response = enrich_response_with_colors(response, inferred_colors)
        write_response_files(args.responses_dir, captions_dir, old_idx, response)
        if args.write_debug_crops:
            write_debug_crop(debug_dir, old_idx, obj, response, args.max_detections_per_object)
        old_to_new[old_idx] = len(pruned_objects)
        obj = dict(obj)
        obj["caption_dict"] = response
        motion_state, is_potential_dynamic, is_short_term = infer_motion_fields(obj, tag)
        obj["motion_state"] = motion_state
        obj["is_potential_dynamic"] = is_potential_dynamic
        obj["is_short_term"] = is_short_term
        pruned_objects.append(obj)
        center, extent = center_extent(obj)
        raw_captions = response.get("captions", [])
        attributes = build_attributes(raw_captions, inferred_colors)
        scene_graph.append(
            {
                "id": len(scene_graph),
                "bbox_extent": np.round(extent, 1).tolist(),
                "bbox_center": np.round(center, 1).tolist(),
                "possible_tags": response.get("response", {}).get("possible_tags", [tag]),
                "object_tag": tag,
                "object_tag_zh": tag_to_cn(tag),
                "caption": build_caption(tag, raw_captions),
                "raw_captions": raw_captions,
                "attributes": attributes,
                "motion_state": motion_state,
                "is_potential_dynamic": is_potential_dynamic,
                "velocity": float(obj.get("velocity", 0.0) or 0.0),
                "is_short_term": is_short_term,
                "last_seen_frame": int(obj.get("last_seen_frame", -1)),
                "center_history": obj.get("center_history", []),
                "frame_history": obj.get("frame_history", []),
                "metadata": {
                    "motion_state": motion_state,
                    "is_potential_dynamic": is_potential_dynamic,
                    "velocity": float(obj.get("velocity", 0.0) or 0.0),
                    "is_short_term": is_short_term,
                    "last_seen_frame": int(obj.get("last_seen_frame", -1)),
                },
            }
        )

    heading = None
    if args.ego_heading:
        heading = np.asarray([float(value) for value in args.ego_heading.split(",")], dtype=float)
    trajectory = load_trajectory_xy(args.trajectory_poses)
    relation_config = {
        "relation_distance_thresholds_m": {
            "near": args.max_edge_distance_m,
            "directional": args.max_edge_distance_m,
            "left_of": args.max_edge_distance_m,
            "right_of": args.max_edge_distance_m,
            "in_front_of": args.max_edge_distance_m,
            "behind": args.max_edge_distance_m,
        },
        "coordinate_frame": args.coordinate_frame,
        "coordinate_frame_fallback": args.coordinate_frame_fallback,
        "minimum_confidence": args.min_relation_confidence,
        "max_out_degree": args.max_edges_per_node,
        "max_in_degree": args.max_edges_per_node,
    }
    relation_result = build_relations(
        scene_graph,
        relation_config,
        ego_heading=heading,
        trajectory=trajectory,
        ego_position=trajectory[-1] if trajectory is not None else None,
    )
    if not relation_result.success:
        raise RuntimeError(f"{relation_result.failure_code}: {relation_result.failure_reason}")
    edges = [(edge.source, edge.target, edge.relation) for edge in relation_result.edges]
    edge_metadata = [edge.to_dict() for edge in relation_result.edges]

    map_dir = args.out_dir / "map"
    map_dir.mkdir(exist_ok=True)
    with gzip.open(map_dir / "scene_map_cfslam_pruned.pkl.gz", "wb") as f:
        pickle.dump(serialize_objects(pruned_objects), f)
    (args.out_dir / "scene_graph.json").write_text(json.dumps(scene_graph, indent=2, ensure_ascii=False), encoding="utf-8")
    save_legacy_edge_pickle(relation_result.edges, args.out_dir / "cfslam_scenegraph_edges.pkl")
    save_structured_edges(relation_result.edges, args.out_dir / "cfslam_scenegraph_edges_structured.pkl")
    save_structured_edges(relation_result.edges, args.out_dir / "cfslam_scenegraph_edges_structured.json")
    save_relation_result(relation_result, args.out_dir / "scenegraph_relation_build_result.json")
    (args.out_dir / "cfslam_scenegraph_edge_metadata.json").write_text(json.dumps(edge_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(scene_graph)} nodes and {len(edges)} edges to {args.out_dir}")


def load_trajectory_xy(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    poses = np.loadtxt(path)
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    centers = []
    for row in poses:
        if row.size >= 16:
            centers.append(row.reshape(4, 4)[:2, 3])
        elif row.size >= 2:
            centers.append(row[:2])
    return None if not centers else np.asarray(centers, dtype=float)


def infer_relation(source: np.ndarray, target: np.ndarray, distance_m: float, max_distance_m: float) -> tuple[str, float]:
    if distance_m <= 1e-6:
        return "near", 1.0
    delta = np.asarray(target[:2] - source[:2], dtype=float)
    abs_dx, abs_dy = abs(float(delta[0])), abs(float(delta[1]))
    confidence = max(0.0, 1.0 - distance_m / max(max_distance_m, 1e-6))
    if max(abs_dx, abs_dy) > 1e-6 and abs(abs_dx - abs_dy) / max(abs_dx, abs_dy) > 0.35:
        if abs_dx > abs_dy:
            relation = "right_of" if delta[0] > 0 else "left_of"
        else:
            relation = "in_front_of" if delta[1] > 0 else "behind"
        confidence = max(confidence, 0.55)
    else:
        relation = "near"
        confidence = max(confidence, 0.5)
    return relation, float(min(confidence, 1.0))


def select_edges_per_node(edge_candidates: list[tuple[int, int, str, dict]], max_edges_per_node: int) -> tuple[list[tuple[int, int, str]], list[dict]]:
    if max_edges_per_node <= 0:
        selected = edge_candidates
    else:
        selected = []
        degree: dict[int, int] = {}
        ranked = sorted(edge_candidates, key=lambda item: (-float(item[3].get("confidence", 0.0)), float(item[3].get("distance_m", 0.0))))
        for item in ranked:
            src, dst, _relation, _meta = item
            if degree.get(src, 0) >= max_edges_per_node or degree.get(dst, 0) >= max_edges_per_node:
                continue
            selected.append(item)
            degree[src] = degree.get(src, 0) + 1
            degree[dst] = degree.get(dst, 0) + 1
    selected = sorted(selected, key=lambda item: (item[0], item[1]))
    return [(src, dst, relation) for src, dst, relation, _meta in selected], [meta for _src, _dst, _relation, meta in selected]


def load_objects(path: Path) -> list[dict]:
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "objects" in payload:
        return payload["objects"]
    return payload


def load_responses(path: Path) -> dict[int, dict]:
    result = {}
    for resp_path in path.glob("*.json"):
        data = json.loads(resp_path.read_text(encoding="utf-8"))
        result[int(data["id"])] = data
    return result


def default_response(obj: dict, idx: int) -> dict:
    names = obj.get("class_name", []) or ["object"]
    tag = str(names[0]).lower()
    return {"id": idx, "captions": [tag], "response": {"summary": tag, "possible_tags": [tag], "object_tag": tag}}


def infer_motion_fields(obj: dict, tag: str, velocity_threshold: float = 0.5) -> tuple[str, bool, bool]:
    tag = str(tag).lower().strip()
    raw_state = str(obj.get("motion_state", "")).lower().strip()
    velocity = float(obj.get("velocity", 0.0) or 0.0)
    is_potential_dynamic = bool(obj.get("is_potential_dynamic", tag in POTENTIAL_DYNAMIC_TAGS))
    if tag in STATIC_TAGS:
        return "static", False, False
    if raw_state in {"static", "dynamic"}:
        return raw_state, is_potential_dynamic, bool(obj.get("is_short_term", raw_state == "dynamic"))
    if is_potential_dynamic and velocity > velocity_threshold:
        return "dynamic", True, True
    # nuScenes parking/roadside objects without observed motion are treated as
    # static map landmarks, especially parked vehicles.
    return "static", is_potential_dynamic, False


def enrich_response_with_colors(response: dict, colors: list[str]) -> dict:
    response = json.loads(json.dumps(response, ensure_ascii=False))
    captions = [str(item) for item in response.get("captions", [])]
    tag = str(response.get("response", {}).get("object_tag", "object"))
    color_phrase = "/".join(colors[:2])
    if captions:
        captions[0] = f"{color_phrase} {captions[0]}"
    else:
        captions = [f"{color_phrase} {tag}"]
    response["captions"] = captions
    summary = str(response.get("response", {}).get("summary", tag))
    if color_phrase not in summary:
        response.setdefault("response", {})["summary"] = f"{color_phrase} {summary}"
    return response


def build_attributes(raw_captions: list[str], colors: list[str]) -> dict:
    raw_text = " ".join(str(item) for item in raw_captions).lower()
    return {
        "colors": colors,
        "brands": [],
        "body_types": [],
        "raw_text": raw_text,
    }


def infer_object_colors(obj: dict, max_samples: int = 4) -> list[str]:
    color_paths = obj.get("color_path", [])
    xyxys = obj.get("xyxy", [])
    masks = obj.get("mask", [])
    confs = np.asarray(obj.get("conf", []), dtype=float)
    if len(color_paths) == 0 or len(xyxys) == 0:
        return []
    order = np.argsort(confs)[::-1] if confs.size else np.arange(len(color_paths))
    rgb_values = []
    for det_idx in [int(i) for i in order[:max_samples]]:
        if det_idx >= len(color_paths) or det_idx >= len(xyxys):
            continue
        image_path = Path(color_paths[det_idx])
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        xyxy = np.asarray(xyxys[det_idx], dtype=float)
        crop, mask_crop = crop_image_and_mask_for_debug(image, masks[det_idx] if det_idx < len(masks) else None, xyxy, padding=0)
        if crop.size == 0:
            continue
        pixels = crop.reshape(-1, 3)
        if mask_crop is not None and mask_crop.shape[:2] == crop.shape[:2] and mask_crop.any():
            pixels = crop[mask_crop.astype(bool)]
        if len(pixels) == 0:
            continue
        rgb_values.append(np.median(pixels.astype(float), axis=0))
    if not rgb_values:
        return []
    rgb = np.median(np.vstack(rgb_values), axis=0)
    brightness = float(np.mean(rgb))
    spread = float(np.max(rgb) - np.min(rgb))
    if brightness < 55:
        return ["black"]
    if brightness > 210 and spread < 35:
        return ["white"]
    if spread < 28:
        return ["gray"]
    distances = {name: float(np.linalg.norm(rgb - value)) for name, value in COLOR_RGB.items()}
    ranked = sorted(distances.items(), key=lambda item: item[1])
    colors = [ranked[0][0]]
    if ranked[1][1] < ranked[0][1] * 1.15:
        colors.append(ranked[1][0])
    return colors[:2]


def write_response_files(responses_dir: Path, captions_dir: Path, idx: int, response: dict) -> None:
    payload = {
        "id": int(response.get("id", idx)),
        "captions": response.get("captions", []),
        "response": response.get("response", {}),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    (responses_dir / f"{idx}.json").write_text(text, encoding="utf-8")
    (captions_dir / f"{idx}.json").write_text(text, encoding="utf-8")


def write_debug_crop(debug_dir: Path, idx: int, obj: dict, response: dict, max_detections: int) -> None:
    color_paths = obj.get("color_path", [])
    xyxys = obj.get("xyxy", [])
    masks = obj.get("mask", [])
    confs = np.asarray(obj.get("conf", []), dtype=float)
    if len(color_paths) == 0 or len(xyxys) == 0:
        return
    order = np.argsort(confs)[::-1] if confs.size else np.arange(len(color_paths))
    order = [int(i) for i in order[:max_detections] if i < len(color_paths) and i < len(xyxys)]
    if not order:
        return
    cols = min(3, len(order))
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.3 * rows))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax in axes_arr:
        ax.axis("off")
    tag = response.get("response", {}).get("object_tag", "object")
    captions = response.get("captions", [])
    for plot_idx, det_idx in enumerate(order):
        ax = axes_arr[plot_idx]
        image_path = Path(color_paths[det_idx])
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        xyxy = np.asarray(xyxys[det_idx], dtype=float)
        crop, mask_crop = crop_image_and_mask_for_debug(image, masks[det_idx] if det_idx < len(masks) else None, xyxy, padding=10)
        ax.imshow(crop)
        if mask_crop is not None:
            green = np.zeros((*mask_crop.shape, 3), dtype=np.uint8)
            green[mask_crop.astype(bool)] = [34, 197, 94]
            ax.imshow(green, alpha=0.22)
        conf = float(confs[det_idx]) if det_idx < confs.size else 0.0
        caption = str(captions[plot_idx] if plot_idx < len(captions) else tag)
        ax.set_title(f"{tag}  置信度:{conf:.2f}\n{caption[:36]}", fontsize=9, fontproperties=CHINESE_FONT)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(debug_dir / f"{idx}.png", dpi=160)
    plt.close(fig)


def crop_image_and_mask_for_debug(image: Image.Image, mask, xyxy: np.ndarray, padding: int) -> tuple[np.ndarray, np.ndarray | None]:
    width, height = image.size
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy[:4]]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    crop = np.asarray(image.crop((x1, y1, x2, y2)))
    if mask is None:
        return crop, None
    mask_np = np.asarray(mask).astype(bool)
    if mask_np.shape[:2] != (height, width):
        return crop, None
    return crop, mask_np[y1:y2, x1:x2]


def center_extent(obj: dict) -> tuple[np.ndarray, np.ndarray]:
    bbox = obj.get("bbox")
    if isinstance(bbox, dict):
        return np.asarray(bbox["center"], dtype=float), np.asarray(bbox["extent"], dtype=float)
    if bbox is not None:
        return np.asarray(bbox.center, dtype=float), np.asarray(bbox.extent, dtype=float)
    bbox_np = obj.get("bbox_np")
    if bbox_np is not None:
        points = np.asarray(bbox_np, dtype=float)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        return (mins + maxs) / 2.0, maxs - mins
    raise KeyError("Object has neither bbox nor bbox_np")


def build_caption(tag: str, captions: list[str]) -> str:
    for caption in captions:
        text = str(caption).strip()
        if text:
            return text[:120]
    return tag


def tag_to_cn(tag: str) -> str:
    mapping = {
        "car": "小汽车",
        "truck": "卡车",
        "bus": "公交车",
        "motorcycle": "摩托车",
        "bicycle": "自行车",
        "pedestrian": "行人",
        "traffic sign": "交通标志",
        "traffic light": "交通信号灯",
        "road": "道路",
        "sidewalk": "人行道",
        "building": "建筑",
        "tree": "树木",
        "fence": "围栏",
        "pole": "杆状物",
        "vehicle": "车辆",
        "object": "物体",
    }
    return mapping.get(str(tag).lower().strip(), str(tag))


def serialize_objects(objects: list[dict]):
    try:
        from conceptgraph.slam.slam_classes import MapObjectList

        return MapObjectList(objects).to_serializable()
    except Exception:
        return objects


if __name__ == "__main__":
    main()
