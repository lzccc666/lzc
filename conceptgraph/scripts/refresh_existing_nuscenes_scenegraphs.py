from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh existing nuScenes scene_graph.json files to the current format.")
    parser.add_argument("--scenes", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for scene in args.scenes:
        scene_dir = Path("/root/autodl-tmp/lizichong/dataset/nuscenes") / scene / "exp_streamlined_yolo_outdoor_objects_v1"
        response_dir = scene_dir / "cfslam_qwen_responses"
        scene_graph_path = scene_dir / "scene_graph.json"
        if not response_dir.exists() or not scene_graph_path.exists():
            continue
        responses_by_id = {}
        for resp_file in sorted(response_dir.glob("*.json")):
            rid = int(resp_file.stem)
            wrapped = json.load(open(resp_file, "r", encoding="utf-8"))
            if isinstance(wrapped, dict) and "response" in wrapped:
                responses_by_id[rid] = {
                    "response": wrapped["response"],
                    "captions": wrapped.get("captions", []),
                }
        old = json.load(open(scene_graph_path, "r", encoding="utf-8"))
        updated = []
        for item in old:
            rid = int(item["id"])
            wrapped = responses_by_id.get(rid, {})
            response = wrapped.get("response", {})
            raw_captions = wrapped.get("captions", [])
            if not raw_captions:
                raw_captions = [str(response.get("summary", item.get("caption", "")))]
            object_tag = str(response.get("object_tag", item.get("object_tag", "unknown")))
            attributes = extract_attributes(raw_captions, response.get("summary", item.get("caption", "")), object_tag)
            updated.append(
                {
                    **item,
                    "caption": build_caption(object_tag, raw_captions, attributes, response.get("summary", item.get("caption", ""))),
                    "raw_captions": raw_captions,
                    "attributes": attributes,
                    "possible_tags": response.get("possible_tags", item.get("possible_tags", [])),
                    "object_tag": object_tag,
                }
            )
        scene_graph_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_attributes(raw_captions: list[str], summary: str, object_tag: str) -> dict:
    text = " ".join([summary, object_tag] + [str(item) for item in raw_captions]).lower()
    colors = ["white", "black", "blue", "red", "gray", "grey", "silver", "green", "yellow", "orange"]
    brands = [
        "honda", "toyota", "nissan", "bmw", "ford", "chevrolet", "mazda", "mitsubishi", "subaru", "volvo",
        "mini", "rogue", "qashqai", "prius", "fit", "mirage", "orlando"
    ]
    body_types = ["suv", "sedan", "hatchback", "van", "truck", "bus", "motorcycle", "bike", "car"]
    return {
        "colors": sorted({c for c in colors if c in text}),
        "brands": sorted({b for b in brands if b in text}),
        "body_types": sorted({t for t in body_types if t in text}),
        "raw_text": text,
    }


def build_caption(object_tag: str, raw_captions: list[str], attributes: dict, fallback_summary: str) -> str:
    colors = attributes.get("colors", [])
    brands = attributes.get("brands", [])
    body_types = [item for item in attributes.get("body_types", []) if item != object_tag]
    parts = []
    if colors:
        parts.append("/".join(colors[:2]))
    if brands:
        parts.append("/".join(brands[:2]))
    if body_types:
        parts.append("/".join(body_types[:2]))
    if object_tag:
        parts.append(str(object_tag))
    if parts:
        return " ".join(parts)
    for caption in raw_captions:
        text = str(caption).strip()
        if text and "observed in nuscenes" not in text.lower():
            return text[:120]
    return str(fallback_summary or object_tag or "object")


if __name__ == "__main__":
    main()
