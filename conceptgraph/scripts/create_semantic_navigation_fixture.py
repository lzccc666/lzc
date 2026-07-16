from __future__ import annotations

import argparse
import gzip
import json
import pickle as pkl
from pathlib import Path

import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class SimpleBBox:
    def __init__(self, center, extent):
        self.center = np.asarray(center, dtype=float)
        self.extent = np.asarray(extent, dtype=float)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    output_dir = args.output_dir
    map_dir = output_dir / "map"
    map_dir.mkdir(parents=True, exist_ok=True)

    objects = []
    fixture_objects = [
        {
            "id": 0,
            "bbox": SimpleBBox(center=[0.0, 0.0, 0.0], extent=[2.0, 2.0, 1.5]),
            "object_tag": "vehicle",
            "caption": "a parked vehicle",
            "possible_tags": ["vehicle", "car", "parked car"],
        },
        {
            "id": 1,
            "bbox": SimpleBBox(center=[8.0, 2.0, 0.0], extent=[3.0, 2.0, 1.5]),
            "object_tag": "parking area",
            "caption": "open parking area near vehicles",
            "possible_tags": ["parking area", "parking spot", "open area"],
        },
        {
            "id": 2,
            "bbox": SimpleBBox(center=[5.0, -4.0, 0.0], extent=[1.0, 1.0, 2.0]),
            "object_tag": "traffic cone",
            "caption": "orange traffic cone",
            "possible_tags": ["traffic cone", "cone"],
        },
    ]

    for item in fixture_objects:
        objects.append(
            {
                "bbox": {
                    "center": item["bbox"].center.tolist(),
                    "extent": item["bbox"].extent.tolist(),
                },
                "caption_dict": {
                    "id": item["id"],
                    "response": {
                        "summary": item["caption"],
                        "possible_tags": item["possible_tags"],
                        "object_tag": item["object_tag"],
                    },
                },
                "conf": [1.0, 0.9, 0.8],
            }
        )

    with gzip.open(map_dir / "scene_map_cfslam_pruned.pkl.gz", "wb") as f:
        pkl.dump(objects, f)

    scene_graph = []
    for item in fixture_objects:
        scene_graph.append(
            {
                "id": item["id"],
                "bbox_extent": item["bbox"].extent.tolist(),
                "bbox_center": item["bbox"].center.tolist(),
                "possible_tags": item["possible_tags"],
                "object_tag": item["object_tag"],
                "caption": item["caption"],
                "bbox": {
                    "center": item["bbox"].center.tolist(),
                    "extent": item["bbox"].extent.tolist(),
                },
            }
        )

    with open(output_dir / "scene_graph.json", "w") as f:
        json.dump(scene_graph, f, indent=2)

    with open(output_dir / "cfslam_scenegraph_edges.pkl", "wb") as f:
        pkl.dump([(0, 1, "a near b"), (2, 1, "a near b")], f)

    print(f"Wrote semantic navigation fixture to {output_dir}")


if __name__ == "__main__":
    main()
