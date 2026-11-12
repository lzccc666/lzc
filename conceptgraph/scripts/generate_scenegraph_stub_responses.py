from __future__ import annotations

import argparse
import gzip
import json
import pickle
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lightweight scenegraph caption responses from mapped object classes.")
    parser.add_argument("--mapfile", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def normalize_tag(tag: str) -> str:
    tag = str(tag).strip().lower()
    if not tag:
        return "object"
    return tag


def main() -> None:
    args = parse_args()
    with gzip.open(args.mapfile, "rb") as f:
        payload = pickle.load(f)
    objects = payload["objects"] if isinstance(payload, dict) and "objects" in payload else payload

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for idx, obj in enumerate(objects):
        class_names = [normalize_tag(name) for name in obj.get("class_name", []) if str(name).strip()]
        if class_names:
            object_tag = Counter(class_names).most_common(1)[0][0]
        else:
            object_tag = "object"
        payload = {
            "id": idx,
            "captions": [f"{object_tag} detected in parking/outdoor scene"],
            "response": {
                "summary": f"{object_tag} detected in parking/outdoor scene",
                "possible_tags": [object_tag, "object"],
                "object_tag": object_tag,
            },
        }
        with open(args.out_dir / f"{idx}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
