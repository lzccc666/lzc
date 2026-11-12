from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_vehicle_classes() -> list[str]:
    classes_path = Path(__file__).resolve().parents[1] / "vehicle_classes.txt"
    return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def infer_object_tag(resp: dict, captions: list[str]) -> str:
    joined = " ".join(captions).lower()
    rules = [
        ("motorcycle", ["motorcycle", "motor bike", "摩托车"]),
        ("bicycle", ["bicycle", "bike", "自行车"]),
        ("pedestrian", ["pedestrian", "person", "human", "行人"]),
        ("traffic sign", ["traffic sign", "road sign", "bus stop sign", "sign", "交通标志"]),
        ("traffic light", ["traffic light", "signal light", "红绿灯"]),
        ("truck", ["truck", "lorry", "货车", "卡车"]),
        ("bus", ["bus", "巴士", "公交"]),
        ("car", ["car", "sedan", "suv", "van", "vehicle", "汽车", "车辆"]),
        ("road", ["road", "street", "道路"]),
        ("sidewalk", ["sidewalk", "人行道"]),
        ("building", ["building", "house", "建筑"]),
        ("tree", ["tree", "树"]),
        ("fence", ["fence", "围栏"]),
        ("pole", ["pole", "杆"]),
    ]
    for tag, words in rules:
        if any(word in joined for word in words):
            return tag
    if str(resp.get("object_tag", "")).strip().lower() in load_vehicle_classes():
        return str(resp["object_tag"]).strip().lower()
    return "invalid"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-dir", required=True)
    parser.add_argument("--captions-json", required=True)
    args = parser.parse_args()

    responses_dir = Path(args.responses_dir)
    captions = json.load(open(args.captions_json, "r", encoding="utf-8"))
    vehicle_classes = load_vehicle_classes()

    for item in captions:
        idx = int(item["id"])
        resp_path = responses_dir / f"{idx}.json"
        if not resp_path.exists():
            continue
        d = json.load(open(resp_path, "r", encoding="utf-8"))
        captions_list = item.get("captions", [])
        d["response"] = {
            "summary": d.get("response", {}).get("summary", ""),
            "possible_tags": vehicle_classes,
            "object_tag": infer_object_tag(d.get("response", {}), captions_list),
        }
        resp_path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
