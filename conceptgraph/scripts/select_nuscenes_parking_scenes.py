from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PARKING_KEYWORDS = ("parking", "parked", "parking lot", "park ")
LOW_DYNAMIC_POSITIVE = ("parked", "parking", "barrier", "bicycle rack", "traffic sign")
HIGH_DYNAMIC_NEGATIVE = ("high speed", "busy intersection", "overtaking", "jaywalker", "many peds", "cross intersection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select parking-like nuScenes scenes for thesis experiments.")
    parser.add_argument("--metadata-root", type=Path, default=Path("/root/autodl-tmp/lizichong/dataset/nuscenes_trainval_raw"))
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--data-roots", nargs="*", type=Path, default=[
        Path("/root/autodl-tmp/lizichong/dataset/nuscenes_trainval_raw"),
        Path("/root/autodl-tmp/lizichong/dataset/nuscenes"),
    ])
    parser.add_argument("--num-scenes", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("/root/autodl-tmp/lizichong/outputs/experiment_summary/selected_parking_scenes.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = args.metadata_root / args.version
    scenes = load_json(meta / "scene.json")
    samples = {item["token"]: item for item in load_json(meta / "sample.json")}
    sample_data = load_json(meta / "sample_data.json")
    calibrated = {item["token"]: item for item in load_json(meta / "calibrated_sensor.json")}
    sensors = {item["token"]: item for item in load_json(meta / "sensor.json")}
    sample_channel_data = build_sample_channel_index(sample_data, calibrated, sensors)

    rows = []
    for scene in scenes:
        desc = str(scene.get("description", ""))
        if not any(keyword in desc.lower() for keyword in PARKING_KEYWORDS):
            continue
        total, available = count_available_keyframes(scene, samples, sample_channel_data, args.data_roots)
        parking_score = sum(desc.lower().count(keyword) for keyword in PARKING_KEYWORDS)
        low_dynamic_score = sum(desc.lower().count(keyword) for keyword in LOW_DYNAMIC_POSITIVE)
        dynamic_penalty = sum(desc.lower().count(keyword) for keyword in HIGH_DYNAMIC_NEGATIVE)
        rows.append(
            {
                "scene_name": scene["name"],
                "description": desc,
                "nbr_samples": int(scene.get("nbr_samples", total)),
                "available_cam_front_lidar_top_frames": int(available),
                "complete_for_local_conversion": bool(available >= min(total, 30)),
                "parking_score": int(parking_score),
                "low_dynamic_score": int(low_dynamic_score),
                "dynamic_penalty": int(dynamic_penalty),
                "selection_score": int(parking_score * 4 + low_dynamic_score * 2 - dynamic_penalty + min(available, 40)),
                "selection_reason": "parking/parked keyword with CAM_FRONT and LIDAR_TOP availability check",
            }
        )
    rows.sort(key=lambda item: (item["complete_for_local_conversion"], item["selection_score"], item["parking_score"]), reverse=True)
    selected = rows[: args.num_scenes]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(selected)} selected scenes to {args.out}")
    for row in selected:
        print(
            row["scene_name"],
            "frames=",
            row["available_cam_front_lidar_top_frames"],
            "complete=",
            row["complete_for_local_conversion"],
            row["description"],
        )


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_sample_channel_index(sample_data: list[dict[str, Any]], calibrated: dict[str, dict], sensors: dict[str, dict]) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    for item in sample_data:
        if not item.get("is_key_frame", False):
            continue
        calibrated_record = calibrated.get(item.get("calibrated_sensor_token", ""), {})
        sensor = sensors.get(calibrated_record.get("sensor_token", ""), {})
        channel = sensor.get("channel")
        if channel not in {"CAM_FRONT", "LIDAR_TOP"}:
            continue
        result.setdefault(item["sample_token"], {})[channel] = item
    return result


def count_available_keyframes(scene: dict, samples: dict[str, dict], sample_channel_data: dict[str, dict[str, dict]], data_roots: list[Path]) -> tuple[int, int]:
    total = 0
    available = 0
    token = scene.get("first_sample_token", "")
    while token:
        sample = samples[token]
        total += 1
        channels = sample_channel_data.get(token, {})
        if all(_file_exists(channels.get(channel), data_roots) for channel in ("CAM_FRONT", "LIDAR_TOP")):
            available += 1
        token = sample.get("next", "")
    return total, available


def _file_exists(sample_data_record: dict | None, data_roots: list[Path]) -> bool:
    if not sample_data_record:
        return False
    filename = sample_data_record.get("filename", "")
    return any((root / filename).exists() for root in data_roots)


if __name__ == "__main__":
    main()
