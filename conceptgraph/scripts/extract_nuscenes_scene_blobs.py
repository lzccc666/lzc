from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CAM_FRONT and LIDAR_TOP blobs for selected nuScenes scenes from trainval tgz files.")
    parser.add_argument("--metadata-root", type=Path, default=Path("/root/autodl-tmp/lizichong/dataset/nuscenes_trainval_raw"))
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--blob-dir", type=Path, default=Path("/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval"))
    parser.add_argument("--output-root", type=Path, default=Path("/root/autodl-tmp/lizichong/dataset/nuscenes_trainval_raw"))
    parser.add_argument("--scenes", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = args.metadata_root / args.version
    scenes = {item["name"]: item for item in load_json(meta / "scene.json")}
    samples = {item["token"]: item for item in load_json(meta / "sample.json")}
    sample_data = load_json(meta / "sample_data.json")
    calibrated = {item["token"]: item for item in load_json(meta / "calibrated_sensor.json")}
    sensors = {item["token"]: item for item in load_json(meta / "sensor.json")}
    by_sample = build_sample_channel_index(sample_data, calibrated, sensors)

    needed = []
    for scene_name in args.scenes:
        scene = scenes[scene_name]
        token = scene["first_sample_token"]
        while token:
            sample = samples[token]
            channels = by_sample.get(token, {})
            for channel in ("CAM_FRONT", "LIDAR_TOP"):
                record = channels.get(channel)
                if record is not None:
                    needed.append(record["filename"])
            token = sample.get("next", "")

    needed = sorted(set(needed))
    missing = [name for name in needed if not (args.output_root / name).exists()]
    if not missing:
        print(f"All {len(needed)} files already exist under {args.output_root}")
        return

    unresolved = set(missing)
    tar_paths = sorted(args.blob_dir.glob("v1.0-trainval*_blobs.tgz"))
    for tar_path in tar_paths:
        if not unresolved:
            break
        members = find_members_with_tar(tar_path, unresolved)
        if not members:
            continue
        print(f"Extracting {len(members)} files from {tar_path.name}")
        cmd = ["tar", "-xzf", str(tar_path), "-C", str(args.output_root), *members]
        subprocess.run(cmd, check=True)
        unresolved.difference_update(members)
    if unresolved:
        raise FileNotFoundError(f"{len(unresolved)} scene blob files not found in trainval tarballs. First missing: {sorted(unresolved)[:5]}")
    print(f"Extracted {len(missing)} missing files for scenes: {', '.join(args.scenes)}")


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
        if channel in {"CAM_FRONT", "LIDAR_TOP"}:
            result.setdefault(item["sample_token"], {})[channel] = item
    return result


def find_members_with_tar(tar_path: Path, missing: set[str]) -> list[str]:
    print(f"Scanning {tar_path.name} ({len(missing)} unresolved)")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        manifest_path = Path(handle.name)
        for name in sorted(missing):
            handle.write(name + "\n")
    try:
        list_cmd = ["tar", "-tzf", str(tar_path)]
        grep_cmd = ["grep", "-Fxf", str(manifest_path)]
        tar_proc = subprocess.Popen(list_cmd, stdout=subprocess.PIPE, text=True)
        grep_proc = subprocess.run(grep_cmd, stdin=tar_proc.stdout, text=True, capture_output=True)
        if tar_proc.stdout is not None:
            tar_proc.stdout.close()
        tar_proc.wait()
        if grep_proc.returncode not in {0, 1}:
            raise RuntimeError(grep_proc.stderr)
        members = [line.strip() for line in grep_proc.stdout.splitlines() if line.strip()]
        return members
    finally:
        manifest_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
