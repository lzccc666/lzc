"""冻结开发集前三视角小试输入，仅按需读取共享归档，不接触方法分数。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time

import imageio.v2 as imageio
import numpy as np

from conceptgraph.nuscenes.expansion_benchmark import iter_records
from conceptgraph.project_paths import PROJECT_ROOT, experiment_directory
from conceptgraph.scripts.preprocess_nuscenes_to_minimal import build_transform, inverse_transform

CHANNELS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def projection(points, transform, intrinsic, height, width):
    """与既有nearest_depth_image同门限/round/zbuffer，同时保留原bin点索引。"""
    xyz = points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    ids = np.flatnonzero(np.isfinite(xyz).all(1) & (xyz[:, 2] > 1) & (xyz[:, 2] < 65))
    xyz = xyz[ids]
    uvz = xyz @ np.asarray(intrinsic).T
    uv = np.rint(uvz[:, :2] / uvz[:, 2, None]).astype(np.int32)
    keep = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv, xyz, ids = uv[keep], xyz[keep], ids[keep]
    linear = uv[:, 1] * width + uv[:, 0]
    order = np.lexsort((ids, xyz[:, 2], linear))
    _, first = np.unique(linear[order], return_index=True)
    take = order[first]
    uv, z, ids = uv[take], xyz[take, 2], ids[take]
    depth = np.zeros((height, width), dtype=np.uint16)
    depth[uv[:, 1], uv[:, 0]] = np.rint(z * 1000).astype(np.uint16)
    return depth, dict(pixel_y=uv[:, 1], pixel_x=uv[:, 0], point_index=ids.astype(np.int32), depth_m=z.astype(np.float32))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default="multiview_20261123_frontend_v1")
    p.add_argument("--wall-s", type=int, default=1800)
    args = p.parse_args()
    begun = time.monotonic()
    root = PROJECT_ROOT
    processed = root / "dataset/processed/multiview_20261123_pilot_v1"
    raw = root / "dataset/raw/multiview_20261123_pilot_v1"
    processed.mkdir(parents=True, exist_ok=False)
    raw.mkdir(parents=True, exist_ok=False)
    run = experiment_directory("nuscenes", "validation", args.run_id, create=True)
    report = root / "outputs/reports/multiview_semantic_tasks_20261123/sensors"
    report.mkdir(parents=True, exist_ok=True)
    source = run / "source_snapshot"
    source.mkdir()
    shutil.copy2(__file__, source / Path(__file__).name)
    meta = root / "dataset/raw/nuscenes_trainval/v1.0-trainval"
    history = root / "outputs/project_v2/lane_qwen_expansion_v1"
    development = {}
    split_files = []
    for base in (history, history / "prospective_supplement_v2"):
        sf = base / "split_manifest.json"
        split = json.loads(sf.read_text())["scene_to_split"]
        split_files.append(sf)
        for row in json.loads((base / "scene_selection_manifest.json").read_text())["scenes"]:
            if split.get(row["scene"]) == "development":
                development[row["scene"]] = dict(row, source_root=str(base))
    names = sorted(development)[:2]
    selected = {s: development[s] for s in names}
    scene_tokens = {r["scene_token"]: name for name, r in selected.items()}
    samples = {r["token"]: r for r in iter_records(meta / "sample.json") if r["scene_token"] in scene_tokens}
    chosen = {}
    for name in names:
        chosen[name] = sorted((r for r in samples.values() if scene_tokens[r["scene_token"]] == name), key=lambda r: (r["timestamp"], r["token"]))[:20]
    wanted = {r["token"] for rows in chosen.values() for r in rows}
    cameras, lidars = {}, {n: [] for n in names}
    for r in iter_records(meta / "sample_data.json"):
        sample = samples.get(r["sample_token"])
        if sample is None:
            continue
        channel = r["filename"].split("/")[1]
        if channel == "LIDAR_TOP":
            lidars[scene_tokens[sample["scene_token"]]].append(r)
        elif r["sample_token"] in wanted and r["is_key_frame"] and channel in CHANNELS:
            cameras[(r["sample_token"], channel)] = r
    calib = {r["token"]: r for r in iter_records(meta / "calibrated_sensor.json")}
    plans = []
    for name, rows in chosen.items():
        for i, sample in enumerate(rows):
            for channel in CHANNELS:
                camera = cameras[(sample["token"], channel)]
                past = [r for r in lidars[name] if r["timestamp"] <= camera["timestamp"] and camera["timestamp"] - r["timestamp"] <= 100000]
                lidar = max(past, key=lambda r: r["timestamp"]) if past else None
                plans.append(dict(scene=name, frame_idx=i, sample=sample, channel=channel, camera=camera, lidar=lidar))
    pose_tokens = {r[k]["ego_pose_token"] for r in plans for k in ("camera", "lidar") if r[k]}
    poses = {r["token"]: r for r in iter_records(meta / "ego_pose.json") if r["token"] in pose_tokens}
    archive = Path("/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval01_blobs.tgz")
    missing, raw_sources = {}, {}
    for plan in plans:
        for channel in ("camera", "lidar"):
            sensor = plan[channel]
            if sensor is None or sensor["filename"] in raw_sources:
                continue
            existing = root / "dataset/raw/nuscenes_trainval" / sensor["filename"]
            old_past = root / "outputs/project_v2/thesis_completion_v1/data_manifest/development_causal_inputs_full_001/raw_past_lidar" / f'{sensor["token"]}.bin'
            path = existing if existing.is_file() else old_past if channel == "lidar" and old_past.is_file() else raw / sensor["filename"]
            raw_sources[sensor["filename"]] = path
            if not path.is_file():
                missing[sensor["filename"]] = path
    scenes = []
    for name, rows in chosen.items():
        groups = []
        for i, sample in enumerate(rows):
            relevant = [r for r in plans if r["scene"] == name and r["frame_idx"] == i]
            latest = max(r["camera"]["timestamp"] for r in relevant)
            group = dict(frame_idx=i, sample_token=sample["token"], reference_timestamp_s=latest / 1e6,
                         official_sample_timestamp_s=sample["timestamp"] / 1e6, views={})
            for plan in relevant:
                camera, lidar, channel = plan["camera"], plan["lidar"], plan["channel"]
                cp, cc = poses[camera["ego_pose_token"]], calib[camera["calibrated_sensor_token"]]
                cw = build_transform(cp["translation"], cp["rotation"]) @ build_transform(cc["translation"], cc["rotation"])
                folder = processed / name / f"{i:06d}" / channel
                folder.mkdir(parents=True)
                lw = None
                if lidar:
                    lp, lc = poses[lidar["ego_pose_token"]], calib[lidar["calibrated_sensor_token"]]
                    lw = build_transform(lp["translation"], lp["rotation"]) @ build_transform(lc["translation"], lc["rotation"])
                view = dict(rgb=str(raw_sources[camera["filename"]]), depth=str(folder / "depth.png"), depth_scale=1000,
                    projection_evidence=str(folder / "projection_evidence.npz"), intrinsic=cc["camera_intrinsic"],
                    camera_pose=cw.tolist(), camera_timestamp_s=camera["timestamp"] / 1e6,
                    lidar_timestamp_s=lidar["timestamp"] / 1e6 if lidar else None,
                    lidar_age_s=(camera["timestamp"] - lidar["timestamp"]) / 1e6 if lidar else None,
                    lidar_token=lidar["token"] if lidar else None,
                    lidar_raw=str(raw_sources[lidar["filename"]]) if lidar else None,
                    lidar_world_pose=lw.tolist() if lw is not None else None,
                    camera_token=camera["token"], camera_source_filename=camera["filename"],
                    lidar_source_filename=lidar["filename"] if lidar else None,
                    width=camera["width"], height=camera["height"],
                    synchronization_wait_s=(latest - camera["timestamp"]) / 1e6,
                    detections=str(run / "detections" / name / f"{i:06d}" / f"{channel}.pkl.gz"))
                group["views"][channel] = view
            groups.append(group)
        scenes.append(dict(scene=name, scene_token=selected[name]["scene_token"], log_token=selected[name]["log_token"],
                           data_role="development_validation_previously_used", groups=groups))
    manifest = dict(schema_version="multiview-20261123-v1", selection_rule="38既有开发场景按名排序首2，各时间排序前20关键帧；不读取方法结果",
        original_development_count=len(development), channels=list(CHANNELS), scenes=scenes,
        inference_time_rule="replay eligibility at max camera timestamp; actual model wall latency reported separately",
        depth_rule="latest same-scene LiDAR <= each camera exposure, <=100ms; missing stays zero; sparse measured returns only; ego motion compensated, no GT motion compensation",
        projection_evidence_schema="pixel_y/pixel_x/point_index(original raw bin row)/depth_m(before millimeter quantization); lidar_token in each view",
        evaluation_ground_truth_separate=True, gt_in_inference=False, new_test_opened=False,
        frontend_run=str(run), source_sha256=digest(__file__), split_sources={str(p): digest(p) for p in split_files},
        shared_archive=dict(path=str(archive), bytes=archive.stat().st_size, mtime_ns=archive.stat().st_mtime_ns),
        budget=dict(total_wall_s=5400, gpu_wall_s=2700, additional_bytes=2 * 1024**3, minimum_free_bytes=20 * 1024**3))
    dump(processed / "manifest.json", manifest)
    dump(processed / "first_group_example.json", scenes[0]["groups"][0])
    dump(processed / "raw_sensor_plan.json", dict(plans=plans, needed_archive_members=sorted(missing)))
    print(json.dumps(dict(manifest=str(processed / "manifest.json"), scenes=names, groups=sum(len(s["groups"]) for s in scenes), views=len(plans), missing_raw_members=len(missing)), ensure_ascii=False), flush=True)
    provenance = []
    # Exactly one bounded read of the identified archive; no unrelated archives scanned.
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if time.monotonic() - begun > args.wall_s:
                raise TimeoutError("按需传感器读取超过预算，冻结manifest与已提取输入保留")
            name = member.name.removeprefix("./")
            if name not in missing:
                continue
            if shutil.disk_usage(raw).free < 20 * 1024**3:
                raise OSError("磁盘低于20GiB，保留进度")
            if not member.isfile() or member.size > 20_000_000:
                raise ValueError("意外归档实体/大小")
            target = missing.pop(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = tar.extractfile(member).read()
            with target.open("xb") as f:
                f.write(data)
            provenance.append(dict(member=name, path=str(target), archive=str(archive), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            if len(provenance) % 20 == 0:
                print(json.dumps(dict(extracted=len(provenance), remaining=len(missing), elapsed_s=time.monotonic()-begun)), flush=True)
            if not missing:
                break
    dump(processed / "extraction_provenance.json", provenance)
    if missing:
        dump(processed / "missing_raw_members.json", sorted(missing))
        raise FileNotFoundError(f"归档侧视/过去LiDAR缺失{len(missing)}项，未伪造输入")
    checks = []
    for scene in scenes:
        for group in scene["groups"]:
            for channel, view in group["views"].items():
                if view["lidar_raw"]:
                    pts = np.fromfile(view["lidar_raw"], dtype=np.float32).reshape(-1, 5)
                    transform = inverse_transform(np.asarray(view["camera_pose"])) @ np.asarray(view["lidar_world_pose"])
                    depth, evidence = projection(pts, transform, view["intrinsic"], view["height"], view["width"])
                else:
                    depth = np.zeros((view["height"], view["width"]), dtype=np.uint16)
                    evidence = dict(pixel_y=np.empty(0, np.int32), pixel_x=np.empty(0, np.int32), point_index=np.empty(0, np.int32), depth_m=np.empty(0, np.float32))
                imageio.imwrite(view["depth"], depth)
                np.savez_compressed(view["projection_evidence"], **evidence)
                checks.append(dict(scene=scene["scene"], frame_idx=group["frame_idx"], channel=channel,
                    rgb_sha256=digest(view["rgb"]), depth_sha256=digest(view["depth"]),
                    projection_sha256=digest(view["projection_evidence"]),
                    lidar_sha256=digest(view["lidar_raw"]) if view["lidar_raw"] else None,
                    valid_lidar_pixels=int(np.count_nonzero(depth)), lidar_age_s=view["lidar_age_s"]))
    dump(processed / "input_verification.json", checks)
    complete = dict(manifest_sha256=digest(processed / "manifest.json"), groups=sum(len(s["groups"]) for s in scenes),
        views=len(checks), missing_past_lidar=sum(r["lidar_age_s"] is None for r in checks), future_lidar_views=0,
        elapsed_s=time.monotonic()-begun, extracted_bytes=sum(r["bytes"] for r in provenance),
        free_after_bytes=shutil.disk_usage(raw).free, frontend_executed=False)
    dump(processed / "inputs_complete.json", complete)
    dump(report / "input_preparation.json", complete)
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
