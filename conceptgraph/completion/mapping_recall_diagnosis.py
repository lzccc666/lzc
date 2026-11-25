"""验证集逐对象漏失诊断；不改评价分母，不把诊断GT送入推理。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from conceptgraph.completion.depth import depth_support
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, available_development, digest
from conceptgraph.project_paths import experiment_directory


def projected_box(obj, pose, intrinsic, width=1600, height=900):
    width_m, length_m, height_m = obj["size_wlh"]
    corners = np.array([[x, y, z] for x in [-length_m / 2, length_m / 2]
                        for y in [-width_m / 2, width_m / 2]
                        for z in [-height_m / 2, height_m / 2]])
    w, x, y, z = obj["rotation_wxyz"]
    world = Rotation.from_quat([x, y, z, w]).apply(corners) + obj["center"]
    camera = (world - pose[:3, 3]) @ pose[:3, :3]
    camera = camera[camera[:, 2] > 0.1]
    if not len(camera):
        return None
    uvz = camera @ intrinsic.T
    uv = uvz[:, :2] / uvz[:, 2:]
    box = np.r_[np.maximum(uv.min(axis=0), 0), np.minimum(uv.max(axis=0), [width, height])]
    return box if np.all(box[2:] > box[:2]) else None


def box_iou(a, b):
    if a is None or b is None:
        return 0.0
    intersection = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0).prod()
    area_a = np.maximum(a[2:] - a[:2], 0).prod()
    area_b = np.maximum(b[2:] - b[:2], 0).prod()
    return float(intersection / max(area_a + area_b - intersection, 1e-9))


def dump(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--wall-budget-s", type=float, default=1800)
    args = parser.parse_args()
    scenes = [s for s in available_development() if not args.scene or s["scene"] in args.scene]
    if not scenes or (args.scene and set(args.scene) != {s["scene"] for s in scenes}):
        raise ValueError("只允许验证／调试场景")
    dest = experiment_directory("nuscenes", "validation", args.run_id, create=True)
    baseline = PROJECT / "outputs/results/nuscenes/validation/mapping"
    source_root = OUTPUT / "mapping_evaluation/causal_mapping_full_001"
    inputs = [source_root / s["scene"] / "frames.jsonl" for s in scenes]
    dump(dest / "release.json", dict(
        scenes=scenes, mode="diagnostic_only", threshold_m=3.0, diagnostic_box_iou=0.25,
        sources={str(p): digest(p) for p in inputs}, source_sha256=digest(Path(__file__)),
        baseline=str(baseline), minimum_free_gib=20, wall_budget_s=args.wall_budget_s,
        new_GPU_calls=0, test_accessed=False, GT_in_inference=False,
        note="2D框由官方3D框投影得到，只作漏失定位，不是官方检测mAP；历史GT时间差仍存在"))
    begun = time.monotonic()
    counts = Counter()
    by_group = defaultdict(Counter)
    frames = 0
    with gzip.open(dest / "objects.jsonl.gz", "wt") as handle:
        for scene, source_path in zip(scenes, inputs):
            base_rows = {a["frame_idx"]: a for a in map(json.loads,
                         (baseline / scene["scene"] / "frames.jsonl").open())}
            scene_counts = Counter()
            for source in map(json.loads, source_path.open()):
                if time.monotonic() - begun > args.wall_budget_s:
                    raise TimeoutError("诊断达到预算，保留已完成记录")
                if shutil.disk_usage(dest).free < 20 * 1024**3:
                    raise OSError("空闲空间低于20GiB")
                base = base_rows[source["frame_idx"]]
                truth = source["evaluation_only_official"]
                recovered = {m["truth_index"] for m in base["metrics"]["full"]["matches"]}
                frames += 1
                if not truth:
                    continue
                pose, intrinsic = np.asarray(source["camera_pose"]), np.asarray(source["intrinsic"])
                with gzip.open(source["detections"], "rb") as stream:
                    detections = pickle.load(stream)
                depth = np.asarray(Image.open(source["depth"]), dtype=float) / 1000.0
                classes = detections["classes"]
                evidence = [depth_support(depth, mask, robust=True)[1] for mask in detections["mask"]]
                for ti, obj in enumerate(truth):
                    box = projected_box(obj, pose, intrinsic)
                    indices = [i for i, cid in enumerate(detections["class_id"])
                               if classes[int(cid)] == obj["category"] and
                               box_iou(box, detections["xyxy"][i]) >= 0.25]
                    confident = [i for i in indices if detections["confidence"][i] >= 0.2]
                    masks = [i for i in confident if detections["mask"][i].sum() >= 25]
                    supported = [i for i in masks if evidence[i]["retained_pixels"] >= 5]
                    if ti in recovered:
                        reason = "recovered_in_map"
                    elif not indices:
                        reason = "no_overlapping_same_class_detection"
                    elif not confident:
                        reason = "all_overlapping_detections_below_mapping_confidence"
                    elif not masks:
                        reason = "mask_too_small"
                    elif not supported:
                        reason = "insufficient_current_mask_depth"
                    else:
                        reason = "detection_and_depth_present_but_3D_or_fusion_unmatched"
                    distance = float(np.linalg.norm(np.asarray(obj["center"][:2]) - pose[:2, 3]))
                    distance_group = "0_20m" if distance < 20 else "20_40m" if distance < 40 else "40m_plus"
                    counts[reason] += 1
                    scene_counts[reason] += 1
                    for group in ["category:" + obj["category"], "range:" + distance_group,
                                  "visibility:" + str(obj["visibility_token"])]:
                        by_group[group]["GT"] += 1
                        by_group[group]["matched"] += ti in recovered
                        by_group[group][reason] += 1
                    record = dict(scene=scene["scene"], frame_idx=source["frame_idx"],
                                  instance=obj["instance_token"], category=obj["category"],
                                  distance_m=distance, visibility=obj["visibility_token"],
                                  official_sample_lidar_pts=obj["num_lidar_pts"],
                                  mapped=ti in recovered, reason=reason,
                                  overlapping_raw_detections=len(indices),
                                  best_box_iou=max((box_iou(box, detections["xyxy"][i]) for i in indices), default=0),
                                  best_confidence=max((float(detections["confidence"][i]) for i in indices), default=None),
                                  maximum_robust_mask_depth_points=max((evidence[i]["retained_pixels"] for i in indices), default=0))
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            print(json.dumps(dict(scene=scene["scene"], objects=sum(scene_counts.values()),
                                  counts=scene_counts), ensure_ascii=False), flush=True)
    groups = {k: dict(v, recall=v["matched"] / v["GT"]) for k, v in sorted(by_group.items())}
    report = dict(frames=frames, scenes=len(scenes), truth_observations=sum(counts.values()),
                  matched=counts["recovered_in_map"], recall=counts["recovered_in_map"] / sum(counts.values()),
                  counts=counts, groups=groups, wall_s=time.monotonic() - begun,
                  records_sha256=digest(dest / "objects.jsonl.gz"),
                  limitations=["漏失优先级分类，不是严格因果消融；同对象可能有多个问题",
                               "2D投影只作诊断，未人工逐框核验；框重叠不保证掩膜身份正确",
                               "官方num_lidar_pts来自标注帧，不等于本方法当前掩膜实际点数",
                               "分组结果附加呈现，不替代全范围主指标"],
                  test_accessed=False, labels_changed=False)
    dump(dest / "report.json", report)
    with (dest / "groups.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "truth", "matched", "recall"])
        writer.writerows((k, v["GT"], v["matched"], v["recall"]) for k, v in groups.items())
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
