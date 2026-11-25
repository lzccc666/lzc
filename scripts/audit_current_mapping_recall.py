"""只读核对当前前端漏失，不改变推理、全范围分母或三米匹配门限。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
from PIL import Image

from conceptgraph.completion.depth import depth_support
from conceptgraph.completion.mapping_recall_diagnosis import box_iou, projected_box
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, available_development, digest, dump
from conceptgraph.project_paths import experiment_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--wall-budget-s", type=float, default=300)
    args = parser.parse_args()
    started = time.monotonic()
    destination = experiment_directory("nuscenes", "validation", args.run_id, create=True)
    reference = PROJECT / "outputs/results/nuscenes/validation/iterations/vehicle_frontend_full_20261118_v1"
    scenes = available_development()
    inputs = {s["scene"]: OUTPUT / "mapping_evaluation/causal_mapping_full_001" / s["scene"] / "frames.jsonl"
              for s in scenes}
    sources = [Path(__file__).resolve(), PROJECT / "conceptgraph/completion/mapping_recall_diagnosis.py",
               PROJECT / "conceptgraph/completion/depth.py"]
    for source in sources:
        target = destination / "source_snapshot" / source.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    dump(destination / "release.json", dict(
        mode="current_frontend_diagnostic_only", scenes=scenes, baseline=str(reference),
        input_sha256={str(p): digest(p) for p in inputs.values()},
        reference_sha256={s["scene"]: digest(reference / s["scene"] / "frames.jsonl.gz") for s in scenes},
        source_sha256={str(p): digest(p) for p in sources}, threshold_m=3.0,
        diagnostic_box_iou=.25, diagnostic_minimum_depth_pixels=4, confidence_floor=.1,
        wall_budget_s=args.wall_budget_s, minimum_free_gib=20, maximum_new_gib=.1,
        labels_changed=False, actual_GPU_calls=0, test_accessed=False,
        limitation="优先级漏失归因，不是严格因果上界；官方框投影非人工二维真值"))
    counts, groups = Counter(), defaultdict(Counter)
    frames = 0
    with gzip.open(destination / "objects.jsonl.gz", "wt") as out:
        for scene in scenes:
            name = scene["scene"]
            with gzip.open(reference / name / "frames.jsonl.gz", "rt") as stream:
                cache = {r["frame_idx"]: r for r in map(json.loads, stream)}
            for row in map(json.loads, inputs[name].open()):
                if time.monotonic() - started > args.wall_budget_s:
                    raise TimeoutError("诊断达到预算，完整汇总不生成，保留真实前缀")
                if shutil.disk_usage(destination).free < 20 * 1024**3:
                    raise OSError("空闲空间低于20GiB")
                idx = row["frame_idx"]
                baseline = cache[idx]
                path = Path(baseline["detections"])
                if digest(path) != baseline["detection_sha256"]:
                    raise ValueError("检测输入指纹改变")
                with gzip.open(path, "rb") as stream:
                    detections = pickle.load(stream)
                depth = np.asarray(Image.open(row["depth"]), dtype=float) / 1000
                pose, intrinsic = np.asarray(row["camera_pose"]), np.asarray(row["intrinsic"])
                evidence = [depth_support(depth, mask, robust=True)[1] for mask in detections["mask"]]
                recovered = {m["truth_index"] for m in baseline["metrics"]["frontend_v2_evidence"]["matches"]}
                frames += 1
                for i, obj in enumerate(row["evaluation_only_official"]):
                    box = projected_box(obj, pose, intrinsic)
                    overlap = [j for j, cid in enumerate(detections["class_id"])
                               if detections["classes"][int(cid)] == obj["category"]
                               and box_iou(box, detections["xyxy"][j]) >= .25]
                    confident = [j for j in overlap if detections["confidence"][j] >= .1]
                    supported = [j for j in confident if evidence[j]["retained_pixels"] >= 4]
                    if i in recovered:
                        reason = "recovered_in_map"
                    elif not overlap:
                        reason = "no_overlapping_same_class_detection"
                    elif not confident:
                        reason = "below_input_confidence"
                    elif not supported:
                        reason = "insufficient_current_mask_depth"
                    else:
                        reason = "observation_present_but_3D_confirmation_or_fusion_unmatched"
                    distance = float(np.linalg.norm(np.asarray(obj["center"][:2]) - pose[:2, 3]))
                    bucket = "0_20m" if distance < 20 else "20_40m" if distance < 40 else "40m_plus"
                    counts[reason] += 1
                    for group in ["category:" + obj["category"], "range:" + bucket]:
                        groups[group]["truth"] += 1
                        groups[group]["matched"] += i in recovered
                        groups[group][reason] += 1
                    out.write(json.dumps(dict(scene=name, frame_idx=idx,
                              instance=obj["instance_token"], category=obj["category"],
                              distance_m=distance, matched=i in recovered, reason=reason), allow_nan=False) + "\n")
            print(name, "diagnosed", flush=True)
    total = sum(counts.values())
    dump(destination / "report.json", dict(
        scenes=len(scenes), frames=frames, truth=total, matched=counts["recovered_in_map"],
        recall=counts["recovered_in_map"] / total, counts=dict(counts),
        groups={k: dict(v, recall=v["matched"] / v["truth"]) for k, v in sorted(groups.items())},
        wall_s=time.monotonic() - started, test_accessed=False,
        labels_changed=False, output_sha256=digest(destination / "objects.jsonl.gz"),
        caveat="未匹配不等于纯关联失败；当前未检测不表示历史帧一定没有证据；诊断不构成80%不可达的证明"))


if __name__ == "__main__":
    main()
