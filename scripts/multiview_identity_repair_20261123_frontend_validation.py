"""固定现有六类960真实YOLO/MobileSAM/CLIP，对三路相同配置逐图缓存。"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import random
import shutil
import time

import numpy as np
import torch

from conceptgraph.completion.vehicle_frontend import CLASSES, PROMPTS, configure_vehicle_prompts, detect_vehicle_frame
from conceptgraph.project_paths import PROJECT_ROOT
from conceptgraph.scripts.streamlined_detections import load_detection_models


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "dataset/processed/multiview_identity_repair_20261123_validation_v1/manifest.json")
    p.add_argument("--gpu-wall-s", type=int, default=900)
    args = p.parse_args()
    started = time.monotonic()
    manifest = json.loads(args.manifest.read_text())
    prepared = json.loads((args.manifest.parent / "inputs_complete.json").read_text())
    assert prepared["manifest_sha256"] == digest(args.manifest)
    run = Path(manifest["frontend_run"])
    if (run / "frontend_release.json").exists():
        raise FileExistsError("新前端run拒绝覆盖，不自动恢复")
    source_paths = [Path(__file__), PROJECT_ROOT / "conceptgraph/completion/vehicle_frontend.py",
                    PROJECT_ROOT / "conceptgraph/scripts/streamlined_detections.py",
                    PROJECT_ROOT / "conceptgraph/utils/model_utils.py"]
    for path in source_paths:
        target = run / "source_snapshot" / path.relative_to(PROJECT_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    weights = [PROJECT_ROOT / "checkpoint/yolov8l-world.pt", PROJECT_ROOT / "checkpoint/mobile_sam.pt",
        PROJECT_ROOT / "checkpoint/hf_models/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_model.safetensors"]
    release = dict(manifest=str(args.manifest), manifest_sha256=digest(args.manifest),
        input_verification_sha256=digest(args.manifest.parent / "input_verification.json"),
        source_sha256={str(x): digest(x) for x in source_paths},
        weights_sha256={str(x): digest(x) for x in weights},
        seed=17, detector_prompts=PROMPTS, canonical_classes=CLASSES,
        detection_image_size=960, detector_confidence=0.1, device="cuda:0",
        model="yolov8l-world + mobile_sam + CLIP ViT-H-14 laion2B-s32B-b79K",
        actual_models=True, train=False, reused_front_cache=False,
        all_views_same_configuration=True, raw_rgb_only_frontend=True, gt_in_frontend=False,
        image_crops_stored=False, crop_reconstruction="source RGB and xyxy plus unchanged compute_clip_features_batched padding; actual image_feats and text_feats preserved",
        gpu_wall_budget_s=args.gpu_wall_s, minimum_free_bytes=20 * 1024**3,
        additional_budget_bytes=768 * 1024**2, evaluation_mode="offline replay, input eligibility timestamps plus measured model latency; no real-time/closed-loop claim",
        torch_version=torch.__version__, cuda_version=torch.version.cuda, gpu_name=torch.cuda.get_device_name(0))
    dump(run / "frontend_release.json", release)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    print(json.dumps(dict(status="loading_existing_local_weights", run=str(run))), flush=True)
    gpu_started = time.monotonic()
    models = load_detection_models(type("Cfg", (), dict(device="cuda:0"))())
    configure_vehicle_prompts(models)
    for model in models[:3]:
        if hasattr(model, "eval"):
            model.eval()
    torch.cuda.synchronize()
    load_s = time.monotonic() - gpu_started
    total_views = sum(len(g["views"]) for s in manifest["scenes"] for g in s["groups"])
    records = []
    with (run / "frontend_frames.jsonl").open("x") as log:
        for scene in manifest["scenes"]:
            for group in scene["groups"]:
                group_started = time.monotonic()
                for channel, view in group["views"].items():
                    if time.monotonic() - gpu_started >= args.gpu_wall_s:
                        raise TimeoutError("GPU前端达到15分钟预算，已有预测保留")
                    if shutil.disk_usage(run).free < 20 * 1024**3:
                        raise OSError("磁盘低于20GiB，已有预测保留")
                    tick = time.monotonic()
                    with torch.inference_mode():
                        detection = detect_vehicle_frame(models, view["rgb"], image_size=960)
                    torch.cuda.synchronize()
                    model_s = time.monotonic() - tick
                    # RGB crops are reproducible and large; masks/actual CLIP features remain untouched.
                    detection.pop("image_crops", None)
                    n = len(detection["xyxy"])
                    assert len(detection["mask"]) == n == len(detection["image_feats"]) == len(detection["class_id"])
                    assert detection["frame_provenance"]["rgb_sha256"] == digest(view["rgb"])
                    if n:
                        assert detection["mask"].shape[1:] == (view["height"], view["width"])
                        assert np.isfinite(detection["image_feats"]).all()
                    output = Path(view["detections"])
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with gzip.open(output, "xb", compresslevel=1) as f:
                        pickle.dump(detection, f, protocol=pickle.HIGHEST_PROTOCOL)
                    row = dict(scene=scene["scene"], log_token=scene["log_token"], frame_idx=group["frame_idx"],
                        sample_token=group["sample_token"], channel=channel, detections=str(output),
                        detections_sha256=digest(output), detections_count=n, model_wall_s=model_s,
                        rgb_sha256=detection["frame_provenance"]["rgb_sha256"], rgb=view["rgb"],
                        mask_shape=list(detection["mask"].shape), image_feats_shape=list(detection["image_feats"].shape),
                        text_feats_shape=list(detection["text_feats"].shape), camera_timestamp_s=view["camera_timestamp_s"],
                        lidar_timestamp_s=view["lidar_timestamp_s"], lidar_token=view["lidar_token"],
                        depth=view["depth"], projection_evidence=view["projection_evidence"],
                        input_eligibility_timestamp_s=group["reference_timestamp_s"],
                        synchronization_wait_s=view["synchronization_wait_s"],
                        camera_pose=view["camera_pose"], intrinsic=view["intrinsic"],
                        scene_role=scene["data_role"], actual_inference=True)
                    records.append(row)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    if len(records) % 10 == 0:
                        progress = dict(completed=len(records), expected=total_views, elapsed_gpu_s=time.monotonic()-gpu_started,
                            model_wall_s=sum(r["model_wall_s"] for r in records), detections=sum(r["detections_count"] for r in records),
                            free_bytes=shutil.disk_usage(run).free)
                        dump(run / "frontend_progress.json", progress)
                        print(json.dumps(progress), flush=True)
                    if len(records) == 20:
                        speed = dict(first_views=20, model_wall_s=sum(r["model_wall_s"] for r in records),
                            model_load_wall_s=load_s, projected_total_gpu_s=(time.monotonic()-gpu_started)*total_views/20,
                            detections=sum(r["detections_count"] for r in records),
                            masks_and_features_verified=True, all_three_views_represented=True)
                        dump(run / "first20_speed_check.json", speed)
                        print(json.dumps(dict(first20=speed)), flush=True)
                # Model completion is measured in wall time, distinct from replay sensor eligibility.
                with (run / "group_latency.jsonl").open("a") as latency:
                    latency.write(json.dumps(dict(scene=scene["scene"], frame_idx=group["frame_idx"],
                        input_eligibility_timestamp_s=group["reference_timestamp_s"],
                        sequential_three_view_wall_s=time.monotonic()-group_started)) + "\n")
    complete = dict(manifest_sha256=digest(args.manifest), views=len(records), groups=sum(len(s["groups"]) for s in manifest["scenes"]),
        scenes=[s["scene"] for s in manifest["scenes"]], actual_new_frontend_calls=len(records),
        reused_frontend_calls=0, all_expected_views_complete=len(records)==total_views,
        detections=sum(r["detections_count"] for r in records),
        model_load_wall_s=load_s, gpu_wall_s=time.monotonic()-gpu_started,
        total_wall_s=time.monotonic()-started, model_wall_s=sum(r["model_wall_s"] for r in records),
        frontend_frames_sha256=digest(run / "frontend_frames.jsonl"),
        free_after_bytes=shutil.disk_usage(run).free, training_executed=False, no_rgb_color_claim=True,
        input_future_lidar_views=prepared["future_lidar_views"],
        input_missing_past_lidar_views=prepared["missing_past_lidar"])
    dump(run / "frontend_complete.json", complete)
    dump(PROJECT_ROOT / "outputs/reports/multiview_identity_repair_20261123/validation/frontend_complete.json", complete)
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
