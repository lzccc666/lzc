"""相同验证序列上真实运行RAM/Grounding DINO、MobileSAM、CLIP与连续对象建图。"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path
import pickle
import shutil
import time
import traceback

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

from conceptgraph.completion.fault_robustness import mapping_config, retain_detections
from conceptgraph.completion.quality import object_frame_metrics
from conceptgraph.completion.ram_dino_frontend import ASSETS, CLASSES, detect_grounding_boxes, load_grounding_models, segment_and_embed
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, available_development, digest, predict
from conceptgraph.completion.temporal_identity_evaluation import identity_counts, prepare_sequence
from conceptgraph.project_paths import experiment_directory
from conceptgraph.scripts.streamlined_detections import load_detection_models
from conceptgraph.slam.evidence_mapper import EvidenceAwareMapper
from conceptgraph.slam.streamlined_mapping import MappingFrame
from run_mapping_iteration import ratios


def dump(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--mode", choices=["both", "dino_fixed", "ram_dino"], default="both")
    parser.add_argument("--native-only", action="store_true", help="仅推荐门限0.25进入分割特征与三维地图；原始低门限框仍保存诊断")
    parser.add_argument("--wall-budget-s", type=float, default=3600)
    args = parser.parse_args()
    scenes = [s for s in available_development() if not args.scene or s["scene"] in args.scene]
    if not scenes or (args.scene and set(args.scene) != {s["scene"] for s in scenes}):
        raise ValueError("只允许固定验证场景")
    modes = ["dino_fixed", "ram_dino"] if args.mode == "both" else [args.mode]
    thresholds = [("_native", 0.25)] if args.native_only else [("_native", 0.25), ("_shared", 0.1)]
    methods = [mode + suffix for mode in modes for suffix, _ in thresholds]
    dest = experiment_directory("nuscenes", "validation", args.run_id, create=True)
    begun = time.monotonic()
    sources = [Path(__file__).resolve(), PROJECT / "conceptgraph/completion/ram_dino_frontend.py",
               PROJECT / "conceptgraph/completion/vehicle_frontend.py", PROJECT / "conceptgraph/slam/evidence_mapper.py",
               PROJECT / "conceptgraph/slam/streamlined_mapping.py", PROJECT / "conceptgraph/slam/utils.py",
               PROJECT / "conceptgraph/slam/mapping.py", PROJECT / "conceptgraph/utils/model_utils.py",
               PROJECT / "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"]
    frozen = {str(p): digest(p) for p in sources}
    for path in sources:
        target = dest / "source_snapshot" / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    inputs = [OUTPUT / "mapping_evaluation/causal_mapping_full_001" / s["scene"] / "frames.jsonl" for s in scenes]
    cfg = mapping_config()
    cfg.merge_with(dict(mask_conf_threshold=0.1, strong_confidence=0.2, weak_confirmation_observations=2))
    weights = [PROJECT / "checkpoint/groundingdino_swint_ogc.pth", PROJECT / "checkpoint/mobile_sam.pt"]
    weights += [ASSETS / "ram_swin_large_14m.pth"] if "ram_dino" in modes else []
    weights += list((PROJECT / "checkpoint/hf_models").glob("**/open_clip_model.safetensors"))
    dump(dest / "release.json", dict(
        scenes=scenes, modes=modes, methods=methods, seed=17, sources_sha256=frozen,
        inputs_sha256={str(p): digest(p) for p in inputs}, weights_sha256={str(p): digest(p) for p in weights},
        mapping_config=OmegaConf.to_container(cfg), dino_save_threshold=0.1, dino_text_threshold=0.2,
        detector_thresholds=dict(native=0.25, shared=0.1), nms_iou=0.5,
        dino_resize="short800/max1333", ram_resize=384, yolo_reference_resize=960,
        class_decoding="highest_mean_full_prompt_token_score；不将混合短语归第一类或直接丢弃",
        SAM="same MobileSAM", CLIP="same ViT-H-14", evaluation_threshold_m=3.0,
        native_note="GSA推荐框门限0.25；shared保留与YOLO相同0.1门限，两组同时报告，不择优隐藏",
        native_only=args.native_only, three_dimensional_conditions=methods,
        runtime_scope_note="低门限154帧二维对照完整保留；三维低门限前缀保留，主表只用完整跑完的推荐配置",
        auto_prompt_GT_injection=False, inference_GT_access=False, test_accessed=False,
        BERT_weights="完整来自GroundingDINO检查点，初始化后检查无missing_keys",
        wall_budget_s=args.wall_budget_s, new_storage_budget_gib=2, minimum_free_gib=20,
        input_root="dataset/processed/nuscenes/validation/converted",
        reference_run="vehicle_frontend_full_20261118_v1", full_ConceptGraphs_reproduction=False))
    reports, total_bytes = [], 0
    try:
        grounding = load_grounding_models(with_ram="ram_dino" in modes)
        models = load_detection_models(type("Cfg", (), dict(device="cuda:0"))())
        common_models = models[1:]
        for scene, source in zip(scenes, inputs):
            rows = [json.loads(line) for line in source.open()]
            mappers = {m: EvidenceAwareMapper(copy.deepcopy(cfg), CLASSES) for m in methods}
            case = dest / scene["scene"]
            (case / "detections").mkdir(parents=True)
            records = []
            with gzip.open(case / "frames.jsonl.gz", "wt") as output:
                for row in rows:
                    if time.monotonic() - begun > args.wall_budget_s:
                        raise TimeoutError("达到事前时间预算，保留完整场景与当前帧")
                    if shutil.disk_usage(dest).free < 20 * 1024**3 or total_bytes > 2 * 1024**3:
                        raise OSError("达到磁盘保护线或事前2GiB结果预算")
                    idx = row["frame_idx"]
                    image = np.asarray(Image.open(row["rgb"]).convert("RGB"))
                    depth = np.asarray(Image.open(row["depth"]), dtype=float) / 1000.0
                    pose, intrinsic = np.asarray(row["camera_pose"]), np.asarray(row["intrinsic"])
                    predictions, times, frontend_times, detection_paths, detection_hashes = {}, {}, {}, {}, {}
                    for mode in modes:
                        torch.cuda.synchronize()
                        start = time.monotonic()
                        boxes = detect_grounding_boxes(grounding, row["rgb"], mode=mode)
                        if args.native_only:
                            # 只省去本来不会进入native地图的候选的SAM/CLIP计算。
                            # 所有0.1原始框和各类分数仍在raw_*诊断字段保留。
                            keep = boxes["confidence"] >= 0.25
                            boxes = dict(boxes)
                            boxes["sam_input_box_threshold"] = 0.25
                            for key in ["xyxy", "confidence", "class_id"]:
                                boxes[key] = boxes[key][keep]
                        detections = segment_and_embed(common_models, row["rgb"], boxes)
                        torch.cuda.synchronize()
                        frontend_times[mode] = time.monotonic() - start
                        detection_path = case / "detections" / f"{idx:06d}_{mode}.pkl.gz"
                        with gzip.open(detection_path, "wb") as stream:
                            pickle.dump(detections, stream)
                        total_bytes += detection_path.stat().st_size
                        detection_paths[mode] = str(detection_path)
                        detection_hashes[mode] = digest(detection_path)
                        for suffix, threshold in thresholds:
                            method = mode + suffix
                            observation = retain_detections(detections, detections["confidence"] >= threshold)
                            np.random.seed(17 + idx)
                            start = time.monotonic()
                            mappers[method].update(MappingFrame(idx, image, depth, intrinsic, pose, observation,
                                                               row["rgb"], timestamp_s=row["camera_timestamp_s"]))
                            predictions[method] = predict(mappers[method].confirmed_objects(), CLASSES, pose=pose, intrinsic=intrinsic)
                            times[method] = time.monotonic() - start
                    # 标准答案只在所有模型与建图计算之后进入评价。
                    truth = row["evaluation_only_official"]
                    full = dict(scene=scene["scene"], frame_idx=idx, camera_timestamp_s=row["camera_timestamp_s"],
                                rgb=row["rgb"], depth=row["depth"], detections=detection_paths, detection_sha256=detection_hashes,
                                predictions=predictions, metrics={k: object_frame_metrics(v, truth) for k, v in predictions.items()},
                                evaluation_only_official=truth, gt_in_inference=False, frontend_s=frontend_times, mapping_s=times)
                    output.write(json.dumps(full, ensure_ascii=False, allow_nan=False) + "\n")
                    output.flush()
                    records.append(full)
            conditions = {}
            for method in methods:
                counts = {k: sum(r["metrics"][method][k] for r in records)
                          for k in ["true_positive", "truth", "predictions", "duplicate_nodes"]}
                counts["center_error_sum_m"] = sum(m["distance_m"] for r in records for m in r["metrics"][method]["matches"])
                counts.update(identity_counts(prepare_sequence(records, method), "all_gated")["counts"])
                conditions[method] = dict(counts=counts, metrics=ratios(counts))
            report = dict(scene=scene["scene"], log=scene["log_token"], frames=len(records), conditions=conditions,
                          frames_sha256=digest(case / "frames.jsonl.gz"),
                          frontend_seconds={mode: sum(r["frontend_s"][mode] for r in records) for mode in modes})
            dump(case / "report.json", report)
            reports.append(report)
            print(json.dumps(dict(scene=scene["scene"], metrics={k: v["metrics"] for k, v in conditions.items()},
                                  bytes=total_bytes, elapsed_s=time.monotonic() - begun)), flush=True)
        totals = {}
        for method in methods:
            counts = {k: sum(r["conditions"][method]["counts"][k] for r in reports)
                      for k in reports[0]["conditions"][method]["counts"]}
            totals[method] = dict(counts=counts, metrics=ratios(counts))
        if any(digest(Path(p)) != h for p, h in frozen.items()):
            raise ValueError("运行源码发生变化，不签发完整报告")
        dump(dest / "report.json", dict(scenes=len(scenes), frames_each=sum(r["frames"] for r in reports), conditions=totals,
                                        raw_detection_bytes=total_bytes, wall_s=time.monotonic() - begun,
                                        actual_DINO_calls=len(modes) * sum(r["frames"] for r in reports),
                                        actual_RAM_calls=sum(r["frames"] for r in reports) if "ram_dino" in modes else 0,
                                        test_accessed=False, actual_arrival_evaluated=False, GT_in_inference=False))
        print(json.dumps(totals, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        dump(dest / "interrupted.json", dict(error=repr(exc), traceback=traceback.format_exc(), completed_scenes=len(reports)))
        raise


if __name__ == "__main__":
    main()
