"""固定154个验证帧比较YOLO、Grounding DINO和RAM自动类别；只评价二维框。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from ultralytics import YOLO

from conceptgraph.completion.association import gated_assignment
from conceptgraph.completion.mapping_recall_diagnosis import box_iou, projected_box
from conceptgraph.completion.ram_dino_frontend import ASSETS, CLASSES, detect_grounding_boxes, load_grounding_models
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, available_development, digest
from conceptgraph.completion.vehicle_frontend import PROMPTS
from conceptgraph.project_paths import experiment_directory


def dump(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def compare_boxes(boxes, classes, truth, pose, intrinsic):
    gt_boxes = [projected_box(o, pose, intrinsic) for o in truth]
    scores = np.full((len(boxes), len(truth)), -np.inf)
    for i, (box, name) in enumerate(zip(boxes, classes)):
        for j, obj in enumerate(truth):
            overlap = box_iou(box, gt_boxes[j])
            if name == obj["category"] and overlap >= 0.25:
                scores[i, j] = overlap
    matched = set(gated_assignment(scores).values())
    counts = dict(GT=len(truth), matched=len(matched), predictions=len(boxes))
    categories = {name: dict(GT=sum(o["category"] == name for o in truth),
                             matched=sum(truth[j]["category"] == name for j in matched)) for name in CLASSES}
    return counts, categories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    selected, sources = [], []
    for scene in available_development():
        source = OUTPUT / "mapping_evaluation/causal_mapping_full_001" / scene["scene"] / "frames.jsonl"
        sources.append(source)
        selected.extend(row for row in map(json.loads, source.open()) if row["frame_idx"] % 10 == 0)
    if args.max_frames:
        selected = selected[:args.max_frames]
    dest = experiment_directory("nuscenes", "validation", args.run_id, create=True)
    method_names = ["yolo_six_960", "dino_fixed_010", "dino_fixed_025", "ram_dino_010", "ram_dino_025"]
    source_files = [Path(__file__).resolve(), PROJECT / "conceptgraph/completion/ram_dino_frontend.py"]
    dump(dest / "release.json", dict(
        methods=method_names, frames_each=len(selected), scenes=len({r["scene"] for r in selected}),
        stride=10, selection="固定验证帧idx%10==0；不根据GT或检测表现选帧", seed=17,
        diagnostic_box_iou=0.25, dino_box_save_threshold=0.1, dino_text_threshold=0.2,
        dino_native_box_threshold=0.25, dino_nms_iou=0.5, dino_input="short800/max1333",
        class_decoding="highest_mean_full_prompt_token_score",
        yolo_input=960, yolo_box_threshold=0.1, auto_tags_injected_GT_classes=False,
        sources_sha256={str(p): digest(p) for p in source_files},
        inputs_sha256={str(p): digest(p) for p in sources},
        assets=json.loads((ASSETS / "assets.json").read_text()),
        wall_budget_s=1200, new_storage_budget_mib=30, minimum_free_gib=20,
        box_diagnostic_only=True, official_mAP=False, test_accessed=False))
    totals = {m: Counter() for m in method_names}
    categories = {m: defaultdict(Counter) for m in method_names}
    timings = defaultdict(list)
    begun = time.monotonic()
    try:
        grounding_models = load_grounding_models()
        yolo = YOLO(str(PROJECT / "checkpoint/yolov8l-world.pt")).to("cuda:0")
        yolo.set_classes(PROMPTS)
        with (dest / "frames.jsonl").open("x") as stream:
            for fi, row in enumerate(selected):
                if time.monotonic() - begun > 1200 or shutil.disk_usage(dest).free < 20 * 1024**3:
                    raise RuntimeError("达到时间或空间保护线，保留已有结果")
                torch.cuda.synchronize()
                started = time.monotonic()
                result = yolo.predict(row["rgb"], conf=0.1, imgsz=960, device=0, verbose=False)[0]
                torch.cuda.synchronize()
                predictions = {"yolo_six_960": dict(xyxy=result.boxes.xyxy.cpu().numpy(),
                                                   confidence=result.boxes.conf.cpu().numpy(),
                                                   class_id=result.boxes.cls.cpu().numpy().astype(int),
                                                   detection_s=time.monotonic() - started)}
                for mode in ["dino_fixed", "ram_dino"]:
                    boxes = detect_grounding_boxes(grounding_models, row["rgb"], mode=mode)
                    for threshold in [0.1, 0.25]:
                        keep = boxes["confidence"] >= threshold
                        method = mode + ("_010" if threshold == 0.1 else "_025")
                        predictions[method] = {k: boxes[k][keep] for k in ["xyxy", "confidence", "class_id"]}
                        predictions[method].update({k: boxes[k] for k in ["ram_tags", "detector_classes", "raw_phrases",
                            "detection_s", "tagging_s", "prompt_token_count", "max_text_len", "raw_unknown_or_out_of_scope_count"]})
                for method, boxes in predictions.items():
                    counts, group = compare_boxes(boxes["xyxy"], [CLASSES[i] for i in boxes["class_id"]],
                                                  row["evaluation_only_official"], np.asarray(row["camera_pose"]),
                                                  np.asarray(row["intrinsic"]))
                    totals[method].update(counts)
                    for name, c in group.items():
                        categories[method][name].update(c)
                    timings[method].append(boxes["detection_s"])
                    saved = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in boxes.items()}
                    stream.write(json.dumps(dict(method=method, scene=row["scene"], frame_idx=row["frame_idx"],
                                                 counts=counts, predictions=saved), ensure_ascii=False) + "\n")
                stream.flush()
                if fi % 20 == 0:
                    print(json.dumps(dict(frames=fi + 1, total=len(selected), totals=totals)), flush=True)
        report = dict(frames_each=len(selected), scenes=len({r["scene"] for r in selected}),
                      totals={m: dict(c, recall=c["matched"] / c["GT"], precision=c["matched"] / c["predictions"] if c["predictions"] else None,
                                      mean_detection_s=float(np.mean(timings[m]))) for m, c in totals.items()},
                      by_class={m: {name: dict(c, recall=c["matched"] / c["GT"] if c["GT"] else None)
                                    for name, c in groups.items()} for m, groups in categories.items()},
                      wall_s=time.monotonic() - begun, box_diagnostic_only=True, test_accessed=False)
        dump(dest / "report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        dump(dest / "interrupted.json", dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
