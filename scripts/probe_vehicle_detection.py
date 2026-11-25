"""固定验证帧比较车载类别提示；真实GPU框检测诊断，不冒充完整建图。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
from ultralytics import YOLO

from conceptgraph.completion.association import gated_assignment
from conceptgraph.completion.mapping_recall_diagnosis import box_iou, projected_box
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, available_development, digest
from conceptgraph.project_paths import experiment_directory


CONFIGS = {
    "original14_640": (None, 640),
    "six_pedestrian_640": (["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"], 640),
    "six_person_640": (["car", "truck", "bus", "motorcycle", "bicycle", "person"], 640),
    "six_person_960": (["car", "truck", "bus", "motorcycle", "bicycle", "person"], 960),
}


def dump(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError("stride必须为正")
    selected = []
    for scene in available_development():
        source = OUTPUT / "mapping_evaluation/causal_mapping_full_001" / scene["scene"] / "frames.jsonl"
        selected.extend(row for row in map(json.loads, source.open()) if row["frame_idx"] % args.stride == 0)
    dest = experiment_directory("nuscenes", "validation", args.run_id, create=True)
    original = [x.strip() for x in (PROJECT / "conceptgraph/vehicle_classes.txt").read_text().splitlines() if x.strip()]
    config = {k: dict(classes=v[0] or original, imgsz=v[1]) for k, v in CONFIGS.items()}
    weights = PROJECT / "checkpoint/yolov8l-world.pt"
    dump(dest / "release.json", dict(
        config=config, frames=len(selected), scenes=len({r["scene"] for r in selected}), stride=args.stride,
        selection="各验证场景预定frame_idx%stride==0，不按检测或GT结果挑帧",
        confidence=0.1, box_iou=0.25, weights_sha256=digest(weights), source_sha256=digest(Path(__file__)),
        box_diagnostic_only=True, segmentation_or_mapping_evaluated=False, test_accessed=False,
        wall_budget_s=600, GPU_device="cuda:0", official_test_metric=False))
    # 首次缓存文字编码器前就固定设备，避免后续改类别时文字token和编码器设备不同。
    model = YOLO(str(weights)).to("cuda:0")
    totals, by_class = {}, {}
    begun = time.monotonic()
    with (dest / "frames.jsonl").open("x") as output:
        for method, cfg in config.items():
            model.set_classes(cfg["classes"])
            counts = Counter()
            categories = defaultdict(Counter)
            for row in selected:
                if time.monotonic() - begun > 600:
                    raise TimeoutError("GPU探查超过冻结预算，保留记录")
                result = model.predict(row["rgb"], conf=0.1, imgsz=cfg["imgsz"], device=0, verbose=False)[0]
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                names = ["pedestrian" if cfg["classes"][i] == "person" else cfg["classes"][i] for i in class_ids]
                # 官方答案只用于这一段评价；没有传给检测器。
                truth = row["evaluation_only_official"]
                gt_boxes = [projected_box(o, np.asarray(row["camera_pose"]), np.asarray(row["intrinsic"])) for o in truth]
                scores = np.full((len(boxes), len(truth)), -np.inf)
                for i, (box, category) in enumerate(zip(boxes, names)):
                    for j, obj in enumerate(truth):
                        overlap = box_iou(box, gt_boxes[j])
                        if category == obj["category"] and overlap >= 0.25:
                            scores[i, j] = overlap
                assignments = gated_assignment(scores)
                matched = set(assignments.values())
                counts.update(GT=len(truth), matched=len(matched), predictions=len(boxes))
                for j, obj in enumerate(truth):
                    categories[obj["category"]]["GT"] += 1
                    categories[obj["category"]]["matched"] += j in matched
                output.write(json.dumps(dict(method=method, scene=row["scene"], frame_idx=row["frame_idx"],
                                             boxes=boxes.tolist(), categories=names,
                                             confidence=result.boxes.conf.cpu().numpy().tolist(),
                                             GT=len(truth), matched=len(matched)), ensure_ascii=False) + "\n")
            totals[method] = dict(counts, recall=counts["matched"] / counts["GT"])
            by_class[method] = {k: dict(v, recall=v["matched"] / v["GT"]) for k, v in categories.items()}
            print(json.dumps(dict(method=method, total=totals[method], by_class=by_class[method])), flush=True)
    dump(dest / "report.json", dict(totals=totals, by_class=by_class, frames_each=len(selected),
                                    wall_s=time.monotonic() - begun, actual_GPU_calls=len(selected) * len(config),
                                    box_diagnostic_only=True, test_accessed=False,
                                    note="3D框投影IoU诊断，不能当官方mAP或最终地图召回；imgsz变化增加计算量"))


if __name__ == "__main__":
    main()
