"""车载六类提示适配，person统一回写pedestrian；几何仍来自真实测量。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from conceptgraph.utils.model_utils import compute_clip_features_batched


PROMPTS = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]
CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"]


def configure_vehicle_prompts(models):
    detector = models[0]
    detector.to("cuda:0")
    detector.set_classes(PROMPTS)


def detect_vehicle_frame(models, color_path, *, image_size=960):
    detector, segmenter, clip_model, preprocess, tokenizer = models
    image = cv2.imread(str(color_path))
    if image is None:
        raise ValueError(f"无法读取图像：{color_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detector.predict(str(color_path), conf=0.1, imgsz=image_size, device=0, verbose=False)[0]
    boxes = result.boxes.xyxy.clone()
    confidence = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    if len(boxes):
        masks = segmenter.predict(str(color_path), bboxes=boxes.clone(), verbose=False)[0].masks.data.cpu().numpy().astype(bool)
        detections = sv.Detections(xyxy=boxes.cpu().numpy(), confidence=confidence, class_id=class_ids, mask=masks)
        crops, image_features, text_features = compute_clip_features_batched(
            rgb, detections, clip_model, preprocess, tokenizer, CLASSES, "cuda:0")
    else:
        masks = np.zeros((0, *rgb.shape[:2]), dtype=bool)
        crops = []
        dimension = int(clip_model.visual.output_dim)
        image_features = np.empty((0, dimension), dtype=np.float32)
        text_features = np.empty((0, dimension), dtype=np.float32)
    return dict(xyxy=boxes.cpu().numpy(), confidence=confidence, class_id=class_ids,
                mask=masks, classes=CLASSES, image_crops=crops, image_feats=image_features,
                text_feats=text_features, frame_provenance=dict(
                    rgb_sha256=hashlib.sha256(Path(color_path).read_bytes()).hexdigest(),
                    image_size_wh=[int(rgb.shape[1]), int(rgb.shape[0])],
                    bbox_coordinate_frame="original_rgb_pixels", sam_prompt_tensor_isolated=True,
                    frontend_version="vehicle-six-person-960-v2", detector_prompts=PROMPTS,
                    canonical_classes=CLASSES, detection_image_size=image_size))
