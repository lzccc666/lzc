"""YOLO/MobileSAM/CLIP detections, including valid frames with no objects."""

import gzip
import hashlib
import json
import pickle
from pathlib import Path

import cv2
import hydra
import numpy as np
import open_clip
import supervision as sv
import torch
from omegaconf import DictConfig
from tqdm import trange
from ultralytics import SAM, YOLO

from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.general_utils import measure_time, save_hydra_config
from conceptgraph.utils.model_utils import compute_clip_features_batched
from conceptgraph.utils.vis import save_video_detections, vis_result_fast


def load_detection_models(cfg):
    """Allow a bounded manifest batch to reuse the same frozen model weights."""
    checkpoint_dir = Path(__file__).resolve().parents[2] / "checkpoint"
    detection_model = YOLO(str(checkpoint_dir / "yolov8l-world.pt"))
    sam_predictor = SAM(str(checkpoint_dir / "mobile_sam.pt"))
    clip_checkpoint = (
        checkpoint_dir
        / "hf_models/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_model.safetensors"
    )
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained=str(clip_checkpoint)
    )
    clip_model = clip_model.to(cfg.device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    return detection_model, sam_predictor, clip_model, clip_preprocess, clip_tokenizer


def run_detections(cfg, models=None, *, skip_existing=False):
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.desired_height,
        desired_width=cfg.desired_width,
        device="cpu",
        dtype=torch.float,
    )
    detection_model, sam_predictor, clip_model, clip_preprocess, clip_tokenizer = (
        models or load_detection_models(cfg)
    )
    with open(cfg.classes_file) as handle:
        classes = [
            line.strip()
            for line in handle
            if line.strip() not in {"wall", "floor", "ceiling", ""}
        ]
    if getattr(detection_model, "_manifest_classes", None) != classes:
        detection_model.set_classes(classes)
        detection_model._manifest_classes = list(classes)
    variant = (
        cfg.exp_suffix.split("_outdoor_objects_v1")[0]
        if cfg.exp_suffix.endswith("_outdoor_objects_v1")
        else "streamlined_yolo"
    )
    classes_out_path = (
        Path(cfg.dataset_root) / cfg.scene_id / f"gsa_classes_{variant}.json"
    )
    if not classes_out_path.exists():
        classes_out_path.write_text(json.dumps(classes, ensure_ascii=False, indent=2))
    exp_out_path = Path(cfg.dataset_root) / cfg.scene_id / f"exp_{cfg.exp_suffix}"
    vis_folder_path, detections_folder_path = (
        exp_out_path / "vis",
        exp_out_path / "detections",
    )
    vis_folder_path.mkdir(exist_ok=True, parents=True)
    detections_folder_path.mkdir(exist_ok=True, parents=True)
    save_hydra_config(cfg, exp_out_path)
    for idx in trange(len(dataset)):
        color_path = dataset.color_paths[idx]
        vis_save_path = vis_folder_path / Path(color_path).name
        destination = detections_folder_path / (Path(color_path).stem + ".pkl.gz")
        if skip_existing and destination.exists():
            with gzip.open(destination, "rb") as handle:
                saved = pickle.load(handle)
            if saved["classes"] != classes or len(saved["xyxy"]) != len(saved["mask"]):
                raise ValueError(f"inconsistent existing detection: {destination}")
            continue
        result, annotated_image = detect_rgb_file(
            cfg, (detection_model, sam_predictor, clip_model, clip_preprocess, clip_tokenizer),
            color_path, classes,
        )
        if not cv2.imwrite(str(vis_save_path), annotated_image):
            raise OSError(f"failed to save image: {vis_save_path}")
        temporary = destination.with_suffix(".tmp")
        with gzip.open(temporary, "wb") as handle:
            pickle.dump(result, handle)
        temporary.replace(destination)
    if cfg.save_video:
        save_video_detections(exp_out_path)


def detect_rgb_file(cfg, models, color_path, classes):
    """录制回放与真实交互环境共用的逐帧YOLO/SAM/CLIP前端。"""
    detection_model, sam_predictor, clip_model, clip_preprocess, clip_tokenizer = models
    if getattr(detection_model, "_manifest_classes", None) != list(classes):
        detection_model.set_classes(list(classes))
        detection_model._manifest_classes = list(classes)
    image = cv2.imread(str(color_path))
    if image is None:
        raise ValueError(f"unreadable image: {color_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    prediction = detection_model.predict(str(color_path), conf=0.1, verbose=False)[
        0
    ]
    boxes = prediction.boxes.xyxy.clone()
    confidences = prediction.boxes.conf.cpu().numpy()
    class_ids = prediction.boxes.cls.cpu().numpy().astype(int)
    if len(boxes):
        masks = (
            sam_predictor.predict(str(color_path), bboxes=boxes.clone(), verbose=False)[0]
            .masks.data.cpu()
            .numpy()
        )
    else:
        # SAM cannot reshape a zero-box batch. Empty detections are evidence,
        # not a preprocessing failure or a reason to discard the frame.
        masks = np.zeros((0, *image.shape[:2]), dtype=bool)
    detections = sv.Detections(
        xyxy=boxes.cpu().numpy(),
        confidence=confidences,
        class_id=class_ids,
        mask=masks,
    )
    if len(boxes):
        crops, image_feats, text_feats = compute_clip_features_batched(
            image_rgb,
            detections,
            clip_model,
            clip_preprocess,
            clip_tokenizer,
            classes,
            cfg.device,
        )
        annotated_image, _ = vis_result_fast(image, detections, classes)
    else:
        dimension = int(clip_model.visual.output_dim)
        crops = []
        image_feats = np.empty((0, dimension), dtype=np.float32)
        text_feats = np.empty((0, dimension), dtype=np.float32)
        annotated_image = image
    result = {
        "xyxy": detections.xyxy,
        "confidence": detections.confidence,
        "class_id": detections.class_id,
        "mask": detections.mask,
        "classes": classes,
        "image_crops": crops,
        "image_feats": image_feats,
        "text_feats": text_feats,
        "frame_provenance": {
            "rgb_sha256": hashlib.sha256(Path(color_path).read_bytes()).hexdigest(),
            "image_size_wh": [int(image.shape[1]), int(image.shape[0])],
            "bbox_coordinate_frame": "original_rgb_pixels",
            "sam_prompt_tensor_isolated": True,
            "frontend_version": "completion-v1-preserve-boxes",
        },
    }
    return result, annotated_image


@hydra.main(
    version_base=None,
    config_path="../hydra_configs/",
    config_name="streamlined_detections",
)
def main(cfg: DictConfig):
    run_detections(cfg)


if __name__ == "__main__":
    measure_time(main)()
