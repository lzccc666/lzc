"""RAM自动类别与固定类别Grounding DINO；同一MobileSAM/CLIP输出协议。"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import time
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import supervision as sv
import torch
from torchvision.ops import box_convert, nms

from conceptgraph.completion.vehicle_frontend import CLASSES, PROMPTS
from conceptgraph.project_paths import PROJECT_ROOT
from conceptgraph.utils.model_utils import compute_clip_features_batched


ASSETS = PROJECT_ROOT / "checkpoint/ram_dino_comparison_v1"
DINO_CONFIG = PROJECT_ROOT / "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
REMOVE_TAGS = {
    "room", "kitchen", "office", "house", "home", "building", "corner", "shadow",
    "carpet", "photo", "shade", "stall", "space", "aquarium", "apartment", "image",
    "city", "blue", "skylight", "hallway", "bureau", "modern", "salon", "doorway",
    "wall lamp", "wood floor", "wall", "floor", "ceiling",
}
# nuScenes的car包含van/SUV；只转换类别词，不把car door、bus stop或vehicle硬归类。
# 来源：https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_nuscenes.md
ALIASES = {
    "car": {"car", "cars", "taxi", "automobile", "sedan", "van", "minivan", "mini van", "suv", "jeep", "hatchback", "station wagon"},
    "truck": {"truck", "trucks", "lorry", "pickup truck", "delivery truck"},
    "bus": {"bus", "buses", "school bus"},
    "motorcycle": {"motorcycle", "motorcycles", "motorbike"},
    "bicycle": {"bicycle", "bicycles", "bike"},
    "pedestrian": {"person", "people", "pedestrian", "pedestrians", "man", "woman", "boy", "girl", "child"},
}


def canonical_category(phrase):
    phrase = " ".join(re.findall(r"[a-z]+", phrase.lower()))
    for name, aliases in ALIASES.items():
        # RAM可能给出person、man等同义标签，DINO一次同时选中二者仍是同一类别。
        pattern = "(?:" + "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True)) + ")"
        if re.fullmatch(pattern + "(?: " + pattern + ")*", phrase):
            return name
    return None


def ram_classes(tags):
    classes = list(dict.fromkeys(t.strip().lower() for t in tags.split("|") if t.strip()))
    # 与项目中ConceptGraphs generate_gsa_results.py的类别后处理保持一致。
    classes = [t for t in classes if not any(bad in t for bad in REMOVE_TAGS)]
    return classes + (["other item"] if "other item" not in classes else [])


def select_prompt_classes(scores, tokenized, classes):
    """按每个完整提示词的token分数选类别，不把混合短语丢弃或硬归第一类。"""
    offsets = tokenized["offset_mapping"]
    start, all_indices = 0, []
    for label in classes:
        end = start + len(label)
        indices = [i for i, (left, right) in enumerate(offsets[:scores.shape[1]])
                   if left < right and left < end and right > start]
        all_indices.append(indices)
        start = end + 2  # caption以'. '连接类别。
    values = torch.full((len(scores), len(classes)), -1.0, dtype=scores.dtype)
    for ci, indices in enumerate(all_indices):
        if indices:
            values[:, ci] = scores[:, indices].mean(dim=1)
    return values.argmax(dim=1).tolist(), values


def load_grounding_models(*, with_ram=True):
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    from ram import get_transform
    from ram.models import ram
    from transformers import BertConfig, BertModel

    cfg = SLConfig.fromfile(str(DINO_CONFIG))
    cfg.device = "cuda:0"
    cfg.bert_base_uncased_path = str(ASSETS / "bert-base-uncased")
    cfg.use_checkpoint = False
    cfg.use_transformer_ckpt = False
    # DINO检查点已包含完整BERT权重。仅从本地配置初始化结构，再严格核对完整加载，
    # 避免先下载另一个BERT权重、随后又被DINO权重全部覆盖。
    def bert_structure(encoder_type, local_path):
        if encoder_type != "bert-base-uncased":
            raise ValueError(encoder_type)
        return BertModel(BertConfig.from_pretrained(local_path, local_files_only=True))
    with patch("groundingdino.util.get_tokenlizer.get_pretrained_language_model", bert_structure):
        dino = build_model(cfg)
    state = torch.load(PROJECT_ROOT / "checkpoint/groundingdino_swint_ogc.pth", map_location="cpu")
    result = dino.load_state_dict(clean_state_dict(state["model"]), strict=False)
    if result.missing_keys:
        raise ValueError(f"Grounding DINO权重缺失：{result.missing_keys}")
    print("dino_unexpected_keys", result.unexpected_keys, flush=True)
    dino.eval().to("cuda:0")
    tagger = None
    if with_ram:
        tagger = ram(pretrained=str(ASSETS / "ram_swin_large_14m.pth"), image_size=384,
                     vit="swin_l", text_encoder_type=str(ASSETS / "bert-base-uncased"))
        tagger.eval().to("cuda:0")
    return dino, tagger, get_transform(image_size=384)


@torch.no_grad()
def detect_grounding_boxes(models, color_path, *, mode):
    from groundingdino.util.inference import Model
    from groundingdino.util.utils import get_phrases_from_posmap
    from ram import inference_ram

    dino, tagger, transform = models
    image = cv2.imread(str(color_path))
    if image is None:
        raise ValueError(f"无法读取图像：{color_path}")
    torch.cuda.synchronize()
    started = time.monotonic()
    if mode == "dino_fixed":
        tags, classes = None, list(PROMPTS)
    elif mode == "ram_dino":
        tensor = transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).unsqueeze(0).to("cuda:0")
        tags, _ = inference_ram(tensor, tagger)
        classes = ram_classes(tags)
    else:
        raise ValueError(mode)
    torch.cuda.synchronize()
    tagging_s = time.monotonic() - started
    caption = ". ".join(classes) + "."
    tokens = dino.tokenizer(caption, return_offsets_mapping=True)
    tensor = Model.preprocess_image(image).to("cuda:0")
    output = dino(tensor[None], captions=[caption])
    scores = output["pred_logits"].sigmoid()[0].cpu()
    boxes = output["pred_boxes"][0].cpu()
    selected = scores.max(dim=1).values > 0.1
    boxes, scores = boxes[selected], scores[selected]
    phrases = [get_phrases_from_posmap(s > 0.2, tokens, dino.tokenizer).replace(".", "").strip() for s in scores]
    selected_classes, prompt_scores = select_prompt_classes(scores, tokens, classes)
    decoded_categories = [canonical_category(classes[i]) for i in selected_classes]
    confidence = scores.max(dim=1).values
    height, width = image.shape[:2]
    boxes = box_convert(boxes * torch.tensor([width, height, width, height]), "cxcywh", "xyxy")
    if len(boxes):
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
        valid = ((boxes[:, 2:] - boxes[:, :2]) > 0).all(dim=1)
        valid_indices = torch.where(valid)[0]
        # 与原GSA检测后处理一致：全类别NMS，不额外为真值类别提供特殊处理。
        kept = valid_indices[nms(boxes[valid], confidence[valid], 0.5)]
    else:
        kept = torch.empty(0, dtype=torch.long)
    raw_phrases = [phrases[i] for i in kept.tolist()]
    known = [i for i in kept.tolist() if decoded_categories[i] is not None]
    categories = [decoded_categories[i] for i in known]
    torch.cuda.synchronize()
    total_s = time.monotonic() - started
    return dict(xyxy=boxes[known].numpy(), confidence=confidence[known].numpy(),
                class_id=np.asarray([CLASSES.index(c) for c in categories], dtype=int),
                classes=CLASSES, phrases=[phrases[i] for i in known], raw_phrases=raw_phrases,
                raw_kept_boxes=boxes[kept].numpy(), raw_kept_confidence=confidence[kept].numpy(),
                raw_selected_prompts=[classes[selected_classes[i]] for i in kept.tolist()],
                raw_prompt_scores=prompt_scores[kept].numpy(),
                ram_tags=tags, detector_classes=classes, caption=caption,
                prompt_token_count=len(tokens["input_ids"]), max_text_len=dino.max_text_len,
                class_decoding="highest_mean_full_prompt_token_score",
                transformed_size_hw=list(tensor.shape[-2:]), tagging_s=tagging_s, detection_s=total_s,
                raw_unknown_or_out_of_scope_count=len(kept) - len(known))


def segment_and_embed(common_models, color_path, boxes):
    """只接收模型框与原图；不接收真值或未来帧。"""
    segmenter, clip_model, preprocess, tokenizer = common_models
    image = cv2.imread(str(color_path))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    xyxy = boxes["xyxy"]
    if len(xyxy):
        masks = segmenter.predict(str(color_path), bboxes=torch.as_tensor(xyxy).clone(), verbose=False)[0].masks.data.cpu().numpy().astype(bool)
        detections = sv.Detections(xyxy=xyxy, confidence=boxes["confidence"], class_id=boxes["class_id"], mask=masks)
        crops, image_features, text_features = compute_clip_features_batched(
            rgb, detections, clip_model, preprocess, tokenizer, CLASSES, "cuda:0")
    else:
        masks = np.zeros((0, *rgb.shape[:2]), dtype=bool)
        crops = []
        dimension = int(clip_model.visual.output_dim)
        image_features = np.empty((0, dimension), dtype=np.float32)
        text_features = np.empty((0, dimension), dtype=np.float32)
    diagnostics = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in boxes.items()
                   if k not in {"xyxy", "confidence", "class_id", "classes"}}
    return dict(xyxy=xyxy, confidence=boxes["confidence"], class_id=boxes["class_id"], classes=CLASSES,
                mask=masks, image_crops=crops, image_feats=image_features, text_feats=text_features,
                frame_provenance=dict(rgb_sha256=hashlib.sha256(Path(color_path).read_bytes()).hexdigest(),
                                      image_size_wh=[int(rgb.shape[1]), int(rgb.shape[0])],
                                      bbox_coordinate_frame="original_rgb_pixels", sam_prompt_tensor_isolated=True,
                                      frontend_version="ram-dino-mobileSAM-CLIP-v1", diagnostics=diagnostics))
