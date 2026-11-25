"""自动类别不能因包含子串就被当作官方评价对象。"""
import torch
import numpy as np
from PIL import Image
from types import SimpleNamespace

from conceptgraph.completion.ram_dino_frontend import canonical_category, ram_classes, select_prompt_classes, segment_and_embed


def test_category_mapping_rejects_object_parts_and_ambiguous_vehicles():
    for phrase in ["car door", "bus stop", "truck driver", "motorcycle helmet", "vehicle", "car truck", ""]:
        assert canonical_category(phrase) is None
    assert canonical_category("school bus") == "bus"
    assert canonical_category("person man") == "pedestrian"
    assert canonical_category("motorbike") == "motorcycle"
    assert canonical_category("van") == "car"


def test_ram_categories_do_not_silently_inject_missing_traffic_classes():
    classes = ram_classes("road | truck | bus stop | road | building")
    assert classes == ["road", "truck", "bus stop", "other item"]
    assert "car" not in classes
    assert "person" not in classes


def test_mixed_phrase_uses_scores_instead_of_first_word_or_dropping_detection():
    tokenized = dict(offset_mapping=[(0, 0), (0, 3), (3, 4), (5, 10), (10, 11), (0, 0)])
    scores = torch.tensor([[0.0, 0.6, 0.0, 0.8, 0.0, 0.0], [0.0, 0.9, 0.0, 0.3, 0.0, 0.0]])
    indices, _ = select_prompt_classes(scores, tokenized, ["car", "truck"])
    assert indices == [1, 0]


def test_truncated_prompt_cannot_win_with_no_visible_tokens():
    tokenized = dict(offset_mapping=[(0, 0), (0, 3)])
    indices, _ = select_prompt_classes(torch.tensor([[0.0, 0.3]]), tokenized, ["car", "truck"])
    assert indices == [0]


def test_frame_diagnostics_do_not_break_mapper_filtering(tmp_path, monkeypatch):
    from conceptgraph.slam.utils import filter_gobs

    rgb = tmp_path / "frame.png"
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(rgb)
    segmenter = SimpleNamespace(predict=lambda *args, **kwargs: [SimpleNamespace(
        masks=SimpleNamespace(data=torch.ones((1, 10, 10))))])
    features = np.ones((1, 2))
    monkeypatch.setattr("conceptgraph.completion.ram_dino_frontend.compute_clip_features_batched",
                        lambda *args: ([Image.open(rgb)], features, features))
    boxes = dict(xyxy=np.array([[1., 1., 9., 9.]]), confidence=np.array([0.8]),
                 class_id=np.array([0]), classes=["car"], ram_tags=None, detection_s=0.5)
    observation = segment_and_embed((segmenter, None, None, None), rgb, boxes)
    cfg = SimpleNamespace(mask_area_threshold=1, skip_bg=True, max_bbox_area_ratio=1., mask_conf_threshold=0.2)
    filtered = filter_gobs(cfg, observation, np.zeros((10, 10, 3)))
    assert len(filtered["xyxy"]) == 1
    assert filtered["frame_provenance"]["diagnostics"]["ram_tags"] is None
