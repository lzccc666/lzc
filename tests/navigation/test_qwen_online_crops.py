from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from conceptgraph.scripts.run_incremental_semantic_navigation import (
    mapping_objects_to_nodes,
    load_camera_timestamps,
)


class _BBox:
    def get_center(self):
        return np.asarray([3.0, 4.0, 0.5])

    def get_extent(self):
        return np.asarray([2.0, 1.5, 1.0])


def test_mapping_object_crop_is_isolated_and_attached_to_node(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (100, 80), "white").save(source)
    crop_dir = tmp_path / "crops"
    objects = [
        {
            "bbox": _BBox(),
            "class_id": [0, 0, 0],
            "num_detections": 3,
            "frame_history": [0, 1, 2],
            "image_idx": [0, 2],
            "color_path": [str(source), str(source)],
            "xyxy": [[0, 0, 5, 5], [20, 10, 60, 50]],
            "inst_color": [0.1, 0.2, 0.3],
        }
    ]

    nodes = mapping_objects_to_nodes(
        objects,
        ["car"],
        2,
        semantic_crop_dir=crop_dir,
    )

    crop = Path(nodes[0].metadata["crop_paths"][0])
    assert crop.is_file()
    assert crop.parent == crop_dir
    assert crop != source
    assert nodes[0].metadata["crop_frame_idx"] == 2
    assert nodes[0].metadata["object_identity"].startswith("inst_color:")
    with Image.open(crop) as image:
        assert image.width >= 40 and image.height >= 40


def test_mapping_object_without_current_observation_has_no_crop(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (40, 40), "white").save(source)
    nodes = mapping_objects_to_nodes(
        [
            {
                "bbox": _BBox(),
                "class_id": [0],
                "image_idx": [1],
                "color_path": [str(source)],
                "xyxy": [[2, 2, 20, 20]],
            }
        ],
        ["car"],
        2,
        semantic_crop_dir=tmp_path / "crops",
    )

    assert "crop_paths" not in nodes[0].metadata


def test_stable_id_survives_object_list_compaction_and_keeps_observation_age():
    remaining = {"bbox": _BBox(), "class_id": [0], "stable_object_id": "track-00000009",
        "last_seen_frame": 3, "timestamp_s": 1.5}
    before = mapping_objects_to_nodes([{"bbox": _BBox(), "class_id": [0],
        "stable_object_id": "track-00000001"}, remaining], ["car"], 4)
    after = mapping_objects_to_nodes([remaining], ["car"], 5)
    assert 9 in before and set(after) == {9}
    assert after[9].metadata["object_identity"] == "track-00000009"
    assert after[9].metadata["last_seen_frame"] == 3
    assert after[9].metadata["timestamp_s"] == 1.5


def test_camera_timestamp_loader_uses_camera_acquisition_time(tmp_path):
    import json
    import pytest
    p=tmp_path/'frame_alignment.json'
    p.write_text(json.dumps([{'timestamp':9000000,'camera_source':'cam__1000000.jpg'},
        {'timestamp':9900000,'camera_timestamp':1500000}]))
    assert load_camera_timestamps(tmp_path,2)==[1.0,1.5]
    with pytest.raises(ValueError):
        load_camera_timestamps(tmp_path,3)
