from __future__ import annotations

import copy

import numpy as np

from conceptgraph.navigation.online import SnapshotNodeTracker


def raw_object(color, center, class_id=0, detections=1):
    center = np.asarray(center, dtype=float)
    offsets = np.asarray(
        [
            [-1, -1, -0.5],
            [-1, 1, -0.5],
            [1, -1, -0.5],
            [1, 1, -0.5],
            [-1, -1, 0.5],
            [-1, 1, 0.5],
            [1, -1, 0.5],
            [1, 1, 0.5],
        ]
    )
    return {
        "bbox_np": center + offsets,
        "class_id": [class_id],
        "inst_color": list(color),
        "num_detections": detections,
    }


def test_snapshot_tracker_stable_ids_changes_and_no_input_mutation() -> None:
    tracker = SnapshotNodeTracker(["car", "traffic sign"])
    first_raw = [raw_object([0.1, 0.2, 0.3], [2, 3, 0])]
    untouched = copy.deepcopy(first_raw)
    first, first_changed = tracker.update(first_raw, 3)
    second_raw = [
        raw_object([0.1, 0.2, 0.3], [2.05, 3, 0], detections=2),
        raw_object([0.5, 0.6, 0.7], [8, 9, 0], class_id=1),
    ]
    second, second_changed = tracker.update(second_raw, 4)

    assert first_changed == [0]
    assert second_changed == [1]
    assert second[0].node_id == first[0].node_id
    assert second[1].object_tag == "traffic sign"
    assert np.array_equal(first_raw[0]["bbox_np"], untouched[0]["bbox_np"])


def test_snapshot_tracker_skips_corrupt_object_and_continues() -> None:
    tracker = SnapshotNodeTracker(["car"])
    nodes, changed = tracker.update(
        [
            {"bbox_np": np.asarray([1.0, 2.0]), "class_id": [0]},
            raw_object([0.1, 0.2, 0.3], [2, 3, 0]),
        ],
        1,
    )
    assert list(nodes) == [0]
    assert changed == [0]
