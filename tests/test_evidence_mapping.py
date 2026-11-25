"""弱观测不能靠重复记录自证可靠，强观测与跨帧证据保留。"""
import numpy as np
import pytest

from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.evidence_mapper import confirmed_observation
from conceptgraph.completion.mapping_recall_diagnosis import box_iou, projected_box


def test_weak_detection_needs_distinct_frames_and_depth():
    obj = dict(conf=[0.12, 0.15], image_idx=[4, 4], depth_evidence=[dict(retained_pixels=9)])
    assert not confirmed_observation(obj)
    obj["image_idx"] = [4, 5]
    assert confirmed_observation(obj)
    obj["depth_evidence"] = [dict(retained_pixels=2)]
    assert not confirmed_observation(obj)


def test_strong_and_invalid_confidence_are_distinguished():
    assert confirmed_observation(dict(conf=[0.2], image_idx=[4]))
    assert not confirmed_observation(dict(conf=[float("nan")], image_idx=[4, 5]))


def test_projection_and_overlap_have_geometric_meaning():
    obj = dict(size_wlh=[2, 2, 2], center=[0, 0, 10], rotation_wxyz=[1, 0, 0, 0])
    box = projected_box(obj, np.eye(4), np.array([[100, 0, 100], [0, 100, 100], [0, 0, 1]]), 200, 200)
    np.testing.assert_allclose(box, [100 - 100 / 9, 100 - 100 / 9, 100 + 100 / 9, 100 + 100 / 9])
    assert box_iou(box, box) == pytest.approx(1)
    assert box_iou(box, np.array([0, 0, 10, 10])) == 0


def test_new_runs_use_new_layout_and_reject_escape():
    path = experiment_directory("nuscenes", "validation", "test_run")
    assert str(path).endswith("outputs/results/nuscenes/validation/iterations/test_run")
    with pytest.raises(ValueError):
        experiment_directory("nuscenes", "../test", "run")
