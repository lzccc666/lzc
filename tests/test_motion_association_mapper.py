"""检查因果运动预测与相邻同类对象的可区分性。"""
import numpy as np
import pytest

from conceptgraph.slam.motion_association_mapper import extrapolated_center, motion_scores
from conceptgraph.slam.streamlined_mapping import StreamlinedMapper


def obj(x, *, history=None, times=None, cls=0):
    value = dict(bbox={"center": [x, 0, 1]}, class_id=[cls])
    if history is not None:
        value["center_history"] = [[v, 0, 1] for v in history]
        value["timestamp_history_s"] = times
    return value


def test_predicts_only_from_past_positions_with_real_time():
    center, age, reason = extrapolated_center(obj(1, history=[0, 1], times=[0, .5]), 1.)
    np.testing.assert_allclose(center, [2, 0, 1])
    assert age == .5 and reason == "past_observations_only"


@pytest.mark.parametrize("times, now", [([0, 0], 1), ([0, 1], .5), ([0, 1], 5)])
def test_bad_or_expired_history_does_not_predict(times, now):
    center, _, reason = extrapolated_center(obj(1, history=[0, 1], times=times), now)
    np.testing.assert_allclose(center, [1, 0, 1])
    assert reason != "past_observations_only"


def test_rejects_implausible_speed_and_does_not_mutate_history():
    value = obj(30, history=[0, 30], times=[0, .5])
    center, _, reason = extrapolated_center(value, 1.)
    assert reason == "implausible_velocity" and center[0] == 30
    assert value["center_history"] == [[0, 0, 1], [30, 0, 1]]


def test_spatial_gate_prevents_distant_same_class_or_wrong_class_merging():
    scores = motion_scores([obj(0), obj(6), obj(0, cls=1)], [obj(0)],
                           np.ones((3, 1)), np.ones((3, 1)), 1.)
    assert scores[0, 0] == pytest.approx(1)
    assert np.isneginf(scores[1:, 0]).all()


def test_motion_can_link_disjoint_boxes_at_the_predicted_position():
    scores = motion_scores([obj(2)], [obj(1, history=[0, 1], times=[0, .5])],
                           np.zeros((1, 1)), np.ones((1, 1)), 1.)
    assert scores[0, 0] > .35


def test_default_mapper_hook_preserves_legacy_scores():
    scores = object()
    assert StreamlinedMapper.refine_association_scores(None, None, None, None, None, scores) is scores
