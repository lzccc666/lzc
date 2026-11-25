import copy

import numpy as np
import pytest

from conceptgraph.slam.observation_tracker import CausalObservationTracker, ObservationTrackingConfig


def observation(x, category="car"):
    return dict(center=[x, 0, 1], category=category, node_id="input-not-an-identity")


def test_maintains_identity_and_predicts_through_a_short_gap_using_past_only():
    tracker = CausalObservationTracker()
    a = tracker.update([observation(0)], 0)[0]
    b = tracker.update([observation(1)], .5)[0]
    c = tracker.update([], 1)[0]
    assert a["node_id"] == b["node_id"] == c["node_id"]
    assert c["center"][0] == pytest.approx(2)
    assert tracker.update([], 2.1) == []


def test_assignment_is_one_to_one_and_class_gated():
    tracker = CausalObservationTracker()
    a = tracker.update([observation(0), observation(1)], 0)
    b = tracker.update([observation(.1), observation(.9), observation(.1, "bus")], .5)
    assert len({x["node_id"] for x in b}) == 3
    assert {x["node_id"] for x in a} <= {x["node_id"] for x in b}


def test_rejects_bad_frame_without_changing_state_and_does_not_alias_input():
    tracker = CausalObservationTracker()
    inputs = [observation(0)]
    tracker.update(inputs, 0)
    inputs[0]["center"][0] = 99
    before = copy.deepcopy(tracker.tracks[0]["object"])
    with pytest.raises(ValueError):
        tracker.update([observation(float("nan"))], .5)
    assert tracker.last_timestamp == 0 and tracker.tracks[0]["object"] == before
    with pytest.raises(ValueError):
        tracker.update([], 0)


def test_long_gap_does_not_extrapolate_and_outputs_are_deterministic():
    runs = []
    for _ in range(2):
        tracker = CausalObservationTracker(ObservationTrackingConfig(prediction_horizon_s=.5))
        tracker.update([observation(0)], 0)
        tracker.update([observation(1)], .5)
        runs.append(tracker.update([], 1.5))
    assert runs[0] == runs[1]
    np.testing.assert_allclose(runs[0][0]["center"], [1, 0, 1])
