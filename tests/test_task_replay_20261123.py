"""Minimal temporal adapter checks; synthetic fixtures are not task scores."""
import numpy as np
import pytest

from conceptgraph.navigation.semantic_tasks_20261123 import Entity
from conceptgraph.navigation.task_replay_20261123 import (
    EpisodeSession, ObjectRoadProvider, PastPoseMotion, ego_pose_from_camera, transform,
)


def test_calibration_inverse_recovers_ego_instead_of_optical_heading():
    ego = transform([10, 20, 0], [1, 0, 0, 0])
    calibration = {"translation": [1, 0, 1.5], "rotation": [0.5, -0.5, 0.5, -0.5]}
    camera = ego @ transform(calibration["translation"], calibration["rotation"])
    assert np.allclose(ego_pose_from_camera(camera, calibration), ego)


def test_past_motion_has_unknown_first_speed_and_never_uses_future():
    provider = PastPoseMotion()
    first = np.eye(4)
    ego, audit = provider.update(1, first, 1.01, "fixture")
    assert ego.get("speed_mps", 1.01)[0] is None and audit["past_interval_s"] is None
    second = np.eye(4)
    second[0, 3] = 2
    ego, audit = provider.update(1.5, second, 1.51, "fixture")
    assert ego.get("speed_mps", 1.51)[0] == 4 and audit["past_interval_s"] == [1, 1.5]
    with pytest.raises(ValueError):
        provider.update(2.1, second, 2, "fixture_future")
    with pytest.raises(ValueError):
        provider.update(1.5, second, 2, "duplicate")


def test_missing_road_and_track_id_do_not_become_lane_or_verified_identity():
    provider = ObjectRoadProvider(None, "missing_road")
    ego = Entity("ego", "ego", (0, 0))
    graph, coverage = provider.build([{"node_id": "track0", "category": "car", "center": [5, 0, 1],
                                       "timestamp_s": 1}], ego, 1, "fixture", "A_front")
    assert not coverage["road_input_available"] and not coverage["relations"]
    car = next(n for n in graph.entities if n.kind == "vehicle")
    assert car.get("continuous_identity_verified", 1)[0] is None
    assert car.get("length_m", 1)[0] is None


def test_one_issue_persists_monitor_across_frames_and_no_replay_completion():
    episode = {"episode_id": "fixture__pull_over", "scene": "fixture", "instruction": "靠边停车", "issue_time_s": 1}
    session = EpisodeSession(episode, "A_front")
    provider = ObjectRoadProvider(None, "missing_road")
    for frame in range(3):
        graph, _ = provider.build([], Entity("ego", "ego", (0, 0)), 1 + frame * 0.5, "fixture", "A_front")
        result = session.step(frame, graph)
        assert result["instruction_issue_count"] == 1
        assert result["monitor"]["status"] == "blocked_no_task_proposal"
        assert not result["parking_success_evaluated"] and not result["actual_vehicle_control"]
    with pytest.raises(ValueError):
        session.step(2, graph)
