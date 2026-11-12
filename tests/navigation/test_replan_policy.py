from __future__ import annotations

import numpy as np

from conceptgraph.navigation.replan_policy import ReplanPolicy


def evaluate(policy, frame, **events):
    return policy.evaluate(
        frame,
        np.asarray(events.pop("vehicle_xy", [0.0, 0.0])),
        has_locked_target=events.pop("has_locked_target", True),
        **events,
    )


def test_priority_and_same_frame_single_plan() -> None:
    policy = ReplanPolicy({"cooldown_frames": 0})
    decision = evaluate(
        policy,
        5,
        target_confirmed=True,
        path_blocked=True,
        target_changed=True,
    )
    duplicate = evaluate(policy, 5, path_blocked=True)

    assert decision.replanned and decision.reason == "path_blocked"
    assert any(
        item["trigger"] == "target_changed" and not item["accepted"]
        for item in decision.audits
    )
    assert not duplicate.replanned
    assert duplicate.audits[0]["reason"] == "already_planned_this_frame"


def test_cooldown_debounce_and_active_event_do_not_repeat() -> None:
    policy = ReplanPolicy(
        {
            "cooldown_frames": 2,
            "cooldown_bypass_triggers": ["target_confirmed"],
            "debounce_frames": {"path_blocked": 2, "vehicle_moved": 1},
        }
    )
    first = evaluate(policy, 0, target_confirmed=True)
    policy.record_plan(0, np.asarray([0.0, 0.0]))
    debounce = evaluate(policy, 1, path_blocked=True)
    cooldown = evaluate(policy, 2, path_blocked=True)
    accepted = evaluate(policy, 3, path_blocked=True)
    repeated = evaluate(policy, 4, path_blocked=True)

    assert first.replanned
    assert not debounce.replanned
    assert any(item["reason"] == "cooldown_active" for item in cooldown.audits)
    assert accepted.replanned and accepted.reason == "path_blocked"
    assert not repeated.replanned
    assert repeated.audits[0]["reason"] == "debounced_active_event"


def test_vehicle_periodic_manual_and_no_target_rejections() -> None:
    policy = ReplanPolicy(
        {
            "cooldown_frames": 0,
            "vehicle_moved_distance_m": 2.0,
            "periodic_interval_frames": 3,
        }
    )
    policy.record_plan(0, np.asarray([0.0, 0.0]))
    moved = evaluate(policy, 1, vehicle_xy=[2.1, 0.0])
    policy.record_plan(1, np.asarray([2.1, 0.0]))
    periodic = evaluate(policy, 4, vehicle_xy=[2.1, 0.0])
    policy.record_plan(4, np.asarray([2.1, 0.0]))
    policy.request_manual_replan()
    manual = evaluate(policy, 5, vehicle_xy=[2.1, 0.0])
    policy.request_manual_replan()
    rejected = evaluate(policy, 6, has_locked_target=False)

    assert moved.reason == "vehicle_moved"
    assert periodic.reason == "periodic_interval"
    assert manual.reason == "manual_request"
    assert not rejected.replanned
    assert rejected.audits[0]["reason"] == "no_locked_target"
