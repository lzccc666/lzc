from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from conceptgraph.navigation.online import OnlineSemanticNavigator
from conceptgraph.navigation.state_machine import NavigationStateMachine
from conceptgraph.navigation.types import SceneGraphNode

from .conftest import make_grid


def candidate(node_id: int, score: float) -> tuple[int, float]:
    return node_id, score


def test_confirmation_lock_hysteresis_and_short_disappearance() -> None:
    machine = NavigationStateMachine(
        {
            "confirmation_observations": 3,
            "minimum_persistent_frames": 3,
            "locked_missing_tolerance_frames": 1,
            "target_lost_after_frames": 3,
        }
    )
    first = machine.observe(0, [candidate(1, 0.8)])
    second = machine.observe(1, [candidate(1, 0.8)])
    locked = machine.observe(2, [candidate(1, 0.8)])
    missing = machine.observe(3, [])
    recovered = machine.observe(4, [candidate(1, 0.81)])

    assert first.state_after == "PROVISIONAL"
    assert second.state_after == "PROVISIONAL"
    assert locked.target_confirmed and locked.state_after == "LOCKED"
    assert missing.transition_reason == "locked_target_temporarily_missing"
    assert recovered.locked_target_id == 1


def test_better_target_requires_margin_and_persistence_then_switches() -> None:
    machine = NavigationStateMachine(
        {
            "confirmation_observations": 1,
            "minimum_persistent_frames": 1,
            "better_target_score_margin": 0.1,
            "better_target_persistence_frames": 3,
        }
    )
    machine.observe(0, [candidate(1, 0.7)])
    machine.planning_succeeded()
    jitter = machine.observe(1, [candidate(2, 0.75), candidate(1, 0.7)])
    machine.observe(2, [candidate(2, 0.85), candidate(1, 0.7)])
    machine.observe(3, [candidate(2, 0.85), candidate(1, 0.7)])
    switched = machine.observe(4, [candidate(2, 0.85), candidate(1, 0.7)])

    assert jitter.locked_target_id == 1
    assert switched.better_target_persisted
    assert switched.locked_target_id == 2


def test_long_target_loss_has_explicit_searching_reason() -> None:
    machine = NavigationStateMachine(
        {
            "confirmation_observations": 1,
            "minimum_persistent_frames": 1,
            "locked_missing_tolerance_frames": 1,
            "target_lost_after_frames": 3,
        }
    )
    machine.observe(0, [candidate(1, 0.8)])
    machine.planning_succeeded()
    machine.observe(1, [])
    machine.observe(2, [])
    lost = machine.observe(3, [])

    assert lost.target_lost
    assert lost.transition_reason == "locked_target_lost"
    assert lost.state_after == "SEARCHING"


def test_long_target_loss_can_be_configured_as_unreachable() -> None:
    machine = NavigationStateMachine(
        {
            "confirmation_observations": 1,
            "minimum_persistent_frames": 1,
            "locked_missing_tolerance_frames": 0,
            "target_lost_after_frames": 2,
            "target_loss_state": "UNREACHABLE",
        }
    )
    machine.observe(0, [candidate(1, 0.8)])
    machine.planning_succeeded()
    machine.observe(1, [])
    lost = machine.observe(2, [])

    assert lost.target_lost
    assert lost.state_after == "UNREACHABLE"


def navigation_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        default_near_distance_m=8.0,
        lambda_text=0.55,
        lambda_relation=0.30,
        lambda_distance_prior=0.25,
        relation_edge_minimum_confidence=0.0,
        start_search_radius_m=3.0,
        goal_search_radius_m=5.0,
        allow_goal_fallback=False,
        semantic_cost_weight=0.0,
        clearance_cost_weight=0.0,
        path_smoothing_iterations=1,
        max_candidates=3,
        grid_resolution=1.0,
        grid_padding_m=5.0,
        robot_radius_m=0.0,
        object_bbox_obstacle_scale=0.2,
        max_object_obstacle_radius_m=1.0,
        ignore_dynamic_obstacles=False,
        obstacle_profile="test",
        bev_map_path=None,
        target_selection=SimpleNamespace(
            mode="joint_utility",
            top_k=3,
            goal_candidates_per_object={"default": 8},
            class_approach_distance_m={"default": 1.0, "car": 1.0},
            min_clearance_m={"default": 0.0, "car": 0.0},
            max_fallback_distance_m={"default": 2.0, "car": 2.0},
            allow_global_fallback=False,
            utility_weights={
                "semantic_score": 0.35,
                "relation_score": 0.15,
                "path_cost": 0.25,
                "clearance_score": 0.10,
                "fallback_penalty": 0.10,
                "dynamic_risk": 0.05,
            },
            dynamic_class_penalties={"default": 0.0},
            deterministic_tie_break="semantic_rank_node_id_goal_rc",
        ),
    )


def node(node_id: int = 1, center=(16.0, 16.0), tag="car") -> SceneGraphNode:
    bbox_center = np.asarray([center[0], center[1], 0.0])
    bbox_extent = np.asarray([2.0, 2.0, 1.0])
    return SceneGraphNode(
        node_id=node_id,
        object_tag=tag,
        possible_tags=[tag],
        caption=tag,
        bbox_center=bbox_center,
        bbox_extent=bbox_extent,
        metadata={
            "num_detections": 4,
            "bbox_center": bbox_center.tolist(),
            "bbox_extent": bbox_extent.tolist(),
        },
    )


def make_navigator(delegate=None) -> OnlineSemanticNavigator:
    return OnlineSemanticNavigator(
        navigation_cfg(),
        semantic_delegate=delegate,
        base_occupancy_grid=make_grid(shape=(36, 36)),
        state_machine_config={
            "confirmation_observations": 2,
            "minimum_persistent_frames": 2,
            "confirmation_confidence": 0.1,
        },
        replan_config={
            "cooldown_frames": 0,
            "vehicle_moved_distance_m": 3.0,
            "periodic_interval_frames": 20,
        },
        relation_config={"minimum_confidence": 0.0},
    )


def test_online_plans_only_after_lock_and_duplicate_is_idempotent() -> None:
    navigator = make_navigator()
    navigator.set_query("前往小汽车附近")
    objects = {1: node()}
    first = navigator.update(0, [2.0, 2.0, 0.0], objects, [1])
    second = navigator.update(1, [2.0, 2.0, 0.0], objects, [])
    duplicate = navigator.update(1, [99.0, 99.0, 0.0], {}, [])
    out_of_order = navigator.update(0, [2.0, 2.0, 0.0], objects, [])
    navigator.close()

    assert first.state_after == "PROVISIONAL" and len(first.path_world) == 0
    assert second.replanned and second.replan_reason == "target_confirmed"
    assert second.state_after == "TRACKING" and len(second.path_world) > 0
    assert duplicate.to_dict() == second.to_dict()
    assert out_of_order.failure.code == "OUT_OF_ORDER_FRAME"


def test_path_block_vehicle_periodic_manual_and_exception_recovery() -> None:
    navigator = make_navigator()
    navigator.set_query("前往小汽车附近")
    objects = {1: node()}
    navigator.update(0, [2.0, 2.0, 0.0], objects, [1])
    planned = navigator.update(1, [2.0, 2.0, 0.0], objects, [])
    blocked = navigator._occupancy_grid.occupancy.copy()
    block_rc = navigator._path_grid[len(navigator._path_grid) // 2]
    blocked[block_rc] = 1
    blocked_update = navigator.update(
        2,
        [2.0, 2.0, 0.0],
        objects,
        [],
        {"occupancy": blocked, "source": "fault_injection"},
    )
    recovered = navigator.update(3, [2.0, 2.0, 0.0], objects, [], navigator._base_occupancy_grid)
    moved = navigator.update(4, [6.0, 2.0, 0.0], objects, [])
    failure = navigator.update(5, [float("nan"), 2.0, 0.0], objects, [])
    after_failure = navigator.update(6, [6.0, 2.0, 0.0], objects, [])
    navigator.request_replan()
    manual = navigator.update(7, [6.0, 2.0, 0.0], objects, [])
    navigator.close()

    assert planned.replanned
    assert blocked_update.replan_reason == "path_blocked"
    assert recovered.failure is None
    assert moved.replan_reason == "vehicle_moved"
    assert failure.failure.code == "FRAME_PROCESSING_EXCEPTION"
    assert failure.state_after == moved.state_after
    assert after_failure.failure is None
    assert manual.replan_reason == "manual_request"


def test_second_blocked_path_is_cleared_when_active_event_is_debounced() -> None:
    navigator = make_navigator()
    navigator.set_query("前往小汽车附近")
    objects = {1: node()}
    pose = [2.0, 2.0, 0.0]
    try:
        navigator.update(0, pose, objects, [1])
        navigator.update(1, pose, objects, [])
        first_obstacle = navigator._occupancy_grid.occupancy.copy()
        first_obstacle[navigator._path_grid[-1]] = 1
        replanned = navigator.update(2, pose, objects, [], {"occupancy": first_obstacle})
        assert replanned.replanned and len(replanned.path_world)
        second_obstacle = navigator._occupancy_grid.occupancy.copy()
        second_obstacle[navigator._path_grid[-1]] = 1
        deferred = navigator.update(3, pose, objects, [], {"occupancy": second_obstacle})
        assert not deferred.replanned
        assert any(a["reason"] == "debounced_active_event" for a in deferred.replan_audit)
        assert deferred.occupancy_audit["unsafe_path_invalidated"]
        assert len(deferred.path_world) == 0
        assert not navigator._path_grid
        assert deferred.failure.code == "PATH_BLOCKED_REPLAN_DEFERRED"
        navigator.request_replan()
        recovered = navigator.update(4, pose, objects, [], navigator._base_occupancy_grid)
        assert recovered.replanned and recovered.failure is None
        assert len(recovered.path_world)
    finally:
        navigator.close()


class FakeConfig:
    enabled = True
    use_qwen_query_parse = True
    use_qwen_object_attributes = False
    use_qwen_rerank = True


class FakeAsyncDelegate:
    config = FakeConfig()

    def __init__(self) -> None:
        self.closed = False

    async def parse_query_async(self, *_args, **_kwargs):
        await asyncio.sleep(0.08)
        return {
            "used_qwen": False,
            "fallback_used": True,
            "failure_code": "BACKEND_TIMEOUT",
            "cache": {"status": "miss"},
        }

    async def rerank_candidates_async(self, *_args, **_kwargs):
        await asyncio.sleep(0.08)
        return {
            "used_qwen": False,
            "fallback_used": True,
            "failure_code": "BACKEND_TIMEOUT",
            "cache": {"status": "miss"},
        }

    def close(self, *, wait=True):
        self.closed = wait


class CacheHitDelegate(FakeAsyncDelegate):
    async def parse_query_async(self, *_args, **_kwargs):
        return {
            "used_qwen": False,
            "fallback_used": True,
            "failure_code": None,
            "cache": {"status": "hit_memory"},
        }


class StaleRerankDelegate(FakeAsyncDelegate):
    async def parse_query_async(self, *_args, **_kwargs):
        return {
            "used_qwen": False,
            "fallback_used": True,
            "failure_code": None,
            "cache": {"status": "miss"},
        }

    async def rerank_candidates_async(self, *_args, **_kwargs):
        await asyncio.sleep(0.08)
        return {
            "used_qwen": True,
            "fallback_used": False,
            "failure_code": None,
            "normalized_output": {
                "selected_node_id": 1,
                "ranking": [1],
                "confidence": 0.9,
            },
            "cache": {"status": "miss"},
        }


class FailingDelegate(FakeAsyncDelegate):
    async def parse_query_async(self, *_args, **_kwargs):
        raise RuntimeError("injected Qwen failure")


def test_qwen_async_is_nonblocking_and_close_leaves_no_tasks() -> None:
    delegate = FakeAsyncDelegate()
    navigator = make_navigator(delegate)
    started = time.perf_counter()
    navigator.set_query("前往小汽车附近")
    elapsed = time.perf_counter() - started
    assert elapsed < 0.05
    assert navigator.pending_qwen_count == 1
    time.sleep(0.1)
    update = navigator.update(0, [2.0, 2.0, 0.0], {1: node()}, [1])
    navigator.close()

    statuses = [event["status"] for event in update.qwen_audit["events"]]
    assert "completed" in statuses
    assert update.qwen_audit["events"][0]["failure_code"] == "BACKEND_TIMEOUT"
    assert navigator.pending_qwen_count == 0
    assert delegate.closed


def test_frame_exception_rolls_back_all_incremental_memory() -> None:
    navigator = make_navigator()
    navigator.set_query("前往小汽车附近")
    navigator.update(0, [2.0, 2.0, 0.0], {1: node()}, [1])
    navigator.update(1, [2.0, 2.0, 0.0], {1: node()}, [])
    object_version = navigator._object_version
    edge_version = navigator._edge_version
    relation_node_ids = set(navigator.relation_graph.nodes)
    trajectory = [point.copy() for point in navigator._trajectory]

    failure = navigator.update(
        2,
        [3.0, 2.0, 0.0],
        {1: node(), 2: {"node_id": 2, "object_tag": "car"}},
        [2],
    )

    assert failure.failure.code == "FRAME_PROCESSING_EXCEPTION"
    assert navigator._object_version == object_version
    assert navigator._edge_version == edge_version
    assert set(navigator.relation_graph.nodes) == relation_node_ids
    assert len(navigator._trajectory) == len(trajectory)
    assert all(
        np.array_equal(actual, expected)
        for actual, expected in zip(navigator._trajectory, trajectory)
    )
    assert navigator.update(3, [3.0, 2.0, 0.0], {1: node()}, []).failure is None
    navigator.close()


def test_qwen_cache_hit_is_audited() -> None:
    navigator = make_navigator(CacheHitDelegate())
    navigator.set_query("前往小汽车附近")
    time.sleep(0.02)
    update = navigator.update(0, [2.0, 2.0, 0.0], {1: node()}, [1])
    navigator.close()

    assert update.qwen_audit["totals"]["cache_hit"] == 1
    assert update.qwen_audit["events"][0]["cache_status"] == "hit_memory"


def test_late_rerank_is_discarded_when_verified_candidate_disappears() -> None:
    navigator = make_navigator(StaleRerankDelegate())
    navigator.set_query("前往小汽车附近")
    time.sleep(0.02)
    navigator.update(0, [2.0, 2.0, 0.0], {1: node()}, [1])
    planned = navigator.update(1, [2.0, 2.0, 0.0], {1: node()}, [])
    assert planned.replanned and navigator.pending_qwen_count == 1
    navigator.update(2, [2.0, 2.0, 0.0], {2: node(2, center=(17.0, 16.0))}, [1, 2])
    time.sleep(0.1)
    audited = navigator.update(3, [2.0, 2.0, 0.0], {2: node(2, center=(17.0, 16.0))}, [])
    navigator.close()

    rerank = [
        event for event in audited.qwen_audit["events"] if event["task"] == "candidate_rerank"
    ]
    assert rerank and not rerank[0]["applied"]
    assert rerank[0]["diagnostic"] == "recommended_candidate_no_longer_present"
    assert audited.qwen_audit["active_rerank"] is None


class AttributeConfig:
    enabled = True
    use_qwen_query_parse = False
    use_qwen_object_attributes = True
    use_qwen_rerank = False


class AttributeDelegate:
    config = AttributeConfig()

    def __init__(self) -> None:
        self.calls = 0

    async def describe_object_async(self, *_args, **_kwargs):
        self.calls += 1
        return {
            "used_qwen": True,
            "fallback_used": False,
            "failure_code": None,
            "normalized_output": {
                "summary": "red sedan",
                "object_tag": "car",
                "possible_tags": ["car", "sedan"],
                "colors": ["red"],
                "body_type": "sedan",
                "motion_state": "static",
                "confidence": 0.94,
            },
            "cache": {"status": "miss"},
        }

    def close(self, *, wait=True):
        return None


def test_object_attributes_are_merged_and_enable_color_query(tmp_path: Path) -> None:
    from PIL import Image

    crop = tmp_path / "crop.png"
    Image.new("RGB", (32, 24), "red").save(crop)
    tracked = node()
    tracked.metadata.update(
        {
            "crop_paths": [str(crop)],
            "object_identity": "track-1",
            "num_detections": 4,
        }
    )
    delegate = AttributeDelegate()
    navigator = make_navigator(delegate)
    navigator.set_query("前往红色小汽车附近")
    first = navigator.update(0, [2.0, 2.0, 0.0], {1: tracked}, [1])
    time.sleep(0.03)
    second = navigator.update(1, [2.0, 2.0, 0.0], {1: tracked}, [])
    navigator.close()

    assert first.candidate_ids == []
    assert second.candidate_ids == [1]
    assert navigator._nodes[1].metadata["attributes"]["colors"] == ["red"]
    assert navigator._nodes[1].caption == "red sedan"
    assert delegate.calls == 1
    applied = [
        event
        for event in [*first.qwen_audit["events"], *second.qwen_audit["events"]]
        if event["task"] == "object_attributes"
    ]
    assert applied and applied[0]["applied"]


def test_crop_motion_advisory_cannot_overwrite_geometry_or_unknown():
    navigator = make_navigator(AttributeDelegate())
    try:
        for observed in ["static", "dynamic", "unknown"]:
            tracked = node()
            tracked.motion_state = observed
            navigator._qwen_node_attributes[1] = {
                "motion_state": "dynamic" if observed == "static" else "static",
                "colors": [],
            }
            navigator._apply_qwen_attributes({1: tracked})
            assert tracked.motion_state == observed
            assert "qwen_motion_advisory" in tracked.metadata
    finally:
        navigator.close()


class RerankConfig:
    enabled = True
    use_qwen_query_parse = False
    use_qwen_object_attributes = False
    use_qwen_rerank = True


class PreferSecondDelegate:
    config = RerankConfig()

    async def rerank_candidates_async(self, _query, candidates, *_args, **_kwargs):
        ids = [int(item["node_id"]) for item in candidates]
        selected = ids[-1]
        ranking = [selected, *[item for item in ids if item != selected]]
        return {
            "used_qwen": True,
            "fallback_used": False,
            "failure_code": None,
            "normalized_output": {
                "selected_node_id": selected,
                "ranking": ranking,
                "confidence": 0.91,
            },
            "cache": {"status": "miss"},
        }

    def close(self, *, wait=True):
        return None


class NoMatchDelegate(PreferSecondDelegate):
    async def rerank_candidates_async(self, _query, candidates, *_args, **_kwargs):
        return dict(used_qwen=True, fallback_used=False, failure_code=None,
            normalized_output=dict(selected_node_id=None, no_match=True,
                ranking=[int(item['node_id']) for item in candidates], confidence=.95),
            cache=dict(status='miss'))


def test_explicit_model_no_match_invalidates_existing_lock_and_path():
    cfg = navigation_cfg()
    cfg.target_selection = vars(cfg.target_selection)
    navigator = OnlineSemanticNavigator(cfg, semantic_delegate=NoMatchDelegate(),
        base_occupancy_grid=make_grid(shape=(36, 36)),
        state_machine_config=dict(confirmation_observations=1, minimum_persistent_frames=1,
            confirmation_confidence=.1),
        replan_config=dict(cooldown_frames=0, periodic_interval_frames=20),
        relation_config=dict(minimum_confidence=0.))
    try:
        navigator.set_query('前往小汽车附近')
        objects = {1: node(1, center=(12., 12.)), 2: node(2, center=(24., 20.))}
        first = navigator.update(0, [2., 2., 0.], objects, [1, 2])
        assert first.selected_node_id == 1 and len(first.path_world)
        # 明确等真实future完成，不用固定30ms假定后台已调度。
        # 首帧末尾可能已经接收结果；接收事件属于实际接收帧。
        for pending in list(navigator._pending_qwen.values()):
            pending.future.result(timeout=2.)
        refused = navigator.update(1, [2., 2., 0.], objects, [])
        assert refused.selected_node_id is None
        assert refused.candidate_ids == [] and len(refused.path_world) == 0
        assert refused.qwen_audit['active_rerank']['no_match'] is True
        received_events = first.qwen_audit['events'] + refused.qwen_audit['events']
        assert any(e.get('diagnostic') == 'verified_no_match_activated' for e in received_events)
        # 相同候选的下一帧不能靠旧路径或迟滞再次锁定。
        later = navigator.update(2, [2., 2., 0.], objects, [])
        assert later.selected_node_id is None and len(later.path_world) == 0
        # 候选集合显著变化后，拒绝证据过期，重新走正常确认流程。
        renewed = navigator._apply_qwen_rerank([(3, .9)])
        assert renewed == [(3, .9)]
        assert navigator._active_qwen_rerank is None
    finally:
        navigator.close()


def test_verified_rerank_switches_only_through_state_machine_and_replans() -> None:
    cfg = navigation_cfg()
    cfg.target_selection = vars(cfg.target_selection)
    navigator = OnlineSemanticNavigator(
        cfg,
        semantic_delegate=PreferSecondDelegate(),
        base_occupancy_grid=make_grid(shape=(36, 36)),
        state_machine_config={
            "confirmation_observations": 1,
            "minimum_persistent_frames": 1,
            "confirmation_confidence": 0.1,
            "better_target_persistence_frames": 1,
            "better_target_score_margin": 0.1,
        },
        replan_config={"cooldown_frames": 0, "periodic_interval_frames": 20},
        relation_config={"minimum_confidence": 0.0},
    )
    objects = {
        1: node(1, center=(12.0, 12.0)),
        2: node(2, center=(24.0, 20.0)),
    }
    navigator.set_query("前往小汽车附近")
    first = navigator.update(0, [2.0, 2.0, 0.0], objects, [1, 2])
    assert first.selected_node_id == 1 and first.replanned
    time.sleep(0.03)
    switched = navigator.update(1, [2.0, 2.0, 0.0], objects, [])
    navigator.close()

    assert switched.selected_node_id == 2
    assert switched.transition_reason == "better_target_persisted"
    assert switched.replan_reason == "better_target_persisted"
    assert switched.replanned and len(switched.path_world) > 0
    assert switched.qwen_audit["active_rerank"]["selected_node_id"] == 2


def test_qwen_exception_and_immediate_close_are_isolated() -> None:
    failing = make_navigator(FailingDelegate())
    failing.set_query("前往小汽车附近")
    time.sleep(0.02)
    update = failing.update(0, [2.0, 2.0, 0.0], {1: node()}, [1])
    failing.close()
    assert update.qwen_audit["events"][0]["status"] == "failed"
    assert update.failure is None

    delayed_delegate = FakeAsyncDelegate()
    cancelled = make_navigator(delayed_delegate)
    cancelled.set_query("前往小汽车附近")
    cancelled.close()
    assert cancelled.pending_qwen_count == 0
    assert cancelled._qwen_totals["cancelled"] == 1
    assert cancelled._qwen_events[0]["status"] in {"cancelled", "cancel_requested"}
    assert delayed_delegate.closed
