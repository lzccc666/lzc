from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from conceptgraph.navigation.pipeline import SemanticNavigationPipeline
from conceptgraph.navigation.planner import (
    OccupancyGrid2D,
    build_occupancy_grid_from_scene_map,
    load_occupancy_grid,
    overlay_scene_objects_on_grid,
    path_is_collision_free,
)
from conceptgraph.navigation.replan_policy import ReplanPolicy, ReplanPolicyConfig
from conceptgraph.navigation.semantic import (
    apply_qwen_query_parse,
    parse_navigation_query,
    rank_candidate_targets,
)
from conceptgraph.navigation.state_machine import (
    NavigationState,
    NavigationStateMachine,
    StateMachineConfig,
)
from conceptgraph.navigation.types import (
    OnlineNavigationFailure,
    OnlineNavigationUpdate,
    ParsedNavigationQuery,
    SceneGraphNode,
)
from conceptgraph.scenegraph.incremental_graph import IncrementalSceneGraph


DYNAMIC_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"}


@dataclass(frozen=True)
class OnlineNavigatorConfig:
    target_change_distance_m: float = 0.75
    qwen_enabled: bool = True
    qwen_async_workers: int = 2
    qwen_candidate_change_ratio: float = 0.34
    qwen_min_submit_interval_frames: int = 2
    qwen_query_min_confidence: float = 0.60
    qwen_apply_object_attributes: bool = True
    qwen_attribute_min_observations: int = 3
    qwen_attribute_refresh_interval_frames: int = 30
    qwen_attribute_retry_interval_frames: int = 10
    qwen_attribute_max_pending: int = 2
    qwen_attribute_max_submissions_per_frame: int = 1
    qwen_attribute_min_confidence: float = 0.60
    qwen_apply_rerank: bool = True
    qwen_rerank_min_confidence: float = 0.60
    qwen_rerank_score_margin: float = 0.13
    relation_coordinate_frame: str = "world"
    pose_forward_axis: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | Any | None) -> OnlineNavigatorConfig:
        if value is None:
            return cls()
        source = value if isinstance(value, Mapping) else {
            name: getattr(value, name)
            for name in cls.__dataclass_fields__
            if hasattr(value, name)
        }
        return cls(
            **{
                name: source[name]
                for name in cls.__dataclass_fields__
                if name in source and source[name] is not None
            }
        )


@dataclass
class _PendingQwenTask:
    task_id: int
    task: str
    future: Future
    binding: dict[str, Any]
    submitted_frame: int | None
    context: dict[str, Any]


class SnapshotNodeTracker:
    """Convert serialized mapping snapshots into stable in-memory navigation nodes."""

    def __init__(self, class_names: Sequence[str], *, frame_dt_s: float = 1.0) -> None:
        if not np.isfinite(frame_dt_s) or frame_dt_s <= 0:
            raise ValueError("frame_dt_s must be positive seconds")
        self.class_names = [str(item) for item in class_names]
        self.frame_dt_s = float(frame_dt_s)
        self.previous_timestamp_s = None
        self.registry: dict[tuple[Any, ...], int] = {}
        self.previous_nodes: dict[int, SceneGraphNode] = {}
        self.first_seen: dict[int, int] = {}

    def update(
        self,
        objects: Sequence[Mapping[str, Any]],
        frame_idx: int,
        *, timestamp_s: float | None = None,
    ) -> tuple[dict[int, SceneGraphNode], list[int]]:
        now_s = float(frame_idx) * self.frame_dt_s if timestamp_s is None else float(timestamp_s)
        if (not np.isfinite(now_s) or (self.previous_timestamp_s is not None
                and now_s <= self.previous_timestamp_s)):
            raise ValueError("snapshot timestamps must be strictly increasing")
        nodes: dict[int, SceneGraphNode] = {}
        for object_index, raw in enumerate(objects):
            bbox_points = np.asarray(raw.get("bbox_np"), dtype=float)
            if bbox_points.ndim != 2 or bbox_points.shape[0] == 0 or bbox_points.shape[1] < 2:
                continue
            dimensions = min(3, bbox_points.shape[1])
            center = np.mean(bbox_points[:, :dimensions], axis=0)
            extent = np.ptp(bbox_points[:, :dimensions], axis=0)
            center = np.pad(center, (0, max(0, 3 - center.size)), constant_values=0.0)[:3]
            extent = np.pad(extent, (0, max(0, 3 - extent.size)), constant_values=0.1)[:3]
            key = self._object_key(raw, object_index)
            if key not in self.registry:
                self.registry[key] = len(self.registry)
            node_id = self.registry[key]
            self.first_seen.setdefault(node_id, int(frame_idx))
            class_id = self._dominant_class_id(raw.get("class_id", []))
            tag = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else "object"
            previous = self.previous_nodes.get(node_id)
            displacement = (
                0.0
                if previous is None
                else float(np.linalg.norm(center[:2] - previous.bbox_center[:2]))
            )
            delta_s = None if previous is None else now_s - previous.metadata["timestamp_s"]
            speed = 0.0 if delta_s is None else displacement / delta_s
            motion_evidence = raw.get("motion_state")
            if motion_evidence not in {"static", "dynamic"}:
                motion_evidence = ("unknown" if timestamp_s is not None and previous is None and tag in DYNAMIC_CLASSES
                                   else "dynamic" if tag in DYNAMIC_CLASSES and speed > 0.75
                                   else "static")
            inst_color = np.asarray(raw.get("inst_color", []), dtype=float).reshape(-1)
            nodes[node_id] = SceneGraphNode(
                node_id=node_id,
                object_tag=tag,
                possible_tags=[tag],
                caption=tag,
                bbox_center=np.asarray(center, dtype=float),
                bbox_extent=np.maximum(np.asarray(extent, dtype=float), 0.1),
                motion_state=motion_evidence,
                is_potential_dynamic=tag in DYNAMIC_CLASSES,
                velocity=float(raw.get("velocity", speed)) if raw.get("timestamp_s") is not None else speed,
                is_short_term=bool(raw.get("is_short_term", False)),
                metadata={
                    "bbox_center": np.asarray(center, dtype=float).tolist(),
                    "bbox_extent": np.maximum(
                        np.asarray(extent, dtype=float),
                        0.1,
                    ).tolist(),
                    "num_detections": int(raw.get("num_detections", 1) or 1),
                    "first_seen_frame": self.first_seen[node_id],
                    "last_seen_frame": int(frame_idx),
                    "frame_displacement_m": displacement,
                    "timestamp_s": now_s,
                    "speed_unit": "m/s",
                    "timestamp_source": "explicit" if timestamp_s is not None else "configured_frame_dt",
                    "snapshot_object_key": list(key),
                    "inst_color": inst_color[:3].tolist(),
                },
            )
        changed = self._changed_ids(nodes)
        self.previous_nodes = _copy_nodes(nodes)
        self.previous_timestamp_s = now_s
        return dict(sorted(nodes.items())), changed

    def _changed_ids(self, nodes: Mapping[int, SceneGraphNode]) -> list[int]:
        changed = set(nodes) ^ set(self.previous_nodes)
        for node_id in set(nodes) & set(self.previous_nodes):
            current = nodes[node_id]
            previous = self.previous_nodes[node_id]
            if (
                np.linalg.norm(current.bbox_center[:2] - previous.bbox_center[:2]) > 0.10
                or current.motion_state != previous.motion_state
                or current.object_tag != previous.object_tag
            ):
                changed.add(node_id)
        return sorted(changed)

    @staticmethod
    def _object_key(raw: Mapping[str, Any], object_index: int) -> tuple[Any, ...]:
        if raw.get("stable_object_id") is not None:
            return ("stable_object_id", str(raw["stable_object_id"]))
        color = np.asarray(raw.get("inst_color", []), dtype=float).reshape(-1)
        if color.size >= 3 and np.all(np.isfinite(color[:3])):
            return ("inst_color",) + tuple(round(float(item), 6) for item in color[:3])
        return ("object_index", int(object_index))

    @staticmethod
    def _dominant_class_id(values: Any) -> int:
        array = np.asarray(values, dtype=int).reshape(-1)
        if array.size == 0:
            return -1
        counts = Counter(int(item) for item in array)
        return min(counts, key=lambda item: (-counts[item], item))


class OnlineSemanticNavigator:
    """Incremental semantic navigator using only the current in-memory snapshot."""

    def __init__(
        self,
        navigation_cfg: Any,
        *,
        online_config: Mapping[str, Any] | OnlineNavigatorConfig | None = None,
        state_machine_config: Mapping[str, Any] | StateMachineConfig | None = None,
        replan_config: Mapping[str, Any] | ReplanPolicyConfig | None = None,
        relation_config: Mapping[str, Any] | None = None,
        semantic_delegate: Any | None = None,
        base_occupancy_grid: OccupancyGrid2D | None = None,
    ) -> None:
        self.navigation_cfg = navigation_cfg
        self.config = (
            online_config
            if isinstance(online_config, OnlineNavigatorConfig)
            else OnlineNavigatorConfig.from_mapping(online_config)
        )
        self.state_machine = NavigationStateMachine(state_machine_config)
        self.replan_policy = ReplanPolicy(replan_config)
        relation_values = dict(relation_config or {})
        relation_values["coordinate_frame"] = self.config.relation_coordinate_frame
        relation_values["coordinate_frame_fallback"] = self.config.relation_coordinate_frame
        self.relation_graph = IncrementalSceneGraph(relation_values)
        self.semantic_delegate = semantic_delegate
        self._planning_pipeline = SemanticNavigationPipeline(navigation_cfg)
        self._base_occupancy_grid = _copy_grid(base_occupancy_grid)
        self._occupancy_grid: OccupancyGrid2D | None = None
        self._nodes: dict[int, SceneGraphNode] = {}
        self._query_text: str | None = None
        self._rule_query: ParsedNavigationQuery | None = None
        self._parsed_query: ParsedNavigationQuery | None = None
        self._query_version = 0
        self._object_version = "empty"
        self._candidate_version = "empty"
        self._edge_version = "empty"
        self._last_frame_idx: int | None = None
        self._last_update: OnlineNavigationUpdate | None = None
        self._path_world = np.empty((0, 3), dtype=float)
        self._path_grid: list[tuple[int, int]] = []
        self._planned_target_center: np.ndarray | None = None
        self._manual_requested = False
        self._trajectory: list[np.ndarray] = []
        self._closed = False
        self._qwen_executor = ThreadPoolExecutor(
            max_workers=max(1, int(self.config.qwen_async_workers)),
            thread_name_prefix="online-qwen-dispatch",
        )
        self._pending_qwen: dict[int, _PendingQwenTask] = {}
        self._next_qwen_task_id = 1
        self._qwen_events: list[dict[str, Any]] = []
        self._qwen_totals = Counter()
        self._last_qwen_candidate_ids: tuple[int, ...] = ()
        self._last_qwen_submit_frame: int | None = None
        self._current_candidate_ids: tuple[int, ...] = ()
        self._qwen_node_attributes: dict[int, dict[str, Any]] = {}
        self._qwen_attribute_pending_nodes: set[int] = set()
        self._qwen_attribute_last_submitted: dict[int, int] = {}
        self._qwen_attribute_last_applied: dict[int, int] = {}
        self._qwen_attribute_retry_after: dict[int, int] = {}
        self._active_qwen_rerank: dict[str, Any] | None = None

    @property
    def state(self) -> NavigationState:
        return self.state_machine.state

    @property
    def pending_qwen_count(self) -> int:
        return sum(not item.future.done() for item in self._pending_qwen.values())

    def set_query(self, query_text: str) -> None:
        if self._closed:
            raise RuntimeError("navigator is closed")
        normalized = " ".join(str(query_text).strip().split())
        if not normalized:
            raise ValueError("query_text must not be empty")
        default_distance = float(self.navigation_cfg.default_near_distance_m)
        rule_query = parse_navigation_query(normalized, default_distance)
        self._cancel_pending_qwen(
            "query_replaced",
            tasks={"query_parse", "candidate_rerank"},
        )
        self._query_text = normalized
        self._rule_query = rule_query
        self._parsed_query = copy.deepcopy(rule_query)
        self._query_version += 1
        self.state_machine.reset()
        self.replan_policy = ReplanPolicy(self.replan_policy.config)
        self._clear_path()
        self._last_update = None
        self._last_frame_idx = None
        self._last_qwen_candidate_ids = ()
        self._last_qwen_submit_frame = None
        self._current_candidate_ids = ()
        self._clear_qwen_rerank()
        if self._qwen_available("use_qwen_query_parse"):
            self._submit_qwen(
                "query_parse",
                self.semantic_delegate.parse_query_async,
                normalized,
                default_near_distance_m=default_distance,
                rule_result=copy.deepcopy(rule_query),
                binding={"query_version": int(self._query_version)},
                submitted_frame=None,
            )

    def request_replan(self) -> None:
        self._manual_requested = True

    def drain_qwen(self, timeout_s: float = 60.0) -> list[dict[str, Any]]:
        """Wait for in-flight semantic tasks without holding up normal frame updates."""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        events: list[dict[str, Any]] = []
        while self._pending_qwen and time.monotonic() < deadline:
            events.extend(self._poll_qwen())
            if self._pending_qwen:
                time.sleep(0.01)
        events.extend(self._poll_qwen())
        return events

    def update(
        self,
        frame_idx: int,
        vehicle_pose,
        objects,
        changed_object_ids,
        occupancy_update=None,
    ) -> OnlineNavigationUpdate:
        started = time.perf_counter()
        frame_idx = int(frame_idx)
        if self._last_frame_idx is not None and frame_idx == self._last_frame_idx:
            return copy.deepcopy(self._last_update)
        if self._last_frame_idx is not None and frame_idx < self._last_frame_idx:
            return self._failure_without_consuming_frame(
                frame_idx,
                "OUT_OF_ORDER_FRAME",
                f"frame {frame_idx} is older than last frame {self._last_frame_idx}",
                started,
            )
        before_machine = copy.deepcopy(self.state_machine)
        before_policy = copy.deepcopy(self.replan_policy)
        before_relation_graph = copy.deepcopy(self.relation_graph)
        before_nodes = _copy_nodes(self._nodes)
        before_occupancy_grid = _copy_grid(self._occupancy_grid)
        before_trajectory = [point.copy() for point in self._trajectory]
        before_versions = (
            self._object_version,
            self._candidate_version,
            self._edge_version,
        )
        before_manual_requested = self._manual_requested
        before_path_world = self._path_world.copy()
        before_path_grid = list(self._path_grid)
        before_target_center = (
            None if self._planned_target_center is None else self._planned_target_center.copy()
        )
        state_before = self.state_machine.state.value
        try:
            update = self._update_impl(
                frame_idx,
                vehicle_pose,
                objects,
                changed_object_ids,
                occupancy_update,
                started,
            )
        except Exception as exc:  # noqa: BLE001 - a bad frame must not terminate replay
            self.state_machine = before_machine
            self.replan_policy = before_policy
            self.relation_graph = before_relation_graph
            self._nodes = before_nodes
            self._occupancy_grid = before_occupancy_grid
            self._trajectory = before_trajectory
            (
                self._object_version,
                self._candidate_version,
                self._edge_version,
            ) = before_versions
            self._manual_requested = before_manual_requested
            self._path_world = before_path_world
            self._path_grid = before_path_grid
            self._planned_target_center = before_target_center
            update = OnlineNavigationUpdate(
                frame_idx=frame_idx,
                state_before=state_before,
                state_after=self.state_machine.state.value,
                transition_reason="frame_exception_recovered",
                state_trace=[state_before],
                candidate_ids=[],
                selected_node_id=self.state_machine.locked_target_id,
                target_confidence=float(self.state_machine.target_confidence),
                path_world=self._path_world.copy(),
                replanned=False,
                replan_reason=None,
                replan_audit=[],
                timings={"total_s": float(time.perf_counter() - started)},
                qwen_audit=self._qwen_frame_audit([]),
                failure=OnlineNavigationFailure(
                    code="FRAME_PROCESSING_EXCEPTION",
                    message=f"{type(exc).__name__}: {str(exc)[:400]}",
                    recoverable=True,
                ),
            )
        self._last_frame_idx = frame_idx
        self._last_update = copy.deepcopy(update)
        return update

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cancel_pending_qwen("navigator_closed")
        self._qwen_executor.shutdown(wait=True, cancel_futures=True)
        if self.semantic_delegate is not None and hasattr(self.semantic_delegate, "close"):
            self.semantic_delegate.close(wait=True)

    def __enter__(self) -> OnlineSemanticNavigator:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _update_impl(
        self,
        frame_idx: int,
        vehicle_pose,
        objects,
        changed_object_ids,
        occupancy_update,
        started: float,
    ) -> OnlineNavigationUpdate:
        timings: dict[str, float] = {}
        qwen_events = self._poll_qwen()
        poll_done = time.perf_counter()
        timings["qwen_poll_s"] = poll_done - started
        if self._query_text is None or self._parsed_query is None:
            return self._simple_failure(
                frame_idx,
                "QUERY_NOT_SET",
                "set_query() must be called before update()",
                started,
                qwen_events,
            )

        pose_vector = self._normalize_vehicle_pose(vehicle_pose)
        vehicle_xy = pose_vector[:2]
        self._trajectory.append(vehicle_xy.copy())
        nodes = _coerce_nodes(objects)
        self._apply_qwen_attributes(nodes)
        changed_ids = sorted({int(item) for item in (changed_object_ids or [])})
        previous_nodes = self._nodes
        self._nodes = nodes
        self._object_version = _nodes_version(nodes)
        self._maybe_submit_object_attributes(frame_idx, nodes)

        relation_started = time.perf_counter()
        heading = pose_vector[3:5] if len(pose_vector) >= 5 else None
        relation_result = self.relation_graph.update_snapshot(
            nodes,
            frame_index=frame_idx,
            ego_heading=heading,
            trajectory=np.asarray(self._trajectory, dtype=float),
            ego_position=vehicle_xy,
        )
        timings["relation_update_s"] = time.perf_counter() - relation_started
        self._edge_version = _edges_version(relation_result.edges)

        occupancy_started = time.perf_counter()
        previous_grid = self._occupancy_grid
        self._occupancy_grid, occupancy_audit = self._resolve_occupancy(
            nodes,
            occupancy_update,
        )
        timings["occupancy_update_s"] = time.perf_counter() - occupancy_started
        path_blocked = self._remaining_path_blocked(vehicle_xy)
        occupancy_audit["remaining_path_blocked"] = path_blocked
        if previous_grid is not None:
            occupancy_audit["changed_cell_count"] = _changed_cell_count(
                previous_grid,
                self._occupancy_grid,
            )

        ranking_started = time.perf_counter()
        query = copy.deepcopy(self._parsed_query)
        if len(pose_vector) >= 5:
            query.vehicle_heading_xy = pose_vector[3:5].copy()
        candidates = rank_candidate_targets(
            parsed_query=query,
            nodes=nodes,
            edges=relation_result.edges,
            vehicle_xy=vehicle_xy,
            lambda_text=float(self.navigation_cfg.lambda_text),
            lambda_relation=float(self.navigation_cfg.lambda_relation),
            lambda_distance=float(self.navigation_cfg.lambda_distance_prior),
            minimum_edge_confidence=float(
                getattr(self.navigation_cfg, "relation_edge_minimum_confidence", 0.35)
            ),
            relation_coordinate_frame=self.config.relation_coordinate_frame,
            relation_coordinate_frame_fallback=self.config.relation_coordinate_frame,
            relation_trajectory=np.asarray(self._trajectory, dtype=float),
            evidence_policy=getattr(self.navigation_cfg, "target_evidence_policy", None),
        )
        timings["query_ranking_s"] = time.perf_counter() - ranking_started
        candidate_pairs = [(item.node_id, item.score) for item in candidates]
        candidate_pairs = self._apply_qwen_rerank(candidate_pairs)
        score_lookup = {int(node_id): float(score) for node_id, score in candidate_pairs}
        candidates = [item for item in candidates if int(item.node_id) in score_lookup]
        candidates.sort(
            key=lambda item: (-round(score_lookup.get(int(item.node_id), item.score), 12), item.node_id)
        )
        for item in candidates:
            item.score = score_lookup.get(int(item.node_id), float(item.score))
        candidate_ids = [int(item.node_id) for item in candidates]
        candidate_scores = {int(item.node_id): float(item.score) for item in candidates}
        self._current_candidate_ids = tuple(candidate_ids)
        self._candidate_version = _candidate_version(candidate_pairs)

        observation = self.state_machine.observe(
            frame_idx,
            candidate_pairs,
            valid_target_ids=set(nodes),
        )
        if observation.target_lost:
            self._clear_path()
        locked_id = self.state_machine.locked_target_id
        target_geometry_changed = self._target_geometry_changed(
            locked_id,
            changed_ids,
            previous_nodes,
            nodes,
        )
        if self._manual_requested:
            self.replan_policy.request_manual_replan()
            self._manual_requested = False
        decision = self.replan_policy.evaluate(
            frame_idx,
            vehicle_xy,
            has_locked_target=locked_id is not None and locked_id in candidate_scores,
            target_confirmed=observation.target_confirmed,
            target_changed=target_geometry_changed,
            better_target_persisted=observation.better_target_persisted,
            path_blocked=path_blocked,
        )

        replanned = False
        failure: OnlineNavigationFailure | None = None
        state_trace = list(observation.state_trace)
        transition_reason = observation.transition_reason
        planning_started = time.perf_counter()
        planning_result = None
        self._last_planning_result = None
        if path_blocked and not decision.replanned:
            # Debounce/cooldown may defer a new plan, never the invalidation of
            # a path that intersects this frame's actual occupancy.
            self._clear_path()
            occupancy_audit["unsafe_path_invalidated"] = True
            transition_reason = "blocked_path_invalidated_while_replan_deferred"
            failure = OnlineNavigationFailure(
                code="PATH_BLOCKED_REPLAN_DEFERRED",
                message="Blocked path cleared while the replan policy defers another attempt.",
                recoverable=True,
                details={"replan_audit": decision.audits},
            )
        if decision.replanned and locked_id is not None:
            replanned = True
            replanning = bool(len(self._path_grid)) and decision.reason != "target_confirmed"
            _, planning_state = self.state_machine.begin_planning(replanning=replanning)
            state_trace.append(planning_state)
            self._planning_pipeline.set_in_memory_inputs(
                nodes=nodes,
                edges=relation_result.edges,
                occupancy_grid=self._occupancy_grid,
            )
            planning_result = self._planning_pipeline.select(
                self._query_text,
                pose_vector,
                parsed_query=copy.deepcopy(self._parsed_query),
                target_node_id=locked_id,
                run_semantic_delegate=False,
            )
            self._last_planning_result = planning_result
            if planning_result.success and planning_result.plan is not None:
                self._path_world = np.asarray(planning_result.plan.path_world, dtype=float).copy()
                self._path_grid = list(planning_result.plan.path_grid)
                self._planned_target_center = nodes[locked_id].bbox_center[:2].copy()
                self.replan_policy.record_plan(frame_idx, vehicle_xy)
                _, tracked_state = self.state_machine.planning_succeeded()
                state_trace.append(tracked_state)
                transition_reason = decision.reason or transition_reason
                self._maybe_submit_candidate_rerank(
                    frame_idx,
                    pose_vector,
                    relation_result.edges,
                    candidate_ids,
                )
            else:
                self._clear_path()
                _, failed_state = self.state_machine.planning_failed()
                state_trace.append(failed_state)
                transition_reason = "planning_failed"
                failure = OnlineNavigationFailure(
                    code=planning_result.failure_code or "PLANNING_FAILED",
                    message=planning_result.failure_reason or "Stage2 target selection failed.",
                    recoverable=True,
                    details={"replan_reason": decision.reason},
                )
        timings["planning_s"] = time.perf_counter() - planning_started

        if not nodes and failure is None:
            failure = OnlineNavigationFailure(
                code="EMPTY_OBJECTS",
                message="The frame contains no usable navigation objects.",
                recoverable=True,
            )
        qwen_events.extend(self._poll_qwen())
        timings["total_s"] = time.perf_counter() - started
        relation_audit = {
            "success": bool(relation_result.success),
            "failure_code": relation_result.failure_code,
            "edge_count": int(relation_result.edge_count),
            "update_statistics": relation_result.update_statistics,
            "changed_object_ids": changed_ids,
        }
        return OnlineNavigationUpdate(
            frame_idx=frame_idx,
            state_before=observation.state_before,
            state_after=self.state_machine.state.value,
            transition_reason=transition_reason,
            state_trace=_dedupe_consecutive(state_trace),
            candidate_ids=candidate_ids,
            candidate_scores=candidate_scores,
            selected_node_id=self.state_machine.locked_target_id,
            target_confidence=float(self.state_machine.target_confidence),
            path_world=self._path_world.copy(),
            replanned=replanned,
            replan_reason=decision.reason,
            replan_audit=decision.audits,
            timings=timings,
            qwen_audit=self._qwen_frame_audit(qwen_events),
            failure=failure,
            relation_update=relation_audit,
            occupancy_audit=occupancy_audit,
            input_version=self._binding(),
        )

    def _resolve_occupancy(
        self,
        nodes: Mapping[int, SceneGraphNode],
        occupancy_update: Any,
    ) -> tuple[OccupancyGrid2D, dict[str, Any]]:
        if occupancy_update is not None:
            grid = _coerce_occupancy_update(occupancy_update, self._occupancy_grid)
            return grid, {"source": "occupancy_update", "object_overlay_applied": False}
        scene_map = _scene_map_from_nodes(nodes)
        if self._base_occupancy_grid is None:
            bev_path = getattr(self.navigation_cfg, "bev_map_path", None)
            if bev_path and str(bev_path):
                self._base_occupancy_grid = load_occupancy_grid(
                    str(bev_path),
                    float(self.navigation_cfg.grid_resolution),
                    [],
                    float(self.navigation_cfg.grid_padding_m),
                    float(self.navigation_cfg.robot_radius_m),
                    object_bbox_scale=float(
                        getattr(self.navigation_cfg, "object_bbox_obstacle_scale", 0.18)
                    ),
                    max_object_radius_m=float(
                        getattr(self.navigation_cfg, "max_object_obstacle_radius_m", 1.2)
                    ),
                    ignore_dynamic_obstacles=bool(
                        getattr(self.navigation_cfg, "ignore_dynamic_obstacles", False)
                    ),
                    obstacle_profile=str(
                        getattr(self.navigation_cfg, "obstacle_profile", "thesis")
                    ),
                )
            elif scene_map:
                self._base_occupancy_grid = build_occupancy_grid_from_scene_map(
                    scene_map,
                    float(self.navigation_cfg.grid_resolution),
                    float(self.navigation_cfg.grid_padding_m),
                    0.0,
                    object_bbox_scale=0.0,
                    max_object_radius_m=0.0,
                    ignore_dynamic_obstacles=True,
                )
            else:
                raise ValueError("cannot initialize occupancy grid from an empty frame")
        grid = overlay_scene_objects_on_grid(
            self._base_occupancy_grid,
            scene_map,
            robot_radius_m=float(self.navigation_cfg.robot_radius_m),
            object_bbox_scale=float(
                getattr(self.navigation_cfg, "object_bbox_obstacle_scale", 0.18)
            ),
            max_object_radius_m=float(
                getattr(self.navigation_cfg, "max_object_obstacle_radius_m", 1.2)
            ),
            ignore_dynamic_obstacles=bool(
                getattr(self.navigation_cfg, "ignore_dynamic_obstacles", False)
            ),
        )
        return grid, {
            "source": grid.source,
            "object_overlay_applied": True,
            "live_object_count": len(scene_map),
        }

    def _remaining_path_blocked(self, vehicle_xy: np.ndarray) -> bool:
        if not self._path_grid or self._occupancy_grid is None or not len(self._path_world):
            return False
        distances = np.linalg.norm(self._path_world[:, :2] - vehicle_xy.reshape(1, 2), axis=1)
        nearest = int(np.argmin(distances))
        remaining = self._path_grid[min(nearest, len(self._path_grid) - 1) :]
        if self._planning_pipeline._selection_config().mode == "lane_constrained":
            from conceptgraph.navigation.lane_constraints import validate_lane_path
            topology = self._planning_pipeline._get_lane_topology()
            return not validate_lane_path(remaining, topology, self._occupancy_grid.occupancy)["valid"]
        return not path_is_collision_free(self._occupancy_grid, remaining)

    def _target_geometry_changed(
        self,
        locked_id: int | None,
        changed_ids: Sequence[int],
        previous: Mapping[int, SceneGraphNode],
        current: Mapping[int, SceneGraphNode],
    ) -> bool:
        if locked_id is None or locked_id not in changed_ids or locked_id not in current:
            return False
        reference = self._planned_target_center
        if reference is None and locked_id in previous:
            reference = previous[locked_id].bbox_center[:2]
        if reference is None:
            return False
        return bool(
            np.linalg.norm(current[locked_id].bbox_center[:2] - reference)
            >= self.config.target_change_distance_m
        )

    def _maybe_submit_candidate_rerank(
        self,
        frame_idx: int,
        pose_vector: np.ndarray,
        edges,
        current_candidate_ids: Sequence[int],
    ) -> None:
        if (
            not self.config.qwen_apply_rerank
            or not self._qwen_available("use_qwen_rerank")
            or any(item.task == "candidate_rerank" for item in self._pending_qwen.values())
        ):
            return
        candidate_tuple = tuple(int(item) for item in current_candidate_ids)
        if not self._candidate_change_significant(self._last_qwen_candidate_ids, candidate_tuple):
            return
        if (
            self._last_qwen_submit_frame is not None
            and frame_idx - self._last_qwen_submit_frame
            < self.config.qwen_min_submit_interval_frames
        ):
            self._qwen_totals["rerank_rejected_submit_interval"] += 1
            return

        # This second Stage2 pass evaluates Top-K alternatives. The authoritative
        # locked-target plan above only contains one candidate by construction.
        validation_result = self._planning_pipeline.select(
            self._query_text,
            pose_vector,
            parsed_query=copy.deepcopy(self._parsed_query),
            target_node_id=None,
            run_semantic_delegate=False,
        )
        verified = [
            candidate
            for candidate in validation_result.evaluated_candidates
            if candidate.reachable
            and any(goal.path_found and goal.rejection_reason is None for goal in candidate.goal_candidates)
        ]
        if not verified:
            return
        payload = [
            {
                "node_id": int(candidate.node_id),
                "semantic_rank": candidate.semantic_rank,
                "stage2_scores": {
                    "combined_semantic": float(candidate.score),
                    "joint_utility": candidate.selection_utility,
                    "path_cost": candidate.path_cost,
                    "min_clearance_m": candidate.min_clearance_m,
                },
                "path_validated": True,
            }
            for candidate in verified
        ]
        verified_ids = {item["node_id"] for item in payload}
        edge_payload = [
            edge.to_dict()
            for edge in edges
            if int(edge.source) in verified_ids or int(edge.target) in verified_ids
        ]
        self._submit_qwen(
            "candidate_rerank",
            self.semantic_delegate.rerank_candidates_async,
            self._query_text,
            payload,
            edge_payload,
            edge_version=self._edge_version,
            binding={"query_version": int(self._query_version)},
            submitted_frame=frame_idx,
            context={
                "candidate_ids": list(candidate_tuple),
                "verified_candidate_ids": sorted(verified_ids),
                "verified_candidate_payload": payload,
            },
        )
        self._last_qwen_candidate_ids = candidate_tuple
        self._last_qwen_submit_frame = frame_idx

    def _candidate_change_significant(
        self,
        previous: Sequence[int],
        current: Sequence[int],
    ) -> bool:
        if not previous:
            return bool(current)
        old = set(previous)
        new = set(current)
        union = old | new
        ratio = 0.0 if not union else len(old ^ new) / len(union)
        return ratio >= self.config.qwen_candidate_change_ratio or tuple(previous) != tuple(current)

    def _maybe_submit_object_attributes(
        self,
        frame_idx: int,
        nodes: Mapping[int, SceneGraphNode],
    ) -> None:
        if (
            not self.config.qwen_apply_object_attributes
            or not self._qwen_available("use_qwen_object_attributes")
        ):
            return
        available_slots = max(
            0,
            int(self.config.qwen_attribute_max_pending)
            - len(self._qwen_attribute_pending_nodes),
        )
        submit_limit = min(
            available_slots,
            max(0, int(self.config.qwen_attribute_max_submissions_per_frame)),
        )
        submitted = 0
        for node_id, node in sorted(nodes.items()):
            if submitted >= submit_limit:
                break
            if node_id in self._qwen_attribute_pending_nodes:
                continue
            if int(node.metadata.get("num_detections", 1)) < int(
                self.config.qwen_attribute_min_observations
            ):
                continue
            if frame_idx < self._qwen_attribute_retry_after.get(node_id, -1):
                continue
            last_submitted = self._qwen_attribute_last_submitted.get(node_id)
            if (
                node_id in self._qwen_node_attributes
                and last_submitted is not None
                and frame_idx - last_submitted
                < int(self.config.qwen_attribute_refresh_interval_frames)
            ):
                continue
            crop_paths = _node_crop_paths(node)
            if not crop_paths:
                continue
            evidence_version = _object_evidence_version(node, crop_paths)
            metadata = {
                "object_tag": node.object_tag,
                "possible_tags": list(node.possible_tags),
                "caption": node.caption,
                "motion_state": node.motion_state,
                "colors": list(node.metadata.get("attributes", {}).get("colors", [])),
                "bbox_extent": np.asarray(node.bbox_extent, dtype=float).tolist(),
                "num_detections": int(node.metadata.get("num_detections", 1)),
            }
            self._submit_qwen(
                "object_attributes",
                self.semantic_delegate.describe_object_async,
                crop_paths,
                metadata,
                object_version=evidence_version,
                binding={},
                submitted_frame=frame_idx,
                context={
                    "node_id": int(node_id),
                    "detector_tag": node.object_tag,
                    "object_identity": node.metadata.get("object_identity"),
                    "evidence_version": evidence_version,
                    "crop_paths": crop_paths,
                },
            )
            self._qwen_attribute_pending_nodes.add(node_id)
            self._qwen_attribute_last_submitted[node_id] = frame_idx
            submitted += 1

    def _apply_qwen_attributes(self, nodes: Mapping[int, SceneGraphNode]) -> None:
        for node_id, node in nodes.items():
            attributes = self._qwen_node_attributes.get(int(node_id))
            if attributes is None:
                continue
            expected_identity = attributes.get("object_identity")
            if (
                expected_identity is not None
                and node.metadata.get("object_identity") != expected_identity
            ):
                self._qwen_node_attributes.pop(int(node_id), None)
                self._qwen_attribute_last_submitted.pop(int(node_id), None)
                self._qwen_attribute_last_applied.pop(int(node_id), None)
                self._qwen_attribute_retry_after.pop(int(node_id), None)
                continue
            possible_tags = [node.object_tag, *node.possible_tags]
            possible_tags.extend(attributes.get("possible_tags", []))
            possible_tags.append(attributes.get("object_tag", node.object_tag))
            node.possible_tags = list(dict.fromkeys(str(item) for item in possible_tags if item))
            summary = str(attributes.get("summary", "")).strip()
            if summary:
                node.caption = summary
            metadata_attributes = copy.deepcopy(node.metadata.get("attributes", {}))
            metadata_attributes["colors"] = list(attributes.get("colors", []))
            if attributes.get("body_type") is not None:
                metadata_attributes["body_type"] = attributes["body_type"]
            node.metadata["attributes"] = metadata_attributes
            node.metadata["qwen_attributes"] = copy.deepcopy(attributes)
            # 单张crop的语言描述只保留为建议，不覆盖多帧几何运动证据，
            # 也不能将尚无时间证据的unknown升级为可靠静止目标。
            node.metadata["qwen_motion_advisory"] = str(attributes.get("motion_state", "unknown"))

    def _apply_qwen_rerank(
        self,
        candidates: Sequence[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        pairs = [(int(node_id), float(score)) for node_id, score in candidates]
        if not pairs or self._active_qwen_rerank is None:
            return pairs
        current_ids = tuple(node_id for node_id, _ in pairs)
        selected = self._active_qwen_rerank["selected_node_id"]
        submitted_ids = tuple(self._active_qwen_rerank["candidate_ids"])
        if (selected is not None and int(selected) not in current_ids) or self._candidate_set_change_ratio(
            submitted_ids,
            current_ids,
        ) >= self.config.qwen_candidate_change_ratio:
            self._clear_qwen_rerank()
            self._qwen_totals["rerank_expired"] += 1
            return pairs
        if selected is None and self._active_qwen_rerank.get("no_match") is True:
            # 明确拒绝使旧锁定与路径失效；不能由迟滞机制继续执行。
            self._clear_path()
            self.state_machine.reset()
            return []
        selected = int(selected)
        max_score = max(score for _, score in pairs)
        boosted = max_score + max(
            float(self.config.qwen_rerank_score_margin),
            float(self.state_machine.config.better_target_score_margin) + 1e-6,
        )
        return [
            (node_id, boosted if node_id == selected else score)
            for node_id, score in pairs
        ]

    @staticmethod
    def _candidate_set_change_ratio(previous: Sequence[int], current: Sequence[int]) -> float:
        old = set(int(item) for item in previous)
        new = set(int(item) for item in current)
        union = old | new
        return 0.0 if not union else len(old ^ new) / len(union)

    def _submit_qwen(
        self,
        task: str,
        method,
        *args: Any,
        binding: dict[str, Any],
        submitted_frame: int | None,
        context: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        task_id = self._next_qwen_task_id
        self._next_qwen_task_id += 1

        def invoke():
            result = method(*args, **kwargs)
            if inspect.isawaitable(result):
                return asyncio.run(result)
            return result

        future = self._qwen_executor.submit(invoke)
        self._pending_qwen[task_id] = _PendingQwenTask(
            task_id=task_id,
            task=task,
            future=future,
            binding=dict(binding),
            submitted_frame=submitted_frame,
            context=copy.deepcopy(dict(context or {})),
        )
        self._qwen_totals["submitted"] += 1

    def _poll_qwen(self) -> list[dict[str, Any]]:
        events = []
        for task_id in sorted(list(self._pending_qwen)):
            pending = self._pending_qwen[task_id]
            if not pending.future.done():
                continue
            del self._pending_qwen[task_id]
            event = {
                "task_id": task_id,
                "task": pending.task,
                "binding": pending.binding,
                "submitted_frame": pending.submitted_frame,
                "context": copy.deepcopy(pending.context),
            }
            if pending.future.cancelled():
                event.update({"status": "cancelled", "applied": False})
                self._qwen_totals["cancelled"] += 1
            else:
                try:
                    result = pending.future.result()
                except Exception as exc:  # noqa: BLE001 - optional enhancement
                    event.update(
                        {
                            "status": "failed",
                            "applied": False,
                            "failure": f"{type(exc).__name__}: {str(exc)[:300]}",
                        }
                    )
                    self._qwen_totals["failed"] += 1
                else:
                    event.update(self._accept_qwen_result(pending, result))
            if pending.task == "object_attributes":
                node_id = pending.context.get("node_id")
                if node_id is not None:
                    self._qwen_attribute_pending_nodes.discard(int(node_id))
                    if not event.get("applied", False):
                        retry_frame = (pending.submitted_frame or 0) + int(
                            self.config.qwen_attribute_retry_interval_frames
                        )
                        self._qwen_attribute_retry_after[int(node_id)] = retry_frame
            events.append(event)
            self._qwen_events.append(event)
        return events

    def _accept_qwen_result(self, pending: _PendingQwenTask, result: Any) -> dict[str, Any]:
        current = self._binding()
        mismatches = [
            key
            for key, expected in pending.binding.items()
            if expected is not None and current.get(key) != expected
        ]
        cache = result.get("cache", {}) if isinstance(result, Mapping) else {}
        if mismatches:
            self._qwen_totals["stale"] += 1
            return {
                "status": "stale_discarded",
                "applied": False,
                "version_mismatches": mismatches,
                "cache_status": cache.get("status"),
            }
        if pending.task == "query_parse":
            applied, diagnostic = self._accept_qwen_query_parse(result)
        elif pending.task == "object_attributes":
            applied, diagnostic = self._accept_qwen_object_attributes(pending, result)
        elif pending.task == "candidate_rerank":
            applied, diagnostic = self._accept_qwen_rerank(pending, result)
        else:
            applied, diagnostic = False, "unknown_task"
        self._qwen_totals["completed"] += 1
        if applied:
            self._qwen_totals[f"{pending.task}_applied"] += 1
        else:
            self._qwen_totals[f"{pending.task}_not_applied"] += 1
        if str(cache.get("status", "")).startswith("hit"):
            self._qwen_totals["cache_hit"] += 1
        return {
            "status": "completed",
            "applied": applied,
            "used_qwen": bool(result.get("used_qwen", False)) if isinstance(result, Mapping) else False,
            "fallback_used": (
                bool(result.get("fallback_used", False)) if isinstance(result, Mapping) else True
            ),
            "failure_code": result.get("failure_code") if isinstance(result, Mapping) else None,
            "cache_status": cache.get("status"),
            "recommendation": (
                result.get("selected_node_id") if isinstance(result, Mapping) else None
            ),
            "diagnostic": diagnostic,
        }

    def _accept_qwen_query_parse(self, result: Any) -> tuple[bool, str]:
        if not isinstance(result, Mapping) or not result.get("used_qwen", False):
            return False, "fallback_or_invalid_result"
        output = result.get("normalized_output")
        if not isinstance(output, Mapping):
            return False, "missing_normalized_output"
        confidence = float(output.get("confidence", 0.0) or 0.0)
        if confidence < float(self.config.qwen_query_min_confidence):
            return False, "confidence_below_threshold"
        previous_signature = _parsed_query_signature(self._parsed_query)
        parsed = apply_qwen_query_parse(
            copy.deepcopy(self._rule_query),
            output,
            float(self.navigation_cfg.default_near_distance_m),
        )
        self._parsed_query = parsed
        if _parsed_query_signature(parsed) != previous_signature:
            self.state_machine.reset()
            self.replan_policy = ReplanPolicy(self.replan_policy.config)
            self._clear_path()
            self._clear_qwen_rerank()
            return True, "semantic_query_changed_state_reset"
        return True, "semantic_query_unchanged"

    def _accept_qwen_object_attributes(
        self,
        pending: _PendingQwenTask,
        result: Any,
    ) -> tuple[bool, str]:
        if not isinstance(result, Mapping):
            return False, "invalid_result_type"
        if not result.get("used_qwen", False) or result.get("fallback_used", False):
            return False, "fallback_or_invalid_result"
        output = result.get("normalized_output")
        if not isinstance(output, Mapping):
            return False, "missing_normalized_output"
        confidence = float(output.get("confidence", result.get("confidence", 0.0)) or 0.0)
        if confidence < float(self.config.qwen_attribute_min_confidence):
            return False, "confidence_below_threshold"
        node_id = int(pending.context.get("node_id", -1))
        current = self._nodes.get(node_id)
        if current is None:
            return False, "object_no_longer_present"
        if current.object_tag != pending.context.get("detector_tag"):
            return False, "detector_identity_changed"
        expected_identity = pending.context.get("object_identity")
        if (
            expected_identity is not None
            and current.metadata.get("object_identity") != expected_identity
        ):
            return False, "object_track_identity_changed"
        expected_evidence = pending.context.get("evidence_version")
        if expected_evidence is not None and _object_evidence_version(
            current, _node_crop_paths(current)
        ) != expected_evidence:
            return False, "object_evidence_version_changed"
        attributes = copy.deepcopy(dict(output))
        attributes["confidence"] = confidence
        attributes["applied_frame"] = self._last_frame_idx
        attributes["evidence_version"] = pending.context.get("evidence_version")
        attributes["detector_tag_authoritative"] = current.object_tag
        attributes["object_identity"] = expected_identity
        self._qwen_node_attributes[node_id] = attributes
        self._qwen_attribute_last_applied[node_id] = int(
            self._last_frame_idx if self._last_frame_idx is not None else pending.submitted_frame or 0
        )
        self._apply_qwen_attributes({node_id: current})
        return True, "attributes_cached_and_merged"

    def _accept_qwen_rerank(
        self,
        pending: _PendingQwenTask,
        result: Any,
    ) -> tuple[bool, str]:
        if not isinstance(result, Mapping):
            return False, "invalid_result_type"
        if not result.get("used_qwen", False) or result.get("fallback_used", False):
            return False, "fallback_or_invalid_result"
        output = result.get("normalized_output")
        if not isinstance(output, Mapping):
            output = result
        confidence = float(output.get("confidence", 0.0) or 0.0)
        if confidence < float(self.config.qwen_rerank_min_confidence):
            return False, "confidence_below_threshold"
        verified_ids = {int(item) for item in pending.context.get("verified_candidate_ids", [])}
        ranking = [int(item) for item in output.get("ranking", [])]
        selected = output.get("selected_node_id")
        no_match = output.get("no_match") is True
        if selected is None and not no_match:
            return False, "missing_selected_node_id"
        if no_match and selected is not None:
            return False, "inconsistent_no_match_selection"
        selected = int(selected) if selected is not None else None
        if (
            (selected is not None and selected not in verified_ids)
            or set(ranking) != verified_ids
            or len(ranking) != len(verified_ids)
        ):
            return False, "recommendation_outside_verified_candidates"
        if selected is not None and selected not in self._current_candidate_ids:
            return False, "recommended_candidate_no_longer_present"
        submitted_ids = tuple(int(item) for item in pending.context.get("candidate_ids", []))
        if self._candidate_set_change_ratio(
            submitted_ids,
            self._current_candidate_ids,
        ) >= float(self.config.qwen_candidate_change_ratio):
            return False, "candidate_set_changed_significantly"
        self._active_qwen_rerank = {
            "selected_node_id": selected,
            "no_match": no_match,
            "ranking": ranking,
            "confidence": confidence,
            "candidate_ids": list(submitted_ids),
            "submitted_frame": pending.submitted_frame,
        }
        return True, "verified_no_match_activated" if no_match else "verified_recommendation_activated"

    def _clear_qwen_rerank(self) -> None:
        self._active_qwen_rerank = None

    def _cancel_pending_qwen(
        self,
        reason: str,
        *,
        tasks: set[str] | None = None,
    ) -> None:
        for task_id in sorted(list(self._pending_qwen)):
            pending = self._pending_qwen[task_id]
            if tasks is not None and pending.task not in tasks:
                continue
            del self._pending_qwen[task_id]
            cancelled = pending.future.cancel()
            event = {
                "task_id": task_id,
                "task": pending.task,
                "binding": pending.binding,
                "submitted_frame": pending.submitted_frame,
                "status": "cancelled" if cancelled else "cancel_requested",
                "applied": False,
                "reason": reason,
            }
            self._qwen_events.append(event)
            self._qwen_totals["cancelled"] += 1
            if pending.task == "object_attributes":
                node_id = pending.context.get("node_id")
                if node_id is not None:
                    self._qwen_attribute_pending_nodes.discard(int(node_id))

    def _qwen_available(self, feature: str) -> bool:
        if not self.config.qwen_enabled or self.semantic_delegate is None:
            return False
        delegate_config = getattr(self.semantic_delegate, "config", None)
        if delegate_config is None:
            return True
        return bool(getattr(delegate_config, "enabled", True) and getattr(delegate_config, feature, True))

    def _qwen_frame_audit(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "events": copy.deepcopy(events),
            "pending_count": self.pending_qwen_count,
            "pending_attribute_node_ids": sorted(self._qwen_attribute_pending_nodes),
            "attribute_node_ids": sorted(self._qwen_node_attributes),
            "active_rerank": copy.deepcopy(self._active_qwen_rerank),
            "totals": dict(sorted(self._qwen_totals.items())),
        }

    def _binding(self) -> dict[str, Any]:
        return {
            "query_version": int(self._query_version),
            "object_version": self._object_version,
            "candidate_version": self._candidate_version,
            "edge_version": self._edge_version,
        }

    def _normalize_vehicle_pose(self, pose: Any) -> np.ndarray:
        array = np.asarray(pose, dtype=float)
        if array.shape == (4, 4):
            xyz = array[:3, 3]
            # nuScenes input matrices are camera-to-world; camera +z is forward.
            heading_axis = 2 if self._planning_pipeline._selection_config().mode == "lane_constrained" else 0
            if self.config.pose_forward_axis is not None:
                heading_axis = int(self.config.pose_forward_axis)
                if heading_axis not in {0, 1, 2}:
                    raise ValueError("pose_forward_axis must be 0, 1 or 2")
            heading = array[:2, heading_axis]
            norm = float(np.linalg.norm(heading))
            if norm <= 1e-9:
                heading = np.asarray([1.0, 0.0])
            else:
                heading = heading / norm
            result = np.concatenate([xyz, heading])
        else:
            result = array.reshape(-1)
            if result.size < 2:
                raise ValueError("vehicle_pose must contain at least x and y")
            if result.size < 3:
                result = np.concatenate([result, np.zeros(3 - result.size)])
            if result.size < 5:
                heading = np.asarray([1.0, 0.0])
                if self._trajectory:
                    delta = result[:2] - self._trajectory[-1]
                    norm = float(np.linalg.norm(delta))
                    if norm > 1e-9:
                        heading = delta / norm
                result = np.concatenate([result[:3], heading])
        if not np.all(np.isfinite(result[:5])):
            raise ValueError("vehicle_pose contains non-finite values")
        return result[:5].astype(float, copy=True)

    def _failure_without_consuming_frame(
        self,
        frame_idx: int,
        code: str,
        message: str,
        started: float,
    ) -> OnlineNavigationUpdate:
        state = self.state_machine.state.value
        return OnlineNavigationUpdate(
            frame_idx=frame_idx,
            state_before=state,
            state_after=state,
            transition_reason="frame_rejected",
            state_trace=[state],
            candidate_ids=[],
            selected_node_id=self.state_machine.locked_target_id,
            target_confidence=float(self.state_machine.target_confidence),
            path_world=self._path_world.copy(),
            replanned=False,
            replan_reason=None,
            timings={"total_s": float(time.perf_counter() - started)},
            qwen_audit=self._qwen_frame_audit([]),
            failure=OnlineNavigationFailure(code=code, message=message, recoverable=True),
        )

    def _simple_failure(
        self,
        frame_idx: int,
        code: str,
        message: str,
        started: float,
        qwen_events: list[dict[str, Any]],
    ) -> OnlineNavigationUpdate:
        state = self.state_machine.state.value
        return OnlineNavigationUpdate(
            frame_idx=frame_idx,
            state_before=state,
            state_after=state,
            transition_reason="frame_failed",
            state_trace=[state],
            candidate_ids=[],
            selected_node_id=self.state_machine.locked_target_id,
            target_confidence=float(self.state_machine.target_confidence),
            path_world=self._path_world.copy(),
            replanned=False,
            replan_reason=None,
            timings={"total_s": float(time.perf_counter() - started)},
            qwen_audit=self._qwen_frame_audit(qwen_events),
            failure=OnlineNavigationFailure(code=code, message=message, recoverable=True),
        )

    def _clear_path(self) -> None:
        self._path_world = np.empty((0, 3), dtype=float)
        self._path_grid = []
        self._planned_target_center = None


def _coerce_nodes(objects: Any) -> dict[int, SceneGraphNode]:
    if objects is None:
        return {}
    items = objects.items() if isinstance(objects, Mapping) else enumerate(objects)
    nodes: dict[int, SceneGraphNode] = {}
    for fallback_id, raw in items:
        if isinstance(raw, SceneGraphNode):
            node = copy.deepcopy(raw)
            node.bbox_center = np.asarray(raw.bbox_center, dtype=float).copy()
            node.bbox_extent = np.asarray(raw.bbox_extent, dtype=float).copy()
        elif isinstance(raw, Mapping):
            node_id = int(raw.get("node_id", raw.get("id", fallback_id)))
            metadata = copy.deepcopy(dict(raw.get("metadata", {}) or {}))
            center = raw.get("bbox_center", raw.get("center"))
            extent = raw.get("bbox_extent", metadata.get("bbox_extent"))
            if center is None and isinstance(raw.get("bbox"), Mapping):
                center = raw["bbox"].get("center")
                extent = extent if extent is not None else raw["bbox"].get("extent")
            if center is None or extent is None:
                raise ValueError(f"object {node_id} is missing bbox center or extent")
            center_array = np.asarray(center, dtype=float).reshape(-1)
            extent_array = np.asarray(extent, dtype=float).reshape(-1)
            if center_array.size < 2 or extent_array.size < 2:
                raise ValueError(f"object {node_id} has invalid bbox dimensions")
            center_array = np.pad(center_array, (0, max(0, 3 - center_array.size)))[:3]
            extent_array = np.pad(
                extent_array,
                (0, max(0, 3 - extent_array.size)),
                constant_values=0.1,
            )[:3]
            tag = str(
                raw.get("object_tag", metadata.get("object_tag", raw.get("class_name", "object")))
            )
            if isinstance(raw.get("class_name"), (list, tuple)) and raw["class_name"]:
                tag = str(raw["class_name"][0])
            motion = str(raw.get("motion_state", "dynamic" if raw.get("dynamic") else "static"))
            node = SceneGraphNode(
                node_id=node_id,
                object_tag=tag,
                possible_tags=[str(item) for item in raw.get("possible_tags", [tag])],
                caption=str(raw.get("caption", tag)),
                bbox_center=center_array,
                bbox_extent=np.maximum(extent_array, 0.1),
                motion_state=motion,
                is_potential_dynamic=bool(raw.get("is_potential_dynamic", tag in DYNAMIC_CLASSES)),
                velocity=float(raw.get("velocity", 0.0) or 0.0),
                is_short_term=bool(raw.get("is_short_term", False)),
                metadata=metadata,
            )
        else:
            raise TypeError(f"unsupported object type: {type(raw).__name__}")
        if not np.all(np.isfinite(node.bbox_center[:2])):
            raise ValueError(f"object {node.node_id} has non-finite center")
        nodes[int(node.node_id)] = node
    return dict(sorted(nodes.items()))


def _scene_map_from_nodes(nodes: Mapping[int, SceneGraphNode]) -> list[dict[str, Any]]:
    return [
        {
            "bbox": {
                "center": node.bbox_center.tolist(),
                "extent": node.bbox_extent.tolist(),
            },
            "object_tag": node.object_tag,
            "class_name": [node.object_tag],
            "motion_state": node.motion_state,
            "is_short_term": node.is_short_term,
        }
        for _, node in sorted(nodes.items())
    ]


def _coerce_occupancy_update(
    value: Any,
    previous: OccupancyGrid2D | None,
) -> OccupancyGrid2D:
    if isinstance(value, OccupancyGrid2D):
        return _copy_grid(value)
    if isinstance(value, Mapping):
        occupancy = value.get("occupancy")
        if occupancy is None and previous is not None:
            occupancy = previous.occupancy
        if occupancy is None:
            raise ValueError("occupancy_update is missing occupancy")
        array = (np.asarray(occupancy) > 0).astype(np.uint8)
        origin = np.asarray(
            value.get("origin_xy", previous.origin_xy if previous is not None else [0.0, 0.0]),
            dtype=float,
        )
        resolution = float(
            value.get("resolution", previous.resolution if previous is not None else 1.0)
        )
        for key, occupied_value in (("blocked_cells", 1), ("free_cells", 0)):
            for cell in value.get(key, []):
                rc = tuple(int(item) for item in cell)
                if 0 <= rc[0] < array.shape[0] and 0 <= rc[1] < array.shape[1]:
                    array[rc] = occupied_value
        from scipy.ndimage import distance_transform_edt

        return OccupancyGrid2D(
            occupancy=array,
            resolution=resolution,
            origin_xy=origin.copy(),
            clearance_m=distance_transform_edt(1 - array) * resolution,
            source=str(value.get("source", "occupancy_update")),
            metadata=copy.deepcopy(dict(value.get("metadata", {}))),
        )
    array = (np.asarray(value) > 0).astype(np.uint8)
    if array.ndim != 2:
        raise ValueError("occupancy_update array must be 2D")
    if previous is None:
        raise ValueError("an array occupancy_update requires an existing grid")
    return _coerce_occupancy_update({"occupancy": array}, previous)


def _copy_grid(grid: OccupancyGrid2D | None) -> OccupancyGrid2D | None:
    if grid is None:
        return None
    return OccupancyGrid2D(
        occupancy=np.asarray(grid.occupancy, dtype=np.uint8).copy(),
        resolution=float(grid.resolution),
        origin_xy=np.asarray(grid.origin_xy, dtype=float).copy(),
        clearance_m=(
            None if grid.clearance_m is None else np.asarray(grid.clearance_m, dtype=float).copy()
        ),
        source=str(grid.source),
        fallback_used=bool(grid.fallback_used),
        metadata=copy.deepcopy(grid.metadata),
    )


def _copy_nodes(nodes: Mapping[int, SceneGraphNode]) -> dict[int, SceneGraphNode]:
    return _coerce_nodes(nodes)


def _nodes_version(nodes: Mapping[int, SceneGraphNode]) -> str:
    payload = [
        {
            "node_id": node_id,
            "tag": node.object_tag,
            "center": np.round(node.bbox_center, 6).tolist(),
            "extent": np.round(node.bbox_extent, 6).tolist(),
            "motion_state": node.motion_state,
            "num_detections": int(node.metadata.get("num_detections", 1)),
        }
        for node_id, node in sorted(nodes.items())
    ]
    return _semantic_hash(payload)


def _edges_version(edges: Sequence[Any]) -> str:
    return _semantic_hash([edge.to_dict() for edge in edges])


def _candidate_version(candidates: Sequence[tuple[int, float]]) -> str:
    return _semantic_hash(
        [{"node_id": int(node_id), "score": round(float(score), 8)} for node_id, score in candidates]
    )


def _node_crop_paths(node: SceneGraphNode) -> list[str]:
    raw = node.metadata.get("crop_paths")
    if raw is None:
        raw = [node.metadata.get("crop_path")] if node.metadata.get("crop_path") else []
    if isinstance(raw, (str, Path)):
        raw = [raw]
    paths = []
    for value in raw or []:
        path = Path(value)
        if path.is_file():
            paths.append(str(path.resolve()))
    return paths


def _object_evidence_version(node: SceneGraphNode, crop_paths: Sequence[str]) -> str:
    evidence = []
    for value in crop_paths:
        path = Path(value)
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            evidence.append({"sha256": digest.hexdigest()})
        except OSError:
            evidence.append({"missing": True})
    return _semantic_hash(
        {
            "node_id": int(node.node_id),
            "object_identity": node.metadata.get("object_identity"),
            "detector_tag": node.object_tag,
            "num_detections": int(node.metadata.get("num_detections", 1)),
            "crops": evidence,
        }
    )


def _parsed_query_signature(query: ParsedNavigationQuery | None) -> str | None:
    if query is None:
        return None
    return _semantic_hash(
        {
            "target_text": query.target_text,
            "target_categories": list(query.target_categories),
            "target_colors": list(query.target_colors),
            "anchor_text": query.anchor_text,
            "anchor_colors": list(query.anchor_colors),
            "relation": query.relation,
            "prefer_nearest": bool(query.prefer_nearest),
            "prefer_front": bool(query.prefer_front),
            "near_distance": query.near_distance,
            "min_target_distance_m": query.min_target_distance_m,
            "max_target_distance_m": query.max_target_distance_m,
        }
    )


def _semantic_hash(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _changed_cell_count(before: OccupancyGrid2D, after: OccupancyGrid2D) -> int | None:
    if before.shape != after.shape:
        return None
    if not np.allclose(before.origin_xy, after.origin_xy) or before.resolution != after.resolution:
        return None
    return int(np.count_nonzero(before.occupancy != after.occupancy))


def _dedupe_consecutive(values: Sequence[str]) -> list[str]:
    result = []
    for value in values:
        if not result or result[-1] != value:
            result.append(value)
    return result


__all__ = [
    "OnlineNavigatorConfig",
    "OnlineSemanticNavigator",
    "SnapshotNodeTracker",
]
