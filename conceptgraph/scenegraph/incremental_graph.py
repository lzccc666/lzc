from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from conceptgraph.scenegraph.relations import (
    RelationBuildResult,
    RelationConfig,
    RelationNode,
    SceneGraphEdge,
    _empty_conflict_statistics,
    _empty_pruning_statistics,
    _json_safe,
    _relation_statistics,
    _sort_edges,
    canonicalize_edge,
    coerce_relation_nodes,
    finalize_relation_candidates,
    generate_relation_candidates,
    merge_duplicate_edges,
    resolve_coordinate_frame,
)


class IncrementalSceneGraph:
    """Maintain relation candidates while recomputing only changed-node neighborhoods."""

    def __init__(self, config: Mapping[str, Any] | RelationConfig | None = None) -> None:
        self.config = RelationConfig.from_mapping(config)
        self.current_frame = -1
        self._nodes: dict[int, RelationNode] = {}
        self._recomputed_centers: dict[int, np.ndarray] = {}
        self._candidate_edges: dict[tuple[int, int, str], SceneGraphEdge] = {}
        self._edges: list[SceneGraphEdge] = []
        self._last_observation_frame: int | None = None
        self._coordinate_signature: tuple | None = None
        self._last_result = self._make_result(
            update_statistics={
                "affected_nodes": [],
                "recomputed_pair_count": 0,
                "recomputed_edge_count": 0,
                "reused_edge_count": 0,
                "expired_dynamic_edge_count": 0,
            }
        )

    @property
    def nodes(self) -> dict[int, RelationNode]:
        return dict(self._nodes)

    @property
    def edges(self) -> list[SceneGraphEdge]:
        return list(self._edges)

    def update_snapshot(
        self,
        nodes: Mapping[int, Any] | Iterable[Any],
        *,
        frame_index: int,
        ego_heading: Sequence[float] | float | None = None,
        trajectory: Sequence[Sequence[float]] | np.ndarray | None = None,
        ego_position: Sequence[float] | None = None,
        full_snapshot: bool = True,
    ) -> RelationBuildResult:
        started = time.perf_counter()
        incoming, node_rejections = coerce_relation_nodes(nodes, frame_index=frame_index)
        old_ids = set(self._nodes)
        incoming_ids = set(incoming)
        added = incoming_ids - old_ids
        removed = old_ids - incoming_ids if full_snapshot else set()
        moved: set[int] = set()
        state_changed: set[int] = set()

        for node_id in sorted(incoming_ids & old_ids):
            previous = self._nodes[node_id]
            baseline = self._recomputed_centers.get(node_id, previous.center)
            distance = float(np.linalg.norm(incoming[node_id].center[:2] - baseline[:2]))
            if distance > self.config.position_update_threshold_m + 1e-12:
                moved.add(node_id)
            if incoming[node_id].dynamic != previous.dynamic:
                state_changed.add(node_id)

        if full_snapshot:
            self._nodes = dict(incoming)
        else:
            self._nodes.update(incoming)
        for node_id in removed:
            self._nodes.pop(node_id, None)
            self._recomputed_centers.pop(node_id, None)

        affected = added | moved | state_changed
        result = self._recompute(
            affected,
            removed,
            frame_index=frame_index,
            ego_heading=ego_heading,
            trajectory=trajectory,
            ego_position=ego_position,
            update_details={
                "added_nodes": sorted(added),
                "moved_nodes": sorted(moved),
                "state_changed_nodes": sorted(state_changed),
                "invalidated_nodes": sorted(removed),
            },
            extra_rejections=node_rejections,
        )
        result.timing["total_s"] = float(time.perf_counter() - started)
        return result

    def add_node(
        self,
        node: Any,
        *,
        frame_index: int | None = None,
        **frame_context: Any,
    ) -> RelationBuildResult:
        frame_index = self._resolve_frame_index(frame_index)
        incoming, rejected = coerce_relation_nodes([node], frame_index=frame_index)
        if not incoming:
            result = self._make_result(
                rejected_edges=rejected,
                update_statistics={
                    "affected_nodes": [],
                    "recomputed_pair_count": 0,
                    "recomputed_edge_count": 0,
                    "reused_edge_count": len(self._edges),
                    "expired_dynamic_edge_count": 0,
                },
            )
            self._last_result = result
            return result
        relation_node = next(iter(incoming.values()))
        self._nodes[relation_node.node_id] = relation_node
        return self._recompute(
            {relation_node.node_id},
            set(),
            frame_index=frame_index,
            update_details={"added_nodes": [relation_node.node_id]},
            extra_rejections=rejected,
            **frame_context,
        )

    def update_node_position(
        self,
        node_id: int,
        center: Sequence[float],
        *,
        frame_index: int | None = None,
        **frame_context: Any,
    ) -> RelationBuildResult:
        node_id = int(node_id)
        if node_id not in self._nodes:
            raise KeyError(f"Unknown relation node: {node_id}")
        frame_index = self._resolve_frame_index(frame_index)
        previous = self._nodes[node_id]
        updated = replace(previous, center=np.asarray(center, dtype=float))
        self._nodes[node_id] = updated
        baseline = self._recomputed_centers.get(node_id, previous.center)
        moved_distance = float(np.linalg.norm(updated.center[:2] - baseline[:2]))
        affected = {node_id} if moved_distance > self.config.position_update_threshold_m + 1e-12 else set()
        return self._recompute(
            affected,
            set(),
            frame_index=frame_index,
            update_details={
                "moved_nodes": sorted(affected),
                "below_threshold_nodes": [] if affected else [node_id],
                "position_delta_m": moved_distance,
            },
            **frame_context,
        )

    def update_node_dynamic_state(
        self,
        node_id: int,
        dynamic: bool,
        *,
        frame_index: int | None = None,
        **frame_context: Any,
    ) -> RelationBuildResult:
        node_id = int(node_id)
        if node_id not in self._nodes:
            raise KeyError(f"Unknown relation node: {node_id}")
        frame_index = self._resolve_frame_index(frame_index)
        changed = self._nodes[node_id].dynamic != bool(dynamic)
        self._nodes[node_id] = replace(self._nodes[node_id], dynamic=bool(dynamic))
        return self._recompute(
            {node_id} if changed else set(),
            set(),
            frame_index=frame_index,
            update_details={"state_changed_nodes": [node_id] if changed else []},
            **frame_context,
        )

    def invalidate_node(
        self,
        node_id: int,
        *,
        frame_index: int | None = None,
        **frame_context: Any,
    ) -> RelationBuildResult:
        node_id = int(node_id)
        frame_index = self._resolve_frame_index(frame_index)
        existed = self._nodes.pop(node_id, None) is not None
        self._recomputed_centers.pop(node_id, None)
        return self._recompute(
            set(),
            {node_id} if existed else set(),
            frame_index=frame_index,
            update_details={"invalidated_nodes": [node_id] if existed else []},
            **frame_context,
        )

    def advance_frame(
        self,
        frame_index: int,
        *,
        ego_heading: Sequence[float] | float | None = None,
        trajectory: Sequence[Sequence[float]] | np.ndarray | None = None,
        ego_position: Sequence[float] | None = None,
    ) -> RelationBuildResult:
        return self._recompute(
            set(),
            set(),
            frame_index=int(frame_index),
            ego_heading=ego_heading,
            trajectory=trajectory,
            ego_position=ego_position,
            update_details={"frame_advanced_without_geometry_change": True},
        )

    def export_snapshot(self) -> RelationBuildResult:
        result = self._make_result(
            update_statistics={
                **self._last_result.update_statistics,
                "snapshot_export": True,
            }
        )
        result.timing = dict(self._last_result.timing)
        result.diagnostics = dict(self._last_result.diagnostics)
        return result

    def _recompute(
        self,
        affected: set[int],
        removed: set[int],
        *,
        frame_index: int,
        ego_heading: Sequence[float] | float | None = None,
        trajectory: Sequence[Sequence[float]] | np.ndarray | None = None,
        ego_position: Sequence[float] | None = None,
        update_details: Mapping[str, Any] | None = None,
        extra_rejections: Iterable[SceneGraphEdge] = (),
    ) -> RelationBuildResult:
        started = time.perf_counter()
        self.current_frame = max(self.current_frame, int(frame_index))
        previous_edges = {(edge.source, edge.target, edge.relation): edge for edge in self._edges}
        previous_candidates = dict(self._candidate_edges)
        affected_active = {node_id for node_id in affected if node_id in self._nodes}

        if not self._nodes:
            self._candidate_edges = {}
            self._edges = []
            update_stats = self._update_statistics(
                affected,
                removed,
                pair_count=0,
                generated_count=0,
                previous_edges=previous_edges,
                expired_count=0,
                update_details=update_details,
            )
            result = self._make_result(
                rejected_edges=list(extra_rejections),
                update_statistics=update_stats,
                timing={"total_s": float(time.perf_counter() - started)},
                diagnostics={"coordinate_frame": {"success": True, "reason": "empty_graph"}},
            )
            self._last_result = result
            self._last_observation_frame = self.current_frame
            return result

        frame = resolve_coordinate_frame(
            self.config.coordinate_frame,
            self.config.coordinate_frame_fallback,
            ego_heading=ego_heading,
            trajectory=trajectory,
            ego_position=ego_position,
        )
        if not frame.success:
            result = self._make_result(
                success=False,
                failure_code=frame.failure_code,
                failure_reason=frame.failure_reason,
                rejected_edges=list(extra_rejections),
                update_statistics={
                    "affected_nodes": sorted(affected | removed),
                    "recomputed_pair_count": 0,
                    "recomputed_edge_count": 0,
                    "reused_edge_count": len(previous_edges),
                    "expired_dynamic_edge_count": 0,
                },
                timing={"total_s": float(time.perf_counter() - started)},
                diagnostics={"coordinate_frame": frame.to_dict()},
            )
            self._last_result = result
            return result

        # 对象不动时，自车转向也会改变驾驶员视角的左右前后。
        # 使用轴而非ego位置比较，避免平移触发无必要的全量重算。
        signature = (
            frame.actual_frame,
            tuple(np.asarray(frame.forward_axis_world, dtype=float)),
            tuple(np.asarray(frame.left_axis_world, dtype=float)),
        )
        frame_changed = self._coordinate_signature is not None and (
            signature[0] != self._coordinate_signature[0]
            or not np.allclose(signature[1:], self._coordinate_signature[1:], atol=1e-9, rtol=0)
        )
        if frame_changed:
            affected = set(self._nodes)
            affected_active = set(self._nodes)
        self._coordinate_signature = signature
        update_details = {**(update_details or {}), "coordinate_frame_changed": frame_changed}

        stale_candidates: dict[tuple[int, int, str], SceneGraphEdge] = {}
        old_affected_candidates: dict[tuple[int, int, str], SceneGraphEdge] = {}
        expired_count = 0
        for key, edge in previous_candidates.items():
            incident_removed = edge.source in removed or edge.target in removed
            incident_affected = edge.source in affected_active or edge.target in affected_active
            if incident_removed:
                continue
            if incident_affected:
                old_affected_candidates[key] = edge
                continue
            if bool(edge.metadata.get("dynamic", False)) and edge.metadata.get("stale_since_frame") is not None:
                last_seen = edge.last_seen_frame
                age = (
                    self.current_frame - last_seen
                    if last_seen is not None
                    else self.config.dynamic_edge_ttl_frames + 1
                )
                if age > self.config.dynamic_edge_ttl_frames:
                    expired_count += 1
                    continue
                stale_candidates[key] = edge
            else:
                stale_candidates[key] = self._observe_reused_edge(edge)

        generated: list[SceneGraphEdge] = []
        geometry_rejections: list[SceneGraphEdge] = []
        pair_count = 0
        if affected_active:
            generated, geometry_rejections, pair_count = generate_relation_candidates(
                self._nodes,
                self.config,
                frame,
                frame_index=self.current_frame,
                pair_filter=lambda source, target: source in affected_active or target in affected_active,
            )

        observed_keys: set[tuple[int, int, str]] = set()
        for new_edge in generated:
            key = _edge_key(new_edge)
            observed_keys.add(key)
            old_edge = old_affected_candidates.get(key)
            if old_edge is None:
                stale_candidates[key] = new_edge
                continue
            merged, _, _ = merge_duplicate_edges([old_edge, new_edge])
            merged_edge = merged[0]
            merged_metadata = dict(merged_edge.metadata)
            merged_metadata.pop("stale_since_frame", None)
            stale_candidates[key] = self._refresh_temporal_confidence(
                replace(merged_edge, metadata=merged_metadata)
            )

        for key, old_edge in old_affected_candidates.items():
            # 原坐标系的方向边不可作为新坐标系的短时遮挡记忆。
            if frame_changed and old_edge.relation != "near":
                continue
            if key in observed_keys or not bool(old_edge.metadata.get("dynamic", False)):
                continue
            last_seen = old_edge.last_seen_frame
            age = self.current_frame - last_seen if last_seen is not None else self.config.dynamic_edge_ttl_frames + 1
            if age <= self.config.dynamic_edge_ttl_frames:
                stale_candidates[key] = replace(
                    old_edge,
                    metadata={
                        **old_edge.metadata,
                        "stale_since_frame": self.current_frame,
                    },
                )
            else:
                expired_count += 1

        for node_id in affected_active:
            self._recomputed_centers[node_id] = self._nodes[node_id].center.copy()
        self._candidate_edges = dict(sorted(stale_candidates.items()))
        edges, finalize_rejections, conflict_stats, pruning_stats = finalize_relation_candidates(
            self._candidate_edges.values(),
            self.config,
        )
        self._edges = edges
        update_stats = self._update_statistics(
            affected,
            removed,
            pair_count=pair_count,
            generated_count=len(generated),
            previous_edges=previous_edges,
            expired_count=expired_count,
            update_details=update_details,
        )
        result = self._make_result(
            rejected_edges=list(extra_rejections) + geometry_rejections + finalize_rejections,
            conflict_statistics=conflict_stats,
            pruning_statistics=pruning_stats,
            update_statistics=update_stats,
            timing={"total_s": float(time.perf_counter() - started)},
            diagnostics={"coordinate_frame": frame.to_dict()},
        )
        self._last_result = result
        self._last_observation_frame = self.current_frame
        return result

    def _observe_reused_edge(self, edge: SceneGraphEdge) -> SceneGraphEdge:
        if self._last_observation_frame is not None and self.current_frame <= self._last_observation_frame:
            return edge
        return self._refresh_temporal_confidence(
            replace(
                edge,
                observation_count=edge.observation_count + 1,
                last_seen_frame=self.current_frame,
                metadata={
                    **edge.metadata,
                    "valid_frame_range": [edge.first_seen_frame, self.current_frame],
                },
            )
        )

    def _refresh_temporal_confidence(self, edge: SceneGraphEdge) -> SceneGraphEdge:
        components = dict(edge.metadata.get("confidence_components", {}))
        if not components:
            return edge
        span = max(
            1,
            (edge.last_seen_frame - edge.first_seen_frame + 1)
            if edge.first_seen_frame is not None and edge.last_seen_frame is not None
            else 1,
        )
        components["observation_count"] = min(
            1.0,
            edge.observation_count / self.config.static_stability_frames,
        )
        components["stability"] = min(
            1.0,
            edge.observation_count / max(self.config.static_stability_frames, span),
        )
        weights = self.config.confidence_weights
        denominator = sum(max(0.0, float(value)) for value in weights.values())
        confidence = (
            sum(max(0.0, weights.get(key, 0.0)) * float(value) for key, value in components.items())
            / max(denominator, 1e-12)
        )
        return replace(
            edge,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            metadata={**edge.metadata, "confidence_components": components},
        )

    def _update_statistics(
        self,
        affected: set[int],
        removed: set[int],
        *,
        pair_count: int,
        generated_count: int,
        previous_edges: Mapping[tuple[int, int, str], SceneGraphEdge],
        expired_count: int,
        update_details: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        new_keys = {_edge_key(edge) for edge in self._edges}
        reused = sum(
            key in new_keys and edge.source not in affected and edge.target not in affected
            for key, edge in previous_edges.items()
        )
        stats = {
            "frame_index": self.current_frame,
            "affected_nodes": sorted(affected | removed),
            "recomputed_pair_count": int(pair_count),
            "recomputed_edge_count": int(generated_count),
            "reused_edge_count": int(reused),
            "expired_dynamic_edge_count": int(expired_count),
            "active_node_count": len(self._nodes),
            "candidate_edge_count": len(self._candidate_edges),
            "output_edge_count": len(self._edges),
        }
        stats.update(_json_safe(dict(update_details or {})))
        return stats

    def _make_result(
        self,
        *,
        success: bool = True,
        rejected_edges: Iterable[SceneGraphEdge] = (),
        conflict_statistics: Mapping[str, Any] | None = None,
        pruning_statistics: Mapping[str, Any] | None = None,
        update_statistics: Mapping[str, Any] | None = None,
        failure_code: str | None = None,
        failure_reason: str | None = None,
        timing: Mapping[str, float] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> RelationBuildResult:
        return RelationBuildResult(
            success=success,
            mode="incremental",
            edges=list(self._edges),
            rejected_edges=_sort_edges(rejected_edges),
            node_count=len(self._nodes),
            edge_count=len(self._edges),
            relation_statistics=_relation_statistics(self._edges),
            conflict_statistics=dict(conflict_statistics or _empty_conflict_statistics()),
            pruning_statistics=dict(pruning_statistics or _empty_pruning_statistics()),
            update_statistics=dict(update_statistics or {}),
            failure_code=failure_code,
            failure_reason=failure_reason,
            config=self.config.to_dict(),
            timing=dict(timing or {}),
            diagnostics=dict(diagnostics or {}),
        )

    def _resolve_frame_index(self, frame_index: int | None) -> int:
        if frame_index is not None:
            return int(frame_index)
        return max(0, self.current_frame + 1)


def _edge_key(edge: SceneGraphEdge) -> tuple[int, int, str]:
    edge = canonicalize_edge(edge)
    return edge.source, edge.target, edge.relation
