from __future__ import annotations

import json
import math
import pickle
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


SYMMETRIC_RELATIONS = frozenset({"near"})
DIRECTIONAL_RELATIONS = frozenset({"left_of", "right_of", "in_front_of", "behind"})
SUPPORTED_RELATIONS = SYMMETRIC_RELATIONS | DIRECTIONAL_RELATIONS
INVERSE_RELATIONS = {
    "near": "near",
    "left_of": "right_of",
    "right_of": "left_of",
    "in_front_of": "behind",
    "behind": "in_front_of",
}
CANONICAL_RELATION_ORDER = {
    "near": 0,
    "left_of": 1,
    "in_front_of": 2,
    "right_of": 3,
    "behind": 4,
}


DEFAULT_RELATION_CONFIG: dict[str, Any] = {
    "relation_distance_thresholds_m": {
        "near": 8.0,
        "directional": 12.0,
        "left_of": 12.0,
        "right_of": 12.0,
        "in_front_of": 12.0,
        "behind": 12.0,
    },
    "direction_angle_threshold_deg": 40.0,
    "minimum_direction_distance_m": 0.50,
    "coordinate_frame": "local_trajectory",
    "coordinate_frame_fallback": "ego_heading",
    "confidence_weights": {
        "distance": 0.30,
        "angle_margin": 0.20,
        "observation_count": 0.15,
        "stability": 0.15,
        "coordinate_frame": 0.10,
        "dynamic_state": 0.10,
    },
    "minimum_confidence": 0.35,
    "static_stability_frames": 4,
    "dynamic_edge_ttl_frames": 3,
    "position_update_threshold_m": 0.10,
    "top_k_per_relation": {
        "near": 8,
        "left_of": 6,
        "right_of": 6,
        "in_front_of": 6,
        "behind": 6,
    },
    "max_out_degree": 16,
    "max_in_degree": 16,
    "max_total_edges": 512,
    "conflict_margin": 0.03,
    "deterministic_tie_break": "confidence_stability_distance_relation_source_target",
}


@dataclass
class SceneGraphEdge:
    source: int
    target: int
    relation: str
    directed: bool | None = None
    confidence: float = 1.0
    distance_m: float | None = None
    coordinate_frame: str = "world"
    relation_type: str = "directional"
    first_seen_frame: int | None = None
    last_seen_frame: int | None = None
    observation_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        self.source = int(self.source)
        self.target = int(self.target)
        self.relation = str(self.relation).strip().lower()
        if self.directed is None:
            self.directed = self.relation not in SYMMETRIC_RELATIONS
        self.directed = bool(self.directed)
        self.confidence = float(np.clip(float(self.confidence), 0.0, 1.0))
        self.distance_m = None if self.distance_m is None else float(self.distance_m)
        self.coordinate_frame = str(self.coordinate_frame or "world")
        if self.relation in SYMMETRIC_RELATIONS:
            self.relation_type = "symmetric"
        elif not self.relation_type:
            self.relation_type = "directional"
        self.observation_count = max(1, int(self.observation_count))
        self.metadata = dict(self.metadata or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "directed": self.directed,
            "confidence": self.confidence,
            "distance_m": self.distance_m,
            "coordinate_frame": self.coordinate_frame,
            "relation_type": self.relation_type,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "observation_count": self.observation_count,
            "metadata": _json_safe(self.metadata),
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SceneGraphEdge:
        return cls(
            source=int(payload["source"]),
            target=int(payload["target"]),
            relation=str(payload["relation"]),
            directed=payload.get("directed"),
            confidence=float(payload.get("confidence", 1.0) or 0.0),
            distance_m=payload.get("distance_m"),
            coordinate_frame=str(payload.get("coordinate_frame", "world")),
            relation_type=str(payload.get("relation_type", "directional")),
            first_seen_frame=_optional_int(payload.get("first_seen_frame")),
            last_seen_frame=_optional_int(payload.get("last_seen_frame")),
            observation_count=int(payload.get("observation_count", 1) or 1),
            metadata=dict(payload.get("metadata", {}) or {}),
            rejection_reason=payload.get("rejection_reason"),
        )


@dataclass
class RelationNode:
    node_id: int
    center: np.ndarray
    dynamic: bool = False
    observation_count: int = 1
    first_seen_frame: int | None = None
    last_seen_frame: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.node_id = int(self.node_id)
        center = np.asarray(self.center, dtype=float).reshape(-1)
        if center.size < 2:
            raise ValueError("RelationNode.center must contain at least x and y")
        if center.size == 2:
            center = np.concatenate([center, np.zeros(1, dtype=float)])
        self.center = center[:3]
        self.dynamic = bool(self.dynamic)
        self.observation_count = max(1, int(self.observation_count))
        self.metadata = dict(self.metadata or {})


@dataclass
class RelationConfig:
    relation_distance_thresholds_m: dict[str, float]
    direction_angle_threshold_deg: float
    minimum_direction_distance_m: float
    coordinate_frame: str
    coordinate_frame_fallback: str | None
    confidence_weights: dict[str, float]
    minimum_confidence: float
    static_stability_frames: int
    dynamic_edge_ttl_frames: int
    position_update_threshold_m: float
    top_k_per_relation: dict[str, int]
    max_out_degree: int
    max_in_degree: int
    max_total_edges: int
    conflict_margin: float
    deterministic_tie_break: str
    directional_mode: str = "primary_axis"
    minimum_direction_component_m: float = 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | RelationConfig | None) -> RelationConfig:
        if isinstance(value, cls):
            return value
        supplied = _plain_mapping(value)
        merged = _deep_merge(DEFAULT_RELATION_CONFIG, supplied)
        if merged.get("directional_mode", "primary_axis") not in {"primary_axis", "all_supported_axes"}:
            raise ValueError("unsupported directional_mode")
        return cls(
            relation_distance_thresholds_m={
                str(key): float(item)
                for key, item in merged["relation_distance_thresholds_m"].items()
            },
            direction_angle_threshold_deg=float(merged["direction_angle_threshold_deg"]),
            minimum_direction_distance_m=max(
                0.0,
                float(merged["minimum_direction_distance_m"]),
            ),
            coordinate_frame=str(merged["coordinate_frame"]),
            coordinate_frame_fallback=(
                None
                if merged.get("coordinate_frame_fallback") in {None, "", "none"}
                else str(merged["coordinate_frame_fallback"])
            ),
            confidence_weights={
                str(key): float(item) for key, item in merged["confidence_weights"].items()
            },
            minimum_confidence=float(merged["minimum_confidence"]),
            static_stability_frames=max(1, int(merged["static_stability_frames"])),
            dynamic_edge_ttl_frames=max(0, int(merged["dynamic_edge_ttl_frames"])),
            position_update_threshold_m=max(0.0, float(merged["position_update_threshold_m"])),
            top_k_per_relation={
                str(key): int(item) for key, item in merged["top_k_per_relation"].items()
            },
            max_out_degree=int(merged["max_out_degree"]),
            max_in_degree=int(merged["max_in_degree"]),
            max_total_edges=int(merged["max_total_edges"]),
            conflict_margin=max(0.0, float(merged["conflict_margin"])),
            deterministic_tie_break=str(merged["deterministic_tie_break"]),
            directional_mode=str(merged.get("directional_mode", "primary_axis")),
            minimum_direction_component_m=max(0.0,float(merged.get("minimum_direction_component_m",1.0))),
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class CoordinateFrameResolution:
    success: bool
    requested_frame: str
    actual_frame: str | None
    forward_axis_world: np.ndarray | None
    left_axis_world: np.ndarray | None
    reliability: float
    fallback_reason: str | None = None
    failure_code: str | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "requested_frame": self.requested_frame,
            "actual_frame": self.actual_frame,
            "forward_axis_world": (
                None if self.forward_axis_world is None else self.forward_axis_world.tolist()
            ),
            "left_axis_world": None if self.left_axis_world is None else self.left_axis_world.tolist(),
            "reliability": self.reliability,
            "fallback_reason": self.fallback_reason,
            "failure_code": self.failure_code,
            "failure_reason": self.failure_reason,
        }


@dataclass
class RelationBuildResult:
    success: bool
    mode: str
    edges: list[SceneGraphEdge]
    rejected_edges: list[SceneGraphEdge]
    node_count: int
    edge_count: int
    relation_statistics: dict[str, Any]
    conflict_statistics: dict[str, Any]
    pruning_statistics: dict[str, Any]
    update_statistics: dict[str, Any]
    failure_code: str | None
    failure_reason: str | None
    config: dict[str, Any]
    timing: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "mode": self.mode,
            "edges": [edge.to_dict() for edge in self.edges],
            "rejected_edges": [edge.to_dict() for edge in self.rejected_edges],
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "relation_statistics": _json_safe(self.relation_statistics),
            "conflict_statistics": _json_safe(self.conflict_statistics),
            "pruning_statistics": _json_safe(self.pruning_statistics),
            "update_statistics": _json_safe(self.update_statistics),
            "failure_code": self.failure_code,
            "failure_reason": self.failure_reason,
            "config": _json_safe(self.config),
            "timing": _json_safe(self.timing),
            "diagnostics": _json_safe(self.diagnostics),
        }


def inverse_relation(relation: str) -> str | None:
    return INVERSE_RELATIONS.get(str(relation).strip().lower())


def canonicalize_edge(edge: SceneGraphEdge) -> SceneGraphEdge:
    relation = str(edge.relation).strip().lower()
    source, target = int(edge.source), int(edge.target)
    if relation == "near":
        source, target = sorted((source, target))
        relation = "near"
    elif relation == "right_of":
        source, target, relation = target, source, "left_of"
    elif relation == "behind":
        source, target, relation = target, source, "in_front_of"
    return replace(
        edge,
        source=source,
        target=target,
        relation=relation,
        directed=relation != "near",
        relation_type="symmetric" if relation == "near" else "directional",
    )


def edge_relation_for_query(edge: SceneGraphEdge, source: int, target: int) -> str | None:
    """Return the relation seen from source toward target without duplicating directed facts."""
    edge = canonicalize_edge(edge)
    source, target = int(source), int(target)
    if edge.relation == "near":
        if {source, target} == {edge.source, edge.target}:
            return "near"
        return None
    if source == edge.source and target == edge.target:
        return edge.relation
    if source == edge.target and target == edge.source:
        return inverse_relation(edge.relation)
    return None


def resolve_coordinate_frame(
    requested_frame: str,
    fallback_frame: str | None = None,
    ego_heading: Sequence[float] | float | None = None,
    trajectory: Sequence[Sequence[float]] | np.ndarray | None = None,
    ego_position: Sequence[float] | None = None,
) -> CoordinateFrameResolution:
    requested = str(requested_frame or "world").strip().lower()
    fallback = None if fallback_frame is None else str(fallback_frame).strip().lower()
    attempted: list[tuple[str, str]] = []

    for frame_name in [requested, fallback]:
        if frame_name is None or any(frame_name == item[0] for item in attempted):
            continue
        axes, reason = _coordinate_axes(
            frame_name,
            ego_heading=ego_heading,
            trajectory=trajectory,
            ego_position=ego_position,
        )
        if axes is not None:
            forward, left, reliability = axes
            fallback_reason = None
            if frame_name != requested:
                first_reason = attempted[0][1] if attempted else "requested_frame_unavailable"
                fallback_reason = f"{requested}_unavailable:{first_reason};fallback_to:{frame_name}"
                reliability *= 0.75
            return CoordinateFrameResolution(
                success=True,
                requested_frame=requested,
                actual_frame=frame_name,
                forward_axis_world=forward,
                left_axis_world=left,
                reliability=float(reliability),
                fallback_reason=fallback_reason,
            )
        attempted.append((frame_name, reason or "unavailable"))

    attempted_text = ";".join(f"{name}:{reason}" for name, reason in attempted)
    return CoordinateFrameResolution(
        success=False,
        requested_frame=requested,
        actual_frame=None,
        forward_axis_world=None,
        left_axis_world=None,
        reliability=0.0,
        failure_code="COORDINATE_FRAME_UNAVAILABLE",
        failure_reason=attempted_text or "no coordinate frame was configured",
    )


def build_relations(
    nodes: Mapping[int, Any] | Iterable[Any],
    config: Mapping[str, Any] | RelationConfig | None = None,
    *,
    frame_index: int | None = None,
    ego_heading: Sequence[float] | float | None = None,
    trajectory: Sequence[Sequence[float]] | np.ndarray | None = None,
    ego_position: Sequence[float] | None = None,
    mode: str = "full_rebuild",
    pair_filter: Callable[[int, int], bool] | None = None,
) -> RelationBuildResult:
    started = time.perf_counter()
    cfg = RelationConfig.from_mapping(config)
    relation_nodes, node_rejections = coerce_relation_nodes(nodes, frame_index=frame_index)
    if not relation_nodes:
        return RelationBuildResult(
            success=True,
            mode=mode,
            edges=[],
            rejected_edges=node_rejections,
            node_count=0,
            edge_count=0,
            relation_statistics=_relation_statistics([]),
            conflict_statistics=_empty_conflict_statistics(),
            pruning_statistics=_empty_pruning_statistics(),
            update_statistics={"candidate_pair_count": 0, "generated_edge_count": 0},
            failure_code=None,
            failure_reason=None,
            config=cfg.to_dict(),
            timing={"total_s": float(time.perf_counter() - started)},
            diagnostics={"coordinate_frame": {"success": True, "reason": "empty_graph"}},
        )
    frame = resolve_coordinate_frame(
        cfg.coordinate_frame,
        cfg.coordinate_frame_fallback,
        ego_heading=ego_heading,
        trajectory=trajectory,
        ego_position=ego_position,
    )
    if not frame.success:
        return RelationBuildResult(
            success=False,
            mode=mode,
            edges=[],
            rejected_edges=node_rejections,
            node_count=len(relation_nodes),
            edge_count=0,
            relation_statistics=_relation_statistics([]),
            conflict_statistics=_empty_conflict_statistics(),
            pruning_statistics=_empty_pruning_statistics(),
            update_statistics={},
            failure_code=frame.failure_code,
            failure_reason=frame.failure_reason,
            config=cfg.to_dict(),
            timing={"total_s": float(time.perf_counter() - started)},
            diagnostics={"coordinate_frame": frame.to_dict()},
        )

    candidates, geometry_rejections, pair_count = generate_relation_candidates(
        relation_nodes,
        cfg,
        frame,
        frame_index=frame_index,
        pair_filter=pair_filter,
    )
    edges, finalize_rejections, conflict_stats, pruning_stats = finalize_relation_candidates(
        candidates,
        cfg,
    )
    rejected = _sort_edges(node_rejections + geometry_rejections + finalize_rejections)
    return RelationBuildResult(
        success=True,
        mode=mode,
        edges=edges,
        rejected_edges=rejected,
        node_count=len(relation_nodes),
        edge_count=len(edges),
        relation_statistics=_relation_statistics(edges),
        conflict_statistics=conflict_stats,
        pruning_statistics=pruning_stats,
        update_statistics={
            "candidate_pair_count": pair_count,
            "generated_edge_count": len(candidates),
        },
        failure_code=None,
        failure_reason=None,
        config=cfg.to_dict(),
        timing={"total_s": float(time.perf_counter() - started)},
        diagnostics={"coordinate_frame": frame.to_dict()},
    )


def coerce_relation_nodes(
    nodes: Mapping[int, Any] | Iterable[Any],
    *,
    frame_index: int | None = None,
) -> tuple[dict[int, RelationNode], list[SceneGraphEdge]]:
    items = nodes.items() if isinstance(nodes, Mapping) else enumerate(nodes)
    result: dict[int, RelationNode] = {}
    rejected: list[SceneGraphEdge] = []
    for fallback_id, raw in items:
        try:
            node = _coerce_relation_node(raw, fallback_id=fallback_id, frame_index=frame_index)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            node_id = _safe_node_id(raw, fallback_id)
            rejected.append(
                SceneGraphEdge(
                    source=node_id,
                    target=node_id,
                    relation="invalid_node",
                    directed=False,
                    confidence=0.0,
                    observation_count=1,
                    rejection_reason="invalid_node_geometry",
                    metadata={"diagnostic": str(exc)},
                )
            )
            continue
        if not np.all(np.isfinite(node.center[:2])):
            rejected.append(
                SceneGraphEdge(
                    source=node.node_id,
                    target=node.node_id,
                    relation="invalid_node",
                    directed=False,
                    confidence=0.0,
                    rejection_reason="non_finite_node_position",
                )
            )
            continue
        result[node.node_id] = node
    return dict(sorted(result.items())), rejected


def generate_relation_candidates(
    nodes: Mapping[int, RelationNode],
    config: RelationConfig,
    frame: CoordinateFrameResolution,
    *,
    frame_index: int | None = None,
    pair_filter: Callable[[int, int], bool] | None = None,
) -> tuple[list[SceneGraphEdge], list[SceneGraphEdge], int]:
    node_ids = sorted(nodes)
    candidates: list[SceneGraphEdge] = []
    rejected: list[SceneGraphEdge] = []
    pair_count = 0
    near_threshold = _distance_threshold(config, "near")
    directional_threshold = max(
        _distance_threshold(config, "left_of"),
        _distance_threshold(config, "in_front_of"),
    )
    max_distance = max(near_threshold, directional_threshold)

    for source_index, source_id in enumerate(node_ids):
        source_node = nodes[source_id]
        for target_id in node_ids[source_index + 1 :]:
            if pair_filter is not None and not pair_filter(source_id, target_id):
                continue
            pair_count += 1
            target_node = nodes[target_id]
            delta = target_node.center[:2] - source_node.center[:2]
            distance = float(np.linalg.norm(delta))
            if not math.isfinite(distance) or distance <= 1e-9:
                rejected.append(
                    SceneGraphEdge(
                        source=source_id,
                        target=target_id,
                        relation="near",
                        directed=False,
                        confidence=0.0,
                        distance_m=distance,
                        coordinate_frame=str(frame.actual_frame),
                        rejection_reason="coincident_nodes",
                    )
                )
                continue
            if distance > max_distance:
                continue

            if distance <= near_threshold:
                near_edge = _make_edge(
                    source_node,
                    target_node,
                    relation="near",
                    distance_m=distance,
                    angle_margin=1.0,
                    threshold_m=near_threshold,
                    config=config,
                    frame=frame,
                    frame_index=frame_index,
                )
                _accept_by_confidence(near_edge, config, candidates, rejected)

            # Direction is not observable below the configured position-change tolerance.
            minimum_direction_distance = max(
                config.position_update_threshold_m,
                config.minimum_direction_distance_m,
            )
            if minimum_direction_distance < distance <= directional_threshold:
                if config.directional_mode == "all_supported_axes":
                    directionals = []
                    for relation, axis in [("left_of", frame.left_axis_world),
                                           ("in_front_of", frame.forward_axis_world)]:
                        component = float(np.dot(delta, axis))
                        if abs(component) <= config.minimum_direction_component_m:
                            continue
                        first, second = (target_node, source_node) if component > 0 else (source_node, target_node)
                        margin = min(1.0,(abs(component)-config.minimum_direction_component_m)/max(distance,1e-9))
                        directionals.append((first,second,relation,margin))
                else:
                    directional = _directional_fact(source_node, target_node, delta, config, frame)
                    directionals = [] if directional is None else [directional]
                for directional in directionals:
                    fact_source, fact_target, relation, angle_margin = directional
                    threshold = _distance_threshold(config, relation)
                    if distance <= threshold:
                        edge = _make_edge(
                            fact_source,
                            fact_target,
                            relation=relation,
                            distance_m=distance,
                            angle_margin=angle_margin,
                            threshold_m=threshold,
                            config=config,
                            frame=frame,
                            frame_index=frame_index,
                        )
                        _accept_by_confidence(edge, config, candidates, rejected)
    return _sort_edges(candidates), _sort_edges(rejected), pair_count


def finalize_relation_candidates(
    candidates: Iterable[SceneGraphEdge],
    config: Mapping[str, Any] | RelationConfig | None,
) -> tuple[list[SceneGraphEdge], list[SceneGraphEdge], dict[str, Any], dict[str, Any]]:
    cfg = RelationConfig.from_mapping(config)
    merged, merge_rejected, duplicate_count = merge_duplicate_edges(candidates)
    conflict_free, conflict_rejected, conflict_stats = resolve_direction_conflicts(merged, cfg)
    pruned, pruning_rejected, pruning_stats = prune_edges(conflict_free, cfg)
    conflict_stats["duplicate_edges_merged"] = duplicate_count
    rejected = _sort_edges(merge_rejected + conflict_rejected + pruning_rejected)
    return _sort_edges(pruned), rejected, conflict_stats, pruning_stats


def merge_duplicate_edges(
    edges: Iterable[SceneGraphEdge],
) -> tuple[list[SceneGraphEdge], list[SceneGraphEdge], int]:
    grouped: dict[tuple[int, int, str], list[SceneGraphEdge]] = {}
    rejected: list[SceneGraphEdge] = []
    for raw_edge in edges:
        edge = canonicalize_edge(raw_edge)
        if edge.source == edge.target:
            rejected.append(replace(edge, rejection_reason="self_loop"))
            continue
        if edge.relation not in {"near", "left_of", "in_front_of"}:
            rejected.append(replace(edge, rejection_reason="unsupported_relation"))
            continue
        grouped.setdefault((edge.source, edge.target, edge.relation), []).append(edge)

    merged: list[SceneGraphEdge] = []
    duplicate_count = 0
    for key in sorted(grouped, key=lambda item: (_relation_order(item[2]), item[0], item[1])):
        group = grouped[key]
        if len(group) == 1:
            merged.append(group[0])
            continue
        duplicate_count += len(group) - 1
        weights = np.asarray([max(1, item.observation_count) for item in group], dtype=float)
        confidence = float(np.average([item.confidence for item in group], weights=weights))
        distances = [item.distance_m for item in group if item.distance_m is not None]
        distance = None if not distances else float(np.average(distances))
        metadata = dict(max(group, key=_edge_rank_key).metadata)
        metadata["merged_observation_sources"] = len(group)
        merged.append(
            SceneGraphEdge(
                source=key[0],
                target=key[1],
                relation=key[2],
                directed=key[2] != "near",
                confidence=confidence,
                distance_m=distance,
                coordinate_frame=max(group, key=_edge_rank_key).coordinate_frame,
                relation_type="symmetric" if key[2] == "near" else "directional",
                first_seen_frame=_minimum_optional(item.first_seen_frame for item in group),
                last_seen_frame=_maximum_optional(item.last_seen_frame for item in group),
                observation_count=sum(item.observation_count for item in group),
                metadata=metadata,
            )
        )
    return _sort_edges(merged), _sort_edges(rejected), duplicate_count


def resolve_direction_conflicts(
    edges: Iterable[SceneGraphEdge],
    config: Mapping[str, Any] | RelationConfig | None,
) -> tuple[list[SceneGraphEdge], list[SceneGraphEdge], dict[str, Any]]:
    cfg = RelationConfig.from_mapping(config)
    groups: dict[tuple[int, int, str], list[SceneGraphEdge]] = {}
    passthrough: list[SceneGraphEdge] = []
    for edge in edges:
        edge = canonicalize_edge(edge)
        axis = _relation_axis(edge.relation)
        if axis is None:
            passthrough.append(edge)
            continue
        pair = tuple(sorted((edge.source, edge.target)))
        groups.setdefault((pair[0], pair[1], axis), []).append(edge)

    kept = list(passthrough)
    rejected: list[SceneGraphEdge] = []
    conflict_groups = 0
    for group_key in sorted(groups):
        group = groups[group_key]
        orientations = {(edge.source, edge.target) for edge in group}
        if len(orientations) <= 1:
            kept.extend(group)
            continue
        conflict_groups += 1
        ranked = sorted(group, key=_edge_rank_key, reverse=True)
        winner = ranked[0]
        kept.append(winner)
        for loser in ranked[1:]:
            confidence_gap = winner.confidence - loser.confidence
            reason = (
                "direction_conflict_lower_confidence"
                if confidence_gap > cfg.conflict_margin
                else "direction_conflict_deterministic_tie_break"
            )
            rejected.append(
                replace(
                    loser,
                    rejection_reason=reason,
                    metadata={
                        **loser.metadata,
                        "conflict_winner": winner.to_dict(),
                        "confidence_gap": confidence_gap,
                    },
                )
            )
    return _sort_edges(kept), _sort_edges(rejected), {
        "conflict_groups_detected": conflict_groups,
        "conflicting_edges_removed": len(rejected),
        "left_right_conflicts": sum(1 for key in groups if key[2] == "lateral" and len({(e.source, e.target) for e in groups[key]}) > 1),
        "front_behind_conflicts": sum(1 for key in groups if key[2] == "longitudinal" and len({(e.source, e.target) for e in groups[key]}) > 1),
        "rejection_reasons": dict(sorted(Counter(edge.rejection_reason for edge in rejected).items())),
    }


def prune_edges(
    edges: Iterable[SceneGraphEdge],
    config: Mapping[str, Any] | RelationConfig | None,
) -> tuple[list[SceneGraphEdge], list[SceneGraphEdge], dict[str, Any]]:
    cfg = RelationConfig.from_mapping(config)
    ranked = sorted(edges, key=_edge_rank_key, reverse=True)
    relation_selected: list[SceneGraphEdge] = []
    rejected: list[SceneGraphEdge] = []
    topk_counts: Counter[tuple[str, int]] = Counter()
    for edge in ranked:
        limit = _top_k_limit(cfg, edge.relation)
        relevant_nodes = (edge.source, edge.target) if edge.relation == "near" else (edge.source,)
        if limit > 0 and any(topk_counts[(edge.relation, node_id)] >= limit for node_id in relevant_nodes):
            rejected.append(replace(edge, rejection_reason="top_k_per_relation"))
            continue
        relation_selected.append(edge)
        for node_id in relevant_nodes:
            topk_counts[(edge.relation, node_id)] += 1

    selected: list[SceneGraphEdge] = []
    out_degree: Counter[int] = Counter()
    in_degree: Counter[int] = Counter()
    for edge in relation_selected:
        out_nodes = (edge.source, edge.target) if edge.relation == "near" else (edge.source,)
        in_nodes = (edge.source, edge.target) if edge.relation == "near" else (edge.target,)
        if cfg.max_out_degree > 0 and any(out_degree[node_id] >= cfg.max_out_degree for node_id in out_nodes):
            rejected.append(replace(edge, rejection_reason="max_out_degree"))
            continue
        if cfg.max_in_degree > 0 and any(in_degree[node_id] >= cfg.max_in_degree for node_id in in_nodes):
            rejected.append(replace(edge, rejection_reason="max_in_degree"))
            continue
        if cfg.max_total_edges > 0 and len(selected) >= cfg.max_total_edges:
            rejected.append(replace(edge, rejection_reason="max_total_edges"))
            continue
        selected.append(edge)
        for node_id in out_nodes:
            out_degree[node_id] += 1
        for node_id in in_nodes:
            in_degree[node_id] += 1

    reason_counts = Counter(edge.rejection_reason for edge in rejected)
    return _sort_edges(selected), _sort_edges(rejected), {
        "before_pruning_edge_count": len(ranked),
        "after_relation_top_k_edge_count": len(relation_selected),
        "after_pruning_edge_count": len(selected),
        "pruned_edge_count": len(rejected),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "maximum_out_degree": max(out_degree.values(), default=0),
        "maximum_in_degree": max(in_degree.values(), default=0),
    }


def load_scene_graph_edges(path: str | Path | None) -> list[SceneGraphEdge]:
    if path is None or not Path(path).exists():
        return []
    edge_path = Path(path)
    if edge_path.suffix.lower() == ".json":
        payload = json.loads(edge_path.read_text(encoding="utf-8"))
    else:
        with edge_path.open("rb") as handle:
            payload = pickle.load(handle)
    if isinstance(payload, Mapping):
        payload = payload.get("edges", payload.get("accepted_edges", []))
    result: list[SceneGraphEdge] = []
    for item in payload or []:
        if isinstance(item, SceneGraphEdge):
            result.append(canonicalize_edge(item))
        elif isinstance(item, Mapping):
            edge = SceneGraphEdge.from_dict(item)
            edge.metadata.setdefault("edge_format", "structured")
            result.append(canonicalize_edge(edge))
        elif isinstance(item, (tuple, list)) and len(item) >= 3:
            edge = SceneGraphEdge(
                source=int(item[0]),
                target=int(item[1]),
                relation=str(item[2]),
                directed=str(item[2]).lower() != "near",
                confidence=1.0,
                coordinate_frame="world",
                relation_type="symmetric" if str(item[2]).lower() == "near" else "directional",
                metadata={
                    "edge_format": "legacy",
                    "metadata_available": False,
                    "legacy_direction_assumption": "source_relation_target",
                },
            )
            result.append(canonicalize_edge(edge))
        else:
            raise ValueError(f"Unsupported scene graph edge payload: {type(item)!r}")
    return _sort_edges(result)


def save_relation_result(result: RelationBuildResult, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def save_structured_edges(edges: Iterable[SceneGraphEdge], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [edge.to_dict() for edge in _sort_edges(edges)]
    if output_path.suffix.lower() == ".json":
        output_path.write_text(
            json.dumps({"schema_version": 1, "edges": payload}, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    else:
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle)
    return output_path


def save_legacy_edge_pickle(edges: Iterable[SceneGraphEdge], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [(edge.source, edge.target, edge.relation) for edge in _sort_edges(edges)]
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return output_path


def _coordinate_axes(
    frame_name: str,
    *,
    ego_heading: Sequence[float] | float | None,
    trajectory: Sequence[Sequence[float]] | np.ndarray | None,
    ego_position: Sequence[float] | None,
) -> tuple[tuple[np.ndarray, np.ndarray, float] | None, str | None]:
    if frame_name == "world":
        # World convention: +Y is forward, -X is left, and +X is right.
        return (np.asarray([0.0, 1.0]), np.asarray([-1.0, 0.0]), 0.90), None
    if frame_name == "ego_heading":
        heading = _heading_vector(ego_heading)
        if heading is None:
            return None, "missing_or_degenerate_ego_heading"
        left = np.asarray([-heading[1], heading[0]], dtype=float)
        return (heading, left, 1.0), None
    if frame_name == "local_trajectory":
        tangent = _trajectory_tangent(trajectory, ego_position=ego_position)
        if tangent is None:
            return None, "missing_or_degenerate_local_trajectory"
        left = np.asarray([-tangent[1], tangent[0]], dtype=float)
        return (tangent, left, 1.0), None
    return None, "unsupported_coordinate_frame"


def _heading_vector(value: Sequence[float] | float | None) -> np.ndarray | None:
    if value is None:
        return None
    if np.isscalar(value):
        angle = float(value)
        heading = np.asarray([math.cos(angle), math.sin(angle)], dtype=float)
    else:
        heading = np.asarray(value, dtype=float).reshape(-1)
        if heading.size < 2:
            return None
        heading = heading[:2]
    if not np.all(np.isfinite(heading)):
        return None
    norm = float(np.linalg.norm(heading))
    if norm <= 1e-9:
        return None
    return heading / norm


def _trajectory_tangent(
    trajectory: Sequence[Sequence[float]] | np.ndarray | None,
    *,
    ego_position: Sequence[float] | None,
) -> np.ndarray | None:
    if trajectory is None:
        return None
    points = np.asarray(trajectory, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return None
    points = points[:, :2]
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if len(points) < 2:
        return None
    if ego_position is None:
        index = len(points) - 1
    else:
        position = np.asarray(ego_position, dtype=float).reshape(-1)[:2]
        index = int(np.argmin(np.linalg.norm(points - position, axis=1)))
    before = max(0, index - 1)
    after = min(len(points) - 1, index + 1)
    tangent = points[after] - points[before]
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-9:
        diffs = np.diff(points, axis=0)
        norms = np.linalg.norm(diffs, axis=1)
        valid = np.flatnonzero(norms > 1e-9)
        if valid.size == 0:
            return None
        nearest = int(valid[np.argmin(np.abs(valid - min(index, len(diffs) - 1)))])
        tangent = diffs[nearest]
        norm = float(norms[nearest])
    return tangent / norm


def _directional_fact(
    source_node: RelationNode,
    target_node: RelationNode,
    delta_world: np.ndarray,
    config: RelationConfig,
    frame: CoordinateFrameResolution,
) -> tuple[RelationNode, RelationNode, str, float] | None:
    forward = float(np.dot(delta_world, frame.forward_axis_world))
    left = float(np.dot(delta_world, frame.left_axis_world))
    if max(abs(forward), abs(left)) <= 1e-9:
        return None
    if abs(left) > abs(forward):
        primary, secondary = abs(left), abs(forward)
        relation = "left_of"
        fact_source, fact_target = (
            (target_node, source_node) if left > 0 else (source_node, target_node)
        )
    else:
        primary, secondary = abs(forward), abs(left)
        relation = "in_front_of"
        fact_source, fact_target = (
            (target_node, source_node) if forward > 0 else (source_node, target_node)
        )
    angle_deg = math.degrees(math.atan2(secondary, primary))
    threshold = max(1e-6, config.direction_angle_threshold_deg)
    if angle_deg > threshold:
        return None
    angle_margin = max(0.0, min(1.0, (threshold - angle_deg) / threshold))
    return fact_source, fact_target, relation, angle_margin


def _make_edge(
    source_node: RelationNode,
    target_node: RelationNode,
    *,
    relation: str,
    distance_m: float,
    angle_margin: float,
    threshold_m: float,
    config: RelationConfig,
    frame: CoordinateFrameResolution,
    frame_index: int | None,
) -> SceneGraphEdge:
    observation_count = max(1, min(source_node.observation_count, target_node.observation_count))
    first_seen = _maximum_optional([source_node.first_seen_frame, target_node.first_seen_frame])
    last_seen = _minimum_optional([source_node.last_seen_frame, target_node.last_seen_frame])
    if frame_index is not None:
        first_seen = frame_index if first_seen is None else first_seen
        last_seen = frame_index
    span = max(1, (last_seen - first_seen + 1) if first_seen is not None and last_seen is not None else 1)
    stability = min(1.0, observation_count / max(config.static_stability_frames, span))
    observation_score = min(1.0, observation_count / config.static_stability_frames)
    dynamic = source_node.dynamic or target_node.dynamic
    distance_score = max(0.0, 1.0 - distance_m / max(threshold_m, 1e-9))
    components = {
        "distance": distance_score,
        "angle_margin": float(angle_margin),
        "observation_count": observation_score,
        "stability": stability,
        "coordinate_frame": frame.reliability,
        "dynamic_state": 0.70 if dynamic else 1.0,
    }
    confidence = _weighted_confidence(components, config.confidence_weights)
    return canonicalize_edge(
        SceneGraphEdge(
            source=source_node.node_id,
            target=target_node.node_id,
            relation=relation,
            directed=relation != "near",
            confidence=confidence,
            distance_m=distance_m,
            coordinate_frame=str(frame.actual_frame),
            relation_type="symmetric" if relation == "near" else "directional",
            first_seen_frame=first_seen,
            last_seen_frame=last_seen,
            observation_count=observation_count,
            metadata={
                "confidence_components": components,
                "confidence_weights": config.confidence_weights,
                "requested_coordinate_frame": frame.requested_frame,
                "coordinate_frame_fallback_reason": frame.fallback_reason,
                "dynamic": dynamic,
                "valid_frame_range": [first_seen, last_seen],
                "edge_format": "structured",
                "metadata_available": True,
                "observation_source": "object_geometry",
                "computed_frame": frame_index,
                "source_observed_frame": source_node.last_seen_frame,
                "target_observed_frame": target_node.last_seen_frame,
                "source_timestamp_s": source_node.metadata.get("timestamp_s"),
                "target_timestamp_s": target_node.metadata.get("timestamp_s"),
                "forward_axis_world": frame.forward_axis_world.tolist(),
                "left_axis_world": frame.left_axis_world.tolist(),
            },
        )
    )


def _accept_by_confidence(
    edge: SceneGraphEdge,
    config: RelationConfig,
    accepted: list[SceneGraphEdge],
    rejected: list[SceneGraphEdge],
) -> None:
    if edge.confidence + 1e-12 < config.minimum_confidence:
        rejected.append(replace(edge, rejection_reason="below_minimum_confidence"))
    else:
        accepted.append(edge)


def _weighted_confidence(components: Mapping[str, float], weights: Mapping[str, float]) -> float:
    positive_weights = {key: max(0.0, float(value)) for key, value in weights.items()}
    denominator = sum(positive_weights.values())
    if denominator <= 1e-12:
        return 0.0
    numerator = sum(positive_weights.get(key, 0.0) * float(value) for key, value in components.items())
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _coerce_relation_node(raw: Any, *, fallback_id: Any, frame_index: int | None) -> RelationNode:
    if isinstance(raw, RelationNode):
        return raw
    if isinstance(raw, Mapping):
        node_id = int(raw.get("node_id", raw.get("id", fallback_id)))
        center = raw.get("center", raw.get("bbox_center"))
        if center is None and isinstance(raw.get("bbox"), Mapping):
            center = raw["bbox"].get("center")
        dynamic = bool(
            raw.get("dynamic", False)
            or raw.get("is_short_term", False)
            or str(raw.get("motion_state", "")).lower() == "dynamic"
        )
        metadata = dict(raw.get("metadata", {}) or {})
        observation_count = int(
            raw.get(
                "observation_count",
                raw.get("num_detections", len(raw.get("frame_history", [])) or 1),
            )
            or 1
        )
        first_seen = raw.get("first_seen_frame")
        if first_seen is None and raw.get("frame_history"):
            first_seen = min(int(item) for item in raw["frame_history"])
        last_seen = raw.get("last_seen_frame")
        if last_seen is None and raw.get("frame_history"):
            last_seen = max(int(item) for item in raw["frame_history"])
        return RelationNode(
            node_id=node_id,
            center=np.asarray(center, dtype=float),
            dynamic=dynamic,
            observation_count=observation_count,
            first_seen_frame=_optional_int(first_seen if first_seen is not None else frame_index),
            last_seen_frame=_optional_int(last_seen if last_seen is not None else frame_index),
            metadata=metadata,
        )
    node_id = int(getattr(raw, "node_id", getattr(raw, "id", fallback_id)))
    center = getattr(raw, "center", getattr(raw, "bbox_center", None))
    dynamic = bool(
        getattr(raw, "dynamic", False)
        or getattr(raw, "is_short_term", False)
        or str(getattr(raw, "motion_state", "")).lower() == "dynamic"
    )
    metadata = dict(getattr(raw, "metadata", {}) or {})
    return RelationNode(
        node_id=node_id,
        center=np.asarray(center, dtype=float),
        dynamic=dynamic,
        observation_count=int(getattr(raw, "observation_count", 1) or 1),
        first_seen_frame=_optional_int(getattr(raw, "first_seen_frame", frame_index)),
        last_seen_frame=_optional_int(getattr(raw, "last_seen_frame", frame_index)),
        metadata=metadata,
    )


def _distance_threshold(config: RelationConfig, relation: str) -> float:
    mapping = config.relation_distance_thresholds_m
    if relation in mapping:
        return max(0.0, float(mapping[relation]))
    if relation in DIRECTIONAL_RELATIONS:
        return max(0.0, float(mapping.get("directional", mapping.get("default", 12.0))))
    return max(0.0, float(mapping.get("default", 8.0)))


def _top_k_limit(config: RelationConfig, relation: str) -> int:
    mapping = config.top_k_per_relation
    if relation in mapping:
        return int(mapping[relation])
    if relation == "left_of":
        return max(int(mapping.get("left_of", 0)), int(mapping.get("right_of", 0)))
    if relation == "in_front_of":
        return max(int(mapping.get("in_front_of", 0)), int(mapping.get("behind", 0)))
    return int(mapping.get("default", 0))


def _edge_rank_key(edge: SceneGraphEdge) -> tuple[Any, ...]:
    stability = _edge_stability(edge)
    distance_rank = -float(edge.distance_m) if edge.distance_m is not None else float("-inf")
    return (
        round(float(edge.confidence), 12),
        round(stability, 12),
        int(edge.observation_count),
        distance_rank,
        -_relation_order(edge.relation),
        -int(edge.source),
        -int(edge.target),
    )


def _edge_stability(edge: SceneGraphEdge) -> float:
    if edge.first_seen_frame is None or edge.last_seen_frame is None:
        return min(1.0, float(edge.observation_count))
    span = max(1, edge.last_seen_frame - edge.first_seen_frame + 1)
    return min(1.0, edge.observation_count / span)


def _sort_edges(edges: Iterable[SceneGraphEdge]) -> list[SceneGraphEdge]:
    return sorted(
        edges,
        key=lambda edge: (
            _relation_order(edge.relation),
            int(edge.source),
            int(edge.target),
            -round(float(edge.confidence), 12),
            str(edge.rejection_reason or ""),
        ),
    )


def _relation_statistics(edges: Iterable[SceneGraphEdge]) -> dict[str, Any]:
    edges = list(edges)
    counts = Counter(edge.relation for edge in edges)
    return {
        "by_relation": dict(sorted(counts.items())),
        "symmetric_edge_count": sum(not edge.directed for edge in edges),
        "directed_edge_count": sum(edge.directed for edge in edges),
        "static_edge_count": sum(not bool(edge.metadata.get("dynamic", False)) for edge in edges),
        "dynamic_edge_count": sum(bool(edge.metadata.get("dynamic", False)) for edge in edges),
    }


def _empty_conflict_statistics() -> dict[str, Any]:
    return {
        "conflict_groups_detected": 0,
        "conflicting_edges_removed": 0,
        "left_right_conflicts": 0,
        "front_behind_conflicts": 0,
        "duplicate_edges_merged": 0,
        "rejection_reasons": {},
    }


def _empty_pruning_statistics() -> dict[str, Any]:
    return {
        "before_pruning_edge_count": 0,
        "after_relation_top_k_edge_count": 0,
        "after_pruning_edge_count": 0,
        "pruned_edge_count": 0,
        "rejection_reasons": {},
        "maximum_out_degree": 0,
        "maximum_in_degree": 0,
    }


def _relation_axis(relation: str) -> str | None:
    if relation == "left_of":
        return "lateral"
    if relation == "in_front_of":
        return "longitudinal"
    return None


def _relation_order(relation: str) -> int:
    return CANONICAL_RELATION_ORDER.get(str(relation), 99)


def _minimum_optional(values: Iterable[int | None]) -> int | None:
    present = [int(value) for value in values if value is not None]
    return min(present) if present else None


def _maximum_optional(values: Iterable[int | None]) -> int | None:
    present = [int(value) for value in values if value is not None]
    return max(present) if present else None


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _safe_node_id(raw: Any, fallback: Any) -> int:
    try:
        if isinstance(raw, Mapping):
            return int(raw.get("node_id", raw.get("id", fallback)))
        return int(getattr(raw, "node_id", getattr(raw, "id", fallback)))
    except (TypeError, ValueError):
        return -1


def _plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    try:
        from omegaconf import OmegaConf

        return dict(OmegaConf.to_container(value, resolve=True))
    except (ImportError, TypeError, ValueError):
        raise TypeError(f"Relation configuration must be a mapping, got {type(value)!r}") from None


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = {str(key): _plain_value(value) for key, value in base.items()}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = _plain_value(value)
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
