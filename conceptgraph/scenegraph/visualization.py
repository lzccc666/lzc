from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from conceptgraph.scenegraph.relations import SceneGraphEdge, canonicalize_edge


RELATION_STYLES: dict[str, dict[str, str]] = {
    "near": {"color": "#5B6770", "matplotlib": "-", "plotly": "solid"},
    "left_of": {"color": "#0072B2", "matplotlib": ":", "plotly": "dot"},
    "right_of": {"color": "#56B4E9", "matplotlib": ":", "plotly": "dot"},
    "in_front_of": {"color": "#009E73", "matplotlib": "--", "plotly": "dash"},
    "behind": {"color": "#D55E00", "matplotlib": "--", "plotly": "dash"},
}


def relation_style(relation: str) -> dict[str, str]:
    return RELATION_STYLES.get(
        str(relation).strip().lower(),
        {"color": "#6B7280", "matplotlib": "-", "plotly": "solid"},
    )


def edge_width(edge: SceneGraphEdge) -> float:
    return 0.8 + 3.0 * float(edge.confidence)


def edge_alpha(edge: SceneGraphEdge) -> float:
    return 0.18 + 0.78 * float(edge.confidence)


def filter_edges(
    edges: Iterable[SceneGraphEdge],
    *,
    relation_types: Iterable[str] | None = None,
    minimum_confidence: float = 0.0,
) -> list[SceneGraphEdge]:
    allowed = None
    if relation_types is not None:
        allowed = {str(item).strip().lower() for item in relation_types if str(item).strip()}
    result = []
    for raw_edge in edges:
        edge = canonicalize_edge(raw_edge)
        if allowed is not None and edge.relation not in allowed:
            continue
        if edge.confidence + 1e-12 < float(minimum_confidence):
            continue
        result.append(edge)
    return sorted(result, key=lambda edge: (edge.relation, edge.source, edge.target))


def navigation_relation_keys(plan: dict[str, Any] | None) -> set[tuple[int, int, str]]:
    if not plan:
        return set()
    relation_query = plan.get("debug", {}).get("relation_query", {})
    chain = relation_query.get("relation_chain", [])
    keys = set()
    for item in chain:
        if item.get("stored_source") is None or item.get("stored_target") is None:
            continue
        keys.add(
            (
                int(item["stored_source"]),
                int(item["stored_target"]),
                str(item.get("stored_relation", item.get("relation", ""))).strip().lower(),
            )
        )
    return keys


def edge_is_highlighted(edge: SceneGraphEdge, keys: set[tuple[int, int, str]]) -> bool:
    edge = canonicalize_edge(edge)
    return (edge.source, edge.target, edge.relation) in keys


def edge_hover_text(edge: SceneGraphEdge) -> str:
    first = "-" if edge.first_seen_frame is None else str(edge.first_seen_frame)
    last = "-" if edge.last_seen_frame is None else str(edge.last_seen_frame)
    distance = "-" if edge.distance_m is None else f"{edge.distance_m:.3f} m"
    fallback = edge.metadata.get("coordinate_frame_fallback_reason") or "无"
    return (
        f"source: {edge.source}"
        f"<br>target: {edge.target}"
        f"<br>relation: {edge.relation}"
        f"<br>distance: {distance}"
        f"<br>confidence: {edge.confidence:.4f}"
        f"<br>coordinate_frame: {edge.coordinate_frame}"
        f"<br>observations: {edge.observation_count}"
        f"<br>valid_frames: [{first}, {last}]"
        f"<br>frame_fallback: {fallback}"
    )


def parse_relation_types(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def absolute_output(path: str | Path) -> str:
    return str(Path(path).resolve())
