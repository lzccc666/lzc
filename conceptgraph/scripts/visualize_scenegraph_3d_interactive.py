from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from conceptgraph.scenegraph.relations import load_scene_graph_edges
from conceptgraph.scenegraph.visualization import (
    edge_alpha,
    edge_hover_text,
    edge_is_highlighted,
    edge_width,
    filter_edges,
    navigation_relation_keys,
    parse_relation_types,
    relation_style,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive 3D scene graph visualization.")
    parser.add_argument("--scene-graph", required=True)
    parser.add_argument("--edges", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vehicle-pose", default=None, help="Optional vehicle pose x,y,z to plot")
    parser.add_argument("--relation-types", default=None, help="Comma-separated relation filter")
    parser.add_argument("--minimum-confidence", type=float, default=0.0)
    parser.add_argument("--navigation-plan", default=None)
    parser.add_argument("--navigation-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = json.load(open(args.scene_graph, "r", encoding="utf-8"))
    edges = filter_edges(
        load_scene_graph_edges(args.edges),
        relation_types=parse_relation_types(args.relation_types),
        minimum_confidence=0.0,
    )
    navigation_plan = None
    if args.navigation_plan and Path(args.navigation_plan).exists():
        navigation_plan = json.loads(Path(args.navigation_plan).read_text(encoding="utf-8"))
    navigation_keys = navigation_relation_keys(navigation_plan)
    vehicle_pose = None
    if args.vehicle_pose:
        vehicle_pose = np.asarray([float(v) for v in args.vehicle_pose.split(",")], dtype=float)

    fig = go.Figure()
    centers = np.asarray([np.asarray(node.get("bbox_center", [0, 0, 0]), dtype=float) for node in nodes], dtype=float)
    xy_span = float(np.max(np.ptp(centers[:, :2], axis=0))) if len(centers) else 1.0
    ground_z = float(np.percentile(centers[:, 2], 2)) if len(centers) else 0.0

    node_lookup = {}
    for node in nodes:
        node_id = int(node["id"])
        tag = str(node.get("object_tag", "unknown"))
        center = np.asarray(node.get("bbox_center", [0, 0, 0]), dtype=float)
        z = display_height(center[2], tag, ground_z, xy_span)
        node_lookup[node_id] = np.asarray([center[0], center[1], z], dtype=float)

    legend_relations = set()
    for edge in edges:
        if edge.source not in node_lookup or edge.target not in node_lookup:
            continue
        p0 = node_lookup[edge.source]
        p1 = node_lookup[edge.target]
        style = relation_style(edge.relation)
        highlighted = edge_is_highlighted(edge, navigation_keys)
        visible = edge.confidence + 1e-12 >= args.minimum_confidence
        opacity = edge_alpha(edge)
        if args.navigation_only:
            opacity = opacity if highlighted else 0.04
        show_legend = edge.relation not in legend_relations
        legend_relations.add(edge.relation)
        trace_meta = {
            "kind": "edge",
            "source": edge.source,
            "target": edge.target,
            "relation": edge.relation,
            "confidence": edge.confidence,
            "highlighted": highlighted,
        }
        fig.add_trace(
            go.Scatter3d(
                x=[p0[0], p1[0]],
                y=[p0[1], p1[1]],
                z=[p0[2], p1[2]],
                mode="lines",
                line=dict(
                    color=style["color"],
                    width=edge_width(edge),
                    dash=style["plotly"],
                ),
                opacity=opacity,
                hovertext=edge_hover_text(edge),
                hoverinfo="text",
                name=edge.relation,
                legendgroup=edge.relation,
                showlegend=show_legend,
                visible=visible,
                meta=trace_meta,
            )
        )
        if edge.directed:
            delta = p1 - p0
            norm = float(np.linalg.norm(delta))
            if norm > 1e-9:
                direction = delta / norm
                tip = p0 + 0.84 * delta
                fig.add_trace(
                    go.Cone(
                        x=[tip[0]],
                        y=[tip[1]],
                        z=[tip[2]],
                        u=[direction[0]],
                        v=[direction[1]],
                        w=[direction[2]],
                        anchor="tip",
                        sizemode="absolute",
                        sizeref=max(0.45, xy_span * 0.012),
                        colorscale=[[0, style["color"]], [1, style["color"]]],
                        showscale=False,
                        hoverinfo="skip",
                        opacity=opacity,
                        legendgroup=edge.relation,
                        showlegend=False,
                        visible=visible,
                        meta=trace_meta,
                    )
                )

    colors = [tag_to_color(str(node.get("object_tag", ""))) for node in nodes]
    fig.add_trace(
        go.Scatter3d(
            x=[node_lookup[int(node["id"])][0] for node in nodes],
            y=[node_lookup[int(node["id"])][1] for node in nodes],
            z=[node_lookup[int(node["id"])][2] for node in nodes],
            mode="markers+text",
            text=[str(node["id"]) for node in nodes],
            textposition="top center",
            marker=dict(size=8, color=colors, line=dict(color="#111827", width=1)),
            hovertext=[
                node_hover_text(node)
                for node in nodes
            ],
            hoverinfo="text",
            name="节点",
            customdata=[int(node["id"]) for node in nodes],
            meta={"kind": "node"},
        )
    )

    if vehicle_pose is not None and vehicle_pose.size >= 3:
        ego_z = max(float(xy_span) * 0.015, 1.05)
        fig.add_trace(
            go.Scatter3d(
                x=[vehicle_pose[0]],
                y=[vehicle_pose[1]],
                z=[ego_z],
                mode="markers+text",
                text=["自车"],
                textposition="top center",
                marker=dict(size=10, color="#111827", symbol="diamond"),
                hoverinfo="skip",
                name="自车",
            )
        )

    fig.update_layout(
        title="场景图三维交互视图",
        width=1500,
        height=940,
        scene=dict(
            xaxis_title="x (米)",
            yaxis_title="y (米)",
            zaxis_title="z (米)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.4, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", groupclick="togglegroup", y=1.02, x=0.0),
    )

    html_path = out_dir / "scenegraph_3d_interactive.html"
    plot_html = fig.to_html(include_plotlyjs=True, full_html=False)
    fallback_figure = build_fallback_figure(
        nodes,
        edges,
        minimum_confidence=args.minimum_confidence,
        navigation_keys=navigation_keys,
        navigation_only=args.navigation_only,
    )
    fallback_html = fallback_figure.to_html(include_plotlyjs=False, full_html=False)
    relation_controls = "".join(
        f'<label><input class="relation-filter" type="checkbox" value="{relation}" checked> {relation}</label>'
        for relation in sorted(legend_relations)
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>阶段3场景图三维交互视图</title>
  <style>
    body {{ margin: 0; color: #17212b; font-family: Arial, sans-serif; background: #ffffff; }}
    .controls {{ display: flex; align-items: center; gap: 18px; min-height: 48px; padding: 8px 18px; border-bottom: 1px solid #d1d5db; flex-wrap: wrap; }}
    .controls label {{ white-space: nowrap; }}
    #confidence-value {{ display: inline-block; min-width: 42px; font-variant-numeric: tabular-nums; }}
    .status {{ margin-left: auto; color: #4b5563; }}
  </style>
</head>
<body>
  <div class="controls" aria-label="关系图过滤器">
    <strong>Relations</strong>{relation_controls}
    <label>Minimum confidence <input id="confidence-filter" type="range" min="0" max="1" step="0.01" value="{args.minimum_confidence:.2f}"></label>
    <span id="confidence-value">{args.minimum_confidence:.2f}</span>
    <button id="reset-focus" type="button">Show all neighbors</button>
    <span class="status" id="focus-status">Click a node to focus its relations</span>
  </div>
  <div id="webgl-container">{plot_html}</div>
  <div id="fallback-container" style="display:none">{fallback_html}</div>
  <script>
    function supportsWebGL() {{
      try {{
        const canvas = document.createElement('canvas');
        return Boolean(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
      }} catch (error) {{ return false; }}
    }}
    const useFallback = !supportsWebGL();
    document.getElementById('webgl-container').style.display = useFallback ? 'none' : 'block';
    document.getElementById('fallback-container').style.display = useFallback ? 'block' : 'none';
    const graph = document.querySelector((useFallback ? '#fallback-container ' : '#webgl-container ') + '.plotly-graph-div');
    window.scenegraphGraph = graph;
    window.scenegraphUsingFallback = useFallback;
    const confidence = document.getElementById('confidence-filter');
    const confidenceValue = document.getElementById('confidence-value');
    const status = document.getElementById('focus-status');
    let focusedNode = null;
    function enabledRelations() {{
      return new Set(Array.from(document.querySelectorAll('.relation-filter:checked')).map(el => el.value));
    }}
    function applyFilters() {{
      const allowed = enabledRelations();
      const threshold = Number(confidence.value);
      confidenceValue.textContent = threshold.toFixed(2);
      graph.data.forEach((trace, index) => {{
        const meta = trace.meta || {{}};
        if (meta.kind !== 'edge') return;
        const visible = allowed.has(meta.relation) && Number(meta.confidence) >= threshold;
        const incident = focusedNode === null || meta.source === focusedNode || meta.target === focusedNode;
        const baseOpacity = 0.18 + 0.78 * Number(meta.confidence);
        Plotly.restyle(graph, {{visible: visible, opacity: incident ? baseOpacity : 0.035}}, [index]);
      }});
    }}
    document.querySelectorAll('.relation-filter').forEach(el => el.addEventListener('change', applyFilters));
    confidence.addEventListener('input', applyFilters);
    document.getElementById('reset-focus').addEventListener('click', () => {{
      focusedNode = null;
      status.textContent = 'Click a node to focus its relations';
      applyFilters();
    }});
    graph.on('plotly_click', event => {{
      const point = event.points && event.points[0];
      if (!point || !point.data || !point.data.meta || point.data.meta.kind !== 'node') return;
      focusedNode = Number(point.customdata);
      status.textContent = `Focused node: ${{focusedNode}}`;
      applyFilters();
    }});
    applyFilters();
  </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def build_fallback_figure(
    nodes: list[dict],
    edges,
    *,
    minimum_confidence: float,
    navigation_keys: set[tuple[int, int, str]],
    navigation_only: bool,
) -> go.Figure:
    figure = go.Figure()
    centers = {
        int(node["id"]): np.asarray(node.get("bbox_center", [0, 0, 0]), dtype=float)[:2]
        for node in nodes
    }
    all_points = np.asarray(list(centers.values()), dtype=float)
    span = float(np.max(np.ptp(all_points, axis=0))) if len(all_points) else 1.0
    arrow_size = max(0.35, span * 0.007)
    legend_relations = set()
    for edge in edges:
        if edge.source not in centers or edge.target not in centers:
            continue
        source = centers[edge.source]
        target = centers[edge.target]
        style = relation_style(edge.relation)
        highlighted = edge_is_highlighted(edge, navigation_keys)
        visible = edge.confidence + 1e-12 >= minimum_confidence
        opacity = edge_alpha(edge)
        if navigation_only:
            opacity = opacity if highlighted else 0.04
        show_legend = edge.relation not in legend_relations
        legend_relations.add(edge.relation)
        meta = {
            "kind": "edge",
            "source": edge.source,
            "target": edge.target,
            "relation": edge.relation,
            "confidence": edge.confidence,
            "highlighted": highlighted,
        }
        figure.add_trace(
            go.Scatter(
                x=[source[0], target[0]],
                y=[source[1], target[1]],
                mode="lines",
                line={
                    "color": style["color"],
                    "width": edge_width(edge),
                    "dash": style["plotly"],
                },
                opacity=opacity,
                hovertext=edge_hover_text(edge),
                hoverinfo="text",
                name=edge.relation,
                legendgroup=edge.relation,
                showlegend=show_legend,
                visible=visible,
                meta=meta,
            )
        )
        delta = target - source
        norm = float(np.linalg.norm(delta))
        if edge.directed and norm > 1e-9:
            direction = delta / norm
            perpendicular = np.asarray([-direction[1], direction[0]])
            tip = source + 0.88 * delta
            base = tip - direction * arrow_size
            polygon = np.vstack(
                [tip, base + perpendicular * arrow_size * 0.45, base - perpendicular * arrow_size * 0.45, tip]
            )
            figure.add_trace(
                go.Scatter(
                    x=polygon[:, 0],
                    y=polygon[:, 1],
                    mode="lines",
                    line={"color": style["color"], "width": 1},
                    fill="toself",
                    fillcolor=style["color"],
                    opacity=opacity,
                    hoverinfo="skip",
                    legendgroup=edge.relation,
                    showlegend=False,
                    visible=visible,
                    meta=meta,
                )
            )
    figure.add_trace(
        go.Scatter(
            x=[centers[int(node["id"])][0] for node in nodes],
            y=[centers[int(node["id"])][1] for node in nodes],
            mode="markers+text",
            text=[str(node["id"]) for node in nodes],
            textposition="top center",
            marker={
                "size": 10,
                "color": [tag_to_color(str(node.get("object_tag", ""))) for node in nodes],
                "line": {"color": "#111827", "width": 1},
            },
            hovertext=[node_hover_text(node) for node in nodes],
            hoverinfo="text",
            name="nodes",
            customdata=[int(node["id"]) for node in nodes],
            meta={"kind": "node"},
        )
    )
    figure.update_layout(
        title="Interactive scene graph BEV",
        width=1500,
        height=940,
        xaxis={"title": "x (m)", "scaleanchor": "y", "scaleratio": 1},
        yaxis={"title": "y (m)"},
        margin={"l": 60, "r": 20, "t": 60, "b": 60},
        legend={"orientation": "h", "groupclick": "togglegroup", "y": 1.02, "x": 0.0},
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#FFFFFF",
    )
    return figure


def tag_to_color(tag: str) -> str:
    colors = {
        "car": "#3b82f6",
        "truck": "#f97316",
        "bus": "#f59e0b",
        "motorcycle": "#ef4444",
        "bicycle": "#8b5cf6",
        "pedestrian": "#ec4899",
        "traffic sign": "#10b981",
        "traffic light": "#06b6d4",
        "road": "#6b7280",
        "sidewalk": "#84cc16",
        "building": "#a855f7",
        "tree": "#16a34a",
        "fence": "#f43f5e",
        "pole": "#14b8a6",
    }
    return colors.get(tag.lower(), "#94a3b8")


def tag_to_cn(tag: str) -> str:
    mapping = {
        "car": "小汽车",
        "truck": "卡车",
        "bus": "公交车",
        "motorcycle": "摩托车",
        "bicycle": "自行车",
        "pedestrian": "行人",
        "traffic sign": "交通标志",
        "traffic light": "红绿灯",
        "road": "道路",
        "sidewalk": "人行道",
        "building": "建筑",
        "tree": "树",
        "fence": "围栏",
        "pole": "杆",
        "vehicle": "车辆",
        "object": "物体",
        "unknown": "未知",
    }
    return mapping.get(tag.lower(), tag)


def node_hover_text(node: dict) -> str:
    tag = str(node.get("object_tag", "unknown"))
    attrs = node.get("attributes", {}) if isinstance(node.get("attributes", {}), dict) else {}
    colors = colors_to_cn(attrs.get("colors", []))
    motion = motion_to_cn(node.get("motion_state", "unknown"))
    caption = str(node.get("caption", node.get("summary", "")))[:180]
    return (
        f"ID: {node.get('id')}"
        f"<br>英文: {tag}"
        f"<br>中文: {tag_to_cn(tag)}"
        f"<br>状态: {motion}"
        f"<br>颜色: {colors}"
        f"<br>描述: {caption}"
    )


def motion_to_cn(value: str) -> str:
    mapping = {"static": "静态", "dynamic": "动态", "unknown": "未知"}
    return mapping.get(str(value).lower().strip(), str(value))


def colors_to_cn(colors) -> str:
    if isinstance(colors, str):
        colors = [colors]
    mapping = {
        "black": "黑色",
        "white": "白色",
        "gray": "灰色",
        "grey": "灰色",
        "silver": "银色",
        "red": "红色",
        "blue": "蓝色",
        "green": "绿色",
        "yellow": "黄色",
        "orange": "橙色",
    }
    labels = [mapping.get(str(c).lower(), str(c)) for c in (colors or []) if str(c).strip()]
    return "/".join(labels[:2]) if labels else "-"


def category_height_offset(tag: str, xy_span: float) -> float:
    order = {
        "road": 0,
        "sidewalk": 1,
        "car": 2,
        "truck": 3,
        "bus": 4,
        "motorcycle": 5,
        "bicycle": 6,
        "pedestrian": 7,
        "traffic sign": 8,
        "traffic light": 9,
        "pole": 10,
        "fence": 11,
        "tree": 12,
        "building": 13,
    }
    step = max(float(xy_span) * 0.004, 0.15)
    return (order.get(tag.lower(), 2) - 2) * step


def display_height(raw_z: float, tag: str, ground_z: float, xy_span: float) -> float:
    base = max(0.0, float(raw_z) - float(ground_z)) * 1.15
    offset = max(0.0, category_height_offset(tag, xy_span)) * 0.08
    return base + offset + max(float(xy_span) * 0.006, 0.35)


if __name__ == "__main__":
    main()
