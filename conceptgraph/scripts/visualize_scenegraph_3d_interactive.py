from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive 3D scene graph visualization.")
    parser.add_argument("--scene-graph", required=True)
    parser.add_argument("--edges", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vehicle-pose", default=None, help="Optional vehicle pose x,y,z to plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = json.load(open(args.scene_graph, "r", encoding="utf-8"))
    edges = []
    if args.edges and Path(args.edges).exists():
        with open(args.edges, "rb") as f:
            edges = pickle.load(f)
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

    for src, dst, relation in edges:
        if int(src) not in node_lookup or int(dst) not in node_lookup:
            continue
        p0 = node_lookup[int(src)]
        p1 = node_lookup[int(dst)]
        fig.add_trace(
            go.Scatter3d(
                x=[p0[0], p1[0]],
                y=[p0[1], p1[1]],
                z=[p0[2], p1[2]],
                mode="lines",
                line=dict(color="#64748b", width=4),
                hoverinfo="skip",
                showlegend=False,
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
        width=1600,
        height=1100,
        scene=dict(
            xaxis_title="x (米)",
            yaxis_title="y (米)",
            zaxis_title="z (米)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.4, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h"),
    )

    html_path = out_dir / "scenegraph_3d_interactive.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)


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
