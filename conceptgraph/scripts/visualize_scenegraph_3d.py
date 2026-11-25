from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

from conceptgraph.scenegraph.relations import load_scene_graph_edges
from conceptgraph.scenegraph.visualization import (
    edge_alpha,
    edge_width,
    filter_edges,
    parse_relation_types,
    relation_style,
)


CHINESE_FONT = FontProperties(fname="/root/autodl-tmp/lizichong/checkpoint/fonts/NotoSansCJKsc-Regular.otf")


def configure_chinese_font() -> None:
    font_path = Path("/root/autodl-tmp/lizichong/checkpoint/fonts/NotoSansCJKsc-Regular.otf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        global CHINESE_FONT
        CHINESE_FONT = FontProperties(fname=str(font_path))
        plt.rcParams["font.family"] = [CHINESE_FONT.get_name()]
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a ConceptGraphs scene graph in 3D.")
    parser.add_argument("--scene-graph", required=True)
    parser.add_argument("--edges", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vehicle-pose", default=None, help="Optional vehicle pose x,y,z to plot")
    parser.add_argument("--relation-types", default=None, help="Comma-separated relation filter")
    parser.add_argument("--minimum-confidence", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    configure_chinese_font()
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = json.load(open(args.scene_graph, "r", encoding="utf-8"))
    edges = filter_edges(
        load_scene_graph_edges(args.edges),
        relation_types=parse_relation_types(args.relation_types),
        minimum_confidence=args.minimum_confidence,
    )
    vehicle_pose = None
    if args.vehicle_pose:
        vehicle_pose = np.asarray([float(v) for v in args.vehicle_pose.split(",")], dtype=float)

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection="3d")

    node_rows = []
    centers_xy = np.asarray([np.asarray(node.get("bbox_center", [0, 0, 0]), dtype=float)[:2] for node in nodes], dtype=float)
    xy_span = float(np.max(np.ptp(centers_xy, axis=0))) if len(centers_xy) else 1.0
    raw_z = np.asarray([np.asarray(node.get("bbox_center", [0, 0, 0]), dtype=float)[2] for node in nodes], dtype=float)
    ground_z = float(np.percentile(raw_z, 2)) if len(raw_z) else 0.0

    for node in nodes:
        center = np.asarray(node.get("bbox_center", [0, 0, 0]), dtype=float)
        center_3d = center.copy()
        tag = str(node.get("object_tag", ""))
        center_3d[2] = display_height(center_3d[2], tag, ground_z, xy_span)
        color = tag_to_color(tag)
        ax.scatter(center_3d[0], center_3d[1], center_3d[2], s=75, c=color)
        ax.text(center_3d[0], center_3d[1], center_3d[2], f" {node['id']}", fontsize=8, fontproperties=CHINESE_FONT)
        attrs = node.get("attributes", {}) if isinstance(node.get("attributes", {}), dict) else {}
        node_rows.append(
            (
                int(node["id"]),
                tag,
                tag_to_cn(tag),
                motion_to_cn(node.get("motion_state", "unknown")),
                colors_to_cn(attrs.get("colors", [])),
            )
        )

    node_map = {}
    for node in nodes:
        center = np.asarray(node.get("bbox_center", [0, 0, 0]), dtype=float)
        center_3d = center.copy()
        center_3d[2] = display_height(center_3d[2], str(node.get("object_tag", "")), ground_z, xy_span)
        node_map[int(node["id"])] = center_3d
    for edge in edges:
        if edge.source in node_map and edge.target in node_map:
            p0 = node_map[edge.source]
            p1 = node_map[edge.target]
            style = relation_style(edge.relation)
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                linewidth=edge_width(edge),
                color=style["color"],
                linestyle=style["matplotlib"],
                alpha=edge_alpha(edge),
            )
            if edge.directed:
                delta = p1 - p0
                start = p0 + 0.72 * delta
                arrow = 0.20 * delta
                ax.quiver(
                    start[0],
                    start[1],
                    start[2],
                    arrow[0],
                    arrow[1],
                    arrow[2],
                    color=style["color"],
                    linewidth=max(1.0, edge_width(edge) * 0.7),
                    alpha=edge_alpha(edge),
                    arrow_length_ratio=0.45,
                    normalize=False,
                )

    if vehicle_pose is not None and vehicle_pose.size >= 3:
        ego_z = max(float(xy_span) * 0.015, 1.05)
        ax.scatter(vehicle_pose[0], vehicle_pose[1], ego_z, s=180, c="#111827", marker="*")
        ax.text(vehicle_pose[0], vehicle_pose[1], ego_z, " 自车", fontsize=10, color="#111827", fontproperties=CHINESE_FONT)

    ax.set_xlabel("x (米)", fontproperties=CHINESE_FONT)
    ax.set_ylabel("y (米)", fontproperties=CHINESE_FONT)
    ax.set_zlabel("z (米)", fontproperties=CHINESE_FONT)
    ax.set_title("场景图三维视图", fontproperties=CHINESE_FONT)
    ax.view_init(elev=28, azim=-58)
    ax.set_proj_type("persp", focal_length=0.8)
    set_axes_equal(ax)

    table_ax = fig.add_axes([0.68, 0.18, 0.30, 0.76])
    table_ax.axis("off")
    table_rows = [["编号", "英文标签", "中文", "状态", "颜色"]]
    table_rows.extend([[str(idx), eng, cn, motion, color] for idx, eng, cn, motion, color in node_rows])
    tbl = table_ax.table(cellText=table_rows, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.4 if len(table_rows) > 45 else 7.0)
    tbl.scale(1.0, 0.95 if len(table_rows) > 45 else 1.04)
    for cell in tbl.get_celld().values():
        cell.get_text().set_fontproperties(CHINESE_FONT)

    legend_ax = fig.add_axes([0.68, 0.04, 0.30, 0.12])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D([0], [0], color="#5B6770", lw=2, linestyle="-", label="附近（无向）"),
            Line2D([0], [0], color="#0072B2", lw=2, linestyle=":", label="左侧（箭头）"),
            Line2D([0], [0], color="#009E73", lw=2, linestyle="--", label="前方（箭头）"),
            Line2D([0], [0], color="#6B7280", lw=1, alpha=0.35, label="线宽/透明度：置信度"),
        ],
        loc="center left",
        frameon=False,
        prop=CHINESE_FONT,
    )

    fig.subplots_adjust(left=0.02, right=0.66, top=0.94, bottom=0.04)
    fig.savefig(out_dir / "scenegraph_3d.png", dpi=180)
    plt.close(fig)


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
    return colors.get(str(tag).lower(), "#94a3b8")


def category_height_offset(tag: str, xy_span: float) -> float:
    tag = str(tag).lower()
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
    return (order.get(tag, 2) - 2) * step


def display_height(raw_z: float, tag: str, ground_z: float, xy_span: float) -> float:
    # Shift all objects above a common ground plane and keep category offsets small.
    base = max(0.0, float(raw_z) - float(ground_z)) * 1.15
    offset = max(0.0, category_height_offset(tag, xy_span)) * 0.08
    return base + offset + max(float(xy_span) * 0.006, 0.35)


def set_axes_equal(ax) -> None:
    x_limits = np.asarray(ax.get_xlim3d(), dtype=float)
    y_limits = np.asarray(ax.get_ylim3d(), dtype=float)
    z_limits = np.asarray(ax.get_zlim3d(), dtype=float)
    centers = np.asarray([x_limits.mean(), y_limits.mean(), z_limits.mean()])
    radius = 0.5 * max(np.ptp(x_limits), np.ptp(y_limits), np.ptp(z_limits))
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


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
    return mapping.get(str(tag).lower(), str(tag))


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


def edge_style(relation: str) -> str:
    return relation_style(relation)["matplotlib"]


def relation_to_cn(relation: str) -> str:
    mapping = {
        "near": "附近",
        "left_of": "左侧",
        "right_of": "右侧",
        "in_front_of": "前方",
        "behind": "后方",
    }
    return mapping.get(str(relation).lower(), str(relation))


if __name__ == "__main__":
    main()
