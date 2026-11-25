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
import networkx as nx
import numpy as np

from conceptgraph.scenegraph.relations import SceneGraphEdge, load_scene_graph_edges
from conceptgraph.scenegraph.visualization import (
    edge_alpha,
    edge_is_highlighted,
    edge_width,
    filter_edges,
    navigation_relation_keys,
    parse_relation_types,
    relation_style,
)


CHINESE_FONT = FontProperties(fname="/root/autodl-tmp/lizichong/checkpoint/fonts/NotoSansCJKsc-Regular.otf")


def configure_chinese_font() -> None:
    candidates = [
        Path("/root/autodl-tmp/lizichong/checkpoint/fonts/NotoSansCJKsc-Regular.otf"),
        Path("/root/autodl-tmp/lizichong/checkpoint/Qwen2-VL-2B-Instruct/SimSun.ttf"),
    ]
    for font_path in candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            global CHINESE_FONT
            CHINESE_FONT = FontProperties(fname=str(font_path))
            plt.rcParams["font.family"] = [CHINESE_FONT.get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a ConceptGraphs scene graph.")
    parser.add_argument("--scene-graph", required=True, help="Path to scene_graph.json")
    parser.add_argument("--edges", default=None, help="Path to cfslam_scenegraph_edges.pkl")
    parser.add_argument("--out-dir", required=True, help="Directory for visualization outputs")
    parser.add_argument("--vehicle-pose", default=None, help="Optional vehicle pose x,y,z to plot")
    parser.add_argument("--navigation-plan", default=None, help="Optional semantic_navigation_plan.json to overlay on BEV")
    parser.add_argument("--trajectory-poses", default=None, help="Optional poses.txt path to draw ego trajectory")
    parser.add_argument("--bev-map", default=None, help="Optional bev_map.npz to render drivable area under the scene graph")
    parser.add_argument("--relation-types", default=None, help="Comma-separated relation filter")
    parser.add_argument("--minimum-confidence", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    configure_chinese_font()
    args = parse_args()
    scene_graph_path = Path(args.scene_graph)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(scene_graph_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    edges = filter_edges(
        load_scene_graph_edges(args.edges),
        relation_types=parse_relation_types(args.relation_types),
        minimum_confidence=args.minimum_confidence,
    )

    graph = nx.MultiDiGraph()
    for node in nodes:
        node_id = int(node["id"])
        tag = str(node.get("object_tag", "unknown"))
        center = np.asarray(node.get("bbox_center", [0, 0, 0]), dtype=float)
        caption = str(node.get("caption", ""))
        attrs = node.get("attributes", {}) if isinstance(node.get("attributes", {}), dict) else {}
        graph.add_node(
            node_id,
            tag=tag,
            center=center,
            caption=caption,
            motion_state=str(node.get("motion_state", node.get("metadata", {}).get("motion_state", "unknown"))),
            colors=attrs.get("colors", []),
        )

    for edge in edges:
        if edge.source in graph and edge.target in graph:
            graph.add_edge(
                edge.source,
                edge.target,
                relation=edge.relation,
                relation_type=edge.relation_type,
                confidence=edge.confidence,
                directed=edge.directed,
                edge=edge,
            )

    vehicle_pose = None
    if args.vehicle_pose:
        vehicle_pose = np.asarray([float(v) for v in args.vehicle_pose.split(",")], dtype=float)

    navigation_plan = load_navigation_plan(args.navigation_plan)
    relation_keys = navigation_relation_keys(navigation_plan)
    trajectory_xy = load_trajectory_xy(args.trajectory_poses)
    bev_map = load_bev_map(args.bev_map)

    save_network_plot(graph, out_dir / "scenegraph_network.png")
    save_bev_plot(graph, out_dir / "scenegraph_bev.png", vehicle_pose=vehicle_pose, trajectory_xy=trajectory_xy, bev_map=bev_map)
    if navigation_plan is not None:
        save_bev_plot(
            graph,
            out_dir / "scenegraph_bev_with_path.png",
            vehicle_pose=vehicle_pose,
            navigation_plan=navigation_plan,
            trajectory_xy=trajectory_xy,
            bev_map=bev_map,
            highlight_relation_keys=relation_keys,
            navigation_only=True,
        )
        save_bev_plot(
            graph,
            out_dir / "navigation_relation_chain.png",
            vehicle_pose=vehicle_pose,
            navigation_plan=navigation_plan,
            trajectory_xy=trajectory_xy,
            bev_map=bev_map,
            highlight_relation_keys=relation_keys,
            navigation_only=True,
            draw_navigation_path=False,
        )
    save_html_summary(nodes, edges, out_dir / "scenegraph_summary.html")
    save_json_summary(nodes, edges, out_dir / "scenegraph_summary.json")


def save_network_plot(graph: nx.Graph, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(20, 11))
    pos = nx.spring_layout(graph, seed=7) if graph.number_of_nodes() else {}
    labels = {node: f"{node}" for node, _data in graph.nodes(data=True)}
    colors = [tag_to_color(data.get("tag", "")) for _node, data in graph.nodes(data=True)]
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=1700, edgecolors="#263238", linewidths=1.2, ax=ax)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_color="#101010", ax=ax, font_family=CHINESE_FONT.get_name())
    for src, dst, data in graph.edges(data=True):
        draw_kwargs = {
            "G": graph,
            "pos": pos,
            "edgelist": [(src, dst)],
            "width": edge_width(data["edge"]),
            "alpha": edge_alpha(data["edge"]),
            "edge_color": edge_color(data),
            "style": edge_style(data.get("relation", "")),
            "arrows": bool(data.get("directed", True)),
            "ax": ax,
        }
        if data.get("directed", True):
            draw_kwargs.update({"arrowstyle": "-|>", "arrowsize": 18})
        nx.draw_networkx_edges(**draw_kwargs)
    ax.set_title("场景图 2D 关系视图", fontproperties=CHINESE_FONT)
    ax.axis("off")
    _add_right_side_table(fig, list(graph.nodes(data=True)))
    _add_relation_legend(fig)
    fig.subplots_adjust(right=0.66)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_navigation_plan(path: str | None) -> dict | None:
    if not path:
        return None
    plan_path = Path(path)
    if not plan_path.exists():
        return None
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_trajectory_xy(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    poses_path = Path(path)
    if not poses_path.exists():
        return None
    poses = np.loadtxt(poses_path)
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    centers = []
    for row in poses:
        if row.size < 16:
            continue
        mat = row.reshape(4, 4)
        centers.append(mat[:2, 3])
    if not centers:
        return None
    return np.asarray(centers, dtype=float)


def load_bev_map(path: str | None) -> dict | None:
    if not path:
        return None
    bev_path = Path(path)
    if not bev_path.exists():
        return None
    data = np.load(bev_path, allow_pickle=True)
    return {
        "occupancy": np.asarray(data["occupancy"], dtype=np.uint8),
        "origin_xy": np.asarray(data["origin_xy"], dtype=float),
        "resolution": float(data["resolution"]),
    }


def save_bev_plot(
    graph: nx.Graph,
    output_path: Path,
    vehicle_pose: np.ndarray | None = None,
    navigation_plan: dict | None = None,
    trajectory_xy: np.ndarray | None = None,
    bev_map: dict | None = None,
    highlight_relation_keys: set[tuple[int, int, str]] | None = None,
    navigation_only: bool = False,
    draw_navigation_path: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(20, 11))
    relation_keys = highlight_relation_keys or set()
    highlighted_nodes = {
        node_id
        for source, target, _relation in relation_keys
        for node_id in (source, target)
    }
    if bev_map is not None:
        occupancy = bev_map["occupancy"]
        origin_xy = bev_map["origin_xy"]
        resolution = bev_map["resolution"]
        extent = [
            origin_xy[0],
            origin_xy[0] + (occupancy.shape[1] - 1) * resolution,
            origin_xy[1],
            origin_xy[1] + (occupancy.shape[0] - 1) * resolution,
        ]
        ax.imshow(1 - occupancy, origin="lower", extent=extent, cmap="Greys", alpha=0.35)

    for node, data in graph.nodes(data=True):
        center = np.asarray(data["center"], dtype=float)
        tag = data.get("tag", "unknown")
        ax.scatter(center[0], center[1], s=120, c=tag_to_color(tag), edgecolors="#263238", linewidths=1.0)
        if navigation_only and node in highlighted_nodes:
            ax.scatter(
                center[0],
                center[1],
                s=340,
                facecolors="none",
                edgecolors="#f59e0b",
                linewidths=3.0,
                zorder=5,
            )
        ax.text(
            center[0],
            center[1],
            f" {node}",
            fontsize=9,
            va="center",
            fontproperties=CHINESE_FONT,
            clip_on=True,
        )

    for src, dst, data in graph.edges(data=True):
        c0 = np.asarray(graph.nodes[src]["center"], dtype=float)
        c1 = np.asarray(graph.nodes[dst]["center"], dtype=float)
        edge = data["edge"]
        highlighted = edge_is_highlighted(edge, relation_keys)
        alpha = edge_alpha(edge)
        linewidth = edge_width(edge)
        if navigation_only:
            alpha = 1.0 if highlighted else 0.025
            linewidth = linewidth + 3.5 if highlighted else 0.45
        if edge.directed:
            ax.annotate(
                "",
                xy=c1[:2],
                xytext=c0[:2],
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": edge_color(data),
                    "linestyle": edge_style(data.get("relation", "")),
                    "linewidth": linewidth,
                    "alpha": alpha,
                    "mutation_scale": 13 + 7 * edge.confidence,
                    "shrinkA": 8,
                    "shrinkB": 8,
                },
                zorder=2 if highlighted else 1,
            )
        else:
            ax.plot(
                [c0[0], c1[0]],
                [c0[1], c1[1]],
                color=edge_color(data),
                linestyle=edge_style(data.get("relation", "")),
                linewidth=linewidth,
                alpha=alpha,
                zorder=2 if highlighted else 1,
            )
        if highlighted and navigation_only and not draw_navigation_path:
            midpoint = (c0[:2] + c1[:2]) / 2.0
            ax.text(
                midpoint[0],
                midpoint[1],
                f"{src} {relation_to_cn(edge.relation)} {dst}\n置信度 {edge.confidence:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#7c2d12",
                weight="bold",
                fontproperties=CHINESE_FONT,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#f59e0b", "alpha": 0.92},
                zorder=6,
            )

    relation_chain_only = navigation_only and not draw_navigation_path
    if vehicle_pose is not None and vehicle_pose.size >= 2 and not relation_chain_only:
        ax.scatter(vehicle_pose[0], vehicle_pose[1], s=180, c="#ef4444", marker="*", edgecolors="#111827", linewidths=1.0)
        ax.text(vehicle_pose[0], vehicle_pose[1], " 自车", fontsize=10, color="#ef4444", weight="bold", fontproperties=CHINESE_FONT)

    if trajectory_xy is not None and len(trajectory_xy) > 1 and not relation_chain_only:
        ax.plot(
            trajectory_xy[:, 0],
            trajectory_xy[:, 1],
            linestyle="--",
            linewidth=1.8,
            color="#6b7280",
            alpha=0.9,
            label="自车轨迹",
        )
        ax.scatter(
            trajectory_xy[0, 0],
            trajectory_xy[0, 1],
            s=70,
            c="#10b981",
            marker="o",
            edgecolors="#111827",
            linewidths=0.8,
            label="轨迹起点",
        )
        ax.scatter(
            trajectory_xy[-1, 0],
            trajectory_xy[-1, 1],
            s=90,
            c="#ef4444",
            marker="X",
            edgecolors="#111827",
            linewidths=0.8,
            label="轨迹终点",
        )

    path_xy = np.empty((0, 2), dtype=float)
    if navigation_plan is not None and draw_navigation_path:
        path_world = navigation_plan.get("path", {}).get("world_xyz", [])
        if path_world:
            path_xy = np.asarray(path_world, dtype=float)[:, :2]
            ax.plot(path_xy[:, 0], path_xy[:, 1], color="#7c3aed", linewidth=2.5, alpha=0.95, label="导航路径")
            ax.scatter(path_xy[0, 0], path_xy[0, 1], s=90, c="#22c55e", marker="o", edgecolors="#111827", linewidths=0.8, label="路径起点")
            ax.scatter(path_xy[-1, 0], path_xy[-1, 1], s=190, c="#f97316", marker="X", edgecolors="#111827", linewidths=0.8, label="路径终点")

        target = navigation_plan.get("selected_target", {})
        target_point = target.get("target_point")
        if target_point:
            target_xy = np.asarray(target_point, dtype=float)[:2]
            target_label = f" 目标 {target.get('node_id', '')}"
            ax.scatter(target_xy[0], target_xy[1], s=230, facecolors="none", edgecolors="#f97316", linewidths=2.2)
            ax.text(target_xy[0], target_xy[1], target_label, fontsize=10, color="#9a3412", weight="bold", fontproperties=CHINESE_FONT)

    title = "导航使用的关系链" if navigation_only and not draw_navigation_path else "场景图俯视图"
    ax.set_title(title, fontproperties=CHINESE_FONT)
    ax.set_xlabel("x (米)", fontproperties=CHINESE_FONT)
    ax.set_ylabel("y (米)", fontproperties=CHINESE_FONT)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    extent_points = []
    if navigation_only and not draw_navigation_path and highlighted_nodes:
        extent_points.append(
            np.asarray(
                [graph.nodes[node_id]["center"][:2] for node_id in sorted(highlighted_nodes) if node_id in graph],
                dtype=float,
            )
        )
    elif len(graph.nodes) > 0:
        extent_points.append(np.asarray([data["center"][:2] for _, data in graph.nodes(data=True)], dtype=float))
    if vehicle_pose is not None and vehicle_pose.size >= 2 and not relation_chain_only:
        extent_points.append(np.asarray(vehicle_pose[:2], dtype=float).reshape(1, 2))
    if path_xy.size > 0:
        extent_points.append(path_xy)
    if trajectory_xy is not None and len(trajectory_xy) > 0 and not relation_chain_only:
        extent_points.append(trajectory_xy)
    if extent_points:
        points = np.concatenate(extent_points, axis=0)
        margin = 8 if navigation_only and not draw_navigation_path else 20
        ax.set_xlim(points[:, 0].min() - margin, points[:, 0].max() + margin)
        ax.set_ylim(points[:, 1].min() - margin, points[:, 1].max() + margin)
    if not relation_chain_only and (
        (navigation_plan is not None and path_xy.size > 0)
        or (trajectory_xy is not None and len(trajectory_xy) > 1)
    ):
        ax.legend(loc="best")
    _add_right_side_table(fig, list(graph.nodes(data=True)))
    _add_relation_legend(fig)
    fig.subplots_adjust(right=0.66)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _add_right_side_table(fig: plt.Figure, nodes: list[tuple[int, dict]]) -> None:
    table_ax = fig.add_axes([0.68, 0.20, 0.30, 0.74])
    table_ax.axis("off")
    rows = [["ID", "英文标签", "中文", "状态", "颜色"]]
    for node_id, data in sorted(((int(n), d) for n, d in nodes), key=lambda item: item[0]):
        eng = str(data.get("tag", "unknown"))
        colors = data.get("colors", []) or []
        rows.append([str(node_id), eng, tag_to_cn(eng), motion_to_cn(data.get("motion_state", "unknown")), colors_to_cn(colors)])
    table = table_ax.table(cellText=rows, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(6.6 if len(rows) > 45 else 7.2)
    table.scale(1.0, 0.95 if len(rows) > 45 else 1.04)
    for cell in table.get_celld().values():
        cell.get_text().set_fontproperties(CHINESE_FONT)


def _add_relation_legend(fig: plt.Figure) -> None:
    legend_ax = fig.add_axes([0.68, 0.04, 0.30, 0.13])
    legend_ax.axis("off")
    handles = [
        Line2D([0], [0], color="#5B6770", lw=2, linestyle="-", label="附近（无向）"),
        Line2D([0], [0], color="#0072B2", lw=2, linestyle=":", label="左侧（箭头）"),
        Line2D([0], [0], color="#009E73", lw=2, linestyle="--", label="前方（箭头）"),
        Line2D([0], [0], color="#6B7280", lw=1, alpha=0.35, label="线宽/透明度：置信度"),
    ]
    legend_ax.legend(handles=handles, loc="center left", frameon=False, prop=CHINESE_FONT)


def save_html_summary(nodes: list[dict], edges: list[SceneGraphEdge], output_path: Path) -> None:
    rows = []
    for node in nodes:
        rows.append(
            "<tr>"
            f"<td>{node.get('id')}</td>"
            f"<td>{escape_html(node.get('object_tag', ''))}</td>"
            f"<td>{escape_html(node.get('caption', ''))}</td>"
            f"<td>{escape_html(json.dumps(node.get('bbox_center', [])))}</td>"
            "</tr>"
        )
    edge_rows = []
    for edge in edges:
        edge_rows.append(
            f"<tr><td>{edge.source}</td><td>{edge.target}</td>"
            f"<td>{escape_html(edge.relation)}</td><td>{edge.confidence:.3f}</td>"
            f"<td>{escape_html(edge.coordinate_frame)}</td></tr>"
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Scene Graph Summary</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #17212b; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #cfd8dc; padding: 8px; text-align: left; }}
    th {{ background: #eceff1; }}
    img {{ max-width: 48%; border: 1px solid #cfd8dc; margin-right: 12px; }}
  </style>
</head>
<body>
  <h1>场景图摘要</h1>
  <p>节点数: {len(nodes)} | 边数: {len(edges)}</p>
  <img src="scenegraph_network.png" alt="network">
  <img src="scenegraph_bev.png" alt="bev">
  <h2>节点</h2>
  <table><tr><th>编号</th><th>类别</th><th>描述</th><th>中心坐标</th></tr>{''.join(rows)}</table>
  <h2>边</h2>
  <table><tr><th>源节点</th><th>目标节点</th><th>关系</th><th>置信度</th><th>坐标系</th></tr>{''.join(edge_rows)}</table>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def save_json_summary(nodes: list[dict], edges: list[SceneGraphEdge], output_path: Path) -> None:
    payload = {
        "nodes": nodes,
        "edges": [edge.to_dict() for edge in edges],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_edge_metadata(scene_graph_path: Path, edges_path: str | None) -> dict[tuple[int, int], dict]:
    candidates = [scene_graph_path.parent / "cfslam_scenegraph_edge_metadata.json"]
    if edges_path:
        candidates.append(Path(edges_path).with_name("cfslam_scenegraph_edge_metadata.json"))
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        result = {}
        for item in payload:
            try:
                src = int(item["source"])
                dst = int(item["target"])
            except Exception:
                continue
            result[(src, dst)] = item
            result[(dst, src)] = item
        return result
    return {}


def edge_style(relation: str) -> str:
    return relation_style(relation)["matplotlib"]


def edge_color(data: dict) -> str:
    return relation_style(str(data.get("relation", "near")))["color"]


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


def tag_to_cn(tag: str) -> str:
    tag = str(tag).lower().strip()
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
        "parking area": "停车区",
        "vehicle": "车辆",
        "object": "物体",
        "unknown": "未知",
    }
    return mapping.get(tag, tag if tag else "未知")


def motion_to_cn(value: str) -> str:
    mapping = {
        "static": "静态",
        "dynamic": "动态",
        "unknown": "未知",
    }
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


def relation_to_cn(relation: str) -> str:
    relation = str(relation).lower().strip()
    mapping = {
        "near": "附近",
        "left_of": "左侧",
        "right_of": "右侧",
        "in_front_of": "前方",
        "behind": "后方",
    }
    return mapping.get(relation, relation)


def escape_html(value) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
