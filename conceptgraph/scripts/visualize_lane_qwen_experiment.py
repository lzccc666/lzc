"""Render planner artifacts and diagnostics without inventing missing Qwen runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from conceptgraph.navigation.lane_constraints import (
    load_lane_topology,
    validate_lane_path,
)
from conceptgraph.navigation.planner import OccupancyGrid2D, smooth_grid_path
from conceptgraph.scripts.run_lane_qwen_experiment import (
    EXPERIMENT_ROOT,
    LANE_ROOT,
    SCENES,
)
from conceptgraph.scripts.run_incremental_semantic_navigation import write_json


def world_path(path, topology):
    return topology.origin_xy + np.asarray(path)[:, ::-1] * topology.resolution


def draw_map(ax, topology, occupancy):
    origin, resolution = topology.origin_xy, topology.resolution
    extent = [
        origin[0] - 0.5 * resolution,
        origin[0] + (occupancy.shape[1] - 0.5) * resolution,
        origin[1] - 0.5 * resolution,
        origin[1] + (occupancy.shape[0] - 0.5) * resolution,
    ]
    ax.imshow(
        occupancy,
        origin="lower",
        extent=extent,
        cmap="Greys",
        alpha=0.25,
        vmin=0,
        vmax=1,
    )
    for geometry in topology.geometries.values():
        polygon, center = geometry.polygon, geometry.centerline
        color = "#d97706" if geometry.lane_type == "lane_connector" else "#2563eb"
        ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.06)
        ax.plot(*np.vstack([polygon, polygon[0]]).T, color=color, alpha=0.25, lw=0.5)
        ax.plot(center[:, 0], center[:, 1], color=color, lw=0.7, alpha=0.6)
        middle = center[len(center) // 2]
        ax.arrow(
            *middle[:2],
            np.cos(middle[2]),
            np.sin(middle[2]),
            color=color,
            width=0.015,
            head_width=0.3,
            length_includes_head=True,
        )
    ax.set(xlim=extent[:2], ylim=extent[2:], xlabel="World x (m)", ylabel="World y (m)")
    ax.set_aspect("equal")


def draw_validation(ax, path, validation, topology, *, label="accepted", annotate=True):
    color = "#15803d" if validation["valid"] else "#dc2626"
    ax.plot(path[:, 0], path[:, 1], color=color, lw=2, label=label)
    for key, marker in (("invalid_cells", "x"),):
        cells = validation.get(key, [])
        if cells:
            points = world_path(cells, topology)
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=10,
                c="#dc2626",
                marker=marker,
                label="outside lane",
            )
    if annotate and validation["reverse_count"]:
        ax.text(
            0.02,
            0.97,
            f"Reverse samples: {validation['reverse_count']}",
            va="top",
            transform=ax.transAxes,
            color="#b91c1c",
        )
    for source, target in zip(
        validation["lane_trace"][:-1], validation["lane_trace"][1:]
    ):
        geometry = topology.geometries[source]
        if target in topology.transitions[source] and geometry.end_pose:
            ax.scatter(
                *geometry.end_pose[:2],
                s=35,
                marker="D",
                facecolor="none",
                edgecolor="#15803d",
            )


def render_scene(scene, run, output):
    topology = load_lane_topology(LANE_ROOT / scene / "lane_constrained_bev.npz")
    with np.load(LANE_ROOT / scene / "lane_constrained_bev.npz") as data:
        occupancy = data["occupancy"]
    output.mkdir(parents=True, exist_ok=True)
    poses = np.loadtxt(Path("dataset/nuscenes") / scene / "poses.txt").reshape(-1, 4, 4)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    rejected = []
    for ax, mode in zip(
        axes.flat, ("free_space", "drivable_area", "lane_area", "lane_constrained")
    ):
        directory = run / "static_comparison" / scene / mode
        if not directory.exists():
            static = json.loads((run / "static_phase_report.json").read_text())
            directory = Path(static[scene][mode]["output_dir"])
        selection = json.loads((directory / "target_selection_result.json").read_text())
        with np.load(directory / "live_grid.npz") as data:
            live = data["occupancy"]
            grid = OccupancyGrid2D(live, float(data["resolution"]), data["origin_xy"])
        with np.load(directory / "input_bev.npz") as data:
            mode_occupancy = data["occupancy"]
        draw_map(ax, topology, mode_occupancy)
        dynamic = np.argwhere((live > 0) & (mode_occupancy == 0))
        if len(dynamic):
            points = world_path(dynamic, topology)
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=2,
                c="#7c3aed",
                alpha=0.5,
                label="object obstacle overlay",
            )
        ax.scatter(
            *poses[0, :2, 3], s=45, marker="o", color="#111827", label="recorded start"
        )
        plan_path = directory / "semantic_navigation_plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            path = np.asarray(plan["path"]["world_xyz"])
            validation = validate_lane_path(
                ((path[:, :2] - topology.origin_xy) / topology.resolution)[:, ::-1],
                topology,
                live,
            )
            draw_validation(ax, path, validation, topology, label="planned path")
            selected = selection["selected_candidate"]
            target = selected["target_point"][:2]
            ax.scatter(
                *target, s=65, marker="*", color="#15803d", label="projected goal"
            )
            for goal in selected["goal_candidates"]:
                if (
                    goal.get("goal_rc") == selected.get("target_grid")
                    and goal["path_found"]
                ):
                    desired = goal["desired_goal_xy"]
                    ax.scatter(
                        *desired,
                        s=40,
                        marker="+",
                        color="#d97706",
                        label="desired approach",
                    )
                    ax.plot(
                        [desired[0], target[0]],
                        [desired[1], target[1]],
                        "--",
                        color="#d97706",
                        lw=1,
                    )
                    break
        else:
            ax.text(
                0.02,
                0.97,
                selection["failure_code"],
                va="top",
                transform=ax.transAxes,
                color="#b91c1c",
            )
        if mode == "lane_constrained":
            for candidate in selection["evaluated_candidates"]:
                for goal in candidate["goal_candidates"]:
                    if (
                        goal["debug"].get("smoothing_rejected")
                        and goal["path_grid_raw"]
                    ):
                        smoothed = smooth_grid_path(
                            grid,
                            [tuple(rc) for rc in goal["path_grid_raw"]],
                            chaikin_iterations=2,
                        )
                        check = validate_lane_path(smoothed, topology, live)
                        rejected.append(
                            {
                                "source_goal_rc": goal["goal_rc"],
                                "reconstructed_smoothed_path": smoothed,
                                "validation": check,
                            }
                        )
                        if (
                            not check["valid"]
                            and sum(not v["validation"]["valid"] for v in rejected) == 1
                        ):
                            draw_validation(
                                ax,
                                world_path(smoothed, topology),
                                check,
                                topology,
                                label="rejected smoothing",
                            )
        ax.set_title(f"{scene}: {mode}")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                dict(zip(labels, handles)).values(),
                dict(zip(labels, handles)).keys(),
                fontsize=7,
                loc="best",
            )
    fig.suptitle(
        "Blue/orange: HD Map lanes/connectors; green: lane-compliant; red: noncompliant or rejected",
        fontsize=10,
    )
    fig.tight_layout()
    comparison = output / "four_map_path_comparison.png"
    fig.savefig(comparison, dpi=160)
    plt.close(fig)
    write_json(
        output / "rejected_smoothing_evidence.json",
        {
            "source": "recorded raw A* path + same deterministic smoother, two iterations",
            "paths": rejected,
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    counts = {}
    for ax, condition in zip(axes, ("drivable_qwen_off", "lane_qwen_off")):
        draw_map(ax, topology, occupancy)
        frames = (
            run
            / "online"
            / condition
            / "first"
            / "end_to_end_replay"
            / scene
            / "frames.jsonl"
        )
        rows = [
            json.loads(line) for line in frames.read_text().splitlines() if line.strip()
        ]
        paths = [
            row
            for row in rows
            if row["replanned"] and row["experiment_audit"]["path_grid"]
        ]
        counts[condition] = len(paths)
        if paths:
            artifact = paths[0]["experiment_audit"]["occupancy_artifact"]
            with np.load(artifact) as data:
                indices = np.argwhere(data["occupancy"] > 0)
                world = data["origin_xy"] + indices[:, ::-1] * float(data["resolution"])
                # Draw live occupied cells only where the HD Map itself is drivable.
                rc = np.round(
                    (world - topology.origin_xy) / topology.resolution
                ).astype(int)
                inside = (
                    (rc[:, 0] >= 0)
                    & (rc[:, 0] < occupancy.shape[1])
                    & (rc[:, 1] >= 0)
                    & (rc[:, 1] < occupancy.shape[0])
                )
                world, rc = world[inside], rc[inside]
                world = world[occupancy[rc[:, 1], rc[:, 0]] == 0]
                ax.scatter(world[:, 0], world[:, 1], s=2, c="#7c3aed", alpha=0.45)
                ax.text(
                    0.02,
                    0.02,
                    f"Purple: live object occupancy, frame {paths[0]['frame_idx']}",
                    transform=ax.transAxes,
                    fontsize=8,
                )
        for row in paths:
            audit = row["experiment_audit"]
            draw_validation(
                ax,
                np.asarray(audit["path_world"]),
                audit["lane_validation"],
                topology,
                annotate=False,
            )
        reverse_events = sum(
            row["experiment_audit"]["lane_validation"]["reverse_event_count"]
            for row in paths
        )
        illegal_events = sum(
            len(row["experiment_audit"]["lane_validation"]["illegal_transitions"])
            for row in paths
        )
        ax.text(
            0.02,
            0.97,
            f"Reverse events: {reverse_events}; illegal transitions: {illegal_events}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            color="#b91c1c" if reverse_events or illegal_events else "#15803d",
        )
        ax.set_title(f"{condition}: {len(paths)} successful replans")
        ax.plot(poses[:, :2, 3][:, 0], poses[:, :2, 3][:, 1], "k--", alpha=0.5, lw=0.7)
    fig.tight_layout()
    replay = output / "online_rule_path_comparison.png"
    fig.savefig(replay, dpi=160)
    plt.close(fig)
    return {
        "scene": scene,
        "static_comparison": str(comparison),
        "online_comparison": str(replay),
        "rejected_smoothing_count": len(rejected),
        "online_path_counts": counts,
    }


def render_fault_fallback(scene, output):
    report = json.loads((EXPERIMENT_ROOT / "failure_injection_report.json").read_text())
    topology = load_lane_topology(LANE_ROOT / scene / "lane_constrained_bev.npz")
    with np.load(LANE_ROOT / scene / "lane_constrained_bev.npz") as loaded:
        occupancy = loaded["occupancy"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, case in zip(axes, ("qwen_off", "exception")):
        draw_map(ax, topology, occupancy)
        directory = Path(report["artifact_root"]) / scene / case
        result = json.loads((directory / "target_selection_result.json").read_text())
        plan_path = directory / "semantic_navigation_plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            world = np.asarray(plan["path"]["world_xyz"])
            validation = validate_lane_path(
                ((world[:, :2] - topology.origin_xy) / topology.resolution)[:, ::-1],
                topology,
            )
            draw_validation(ax, world, validation, topology)
        else:
            ax.text(
                0.02,
                0.95,
                result["failure_code"],
                transform=ax.transAxes,
                va="top",
                color="#b91c1c",
            )
        ax.set_title(
            f"{scene}: {'rule baseline' if case == 'qwen_off' else 'injected exception -> safe rule fallback'}"
        )
    fig.suptitle("CPU fault injection: no Qwen model loaded", fontsize=11)
    fig.tight_layout()
    artifact = output / "rule_vs_fault_fallback.png"
    fig.savefig(artifact, dpi=160)
    plt.close(fig)
    return str(artifact)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    validation = json.loads((EXPERIMENT_ROOT / "validation_report.json").read_text())
    run = args.run_dir or Path(validation["run_dir"])
    output = EXPERIMENT_ROOT / "visualizations" / run.name
    scenes = [render_scene(scene, run, output / scene) for scene in SCENES]
    for scene in scenes:
        scene["fault_fallback_comparison"] = render_fault_fallback(
            scene["scene"], output / scene["scene"]
        )
    runtime = json.loads((EXPERIMENT_ROOT / "qwen_runtime_report.json").read_text())
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    ax.text(0.02, 0.75, "Qwen backend / cache / reranking evidence", fontsize=16)
    ax.text(
        0.02,
        0.45,
        f"Visible CUDA devices: {runtime['cuda_device_count']} | Actual backend calls: {runtime['backend_call_count']}",
        fontsize=12,
    )
    ax.text(
        0.02,
        0.18,
        "GPU inference pending; no Qwen ranking or model-quality comparison is claimed."
        if not runtime["real_inference_completed"]
        else "See per-frame qwen_audit and per-call model/adapter audit files.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output / "qwen_evidence_status.png", dpi=160)
    plt.close(fig)
    write_json(
        output / "visualization_manifest.json",
        {
            "scenes": scenes,
            "run_dir": str(run),
            "runtime": runtime,
            "data_source": "same lane topology, planner grids, raw/accepted paths and frame audits",
            "qwen_rerank_visualization_pending": not runtime[
                "real_inference_completed"
            ],
        },
    )
    print(json.dumps(scenes), flush=True)


if __name__ == "__main__":
    main()
