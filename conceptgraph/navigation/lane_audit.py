"""Experiment audits derived from the live planner grid, without changing decisions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from conceptgraph.navigation.lane_constraints import validate_lane_path
from conceptgraph.navigation.lane_geometry import continuous_collision_cells
from conceptgraph.navigation.planner import minimum_path_clearance_m, path_length_m
from conceptgraph.navigation.target_selector import target_selection_result_to_dict


def audit_online_frame(
    navigator, update, scene_output: Path, topology, *, write_outputs: bool
) -> dict:
    grid = navigator._occupancy_grid
    path = list(navigator._path_grid)
    result = getattr(navigator, "_last_planning_result", None)
    audit = {
        "planning_result": None
        if result is None
        else target_selection_result_to_dict(result),
        "path_grid": [list(rc) for rc in path],
        "path_world": update.path_world.tolist(),
        "lane_validation": validate_lane_path(
            ((update.path_world[:, :2] - topology.origin_xy) / topology.resolution)[
                :, ::-1
            ],
            topology,
        )
        if path
        else None,
        "occupancy_artifact": None,
    }
    if grid is None:
        return audit
    if write_outputs:
        output = scene_output / "live_grids" / f"frame_{update.frame_idx:04d}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            occupancy=grid.occupancy,
            clearance_m=grid.clearance_m,
            origin_xy=grid.origin_xy,
            resolution=grid.resolution,
        )
        audit["occupancy_artifact"] = str(output.resolve())
    # For reused paths evaluate the portion ahead of the current recorded pose.
    trajectory = navigator._trajectory
    nearest = 0
    if path and trajectory:
        nearest = int(
            np.argmin(np.linalg.norm(update.path_world[:, :2] - trajectory[-1], axis=1))
        )
    remaining = path[nearest:]
    collisions = set()
    if len(remaining) == 1:
        collisions.update(
            map(
                tuple,
                continuous_collision_cells(
                    np.asarray(remaining[0]), np.asarray(remaining[0]), grid.occupancy
                ),
            )
        )
    for a, b in zip(remaining[:-1], remaining[1:]):
        collisions.update(
            map(
                tuple,
                continuous_collision_cells(
                    np.asarray(a), np.asarray(b), grid.occupancy
                ),
            )
        )
    audit["remaining_path_collision_cells"] = [list(rc) for rc in sorted(collisions)]
    audit["remaining_path_start_index"] = nearest
    audit["path_length_m"] = path_length_m(grid, path) if path else None
    audit["minimum_clearance_m"] = (
        minimum_path_clearance_m(grid, remaining) if remaining else None
    )
    points = np.asarray(update.path_world, dtype=float)
    turns = []
    if len(points) >= 3:
        directions = np.diff(points[:, :2], axis=0)
        headings = np.arctan2(directions[:, 1], directions[:, 0])
        turns = np.abs((np.diff(headings) + np.pi) % (2 * np.pi) - np.pi).tolist()
    audit["total_absolute_turn_rad"] = float(sum(turns)) if path else None
    return audit
