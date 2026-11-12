"""Independent continuous lane-path validation, including overlapping lanes."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from conceptgraph.navigation.lane_geometry import (
    contains_points,
    continuous_collision_cells,
    segment_breaks,
)

if TYPE_CHECKING:
    from conceptgraph.navigation.lane_constraints import LaneTopology


def transition_allowed(
    topology: LaneTopology, source: int, target: int, a: np.ndarray, b: np.ndarray
) -> bool:
    if source == target:
        return True
    source_token, target_token = (
        topology.lane_tokens[source],
        topology.lane_tokens[target],
    )
    if target_token not in topology.transitions.get(source_token, ()):
        return False
    source_geometry = topology.geometries.get(source_token)
    target_geometry = topology.geometries.get(target_token)
    if (
        source_geometry is None
        or target_geometry is None
        or source_geometry.end_pose is None
        or target_geometry.start_pose is None
    ):
        return False
    tolerance = float(topology.diagnostics.get("transition_endpoint_tolerance_m", 2.0))
    # A legal edge can only be traversed near the two connected arcline ends.
    return bool(
        np.linalg.norm(a - source_geometry.end_pose[:2]) <= tolerance
        and np.linalg.norm(b - target_geometry.start_pose[:2]) <= tolerance
    )


def validate_continuous_lane_path(
    path,
    topology: LaneTopology,
    occupancy=None,
    *,
    start_heading_xy=None,
    direction_tolerance_deg=75.0,
) -> dict:
    if not 0 < direction_tolerance_deg <= 90:
        raise ValueError(
            "direction tolerance must be in (0,90] to prohibit reverse driving"
        )
    cells = np.asarray(list(path), dtype=float)
    if cells.size == 0:
        cells = np.empty((0, 2))
    if cells.ndim != 2 or cells.shape[1] != 2 or not np.isfinite(cells).all():
        raise ValueError("path must contain finite row/column pairs")
    worlds = topology.origin_xy + cells[:, ::-1] * topology.resolution
    polygons = [geometry.polygon for geometry in topology.geometries.values()]
    points, directions, lengths = [], [], []
    collisions = set()
    out_length = 0.0
    invalid, reverse, missing, illegal = [], [], [], []
    if len(cells):
        points.append(worlds[0])
        directions.append(None)
        lengths.append(0.0)
        if occupancy is not None:
            collisions.update(
                map(tuple, continuous_collision_cells(cells[0], cells[0], occupancy))
            )
    for a_rc, b_rc, a, b in zip(cells[:-1], cells[1:], worlds[:-1], worlds[1:]):
        length = float(np.linalg.norm(b - a))
        if occupancy is not None:
            collisions.update(
                map(tuple, continuous_collision_cells(a_rc, b_rc, occupancy))
            )
        if length <= 1e-10:
            continue
        breaks = segment_breaks(a, b, polygons)
        # Polygon edge partitions prove coverage; the extra samples resolve the
        # local arcline tangent and provide a stable physical length measure.
        breaks = np.unique(
            np.r_[
                breaks,
                np.linspace(
                    0, 1, max(2, math.ceil(length / (topology.resolution / 4)) + 1)
                ),
            ]
        )
        for lo, hi in zip(breaks[:-1], breaks[1:]):
            for fraction, distance in (((lo + hi) / 2, length * (hi - lo)), (hi, 0.0)):
                points.append(a + fraction * (b - a))
                directions.append((b - a) / length)
                lengths.append(distance)
    samples = np.asarray(points, dtype=float).reshape(-1, 2)
    memberships_by_lane = np.zeros(
        (len(topology.lane_tokens), len(samples)), dtype=bool
    )
    headings = np.full(memberships_by_lane.shape, np.nan)
    for lane_id, token in enumerate(topology.lane_tokens):
        geometry = topology.geometries.get(token)
        if geometry is not None:
            memberships_by_lane[lane_id] = contains_points(geometry.polygon, samples)
        else:
            for index, point in enumerate(samples):
                col, row = np.round(
                    (point - topology.origin_xy) / topology.resolution
                ).astype(int)
                memberships_by_lane[lane_id, index] = lane_id in topology.lanes_at(
                    (row, col)
                )
        indices = np.flatnonzero(memberships_by_lane[lane_id])
        if lane_id in topology._trees and len(indices):
            _, nearest = topology._trees[lane_id].query(samples[indices])
            headings[lane_id, indices] = geometry.centerline[nearest, 2]
        else:
            headings[lane_id, indices] = [
                topology.heading_at(lane_id, samples[i]) for i in indices
            ]
    states: dict[int, tuple[int, ...]] = {}
    direction_violations = []
    structural_states = set()
    previous_structural_point = None
    cosine = math.cos(math.radians(direction_tolerance_deg))
    previous_point = None
    for index, (point, movement, distance) in enumerate(
        zip(points, directions, lengths)
    ):
        memberships = np.flatnonzero(memberships_by_lane[:, index]).tolist()
        if not memberships:
            invalid.append(
                [
                    float(v)
                    for v in (point - topology.origin_xy)[::-1] / topology.resolution
                ]
            )
            out_length += distance
        if memberships:
            if previous_structural_point is None:
                structural_states = set(memberships)
            else:
                next_structural = {
                    target
                    for target in memberships
                    if any(
                        transition_allowed(
                            topology, source, target, previous_structural_point, point
                        )
                        for source in structural_states
                    )
                }
                if structural_states and not next_structural:
                    illegal.append(
                        {
                            "index": index,
                            "from": [
                                topology.lane_tokens[i]
                                for i in sorted(structural_states)
                            ],
                            "to": [topology.lane_tokens[i] for i in memberships],
                        }
                    )
                # Restart diagnostics after a violation; acceptance states below never restart.
                structural_states = next_structural or set(memberships)
            previous_structural_point = point
        legal_directions = []
        direction_dots = []
        for lane_id in memberships:
            heading = float(headings[lane_id, index])
            if not math.isfinite(heading):
                continue
            if movement is not None:
                direction_dots.append(
                    float(movement @ np.asarray([math.cos(heading), math.sin(heading)]))
                )
            if (
                movement is None
                or float(movement @ np.asarray([math.cos(heading), math.sin(heading)]))
                >= cosine - 1e-9
            ):
                legal_directions.append(lane_id)
        if memberships and not any(
            math.isfinite(headings[lane_id, index]) for lane_id in memberships
        ):
            missing.append(index)
        elif memberships and movement is not None and not legal_directions:
            direction_violations.append(index)
            if direction_dots and max(direction_dots) < -1e-9:
                reverse.append(index)
        if index == 0:
            for lane_id in legal_directions:
                if start_heading_xy is not None:
                    vector = np.asarray(start_heading_xy, dtype=float)
                    norm = np.linalg.norm(vector)
                    if (
                        vector.shape != (2,)
                        or norm < 1e-9
                        or not np.isfinite(vector).all()
                    ):
                        continue
                    heading = float(headings[lane_id, index])
                    if (
                        float(
                            vector
                            @ np.asarray([math.cos(heading), math.sin(heading)])
                            / norm
                        )
                        < 0
                    ):
                        continue
                states[lane_id] = (lane_id,)
        else:
            next_states = {}
            for target in legal_directions:
                for source, trace in sorted(states.items()):
                    if transition_allowed(
                        topology, source, target, previous_point, point
                    ):
                        next_states[target] = (
                            trace if source == target else (*trace, target)
                        )
                        break
            states = next_states
        previous_point = point
    trace = list(next(iter(states.values()))) if states else []
    return {
        "valid": bool(
            len(cells)
            and states
            and not collisions
            and not invalid
            and not direction_violations
            and not illegal
            and not missing
        ),
        "path_length_cells": len(cells),
        "sample_count": len(points),
        "lane_ids": [topology.lane_tokens[i] for i in trace],
        "lane_trace": [topology.lane_tokens[i] for i in trace],
        "invalid_cells": invalid,
        "illegal_transitions": illegal,
        "direction_violation_steps": direction_violations,
        "direction_violation_count": len(direction_violations),
        "reverse_steps": reverse,
        "reverse_count": len(reverse),
        "reverse_event_count": sum(
            i == 0 or value != reverse[i - 1] + 1 for i, value in enumerate(reverse)
        ),
        "missing_heading_steps": missing,
        "collision_cells": [list(v) for v in sorted(collisions)],
        "out_of_lane_ratio": len(invalid) / len(points) if points else None,
        "out_of_lane_length_m": float(out_length),
        "path_length_m": float(np.linalg.norm(np.diff(worlds, axis=0), axis=1).sum())
        if len(worlds)
        else 0.0,
        "continuous_polygon_check": True,
        "continuous_collision_check": True,
    }
