"""Lane-level map data and deterministic safety checks for navigation.

The semantic modules may suggest a target, but this module is deliberately
independent of language-model output.  A path is accepted only when every
grid cell belongs to a lane and every lane change is present in the HD map
connectivity table.
"""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial import cKDTree

from conceptgraph.navigation.lane_geometry import (
    contains_points,
    segment_covered,
)

from conceptgraph.navigation.lane_validation import (
    transition_allowed,
    validate_continuous_lane_path,
)
from conceptgraph.navigation.planner import OccupancyGrid2D


INVALID_LANE = -1


@dataclass(frozen=True)
class LaneGeometry:
    token: str
    lane_type: str
    polygon: np.ndarray
    centerline: np.ndarray
    heading: float | None
    start_pose: tuple[float, float, float] | None
    end_pose: tuple[float, float, float] | None
    incoming: tuple[str, ...] = ()
    outgoing: tuple[str, ...] = ()


@dataclass
class LaneTopology:
    lane_id_grid: np.ndarray
    lane_heading_grid: np.ndarray
    centerline_distance_grid: np.ndarray
    lane_tokens: list[str]
    lane_types: list[str]
    transitions: dict[str, tuple[str, ...]]
    geometries: dict[str, LaneGeometry] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    origin_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    resolution: float = 1.0
    memberships: np.ndarray | None = field(default=None, repr=False)
    headings: np.ndarray | None = field(default=None, repr=False)
    distances: np.ndarray | None = field(default=None, repr=False)
    _trees: dict[int, cKDTree] = field(default_factory=dict, repr=False)
    _edge_geometry_cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if (
            self.lane_id_grid.ndim != 2
            or self.resolution <= 0
            or not math.isfinite(self.resolution)
        ):
            raise ValueError("invalid lane raster dimensions or resolution")
        if (
            self.lane_heading_grid.shape != self.lane_id_grid.shape
            or self.centerline_distance_grid.shape != self.lane_id_grid.shape
        ):
            raise ValueError("lane arrays must have identical shapes")
        if len(set(self.lane_tokens)) != len(self.lane_tokens) or len(
            self.lane_types
        ) != len(self.lane_tokens):
            raise ValueError("duplicate lane tokens or inconsistent types")
        if self.origin_xy.shape != (2,) or not np.isfinite(self.origin_xy).all():
            raise ValueError("invalid lane origin")
        if any(
            source not in self.lane_tokens
            or any(target not in self.lane_tokens for target in targets)
            for source, targets in self.transitions.items()
        ):
            raise ValueError("unknown lane token in connectivity")
        count = len(self.lane_tokens)
        if not np.all((self.lane_id_grid >= -1) & (self.lane_id_grid < count)):
            raise ValueError("lane_id_grid contains an unknown lane index")
        self.memberships = np.zeros((count, *self.lane_id_grid.shape), dtype=bool)
        self.headings = np.full(self.memberships.shape, np.nan)
        self.distances = np.full(self.memberships.shape, np.inf)
        rows, cols = np.indices(self.lane_id_grid.shape)
        points = (
            self.origin_xy
            + np.column_stack([cols.ravel(), rows.ravel()]) * self.resolution
        )
        for lane_id, token in enumerate(self.lane_tokens):
            geometry = self.geometries.get(token)
            if geometry is None:
                mask = self.lane_id_grid == lane_id
                self.memberships[lane_id] = mask
                self.headings[lane_id, mask] = self.lane_heading_grid[mask]
                self.distances[lane_id, mask] = self.centerline_distance_grid[mask]
                continue
            polygon = np.asarray(geometry.polygon, dtype=float)
            centerline = np.asarray(geometry.centerline, dtype=float)
            if (
                polygon.ndim != 2
                or polygon.shape[1:] != (2,)
                or len(polygon) < 3
                or not np.isfinite(polygon).all()
            ):
                raise ValueError(f"invalid polygon: {token}")
            if (
                centerline.ndim != 2
                or centerline.shape[1:] != (3,)
                or len(centerline) < 2
                or not np.isfinite(centerline).all()
            ):
                raise ValueError(f"missing centerline/heading: {token}")
            tree = cKDTree(centerline[:, :2])
            self._trees[lane_id] = tree
            mask = contains_points(polygon, points).reshape(self.lane_id_grid.shape)
            self.memberships[lane_id] = mask
            if mask.any():
                distances, nearest = tree.query(points[mask.ravel()])
                self.headings[lane_id, mask] = centerline[nearest, 2]
                self.distances[lane_id, mask] = distances

    @property
    def lane_mask(self) -> np.ndarray:
        return self.memberships.any(axis=0)

    def lanes_at(self, rc: tuple[int, int]) -> list[int]:
        if not (
            0 <= rc[0] < self.lane_id_grid.shape[0]
            and 0 <= rc[1] < self.lane_id_grid.shape[1]
        ):
            return []
        return np.flatnonzero(self.memberships[:, rc[0], rc[1]]).tolist()

    def world(self, rc) -> np.ndarray:
        return self.origin_xy + np.asarray([rc[1], rc[0]]) * self.resolution

    def heading_at(self, lane_id: int, xy: np.ndarray) -> float:
        if lane_id in self._trees:
            _, index = self._trees[lane_id].query(xy)
            return float(
                self.geometries[self.lane_tokens[lane_id]].centerline[index, 2]
            )
        col, row = np.round((xy - self.origin_xy) / self.resolution).astype(int)
        if (
            0 <= row < self.lane_id_grid.shape[0]
            and 0 <= col < self.lane_id_grid.shape[1]
        ):
            return float(self.headings[lane_id, row, col])
        return float("nan")

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.lane_tokens)}

    def token_at(self, rc: tuple[int, int]) -> str | None:
        row, col = rc
        if not (
            0 <= row < self.lane_id_grid.shape[0]
            and 0 <= col < self.lane_id_grid.shape[1]
        ):
            return None
        index = int(self.lane_id_grid[row, col])
        return self.lane_tokens[index] if 0 <= index < len(self.lane_tokens) else None


@dataclass
class LanePlanResult:
    path: list[tuple[int, int]] | None
    start_rc: tuple[int, int] | None
    goal_rc: tuple[int, int] | None
    projected_start_xy: np.ndarray | None
    projected_goal_xy: np.ndarray | None
    start_projection_distance_m: float | None
    goal_projection_distance_m: float | None
    cost: float | None
    expanded_nodes: int
    validation: dict[str, Any]
    failure_code: str | None = None
    failure_reason: str | None = None


def load_lane_topology(
    npz_path: str | Path, topology_path: str | Path | None = None
) -> LaneTopology:
    """Load lane grids and metadata produced by the lane BEV builder."""
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as loaded:
        required = {"lane_id_grid", "lane_heading_grid", "centerline_distance_grid"}
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"lane map is missing arrays: {missing}")
        lane_id_grid = np.asarray(loaded["lane_id_grid"], dtype=np.int32)
        lane_heading_grid = np.asarray(loaded["lane_heading_grid"], dtype=np.float32)
        centerline_distance_grid = np.asarray(
            loaded["centerline_distance_grid"], dtype=np.float32
        )
        origin_xy = (
            np.asarray(loaded["origin_xy"], dtype=float)
            if "origin_xy" in loaded.files
            else np.zeros(2, dtype=float)
        )
        resolution = (
            float(loaded["resolution"]) if "resolution" in loaded.files else 1.0
        )
    if (
        lane_id_grid.shape != lane_heading_grid.shape
        or lane_id_grid.shape != centerline_distance_grid.shape
    ):
        raise ValueError("lane grid arrays must have identical shapes")

    if topology_path is None:
        topology_path = npz_path.with_name("lane_topology.json")
    payload = json.loads(Path(topology_path).read_text(encoding="utf-8"))
    tokens = [str(item) for item in payload.get("lane_tokens", [])]
    types = [str(item) for item in payload.get("lane_types", [])]
    if len(tokens) != len(types):
        raise ValueError("lane_tokens and lane_types length mismatch")
    transitions = {
        str(token): tuple(str(value) for value in values)
        for token, values in payload.get("transitions", {}).items()
    }
    geometries: dict[str, LaneGeometry] = {}
    for token, item in payload.get("geometries", {}).items():
        geometries[str(token)] = LaneGeometry(
            token=str(token),
            lane_type=str(item.get("lane_type", "lane")),
            polygon=np.asarray(item.get("polygon", []), dtype=float),
            centerline=np.asarray(item.get("centerline", []), dtype=float),
            heading=_finite_or_none(item.get("heading")),
            start_pose=_pose_or_none(item.get("start_pose")),
            end_pose=_pose_or_none(item.get("end_pose")),
            incoming=tuple(str(value) for value in item.get("incoming", [])),
            outgoing=tuple(str(value) for value in item.get("outgoing", [])),
        )
    diagnostics = dict(payload.get("diagnostics", {}))
    if not np.all((lane_id_grid >= INVALID_LANE) & (lane_id_grid < len(tokens))):
        raise ValueError("lane_id_grid contains an unknown lane index")
    if set(geometries) != set(tokens):
        raise ValueError(
            "lane metadata must include geometry and heading for every lane token"
        )
    return LaneTopology(
        lane_id_grid=lane_id_grid,
        lane_heading_grid=lane_heading_grid,
        centerline_distance_grid=centerline_distance_grid,
        lane_tokens=tokens,
        lane_types=types,
        transitions=transitions,
        geometries=geometries,
        diagnostics=diagnostics,
        origin_xy=origin_xy,
        resolution=resolution,
    )


def project_to_legal_lane(
    grid: OccupancyGrid2D,
    topology: LaneTopology,
    xy: np.ndarray,
    *,
    max_distance_m: float = 8.0,
) -> tuple[tuple[int, int], np.ndarray, float] | None:
    """Project a point to the nearest free lane cell with a known heading."""
    check_lane_grid_alignment(grid, topology)
    query = np.asarray(xy, dtype=float)[:2]
    if query.shape != (2,) or not np.all(np.isfinite(query)):
        return None
    candidates = np.argwhere(
        topology.lane_mask
        & np.any(np.isfinite(topology.headings), axis=0)
        & (grid.occupancy == 0)
    )
    if candidates.size == 0:
        return None
    points = np.column_stack(
        (
            grid.origin_xy[0] + candidates[:, 1] * grid.resolution,
            grid.origin_xy[1] + candidates[:, 0] * grid.resolution,
        )
    )
    distances = np.linalg.norm(points - query.reshape(1, 2), axis=1)
    index = int(np.argmin(distances))
    distance = float(distances[index])
    if distance > float(max_distance_m) + 1e-9:
        return None
    rc = (int(candidates[index, 0]), int(candidates[index, 1]))
    return rc, points[index], distance


def validate_lane_path(
    path: Iterable[tuple[int, int]],
    topology: LaneTopology,
    occupancy: np.ndarray | None = None,
    *,
    start_heading_xy: np.ndarray | None = None,
    direction_tolerance_deg: float = 75.0,
) -> dict[str, Any]:
    """Independently validate every segment, including polygon and cell edges."""
    return validate_continuous_lane_path(
        path,
        topology,
        occupancy,
        start_heading_xy=start_heading_xy,
        direction_tolerance_deg=direction_tolerance_deg,
    )


def check_lane_grid_alignment(grid: OccupancyGrid2D, topology: LaneTopology) -> None:
    if (
        grid.occupancy.shape != topology.lane_id_grid.shape
        or not np.allclose(grid.origin_xy, topology.origin_xy, atol=1e-6, rtol=0)
        or not math.isclose(grid.resolution, topology.resolution, abs_tol=1e-9)
    ):
        raise ValueError("lane and occupancy grids have different coordinate frames")


def lane_astar_plan(
    grid: OccupancyGrid2D,
    topology: LaneTopology,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    *,
    start_heading_xy: np.ndarray | None = None,
    max_projection_distance_m: float = 8.0,
    centerline_weight: float = 0.25,
    direction_tolerance_deg: float = 75.0,
) -> LanePlanResult:
    """A* over (row, column, lane ID); overlapping lanes retain their own states."""
    if not 0 < direction_tolerance_deg <= 90 or centerline_weight < 0:
        raise ValueError("invalid lane planning cost/direction configuration")
    check_lane_grid_alignment(grid, topology)
    start = project_to_legal_lane(
        grid, topology, start_xy, max_distance_m=max_projection_distance_m
    )
    goal = project_to_legal_lane(
        grid, topology, goal_xy, max_distance_m=max_projection_distance_m
    )
    if start is None:
        return _failed_lane_plan(
            "NO_LEGAL_LANE_START",
            "pose exceeds legal lane projection bound",
            None,
            None,
        )
    if goal is None:
        return _failed_lane_plan(
            "NO_LEGAL_LANE_GOAL",
            "target exceeds legal lane projection bound",
            start[0],
            None,
            start,
        )
    start_rc, start_world, start_distance = start
    goal_rc, goal_world, goal_distance = goal
    directions = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 0),
    )
    cosine = math.cos(math.radians(direction_tolerance_deg))
    queue, came_from, costs = [], {}, {}
    serial = 0
    for lane_id in topology.lanes_at(start_rc):
        heading = topology.heading_at(lane_id, start_world)
        if not math.isfinite(heading):
            continue
        if start_heading_xy is not None:
            vector = np.asarray(start_heading_xy, dtype=float)
            if (
                vector.shape != (2,)
                or not np.isfinite(vector).all()
                or np.linalg.norm(vector) < 1e-9
            ):
                continue
            if float(vector @ [math.cos(heading), math.sin(heading)]) < 0:
                continue
        state = (*start_rc, lane_id)
        came_from[state], costs[state] = None, 0.0
        heapq.heappush(queue, (0.0, serial, 0.0, state))
        serial += 1
    expanded = 0
    edge_cache = topology._edge_geometry_cache
    while queue:
        _, _, popped_cost, current = heapq.heappop(queue)
        if popped_cost > costs[current] + 1e-9:
            continue
        expanded += 1
        row, col, source = current
        if (row, col) == goal_rc:
            states = [current]
            while came_from[states[-1]] is not None:
                states.append(came_from[states[-1]])
            path = []
            for state in reversed(states):
                if not path or path[-1] != state[:2]:
                    path.append(state[:2])
            validation = validate_lane_path(
                path,
                topology,
                grid.occupancy,
                start_heading_xy=start_heading_xy,
                direction_tolerance_deg=direction_tolerance_deg,
            )
            if validation["valid"]:
                return LanePlanResult(
                    path,
                    start_rc,
                    goal_rc,
                    start_world,
                    goal_world,
                    start_distance,
                    goal_distance,
                    costs[current],
                    expanded,
                    validation,
                )
            # An independent validator rejection is a failure, never a free-space fallback.
            continue
        a = topology.world((row, col))
        for dr, dc in directions:
            rc = (row + dr, col + dc)
            if not grid.in_bounds(rc) or grid.occupancy[rc]:
                continue
            if dr and dc and (grid.occupancy[row, rc[1]] or grid.occupancy[rc[0], col]):
                continue
            for target in topology.lanes_at(rc):
                if dr == dc == 0 and source == target:
                    continue
                b = topology.world(rc)
                if not transition_allowed(topology, source, target, a, b):
                    continue
                key = (row, col, source, *rc, target, direction_tolerance_deg)
                legal = edge_cache.get(key)
                if legal is None:
                    legal = True
                    geometries = [
                        topology.geometries.get(topology.lane_tokens[i])
                        for i in {source, target}
                    ]
                    if any(g is None for g in geometries):
                        legal = legal and source == target
                    else:
                        legal = legal and segment_covered(
                            a, b, [g.polygon for g in geometries]
                        )
                    if dr or dc:
                        movement = np.array([dc, dr]) / math.hypot(dr, dc)
                        for lane_id, point in ((source, a), (target, b)):
                            heading = topology.heading_at(lane_id, point)
                            legal = (
                                legal
                                and math.isfinite(heading)
                                and float(
                                    movement @ [math.cos(heading), math.sin(heading)]
                                )
                                >= cosine - 1e-9
                            )
                    edge_cache[key] = legal
                if not legal:
                    continue
                neighbor = (*rc, target)
                penalty = float(topology.distances[target, rc[0], rc[1]])
                step_cost = max(math.hypot(dr, dc), 1e-6) * (
                    1 + centerline_weight * penalty / grid.resolution
                )
                new_cost = costs[current] + step_cost
                if new_cost + 1e-9 < costs.get(neighbor, float("inf")):
                    costs[neighbor], came_from[neighbor] = new_cost, current
                    serial += 1
                    heuristic = math.hypot(rc[0] - goal_rc[0], rc[1] - goal_rc[1])
                    heapq.heappush(
                        queue, (new_cost + heuristic, serial, new_cost, neighbor)
                    )
    result = _failed_lane_plan(
        "NO_LEGAL_LANE_PATH",
        "blocked/disconnected topology or incompatible direction",
        start_rc,
        goal_rc,
        start,
        goal,
    )
    result.expanded_nodes = expanded
    return result


def _failed_lane_plan(
    code: str, reason: str, start_rc=None, goal_rc=None, start=None, goal=None
) -> LanePlanResult:
    return LanePlanResult(
        None,
        start_rc,
        goal_rc,
        None if start is None else start[1],
        None if goal is None else goal[1],
        None if start is None else float(start[2]),
        None if goal is None else float(goal[2]),
        None,
        0,
        {"valid": False, "failure_code": code},
        code,
        reason,
    )


def _finite_or_none(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _pose_or_none(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    values = tuple(float(item) for item in value[:3])
    return values if all(math.isfinite(item) for item in values) else None
