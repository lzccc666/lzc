from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, label as connected_component_label


@dataclass
class OccupancyGrid2D:
    occupancy: np.ndarray
    resolution: float
    origin_xy: np.ndarray
    clearance_m: np.ndarray | None = None
    source: str = "scene_map"
    fallback_used: bool = False
    metadata: dict | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.occupancy.shape)

    def world_to_grid(self, xy: np.ndarray) -> tuple[int, int]:
        xy = np.asarray(xy, dtype=float)
        col = int(np.round((xy[0] - self.origin_xy[0]) / self.resolution))
        row = int(np.round((xy[1] - self.origin_xy[1]) / self.resolution))
        return row, col

    def grid_to_world(self, rc: tuple[int, int]) -> np.ndarray:
        row, col = rc
        return np.asarray(
            [
                self.origin_xy[0] + col * self.resolution,
                self.origin_xy[1] + row * self.resolution,
            ],
            dtype=float,
        )

    def in_bounds(self, rc: tuple[int, int]) -> bool:
        row, col = rc
        return 0 <= row < self.occupancy.shape[0] and 0 <= col < self.occupancy.shape[1]

    def is_free(self, rc: tuple[int, int]) -> bool:
        return self.in_bounds(rc) and bool(self.occupancy[rc] == 0)


@dataclass
class ReachableGoalResult:
    goal_xy: np.ndarray
    goal_rc: tuple[int, int]
    reason: str
    fallback_used: bool
    fallback_distance_m: float
    target_in_map: bool
    target_free: bool
    start_connected: bool


@dataclass
class AStarResult:
    path: list[tuple[int, int]]
    cost: float
    expanded_nodes: int


@dataclass
class FreeComponent:
    mask: np.ndarray
    cells: np.ndarray

    def __bool__(self) -> bool:
        return bool(self.cells.size)

    def __len__(self) -> int:
        return int(len(self.cells))

    def __iter__(self):
        for row, col in self.cells:
            yield int(row), int(col)

    def contains(self, rc: tuple[int, int]) -> bool:
        row, col = rc
        return 0 <= row < self.mask.shape[0] and 0 <= col < self.mask.shape[1] and bool(self.mask[row, col])


def build_occupancy_grid_from_scene_map(
    scene_map,
    resolution: float,
    padding_m: float,
    robot_radius_m: float,
    object_bbox_scale: float = 0.18,
    max_object_radius_m: float = 1.2,
    ignore_dynamic_obstacles: bool = False,
    obstacle_profile: str = "thesis",
) -> OccupancyGrid2D:
    centers = []
    for obj in scene_map:
        center, _extent = _extract_bbox_center_extent(obj)
        centers.append(np.asarray(center, dtype=float)[:2])
    if not centers:
        raise ValueError("Scene map is empty, cannot build occupancy grid.")

    all_xy = np.vstack(centers)
    min_xy = all_xy.min(axis=0) - padding_m
    max_xy = all_xy.max(axis=0) + padding_m

    width = int(np.ceil((max_xy[0] - min_xy[0]) / resolution)) + 1
    height = int(np.ceil((max_xy[1] - min_xy[1]) / resolution)) + 1
    occupancy = np.zeros((height, width), dtype=np.uint8)

    inflate_cells = max(0, int(np.ceil(robot_radius_m / resolution)))
    for obj in scene_map:
        if ignore_dynamic_obstacles and _is_dynamic_obstacle(obj):
            continue
        center, extent = _extract_bbox_center_extent(obj)
        # Real scene maps in this project are sparse object-level abstractions.
        # Inflating the full 2D bbox footprint can saturate the whole local map,
        # so keep a narrower obstacle support around each object center.
        half = _object_obstacle_half_extent(extent, resolution, object_bbox_scale, max_object_radius_m)
        min_obj = center[:2] - half
        max_obj = center[:2] + half
        row_min, col_min = _world_to_grid_raw(min_obj, min_xy, resolution)
        row_max, col_max = _world_to_grid_raw(max_obj, min_xy, resolution)

        row_lo = max(0, min(row_min, row_max) - inflate_cells)
        row_hi = min(height - 1, max(row_min, row_max) + inflate_cells)
        col_lo = max(0, min(col_min, col_max) - inflate_cells)
        col_hi = min(width - 1, max(col_min, col_max) + inflate_cells)
        occupancy[row_lo:row_hi + 1, col_lo:col_hi + 1] = 1

    clearance = distance_transform_edt(1 - occupancy) * resolution
    return OccupancyGrid2D(
        occupancy=occupancy,
        resolution=resolution,
        origin_xy=min_xy,
        clearance_m=clearance,
        source="scene_map",
        fallback_used=True,
        metadata={
            "obstacle_profile": obstacle_profile,
            "robot_radius_m": float(robot_radius_m),
            "object_bbox_scale": float(object_bbox_scale),
            "max_object_radius_m": float(max_object_radius_m),
            "ignore_dynamic_obstacles": bool(ignore_dynamic_obstacles),
        },
    )


def load_occupancy_grid(
    bev_map_path: str | None,
    resolution: float,
    scene_map,
    padding_m: float,
    robot_radius_m: float,
    object_bbox_scale: float = 0.18,
    max_object_radius_m: float = 1.2,
    ignore_dynamic_obstacles: bool = False,
    obstacle_profile: str = "thesis",
) -> OccupancyGrid2D:
    if bev_map_path is None or not str(bev_map_path):
        return build_occupancy_grid_from_scene_map(
            scene_map,
            resolution,
            padding_m,
            robot_radius_m,
            object_bbox_scale=object_bbox_scale,
            max_object_radius_m=max_object_radius_m,
            ignore_dynamic_obstacles=ignore_dynamic_obstacles,
            obstacle_profile=obstacle_profile,
        )
    if not isinstance(bev_map_path, str):
        bev_map_path = str(bev_map_path)
    from pathlib import Path
    if not Path(bev_map_path).exists():
        return build_occupancy_grid_from_scene_map(
            scene_map,
            resolution,
            padding_m,
            robot_radius_m,
            object_bbox_scale=object_bbox_scale,
            max_object_radius_m=max_object_radius_m,
            ignore_dynamic_obstacles=ignore_dynamic_obstacles,
            obstacle_profile=obstacle_profile,
        )

    loaded = np.load(bev_map_path, allow_pickle=True)
    if isinstance(loaded, np.ndarray):
        occupancy = loaded
        origin_xy = np.zeros(2, dtype=float)
        grid_resolution = resolution
    else:
        keys = set(loaded.files)
        occupancy_key = "occupancy" if "occupancy" in keys else "bev"
        occupancy = loaded[occupancy_key]
        origin_xy = np.asarray(loaded["origin_xy"], dtype=float) if "origin_xy" in keys else np.zeros(2, dtype=float)
        grid_resolution = float(loaded["resolution"]) if "resolution" in keys else resolution
    occupancy = (occupancy > 0).astype(np.uint8)
    occupancy = _overlay_scene_object_obstacles(
        occupancy=occupancy,
        origin_xy=origin_xy,
        resolution=grid_resolution,
        scene_map=scene_map,
        robot_radius_m=robot_radius_m,
        object_bbox_scale=object_bbox_scale,
        max_object_radius_m=max_object_radius_m,
        ignore_dynamic_obstacles=ignore_dynamic_obstacles,
    )
    clearance = distance_transform_edt(1 - occupancy) * grid_resolution
    return OccupancyGrid2D(
        occupancy=occupancy,
        resolution=grid_resolution,
        origin_xy=origin_xy,
        clearance_m=clearance,
        source="bev_map",
        fallback_used=False,
        metadata={
            "obstacle_profile": obstacle_profile,
            "robot_radius_m": float(robot_radius_m),
            "object_bbox_scale": float(object_bbox_scale),
            "max_object_radius_m": float(max_object_radius_m),
            "ignore_dynamic_obstacles": bool(ignore_dynamic_obstacles),
        },
    )


def find_reachable_goal_near_point(
    grid: OccupancyGrid2D,
    target_xy: np.ndarray,
    max_search_radius_m: float,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    target_rc = grid.world_to_grid(target_xy)
    if grid.is_free(target_rc):
        return grid.grid_to_world(target_rc), target_rc

    if not grid.in_bounds(target_rc):
        free_cells = np.argwhere(grid.occupancy == 0)
        if free_cells.size == 0:
            return None
        deltas = free_cells - np.asarray(target_rc, dtype=float)
        distances = np.hypot(deltas[:, 0], deltas[:, 1])
        best_idx = int(np.argmin(distances))
        best_rc = tuple(int(v) for v in free_cells[best_idx])
        return grid.grid_to_world(best_rc), best_rc

    # Search progressively larger rings around the target until we find a free cell.
    max_radius_cells = max(1, int(np.ceil(max_search_radius_m / grid.resolution)))
    best = None
    best_score = float("inf")
    for radius in range(1, max_radius_cells + 1):
        for d_row in range(-radius, radius + 1):
            for d_col in range(-radius, radius + 1):
                if max(abs(d_row), abs(d_col)) != radius:
                    continue
                rc = (target_rc[0] + d_row, target_rc[1] + d_col)
                if not grid.is_free(rc):
                    continue
                dist = float(np.hypot(d_row, d_col))
                clearance_penalty = 0.0
                if grid.clearance_m is not None and grid.in_bounds(rc):
                    clearance_penalty = 1.0 / max(float(grid.clearance_m[rc]), 1e-3)
                score = dist + 0.5 * clearance_penalty
                if score < best_score:
                    best_score = score
                    best = rc
        if best is not None:
            break
    if best is None:
        free_cells = np.argwhere(grid.occupancy == 0)
        if free_cells.size == 0:
            return None
        deltas = free_cells - np.asarray(target_rc, dtype=float)
        distances = np.hypot(deltas[:, 0], deltas[:, 1])
        best_idx = int(np.argmin(distances))
        best = tuple(int(v) for v in free_cells[best_idx])
    return grid.grid_to_world(best), best


def find_reachable_goal_near_point_from_start(
    grid: OccupancyGrid2D,
    target_xy: np.ndarray,
    start_rc: tuple[int, int],
    max_search_radius_m: float,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Find a target-adjacent free cell that is connected to the current start."""
    if not grid.is_free(start_rc):
        return None

    component = _free_component(grid, start_rc)
    if not component:
        return None

    target_rc = grid.world_to_grid(target_xy)
    max_radius_cells = max(1, int(np.ceil(max_search_radius_m / grid.resolution)))
    target_arr = np.asarray(target_rc, dtype=float)

    best = None
    best_score = float("inf")
    for rc in component:
        delta = np.asarray(rc, dtype=float) - target_arr
        dist_cells = float(np.hypot(delta[0], delta[1]))
        if dist_cells > max_radius_cells:
            continue
        clearance_penalty = 0.0
        if grid.clearance_m is not None:
            clearance_penalty = 1.0 / max(float(grid.clearance_m[rc]), 1e-3)
        score = dist_cells + 0.5 * clearance_penalty
        if score < best_score:
            best_score = score
            best = rc

    if best is None:
        # Fall back to the nearest start-connected cell; mark this in caller debug.
        for rc in component:
            delta = np.asarray(rc, dtype=float) - target_arr
            dist_cells = float(np.hypot(delta[0], delta[1]))
            if dist_cells < best_score:
                best_score = dist_cells
                best = rc

    if best is None:
        return None
    return grid.grid_to_world(best), best


def evaluate_reachable_goal_near_point_from_start(
    grid: OccupancyGrid2D,
    target_xy: np.ndarray,
    start_rc: tuple[int, int],
    max_search_radius_m: float,
    allow_global_fallback: bool = True,
    component: FreeComponent | set[tuple[int, int]] | None = None,
) -> ReachableGoalResult | None:
    if not grid.is_free(start_rc):
        return None

    component = _free_component(grid, start_rc) if component is None else component
    if not component:
        return None

    target_rc = grid.world_to_grid(target_xy)
    target_in_map = grid.in_bounds(target_rc)
    target_free = grid.is_free(target_rc)
    target_arr = np.asarray(target_rc, dtype=float)
    max_radius_cells = max(1, int(np.ceil(max_search_radius_m / grid.resolution)))

    best = None
    best_score = float("inf")
    if isinstance(component, FreeComponent):
        best, best_score = _best_component_cell_near_target(
            grid=grid,
            cells=component.cells,
            target_arr=target_arr,
            max_radius_cells=max_radius_cells,
        )
    else:
        for rc in component:
            delta = np.asarray(rc, dtype=float) - target_arr
            dist_cells = float(np.hypot(delta[0], delta[1]))
            if dist_cells > max_radius_cells:
                continue
            clearance_penalty = 0.0
            if grid.clearance_m is not None:
                clearance_penalty = 1.0 / max(float(grid.clearance_m[rc]), 1e-3)
            score = dist_cells + 0.5 * clearance_penalty
            if score < best_score:
                best_score = score
                best = rc

    reason = "direct_free" if target_free and best == target_rc else "fallback_near_target"
    if best is None and allow_global_fallback:
        reason = "out_of_map" if not target_in_map else ("occupied_target" if not target_free else "not_connected_to_start")
        if isinstance(component, FreeComponent):
            best, best_score = _nearest_component_cell(component.cells, target_arr)
        else:
            for rc in component:
                delta = np.asarray(rc, dtype=float) - target_arr
                dist_cells = float(np.hypot(delta[0], delta[1]))
                if dist_cells < best_score:
                    best_score = dist_cells
                    best = rc
        if best is not None:
            reason = f"{reason}_nearest_connected"

    if best is None:
        return None

    goal_xy = grid.grid_to_world(best)
    fallback_distance_m = float(np.linalg.norm(goal_xy - np.asarray(target_xy, dtype=float)[:2]))
    return ReachableGoalResult(
        goal_xy=goal_xy,
        goal_rc=best,
        reason=reason,
        fallback_used=fallback_distance_m > max(grid.resolution * 1.5, 1e-6),
        fallback_distance_m=fallback_distance_m,
        target_in_map=bool(target_in_map),
        target_free=bool(target_free),
        start_connected=True,
    )


def astar_plan(
    grid: OccupancyGrid2D,
    start_rc: tuple[int, int],
    goal_rc: tuple[int, int],
    semantic_bias: np.ndarray | None = None,
    clearance_weight: float = 0.0,
) -> tuple[list[tuple[int, int]], float] | None:
    result = astar_plan_with_stats(
        grid=grid,
        start_rc=start_rc,
        goal_rc=goal_rc,
        semantic_bias=semantic_bias,
        clearance_weight=clearance_weight,
    )
    if result is None:
        return None
    return result.path, result.cost


def astar_plan_with_stats(
    grid: OccupancyGrid2D,
    start_rc: tuple[int, int],
    goal_rc: tuple[int, int],
    semantic_bias: np.ndarray | None = None,
    clearance_weight: float = 0.0,
) -> AStarResult | None:
    if not grid.is_free(start_rc) or not grid.is_free(goal_rc):
        return None

    open_heap: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, start_rc))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_cost = {start_rc: 0.0}
    expanded_nodes = 0

    while open_heap:
        _, current = heapq.heappop(open_heap)
        expanded_nodes += 1
        if current == goal_rc:
            path = _reconstruct_path(came_from, current)
            return AStarResult(path=path, cost=float(g_cost[current]), expanded_nodes=expanded_nodes)

        for neighbor, step_cost in _neighbors(grid, current):
            semantic_cost = 0.0
            if semantic_bias is not None:
                semantic_cost = float(semantic_bias[neighbor])
            clearance_cost = 0.0
            if clearance_weight > 0 and grid.clearance_m is not None:
                clearance = float(grid.clearance_m[neighbor])
                clearance_cost = clearance_weight / max(clearance, 1e-3)
            tentative = g_cost[current] + step_cost + semantic_cost
            tentative += clearance_cost
            if tentative >= g_cost.get(neighbor, float("inf")):
                continue
            came_from[neighbor] = current
            g_cost[neighbor] = tentative
            f_score = tentative + _heuristic(neighbor, goal_rc)
            heapq.heappush(open_heap, (f_score, neighbor))
    return None


def compute_semantic_bias_grid(
    grid: OccupancyGrid2D,
    preferred_goal_rc: tuple[int, int],
    weight: float,
) -> np.ndarray:
    if weight <= 0:
        return np.zeros_like(grid.occupancy, dtype=float)
    rows, cols = np.indices(grid.occupancy.shape)
    distances = np.hypot(rows - preferred_goal_rc[0], cols - preferred_goal_rc[1])
    distances = distances / max(float(np.max(distances)), 1.0)
    return weight * distances


def compute_clearance_bias_grid(grid: OccupancyGrid2D, weight: float) -> np.ndarray:
    if weight <= 0 or grid.clearance_m is None:
        return np.zeros_like(grid.occupancy, dtype=float)
    clearance = np.asarray(grid.clearance_m, dtype=float)
    return weight / np.maximum(clearance, 1e-3)


def path_grid_to_world(grid: OccupancyGrid2D, path_grid: list[tuple[int, int]], z: float) -> np.ndarray:
    world_path = []
    for rc in path_grid:
        xy = grid.grid_to_world(rc)
        world_path.append([xy[0], xy[1], z])
    return np.asarray(world_path, dtype=float)


def smooth_grid_path(
    grid: OccupancyGrid2D,
    path_grid: list[tuple[int, int]],
    chaikin_iterations: int = 2,
) -> list[tuple[int, int]]:
    """Shortcut and smooth an A* grid path while keeping points in free cells."""
    if len(path_grid) <= 2:
        return path_grid
    shortcut = _shortcut_path(grid, path_grid)
    smoothed_xy = _chaikin_smooth([grid.grid_to_world(rc) for rc in shortcut], chaikin_iterations)
    smoothed_rc: list[tuple[int, int]] = []
    for xy in smoothed_xy:
        rc = grid.world_to_grid(xy)
        if not grid.is_free(rc):
            nearest = _nearest_free_cell(grid, rc, max_radius_cells=4)
            if nearest is None:
                return shortcut
            rc = nearest
        if not smoothed_rc or rc != smoothed_rc[-1]:
            smoothed_rc.append(rc)
    if smoothed_rc[0] != path_grid[0]:
        smoothed_rc.insert(0, path_grid[0])
    if smoothed_rc[-1] != path_grid[-1]:
        smoothed_rc.append(path_grid[-1])
    return smoothed_rc


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _neighbors(grid: OccupancyGrid2D, current: tuple[int, int]):
    row, col = current
    for d_row, d_col in [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]:
        nxt = (row + d_row, col + d_col)
        if grid.is_free(nxt):
            yield nxt, float(np.hypot(d_row, d_col))


def _free_component(grid: OccupancyGrid2D, start_rc: tuple[int, int]) -> FreeComponent:
    mask = np.zeros_like(grid.occupancy, dtype=bool)
    if not grid.is_free(start_rc):
        return FreeComponent(mask=mask, cells=np.empty((0, 2), dtype=np.int32))
    labels, _num_labels = connected_component_label(grid.occupancy == 0, structure=np.ones((3, 3), dtype=np.uint8))
    start_label = int(labels[start_rc])
    if start_label <= 0:
        return FreeComponent(mask=mask, cells=np.empty((0, 2), dtype=np.int32))
    mask = labels == start_label
    return FreeComponent(mask=mask, cells=np.argwhere(mask).astype(np.int32))


def free_component(grid: OccupancyGrid2D, start_rc: tuple[int, int]) -> FreeComponent:
    return _free_component(grid, start_rc)


def _best_component_cell_near_target(
    grid: OccupancyGrid2D,
    cells: np.ndarray,
    target_arr: np.ndarray,
    max_radius_cells: int,
) -> tuple[tuple[int, int] | None, float]:
    if cells.size == 0:
        return None, float("inf")
    deltas = cells.astype(float) - target_arr.reshape(1, 2)
    distances = np.hypot(deltas[:, 0], deltas[:, 1])
    within = distances <= float(max_radius_cells)
    if not bool(np.any(within)):
        return None, float("inf")
    scores = distances[within]
    candidate_cells = cells[within]
    if grid.clearance_m is not None:
        clearances = grid.clearance_m[candidate_cells[:, 0], candidate_cells[:, 1]]
        scores = scores + 0.5 / np.maximum(clearances, 1e-3)
    best_idx = int(np.argmin(scores))
    best_cell = candidate_cells[best_idx]
    return (int(best_cell[0]), int(best_cell[1])), float(scores[best_idx])


def _nearest_component_cell(cells: np.ndarray, target_arr: np.ndarray) -> tuple[tuple[int, int] | None, float]:
    if cells.size == 0:
        return None, float("inf")
    deltas = cells.astype(float) - target_arr.reshape(1, 2)
    distances = np.hypot(deltas[:, 0], deltas[:, 1])
    best_idx = int(np.argmin(distances))
    best_cell = cells[best_idx]
    return (int(best_cell[0]), int(best_cell[1])), float(distances[best_idx])


def _reconstruct_path(came_from: dict[tuple[int, int], tuple[int, int]], current: tuple[int, int]) -> list[tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _shortcut_path(grid: OccupancyGrid2D, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(path) <= 2:
        return path
    result = [path[0]]
    anchor_idx = 0
    while anchor_idx < len(path) - 1:
        next_idx = len(path) - 1
        while next_idx > anchor_idx + 1 and not _line_is_free(grid, path[anchor_idx], path[next_idx]):
            next_idx -= 1
        result.append(path[next_idx])
        anchor_idx = next_idx
    return result


def _line_is_free(grid: OccupancyGrid2D, start: tuple[int, int], end: tuple[int, int]) -> bool:
    for rc in _bresenham_cells(start, end):
        if not grid.is_free(rc):
            return False
    return True


def _bresenham_cells(start: tuple[int, int], end: tuple[int, int]):
    r0, c0 = start
    r1, c1 = end
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        yield (r, c)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


def _chaikin_smooth(points: list[np.ndarray], iterations: int) -> list[np.ndarray]:
    if len(points) <= 2 or iterations <= 0:
        return points
    smoothed = [np.asarray(point, dtype=float) for point in points]
    for _ in range(iterations):
        next_points = [smoothed[0]]
        for p0, p1 in zip(smoothed[:-1], smoothed[1:]):
            next_points.append(0.75 * p0 + 0.25 * p1)
            next_points.append(0.25 * p0 + 0.75 * p1)
        next_points.append(smoothed[-1])
        smoothed = next_points
    return smoothed


def _nearest_free_cell(grid: OccupancyGrid2D, center: tuple[int, int], max_radius_cells: int) -> tuple[int, int] | None:
    if grid.is_free(center):
        return center
    best = None
    best_dist = float("inf")
    for radius in range(1, max_radius_cells + 1):
        for d_row in range(-radius, radius + 1):
            for d_col in range(-radius, radius + 1):
                if max(abs(d_row), abs(d_col)) != radius:
                    continue
                rc = (center[0] + d_row, center[1] + d_col)
                if not grid.is_free(rc):
                    continue
                dist = float(np.hypot(d_row, d_col))
                if dist < best_dist:
                    best_dist = dist
                    best = rc
        if best is not None:
            return best
    return None


def _world_to_grid_raw(xy: np.ndarray, origin_xy: np.ndarray, resolution: float) -> tuple[int, int]:
    col = int(np.round((xy[0] - origin_xy[0]) / resolution))
    row = int(np.round((xy[1] - origin_xy[1]) / resolution))
    return row, col


def _extract_bbox_center_extent(obj) -> tuple[np.ndarray, np.ndarray]:
    bbox = obj.get("bbox")
    if bbox is not None:
        if isinstance(bbox, dict):
            center = np.asarray(bbox["center"], dtype=float)
            extent = np.asarray(bbox["extent"], dtype=float)
        else:
            center = np.asarray(bbox.center, dtype=float)
            extent = np.asarray(bbox.extent, dtype=float)
        return center, extent

    bbox_np = obj.get("bbox_np")
    if bbox_np is not None:
        points = np.asarray(bbox_np, dtype=float)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Invalid bbox_np shape: {points.shape}")
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = (mins + maxs) / 2.0
        extent = maxs - mins
        return center, extent

    raise KeyError("Scene object has neither bbox nor bbox_np")


def _overlay_scene_object_obstacles(
    occupancy: np.ndarray,
    origin_xy: np.ndarray,
    resolution: float,
    scene_map,
    robot_radius_m: float,
    object_bbox_scale: float = 0.18,
    max_object_radius_m: float = 1.2,
    ignore_dynamic_obstacles: bool = False,
) -> np.ndarray:
    combined = np.array(occupancy, copy=True, dtype=np.uint8)
    inflate_cells = max(0, int(np.ceil(robot_radius_m / resolution)))
    height, width = combined.shape

    for obj in scene_map:
        if ignore_dynamic_obstacles and _is_dynamic_obstacle(obj):
            continue
        center, extent = _extract_bbox_center_extent(obj)
        half = _object_obstacle_half_extent(extent, resolution, object_bbox_scale, max_object_radius_m)
        min_obj = center[:2] - half
        max_obj = center[:2] + half
        row_min, col_min = _world_to_grid_raw(min_obj, origin_xy, resolution)
        row_max, col_max = _world_to_grid_raw(max_obj, origin_xy, resolution)

        row_lo = max(0, min(row_min, row_max) - inflate_cells)
        row_hi = min(height - 1, max(row_min, row_max) + inflate_cells)
        col_lo = max(0, min(col_min, col_max) - inflate_cells)
        col_hi = min(width - 1, max(col_min, col_max) + inflate_cells)
        if row_lo > row_hi or col_lo > col_hi:
            continue
        combined[row_lo:row_hi + 1, col_lo:col_hi + 1] = 1

    return combined


def _object_obstacle_half_extent(
    extent: np.ndarray,
    resolution: float,
    object_bbox_scale: float,
    max_object_radius_m: float,
) -> np.ndarray:
    extent = np.asarray(extent, dtype=float)
    half = np.maximum(extent[:2] * float(object_bbox_scale), float(resolution))
    if max_object_radius_m > 0:
        half = np.minimum(half, float(max_object_radius_m))
    return half


def _is_dynamic_obstacle(obj: dict) -> bool:
    return str(obj.get("motion_state", "")).lower() == "dynamic" or bool(obj.get("is_short_term", False))
