from __future__ import annotations

from conceptgraph.navigation.planner import (
    astar_plan,
    path_is_collision_free,
    smooth_grid_path,
)

from .conftest import make_grid


def test_astar_returns_none_for_disconnected_goal() -> None:
    grid = make_grid(shape=(12, 12), occupied=[(slice(None), 6)])

    assert astar_plan(grid, (2, 2), (8, 9)) is None


def test_continuous_collision_check_detects_obstacle_between_waypoints() -> None:
    grid = make_grid(shape=(12, 12), occupied=[(5, 5)])

    assert not path_is_collision_free(grid, [(2, 2), (8, 8)])


def test_smoothed_path_is_rechecked_continuously() -> None:
    grid = make_grid(shape=(14, 14), occupied=[(slice(5, 9), slice(5, 9))])
    planned = astar_plan(grid, (2, 2), (11, 11))
    assert planned is not None

    path, _cost = planned
    smoothed = smooth_grid_path(grid, path, chaikin_iterations=3)

    assert smoothed[0] == path[0]
    assert smoothed[-1] == path[-1]
    assert path_is_collision_free(grid, smoothed)
