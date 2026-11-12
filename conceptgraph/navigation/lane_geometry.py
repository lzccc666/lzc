"""Exact polygon boundary partitions for the lane safety validator."""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path


def contains_points(polygon: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Closed polygon membership, including edges (absolute tolerance 1e-8 m)."""
    polygon = np.asarray(polygon, dtype=float)
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 3:
        return np.zeros(len(points), dtype=bool)
    result = Path(polygon).contains_points(points)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = b - a
        denominator = float(edge @ edge)
        if denominator <= 1e-20:
            continue
        position = np.clip(np.sum((points - a) * edge, axis=1) / denominator, 0.0, 1.0)
        result |= np.linalg.norm(points - a - position[:, None] * edge, axis=1) <= 1e-8
    return result


def segment_breaks(
    a: np.ndarray, b: np.ndarray, polygons: list[np.ndarray]
) -> np.ndarray:
    """Split a segment at every polygon crossing, including collinear edges."""
    direction = b - a
    length2 = float(direction @ direction)
    if length2 < 1e-20:
        return np.asarray([0.0, 1.0])
    if not polygons:
        return np.asarray([0.0, 1.0])
    starts = np.concatenate(polygons)
    ends = np.concatenate([np.roll(polygon, -1, axis=0) for polygon in polygons])
    edges = ends - starts
    offsets = starts - a
    denominator = direction[0] * edges[:, 1] - direction[1] * edges[:, 0]
    nonparallel = np.abs(denominator) >= 1e-12
    indices = np.flatnonzero(nonparallel)
    along = (
        offsets[indices, 0] * edges[indices, 1]
        - offsets[indices, 1] * edges[indices, 0]
    ) / denominator[indices]
    cross = (
        offsets[indices, 0] * direction[1] - offsets[indices, 1] * direction[0]
    ) / denominator[indices]
    mask = (
        (along >= -1e-9) & (along <= 1 + 1e-9) & (cross >= -1e-9) & (cross <= 1 + 1e-9)
    )
    collinear = (~nonparallel) & (
        np.abs(offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0]) < 1e-9
    )
    values = np.r_[
        0.0,
        1.0,
        along[mask],
        np.sum((starts[collinear] - a) * direction, axis=1) / length2,
        np.sum((ends[collinear] - a) * direction, axis=1) / length2,
    ]
    return np.unique(np.round(np.clip(values, 0, 1), 12))


def segment_covered(a: np.ndarray, b: np.ndarray, polygons: list[np.ndarray]) -> bool:
    breaks = segment_breaks(a, b, polygons)
    values = np.unique(np.r_[breaks, (breaks[:-1] + breaks[1:]) / 2])
    points = a + values[:, None] * (b - a)
    covered = np.zeros(len(points), dtype=bool)
    for polygon in polygons:
        covered |= contains_points(polygon, points)
    return bool(np.all(covered))


def continuous_collision_cells(
    a_rc: np.ndarray, b_rc: np.ndarray, occupancy: np.ndarray
) -> list[list[int]]:
    """Supercover: include every occupied closed raster cell touched by a segment."""
    delta = b_rc - a_rc
    positions = [0.0, 1.0]
    for axis in (0, 1):
        if abs(delta[axis]) > 1e-12:
            lo, hi = sorted([a_rc[axis], b_rc[axis]])
            edges = np.arange(np.ceil(lo - 0.5), np.floor(hi - 0.5) + 1) + 0.5
            positions.extend(((edges - a_rc[axis]) / delta[axis]).tolist())
    breaks = np.unique(np.clip(positions, 0.0, 1.0))
    positions = np.unique(np.r_[breaks, (breaks[:-1] + breaks[1:]) / 2])
    touched = set()
    for point in a_rc + positions[:, None] * delta:
        lower = np.ceil(point - 0.5 - 1e-9).astype(int)
        upper = np.floor(point + 0.5 + 1e-9).astype(int)
        for row in range(lower[0], upper[0] + 1):
            for col in range(lower[1], upper[1] + 1):
                if (
                    not (
                        0 <= row < occupancy.shape[0] and 0 <= col < occupancy.shape[1]
                    )
                    or occupancy[row, col]
                ):
                    touched.add((row, col))
    return [list(cell) for cell in sorted(touched)]
