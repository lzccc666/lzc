"""显式无损移除连续相同XY采样；保留原输入、道路、端点和有向拓扑。"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from shapely.geometry import LineString

from conceptgraph.completion.topology_vehicle import DirectedVehicleCorridor


def canonical_lane(lane):
    points = np.asarray(lane['centerline'], dtype=float)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('中心线需至少两个有限XY采样')
    # 只删除相邻XY完全相等的零长度边；不使用容差，不删除非相邻返回点。
    keep = np.r_[True, np.any(points[1:, :2] != points[:-1, :2], axis=1)]
    selected = np.flatnonzero(keep)
    if len(selected) < 2:
        raise ValueError('中心线仅有一个不同位置')
    original = LineString(points[:, :2])
    canonical = LineString(points[selected, :2])
    before = np.diff(points[:, :2], axis=0)
    nonzero = before[np.any(before != 0, axis=1)]
    assert np.array_equal(nonzero, np.diff(points[selected, :2], axis=0))
    assert original.equals(canonical) and original.hausdorff_distance(canonical) == 0
    assert original.length == canonical.length
    source = json.dumps(lane['centerline'], ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    result = dict(lane, centerline=points[selected].tolist())
    receipt = dict(token=lane['token'], original_rows=len(points), canonical_rows=len(selected),
        removed_exact_zero_segments=int((~keep).sum()), kept_original_indices=selected.tolist(),
        removed_original_indices=np.flatnonzero(~keep).tolist(), original_centerline_sha256=hashlib.sha256(source).hexdigest(),
        ordered_nonzero_vectors_identical=True, geometric_curve_equal=True, hausdorff_distance_m=0.,
        original_length_m=original.length, canonical_length_m=canonical.length,
        endpoints_and_polygon_and_connectivity_modified=False, nonzero_short_segments_removed=0)
    return result, receipt


def canonical_vehicle_corridor(lanes, transitions, drivable, obstacles):
    converted = [canonical_lane(lane) for lane in lanes]
    corridor = DirectedVehicleCorridor([lane for lane, _ in converted], transitions, drivable, obstacles)
    return corridor, [receipt for _, receipt in converted]
