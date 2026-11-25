"""External path audit using Shapely, independent of planner geometry routines."""
from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union


def continuous_collisions(cells, occupancy):
    cells = np.asarray(cells, dtype=float).reshape(-1, 2)
    if not len(cells):
        return []
    found = set()
    domain = box(-0.5, -0.5, occupancy.shape[1]-0.5, occupancy.shape[0]-0.5)
    segments = list(zip(cells[:-1], cells[1:])) if len(cells) > 1 else [(cells[0], cells[0])]
    for a, b in segments:
        geometry = Point(a[::-1]) if np.allclose(a, b, rtol=0, atol=1e-12) else LineString([a[::-1], b[::-1]])
        if not domain.covers(geometry):
            found.add((-1, -1))
        low = np.maximum(np.floor(np.minimum(a, b)-0.5).astype(int), 0)
        high = np.minimum(np.ceil(np.maximum(a, b)+0.5).astype(int), np.array(occupancy.shape)-1)
        rows, cols = np.nonzero(occupancy[low[0]:high[0]+1, low[1]:high[1]+1])
        rows, cols = rows+low[0], cols+low[1]
        if len(rows):
            rectangles = shapely.box(cols-0.5, rows-0.5, cols+0.5, rows+0.5)
            touched = shapely.intersects(rectangles, geometry)
            found.update((int(r), int(c)) for r, c in zip(rows[touched], cols[touched]))
    return sorted(found)


def nearest_heading_breaks(a, b, centerline):
    """Exact 1D lower envelope of squared distances to centerline samples."""
    d = b-a
    offsets = a-np.asarray(centerline)[:, :2]
    slopes = 2*offsets@d
    intercepts = np.sum(offsets**2, axis=1)
    ordered = sorted(zip(slopes, intercepts), key=lambda pair: (-pair[0], pair[1]))
    unique = []
    for slope, intercept in ordered:
        if unique and abs(slope-unique[-1][0]) < 1e-12:
            continue
        unique.append((float(slope), float(intercept)))
    hull, starts = [], []
    for slope, intercept in unique:
        start = -math.inf
        while hull:
            previous_slope, previous_intercept = hull[-1]
            start = (intercept-previous_intercept)/(previous_slope-slope)
            if start > starts[-1]+1e-12:
                break
            hull.pop()
            starts.pop()
        if not hull:
            start = -math.inf
        hull.append((slope, intercept))
        starts.append(start)
    return [t for t in starts if 0 < t < 1]


def geometry_coordinates(geometry):
    if geometry.is_empty:
        return []
    if hasattr(geometry, 'geoms'):
        return [xy for part in geometry.geoms for xy in geometry_coordinates(part)]
    return list(geometry.coords)


class IndependentLaneAudit:
    def __init__(self, topology):
        self.topology = topology
        self.polygons = {token: Polygon(g.polygon) for token, g in topology.geometries.items()}
        self.union = unary_union(list(self.polygons.values()))

    def _legal_handover(self, source, target, xy):
        if source == target:
            return True
        topology = self.topology
        a, b = topology.geometries[source], topology.geometries[target]
        tolerance = float(topology.diagnostics.get('transition_endpoint_tolerance_m', 2.0))
        return target in topology.transitions.get(source, ()) and a.end_pose is not None and b.start_pose is not None and np.linalg.norm(xy-np.asarray(a.end_pose[:2])) <= tolerance+1e-8 and np.linalg.norm(xy-np.asarray(b.start_pose[:2])) <= tolerance+1e-8

    def validate(self, worlds):
        worlds = np.asarray(worlds, dtype=float).reshape(-1, 2)
        if not len(worlds):
            return {'valid': False, 'reason': 'empty_path', 'checked_intervals': 0}
        if len(worlds) == 1:
            return {'valid': bool(self.union.buffer(1e-8).covers(Point(worlds[0]))), 'reason': 'stationary_point', 'checked_intervals': 1}
        states, intervals, total_outside = None, 0, 0.0
        for a, b in zip(worlds[:-1], worlds[1:]):
            if np.linalg.norm(b-a) < 1e-10:
                continue
            line = LineString([a, b])
            outside = line.difference(self.union.buffer(1e-8)).length
            total_outside += outside
            if outside > 1e-7:
                return {'valid': False, 'reason': 'polygon_segment_gap', 'outside_length_m': total_outside, 'checked_intervals': intervals}
            intersecting = {token: p for token, p in self.polygons.items() if p.intersects(line)}
            breaks = [0.0, 1.0]
            for token, polygon in intersecting.items():
                breaks += [float(line.project(Point(xy))/line.length) for xy in geometry_coordinates(line.intersection(polygon.boundary))]
                breaks += nearest_heading_breaks(a, b, self.topology.geometries[token].centerline)
            breaks = np.unique(np.round(np.clip(breaks, 0, 1), 12))
            direction = math.atan2((b-a)[1], (b-a)[0])
            for left, right in zip(breaks[:-1], breaks[1:]):
                midpoint = a+(b-a)*((left+right)/2)
                point = Point(midpoint)
                choices = set()
                for token, polygon in intersecting.items():
                    if not polygon.buffer(1e-8).covers(point):
                        continue
                    centerline = self.topology.geometries[token].centerline
                    index = np.argmin(np.sum((centerline[:, :2]-midpoint)**2, axis=1))
                    angle = abs((direction-float(centerline[index, 2])+math.pi)%(2*math.pi)-math.pi)
                    if angle <= math.radians(75)+1e-8:
                        choices.add(token)
                intervals += 1
                if not choices:
                    return {'valid': False, 'reason': 'direction_or_coverage', 'checked_intervals': intervals}
                if states is not None:
                    boundary = a+(b-a)*left
                    choices = {target for target in choices if any(self._legal_handover(source, target, boundary) for source in states)}
                    if not choices:
                        return {'valid': False, 'reason': 'illegal_transition', 'checked_intervals': intervals}
                states = choices
        return {'valid': states is not None, 'reason': None if states else 'all_segments_stationary', 'checked_intervals': intervals, 'outside_length_m': total_outside, 'implementation': 'Shapely full segment geometry + exact heading Voronoi partition + endpoint-constrained directed DP'}
