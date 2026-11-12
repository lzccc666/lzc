"""Describe physical coverage from official polygons and recorded poses."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np
from shapely import make_valid
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT, PROJECT_ROOT
from conceptgraph.scripts.run_incremental_semantic_navigation import write_json


def main():
    root = DEFAULT_ROOT
    scenes = json.loads((root / 'scene_selection_manifest.json').read_text())['scenes']
    maps, sources, repairs = {}, [], []
    for location in sorted({scene['strata']['location'] for scene in scenes}):
        path = PROJECT_ROOT / 'dataset/nuscenes_map_expansion/expansion' / f'{location}.json'
        content = path.read_bytes()
        data = json.loads(content)
        sources.append({'path': str(path), 'sha256': hashlib.sha256(content).hexdigest()})
        nodes = {row['token']: (row['x'], row['y']) for row in data['node']}
        polygons = {row['token']: row for row in data['polygon']}
        layers = {}
        for layer in ('carpark_area', 'road_segment'):
            shapes = []
            for record in data[layer]:
                if layer == 'road_segment' and not record['is_intersection']:
                    continue
                polygon = polygons[record['polygon_token']]
                shape = Polygon([nodes[token] for token in polygon['exterior_node_tokens']], [[nodes[token] for token in hole['node_tokens']] for hole in polygon['holes']])
                if not shape.is_valid:
                    repairs.append({'location': location, 'layer': layer, 'polygon_token': polygon['token'], 'operation': 'Shapely make_valid for descriptive coverage only; planning maps unchanged'})
                    shape = make_valid(shape)
                shapes.append(shape)
            layers[layer] = (STRtree(shapes), shapes)
        maps[location] = layers
        sources[-1]['layer_geometry_counts'] = {layer: len(value[1]) for layer, value in layers.items()}
    results = []
    for scene in scenes:
        pose_path = root / 'converted' / scene['scene'] / 'poses.txt'
        poses = np.loadtxt(pose_path).reshape(-1, 4, 4)
        points = [Point(xy) for xy in poses[:, :2, 3]]
        layers = maps[scene['strata']['location']]
        parking_tree, parking_shapes = layers['carpark_area']
        intersection_tree, _ = layers['road_segment']
        distances = [point.distance(parking_shapes[int(parking_tree.nearest(point))]) for point in points] if parking_shapes else []
        parking_frames = sum(bool(len(parking_tree.query(point, predicate='intersects'))) for point in points) if parking_shapes else None
        intersection_frames = sum(bool(len(intersection_tree.query(point, predicate='intersects'))) for point in points)
        minimum_heading_norm = float(np.linalg.norm(poses[:, :2, 2], axis=1).min())
        if minimum_heading_norm < 0.1:
            raise ValueError(f'camera heading projection too small for reliable coverage classification: {scene["scene"]}')
        headings = np.unwrap(np.arctan2(poses[:, 1, 2], poses[:, 0, 2]))
        heading_span = float(np.rad2deg(np.ptp(headings)))
        results.append({'scene': scene['scene'], 'role': scene['role'], 'location': scene['strata']['location'], 'pose_file_sha256': hashlib.sha256(pose_path.read_bytes()).hexdigest(), 'frames': len(points), 'camera_trajectory_frames_in_official_carpark': parking_frames, 'camera_trajectory_frames_in_official_intersection': intersection_frames, 'minimum_official_carpark_distance_m': min(distances) if distances else None, 'minimum_horizontal_forward_axis_norm': minimum_heading_norm, 'heading_span_deg': heading_span, 'heading_total_absolute_turn_deg': float(np.rad2deg(np.abs(np.diff(headings)).sum())), 'physical_parking_context': 'inside_carpark' if parking_frames else 'map_layer_absent' if not distances else 'within_10m_of_carpark' if min(distances) <= 10 else 'farther_than_10m_of_carpark', 'physical_intersection_context': 'intersects_official_intersection' if intersection_frames else 'no_recorded_pose_in_intersection', 'physical_turn_context': 'heading_span_at_least_15deg' if heading_span >= 15 else 'heading_span_under_15deg'})
    groups = []
    for role in ('legacy_regression', 'development', 'sealed_test'):
        members = [row for row in results if row['role'] == role]
        groups.append({'role': role, 'scene_count': len(members), 'strata': {dimension: dict(Counter(row[dimension] for row in members)) for dimension in ('physical_parking_context', 'physical_intersection_context', 'physical_turn_context')}})
    report = {'passed': bool(scenes) and len(results) == len(scenes), 'expected_scene_count': len(scenes), 'selection_or_queries_changed': False, 'prospective_sampling_claimed': False, 'scope': 'Post-selection descriptive coverage from official carpark/intersection polygons and actual CAM_FRONT +Z heading/translation in the aligned global XY map frame; no method outcomes used.', 'stratum_thresholds': {'carpark_near_distance_m': 10, 'heading_span_deg': 15}, 'scene_pose_points_are_not_independent_statistical_samples': True, 'map_sources': sources, 'descriptive_polygon_repairs': repairs, 'scenes': results, 'role_summaries': groups, 'limitations': 'A parked vehicle in a scene description does not prove the ego camera entered a mapped carpark. No recorded pose in an intersection does not rule out a visible/nearby intersection. Camera trajectory and heading bins are coverage descriptors, not manually reviewed road-type truth.'}
    write_json(root / 'physical_coverage_audit.json', report)
    print(json.dumps(groups, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
