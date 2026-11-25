"""独立复核隔离终点标注的文件、整车包络及当前官方盒，不评价可达性。"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon, shape
from shapely.ops import unary_union

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump
from conceptgraph.nuscenes.expansion_benchmark import iter_records


def audit_region(region, target, context, policy, vehicle):
    """从数值和保存的矢量图重建，不调用生成器或待评测规划器。"""
    nominal = np.asarray(region['nominal_rear_axle_xy'])
    yaw = float(region['heading_rad'])
    region_polygon = Polygon(region['polygon_xy'])
    envelope = shape(region['envelope_geojson'])
    road, lanes = shape(context['drivable_geojson']), shape(context['lane_union_geojson'])
    obstacles = unary_union([shape(o['polygon_geojson']) for o in context['objects']])
    failures = []
    if not region_polygon.is_valid or region_polygon.area <= 0:
        failures.append('invalid_position_region')
    if not road.covers(envelope) or not lanes.covers(envelope):
        failures.append('envelope_outside_road_or_lanes')
    if envelope.intersects(obstacles):
        failures.append('envelope_intersects_official_obstacle')
    radius = max(math.dist(nominal, xy) for xy in list(region_polygon.exterior.coords)[:-1])
    distance = target.distance(Point(nominal))
    if (distance-radius < policy['minimum_reference_distance_m']-1e-8
            or distance+radius > policy['maximum_reference_distance_m']+1e-8):
        failures.append('whole_position_region_distance_bound_failed')
    if region['max_heading_error_rad'] != policy['heading_tolerance_rad']:
        failures.append('heading_tolerance_mismatch')
    local = np.array([[-vehicle['rear_overhang_m'], -vehicle['width_m']/2],
        [vehicle['front_from_rear_axle_m'], -vehicle['width_m']/2],
        [vehicle['front_from_rear_axle_m'], vehicle['width_m']/2],
        [-vehicle['rear_overhang_m'], vehicle['width_m']/2]])
    def body(xy, heading):
        c, s = math.cos(heading), math.sin(heading)
        return Polygon(local@np.array([[c, s], [-s, c]])+xy)
    corner_radius = float(np.linalg.norm(local, axis=1).max())
    conservative_radius = (radius+2*corner_radius*math.sin(policy['heading_tolerance_rad']/2)
                           +policy['body_clearance_m'])
    # 独立重建解析半径外包络；微米容差只用于序列化浮点，不扩大合法区域。
    independent_envelope = body(nominal, yaw).buffer(
        conservative_radius/math.cos(math.pi/(4*policy['buffer_quad_segs'])),
        quad_segs=policy['buffer_quad_segs'])
    if not envelope.buffer(1e-7).covers(independent_envelope):
        failures.append('saved_envelope_does_not_cover_analytic_bound')
    # 再检查9个平移点×11个航向，覆盖名义姿态以外的真实车身位置。
    c, s = math.cos(yaw), math.sin(yaw)
    rotations = np.array([[c, s], [-s, c]])
    positions = [np.array([a*policy['position_length_m']/2, b*policy['position_width_m']/2])@rotations+nominal
                 for a, b in itertools.product([-1., 0., 1.], repeat=2)]
    checked = 0
    for xy in positions:
        if not region_polygon.buffer(1e-7).covers(Point(xy)):
            failures.append('position_rectangle_mismatch')
        for heading in np.linspace(yaw-policy['heading_tolerance_rad'], yaw+policy['heading_tolerance_rad'], 11):
            rectangle = body(xy, float(heading))
            if (not envelope.covers(rectangle) or not road.covers(rectangle)
                    or not lanes.covers(rectangle) or rectangle.intersects(obstacles)):
                failures.append('independent_perturbed_body_failed')
            checked += 1
    if region.get('reachability') != 'unknown':
        failures.append('endpoint_certificate_claims_reachability')
    return dict(passed=not failures, failures=sorted(set(failures)), checked_pose_samples=checked,
        reference_point_distance_m=distance, position_radius_m=radius,
        conservative_extra_radius_m=conservative_radius)


def raw_box_polygon(record):
    """与生成器分开写的四元数转矩阵和八角点投影。"""
    quat = np.asarray(record['rotation_wxyz'], dtype=float)
    quat = quat/np.linalg.norm(quat)
    w, xyz = quat[0], quat[1:]
    cross = np.array([[0., -xyz[2], xyz[1]], [xyz[2], 0., -xyz[0]], [-xyz[1], xyz[0], 0.]])
    rotation = np.eye(3)+2*w*cross+2*cross@cross
    size = np.asarray(record['size_wlh'])[[1, 0, 2]]
    local = np.asarray(list(itertools.product([-.5, .5], repeat=3)))*size
    return MultiPoint((local@rotation.T+record['center'])[:, :2]).convex_hull


def independent_map(data):
    nodes = {n['token']: (n['x'], n['y']) for n in data['node']}
    polygons = {}
    for p in data['polygon']:
        polygons[p['token']] = Polygon([nodes[t] for t in p['exterior_node_tokens']],
            [[nodes[t] for t in h['node_tokens']] for h in p.get('holes', [])])
    valid_road_tokens = {t for r in data['drivable_area'] for t in r['polygon_tokens']
                         if isinstance(t, str) and t in polygons}
    lanes = {r['token']: polygons[r['polygon_token']] for kind in ['lane', 'lane_connector']
             for r in data[kind] if r['polygon_token'] in polygons}
    return dict(road=unary_union([polygons[t] for t in sorted(valid_road_tokens)]), lanes=lanes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--audit-id', default='audit_001')
    args = parser.parse_args()
    started = time.monotonic()
    source = OUTPUT/'annotation'/args.run_id
    summary = json.loads((source/'summary.json').read_text())
    release = json.loads((source/'release.json').read_text())
    dest = source/args.audit_id
    dest.mkdir(exist_ok=False)
    shutil.copy2(__file__, dest/'auditor_source.py')
    for name, sha in summary['files_sha256'].items():
        if digest(source/name) != sha:
            raise ValueError(f'changed label artifact: {name}')
    for group in ['input_sha256', 'source_sha256']:
        for name, sha in release[group].items():
            if digest(name) != sha:
                raise ValueError(f'changed label source: {name}')
    maps = {}
    for location, sha in release['map_sha256'].items():
        path = PROJECT/'dataset/nuscenes_map_expansion/expansion'/f'{location}.json'
        if digest(path) != sha:
            raise ValueError('changed official HD map')
        maps[sha] = independent_map(json.loads(path.read_text()))
    contexts, records, checks, objects = {}, [], [], 0
    for row in map(json.loads, (source/'targets.jsonl').open()):
        name = row['context']['path']
        if name not in contexts:
            context = json.loads((source/name).read_text())
            if digest(source/name) != row['context']['sha256']:
                raise ValueError('context fingerprint differs')
            official_map = maps[context['prior_sha256']]
            window = shape(context['window_geojson'])
            expected_road = official_map['road'].intersection(window)
            saved_road = shape(context['drivable_geojson'])
            if saved_road.symmetric_difference(expected_road).area > 1e-7:
                raise ValueError('context road differs from original official polygon/holes')
            saved_lanes = []
            for lane in context['lanes']:
                saved = shape(lane['polygon_geojson'])
                if (lane['token'] not in official_map['lanes']
                        or saved.symmetric_difference(official_map['lanes'][lane['token']]).area > 1e-7):
                    raise ValueError('context contains an unofficial lane polygon')
                saved_lanes.append(saved)
            if (shape(context['lane_union_geojson']).symmetric_difference(
                    unary_union(saved_lanes).intersection(window)).area > 1e-7):
                raise ValueError('context lane union changes official lane polygons')
            for official in context['objects']:
                projected = raw_box_polygon(official)
                saved = shape(official['polygon_geojson'])
                if projected.hausdorff_distance(saved) > 1e-8:
                    raise ValueError('official box projection mismatch')
                objects += 1
            contexts[name] = context
        context = contexts[name]
        target = raw_box_polygon(row['official_annotation'])
        for j, region in enumerate(row['regions']):
            result = audit_region(region, target, context, release['policy'], release['vehicle'])
            checks.append(dict(scene=row['scene'], frame_idx=row['frame_idx'],
                instance_token=row['instance_token'], region_index=j, **result))
        records.append(row)
    # 用原始完整annotation表核对上下文，排除只保存目标或只保存静态盒的遗漏。
    by_sample = {context['sample_token']: context for context in contexts.values()}
    expected_boxes = {token: set() for token in by_sample}
    raw_table = next(name for name in release['input_sha256'] if name.endswith('/sample_annotation.json'))
    for annotation in iter_records(Path(raw_table)):
        context = by_sample.get(annotation['sample_token'])
        if context is None:
            continue
        projected = raw_box_polygon(dict(center=annotation['translation'], size_wlh=annotation['size'],
                                         rotation_wxyz=annotation['rotation']))
        if projected.intersects(shape(context['window_geojson'])):
            expected_boxes[annotation['sample_token']].add(annotation['token'])
    for sample, context in by_sample.items():
        if expected_boxes[sample] != {o['annotation_token'] for o in context['objects']}:
            raise ValueError('a current official box was omitted or substituted')
    queries = [json.loads(line) for line in (source/'queries.jsonl').open()]
    original = OUTPUT/'annotation'/release['input_annotation_run']
    original_queries = [q for name in ['main_queries.jsonl', 'challenge_queries.jsonl']
                        for q in map(json.loads, (original/name).open())]
    scenes = sorted({q['scene'] for q in original_queries})
    if release['scene_limit']:
        scenes = scenes[:release['scene_limit']]
    original_queries = [q for q in original_queries if q['scene'] in scenes]
    if release['query_limit']:
        original_queries = original_queries[:release['query_limit']]
    if [q['query_id'] for q in queries] != [q['query_id'] for q in original_queries]:
        raise ValueError('reference labels dropped, duplicated or reordered original queries')
    targets = {(r['scene'], r['frame_idx'], r['instance_token']): r for r in records}
    for query, original_query in zip(queries, original_queries):
        if any(query[field] != original_query[field] for field in
               ['accepted_instance_tokens', 'scene', 'frame_idx', 'cohort', 'no_matching_target']):
            raise ValueError('reference labels changed independently accepted official targets')
        expected = [region for token in query['accepted_instance_tokens']
                    for region in targets[(query['scene'], query['frame_idx'], token)]['regions']]
        if query['safe_regions'] != expected:
            raise ValueError('query regions differ from accepted official target regions')
    preserved = (all(q['reachable_label'] is None and q['reachability'] == 'unknown'
                     and q['actual_arrival'] is None and not q['new_test_opened'] for q in queries)
                 and all(r['reachability'] == 'unknown' and r['actual_arrival'] is None for r in records))
    with (dest/'region_checks.jsonl').open('x') as handle:
        for check in checks:
            handle.write(json.dumps(check, ensure_ascii=False, allow_nan=False)+'\n')
    report = dict(passed=all(c['passed'] for c in checks) and preserved,
        queries=len(queries), target_records=len(records), certified_regions=len(checks),
        contexts=len(contexts), official_boxes_projection_checked=objects,
        perturbed_body_pose_checks=sum(c['checked_pose_samples'] for c in checks),
        failed_regions=sum(not c['passed'] for c in checks),
        failure_reasons=dict(Counter(reason for c in checks for reason in c['failures'])),
        unknown_reachability_and_no_execution_preserved=preserved,
        all_current_official_boxes_present=True, query_to_target_region_mapping_passed=True,
        original_query_population_unchanged=True, official_map_polygons_and_holes_verified=True,
        all_input_and_artifact_fingerprints_passed=True,
        auditor_source_sha256=digest(__file__), generator_summary_sha256=digest(source/'summary.json'),
        checks_sha256=digest(dest/'region_checks.jsonl'), elapsed_s=time.monotonic()-started,
        independent_method_evaluation=False, correct_arrival_credit=False, goal_complete=False)
    dump(dest/'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report['passed']:
        raise AssertionError('reference region certificate audit failed; artifacts retained')


if __name__ == '__main__':
    main()
