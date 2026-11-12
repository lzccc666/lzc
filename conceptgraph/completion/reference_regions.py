"""隔离评价端生成官方对象附近的保守整车静态区域；不判定路线可达性。"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Point, Polygon, box, mapping
from shapely.ops import unary_union

from conceptgraph.completion.map_constraint_ablation import polygon_with_holes
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump
from conceptgraph.navigation.vehicle_geometry import VehicleSpec, VehicleState, footprint_polygon
from conceptgraph.nuscenes.expansion_benchmark import iter_records
from conceptgraph.scripts.build_lane_constrained_bev import _lane_records


@dataclass(frozen=True)
class RegionPolicy:
    # 坐标均为世界米；区域位置是后轴中心，不是相机或对象中心。
    position_length_m: float = .4
    position_width_m: float = .2
    heading_tolerance_rad: float = math.radians(5.)
    body_clearance_m: float = .1
    minimum_reference_distance_m: float = 2.
    maximum_reference_distance_m: float = 8.
    map_window_radius_m: float = 100.
    nearest_centerline_samples: int = 256
    maximum_candidates_per_target: int = 256
    maximum_regions_per_target: int = 4
    region_separation_m: float = 2.
    lateral_offsets_m: tuple = (0., -.5, .5)
    buffer_quad_segs: int = 16


def reference_prior_from_json(data):
    """缺失引用不提供可行驶区域；保留所有有效官方多边形和孔洞。"""
    nodes = {n['token']: n for n in data['node']}
    polygons = {p['token']: p for p in data['polygon']}
    missing, tokens = [], set()
    for record in data['drivable_area']:
        for token in record['polygon_tokens']:
            if not isinstance(token, str) or token not in polygons:
                missing.append(dict(record_token=record['token'], missing_polygon_token=token))
            else:
                tokens.add(token)
    road = unary_union([polygon_with_holes(polygons[t], nodes) for t in sorted(tokens)])
    xy = np.array([[n['x'], n['y']] for n in nodes.values()])
    lanes, diagnostics = _lane_records(data, xy.min(axis=0)-1, xy.max(axis=0)+1, .5)
    return dict(lanes=lanes, drivable=road, diagnostics=dict(**diagnostics,
        missing_drivable_polygon_references=missing, missing_geometry_not_filled=True),
        polygon_holes=sum(len(polygons[t].get('holes', [])) for t in tokens))


def load_reference_prior(location):
    path = PROJECT/'dataset/nuscenes_map_expansion/expansion'/f'{location}.json'
    return dict(**reference_prior_from_json(json.loads(path.read_text())), path=str(path), sha256=digest(path))


def object_polygon(record):
    """投影完整3D盒八角点，保留roll/pitch；不以仅yaw的盒缩小障碍。"""
    center = np.asarray(record.get('center', record.get('translation')), dtype=float)
    size = np.asarray(record.get('size_wlh', record.get('size')), dtype=float)
    quat = np.asarray(record.get('rotation_wxyz', record.get('rotation')), dtype=float)
    if (center.shape != (3,) or size.shape != (3,) or quat.shape != (4,)
            or not np.isfinite(np.r_[center, size, quat]).all() or np.any(size <= 0)
            or abs(np.linalg.norm(quat)-1.) > 1e-5):
        raise ValueError('invalid official box; cannot omit it from safety evidence')
    w, x, y, z = quat/np.linalg.norm(quat)
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    corners = np.array(list(itertools.product([-1., 1.], repeat=3))) * size[[1, 0, 2]]/2
    polygon = MultiPoint((corners@rotation.T+center)[:, :2]).convex_hull
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError('invalid official footprint')
    return polygon


def position_region(x, y, yaw, policy=RegionPolicy()):
    local = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    local *= [policy.position_length_m/2, policy.position_width_m/2]
    c, s = math.cos(yaw), math.sin(yaw)
    return Polygon(local@np.array([[c, s], [-s, c]])+[x, y])


def conservative_envelope(x, y, yaw, policy=RegionPolicy(), spec=VehicleSpec()):
    """包住区域内任意平移及±航向容差的车身，另加固定安全间距。"""
    translation_radius = math.hypot(policy.position_length_m/2, policy.position_width_m/2)
    corner_radius = math.hypot(max(spec.front_from_rear_axle_m, spec.rear_overhang_m), spec.width_m/2)
    rotational_displacement = 2*corner_radius*math.sin(policy.heading_tolerance_rad/2)
    radius = translation_radius+rotational_displacement+policy.body_clearance_m
    # GEOS圆弧是内接折线；补偿弦的内缩，保证这里用的是外包络。
    outward_radius = radius/math.cos(math.pi/(4*policy.buffer_quad_segs))
    return footprint_polygon(VehicleState(x, y, yaw), spec).buffer(
        outward_radius, quad_segs=policy.buffer_quad_segs)


def check_region(x, y, yaw, target, road, lanes, obstacles, policy=RegionPolicy()):
    region = position_region(x, y, yaw, policy)
    radius = math.hypot(policy.position_length_m/2, policy.position_width_m/2)
    distance = target.distance(Point(x, y))
    if (distance-radius < policy.minimum_reference_distance_m
            or distance+radius > policy.maximum_reference_distance_m):
        return None, 'rear_axle_distance_outside_declared_annulus'
    envelope = conservative_envelope(x, y, yaw, policy)
    if not road.covers(envelope):
        return None, 'envelope_outside_official_drivable_area'
    if not lanes.covers(envelope):
        return None, 'envelope_outside_official_lane_area'
    if envelope.intersects(obstacles):
        return None, 'envelope_intersects_current_official_box'
    # GEOS在极细的官方车道拼接边界可能出现covers谓词不满足传递性的数值情形。
    # 不扩大或修补地图：任一实际扰动矩形谓词不通过就保守拒绝该候选。
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, s], [-s, c]])
    for a, b in itertools.product([-1., 0., 1.], repeat=2):
        xy = np.array([a*policy.position_length_m/2, b*policy.position_width_m/2])@rotation+[x, y]
        for heading in np.linspace(yaw-policy.heading_tolerance_rad, yaw+policy.heading_tolerance_rad, 11):
            body = footprint_polygon(VehicleState(float(xy[0]), float(xy[1]), float(heading)))
            if (not envelope.covers(body) or not road.covers(body) or not lanes.covers(body)
                    or body.intersects(obstacles)):
                return None, 'explicit_perturbed_body_predicate_failed_conservatively'
    return dict(polygon_xy=list(region.exterior.coords), heading_rad=yaw,
        max_heading_error_rad=policy.heading_tolerance_rad,
        nominal_rear_axle_xy=[x, y], envelope_geojson=mapping(envelope),
        reference_point_target_box_distance_m=distance,
        guaranteed_minimum_reference_distance_m=distance-radius,
        guaranteed_maximum_reference_distance_m=distance+radius,
        explicit_perturbed_pose_checks=99, robust_static_body_certificate=True, reachability='unknown'), None


def prior_index(prior):
    samples = np.concatenate([lane['centerline'] for lane in prior['lanes']])
    lane_ids = np.concatenate([np.full(len(lane['centerline']), i, dtype=int)
                               for i, lane in enumerate(prior['lanes'])])
    return dict(tree=cKDTree(samples[:, :2]), samples=samples, lane_ids=lane_ids)


def make_context(prior, frame, annotations, policy):
    xy = np.asarray(frame['camera_pose'])[:2, 3]
    window = box(*(xy-policy.map_window_radius_m), *(xy+policy.map_window_radius_m))
    lanes = []
    for lane in prior['lanes']:
        polygon = Polygon(lane['polygon'])
        if polygon.intersects(window):
            lanes.append(dict(token=lane['token'], polygon_geojson=mapping(polygon)))
    objects = []
    for annotation in annotations:
        polygon = object_polygon(annotation)
        if polygon.intersects(window):
            objects.append(dict(annotation_token=annotation['token'],
                instance_token=annotation['instance_token'], polygon_geojson=mapping(polygon),
                center=annotation['translation'], size_wlh=annotation['size'],
                rotation_wxyz=annotation['rotation'], official_attribute_tokens=annotation['attribute_tokens']))
    road = prior['drivable'].intersection(window)
    lane_union = unary_union([Polygon(lane['polygon']) for lane in prior['lanes']
                              if Polygon(lane['polygon']).intersects(window)]).intersection(window)
    payload = dict(scene=frame['scene'], frame_idx=frame['frame_idx'], sample_token=frame['sample_token'],
        camera_timestamp_s=frame['camera_timestamp_s'], official_annotation_timestamp_s=frame['sample_timestamp_s'],
        annotation_minus_camera_s=frame['sample_timestamp_s']-frame['camera_timestamp_s'],
        window_geojson=mapping(window), drivable_geojson=mapping(road), lane_union_geojson=mapping(lane_union),
        lanes=lanes, objects=objects, prior_sha256=prior['sha256'],
        prior_diagnostics=prior['diagnostics'],
        all_official_categories_included=True, all_motion_states_included=True,
        temporal_scope='官方sample时刻的冻结盒和地图；不保证相机时刻精确同步或未来交通安全',
        inference_may_access=False, future_observations_used_for_method=False)
    return payload, road, lane_union, unary_union([object_polygon(a) for a in annotations])


def target_candidates(target, prior, index, policy):
    center = np.asarray(target['center'])[:2]
    _, neighbors = index['tree'].query(center, k=min(policy.nearest_centerline_samples, len(index['samples'])))
    polygon = object_polygon(target)
    candidates = {}
    for neighbor in np.atleast_1d(neighbors):
        lane = prior['lanes'][int(index['lane_ids'][neighbor])]
        x, y, yaw = map(float, index['samples'][neighbor])
        for lateral in policy.lateral_offsets_m:
            a, b = x-lateral*math.sin(yaw), y+lateral*math.cos(yaw)
            if not Polygon(lane['polygon']).covers(Point(a, b)):
                continue
            distance = polygon.distance(Point(a, b))
            if not policy.minimum_reference_distance_m <= distance <= policy.maximum_reference_distance_m:
                continue
            key = tuple(round(v, 5) for v in (a, b, yaw))
            value = dict(x=a, y=b, yaw=yaw, lane_token=lane['token'], lateral_offset_m=lateral)
            candidates.setdefault(key, value)
    # 事前固定几何顺序；不存在检测、候选排序、模型、待评测规划器或方法结果输入。
    middle = (policy.minimum_reference_distance_m+policy.maximum_reference_distance_m)/2
    return sorted(candidates.values(), key=lambda r: (
        abs(polygon.distance(Point(r['x'], r['y']))-middle), r['lane_token'], r['x'], r['y'], r['yaw']
    ))[:policy.maximum_candidates_per_target]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--annotation-run', default='reserved_official_annotation_001')
    parser.add_argument('--scene-limit', type=int, default=0)
    parser.add_argument('--query-limit', type=int, default=0)
    parser.add_argument('--wall-budget-s', type=float, default=3600.)
    args = parser.parse_args()
    started = time.monotonic()
    policy = RegionPolicy()
    source = OUTPUT/'annotation'/args.annotation_run
    all_queries = [json.loads(line) for name in ['main_queries.jsonl', 'challenge_queries.jsonl']
                   for line in (source/name).open()]
    scenes = sorted({q['scene'] for q in all_queries})
    if args.scene_limit:
        scenes = scenes[:args.scene_limit]
    queries = [q for q in all_queries if q['scene'] in scenes]
    if args.query_limit:
        queries = queries[:args.query_limit]
    wanted = {(q['scene'], q['frame_idx']) for q in queries if q['accepted_instance_tokens']}
    frames = {(f['scene'], f['frame_idx']): f for f in map(json.loads,
        (source/'evaluation_only_frames.jsonl').open()) if (f['scene'], f['frame_idx']) in wanted}
    if set(frames) != wanted:
        raise ValueError('missing independently annotated positive frame')
    dest = OUTPUT/'annotation'/args.run_id
    dest.mkdir(parents=True, exist_ok=False)
    (dest/'contexts').mkdir()
    meta = PROJECT/'dataset/nuscenes_trainval_raw/v1.0-trainval'
    locations = {r['scene']: r['strata']['location'] for r in json.loads(
        (OUTPUT/'splits/reserved_test_scenes_v2.json').read_text())['scenes']}
    dependencies = [Path(__file__), PROJECT/'conceptgraph/completion/map_constraint_ablation.py',
        PROJECT/'conceptgraph/navigation/vehicle_geometry.py', PROJECT/'conceptgraph/scripts/build_lane_constrained_bev.py']
    fingerprints = {str(p): digest(p) for p in dependencies}
    dump(dest/'release.json', dict(version='official-static-region-certificate-v3', policy=asdict(policy),
        vehicle=asdict(VehicleSpec()), input_annotation_run=args.annotation_run,
        input_sha256={str(p): digest(p) for p in [source/'evaluation_only_frames.jsonl',
            source/'main_queries.jsonl', source/'challenge_queries.jsonl',
            meta/'sample_annotation.json', meta/'sample.json', OUTPUT/'splits/reserved_test_scenes_v2.json']},
        source_sha256=fingerprints,
        map_sha256={loc: digest(PROJECT/'dataset/nuscenes_map_expansion/expansion'/f'{loc}.json')
                    for loc in sorted({locations[s] for s in scenes})},
        queries=len(queries), frames=len(frames), scene_limit=args.scene_limit, query_limit=args.query_limit,
        wall_budget_s=args.wall_budget_s, storage_budget_bytes=500000000, minimum_free_bytes=6000000000,
        method_inference_executed=False, gpu_model_calls=0, new_test_opened=False,
        reachability_evaluated=False, no_region_does_not_mean_inaccessible=True,
        independent_label_preparation_only=True, goal_complete=False))
    shutil.copy2(__file__, dest/'generator_source.py')
    samples = {f['sample_token']: [] for f in frames.values()}
    timestamps = {r['token']: r['timestamp']/1e6 for r in iter_records(meta/'sample.json') if r['token'] in samples}
    for annotation in iter_records(meta/'sample_annotation.json'):
        if annotation['sample_token'] in samples:
            samples[annotation['sample_token']].append(annotation)
    priors, indices, context_refs, target_rows = {}, {}, {}, {}
    with (dest/'targets.jsonl').open('x') as target_handle:
        for key, frame in sorted(frames.items()):
            if time.monotonic()-started > args.wall_budget_s:
                raise TimeoutError('reference label wall budget; partial run retained')
            if shutil.disk_usage(dest).free < 6000000000:
                raise OSError('disk reserve; partial run retained')
            location = locations[frame['scene']]
            if location not in priors:
                priors[location] = load_reference_prior(location)
                indices[location] = prior_index(priors[location])
            frame['sample_timestamp_s'] = timestamps[frame['sample_token']]
            context, road, lanes, obstacles = make_context(priors[location], frame, samples[frame['sample_token']], policy)
            filename = f'contexts/{frame["scene"]}_{frame["frame_idx"]:04d}.json'
            dump(dest/filename, context)
            if sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()) > 500000000:
                raise OSError('reference label storage budget; partial run retained')
            context_refs[key] = dict(path=filename, sha256=digest(dest/filename))
            accepted = sorted({t for q in queries if (q['scene'], q['frame_idx']) == key
                               for t in q['accepted_instance_tokens']})
            for token in accepted:
                target = next(o for o in frame['objects'] if o['instance_token'] == token)
                if target['official_motion'] != 'static':
                    raise ValueError('a non-static target entered the static-task truth')
                official = next(a for a in samples[frame['sample_token']] if a['instance_token'] == token)
                if not object_polygon(target).equals_exact(object_polygon(official), 1e-8):
                    raise ValueError('prepared target box differs from raw official annotation')
                pool = target_candidates(target, priors[location], indices[location], policy)
                regions, attempts, reasons = [], [], Counter()
                for candidate in pool:
                    if any(math.hypot(candidate['x']-r['nominal_rear_axle_xy'][0],
                                      candidate['y']-r['nominal_rear_axle_xy'][1]) < policy.region_separation_m
                           for r in regions):
                        continue
                    region, failure = check_region(candidate['x'], candidate['y'], candidate['yaw'],
                        object_polygon(target), road, lanes, obstacles, policy)
                    attempts.append(dict(**candidate, rejection=failure, certified=region is not None))
                    reasons[failure or 'certified'] += 1
                    if region is not None:
                        regions.append(dict(**region, instance_id=token, lane_token=candidate['lane_token']))
                    if len(regions) >= policy.maximum_regions_per_target:
                        break
                record = dict(scene=frame['scene'], frame_idx=frame['frame_idx'], sample_token=frame['sample_token'],
                    instance_token=token, category=target['category'], official_annotation=target,
                    context=context_refs[key], regions=regions, candidate_pool_size=len(pool),
                    evaluated_candidates=attempts, candidate_reasons=dict(reasons),
                    endpoint_status='certified_static_regions' if regions else 'unknown_no_certificate',
                    reachability='unknown', actual_arrival=None, actual_route_generated=None,
                    temporal_scope=context['temporal_scope'], human_verified=False,
                    label_source='isolated_official_boxes_and_declared_HD_map_only')
                target_rows[(key, token)] = record
                target_handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                target_handle.flush()
            print(json.dumps(dict(frame=f'{key[0]}:{key[1]}', completed_frames=len(context_refs),
                total_frames=len(frames), completed_target_records=len(target_rows)), ensure_ascii=False), flush=True)
    query_rows = []
    for query in queries:
        key = (query['scene'], query['frame_idx'])
        records = [target_rows[(key, t)] for t in query['accepted_instance_tokens']]
        regions = [r for record in records for r in record['regions']]
        query_rows.append(dict(query_id=query['query_id'], scene=query['scene'], frame_idx=query['frame_idx'],
            cohort=query['cohort'], accepted_instance_tokens=query['accepted_instance_tokens'],
            no_matching_target=query['no_matching_target'], safe_regions=regions,
            endpoint_status=('not_applicable_no_matching_target' if not records else
                             'certified_static_regions' if regions else 'unknown_no_certificate'),
            reachability='unknown', reachable_label=None, actual_arrival=None,
            source_query_unchanged=True, human_verified=False, method_inference_executed=False,
            new_test_opened=False))
    with (dest/'queries.jsonl').open('x') as handle:
        for record in query_rows:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
    if any(digest(name) != sha for name, sha in fingerprints.items()):
        raise ValueError('generator dependencies changed during label preparation')
    files = {str(p.relative_to(dest)): digest(p) for p in dest.rglob('*') if p.is_file()}
    dump(dest/'summary.json', dict(queries=len(query_rows), positive_queries=sum(not q['no_matching_target'] for q in query_rows),
        query_endpoint_status=dict(Counter(q['endpoint_status'] for q in query_rows)),
        scenes=len({q['scene'] for q in query_rows}), frames=len(frames), target_records=len(target_rows),
        targets_with_certificates=sum(bool(r['regions']) for r in target_rows.values()),
        certified_target_regions=sum(len(r['regions']) for r in target_rows.values()),
        current_official_boxes=sum(len(c['objects']) for c in [json.loads((dest/r['path']).read_text()) for r in context_refs.values()]),
        elapsed_s=time.monotonic()-started, storage_bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),
        files_sha256=files, independently_audited=False, reachability_unknown_queries=len(query_rows),
        actual_vehicle_executions=0, actual_arrivals=None, model_calls=0, new_test_opened=False, goal_complete=False,
        limitations=['仅产生冻结官方sample时刻的静态安全终点区域，不是拓扑/运动学可达性或控制执行证据',
            '完整3D盒投影包含所有官方类别、所有运动状态；未知和动态对象不被忽略，但不预测它们未来运动',
            '地图保留drivable孔洞，未知/缺失arcline车道不产生候选；候选有限采样，无证书保持unknown',
            '保守包络覆盖0.4m×0.2m后轴位置区域与±5度姿态误差并另加0.1m；不得用默认15度终点容差替换',
            '目标参考点距目标完整投影盒2—8m；车身和全部已标注盒另作碰撞检查，不是对象中心距离',
            'GT仅存在annotation评价端；没有检测、语义、待测规划、训练、调参或新测试方法调用']))
    print(json.dumps({k: v for k, v in json.loads((dest/'summary.json').read_text()).items()
                      if k not in ['files_sha256', 'limitations']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
