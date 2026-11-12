"""完整过去LiDAR六地图记录的独立几何/计数/区间复核；不重复执行规划。"""
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import itertools
import json
import math
import time

import numpy as np
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import unary_union

ROOT = Path('/root/autodl-tmp/lizichong')
OUT = ROOT/'outputs/project_v2/thesis_completion_v1'
SOURCE = OUT/'ablations/causal_maps_full_001'
DEST = SOURCE/'audit_001'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def ratio(counts):
    return np.divide(counts[..., 0], counts[..., 1],
        out=np.full(counts.shape[:-1], np.nan), where=counts[..., 1] > 0)


def box_at(xy, yaw):
    # 独立显式后轴坐标计算；不调用生产footprint/VehicleSpec函数。
    c, s = math.cos(yaw), math.sin(yaw)
    return Polygon([(xy[0]+u*c-v*s, xy[1]+u*s+v*c)
        for u, v in [(-.9, -.925), (3.7, -.925), (3.7, .925), (-.9, .925)]])


def main():
    begun = time.monotonic()
    DEST.mkdir(exist_ok=False)
    release, summary = read(SOURCE/'release.json'), read(SOURCE/'summary.json')
    statistics = read(OUT/'statistics/causal_maps_full_001/report.json')
    assert sha(SOURCE/'cases.jsonl') == summary['records_sha256']
    rows = [json.loads(line) for line in (SOURCE/'cases.jsonl').open()]
    methods = release['conditions']
    expected = {(q['scene'], q['frame_idx'], q['query_index'], m)
        for q in release['selected_queries'] for m in methods}
    key = lambda r: (r['scene'], r['frame_idx'], r['query_index'], r['condition'])
    assert len(rows) == len(expected) == 1272 and {key(r) for r in rows} == expected
    lookup = {key(r): r for r in rows}
    sources = {SOURCE/'release.json': sha(SOURCE/'release.json'),
        SOURCE/'summary.json': sha(SOURCE/'summary.json'),
        SOURCE/'cases.jsonl': sha(SOURCE/'cases.jsonl'),
        OUT/'statistics/causal_maps_full_001/report.json': sha(OUT/'statistics/causal_maps_full_001/report.json')}
    for k, v in release['source_frames_sha256'].items():
        sources[Path(k)] = v
    for k, v in release['source_sha256'].items():
        sources[ROOT/k] = v
    for k, v in release['map_sha256'].items():
        sources[ROOT/'dataset/nuscenes_map_expansion/expansion'/f'{k}.json'] = v
    for k, v in release['metadata_sha256'].items():
        sources[ROOT/'dataset/nuscenes_trainval_raw/v1.0-trainval'/k] = v
    contexts = {}
    geometry_cache = {}
    geometry_routes = 0
    geometry_poses = 0
    for row in rows:
        baseline = lookup[(*key(row)[:3], methods[0])]
        for f in ['common_start', 'common_goal', 'target_instance', 'context']:
            assert row[f] == baseline[f]
        assert row['log_token'] == release['scene_logs'][row['scene']]
        assert not row['vehicle_executed'] and not row['actual_arrival_evaluated']
        e = row['evaluation']
        assert e['actual_arrival'] is None and e['selected_correct_object'] is None and e['stable_correct_lock'] is None
        context_path = Path(row['context']['path'])
        if context_path not in contexts:
            c = read(context_path/'context.json')
            rasters = np.load(context_path/'rasters.npz')
            contexts[context_path] = dict(c=c, origin=rasters['origin_xy'].copy(),
                resolution=float(rasters['resolution']), drivable=shape(c['drivable_geojson']),
                lanes=unary_union([Polygon(g['polygon']) for g in c['lane_geometries'].values()]),
                obstacles=[shape(g) for g in c['obstacles_geojson']])
            sources[context_path/'context.json'] = row['context']['context_sha256']
            sources[context_path/'rasters.npz'] = row['context']['rasters_sha256']
            assert c['obstacle_input_source'] == 'current predictions.one_to_one_depth_quality'
        path = row['planner']['path']
        assert e['path_generated'] == (path is not None)
        if path is None:
            assert e['full_lane_legal'] is None and e['body_pose_legal'] is None
            continue
        assert path[0] == row['common_start']['rc'] and path[-1] == row['common_goal']['rc']
        geometry_routes += 1
        cache_key = (context_path, tuple(map(tuple, path)))
        if cache_key not in geometry_cache:
            c = contexts[context_path]
            xy = np.array(path)[:, ::-1]*c['resolution']+c['origin']
            point = LineString(xy) if len(xy)>1 else Point(xy[0])
            pose = np.asarray(c['c']['camera_pose'])
            initial = math.atan2(pose[1, 2], pose[0, 2])
            def reasons_at(p, yaw):
                body = box_at(p, yaw)
                return [name for name, fail in [
                    ('rectangle_outside_drivable', not c['drivable'].covers(body)),
                    ('rectangle_outside_lane_union', not c['lanes'].covers(body)),
                    ('rectangle_intersects_observed_obstacle', any(body.intersects(o) for o in c['obstacles']))] if fail]
            initial_legal = not reasons_at(xy[0], initial)
            violations, yaws = [], []
            for i, p in enumerate(xy):
                delta = xy[min(i+1, len(xy)-1)]-xy[max(0, i-1)]
                yaw = math.atan2(delta[1], delta[0]) if np.linalg.norm(delta)>1e-9 else initial
                yaws.append(yaw)
                reasons = reasons_at(p, yaw)
                if reasons:
                    violations.append(dict(index=i, reasons=reasons))
            yaw_delta = np.abs((np.diff(yaws)+math.pi)%(2*math.pi)-math.pi)
            distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            curvature = np.divide(yaw_delta, distances, out=np.zeros_like(distances), where=distances>1e-9)
            geometry_cache[cache_key] = dict(continuous_point_drivable=bool(c['drivable'].covers(point)),
                initial_body_pose_legal=initial_legal, body_pose_legal=not violations and initial_legal,
                body_pose_violations=violations, curvature_exceeds_vehicle_bound=bool(np.any(curvature>math.tan(math.radians(32))/2.7)),
                waypoint_tangent_curvature_max_per_m=float(curvature.max()) if len(curvature) else None)
            geometry_poses += len(xy)+1
        for f, value in geometry_cache[cache_key].items():
            assert e[f] == value, (key(row), f, e[f], value)
        # 车道方向/连接计数来自既有独立连续审计器；此处不称第二套独立车道实现。
        assert e['full_lane_legal'] == e['independent_lane_audit']['valid']
    metrics = list(statistics['conditions'][methods[0]])
    logs = sorted(set(release['scene_logs'].values()))
    counts = np.zeros((len(logs),len(methods),len(metrics),2),dtype=int)
    for row in rows:
        e = row['evaluation']; lane = e.get('independent_lane_audit', {})
        route = bool(e['path_generated']); legal = e['full_lane_legal'] is True
        road = legal and e.get('continuous_point_drivable') is True
        values = [route, legal, road, legal, road, e.get('continuous_point_drivable') is True,
            bool(lane.get('direction_violation_count')), bool(lane.get('illegal_transitions')),
            bool(lane.get('reverse_count')), bool(lane.get('invalid_cells')), e['body_pose_legal'] is True,
            e.get('curvature_exceeds_vehicle_bound') is True,
            row['planner']['failure'] in ['PLANNING_WALL_BUDGET','PLANNING_EXPANSION_BUDGET'],
            row['planner']['failure']=='COMMON_ENDPOINT_PROJECTION_FAILED']
        denominators = [True]*3+[route]*9+[True]*2
        assert len(values) == len(metrics)
        counts[logs.index(row['log_token']),methods.index(row['condition'])] += np.array([values, denominators],dtype=int).T
    point = ratio(counts.sum(axis=0))
    random = np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)))
    boot = ratio(counts[random].sum(axis=1))
    def check(value, target):
        if target is None:
            assert not np.isfinite(value)
        else:
            assert np.allclose(value,target,rtol=0,atol=1e-12)
    for j,m in enumerate(methods):
        for k,f in enumerate(metrics):
            result = statistics['conditions'][m][f]
            assert counts[:,j,k].sum(axis=0).tolist()==[result['count'],result['denominator']]
            check(point[j,k], result['estimate'])
            valid = boot[:,j,k][np.isfinite(boot[:,j,k])]
            check(np.quantile(valid,[.025,.975]) if len(valid) else np.nan,result['ci95'])
            assert len(valid)==result['valid_resamples']
    for comparison in statistics['paired_differences']:
        a,b,k=methods.index(comparison['a']),methods.index(comparison['b']),metrics.index(comparison['metric'])
        check(point[a,k]-point[b,k],comparison['difference'])
        valid=boot[:,a,k]-boot[:,b,k];valid=valid[np.isfinite(valid)]
        check(np.quantile(valid,[.025,.975]) if len(valid) else np.nan,comparison['ci95'])
        assert len(valid)==comparison['valid_resamples']
    assert len(statistics['paired_differences'])==len(list(itertools.combinations(methods,2)))*len(metrics)
    for p,digest in sources.items():
        assert sha(p)==digest, str(p)
    with (DEST/'counts_by_log.csv').open('x') as handle:
        writer=csv.writer(handle);writer.writerow(['log','method','metric','numerator','denominator'])
        for i,log in enumerate(logs):
            for j,m in enumerate(methods):
                for k,f in enumerate(metrics):writer.writerow([log,m,f,*counts[i,j,k]])
    report=dict(passed=True,updated_utc=datetime.now(timezone.utc).isoformat(),
        records=len(rows),queries=len(expected)//len(methods),unique_geometric_queries=statistics['unique_geometric_queries'],
        contexts=len(contexts),route_condition_records=geometry_routes,unique_geometry_paths=len(geometry_cache),
        unique_static_body_poses_recomputed=geometry_poses,source_files_checked=len(sources),
        point_estimate_and_CI_checked=len(methods)*len(metrics),paired_CI_checked=len(statistics['paired_differences']),
        same_endpoints_and_prediction_contexts=True,all_failures_retained=True,
        independent_point_and_rectangle_geometry=True,second_independent_lane_auditor=False,
        source_statistics_sha256=sha(OUT/'statistics/causal_maps_full_001/report.json'),
        audit_source_sha256=sha(__file__),sources={str(p):s for p,s in sources.items()},
        elapsed_s=time.monotonic()-begun,vehicle_executed=False,actual_arrival_credit=False,new_test_opened=False,goal_complete=False)
    (DEST/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 过去LiDAR六地图完整记录复核','',
        '212查询、1272条件、176个不同几何查询。固定端点、输入指纹、全部失败和84项指标/210项配对区间均通过。',
        '道路连续点及后轴矩形姿态以独立显式几何重算；车道方向/连接沿用已有独立连续审计器，本次没有第二套车道审计实现。',
        '正确对象由隔离评价端提供；未执行车辆，没有选对、正确锁定或实际到达信用。','',
        '|条件|生成路线/212|车道且可行驶/212|静态矩形通过/路线|','|---|---:|---:|---:|']
    for m in methods:
        s=statistics['conditions'][m];body=s['static_rectangle_legal_among_routes']
        lines.append(f"|{m}|{s['route_generated']['count']}|{s['full_road_legal_all_queries']['count']}|{body['count']}/{body['denominator']}|")
    (DEST/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='sources'},ensure_ascii=False))


if __name__=='__main__':
    main()
