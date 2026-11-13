"""官方连接/逐步轨迹/参考位姿与分母独立核验，不运行规划器或控制器。"""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import tarfile

import numpy as np
from shapely.geometry import Point, Polygon, box, shape
from shapely.ops import unary_union

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, dump


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rear_arc_covered(before, after, region, dt, *, max_depth=12):
    """另一种连续位置界：圆弧中点+路程半径方框，不复用原弦弓高管。"""
    travel = .5*(before['speed_mps']+after['speed_mps'])*dt
    turn = (after['yaw_rad']-before['yaw_rad']+math.pi)%(2*math.pi)-math.pi
    def point(f):
        half = turn*f/2
        scale = travel*f*(math.sin(half)/half if abs(half) > 1e-12 else 1.)
        angle = before['yaw_rad']+half
        return before['x']+scale*math.cos(angle), before['y']+scale*math.sin(angle)

    def certify(a, b, depth):
        x, y = point((a+b)/2)
        radius = travel*(b-a)/2+1e-9
        if region.covers(box(x-radius, y-radius, x+radius, y+radius)):
            return True
        if not all(region.covers(Point(point(f))) for f in [a, (a+b)/2, b]):
            return False
        if depth == max_depth:
            return None
        left, right = certify(a, (a+b)/2, depth+1), certify((a+b)/2, b, depth+1)
        return False if left is False or right is False else True if left is True and right is True else None

    return certify(0., 1., 0)


def main():
    started = time.monotonic()
    source = OUTPUT/'vehicle_execution/HDmap_reference_guided_full_001'
    root = OUTPUT/'vehicle_execution/HDmap_reference_guided_audit_001'
    root.mkdir(exist_ok=False)
    release, summary = read(source/'release.json'), read(source/'summary.json')
    assert summary['attempted'] == summary['expected'] == 228
    assert sha(source/'trials.jsonl') == summary['trial_records_sha256']
    tasks_path = OUTPUT/'vehicle_execution/HDmap_connection_tasks_001/tasks.json'
    assert sha(tasks_path) == release['tasks_sha256']
    tasks = {t['task_id']: t for t in read(tasks_path)}
    cases = [json.loads(line) for line in (source/'trials.jsonl').open()]
    expected = {(t, seed, method) for t in tasks for seed in [17, 29, 43] for method in ['hybrid_astar', 'reference_guided']}
    assert len(cases) == len(expected) == 228 and {(r['task_id'], r['seed'], r['method']) for r in cases} == expected
    assert [(c['task_id'], c['seed'], c['method']) for c in cases] == [(c['task_id'], c['seed'], c['method']) for c in release['plans']]
    for pair_index in range(len(cases)//2):
        pair = cases[2*pair_index:2*pair_index+2]
        assert pair[0]['task_id'] == pair[1]['task_id'] and pair[0]['seed'] == pair[1]['seed']
        order = ['hybrid_astar', 'reference_guided'] if pair_index%2 == 0 else ['reference_guided', 'hybrid_astar']
        assert [c['method'] for c in pair] == order
    source_receipt = read(source/'source_snapshot_receipt.json')
    assert sha(source/'source_snapshot.tar.gz') == source_receipt['archive_sha256']
    with tarfile.open(source/'source_snapshot.tar.gz', 'r:gz') as archive:
        assert len(archive.getmembers()) == len(release['source_sha256']) == 8
        for member in archive.getmembers():
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == release['source_sha256'][member.name]
    assert all(sha(PROJECT/p) == value for p, value in release['source_sha256'].items())
    files = {str(tasks_path): sha(tasks_path), str(source/'release.json'): sha(source/'release.json'),
             str(source/'summary.json'): sha(source/'summary.json'), str(source/'trials.jsonl'): sha(source/'trials.jsonl')}
    dump(root/'release.json', dict(source_sha256=sha(__file__), original_release_sha256=sha(source/'release.json'),
        original_trials_sha256=sha(source/'trials.jsonl'), budget_wall_s=900, budget_bytes=15000000,
        scope='官方地图连接、独立后轴圆弧方框界、连续车身证书账目、参考位姿保持与配对统计',
        new_executions=0, new_model_calls=0, semantic_arrival_credit=0, goal_complete=False))
    map_cache = {}
    for location in ['boston-seaport', 'singapore-onenorth']:
        path = PROJECT/'dataset/nuscenes_map_expansion/expansion'/f'{location}.json'
        data = read(path)
        maps = dict(data=data, nodes={n['token']: (n['x'], n['y']) for n in data['node']},
                    polygons={p['token']: p for p in data['polygon']},
                    lanes={p['token']: p for p in data['lane']+data['lane_connector']})
        map_cache[location] = maps
        files[str(path)] = sha(path)
    results = []
    counts = Counter()
    for index, case in enumerate(cases):
        assert time.monotonic()-started < 900
        task = tasks[case['task_id']]
        folder = source/f"{index:03d}_{task['scene']}_seed{case['seed']}_{case['method']}"
        assert read(folder/'result.json') == case
        assert sha(task['context_path']) == task['context_sha256'] == case['context_sha256']
        context = read(task['context_path'])
        files[task['context_path']] = task['context_sha256']
        checked = dict(task_id=case['task_id'], scene=case['scene'], log_token=case['log_token'], seed=case['seed'],
                       method=case['method'], preparation_status=task['status'], original_status=case['status'],
                       actual_motion=case['actual_vehicle_motion'], path_generated=case['path_generated'], reference_pose_reached=False,
                       independently_confirmed_reference_pose=False, accepted_distinct_connections=0,
                       rear_arc_bound_status='not_exercised', independent_direction_global_cone='not_exercised',
                       complete_body_certificate=None, semantic_arrival_credit=0)
        if not (folder/'execution.jsonl').exists():
            assert case['saved_control_records'] == 0 and not case['reference_pose_reached']
        else:
            assert sha(folder/'execution.jsonl') == case['source_execution_sha256']
            raw = [json.loads(line) for line in (folder/'execution.jsonl').open()]
            states = read(folder/'states.json')
            geometry = read(folder/'geometry.json')
            assert len(states) == len(raw)+1 and len(raw) == case['saved_control_records']
            assert states[0] == case['initial_state'] and states[-1] == case['final_state']
            assert [r['state'] for r in raw] == states[1:]
            lanes = [context['lane_geometries'][token] for token in task['lane_tokens']]
            city = next(v for v in map_cache.values() if all(t in v['lanes'] for t in task['lane_tokens']))
            official = city['data']
            polygons = []
            for lane in lanes:
                record = city['lanes'][lane['token']]
                polygon = city['polygons'][record['polygon_token']]
                official_shape = Polygon([city['nodes'][t] for t in polygon['exterior_node_tokens']],
                    [[city['nodes'][t] for t in h['node_tokens']] for h in polygon.get('holes', [])])
                actual = Polygon(lane['polygon'])
                assert actual.hausdorff_distance(official_shape) < 1e-8 and abs(actual.area-official_shape.area) < 1e-8
                arcline = official['arcline_path_3'][lane['token']]
                assert np.allclose(lane['start_pose'], arcline[0]['start_pose'], atol=1e-8, rtol=0)
                assert np.allclose(lane['end_pose'], arcline[-1]['end_pose'], atol=1e-8, rtol=0)
                polygons.append(actual)
            for a, b in zip(task['lane_tokens'], task['lane_tokens'][1:]):
                assert b in official['connectivity'][a]['outgoing']
            road = unary_union(polygons).intersection(shape(context['drivable_geojson']))
            assert road.equals(shape(geometry['road']))
            assert len(geometry['obstacles']) == len(context['obstacles_geojson'])
            assert all(shape(a).equals(shape(b)) for a, b in zip(geometry['obstacles'], context['obstacles_geojson']))
            rng = np.random.default_rng(case['seed'])
            initial_xy = np.asarray(task['start_xy'])+rng.normal(0, .03, 2)
            initial_yaw = task['start_yaw_rad']+rng.normal(0, math.radians(.5))
            assert np.allclose([states[0]['x'], states[0]['y']], initial_xy, atol=1e-12, rtol=0)
            assert states[0]['yaw_rad'] == initial_yaw and states[0]['time_s'] == 0
            if raw:
                body = read(folder/'independent_continuous_body.json')
                assert body['audited_intervals'] == len(body['intervals']) == len(raw)
                assert [r['index'] for r in body['intervals']] == list(range(len(raw)))
                checked['complete_body_certificate'] = body['passed']
                assert body['passed'] == all(r['status'] == 'certificate_safe' for r in body['intervals'])
                for interval in body['intervals']:
                    for field, residual in interval['state_residuals'].items():
                        assert residual <= body['tolerances'][field]
                # 从完整末端状态回溯一个贯穿全程的合法车道见证；不能将每步
                # 可选切换候选累加为实际连接次数。
                end_stage = 2 if case['reference_pose_reached'] else min(raw[-1]['lane_transition']['next_stages'])
                chain = [end_stage]
                witnesses = []
                for row in reversed(raw):
                    matches = [w for w in row['lane_transition']['witnesses'] if w['to_stage'] == chain[-1]]
                    assert matches
                    witness = min(matches, key=lambda w: w['from_stage'])
                    chain.append(witness['from_stage'])
                    witnesses.append(witness)
                chain.reverse()
                witnesses.reverse()
                assert chain[0] == 0 and len(chain) == len(raw)+1
                rear_status, direction_status = [], []
                held = 0.
                for j, (before, after, row, witness) in enumerate(zip(states, states[1:], raw, witnesses)):
                    source_stage, target_stage = chain[j:j+2]
                    assert 0 <= target_stage-source_stage <= 1
                    assert witness['source_token'] == lanes[source_stage]['token'] and witness['target_token'] == lanes[target_stage]['token']
                    assert polygons[source_stage].covers(Point(before['x'], before['y']))
                    assert polygons[target_stage].covers(Point(after['x'], after['y']))
                    turn = (after['yaw_rad']-before['yaw_rad']+math.pi)%(2*math.pi)-math.pi
                    travel = .5*(before['speed_mps']+after['speed_mps'])*release['vehicle']['dt_s']
                    if source_stage != target_stage:
                        assert lanes[target_stage]['token'] in official['connectivity'][lanes[source_stage]['token']]['outgoing']
                        endpoint = [lanes[source_stage]['end_pose'][:2], lanes[target_stage]['start_pose'][:2]]
                        assert max(math.dist([s['x'], s['y']], p) for s in [before, after] for p in endpoint) <= 2.+1e-9
                    area = unary_union([polygons[k] for k in set([source_stage, target_stage])])
                    rear_status.append(rear_arc_covered(before, after, area, release['vehicle']['dt_s']))
                    if travel > 1e-12:
                        vector = np.array([math.cos(before['yaw_rad']+turn/2), math.sin(before['yaw_rad']+turn/2)])
                        limits = []
                        for stage in set([source_stage, target_stage]):
                            v = np.diff(np.asarray(lanes[stage]['centerline'])[:, :2], axis=0)
                            v = v[np.linalg.norm(v, axis=1) > 1e-12]
                            u = v/np.linalg.norm(v, axis=1)[:, None]
                            angles = np.abs(np.arctan2(u[:, 0]*vector[1]-u[:, 1]*vector[0], u@vector))
                            limits.append(float(angles.max())+abs(turn)/2)
                        direction_status.append(max(limits) < math.pi/2-1e-10)
                    inside = math.dist([after['x'], after['y']], task['goal_xy']) <= .75
                    yaw_ok = abs(math.atan2(math.sin(after['yaw_rad']-task['goal_yaw_rad']), math.cos(after['yaw_rad']-task['goal_yaw_rad']))) <= math.radians(12)
                    stopped = after['speed_mps'] <= .1
                    final_lane = target_stage == 2
                    held = held+release['vehicle']['dt_s'] if inside and yaw_ok and stopped and final_lane else 0.
                checked.update(accepted_distinct_connections=sum(a != b for a, b in zip(chain, chain[1:])),
                    connection_candidate_intervals=case['connection_candidate_intervals'],
                    accepted_lane_stage_path=chain, independent_terminal_held_s=held,
                    reference_pose_reached=held >= 2.-1e-8,
                    rear_arc_bound_status='violation' if False in rear_status else 'unknown' if None in rear_status else 'certificate',
                    independent_direction_global_cone='certificate' if all(direction_status) else 'unknown_global_cone_too_wide',
                    actual_connection_tokens=[dict(source=lanes[a]['token'], target=lanes[b]['token'])
                        for a, b in zip(chain, chain[1:]) if a != b])
                assert case['accepted_lane_stage_path'] == chain
                assert case['accepted_distinct_connections'] == checked['accepted_distinct_connections']
                assert checked['reference_pose_reached'] == case['reference_pose_reached']
                checked['independently_confirmed_reference_pose'] = bool(checked['reference_pose_reached'] and body['passed']
                    and checked['rear_arc_bound_status'] == 'certificate' and checked['independent_direction_global_cone'] == 'certificate')
                counts['control_intervals'] += len(raw)
                counts['rear_arc_independent_certified_intervals'] += sum(v is True for v in rear_status)
                counts['rear_arc_independent_unknown_intervals'] += sum(v is None for v in rear_status)
                counts['rear_arc_independent_violation_intervals'] += sum(v is False for v in rear_status)
                counts['distinct_assigned_connections'] += checked['accepted_distinct_connections']
            else:
                assert not case['actual_vehicle_motion'] and not case['reference_pose_reached']
            for path in folder.iterdir():
                if path.is_file():
                    files[str(path)] = sha(path)
        results.append(checked)
    assert Counter(r['original_status'] for r in results) == summary['status_counts']
    for method in ['hybrid_astar', 'reference_guided']:
        selected = [r for r in results if r['method'] == method]
        assert len(selected) == summary['by_method'][method]['trials'] == 114
        assert sum(r['reference_pose_reached'] for r in selected) == summary['by_method'][method]['reference_pose_reached']
    with (root/'cases.jsonl').open('x') as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    by_log = defaultdict(lambda: Counter())
    for row in results:
        key = row['log_token']
        by_log[key][row['method']+'_trials'] += 1
        by_log[key][row['method']+'_reference_reached'] += row['reference_pose_reached']
        by_log[key][row['method']+'_independent_reference'] += row['independently_confirmed_reference_pose']
    with (root/'by_log.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['log_token']+sorted({k for v in by_log.values() for k in v}))
        writer.writeheader()
        writer.writerows(dict(log_token=k, **v) for k, v in sorted(by_log.items()))
    rng = np.random.default_rng(5723)
    logs = sorted(by_log)
    differences = []
    for _ in range(4000):
        selected = rng.choice(logs, size=len(logs), replace=True)
        aggregated = Counter()
        for log in selected:
            aggregated.update(by_log[log])
        differences.append(aggregated['reference_guided_reference_reached']/aggregated['reference_guided_trials']
                           -aggregated['hybrid_astar_reference_reached']/aggregated['hybrid_astar_trials'])
    dump(root/'files.json', files)
    report = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(), trials=228, scenes=38, logs=len(by_log),
        source_sha256=sha(__file__), totals=dict(counts), source_files=len(files), files_manifest_sha256=sha(root/'files.json'),
        methods={method: dict(trials=sum(r['method'] == method for r in results),
            planned_routes=sum(r['method'] == method and r['path_generated'] for r in results),
            actual_motion=sum(r['method'] == method and r['actual_motion'] for r in results),
            reference_pose_reached=sum(r['method'] == method and r['reference_pose_reached'] for r in results),
            independently_confirmed_reference_pose=sum(r['method'] == method and r['independently_confirmed_reference_pose'] for r in results))
            for method in ['hybrid_astar', 'reference_guided']},
        guided_minus_original_reference_reached_rate=sum(r['method'] == 'reference_guided' and r['reference_pose_reached'] for r in results)/114
            -sum(r['method'] == 'hybrid_astar' and r['reference_pose_reached'] for r in results)/114,
        paired_log_cluster_95ci=np.quantile(differences, [.025, .975]).tolist(), bootstrap_replicates=4000, bootstrap_seed=5723,
        raw_counter_interpretation='connection_candidate_intervals是候选控制区间，accepted_distinct_connections是完整见证的实际过渡，不混用',
        reference_pose_is_not_semantic_arrival=True, new_executions=0, new_model_calls=0, actual_semantic_arrival_credit=0,
        wall_CPU_shared_s=time.monotonic()-started,
        limits=['原始失败已查看后的开发改进；两者3秒/2000扩展，但扩展工作单位不同，共享CPU不能作为独占性能',
                '新版本同时引入中心线连续建议和逐步有向拓扑搜索，增益不单独归因于转角建议',
                '独立连续车身证书由不同于执行器的积分/几何实现产生，本检查只复核其账目和输入，非第三种车身算法',
                '后轴轨迹用不同的圆弧中点方框界复核，未决保留；方向全线段锥过宽时仅给unknown',
                '人工参考起始与地图路线，静态预测障碍，非录制实车后轴、感知闭环或语义正确到达'], goal_complete=False)
    dump(root/'report.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
