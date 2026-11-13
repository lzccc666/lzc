"""独立全线段航向锥及GEOS进度核对；不复用方向模块的候选或界。"""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from shapely.geometry import LineString, Point

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, dump


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while data := handle.read(1048576):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    started = time.monotonic()
    destination = OUTPUT/'vehicle_execution/execution_direction_evidence_check_002'
    destination.mkdir(exist_ok=False)
    data_path = OUTPUT/'interactive_simulation/synthetic_tasks_v1.json'
    data = read(data_path)
    config = {c['config_id']: c for c in data['configs']}
    files, cohorts = {}, []
    totals = Counter()
    minimum_global_margin = math.inf
    for name, body_name, expected in [('execution_direction_rule_full_001', 'independent_body_rule_full_001', 150),
                                      ('execution_direction_fault_cpu_full_001', 'independent_body_fault_cpu_full_001', 6)]:
        root = OUTPUT/'vehicle_execution'/name
        report, release = read(root/'report.json'), read(root/'release.json')
        assert report['cases'] == report['expected_cases'] == expected and not report['is_smoke']
        assert release['task_release_sha256'] == sha(data_path)
        for path, value in release['source_sha256'].items():
            assert sha(PROJECT/path) == value
        body_root = OUTPUT/'vehicle_execution'/body_name
        assert sha(body_root/'report.json') == release['parent_body_report_sha256']
        assert sha(body_root/'cases.jsonl') == release['parent_body_cases_sha256']
        body = [json.loads(line) for line in (body_root/'cases.jsonl').open()]
        records = [json.loads(line) for line in (root/'cases.jsonl').open()]
        def key(r):
            return r['task'], r['seed'], r['method']
        assert len(records) == len({key(r) for r in records}) == expected
        assert {key(r) for r in records} == {key(r) for r in body}
        counts, by_config = Counter(), {}
        for index, record in enumerate(records):
            folder = root/f"{index:04d}_{record['task']}_seed{record['seed']}_{record['method']}"
            assert read(folder/'report.json') == record
            audit = read(folder/'interval_audit.json')
            case = Path(record['raw_case'])
            rows = [json.loads(line) for line in (case/'execution.jsonl').open()]
            initial = json.loads(next((case/'perception.jsonl').open()))['sensor']['state']
            assert initial['time_s'] == 0 and len(rows) == len(audit['intervals']) == record['audited_intervals']
            assert [r['index'] for r in audit['intervals']] == list(range(len(rows)))
            for path, value in record['source_sha256'].items():
                assert sha(path) == value
                files[path] = value
            c = config[record['config_id']]
            assert c['topology'] == '单向单车道，没有变道、倒车、逆行或未声明连接'
            points = np.asarray(c['centerline_xy'])
            vectors = np.diff(points, axis=0)
            units = vectors/np.linalg.norm(vectors, axis=1)[:, None]
            reference = LineString(c['centerline_xy'])
            previous = initial
            moving, globally_forward = 0, 0
            enumerated = Counter()
            for raw, interval in zip(rows, audit['intervals']):
                assert time.monotonic()-started < 900
                current = raw['state']
                travel = .5*(previous['speed_mps']+current['speed_mps'])*data['vehicle']['dt_s']
                turn = (current['yaw_rad']-previous['yaw_rad']+math.pi)%(2*math.pi)-math.pi
                enumerated[interval['status']] += 1
                assert abs(travel-interval['travel_m']) < 1e-12
                assert abs(turn-interval['heading_change_rad']) < 1e-12
                if travel > 1e-12:
                    moving += 1
                    # 第二种更保守的充分条件：对中心线所有线段同时计算
                    # 中点航向与切线夹角，再加半步转角。全锥小于90°即
                    # 连续前向。无需局部最近段候选、空间半径或递归细分。
                    middle = previous['yaw_rad']+turn/2
                    heading = np.array([math.cos(middle), math.sin(middle)])
                    dot = units@heading
                    cross = units[:, 0]*heading[1]-units[:, 1]*heading[0]
                    half_angle = float(np.max(np.abs(np.arctan2(cross, dot))))+abs(turn)/2
                    margin = math.pi/2-half_angle
                    minimum_global_margin = min(minimum_global_margin, margin)
                    if margin > 1e-10:
                        globally_forward += 1
                    else:
                        raise AssertionError('全线段锥无法确认，须保留并另审，不能声称独立复核通过')
                    assert interval['status'] == 'certificate_forward'
                    spans = interval['spans']
                    assert spans[0]['fraction_start'] == 0 and spans[-1]['fraction_end'] == 1
                    assert all(a['fraction_end'] == b['fraction_start'] for a, b in zip(spans, spans[1:]))
                    assert all(s['status'] == 'certificate_forward' and s['cosine_lower_bound'] > 1e-10 for s in spans)
                    for state, prefix in [(previous, 'before'), (current, 'after')]:
                        projected = reference.project(Point(state['x'], state['y']))
                        assert interval[f'progress_{prefix}_min_m']-1e-8 <= projected <= interval[f'progress_{prefix}_max_m']+1e-8
                    counts['GEOS_progress_checks'] += 2
                else:
                    assert interval['status'] == 'not_exercised_stationary' and not interval['moving']
                previous = current
            assert dict(enumerated) == audit['counts'] == record['counts']
            assert moving == globally_forward == record['moving_intervals']
            if moving:
                assert record['passed'] is True and record['combined_body_and_moving_direction_certificate']
            else:
                assert record['status'] == 'not_exercised_stationary' and record['passed'] is None
                assert not record['combined_body_and_moving_direction_certificate']
            assert record['multi_lane_connection_accuracy'] is None
            counts['cases'] += 1
            counts['intervals'] += len(rows)
            counts['moving_intervals'] += moving
            counts['global_cone_forward_intervals'] += globally_forward
            counts['moving_cases'] += moving > 0
            counts['stationary_cases'] += moving == 0
            counts['old_arrivals'] += record['prior_reported_arrival']
            counts['old_arrivals_with_body_direction'] += record['prior_reported_arrival'] and record['combined_body_and_moving_direction_certificate']
            group = by_config.setdefault(record['config_id'], Counter())
            group['cases'] += 1
            group['moving_cases'] += moving > 0
            group['forward_certified_cases'] += moving > 0
            group['stationary_not_exercised'] += moving == 0
            for path in [folder/'report.json', folder/'interval_audit.json']:
                files[str(path)] = sha(path)
        assert counts['cases'] == report['cases'] and counts['intervals'] == report['audited_intervals']
        assert counts['moving_cases'] == report['moving_cases'] == report['direction_certified_moving_cases']
        assert counts['moving_intervals'] == report['moving_intervals']
        assert counts['old_arrivals'] == counts['old_arrivals_with_body_direction'] == report['previous_arrivals_with_body_and_direction']
        with (root/'by_config.csv').open(newline='') as handle:
            table = list(csv.DictReader(handle))
        assert len(table) == len(by_config)
        for row in table:
            expected_row = by_config[row['config_id']]
            assert all(int(row[field]) == expected_row[field] for field in
                       ['cases', 'moving_cases', 'forward_certified_cases', 'stationary_not_exercised', 'wrong_way_cases', 'unknown_cases'])
        if len(by_config) == 1:
            assert report['config_cluster_bootstrap_95ci'] is None and report['bootstrap_replicates'] == 0
        else:
            rng = np.random.default_rng(5723)
            keys = sorted(by_config)
            values = []
            for _ in range(4000):
                sampled = rng.choice(keys, size=len(keys), replace=True)
                denominator = sum(by_config[k]['moving_cases'] for k in sampled)
                if denominator:
                    values.append(sum(by_config[k]['forward_certified_cases'] for k in sampled)/denominator)
            assert np.allclose(np.percentile(values, [2.5, 97.5]), report['config_cluster_bootstrap_95ci'], rtol=0, atol=1e-12)
        assert report['actual_model_calls'] == report['new_executions'] == report['new_arrivals'] == 0
        assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) < 1024**3
        cohorts.append(dict(run=name, **dict(counts)))
        totals.update(counts)
        for path in [root/'report.json', root/'release.json', root/'by_config.csv', root/'cases.jsonl']:
            files[str(path)] = sha(path)
    assert totals['cases'] == 156 and totals['intervals'] == 33076
    assert totals['moving_cases'] == 126 and totals['stationary_cases'] == 30
    dump(destination/'files.json', files)
    report = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(), cohorts=cohorts,
        totals=dict(totals), source_sha256=sha(__file__), source_files=len(files),
        files_manifest_sha256=sha(destination/'files.json'), minimum_global_heading_margin_rad=minimum_global_margin,
        independent_direction_bound='所有中心线切线的全局航向锥，与主审核的局部距离界/递归不同',
        independent_progress_backend='GEOS最近点投影，核对主审核NumPy投影的并列范围',
        multi_lane_connection_accuracy=None, new_model_calls=0, new_executions=0, new_arrivals=0,
        wall_CPU_shared_s=time.monotonic()-started,
        limits=['第二种方向界，不是另一套积分/车身或多车道拓扑证明', '单向单车道没有连接过渡，连接评价未检验',
                '只核对冻结保存模型与静态地图；不外推真实动态交通或未知道路安全'], goal_complete=False)
    dump(destination/'report.json', report)
    lines = ['# 保存执行连续方向独立复核', '',
        '150规则与6CPU故障试验、33076区间全部枚举。126行驶试验方向证书获全线段航向锥的第二种更保守界复核；30未行驶不计方向安全。', '',
        f"实际运动区间{totals['moving_intervals']}，全部获第二种连续方向界；GEOS进度核对{totals['GEOS_progress_checks']}次。", '',
        '既有21规则与6CPU故障到达均有原独立车身和本次方向证据，不增加模型、执行或到达。', '',
        '十配置是单向单车道，连接准确率null、未检验；方向通过不代替道路包含、碰撞、身份和动态安全。', '',
        '证据与逐配置分母见同目录report.json和各cohort的逐区间报告；全部原结果保持。']
    (destination/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
