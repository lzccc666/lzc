"""独立复核主450车身审核的冻结输入与逐区间账目；缺结果保留未知。"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box


PROJECT = Path('/root/autodl-tmp/lizichong')
OUT = PROJECT/'outputs/project_v2/thesis_completion_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    started = time.monotonic()
    root = OUT/'vehicle_execution/independent_body_main_full_001'
    report, frozen = read(root/'report.json'), read(root/'release.json')
    assert report['cases'] == report['expected_cases'] == 450 and not report['is_smoke']
    taskpath = OUT/'interactive_simulation/synthetic_tasks_v1.json'
    task_release = read(taskpath)
    configs = {row['config_id']: row for row in task_release['configs']}
    tasks = {row['task_id']: row for row in task_release['tasks']}
    source = OUT/'interactive_simulation/realtime_three_model_full_001'
    summary = read(source/'summary.json')
    assert summary['attempted'] == summary['expected'] == 450
    assert sha(source/'trial_records.jsonl') == summary['trial_records_sha256']
    assert sha(source/'summary.json') == frozen['original_summary_sha256'] == report['source_summary_sha256']
    assert sha(source/'release.json') == frozen['original_release_sha256']
    assert sha(taskpath) == frozen['task_release_sha256']
    files, totals, maxima = {}, Counter(), Counter()
    for path, value in frozen['source_sha256'].items():
        assert sha(PROJECT/path) == value
        files[str(PROJECT/path)] = value
    cases = [json.loads(line) for line in (root/'cases.jsonl').open()]
    raw_cases = [json.loads(line) for line in (source/'trial_records.jsonl').open()]
    expected = {(method, task, seed) for method in ['base_full', 'new_full', 'rule']
                for task in tasks for seed in task_release['seeds']}
    keys = [(row['method'], row['task'], row['seed']) for row in cases]
    assert len(keys) == len(set(keys)) == 450 and set(keys) == expected
    assert keys == [(row['variant'], row['task'], row['seed']) for row in raw_cases]
    destination = OUT/'vehicle_execution/main_body_evidence_check_001'
    destination.mkdir(exist_ok=False)
    statuses, deepest = Counter(), 0
    for index, (record, original) in enumerate(zip(cases, raw_cases)):
        assert time.monotonic()-started < 900
        folder = root/f"{index:04d}_{record['task']}_seed{record['seed']}_{record['method']}"
        assert read(folder/'report.json') == record
        assert record['raw_case'] == original['raw_case']
        assert record['prior_reported_arrival'] == original['actual_arrival']
        for path, value in record['source_sha256'].items():
            assert sha(path) == value
            files[path] = value
        geometry = read(folder/'geometry.json')
        config, task = configs[record['config_id']], tasks[record['task']]
        assert geometry['road_polygon_xy'] == config['road_polygon_xy']
        assert Polygon(geometry['road_polygon_xy']).is_valid
        for position, target in enumerate(config['targets']):
            reference = translate(rotate(box(-target['length_m']/2, -target['width_m']/2,
                                            target['length_m']/2, target['width_m']/2),
                                         target['yaw_rad'], origin=(0, 0), use_radians=True),
                                  *target['center_xy'])
            actual = Polygon(geometry['obstacles_xy'][position])
            assert actual.hausdorff_distance(reference) < 1e-12
            assert abs(actual.area-target['length_m']*target['width_m']) < 1e-11
            totals['independent_parked_rectangle_checks'] += 1
        if task['kind'] == 'inaccessible':
            assert len(geometry['obstacles_xy']) == 4
            barrier = Polygon(geometry['obstacles_xy'][-1])
            assert abs(barrier.area-200.) < 1e-9
            progress, center = 0., None
            for a, b in zip(config['centerline_xy'], config['centerline_xy'][1:]):
                length = math.dist(a, b)
                if progress+length >= 18.:
                    ratio = (18.-progress)/length
                    center = [a[k]+ratio*(b[k]-a[k]) for k in [0, 1]]
                    break
                progress += length
            assert center is not None and math.dist(center, [barrier.centroid.x, barrier.centroid.y]) < 1e-12
            totals['independent_barrier_checks'] += 1
        else:
            assert len(geometry['obstacles_xy']) == 3
        statuses[record['status']] += 1
        if record['status'] in ['unknown_missing_complete_evidence', 'unknown_missing_initial_state']:
            assert record['passed'] is False and record['audited_intervals'] == 0
            assert not (folder/'interval_audit.json').exists()
            if record['status'] == 'unknown_missing_complete_evidence':
                assert original['result'] is None and not record['prior_reported_arrival']
                totals['unknown_missing_complete_results'] += 1
            else:
                assert original['result'] is not None
                observed = json.loads(next((Path(record['raw_case'])/'perception.jsonl').open()))['sensor']['state']
                assert observed['time_s'] == record['initial_observed_time_s'] > 0.
                totals['unknown_missing_initial_states'] += 1
        else:
            original_path = Path(record['raw_case'])
            intervals = read(folder/'interval_audit.json')
            execution = [json.loads(line) for line in (original_path/'execution.jsonl').open()]
            initial = json.loads(next((original_path/'perception.jsonl').open()))['sensor']['state']
            assert initial['time_s'] == 0.
            assert len(execution) == len(intervals['intervals']) == intervals['audited_intervals'] == record['audited_intervals']
            assert [row['index'] for row in intervals['intervals']] == list(range(len(execution)))
            counts = Counter(row['status'] for row in intervals['intervals'])
            assert all(intervals['counts'][key] == value for key, value in counts.items())
            assert intervals['passed'] == (counts['certificate_safe'] == len(execution) and len(execution) > 0)
            assert record['passed'] == intervals['passed'] and record['status'] == intervals['status']
            assert intervals['numerical_padding_m'] == 2e-6 and intervals['maximum_subdivision_depth'] == 12
            for interval in intervals['intervals']:
                for field, value in interval['state_residuals'].items():
                    assert 0 <= value <= intervals['tolerances'][field]
                    maxima[field] = max(maxima[field], value)
                deepest = max(deepest, interval.get('deepest_subdivision', 0))
            moving = sum(row['state']['speed_mps'] > 1e-6 for row in execution)
            assert moving == record['moving_control_records']
            totals['audited_intervals'] += len(execution)
            totals['moving_control_records'] += moving
            totals['moving_cases'] += moving > 0
            totals['certificate_safe_cases'] += record['passed']
            files[str(folder/'interval_audit.json')] = sha(folder/'interval_audit.json')
        totals['previously_reported_arrivals'] += record['prior_reported_arrival']
        totals['arrivals_with_body_certificate'] += record['prior_reported_arrival'] and record['passed']
        for path in [folder/'report.json', folder/'geometry.json']:
            files[str(path)] = sha(path)
    assert dict(statuses) == report['status_counts']
    for field in ['audited_intervals', 'moving_control_records', 'moving_cases', 'previously_reported_arrivals']:
        assert totals[field] == report[field]
    assert totals['certificate_safe_cases'] == report['certificate_safe_cases']
    assert totals['arrivals_with_body_certificate'] == report['previous_arrivals_with_continuous_body_certificate']
    assert report['new_arrivals'] == report['new_executions'] == report['actual_model_calls'] == 0
    for path in [root/'report.json', root/'release.json', root/'cases.jsonl',
                 source/'summary.json', source/'release.json', source/'trial_records.jsonl', taskpath]:
        files[str(path)] = sha(path)
    save(destination/'files.json', files)
    save(destination/'report.json', dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(), cases=450,
         totals=dict(totals), status_counts=dict(statuses), maximum_state_residuals=dict(maxima),
         deepest_subdivision=deepest, SHA_files=len(files), files_manifest_sha256=sha(destination/'files.json'),
         source_sha256=sha(__file__), wall_CPU_shared_s=time.monotonic()-started,
         actual_model_calls=0, new_executions=0, new_arrivals=0,
         limits=['逐区间证书和输入账目审核；没有第三套连续车身算法',
                 '缺结果保留未知，原始部分轨迹不增加完整安全证据信用',
                 '不重评语言身份、区域保持或方向；双精度GEOS不是形式化区间算术'], goal_complete=False))
    print(json.dumps(read(destination/'report.json'), ensure_ascii=False))


if __name__ == '__main__':
    main()
