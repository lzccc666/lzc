"""初态重建联合账目的独立输入与区间审核；用显式PCG64标准正态复核种子。"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


OUT = Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    original = OUT/'vehicle_execution/independent_body_main_full_001'
    root = OUT/'vehicle_execution/independent_body_main_seed_initial_001'
    report, release = read(root/'report.json'), read(root/'release.json')
    assert report['cases'] == report['expected_cases'] == 450 and not report['is_smoke']
    assert sha(original/'report.json') == release['original_report_sha256']
    assert sha(original/'cases.jsonl') == release['original_cases_sha256']
    task_path = OUT/'interactive_simulation/synthetic_tasks_v1.json'
    assert sha(task_path) == release['task_release_sha256']
    data = read(task_path)
    configs = {row['config_id']: row for row in data['configs']}
    rows = [json.loads(line) for line in (root/'cases.jsonl').open()]
    parents = [json.loads(line) for line in (original/'cases.jsonl').open()]
    assert len(rows) == len(parents) == len({(r['method'], r['task'], r['seed']) for r in rows}) == 450
    files, totals, statuses = {}, Counter(), Counter()
    maximum_initial_residual = 0.
    for index, (row, parent) in enumerate(zip(rows, parents)):
        for field in ['method', 'task', 'seed', 'config_id', 'raw_case', 'prior_reported_arrival', 'source_sha256']:
            assert row[field] == parent[field]
        name = f"{index:04d}_{row['task']}_seed{row['seed']}_{row['method']}"
        directory, old = root/name, original/name
        assert read(directory/'report.json') == row
        assert sha(old/'report.json') == row['original_strict_report_sha256']
        assert row['original_strict_status'] == parent['status']
        for path, value in row['source_sha256'].items():
            assert sha(path) == value
            files[path] = value
        if parent['status'] == 'certificate_safe':
            assert row['initial_state_provenance'] == 'saved_time_zero; prior_certificate_reused'
            assert row['audited_intervals'] == parent['audited_intervals']
            assert sha(old/'interval_audit.json') == row['reused_interval_audit_sha256']
            totals['reused_certificates'] += 1
            totals['reused_intervals'] += parent['audited_intervals']
        elif parent['status'] == 'unknown_missing_complete_evidence':
            assert row['status'] == parent['status'] and row['passed'] is False
            assert not row['prior_reported_arrival'] and row['audited_intervals'] == 0
            totals['unknown_missing_complete_results'] += 1
        else:
            assert parent['status'] == 'unknown_missing_initial_state'
            proof = read(directory/'initial_reconstruction.json')
            config = configs[row['config_id']]
            # 显式PCG64及一次标准正态向量，与default_rng/两次normal调用分别核验。
            noise = np.random.Generator(np.random.PCG64(row['seed'])).standard_normal(3)
            expected = dict(x=config['start_xy'][0]+float(noise[0])*.03,
                            y=config['start_xy'][1]+float(noise[1])*.03,
                            yaw_rad=config['start_yaw_rad']+float(noise[2])*math.pi/360,
                            speed_mps=0., steering_rad=0., time_s=0.)
            for field, value in expected.items():
                residual = abs(value-proof['reconstructed_initial'][field])
                assert residual < 1e-12
                maximum_initial_residual = max(maximum_initial_residual, residual)
            raw = Path(row['raw_case'])
            assert sha(raw/'execution.jsonl') == proof['raw_execution_sha256']
            assert sha(raw/'perception.jsonl') == proof['raw_perception_sha256']
            assert sha(old/'geometry.json') == proof['original_geometry_sha256']
            assert read(directory/'geometry.json') == read(old/'geometry.json')
            execution = [json.loads(line) for line in (raw/'execution.jsonl').open()]
            observed = json.loads(next((raw/'perception.jsonl').open()))['sensor']['state']
            assert observed == proof['first_observed_state'] and observed['time_s'] > 0
            prefix = [record for record in execution if record['state']['time_s'] <= observed['time_s']]
            assert len(prefix) == proof['stationary_prefix_records'] and prefix[-1]['state'] == observed
            for record in prefix:
                assert record['stale_sensor_braking'] and record['sensor_age_s'] is None
                assert record['state']['speed_mps'] == record['state']['steering_rad'] == 0
                for field in ['x', 'y', 'yaw_rad']:
                    assert abs(record['state'][field]-expected[field]) < 1e-12
            audit = read(directory/'interval_audit.json')
            assert len(audit['intervals']) == len(execution) == row['audited_intervals']
            assert [record['index'] for record in audit['intervals']] == list(range(len(execution)))
            counts = Counter(record['status'] for record in audit['intervals'])
            assert counts['certificate_safe'] == len(execution) and audit['passed'] is row['passed'] is True
            for interval in audit['intervals']:
                assert all(0 <= value <= audit['tolerances'][field] for field, value in interval['state_residuals'].items())
            totals['reconstructed_initials'] += 1
            totals['stationary_prefix_records'] += len(prefix)
            totals['newly_audited_intervals'] += len(execution)
            for path in [directory/'initial_reconstruction.json', directory/'interval_audit.json', directory/'geometry.json']:
                files[str(path)] = sha(path)
        statuses[row['status']] += 1
        totals['previously_reported_arrivals'] += row['prior_reported_arrival']
        totals['reported_arrivals_with_body_certificate'] += row['prior_reported_arrival'] and row['passed']
        files[str(directory/'report.json')] = sha(directory/'report.json')
        files[str(old/'report.json')] = sha(old/'report.json')
    assert dict(statuses) == report['status_counts']
    assert totals['reconstructed_initials'] == 54 and totals['reused_certificates'] == 395
    assert totals['unknown_missing_complete_results'] == 1 and totals['stationary_prefix_records'] == 68
    assert totals['reused_intervals']+totals['newly_audited_intervals'] == report['audited_intervals'] == 333929
    assert totals['previously_reported_arrivals'] == report['previously_reported_arrivals'] == 93
    assert totals['reported_arrivals_with_body_certificate'] == report['previous_arrivals_with_continuous_body_certificate'] == 93
    for path in [task_path, root/'report.json', root/'release.json', root/'cases.jsonl', original/'report.json', original/'cases.jsonl']:
        files[str(path)] = sha(path)
    destination = OUT/'vehicle_execution/main_seed_initial_ledger_001'
    destination.mkdir(exist_ok=False)
    with (destination/'files.json').open('x') as handle:
        json.dump(files, handle, ensure_ascii=False, indent=2, allow_nan=False)
    result = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(), cases=450, totals=dict(totals),
                  status_counts=dict(statuses), maximum_reconstructed_initial_residual=maximum_initial_residual,
                  source_sha256=sha(__file__), SHA_files=len(files), files_manifest_sha256=sha(destination/'files.json'),
                  RNG_crosscheck='Generator(PCG64(seed)).standard_normal(3), explicit scales',
                  initial_is_reconstructed_not_observed=True, old_strict_unknowns_preserved=True,
                  actual_model_calls=0, new_executions=0, new_arrivals=0, goal_complete=False)
    with (destination/'report.json').open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
