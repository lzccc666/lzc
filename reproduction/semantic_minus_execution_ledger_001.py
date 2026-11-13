"""独立复核缺一450审核的固定已完成前缀、输入SHA、初态和区间账目。"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prefix(path):
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        data = data[:data.rfind(b'\n')+1]
    return data, [json.loads(line) for line in data.splitlines()]


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run', default='semantic_minus_execution_full_001')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    source = ROOT/'vehicle_execution'/args.input_run
    release = read(source/'release.json')
    original = ROOT/'interactive_simulation/realtime_semantic_minus_full_001'
    assert sha(original/'release.json') == release['original_release_sha256']
    data, audits = prefix(source/'cases.jsonl')
    assert audits and len(audits) <= 450
    _, records = prefix(original/'trial_records.jsonl')
    assert len(records) >= len(audits)
    tasks = read(ROOT/'interactive_simulation/synthetic_tasks_v1.json')
    configs = {c['config_id']:c for c in tasks['configs']}
    destination = ROOT/'vehicle_execution'/args.run_id
    destination.mkdir(exist_ok=False)
    counts, files, strict, rebuilt, controls, moving, unresolved = Counter(), {}, 0, 0, 0, 0, []
    for i, (audit, record) in enumerate(zip(audits, records)):
        assert audit['index'] == i
        for key in ['variant', 'task', 'config_id', 'seed', 'raw_case', 'status']:
            assert audit[key] == record[key]
        assert audit['reported_arrival'] == record['actual_arrival']
        if record['result'] is None:
            assert audit['audit_status'] == 'unknown_missing_complete_result'
            assert not audit['reported_arrival']
            counts['unknown_missing_complete_result'] += 1
            continue
        directory = source/f'{i:04d}'
        assert read(directory/'report.json') == audit
        for name, expected in audit['source_sha256'].items():
            assert sha(name) == expected
            files[name] = expected
        raw = Path(audit['raw_case'])
        execution = [json.loads(line) for line in (raw/'execution.jsonl').open()]
        observed = json.loads(next((raw/'perception.jsonl').open()))['sensor']['state']
        proof = read(directory/'initial_state_proof.json')
        config = configs[record['config_id']]
        noise = np.random.Generator(np.random.PCG64(record['seed'])).standard_normal(3)
        expected = dict(x=config['start_xy'][0]+float(noise[0])*.03,
            y=config['start_xy'][1]+float(noise[1])*.03,
            yaw_rad=config['start_yaw_rad']+float(noise[2])*math.pi/360,
            speed_mps=0., steering_rad=0., time_s=0.)
        if proof['strict_observed_time_zero']:
            assert observed['time_s'] == 0 and not proof['reconstructed']
            assert all(abs(observed[key]-value) < 1e-12 for key, value in expected.items())
            strict += 1
        elif proof['reconstructed']:
            assert proof['reconstructed_is_not_observed'] and observed == proof['first_observed_state']
            assert all(abs(proof['reconstructed_initial'][key]-value) < 1e-12 for key, value in expected.items())
            stationary = [row for row in execution if row['state']['time_s'] <= observed['time_s']]
            assert len(stationary) == proof['stationary_prefix_records'] and stationary[-1]['state'] == observed
            for row in stationary:
                assert row['stale_sensor_braking'] and row['sensor_age_s'] is None
                assert row['state']['speed_mps'] == row['state']['steering_rad'] == 0.
                assert all(abs(row['state'][key]-expected[key]) < 1e-12 for key in ['x', 'y', 'yaw_rad'])
            rebuilt += 1
        else:
            assert not audit['augmented_body_certificate']
            counts['unknown_initial'] += 1
            continue
        certificate = directory/'certificates.json.gz'
        assert sha(certificate) == audit['certificates_sha256']
        with gzip.open(certificate, 'rt', encoding='utf-8') as handle:
            body, direction, held = (lambda x:(x['body'], x['direction'], x['hold']))(json.load(handle))
        assert body['audited_intervals'] == len(body['intervals']) == len(execution) == audit['audit_intervals']
        assert len(direction['intervals']) == len(execution) == audit['saved_control_records']
        assert body['passed'] == audit['augmented_body_certificate']
        assert direction['moving_intervals'] == audit['moving_intervals']
        assert direction['status'] == audit['direction_status']
        assert audit['conservative_hold_certificate'] == bool(held['certified'] and body['passed'] and direction['passed'] is True)
        assert not held['uses_result_arrival_as_truth']
        for j, interval in enumerate(body['intervals']):
            assert interval['index'] == j
            if body['passed']:
                assert interval['status'] == 'certificate_safe'
                assert all(0 <= value <= body['tolerances'][key] for key, value in interval['state_residuals'].items())
        witnesses = held['valid_control_intervals']
        longest, beginning, last = 0., None, None
        for witness in witnesses:
            index = witness['control_index']
            row = execution[index]
            assert witness['end_s'] == row['state']['time_s']
            assert abs(witness['end_s']-witness['start_s']-tasks['vehicle']['dt_s']) < 1e-7
            if last is None or index != last+1:
                beginning = witness['start_s']
            duration = witness['end_s']-beginning
            assert abs(witness['held_s']-duration) < 1e-7
            longest = max(longest, duration)
            last = index
        assert abs(longest-held['longest_whole_interval_hold_s']) < 1e-7
        assert held['certified'] == (longest >= 2.-1e-8)
        assert audit['reported_arrival_without_full_certificate'] == (record['actual_arrival'] and not audit['conservative_hold_certificate'])
        if audit['reported_arrival_without_full_certificate']:
            unresolved.append(dict(index=i, task=record['task'], variant=record['variant'], hold_s=longest))
        controls += len(execution)
        moving += direction['moving_intervals']
        counts['body_'+body['status']] += 1
        counts['direction_'+direction['status']] += 1
        counts['reported_arrivals'] += audit['reported_arrival']
        counts['certified_reported_arrivals'] += audit['reported_arrival'] and audit['conservative_hold_certificate']
        files[str(certificate)] = sha(certificate)
        files[str(directory/'report.json')] = sha(directory/'report.json')
        files[str(directory/'initial_state_proof.json')] = sha(directory/'initial_state_proof.json')
    save(destination/'input_prefix.json', dict(rows=audits,
        sha256=hashlib.sha256(data).hexdigest(), complete_prefix_bytes=len(data)))
    save(destination/'files.json', files)
    save(destination/'report.json', dict(passed=True, updated_utc=datetime.now(timezone.utc).isoformat(),
        audited_prefix_cases=len(audits), expected_cohort_cases=450, full_cohort_verified=len(audits) == 450,
        source_progress_can_continue=True, strict_observed_initials=strict, reconstructed_initials=rebuilt,
        control_intervals=controls, moving_intervals=moving, counts=dict(counts),
        unresolved_reported_arrivals=unresolved, source_release_sha256=sha(source/'release.json'),
        audited_prefix_sha256=hashlib.sha256(data).hexdigest(), SHA_files=len(files),
        files_manifest_sha256=sha(destination/'files.json'), source_sha256=sha(__file__),
        original_result_values_unchanged=True, actual_model_calls=0, new_executions=0, new_arrivals=0,
        initial_RNG_independent_crosscheck='explicit PCG64 standard_normal(3) and explicit scales',
        conservative_hold_times_recomputed_from_saved_interval_witnesses=True,
        safety_status_is_not_assumed_from_passed_ledger=True, goal_complete=False))
    print(json.dumps(read(destination/'report.json'), ensure_ascii=False))


if __name__ == '__main__':
    main()
