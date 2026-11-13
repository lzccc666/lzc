"""用冻结种子/初始化归档及真实静止前缀重建缺失零时刻观测；原审计只读。"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tarfile
import time

import numpy as np
from shapely.geometry import Polygon

from conceptgraph.completion.independent_continuous_body import audit_trajectory


PROJECT = Path('/root/autodl-tmp/lizichong')
OUT = PROJECT/'outputs/project_v2/thesis_completion_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def initial_state(config, seed):
    rng = np.random.default_rng(seed)
    xy = np.asarray(config['start_xy'])+rng.normal(0, .03, 2)
    return dict(x=float(xy[0]), y=float(xy[1]),
                yaw_rad=float(config['start_yaw_rad']+rng.normal(0, math.radians(.5))),
                speed_mps=0., steering_rad=0., time_s=0.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    assert args.limit is None or args.limit > 0
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    started = time.monotonic()
    source = OUT/'interactive_simulation/realtime_three_model_full_001'
    original = OUT/'vehicle_execution/independent_body_main_full_001'
    strict = read(original/'report.json')
    assert strict['cases'] == 450 and strict['status_counts'] == dict(
        certificate_safe=395, unknown_missing_initial_state=54, unknown_missing_complete_evidence=1)
    rows = [json.loads(line) for line in (original/'cases.jsonl').open()]
    delayed = [row for row in rows if row['status'] == 'unknown_missing_initial_state']
    plans = delayed[:args.limit] if args.limit is not None else rows
    main_release = read(source/'release.json')
    assert read(source/'summary.json')['attempted'] == 450
    task_path = OUT/'interactive_simulation/synthetic_tasks_v1.json'
    assert sha(task_path) == main_release['task_release_sha256']
    data = read(task_path)
    configs = {config['config_id']: config for config in data['configs']}
    module = PROJECT/'conceptgraph/completion/realtime_navigation.py'
    assert sha(module) == main_release['source_sha256'][str(module.relative_to(PROJECT))]
    with tarfile.open(source/'pipeline_snapshot.tar.gz') as archive:
        archived = archive.extractfile(str(module.relative_to(PROJECT))).read()
    assert archived == module.read_bytes()
    initialization = """    rng = np.random.default_rng(seed)
    state = VehicleState(*(np.asarray(config['start_xy'])+rng.normal(0,.03,2)),
        config['start_yaw_rad']+rng.normal(0,math.radians(.5)))"""
    assert initialization in archived.decode()
    destination = OUT/'vehicle_execution'/args.run_id
    destination.mkdir(exist_ok=False)
    frozen = {str(path.relative_to(PROJECT)): sha(path) for path in [Path(__file__).resolve(),
              PROJECT/'conceptgraph/completion/independent_continuous_body.py',
              OUT/'task_protocol/main_seed_initial_reconstruction_001.md']}
    save(destination/'release.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=frozen,
         task_release_sha256=sha(task_path), original_report_sha256=sha(original/'report.json'),
         original_cases_sha256=sha(original/'cases.jsonl'), main_release_sha256=sha(source/'release.json'),
         original_initialization_source_sha256=sha(module), source_snapshot_sha256=sha(source/'pipeline_snapshot.tar.gz'),
         initialization_excerpt_sha256=hashlib.sha256(initialization.encode()).hexdigest(),
         NumPy_version=np.__version__, RNG='NumPy default_rng / PCG64; archived initialization order',
         selected_limit=args.limit, expected_original_cases=450, delayed_cases=54,
         expected_output_cases=len(plans), wall_budget_s=900, storage_budget_bytes=1024**3,
         minimum_free_bytes=6*1024**3, CUDA_visible='', shared_CPU=True,
         posthoc_development_audit=True, reconstructed_is_not_observed_time_zero=True, goal_complete=False))
    outputs, counts = [], Counter()
    for record in plans:
        assert time.monotonic()-started < 900 and shutil.disk_usage(OUT).free > 6*1024**3
        assert all(sha(PROJECT/path) == value for path, value in frozen.items())
        original_index = rows.index(record)
        name = f"{original_index:04d}_{record['task']}_seed{record['seed']}_{record['method']}"
        parent = original/name
        output = dict(record)
        output['original_strict_report_sha256'] = sha(parent/'report.json')
        output['original_strict_status'] = record['status']
        if record['status'] == 'unknown_missing_complete_evidence':
            output['initial_state_provenance'] = 'unknown_missing_complete_result'
            counts['unknown_missing_complete_results'] += 1
        else:
            raw = Path(record['raw_case'])
            first = json.loads(next((raw/'perception.jsonl').open()))['sensor']['state']
            initial = initial_state(configs[record['config_id']], record['seed'])
            for path, value in record['source_sha256'].items():
                assert sha(path) == value
            if record['status'] == 'certificate_safe':
                assert first['time_s'] == 0.
                assert max(abs(first[field]-initial[field]) for field in initial) < 1e-12
                output['initial_state_provenance'] = 'saved_time_zero; prior_certificate_reused'
                output['reused_interval_audit_sha256'] = sha(parent/'interval_audit.json')
                counts['reused_saved_zero_certificates'] += 1
                counts['reused_intervals'] += record['audited_intervals']
            else:
                assert record['status'] == 'unknown_missing_initial_state' and first['time_s'] > 0
                execution = [json.loads(line) for line in (raw/'execution.jsonl').open()]
                prefix = [row for row in execution if row['state']['time_s'] <= first['time_s']]
                assert prefix and prefix[-1]['state'] == first
                assert abs(prefix[0]['state']['time_s']-data['vehicle']['dt_s']) < 1e-12
                for row in prefix:
                    state = row['state']
                    assert row['stale_sensor_braking'] and not row['stop_requested']
                    assert row['sensor_age_s'] is None and row['control']['acceleration_mps2'] == -2.
                    assert state['speed_mps'] == state['steering_rad'] == 0.
                    assert math.dist([state['x'], state['y']], [initial['x'], initial['y']]) < 1e-12
                    assert abs(math.remainder(state['yaw_rad']-initial['yaw_rad'], 2*math.pi)) < 1e-12
                geometry = read(parent/'geometry.json')
                audited = audit_trajectory(initial, execution, data['vehicle'], Polygon(geometry['road_polygon_xy']),
                                           [Polygon(points) for points in geometry['obstacles_xy']])
                save(destination/name/'initial_reconstruction.json', dict(
                    reconstructed_initial=initial, first_observed_state=first, stationary_prefix_records=len(prefix),
                    initial_not_a_saved_sensor_observation=True, raw_perception_sha256=sha(raw/'perception.jsonl'),
                    raw_execution_sha256=sha(raw/'execution.jsonl'), original_geometry_sha256=sha(parent/'geometry.json')))
                save(destination/name/'interval_audit.json', audited)
                save(destination/name/'geometry.json', geometry)
                output.update({field: value for field, value in audited.items() if field != 'intervals'})
                output['initial_state_provenance'] = 'frozen_seed_reconstruction_and_saved_stationary_prefix'
                output['moving_control_records'] = sum(row['state']['speed_mps'] > 1e-6 for row in execution)
                output['saved_control_records'] = len(execution)
                counts['reconstructed_initial_states'] += 1
                counts['reconstructed_prefix_control_records'] += len(prefix)
                counts['newly_audited_intervals'] += audited['audited_intervals']
        save(destination/name/'report.json', output)
        outputs.append(output)
        with (destination/'cases.jsonl').open('a') as handle:
            handle.write(json.dumps(output, ensure_ascii=False, allow_nan=False)+'\n')
    status_counts = dict(Counter(row['status'] for row in outputs))
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), cases=len(outputs), expected_cases=len(plans),
        is_smoke=args.limit is not None, counts=dict(counts), status_counts=status_counts,
        audited_intervals=sum(row['audited_intervals'] for row in outputs),
        certificate_safe_cases=sum(row['passed'] is True for row in outputs),
        previously_reported_arrivals=sum(row['prior_reported_arrival'] for row in outputs),
        previous_arrivals_with_continuous_body_certificate=sum(row['prior_reported_arrival'] and row['passed'] for row in outputs),
        original_strict_report_sha256=sha(original/'report.json'), source_sha256=frozen,
        original_strict_results_preserved=True, reconstructed_is_not_observed_time_zero=True,
        actual_model_calls=0, new_executions=0, new_arrivals=0, wall_CPU_shared_s=time.monotonic()-started,
        limits=['395个旧证书仅绑定复用；54个重建初态另外审计，不增加独立试验或到达',
                '初态来自冻结种子与归档初始化及保存静止前缀，不是新增零时刻观测',
                '方向与终点保持未在此获得信用；静态模型数值证书不是动态交通证明'], goal_complete=False)
    save(destination/'report.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
