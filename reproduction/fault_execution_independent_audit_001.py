"""故障闭环独立复核：连续整车、方向、目标身份及到达保持。"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
import time

OUTPUT = Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')
PROJECT = OUTPUT.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-run', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    source = OUTPUT / 'interactive_simulation' / args.input_run
    release = read(source / 'release.json')
    summary = read(source / 'summary.json')
    rows = [json.loads(s) for s in (source / 'trial_records.jsonl').read_text().splitlines()]
    assert len(rows) == len(release['trials']) == summary['attempted'] == summary['expected']
    assert sha(source / 'trial_records.jsonl') == summary['trial_records_sha256']
    task_path = OUTPUT / 'interactive_simulation/synthetic_tasks_v1.json'
    assert sha(task_path) == release['task_release_sha256']
    task_data = read(task_path)
    tasks = {t['task_id']: t for t in task_data['tasks']}
    configs = {c['config_id']: c for c in task_data['configs']}
    with tarfile.open(source / 'source_snapshot.tar.gz') as archive:
        initializer = archive.extractfile('conceptgraph/completion/realtime_navigation.py').read()
    assert hashlib.sha256(initializer).hexdigest() == release['source_sha256']['conceptgraph/completion/realtime_navigation.py']
    helper = OUTPUT / 'source_bundle/semantic_minus_execution_audit_001.py'
    expected = read(OUTPUT / 'vehicle_execution/semantic_minus_execution_full_002/report.json')['source_sha256']
    for name, value in expected.items():
        assert sha(PROJECT / name) == value
    spec = importlib.util.spec_from_file_location('independent_execution', helper)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    destination = OUTPUT / 'vehicle_execution' / args.run_id
    destination.mkdir(exist_ok=False)
    frozen = dict(expected, **{str(Path(__file__).relative_to(PROJECT)): sha(Path(__file__))})
    (destination / 'release.json').write_text(json.dumps(dict(source_run=args.input_run,
        expected=len(rows), sources=frozen, input_release_sha256=sha(source / 'release.json'),
        budget_wall_s=1800, budget_bytes=1_000_000_000, minimum_free_bytes=6_000_000_000,
        actual_model_calls=0, goal_complete=False), ensure_ascii=False, indent=2))
    begun = time.monotonic()
    outputs = []
    try:
        with (destination / 'cases.jsonl').open('x') as f:
            for index, (row, plan) in enumerate(zip(rows, release['trials'])):
                assert all(row[k] == v for k, v in plan.items())
                assert time.monotonic()-begun < 1800 and shutil.disk_usage(OUTPUT).free > 6_000_000_000
                result = auditor.case(row, index, destination, tasks, configs, task_data['vehicle'])
                outputs.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                if index % 20 == 0:
                    assert sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()) < 1_000_000_000
        for name, value in frozen.items():
            assert sha(PROJECT / name) == value
        counts = Counter(r['audit_status'] for r in outputs)
        unresolved = [r['index'] for r in outputs if r.get('reported_arrival_without_full_certificate')]
        report = dict(audit_completed=True, source_run=args.input_run, cases=len(outputs),
                      status_counts=dict(counts), reported_arrivals=sum(r['reported_arrival'] for r in outputs),
                      conservative_hold_certificates=sum(r['conservative_hold_certificate'] for r in outputs),
                      explicit_initial_reconstructed=sum(r.get('explicit_initial_reconstructed', False) for r in outputs),
                      unresolved_arrivals=unresolved, control_records=sum(r['saved_control_records'] for r in outputs),
                      cases_sha256=sha(destination / 'cases.jsonl'), sources=frozen,
                      initial_reconstruction_is_not_observation=True, single_lane_connection_accuracy=None,
                      dynamic_traffic_safety_not_proven=True, new_model_calls=0, new_arrival_credit=0,
                      execution_wall_s=time.monotonic()-begun, goal_complete=False)
        (destination / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False))
    except BaseException as error:
        (destination / 'interrupted.json').write_text(json.dumps(dict(error=repr(error),
            audited_cases=len(outputs), goal_complete=False), ensure_ascii=False, indent=2))
        raise


if __name__ == '__main__':
    main()
