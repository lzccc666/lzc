"""主450和独立车身审核完整后，仅CPU审核全部保存方向；排队不算完成。"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path('/root/autodl-tmp/lizichong')
OUT = PROJECT/'outputs/project_v2/thesis_completion_v1'
ROOT = OUT/'vehicle_execution/execution_direction_after_body_001'
ROOT.mkdir(exist_ok=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def ticks(pid):
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[19])
    except FileNotFoundError:
        return None


main = OUT/'interactive_simulation/realtime_three_model_full_001'
body_waiter = OUT/'vehicle_execution/independent_body_after_main_001'
body_result = OUT/'vehicle_execution/independent_body_main_full_001'
driver = OUT/'source_bundle/execution_direction_audit_001.py'
frozen = {str(p.relative_to(PROJECT)): sha(p) for p in [Path(__file__).resolve(), driver,
    PROJECT/'conceptgraph/completion/execution_direction_audit.py', OUT/'task_protocol/execution_direction_audit_001.md']}
assert ticks(224212) is not None
release = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=frozen,
    main_release_sha256=sha(main/'release.json'), parent_body_waiter_release_sha256=sha(body_waiter/'release.json'),
    body_waiter_pid=224212, body_waiter_start_ticks=ticks(224212), expected_cases=450,
    wait_budget_s=176400, child_budget_s=900, budget_bytes=1024**3, minimum_free_bytes=6*1024**3,
    run_id='execution_direction_main_full_001', CUDA_visible='', shared_CPU=True,
    queued_is_not_completed=True, new_model_calls=0, new_executions=0, goal_complete=False)
save(ROOT/'release.json', release)
print(json.dumps(dict(pid=os.getpid(), waiting_for=str(body_waiter/'complete.json'), goal_complete=False)), flush=True)
started = time.monotonic()
try:
    while True:
        completed = None
        try:
            completed = read(body_waiter/'complete.json')
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if completed is not None:
            break
        if time.monotonic()-started > release['wait_budget_s']:
            raise TimeoutError('前置未在预算内完整，保留等待失败')
        if ticks(224212) != release['body_waiter_start_ticks']:
            # 退出时再读一次，避免刚好位于成功回执与进程退出之间。
            completed = read(body_waiter/'complete.json')
            break
        time.sleep(5)
    assert completed['cases'] == 450
    assert sha(body_result/'report.json') == completed['report_sha256']
    assert sha(main/'release.json') == release['main_release_sha256']
    summary = read(main/'summary.json')
    assert summary['attempted'] == summary['expected'] == 450
    assert sha(main/'trial_records.jsonl') == summary['trial_records_sha256']
    assert all(sha(PROJECT/p) == value for p, value in frozen.items())
    for dependency in [main/'release.json', body_waiter/'release.json']:
        for path, value in read(dependency)['source_sha256'].items():
            assert sha(PROJECT/path) == value
    assert shutil.disk_usage(OUT).free > release['minimum_free_bytes']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    with (ROOT/'audit.log').open('x') as log:
        child = subprocess.run([sys.executable, str(driver), '--cohort', 'main', '--run-id', release['run_id']],
            cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=release['child_budget_s'])
    assert child.returncode == 0, '主方向审核未完整，保留失败，不自动重试'
    result = OUT/'vehicle_execution'/release['run_id']
    report = read(result/'report.json')
    assert report['cases'] == report['expected_cases'] == 450 and not report['is_smoke']
    assert report['actual_model_calls'] == report['new_executions'] == report['new_arrivals'] == 0
    save(ROOT/'complete.json', dict(cases=450, report_sha256=sha(result/'report.json'),
        status_counts=report['status_counts'], moving_cases=report['moving_cases'],
        previous_arrivals_with_body_and_direction=report['previous_arrivals_with_body_and_direction'],
        still_requires_independent_global_direction_and_ledger_verification=True,
        new_arrivals=0, goal_complete=False))
except BaseException as exc:
    save(ROOT/'interrupted.json', dict(error=repr(exc), goal_complete=False))
    raise
