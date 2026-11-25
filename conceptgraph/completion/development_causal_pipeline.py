"""有界顺序执行开发集过去LiDAR转换、独立审计、重建和统计。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not args.run_id.replace('_', '').replace('-', '').isalnum():
        raise ValueError('非法run-id')
    source_run = 'development_causal_inputs_'+args.run_id
    binding_run = 'causal_frontend_reuse_'+args.run_id
    mapping_run = 'causal_mapping_'+args.run_id
    destination = OUTPUT/'source_bundle'/('development_causal_pipeline_'+args.run_id)
    destination.mkdir(exist_ok=False)
    shutil.copy2(__file__, destination/'pipeline_source.py')
    sources = [Path(__file__).with_name(f'development_causal_{name}.py')
               for name in ['inputs', 'audit', 'mapping', 'statistics', 'pipeline']]
    sources += [PROJECT/'conceptgraph/completion'/name for name in
        ['stage9.py', 'causal_input_audit.py', 'natural_inputs.py', 'reserved_causal_inputs.py',
         'association.py', 'quality.py', 'depth.py', 'motion.py']]
    sources += [PROJECT/'conceptgraph/slam'/name for name in ['streamlined_mapping.py', 'mapping.py', 'utils.py']]
    sources += [PROJECT/'conceptgraph/hydra_configs/base_mapping.yaml',
        OUTPUT/'task_protocol/development_causal_mapping_001.md']
    frozen = {str(path): digest(path) for path in sources}
    for path in sources:
        relative = path.relative_to(PROJECT)
        target = destination/'frozen_source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    release = dict(source_sha256=frozen, input_run=source_run, frontend_run=binding_run,
        mapping_run=mapping_run, input_protocol_sha256=digest(OUTPUT/'task_protocol/development_causal_mapping_001.md'),
        sequential_stages=True, CPU_threads=2, actual_new_frontend_calls=0,
        conversion_storage_limit_bytes=3*1024**3, mapping_storage_limit_bytes=3*1024**3,
        minimum_free_bytes=6*1024**3, stage_wall_budgets_s=[7200, 1800, 10800, 600],
        new_test_opened=False, performance_credit=False, goal_complete=False)
    dump(destination/'release.json', release)
    steps = [
        ('convert', 7200, [sys.executable, '-m', 'conceptgraph.completion.development_causal_inputs',
            '--run-id', source_run, '--wall-budget-s', '7200', '--storage-budget-gb', '3', '--minimum-free-gb', '6']),
        ('audit', 1800, [sys.executable, '-m', 'conceptgraph.completion.development_causal_audit', '--input-run', source_run]),
        ('mapping', 10800, [sys.executable, '-m', 'conceptgraph.completion.development_causal_mapping',
            '--input-run', source_run, '--frontend-run', binding_run, '--mapping-run', mapping_run, '--execute']),
        ('statistics', 600, [sys.executable, '-m', 'conceptgraph.completion.development_causal_statistics',
            '--input-run', source_run, '--mapping-run', mapping_run, '--run-id', mapping_run])]
    events = OUTPUT/'interactive_simulation/realtime_three_model_full_001/resource_events.jsonl'
    def record_resource(event):
        if events.exists():
            with events.open('a') as handle:
                handle.write(json.dumps(dict(event=event, pipeline=args.run_id, time_unix_s=time.time(),
                    device='cpu', GPU_calls=0, performance_credit=False))+'\n')
    started = time.monotonic()
    record_resource('development_causal_pipeline_start')
    try:
        for name, wall_s, command in steps:
            if any(digest(path) != sha for path, sha in frozen.items()):
                raise ValueError('流水线源码变化，保留结果并停止')
            if shutil.disk_usage(OUTPUT).free < release['minimum_free_bytes']:
                raise OSError('磁盘低于保护线')
            stage_started = time.monotonic()
            dump(destination/(name+'_command.json'), dict(command=command, wall_budget_s=wall_s))
            print(json.dumps(dict(stage=name, status='starting', elapsed_s=stage_started-started)), flush=True)
            with (destination/(name+'.log')).open('x') as log:
                process = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                try:
                    while True:
                        try:
                            code = process.wait(timeout=5)
                            break
                        except subprocess.TimeoutExpired:
                            if time.monotonic()-stage_started > wall_s:
                                raise TimeoutError(f'{name}达到事前预算')
                            if shutil.disk_usage(OUTPUT).free < release['minimum_free_bytes']:
                                raise OSError('磁盘低于保护线')
                            if name == 'mapping':
                                folder = OUTPUT/'mapping_evaluation'/mapping_run
                                size = sum(path.lstat().st_size for path in folder.rglob('*') if path.is_file()) if folder.exists() else 0
                                if size > release['mapping_storage_limit_bytes']:
                                    raise OSError('重建达到事前存储预算')
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
                    raise
            dump(destination/(name+'_exit.json'), dict(returncode=code, elapsed_s=time.monotonic()-stage_started))
            if code:
                raise RuntimeError(f'{name}失败，退出码{code}，不自动重试')
        if any(digest(path) != sha for path, sha in frozen.items()):
            raise ValueError('流水线源码变化')
        report = OUTPUT/'statistics'/mapping_run/'report.json'
        dump(destination/'complete.json', dict(completed_stages=[s[0] for s in steps],
            statistics=str(report), statistics_sha256=digest(report), elapsed_s=time.monotonic()-started,
            full_development_causal_mapping_completed=True, new_test_opened=False, goal_complete=False))
        print(json.dumps(dict(pipeline_completed=True, statistics=str(report), goal_complete=False)), flush=True)
    except BaseException as exc:
        dump(destination/f'interrupted_{time.time_ns()}.json', dict(error=repr(exc),
            elapsed_s=time.monotonic()-started, goal_complete=False))
        raise
    finally:
        record_resource('development_causal_pipeline_end')


if __name__ == '__main__':
    main()
