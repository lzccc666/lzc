"""同版过去LiDAR建图完成后顺序重做开发关系、上限诊断与四方法检索。"""
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
    parser.add_argument('--mapping-run', required=True)
    parser.add_argument('--wait-for-pipeline')
    parser.add_argument('--wait-budget-s', type=float, default=21000)
    args = parser.parse_args()
    for value in [args.run_id, args.mapping_run, args.wait_for_pipeline or 'none']:
        if not value.replace('_', '').replace('-', '').isalnum():
            raise ValueError('非法run-id')
    destination = OUTPUT/'source_bundle'/('causal_downstream_'+args.run_id)
    destination.mkdir(exist_ok=False)
    sources = [Path(__file__)]
    sources += [Path(__file__).with_name(name+'.py') for name in
        ['relations_audit', 'retrieval_baselines', 'retrieval_statistics', 'navigation_diagnostics', 'depth', 'quality']]
    sources += [PROJECT/'conceptgraph/scenegraph'/name for name in ['relations.py', 'incremental_graph.py']]
    sources += [PROJECT/'conceptgraph/navigation'/name for name in ['semantic.py', 'types.py']]
    protocol = OUTPUT/'task_protocol/development_causal_downstream_001.md'
    sources.append(protocol)
    frozen = {str(path): digest(path) for path in sources}
    for path in sources:
        target = destination/'frozen_source'/path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    release = dict(source_sha256=frozen, mapping_run=args.mapping_run, expected_frontend_model_calls=0,
        wait_for_pipeline=args.wait_for_pipeline, wait_budget_s=args.wait_budget_s,
        maximum_stage_wall_s=[900, 900, 1800, 600], minimum_free_bytes=6*1024**3,
        maximum_new_physical_bytes=1024**3, only_development=True, CPU_threads=2,
        actual_arrivals=0, performance_credit=False, new_test_opened=False, goal_complete=False)
    dump(destination/'release.json', release)
    begun = time.monotonic()
    events = OUTPUT/'interactive_simulation/realtime_three_model_full_001/resource_events.jsonl'
    def unchanged():
        if any(digest(path) != sha for path, sha in frozen.items()):
            raise ValueError('等待或运行中源码改变，不自动转用新版本')
    def resource(event):
        if events.exists():
            with events.open('a') as handle:
                handle.write(json.dumps(dict(event=event, time_unix_s=time.time(), run_id=args.run_id,
                    device='cpu', GPU_calls=0, performance_credit=False))+'\n')
    records = []
    try:
        if args.wait_for_pipeline:
            pipeline = OUTPUT/'source_bundle'/('development_causal_pipeline_'+args.wait_for_pipeline)
            if not (pipeline/'release.json').exists():
                raise ValueError('依赖流水线不存在')
            dump(destination/'wait_started.json', dict(pipeline=str(pipeline),
                pipeline_release_sha256=digest(pipeline/'release.json'), budget_s=args.wait_budget_s))
            while not (pipeline/'complete.json').exists():
                if list(pipeline.glob('interrupted_*.json')):
                    raise RuntimeError('依赖流水线失败，保留证据并停止等待')
                if time.monotonic()-begun > args.wait_budget_s:
                    raise TimeoutError('达到等待预算')
                unchanged()
                time.sleep(5)
            completed = json.loads((pipeline/'complete.json').read_text())
            if not completed['full_development_causal_mapping_completed']:
                raise ValueError('依赖结果不是完整开发重建')
            if digest(completed['statistics']) != completed['statistics_sha256']:
                raise ValueError('依赖统计结果指纹改变')
        summary_path = OUTPUT/'mapping_evaluation'/args.mapping_run/'summary.json'
        summary = json.loads(summary_path.read_text())
        if summary['scene_count'] != 38 or summary['frames'] != 1508:
            raise ValueError('正式下游要求全38开发场景1508帧，不剔除失败')
        mapping_sha = digest(summary_path)
        base = [sys.executable, '-m']
        steps = [
            ('relations', 900, base+['conceptgraph.completion.relations_audit', '--mapping-run', args.mapping_run,
                '--run-id', 'causal_relations_'+args.run_id, '--method', 'one_to_one_depth_quality',
                '--directional-mode', 'all_supported_axes'], OUTPUT/'mapping_evaluation'/('causal_relations_'+args.run_id)),
            ('selection', 900, base+['conceptgraph.completion.navigation_diagnostics', '--mapping-run', args.mapping_run,
                '--run-id', 'causal_selection_'+args.run_id, '--method', 'one_to_one_depth_quality'], OUTPUT/'replay'/('causal_selection_'+args.run_id)),
            ('retrieval', 1800, base+['conceptgraph.completion.retrieval_baselines', '--mapping-run', args.mapping_run,
                '--run-id', 'causal_retrieval_'+args.run_id, '--wall-budget-s', '1800'], OUTPUT/'main_experiments'/('causal_retrieval_'+args.run_id)),
            ('retrieval_statistics', 600, base+['conceptgraph.completion.retrieval_statistics',
                '--input-run', 'causal_retrieval_'+args.run_id, '--run-id', 'causal_retrieval_'+args.run_id], OUTPUT/'statistics'/('causal_retrieval_'+args.run_id))]
        resource('causal_downstream_start')
        output_folders = [step[3] for step in steps]
        for name, budget, command, output in steps:
            unchanged()
            if digest(summary_path) != mapping_sha:
                raise ValueError('完整建图汇总改变')
            if shutil.disk_usage(OUTPUT).free < release['minimum_free_bytes']:
                raise OSError('磁盘低于保护线')
            started = time.monotonic()
            dump(destination/(name+'_command.json'), dict(command=command, budget_s=budget))
            print(json.dumps(dict(stage=name, status='starting')), flush=True)
            with (destination/(name+'.log')).open('x') as log:
                process = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
                try:
                    while True:
                        try:
                            code = process.wait(timeout=5)
                            break
                        except subprocess.TimeoutExpired:
                            if time.monotonic()-started > budget:
                                raise TimeoutError(f'{name}达到预算')
                            if shutil.disk_usage(OUTPUT).free < release['minimum_free_bytes']:
                                raise OSError('磁盘低于保护线')
                            size = sum(p.lstat().st_size for folder in output_folders
                                if folder.exists() for p in folder.rglob('*') if p.is_file())
                            if size > release['maximum_new_physical_bytes']:
                                raise OSError('下游达到新增存储预算')
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
                    raise
            record = dict(stage=name, returncode=code, elapsed_s=time.monotonic()-started, output=str(output))
            dump(destination/(name+'_exit.json'), record)
            records.append(record)
            if code:
                raise RuntimeError(f'{name}失败，不自动重试')
        unchanged()
        dump(destination/'complete.json', dict(stages=records, mapping_summary_sha256=mapping_sha,
            total_wall_s_including_wait=time.monotonic()-begun, actual_arrivals=0,
            new_test_opened=False, goal_complete=False))
        print(json.dumps(dict(completed=True, actual_arrivals=0, goal_complete=False)), flush=True)
    except BaseException as exc:
        dump(destination/f'interrupted_{time.time_ns()}.json', dict(error=repr(exc), stages=records, goal_complete=False))
        raise
    finally:
        resource('causal_downstream_end')


if __name__ == '__main__':
    main()
