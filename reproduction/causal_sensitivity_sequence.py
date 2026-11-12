"""预注册过去LiDAR开发敏感性：先实际离线选择，再等待同源故障基线重建。"""
from pathlib import Path
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=['smoke', 'full'], required=True)
    args = parser.parse_args()
    suffix = args.preset+'_001'
    destination = OUTPUT/'source_bundle'/('causal_sensitivity_'+suffix)
    destination.mkdir(exist_ok=False)
    mapping = 'causal_mapping_'+suffix
    baseline = 'causal_fault_'+suffix
    retrieval = 'causal_retrieval_scene0013_smoke_001' if args.preset == 'smoke' else 'causal_retrieval_full_001'
    sources = [Path(__file__), OUTPUT/'task_protocol/development_causal_sensitivity_001.md']
    sources += [PROJECT/'conceptgraph/completion'/f'{name}.py' for name in [
        'selection_parameters', 'development_selection_ablation', 'selection_ablation', 'retrieval_statistics',
        'road_group_statistics', 'map_constraint_ablation', 'mapping_parameters', 'mapping_parameter_statistics',
        'mapping_module_ablation', 'fault_robustness', 'fault_statistics', 'navigation_diagnostics',
        'stage9', 'association', 'depth', 'quality', 'motion']]
    sources += sorted((PROJECT/'conceptgraph/slam').glob('*.py'))
    sources += [PROJECT/'conceptgraph/navigation'/name for name in ['semantic.py', 'types.py']]
    sources += [PROJECT/'conceptgraph/scenegraph/relations.py', PROJECT/'conceptgraph/hydra_configs/base_mapping.yaml']
    frozen = {str(path): digest(path) for path in sources}
    dump(destination/'release.json', dict(source_sha256=frozen, preset=args.preset, mapping_run=mapping,
        retrieval_run=retrieval, baseline_fault_run=baseline, maximum_new_bytes=512*1024**2,
        minimum_free_bytes=6*1024**3, wait_budget_s=21600, CPU_threads=2, model_calls=0,
        performance_credit=False, independent_test=False, goal_complete=False))
    for path in sources:
        target = destination/'frozen_source'/path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    base = [sys.executable, '-m']
    short = ['--scene', 'scene-0012'] if args.preset == 'smoke' else []
    steps = [
        ('selection_parameters', 600, base+['conceptgraph.completion.selection_parameters',
            '--run-id', 'causal_selection_parameters_'+suffix, '--retrieval-run', retrieval],
            OUTPUT/'performance'/('causal_selection_parameters_'+suffix)),
        ('selection_ablation', 600, base+['conceptgraph.completion.development_selection_ablation',
            '--run-id', 'causal_selection_ablation_'+suffix, '--retrieval-run', retrieval],
            OUTPUT/'ablations'/('causal_selection_ablation_'+suffix)),
        ('road_groups', 300, base+['conceptgraph.completion.road_group_statistics',
            '--run-id', 'causal_road_groups_'+suffix, '--retrieval-run', retrieval],
            OUTPUT/'performance'/('causal_road_groups_'+suffix)),
        ('mapping_parameters', 3600, base+['conceptgraph.completion.mapping_parameters',
            '--run-id', 'causal_mapping_parameters_'+suffix, '--mapping-run', mapping,
            '--baseline-fault-run', baseline, '--wall-budget-s', '3600']+short,
            OUTPUT/'performance'/('causal_mapping_parameters_'+suffix)),
        ('mapping_statistics', 900, base+['conceptgraph.completion.mapping_parameter_statistics',
            '--input-run', 'causal_mapping_parameters_'+suffix, '--run-id', 'causal_mapping_parameters_'+suffix],
            OUTPUT/'statistics'/('causal_mapping_parameters_'+suffix))]
    events = OUTPUT/'interactive_simulation/realtime_three_model_full_001/resource_events.jsonl'
    def resource(event):
        with events.open('a') as handle:
            handle.write(json.dumps(dict(event=event, time_unix_s=time.time(), run_id=destination.name,
                device='cpu', GPU_calls=0, performance_credit=False))+'\n')
    def unchanged():
        if any(digest(p) != sha for p, sha in frozen.items()):
            raise ValueError('敏感性试验冻结源码改变')
    records = []
    resource('causal_sensitivity_start')
    try:
        for name, budget, command, folder in steps:
            unchanged()
            if name == 'mapping_parameters':
                waited = time.monotonic()
                dependency = OUTPUT/'faults'/baseline
                dump(destination/'baseline_wait.json', dict(dependency=str(dependency), budget_s=21600))
                while not (dependency/'summary.json').exists():
                    if (dependency/'interrupted.json').exists() or time.monotonic()-waited > 21600:
                        raise RuntimeError('同源故障基线失败或等待超时，不替换其他输入')
                    unchanged()
                    time.sleep(5)
            if shutil.disk_usage(OUTPUT).free < 6*1024**3:
                raise OSError('磁盘低于6GiB保护线')
            started = time.monotonic()
            dump(destination/(name+'_command.json'), dict(command=command, wall_budget_s=budget))
            print(json.dumps(dict(stage=name, status='starting')), flush=True)
            with (destination/(name+'.log')).open('x') as log:
                process = subprocess.Popen(command, cwd=PROJECT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    while True:
                        try:
                            code = process.wait(timeout=5)
                            break
                        except subprocess.TimeoutExpired:
                            size = sum(p.stat().st_size for _, _, _, output in steps if output.exists()
                                for p in output.rglob('*') if p.is_file())
                            if (time.monotonic()-started > budget or size > 512*1024**2
                                    or shutil.disk_usage(OUTPUT).free < 6*1024**3):
                                raise RuntimeError('时间或存储预算达到上限，保留失败前缀')
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
                    raise
            record = dict(stage=name, returncode=code, elapsed_s=time.monotonic()-started, output=str(folder))
            records.append(record)
            dump(destination/(name+'_exit.json'), record)
            if code:
                raise RuntimeError(f'{name}失败，不自动重试')
        unchanged()
        dump(destination/'complete.json', dict(stages=records, new_test_opened=False,
            actual_arrivals=0, performance_credit=False, goal_complete=False))
    except BaseException as exc:
        dump(destination/f'interrupted_{time.time_ns()}.json', dict(error=repr(exc), stages=records, goal_complete=False))
        raise
    finally:
        resource('causal_sensitivity_end')


if __name__ == '__main__':
    main()
