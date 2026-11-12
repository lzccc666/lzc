"""等待同版过去LiDAR下游完成，执行原定义的录制故障与独立消融。"""
from pathlib import Path
import json
import os
import shutil
import signal
import subprocess
import sys
import time

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


destination = OUTPUT/'source_bundle/causal_ablations_full_001'
destination.mkdir(exist_ok=False)
modules = ['fault_robustness', 'fault_statistics', 'mapping_module_ablation',
    'mapping_module_motion_statistics', 'map_constraint_ablation', 'map_constraint_statistics',
    'goal_selection_ablation', 'selection_ablation', 'navigation_diagnostics', 'retrieval_statistics', 'stage9']
paths = [PROJECT/'conceptgraph/completion'/f'{name}.py' for name in modules]
paths += sorted((PROJECT/'conceptgraph/navigation').glob('*.py'))
paths += [PROJECT/'conceptgraph/scenegraph/relations.py',
    OUTPUT/'task_protocol/development_causal_ablations_001.md', Path(__file__)]
frozen = {str(path): digest(path) for path in paths}
for path in paths:
    target = destination/'frozen_source'/path.relative_to(PROJECT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
release = dict(source_sha256=frozen, wait_budget_s=21600, maximum_new_bytes=3*1024**3,
    minimum_free_bytes=6*1024**3, mapping_run='causal_mapping_full_001',
    retrieval_run='causal_retrieval_full_001', actual_model_calls=0,
    performance_credit=False, new_test_opened=False, goal_complete=False)
dump(destination/'release.json', release)
prefix = [sys.executable, '-m']
steps = [
    ('faults', 7200, prefix+['conceptgraph.completion.fault_robustness', '--run-id', 'causal_fault_full_001',
        '--mapping-run', 'causal_mapping_full_001', '--wall-budget-s', '7200']),
    ('fault_statistics', 900, prefix+['conceptgraph.completion.fault_statistics', '--input-run', 'causal_fault_full_001', '--run-id', 'causal_fault_full_001']),
    ('modules', 3600, prefix+['conceptgraph.completion.mapping_module_ablation', '--run-id', 'causal_modules_full_001',
        '--mapping-run', 'causal_mapping_full_001', '--baseline-fault-run', 'causal_fault_full_001', '--wall-budget-s', '3600']),
    ('motion_statistics', 900, prefix+['conceptgraph.completion.mapping_module_motion_statistics', '--input-run', 'causal_modules_full_001', '--run-id', 'causal_modules_motion_full_001']),
    ('maps', 7200, prefix+['conceptgraph.completion.map_constraint_ablation', '--run-id', 'causal_maps_full_001',
        '--retrieval-run', 'causal_retrieval_full_001', '--wall-budget-s', '7200']),
    ('map_statistics', 900, prefix+['conceptgraph.completion.map_constraint_statistics', '--input-run', 'causal_maps_full_001', '--run-id', 'causal_maps_full_001']),
    ('goals', 3600, prefix+['conceptgraph.completion.goal_selection_ablation', '--run-id', 'causal_goals_full_001',
        '--retrieval-run', 'causal_retrieval_full_001', '--wall-budget-s', '3600'])]
dump(destination/'commands.json', [dict(stage=name, wall_budget_s=budget, command=command) for name,budget,command in steps])
begun = time.monotonic()
previous = OUTPUT/'source_bundle/causal_downstream_full_001'
records = []
resource_started = False
events = OUTPUT/'interactive_simulation/realtime_three_model_full_001/resource_events.jsonl'
def resource(event):
    with events.open('a') as handle:
        handle.write(json.dumps(dict(event=event, time_unix_s=time.time(), device='cpu',
            GPU_calls=0, performance_credit=False))+'\n')
def unchanged():
    if any(digest(path) != sha for path, sha in frozen.items()):
        raise ValueError('消融等待或运行中源码变化')
try:
    while not (previous/'complete.json').exists():
        if list(previous.glob('interrupted_*.json')):
            raise RuntimeError('前置开发下游失败，不自动重试')
        if time.monotonic()-begun > release['wait_budget_s']:
            raise TimeoutError('达到前置结果等待预算')
        unchanged()
        time.sleep(5)
    resource('causal_ablations_start')
    resource_started = True
    folders = [OUTPUT/'faults/causal_fault_full_001', OUTPUT/'ablations/causal_modules_full_001',
        OUTPUT/'ablations/causal_maps_full_001', OUTPUT/'ablations/causal_goals_full_001']
    for name, budget, command in steps:
        unchanged()
        if shutil.disk_usage(OUTPUT).free < release['minimum_free_bytes']:
            raise OSError('低于6GiB磁盘保护线')
        started = time.monotonic()
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
                            raise TimeoutError(name+'达到墙钟预算')
                        size = sum(path.lstat().st_size for folder in folders if folder.exists()
                            for path in folder.rglob('*') if path.is_file())
                        if size > release['maximum_new_bytes'] or shutil.disk_usage(OUTPUT).free < release['minimum_free_bytes']:
                            raise OSError('达到消融存储保护线')
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
                raise
        record = dict(stage=name, returncode=code, elapsed_s=time.monotonic()-started)
        records.append(record)
        dump(destination/(name+'_exit.json'), record)
        if code:
            raise RuntimeError(name+'失败，不自动重试')
    unchanged()
    dump(destination/'complete.json', dict(stages=records, elapsed_s_including_wait=time.monotonic()-begun,
        actual_arrivals=0, new_test_opened=False, goal_complete=False))
except BaseException as error:
    dump(destination/f'interrupted_{time.time_ns()}.json', dict(error=repr(error), stages=records, goal_complete=False))
    raise
finally:
    if resource_started:
        resource('causal_ablations_end')
