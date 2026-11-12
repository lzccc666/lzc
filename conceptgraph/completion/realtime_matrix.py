"""冻结50任务三种子的完整真实时钟矩阵；逐进程隔离，保留失败，可核验续跑。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tarfile
import time

from conceptgraph.completion.realtime_batch import pipeline_sources, verify_sources
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


PRESETS = {
    'main': [('rule', 'rule', ''), ('base_full', 'base', 'query_parse,object_attributes,candidate_rerank'),
             ('new_full', 'new', 'query_parse,object_attributes,candidate_rerank')],
    'semantic-minus': [('new_minus_parse', 'new', 'object_attributes,candidate_rerank'),
                       ('new_minus_attributes', 'new', 'query_parse,candidate_rerank'),
                       ('new_minus_rerank', 'new', 'query_parse,object_attributes')],
}
PROTOCOL = dict(maximum_wall_s=90, semantic_ttl_s=30, sensor_ttl_s=1.5,
                candidate_plan_budget_s=10, maximum_candidates=3, plan_reuse=True,
                subprocess_timeout_s=240)


def trial_plan(tasks, seeds, preset, run_id):
    """按身份构造笛卡尔积；随机顺序与方法结果无关。"""
    ids = [t['task_id'] for t in tasks]
    if len(ids) != len(set(ids)) or len(seeds) != len(set(seeds)):
        raise ValueError('duplicate task or seed would inflate the denominator')
    rows = [dict(variant=variant, model=model, semantic_tasks=enabled, task=t['task_id'],
                 config_id=t['config_id'], kind=t['kind'], seed=seed,
                 run_id=f"{run_id}_{variant}_{t['task_id'].replace('-', '_')}_s{seed}")
            for (variant, model, enabled), t, seed in itertools.product(PRESETS[preset], tasks, seeds)]
    random.Random(17).shuffle(rows)
    return rows


def verified_records(dest, release):
    """已完成失败也不重试；断电时未记账的试验只能记为中断。"""
    path = dest/'trial_records.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    expected = {r['run_id']: r for r in release['trials']}
    seen = set()
    for row in rows:
        key = row['run_id']
        if key in seen or key not in expected or any(row.get(k) != v for k, v in expected[key].items()):
            raise ValueError('resume record is duplicated or differs from frozen trial')
        seen.add(key)
        if row['result_sha256'] is not None:
            if digest(Path(row['raw_case'])/'result.json') != row['result_sha256']:
                raise ValueError('saved trial result changed')
    return rows


def bytes_in(paths):
    return sum(p.stat().st_size for root in paths if root.exists()
               for p in root.rglob('*') if p.is_file())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--preset', choices=PRESETS, default='main')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wall-budget-s', type=float, default=86400.)
    parser.add_argument('--storage-budget-gb', type=float, default=20.)
    parser.add_argument('--minimum-free-gb', type=float, default=6.)
    args = parser.parse_args()
    dest = OUTPUT/'interactive_simulation'/args.run_id
    release_path = OUTPUT/'interactive_simulation/synthetic_tasks_v1.json'
    selected_path = OUTPUT/'training/selected_three_task_v2_model_001.json'
    tasks = json.loads(release_path.read_text())
    if len(tasks['tasks']) != 50 or len(tasks['configs']) != 10 or len(tasks['seeds']) != 3:
        raise ValueError('required 50 tasks / 10 configurations / three seeds not found')
    if Counter(t['kind'] for t in tasks['tasks']) != dict(reachable_positive=30, no_match=10, inaccessible=10):
        raise ValueError('frozen task type denominator differs')
    if args.resume:
        release = json.loads((dest/'release.json').read_text())
        if release['preset'] != args.preset:
            raise ValueError('resume preset differs')
    else:
        dest.mkdir(parents=True, exist_ok=False)
        sources = sorted(set(pipeline_sources()+[Path(__file__).resolve()]))
        training = OUTPUT/'training/three_task_v2_formal_seed17_001/training_report.json'
        adapter = Path(json.loads(training.read_text())['best_checkpoint']['path'])
        release = dict(preset=args.preset,
            trials=trial_plan(tasks['tasks'], tasks['seeds'], args.preset, args.run_id),
            task_release_sha256=digest(release_path), model_selection_sha256=digest(selected_path),
            adapter_weights_path=str(adapter/'adapter_model.safetensors'),
            adapter_weights_sha256=digest(adapter/'adapter_model.safetensors'),
            source_sha256={str(p.relative_to(PROJECT)): digest(p) for p in sources},
            protocol=PROTOCOL, wall_budget_s=args.wall_budget_s,
            storage_budget_bytes=int(args.storage_budget_gb*1e9), minimum_free_bytes=int(args.minimum_free_gb*1e9),
            frozen_utc=datetime.now(timezone.utc).isoformat(),
            design='全部冻结50任务×3固定种子×3条件；每试验独立进程，同前端/执行器/候选验证/时钟预算',
            independent_test=False, natural_language_distribution=False,
            configuration_dependence='10配置共享3种道路家族；正任务仅颜色表达，不代表复杂关系任务',
            performance_credit=False, cold_start_in_process_wall_only=True,
            retry='不自动重试；失败/无结果留在450分母；中断续跑不得重新选择已尝试的任务',
            goal_complete=False)
        dump(dest/'release.json', release)
        with tarfile.open(dest/'pipeline_snapshot.tar.gz', 'x:gz') as archive:
            for source in sources:
                archive.add(source, arcname=str(source.relative_to(PROJECT)))
    if digest(release_path) != release['task_release_sha256'] or digest(selected_path) != release['model_selection_sha256']:
        raise ValueError('task/model release changed')
    if digest(Path(release['adapter_weights_path'])) != release['adapter_weights_sha256']:
        raise ValueError('adapter weights changed')
    verify_sources(release['source_sha256'])
    rows = verified_records(dest, release)
    seen = {r['run_id'] for r in rows}
    started = time.monotonic()
    elapsed_previous = sum(r.get('process_wall_s', 0.) for r in rows)
    env = dict(os.environ, LD_LIBRARY_PATH='/root/autodl-fs/third_party/graphics_runtime/lib:/root/miniconda3/envs/conceptgraph/lib',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYOPENGL_PLATFORM='egl', PYTHONFAULTHANDLER='1')
    with (dest/'trial_records.jsonl').open('a') as handle:
        for index, trial in enumerate(release['trials']):
            if trial['run_id'] in seen:
                continue
            verify_sources(release['source_sha256'])
            roots = [dest]+[OUTPUT/folder/trial['run_id'] for folder in ['interactive_simulation', 'cache_audit']]
            # 前次试验的实际体积逐项累计；检查当前试验目录，避免续跑重复。
            storage = (sum(r.get('saved_bytes', 0) for r in rows)+bytes_in(roots))
            free = shutil.disk_usage(OUTPUT).free
            elapsed = elapsed_previous+time.monotonic()-started
            if (elapsed > release['wall_budget_s'] or storage > release['storage_budget_bytes']
                    or free < release['minimum_free_bytes']):
                dump(dest/'budget_stop.json', dict(attempted=len(rows), expected=len(release['trials']),
                    elapsed_s=elapsed, saved_bytes=storage, free_bytes=free, goal_complete=False))
                raise RuntimeError('frozen time/storage budget reached; keep all attempted results')
            gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, check=True).stdout.strip()
            if gpu:
                dump(dest/'gpu_busy.json', dict(pids=gpu, next_trial=trial, goal_complete=False))
                raise RuntimeError('another CUDA process present; do not compete')
            raw = OUTPUT/'interactive_simulation'/trial['run_id']
            case = raw/trial['task']/f"seed{trial['seed']}"
            begun = time.monotonic()
            timeout = False
            if raw.exists():
                # 上次进程启动后主进程中断；保存已有结果或明确记为中断，不再调用模型。
                returncode = None
                status_override = 'PREVIOUS_ATTEMPT_INTERRUPTED'
            else:
                status_override = None
                command = [sys.executable, '-m', 'conceptgraph.completion.realtime_navigation',
                    '--run-id', trial['run_id'], '--model', trial['model'], '--task', trial['task'],
                    '--seed', str(trial['seed']), '--semantic-tasks', trial['semantic_tasks']]
                for flag, key in [('maximum-wall-s', 'maximum_wall_s'), ('semantic-ttl-s', 'semantic_ttl_s'),
                                  ('sensor-ttl-s', 'sensor_ttl_s'), ('candidate-plan-budget-s', 'candidate_plan_budget_s'),
                                  ('maximum-candidates', 'maximum_candidates')]:
                    command += ['--'+flag, str(release['protocol'][key])]
                with (dest/f'{index:03d}.log').open('x') as log:
                    try:
                        process = subprocess.run(command, cwd=PROJECT, env=env, stdout=log,
                            stderr=subprocess.STDOUT, timeout=release['protocol']['subprocess_timeout_s'])
                        returncode = process.returncode
                    except subprocess.TimeoutExpired:
                        returncode, timeout = None, True
            path = case/'result.json'
            result = json.loads(path.read_text()) if path.exists() else None
            row = dict(**trial, raw_case=str(case), returncode=returncode, process_timeout=timeout,
                process_wall_s=time.monotonic()-begun, result=result,
                status=result['status'] if result else status_override or 'PROCESS_FAILED_NO_RESULT',
                result_sha256=digest(path) if result else None,
                actual_arrival=bool(result and result['actual_full_perception_arrival']),
                saved_bytes=bytes_in([raw, OUTPUT/'cache_audit'/trial['run_id']]))
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            handle.flush()
            print(json.dumps(dict(index=index, variant=trial['variant'], task=trial['task'],
                                  seed=trial['seed'], status=row['status'], actual_arrival=row['actual_arrival']),
                             ensure_ascii=False), flush=True)
    dump(dest/'summary.json', dict(attempted=len(rows), expected=len(release['trials']),
        processes_with_result=sum(r['result'] is not None for r in rows),
        process_failures=sum(r['returncode'] != 0 or r['process_timeout'] for r in rows),
        trial_records_sha256=digest(dest/'trial_records.jsonl'),
        elapsed_s=elapsed_previous+time.monotonic()-started, saved_trial_bytes=sum(r['saved_bytes'] for r in rows),
        full_frozen_development_matrix_finished=len(rows) == len(release['trials']),
        independent_test=False, stage11_complete=False, stage12_complete=False, goal_complete=False))


if __name__ == '__main__':
    main()
