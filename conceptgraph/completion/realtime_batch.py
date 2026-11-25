"""按冻结顺序逐进程执行三模型真实时钟开发smoke；保留每个失败且不并发Qwen。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import os
import random
import subprocess
import sys
import tarfile
import time

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


def pipeline_sources():
    names=['realtime_batch','realtime_navigation','interactive_navigation','navigation_semantics',
        'realtime_semantics','interactive_environment','render_probe','kinematic_benchmark','protocol',
        'stage9','semantic_contract','semantic_evaluation','depth','association','motion','quality']
    names.append('planning_reuse')
    selected={PROJECT/f'conceptgraph/completion/{name}.py' for name in names}
    for folder in ['navigation','scenegraph','slam','vlm','utils']:
        selected.update((PROJECT/'conceptgraph'/folder).rglob('*.py'))
    for name in ['streamlined_detections','run_incremental_semantic_navigation']:
        selected.add(PROJECT/f'conceptgraph/scripts/{name}.py')
    return sorted(selected)


def verify_sources(frozen):
    for name,sha in frozen.items():
        if digest(PROJECT/name)!=sha:
            raise ValueError('frozen execution dependency changed: '+name)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--after-evaluation-run')
    parser.add_argument('--wait-budget-s',type=float,default=3600.)
    parser.add_argument('--wall-budget-s',type=float,default=6000.)
    args=parser.parse_args()
    started=time.monotonic()
    dest=OUTPUT/'interactive_simulation'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    task_names=['config00-positive0','config00-no_match','config00-inaccessible']
    order=list(itertools.product(['rule','base','new'],task_names))
    random.Random(17).shuffle(order)
    plan=[dict(model=model,task=task,seed=17,run_id=f"{args.run_id}_{model}_{task.replace('-','_')}_seed17")
        for model,task in order]
    sources=pipeline_sources()
    frozen={str(p.relative_to(PROJECT)):digest(p) for p in sources}
    release=dict(trials=plan,source_sha256=frozen,
        task_release_sha256=digest(OUTPUT/'interactive_simulation/synthetic_tasks_v1.json'),
        model_selection_sha256=digest(OUTPUT/'training/selected_three_task_v2_model_001.json'),
        maximum_wall_s=90,semantic_ttl_s=30,sensor_ttl_s=1.5,candidate_plan_budget_s=10,
        maximum_candidates=3,subprocess_timeout_s=240,wait_budget_s=args.wait_budget_s,
        wall_budget_s=args.wall_budget_s,after_evaluation_run=args.after_evaluation_run,
        gpu_policy='wait for fair evaluation complete, then one child at a time; no concurrent Qwen process',
        performance_credit=False,cpu_shared_with_recorded_fault_run=True,
        design='三个事前指定任务组×3方法×seed17；开发smoke，不增加独立场景或替代50任务三种子全量',
        retry='每一冻结试验仅一次；进程失败、超时和无结果都保留，不自动重跑挑优',
        independent_test=False,gt_in_agent=False,goal_complete=False)
    dump(dest/'release.json',release)
    with tarfile.open(dest/'pipeline_snapshot.tar.gz','x:gz') as archive:
        for p in sources:
            archive.add(p,arcname=str(p.relative_to(PROJECT)))
    if args.after_evaluation_run:
        previous=OUTPUT/'main_experiments'/args.after_evaluation_run/'report.json'
        while not previous.exists():
            if time.monotonic()-started>min(args.wait_budget_s,args.wall_budget_s):
                dump(dest/'wait_failed.json',dict(reason='fair evaluation report not completed within wait budget',
                    previous=str(previous),goal_complete=False))
                raise TimeoutError('fair evaluation still incomplete; no second Qwen started')
            time.sleep(5)
        report=json.loads(previous.read_text())
        if report.get('sample_count')!=603 or len(report.get('variants',{}))!=5 or any(
                v.get('cache_replay_added_calls')!=0 for v in report['variants'].values()):
            raise ValueError('fair evaluation does not satisfy completed three-task release')
        # 模型释放发生在总报告写入之前；还要求CUDA进程列表为空，避免与其他工作竞争。
        dump(dest/'preceding_evaluation_verified.json',dict(path=str(previous),sha256=digest(previous),
            verified_utc=datetime.now(timezone.utc).isoformat()))
    env=dict(os.environ,LD_LIBRARY_PATH='/root/autodl-fs/third_party/graphics_runtime/lib:/root/miniconda3/envs/conceptgraph/lib',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYOPENGL_PLATFORM='egl',PYTHONFAULTHANDLER='1')
    records=[]
    with (dest/'trial_records.jsonl').open('x') as handle:
        for index,trial in enumerate(plan):
            verify_sources(frozen)
            if time.monotonic()-started>args.wall_budget_s:
                raise TimeoutError('batch wall budget; all completed records retained')
            idle_deadline=min(started+args.wall_budget_s,time.monotonic()+30.)
            while True:
                gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                    capture_output=True,text=True,check=True).stdout.strip()
                if not gpu or time.monotonic()>=idle_deadline:
                    break
                time.sleep(2)
            if gpu:
                dump(dest/f'gpu_busy_{index:02d}.json',dict(pids=gpu,trial=trial,goal_complete=False))
                raise RuntimeError('GPU has another process; do not start competing model')
            command=[sys.executable,'-m','conceptgraph.completion.realtime_navigation',
                '--run-id',trial['run_id'],'--model',trial['model'],'--task',trial['task'],
                '--seed','17','--maximum-wall-s','90','--semantic-ttl-s','30',
                '--candidate-plan-budget-s','10','--sensor-ttl-s','1.5']
            trial_started=time.monotonic()
            with (dest/f'{index:02d}_{trial["model"]}_{trial["task"]}.log').open('x') as log:
                try:
                    process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                        cwd=PROJECT,env=env,timeout=240)
                    returncode=process.returncode
                    timed_out=False
                except subprocess.TimeoutExpired:
                    returncode=None
                    timed_out=True
            case=OUTPUT/'interactive_simulation'/trial['run_id']/trial['task']/'seed17'
            result_path=case/'result.json'
            result=json.loads(result_path.read_text()) if result_path.exists() else None
            row=dict(**trial,returncode=returncode,process_timeout=timed_out,
                process_wall_s=time.monotonic()-trial_started,
                status=result['status'] if result is not None else 'PROCESS_FAILED_NO_RESULT',
                result=result,result_sha256=digest(result_path) if result_path.exists() else None,
                raw_case=str(case),actual_arrival=bool(result and result['actual_full_perception_arrival']))
            records.append(row)
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            handle.flush()
            print(json.dumps({k:v for k,v in row.items() if k!='result'},ensure_ascii=False),flush=True)
    dump(dest/'summary.json',dict(attempted=len(records),expected=len(plan),
        processes_with_result=sum(r['result'] is not None for r in records),
        process_failures=sum(r['returncode']!=0 or r['process_timeout'] for r in records),
        trial_records_sha256=digest(dest/'trial_records.jsonl'),elapsed_s=time.monotonic()-started,
        only_development_smoke=True,complete_three_model_navigation_main_table=False,
        stage11_complete=False,stage12_complete=False,goal_complete=False))


if __name__=='__main__':
    main()
