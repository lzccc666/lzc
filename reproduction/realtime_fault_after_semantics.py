"""主矩阵/语义缺一/CPU试跑均完整后，先六GPU试跑，再180完整故障；不并发模型。"""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import shutil
import subprocess
import sys
import time

from conceptgraph.completion.stage9 import PROJECT,OUTPUT,digest,dump
from conceptgraph.completion.realtime_batch import verify_sources


def main():
    root=OUTPUT/'faults/realtime_fault_after_semantics_001'
    root.mkdir(exist_ok=False)
    script=OUTPUT/'source_bundle/realtime_fault_matrix_001.py'
    runs=[('realtime_faults_gpu_smoke_001',True,2400,1),('realtime_faults_gpu_full_001',False,43200,8)]
    for run,smoke,wall,storage in runs:
        command=[sys.executable,str(script),'--run-id',run,'--prepare-only','--wall-budget-s',str(wall),'--storage-budget-gb',str(storage)]
        if smoke:command+=['--smoke']
        subprocess.run(command,cwd=PROJECT,check=True)
    frozen=json.loads((OUTPUT/'interactive_simulation'/runs[0][0]/'release.json').read_text())['source_sha256']
    release=dict(source_sha256=frozen,queue_source_sha256=digest(__file__),runs=runs,
        wait_budget_s=176400,minimum_free_bytes_before_GPU=16000000000,
        required_complete=['interactive_simulation/realtime_three_model_full_001/summary.json',
            'ablations/semantic_navigation_chain_001/complete.json','faults/realtime_fault_cpu_smoke_002/summary.json'],
        GPU_policy='只在前置全部完整且nvidia-smi无CUDA进程时逐子进程启动；不重试失败或重复活跃试验',
        independent_test=False,queued_trials_not_completed=True,goal_complete=False)
    dump(root/'release.json',release)
    started=time.monotonic()
    while not all((OUTPUT/p).exists() for p in release['required_complete']):
        if (OUTPUT/'faults/realtime_fault_cpu_smoke_002/interrupted.json').exists():
            dump(root/'wait_failed.json',dict(reason='CPU实际链试跑中断，不启动GPU故障扩展',goal_complete=False))
            raise RuntimeError('CPU smoke failed')
        if time.monotonic()-started>release['wait_budget_s']:
            dump(root/'wait_failed.json',dict(reason='前置任务未在冻结等待预算内完成',goal_complete=False))
            raise TimeoutError('preceding jobs incomplete')
        time.sleep(5)
    verify_sources(frozen)
    assert digest(__file__)==release['queue_source_sha256']
    main_summary=json.loads((OUTPUT/release['required_complete'][0]).read_text())
    semantic=json.loads((OUTPUT/release['required_complete'][1]).read_text())
    cpu=json.loads((OUTPUT/release['required_complete'][2]).read_text())
    assert main_summary['attempted']==main_summary['expected']==450
    assert semantic['new_trials']==450 and semantic['combined_trial_records']==750
    assert cpu['attempted']==cpu['expected']==6 and cpu['complete_input_output_smoke']
    if shutil.disk_usage(OUTPUT).free<release['minimum_free_bytes_before_GPU']:
        dump(root/'storage_stop.json',dict(free_bytes=shutil.disk_usage(OUTPUT).free,goal_complete=False))
        raise RuntimeError('cannot retain full GPU fault evidence with frozen disk margin')
    environment=dict(os.environ,LD_LIBRARY_PATH='/root/miniconda3/envs/conceptgraph/lib',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    environment.pop('CUDA_VISIBLE_DEVICES',None)
    for run,smoke,wall,storage in runs:
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
        if gpu:
            dump(root/'gpu_busy.json',dict(pids=gpu,run=run,goal_complete=False));raise RuntimeError('GPU occupied after preceding complete')
        command=[sys.executable,str(script),'--run-id',run,'--resume']
        if smoke:command+=['--smoke']
        with (root/(run+'.log')).open('x') as log:
            child=subprocess.run(command,cwd=PROJECT,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=wall+600)
        if child.returncode:
            dump(root/'child_failed.json',dict(run=run,returncode=child.returncode,goal_complete=False));raise RuntimeError('fault child failed; no automatic retry')
        target=OUTPUT/'interactive_simulation'/run
        summary=json.loads((target/'summary.json').read_text())
        assert summary['attempted']==summary['expected']==(6 if smoke else 180)
        if smoke:
            rows=list(map(json.loads,(target/'trial_records.jsonl').open()))
            assert all(r['returncode']==0 and r['result'] is not None and r['fault_exposure']['actual_detector_calls']>=2 for r in rows)
            blackout=next(r for r in rows if r['fault']=='sensor_blackout_4s')
            if not blackout['fault_exposure']['actual_fault_exposed'] or not blackout['fault_exposure']['blackout_control_ticks']:
                dump(root/'smoke_exposure_incomplete.json',dict(reason='真实GPU试跑未产生移动后的断流暴露；不能以预备脚本替代',goal_complete=False))
                raise RuntimeError('moving blackout exposure absent; do not expand')
    dump(root/'complete.json',dict(new_GPU_trials=186,full_trials=180,smoke_trials=6,
        elapsed_s_including_wait=time.monotonic()-started,source_sha_unchanged=True,
        independent_test=False,stage12_complete=False,goal_complete=False))


if __name__=='__main__':main()
