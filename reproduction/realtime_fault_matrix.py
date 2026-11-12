"""六条件×10事前指定正任务×三种子；先完整真实GPU试跑，再独立180次全量。"""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import argparse
import itertools
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time

from conceptgraph.completion.realtime_faults import CONDITIONS
from conceptgraph.completion.realtime_batch import pipeline_sources,verify_sources
from conceptgraph.completion.realtime_matrix import bytes_in,verified_records,PROTOCOL
from conceptgraph.completion.stage9 import PROJECT,OUTPUT,digest,dump


def exposure_metrics(case):
    """从真实事件及控制心跳计算暴露/停止/恢复；未触发不作安全成功。"""
    path=case/'fault_exposure.json'
    if not path.exists():
        return dict(exposure_unknown=True,actual_fault_exposed=None)
    exposure=json.loads(path.read_text())
    events=list(map(json.loads,(case/'fault_events.jsonl').open()))
    controls=list(map(json.loads,(case/'execution.jsonl').open()))
    clock=json.loads((case/'fault_driver_clock.json').read_text())
    sensor=[r for r in events if r['event']=='sensor']
    detectors=[r for r in events if r['event']=='detector']
    assert len({r['frame_idx'] for r in sensor})==len(sensor)
    assert len({r['frame_idx'] for r in detectors})==len(detectors)
    assert all(b['wall_elapsed_s']>=a['wall_elapsed_s'] for a,b in zip(controls,controls[1:]))
    begins=[r for r in events if r['event']=='blackout_begin']
    ends=[r for r in events if r['event']=='blackout_end']
    result=dict(exposure_unknown=False,**exposure,actual_sensor_frames=len(sensor),actual_detector_calls=len(detectors),
        recorded_control_ticks=len(controls),blackout_control_ticks=None,
        blackout_stale_braking_ticks=None,blackout_stopped_during_gap=None,after_gap_fresh_motion_observed=None,
        actual_detection_removals=sum(r['input_detections']-r['retained_detections'] for r in detectors),
        actual_depth_pixels_removed=sum(max(0,r['input_depth_pixels']-r['retained_depth_pixels']) for r in sensor),
        actual_noisy_pose_frames=sum(any(abs(v)>0 for v in r['position_delta_m']) for r in sensor),
        raw_event_sha256=digest(case/'fault_events.jsonl'),raw_execution_sha256=digest(case/'execution.jsonl'))
    if begins:
        assert len(begins)==len(ends)==1
        first,last=begins[0]['wall_s'],ends[0]['wall_s']
        assert last-first>=4.
        ticks=[r for r in controls if first<=clock['started_monotonic_s']+r['wall_elapsed_s']<=last]
        after=[r for r in controls if clock['started_monotonic_s']+r['wall_elapsed_s']>last]
        result.update(blackout_duration_s=last-first,blackout_control_ticks=len(ticks),
            blackout_stale_braking_ticks=sum(r['stale_sensor_braking'] for r in ticks),
            blackout_stopped_during_gap=any(r['stale_sensor_braking'] and r['state']['speed_mps']==0 for r in ticks),
            after_gap_fresh_motion_observed=any(not r['stale_sensor_braking'] and r['state']['speed_mps']>.2 for r in after))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--wall-budget-s',type=float,default=43200.)
    parser.add_argument('--storage-budget-gb',type=float,default=8.)
    args=parser.parse_args()
    dest=OUTPUT/'interactive_simulation'/args.run_id
    tasks_path=OUTPUT/'interactive_simulation/synthetic_tasks_v1.json'
    selected_path=OUTPUT/'training/selected_three_task_v2_model_001.json'
    tasks=json.loads(tasks_path.read_text())
    if args.resume:
        release=json.loads((dest/'release.json').read_text())
        assert release['smoke']==args.smoke
    else:
        dest.mkdir(exist_ok=False)
        # 每配置固定positive0，三种子及条件全部保留；与运行成功/失败无关。
        chosen=[t for t in tasks['tasks'] if t['task_id']==t['config_id']+'-positive0']
        assert len(chosen)==10 and len({t['config_id'] for t in chosen})==10
        if args.smoke:
            chosen=[t for t in chosen if t['task_id']=='config00-positive0']
        seeds=[17] if args.smoke else tasks['seeds']
        plan=[dict(variant=c,model='new',semantic_tasks='query_parse,object_attributes,candidate_rerank',
            fault=c,task=t['task_id'],config_id=t['config_id'],kind=t['kind'],seed=s,
            run_id=f"{args.run_id}_{c}_{t['task_id'].replace('-','_')}_s{s}")
            for c,t,s in itertools.product(CONDITIONS,chosen,seeds)]
        random.Random(17).shuffle(plan)
        sources=sorted(set(pipeline_sources()+[Path(__file__).resolve(),
            PROJECT/'conceptgraph/completion/realtime_faults.py',
            PROJECT/'conceptgraph/completion/realtime_matrix.py',
            PROJECT/'conceptgraph/completion/realtime_matrix_statistics.py',
            PROJECT/'conceptgraph/completion/interactive_statistics.py',
            PROJECT/'conceptgraph/completion/vehicle_statistics.py',
            PROJECT/'conceptgraph/hydra_configs/base_mapping.yaml',PROJECT/'conceptgraph/vehicle_classes.txt']))
        training=json.loads((OUTPUT/'training/three_task_v2_formal_seed17_001/training_report.json').read_text())
        adapter=Path(training['best_checkpoint']['path'])/'adapter_model.safetensors'
        release=dict(trials=plan,smoke=args.smoke,task_release_sha256=digest(tasks_path),
            model_selection_sha256=digest(selected_path),adapter_weights_path=str(adapter),adapter_weights_sha256=digest(adapter),
            source_sha256={str(p.relative_to(PROJECT)):digest(p) for p in sources},protocol=PROTOCOL,
            fault_duration_s=6.,blackout_delay_s=4.,onset_distance_m=.5,minimum_speed_mps=.2,
            wall_budget_s=args.wall_budget_s,storage_budget_bytes=int(args.storage_budget_gb*1e9),minimum_free_bytes=6000000000,
            frozen_utc=datetime.now(timezone.utc).isoformat(),seed_order=seeds,
            design='6条件×每配置固定positive0×三种子；每试验独立进程；smoke只config00/seed17；不复用主矩阵baseline',
            trigger='当前实测位移至少0.5m且速度至少0.2m/s；未触发/未移除任何点或框分别保存，不算抗扰成功',
            scopes=['开发合成正任务；不代替完整50任务/自然分布/新封存',
                '动作后传感器可不同，配对的是任务/种子而不是声称相同闭环RGB',
                '位姿噪声只注入相机观测位姿，物理车态/道路/评价器不改；非完整惯导漂移模型',
                '合成高斯/丢失幅度未经真实车载传感器误差标定','不提供独占性能或真实道路动态交通安全结论'],
            failed_trials_retained=True,automatic_retry=False,independent_test=False,performance_credit=False,goal_complete=False)
        dump(dest/'release.json',release)
        with tarfile.open(dest/'source_snapshot.tar.gz','x:gz') as archive:
            for path in sources:archive.add(path,arcname=str(path.relative_to(PROJECT)))
    assert digest(tasks_path)==release['task_release_sha256'] and digest(selected_path)==release['model_selection_sha256']
    assert digest(Path(release['adapter_weights_path']))==release['adapter_weights_sha256']
    verify_sources(release['source_sha256'])
    if args.prepare_only:
        print(json.dumps(dict(prepared_trials=len(release['trials']),model_calls=0,goal_complete=False)))
        return
    rows=verified_records(dest,release);seen={r['run_id'] for r in rows}
    begun=time.monotonic();previous=sum(r['process_wall_s'] for r in rows)
    environment=dict(os.environ,LD_LIBRARY_PATH='/root/autodl-fs/third_party/graphics_runtime/lib:/root/miniconda3/envs/conceptgraph/lib',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYOPENGL_PLATFORM='egl',PYTHONFAULTHANDLER='1')
    # 不继承隔离CPU软件渲染运行库；正式GPU条件全部使用既有环境。
    environment.pop('PYTHONPATH',None)
    with (dest/'trial_records.jsonl').open('a') as handle:
        for index,trial in enumerate(release['trials']):
            if trial['run_id'] in seen:continue
            verify_sources(release['source_sha256'])
            gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
            if gpu:
                dump(dest/'gpu_busy.json',dict(pids=gpu,next_trial=trial,goal_complete=False));raise RuntimeError('GPU occupied; no concurrent model')
            if previous+time.monotonic()-begun>release['wall_budget_s'] or sum(r['saved_bytes'] for r in rows)>release['storage_budget_bytes'] or shutil.disk_usage(OUTPUT).free<release['minimum_free_bytes']:
                dump(dest/'budget_stop.json',dict(attempted=len(rows),expected=len(release['trials']),goal_complete=False));raise RuntimeError('frozen fault budget reached')
            raw=OUTPUT/'interactive_simulation'/trial['run_id'];protocol=OUTPUT/'faults'/trial['run_id'];case=raw/trial['task']/f"seed{trial['seed']}"
            started=time.monotonic();timeout=False;returncode=None
            if raw.exists() or protocol.exists():
                status_override='PREVIOUS_ATTEMPT_INTERRUPTED'
            else:
                status_override=None
                command=[sys.executable,'-m','conceptgraph.completion.realtime_faults',
                    '--run-id',trial['run_id'],'--fault',trial['fault'],'--fault-seed',str(trial['seed']),
                    '--model','new','--task',trial['task'],'--seed',str(trial['seed']),'--semantic-tasks',trial['semantic_tasks']]
                for flag,key in [('maximum-wall-s','maximum_wall_s'),('semantic-ttl-s','semantic_ttl_s'),('sensor-ttl-s','sensor_ttl_s'),
                    ('candidate-plan-budget-s','candidate_plan_budget_s'),('maximum-candidates','maximum_candidates')]:command+=['--'+flag,str(PROTOCOL[key])]
                with (dest/f'{index:03d}.log').open('x') as log:
                    try:
                        child=subprocess.run(command,cwd=PROJECT,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=PROTOCOL['subprocess_timeout_s'])
                        returncode=child.returncode
                    except subprocess.TimeoutExpired:timeout=True
            path=case/'result.json';result=json.loads(path.read_text()) if path.exists() else None
            metrics=exposure_metrics(case) if result else dict(exposure_unknown=True,actual_fault_exposed=None)
            row=dict(**trial,raw_case=str(case),returncode=returncode,process_timeout=timeout,
                process_wall_s=time.monotonic()-started,result=result,result_sha256=digest(path) if result else None,
                status=result['status'] if result else status_override or 'PROCESS_FAILED_NO_RESULT',
                actual_arrival=bool(result and result['actual_full_perception_arrival']),fault_exposure=metrics,
                saved_bytes=bytes_in([raw,protocol,OUTPUT/'cache_audit'/trial['run_id']]))
            rows.append(row);handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');handle.flush()
            print(json.dumps(dict(index=index,condition=trial['fault'],task=trial['task'],seed=trial['seed'],status=row['status'],
                actual_fault_exposed=metrics['actual_fault_exposed']),ensure_ascii=False),flush=True)
    dump(dest/'summary.json',dict(attempted=len(rows),expected=len(release['trials']),trial_records_sha256=digest(dest/'trial_records.jsonl'),
        all_frozen_trials_attempted=len(rows)==len(release['trials']),processes_with_result=sum(r['result'] is not None for r in rows),
        condition_counts=dict(Counter(r['fault'] for r in rows)),goal_complete=False))
    subprocess.run([sys.executable,'-m','conceptgraph.completion.realtime_matrix_statistics','--input-run',args.run_id,'--run-id',args.run_id],cwd=PROJECT,check=True)
    dump(dest/'exposure_summary.json',dict(conditions={c:dict(frozen_trials=sum(r['fault']==c for r in rows),
        triggered=sum(r['fault']==c and r['fault_exposure']['actual_fault_exposed'] is True for r in rows),
        unknown=sum(r['fault']==c and r['fault_exposure']['actual_fault_exposed'] is None for r in rows),
        blackout_stopped=sum(r['fault']==c and r['fault_exposure'].get('blackout_stopped_during_gap') is True for r in rows)) for c in CONDITIONS},
        exposure_is_not_arrival=True,no_unexposed_safety_credit=True,goal_complete=False))


if __name__=='__main__':main()
