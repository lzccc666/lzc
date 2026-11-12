"""CPU与llvmpipe实际前端的六条件有界试跑；不与当前主矩阵争用CUDA。"""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

from conceptgraph.completion.stage9 import PROJECT,OUTPUT,digest,dump
from conceptgraph.completion.realtime_faults import CONDITIONS


def main():
    root=OUTPUT/'faults/realtime_fault_cpu_smoke_002'
    root.mkdir(exist_ok=False)
    runtime=OUTPUT/'source_bundle/software_render_runtime_001'
    assert 'llvmpipe' in (runtime/'probe_004.log').read_text()
    sources={str(p.relative_to(PROJECT)):digest(p) for p in
        [Path(__file__).resolve(),PROJECT/'conceptgraph/completion/realtime_faults.py',
         OUTPUT/'source_bundle/realtime_fault_matrix_001.py']}
    plan=[dict(condition=c,run_id=f'realtime_fault_cpu_smoke_002_{c}') for c in CONDITIONS]
    release=dict(plan=plan,source_sha256=sources,task='config00-positive0',seed=17,model='rule',
        frontend_device='cpu',render_backend='OSMesa llvmpipe',maximum_wall_s=70,sensor_ttl_s=15.,
        child_timeout_s=240,total_wall_budget_s=1500,storage_budget_bytes=1000000000,minimum_free_bytes=6000000000,
        rationale='CPU感知较慢，15秒TTL仅为此独立试跑；正式GPU故障/主矩阵保持1.5秒，不横比到达或性能',
        smoke_expansion_gate='先baseline/blackout取得实际前端与完整结果，才继续余四条件；不以导航成功选样本',
        model_GPU_calls=0,independent_test=False,performance_credit=False,goal_complete=False)
    dump(root/'release.json',release)
    spec=importlib.util.spec_from_file_location('fault_matrix',OUTPUT/'source_bundle/realtime_fault_matrix_001.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',LP_NUM_THREADS='2',
        PYOPENGL_PLATFORM='osmesa',PYGLET_SHADOW_WINDOW='false',
        PYTHONPATH=str(runtime/'python_deps'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        LD_LIBRARY_PATH=f'{runtime}/extracted/usr/lib/x86_64-linux-gnu:/root/autodl-fs/third_party/graphics_runtime/lib:/root/miniconda3/envs/conceptgraph/lib')
    events=OUTPUT/'interactive_simulation/realtime_three_model_full_001/resource_events.jsonl'
    def note(state,**values):
        with events.open('a') as h:h.write(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(),event='realtime_fault_cpu_smoke_002',
            status=state,CUDA_VISIBLE_DEVICES='',software_rendering=True,shared_CPU_performance_credit=False,**values))+'\n')
    records=[];started=time.monotonic();note('start')
    try:
        for index,trial in enumerate(plan):
            assert all(digest(PROJECT/name)==sha for name,sha in sources.items())
            assert time.monotonic()-started<1500 and shutil.disk_usage(OUTPUT).free>6000000000
            command=[sys.executable,'-m','conceptgraph.completion.realtime_faults','--run-id',trial['run_id'],
                '--fault',trial['condition'],'--fault-seed','17','--frontend-device','cpu','--model','rule',
                '--task','config00-positive0','--seed','17','--semantic-tasks','','--maximum-wall-s','70',
                '--candidate-plan-budget-s','10','--sensor-ttl-s','15']
            note('case_start',condition=trial['condition'])
            with (root/f'{index:02d}.log').open('x') as log:
                child=subprocess.run(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
            case=OUTPUT/'interactive_simulation'/trial['run_id']/'config00-positive0/seed17'
            path=case/'result.json';result=json.loads(path.read_text()) if path.exists() else None
            metrics=module.exposure_metrics(case) if result else None
            record=dict(**trial,returncode=child.returncode,raw_case=str(case),result=result,exposure=metrics,
                result_sha256=digest(path) if path.exists() else None)
            records.append(record);dump(root/f'{index:02d}_record.json',record)
            note('case_end',condition=trial['condition'],returncode=child.returncode)
            print(json.dumps(dict(condition=trial['condition'],returncode=child.returncode,
                status=result['status'] if result else None,exposure=metrics),ensure_ascii=False),flush=True)
            if child.returncode!=0 or result is None or not metrics['actual_detector_calls']:
                raise RuntimeError('actual frontend smoke incomplete; preserve failure, do not expand')
        dump(root/'summary.json',dict(attempted=len(records),expected=6,model='rule',frontend_device='cpu',
            complete_input_output_smoke=True,elapsed_s=time.monotonic()-started,
            results=[dict(condition=r['condition'],status=r['result']['status'],actual_fault_exposed=r['exposure']['actual_fault_exposed']) for r in records],
            no_Qwen_model_called=True,formal_GPU_matrix_completed=False,arrival_credit=False,goal_complete=False))
    except BaseException as exc:
        dump(root/'interrupted.json',dict(error=repr(exc),attempted=len(records),goal_complete=False))
        raise
    finally:
        note('end',attempted=len(records))


if __name__=='__main__':main()
