"""原审核失败保留，续行未执行的实际排序条件；已完成rule不重跑。"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


def read(path):
    return json.loads(path.read_text())


def ticks(pid):
    try:
        row=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return row[19] if row[0]!='Z' else None
    except OSError:
        return None


def main():
    base=OUTPUT/'faults/perceived_blockage_cpu_002'
    dest=base/'recovery_001'
    dest.mkdir(exist_ok=False)
    previous=read(base/'worker_process.json')
    assert ticks(previous['pid'])!=previous['start_ticks']
    assert read(base/'00_worker_exit.json')['returncode']==0
    assert read(base/'00_audit_exit.json')['returncode']==1
    assert not (base/'delayed_actual_rank').exists(),'未执行条件只能启动一次'
    rule=read(base/'rule_blockage/independent_audit_003/report.json')
    assert rule['passed']
    release=read(base/'release.json')
    assert all(digest(PROJECT/p)==h for p,h in release['source_sha256'].items())
    auxiliary=[Path(__file__),OUTPUT/'source_bundle/perceived_blockage_cpu_audit_004.py',
        OUTPUT/'task_protocol/perceived_blockage_audit_correction_003.md']
    dump(dest/'release.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        original_release_sha256=digest(base/'release.json'),original_parent=previous,
        auxiliary_source_sha256={str(p.relative_to(PROJECT)):digest(p) for p in auxiliary},
        existing_rule_result_sha256=digest(base/'rule_blockage/navigation/result.json'),
        rule_rerun=False,unexecuted_condition='delayed_actual_rank',
        worker_budget_s=420,audit_budget_s=60,automatic_retry=False,goal_complete=False))
    dump(dest/'worker_process.json',dict(pid=os.getpid(),start_ticks=ticks(os.getpid()),source_sha256=digest(__file__)))
    runtime=OUTPUT/'source_bundle/software_render_runtime_001'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',LP_NUM_THREADS='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1',
        PYOPENGL_PLATFORM='osmesa',PYGLET_SHADOW_WINDOW='false',
        PYTHONPATH=str(PROJECT)+':'+str(runtime/'python_deps'),
        LD_LIBRARY_PATH=str(runtime/'extracted/usr/lib/x86_64-linux-gnu')+
            ':/root/autodl-fs/third_party/graphics_runtime/lib:/root/miniconda3/envs/conceptgraph/lib')
    try:
        commands=[('worker',420,[sys.executable,str(OUTPUT/'source_bundle/perceived_blockage_cpu_worker_002.py'),
            '--release',str(base/'release.json'),'--condition','delayed_actual_rank']),
            ('audit',60,[sys.executable,str(OUTPUT/'source_bundle/perceived_blockage_cpu_audit_004.py'),
            '--condition','delayed_actual_rank'])]
        for role,budget,command in commands:
            with (dest/f'{role}.log').open('x') as handle:
                child=subprocess.run(command,cwd=PROJECT,env=env,stdout=handle,stderr=subprocess.STDOUT,
                    timeout=budget,check=False)
            dump(dest/f'{role}_exit.json',dict(returncode=child.returncode,goal_complete=False))
            assert child.returncode==0,'不重试；保留实际数据与故障'
        delayed=read(base/'delayed_actual_rank/independent_audit_003/report.json')
        dump(dest/'complete.json',dict(execution_complete=True,all_required_exposures_passed=rule['passed'] and delayed['passed'],
            conditions=[rule,delayed],existing_rule_rerun=False,
            actual_frontend_calls=rule['actual_frontend_calls']+delayed['actual_frontend_calls'],
            actual_controls=rule['actual_controls']+delayed['actual_controls'],
            actual_CPU_Qwen_calls=rule['actual_CPU_Qwen_calls']+delayed['actual_CPU_Qwen_calls'],
            formal_GPU_fault_gate=False,new_independent_tasks=0,new_arrival_credit=0,goal_complete=False))
    except BaseException as error:
        dump(dest/'interrupted.json',dict(error=repr(error),goal_complete=False))
        raise


if __name__=='__main__':
    main()
