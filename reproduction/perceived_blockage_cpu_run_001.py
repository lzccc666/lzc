"""固定CPU真实感知组合故障预算，依次实际执行并独立审计，不自动重试。"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


def read(path):
    return json.loads(path.read_text())


def ticks(pid):
    try:
        row = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return row[19] if row[0]!='Z' else None
    except OSError:
        return None


def main():
    dest = OUTPUT/'faults/perceived_blockage_cpu_001'
    dest.mkdir(exist_ok=False)
    previous = read(OUTPUT/'faults/semantic_async_cpu_003/worker_process.json')
    assert ticks(previous['pid']) != previous['start_ticks']
    sources = dict(read(OUTPUT/'source_bundle/source_checkpoint_037/checkpoint.json')['sources'])
    for name in ['perceived_blockage_cpu_worker_001.py','perceived_blockage_cpu_audit_001.py','perceived_blockage_cpu_run_001.py']:
        path = OUTPUT/'source_bundle'/name
        sources[str(path.relative_to(PROJECT))] = digest(path)
    protocol = OUTPUT/'task_protocol/perceived_blockage_cpu_001.md'
    sources[str(protocol.relative_to(PROJECT))] = digest(protocol)
    conditions = ['rule_blockage','delayed_actual_rank']
    dump(dest/'release.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        conditions=conditions,source_sha256=sources,
        Qwen_model_sha256=read(OUTPUT/'faults/semantic_async_cpu_001/release.json')['Qwen_model_sha256'],
        maximum_wall_s_per_case=240,child_wall_budget_s=420,total_wall_budget_s=900,
        minimum_free_bytes=6_000_000_000,maximum_saved_bytes=1_000_000_000,
        actual_GPU_calls=0,no_automatic_retry=True,goal_complete=False))
    dump(dest/'worker_process.json',dict(pid=os.getpid(),start_ticks=ticks(os.getpid()),source_sha256=digest(__file__)))
    runtime = OUTPUT/'source_bundle/software_render_runtime_001'
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',LP_NUM_THREADS='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1',
        PYOPENGL_PLATFORM='osmesa',PYGLET_SHADOW_WINDOW='false',
        PYTHONPATH=str(PROJECT)+':'+str(runtime/'python_deps'),
        LD_LIBRARY_PATH=str(runtime/'extracted/usr/lib/x86_64-linux-gnu')+
            ':/root/autodl-fs/third_party/graphics_runtime/lib:/root/miniconda3/envs/conceptgraph/lib')
    begun = time.monotonic()
    reports = []
    try:
        for index, condition in enumerate(conditions):
            assert all(digest(PROJECT/p)==h for p,h in sources.items())
            assert shutil.disk_usage(OUTPUT).free > 6_000_000_000
            assert sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()) < 1_000_000_000
            for role, budget in [('worker',420),('audit',60)]:
                assert 900-(time.monotonic()-begun)>budget
                command = [sys.executable,str(OUTPUT/'source_bundle'/f'perceived_blockage_cpu_{role}_001.py'),
                           '--condition',condition]
                if role=='worker':
                    command += ['--release',str(dest/'release.json')]
                with (dest/f'{index:02d}_{role}.log').open('x') as log:
                    child = subprocess.run(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                           timeout=budget,check=False)
                dump(dest/f'{index:02d}_{role}_exit.json',dict(condition=condition,returncode=child.returncode,goal_complete=False))
                assert child.returncode==0,'保留原失败，不自动重试'
            reports.append(read(dest/condition/'independent_audit_001/report.json'))
            dump(dest/f'{index:02d}_result.json',reports[-1])
        dump(dest/'complete.json',dict(execution_complete=True,conditions=reports,
            all_fault_exposures_passed=all(r['passed'] for r in reports),
            actual_frontend_calls=sum(r['actual_frontend_calls'] for r in reports),
            actual_CPU_Qwen_calls=sum(r['actual_CPU_Qwen_calls'] for r in reports),
            actual_controls=sum(r['actual_controls'] for r in reports),
            formal_GPU_fault_gate=False,new_independent_tasks=0,new_arrival_credit=0,goal_complete=False))
    except BaseException as error:
        dump(dest/'interrupted.json',dict(error=repr(error),conditions_completed=len(reports),goal_complete=False))
        raise


if __name__ == '__main__':
    main()
