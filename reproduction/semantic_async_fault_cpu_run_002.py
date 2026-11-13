"""等待原CPU父进程终止后，补单列期限版本和未执行的绑定故障。"""
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
    return json.loads(Path(path).read_text())


def ticks(pid):
    try:
        row = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
        return row[19] if row[0] != 'Z' else None
    except OSError:
        return None


def main():
    base = OUTPUT/'faults/semantic_async_cpu_002'
    base.mkdir(exist_ok=False)
    prior = OUTPUT/'faults/semantic_async_cpu_001'
    parent = read(prior/'worker_process.json')
    assert ticks(parent['pid']) == parent['start_ticks']
    original = read(prior/'release.json')
    sources = dict(read(OUTPUT/'source_bundle/source_checkpoint_036/checkpoint.json')['sources'])
    for name in ['semantic_async_fault_cpu_worker_002.py', 'semantic_async_fault_cpu_audit_002.py', 'semantic_async_fault_cpu_run_002.py']:
        path = OUTPUT/'source_bundle'/name
        sources[str(path.relative_to(PROJECT))] = digest(path)
    protocol = OUTPUT/'task_protocol/semantic_async_delivery_expiry_cpu_002.md'
    sources[str(protocol.relative_to(PROJECT))] = digest(protocol)
    dump(base/'release.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        conditions=['expired_before_delivery', 'binding_changed'], source_sha256=sources,
        Qwen_model_sha256=original['Qwen_model_sha256'], original_parent=parent,
        original_release_sha256=digest(prior/'release.json'), wait_budget_s=3000,
        work_budget_s=650, expired_delivery_TTL_s=120, expired_delivery_poll_after_s=140,
        actual_GPU_calls=0, no_automatic_retry=True, original_failure_preserved=True, goal_complete=False))
    pid = os.getpid()
    dump(base/'worker_process.json', dict(pid=pid, start_ticks=ticks(pid), CUDA_VISIBLE_DEVICES='', source_sha256=digest(__file__)))
    runtime = OUTPUT/'source_bundle/software_render_runtime_001'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', LP_NUM_THREADS='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONNOUSERSITE='1',
        PYOPENGL_PLATFORM='osmesa', PYGLET_SHADOW_WINDOW='false',
        PYTHONPATH=str(PROJECT)+':'+str(runtime/'python_deps'),
        LD_LIBRARY_PATH=str(runtime/'extracted/usr/lib/x86_64-linux-gnu')+
            ':/root/autodl-fs/third_party/graphics_runtime/lib:/root/miniconda3/envs/conceptgraph/lib')
    begun = time.monotonic()
    results = []
    try:
        while ticks(parent['pid']) == parent['start_ticks']:
            assert time.monotonic()-begun < 3000, '原CPU进程仍活跃；保留，不重启'
            time.sleep(5)
        assert (prior/'complete.json').exists() or (prior/'interrupted.json').exists()
        dump(base/'original_terminal.json', dict(original_failure=read(prior/'interrupted.json') if (prior/'interrupted.json').exists() else None,
            original_complete=(prior/'complete.json').exists(), goal_complete=False))
        started = time.monotonic()
        for index, condition in enumerate(['expired_before_delivery', 'binding_changed']):
            old_report = prior/condition/'independent_audit_001/report.json'
            if old_report.exists() and read(old_report)['passed']:
                dump(base/f'{index:02d}_not_repeated.json', dict(reason='首版已经通过，不重复实验', evidence_sha256=digest(old_report)))
                continue
            for name, expected in sources.items():
                assert digest(PROJECT/name) == expected
            assert shutil.disk_usage(OUTPUT).free > 6_000_000_000
            for role in ['worker', 'audit']:
                command = [sys.executable, str(OUTPUT/f'source_bundle/semantic_async_fault_cpu_{role}_002.py'), '--condition', condition]
                if role == 'worker':
                    command += ['--release', str(base/'release.json')]
                with (base/f'{index:02d}_{role}.log').open('x') as log:
                    child = subprocess.run(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT,
                        timeout=min(300, max(1, 650-(time.monotonic()-started))), check=False)
                dump(base/f'{index:02d}_{role}_exit.json', dict(returncode=child.returncode, condition=condition, goal_complete=False))
                assert child.returncode == 0, '第二版本失败保留，不自动重试'
            report = read(base/condition/'independent_audit_001/report.json')
            assert report['passed']
            results.append(report)
        dump(base/'complete.json', dict(passed=True, results=results, original_failure_preserved=True,
            actual_Qwen_calls=sum(r['actual_Qwen_calls'] for r in results), actual_GPU_calls=0,
            new_independent_tasks=0, frozen_main_table_credit=False, formal_GPU_fault_gate_complete=False,
            model_accuracy_credit=False, goal_complete=False))
    except BaseException as exc:
        dump(base/'interrupted.json', dict(error=repr(exc), completed_conditions=len(results), goal_complete=False))
        raise


if __name__ == '__main__':
    main()
