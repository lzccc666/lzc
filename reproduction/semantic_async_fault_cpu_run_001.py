"""先两项实际smoke及独立审核，再按冻结次序执行余下六项。"""
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


def main():
    base = OUTPUT/'faults/semantic_async_cpu_001'
    base.mkdir(exist_ok=False)
    prior = read(OUTPUT/'source_bundle/source_checkpoint_036/checkpoint.json')
    original_index = digest(PROJECT/'.git/index')
    assert original_index == prior['original_index_sha256']
    source = dict(prior['sources'])
    names = ['semantic_async_fault_cpu_worker_001.py', 'semantic_async_fault_cpu_audit_001.py',
        'semantic_async_fault_cpu_run_001.py']
    source.update({str((OUTPUT/'source_bundle'/name).relative_to(PROJECT)):digest(OUTPUT/'source_bundle'/name) for name in names})
    protocol = OUTPUT/'task_protocol/semantic_async_fault_cpu_001.md'
    source[str(protocol.relative_to(PROJECT))] = digest(protocol)
    model = read(OUTPUT/'reproduction/three_task_fixedlogical_001/experiment_release.json')['Qwen_model_sha256']
    conditions = ['baseline', 'backend_exception', 'empty_output', 'malformed_output', 'backend_timeout',
        'late_and_queue_expiry', 'expired_before_delivery', 'binding_changed']
    release = dict(created_utc=datetime.now(timezone.utc).isoformat(), conditions=conditions,
        source_sha256=source, Qwen_model_sha256=model, initial_root_index_sha256=original_index,
        per_navigation_wall_budget_s=180, per_child_wall_budget_s=300, total_wall_budget_s=2400,
        minimum_free_bytes=6_000_000_000, storage_budget_bytes=2_000_000_000,
        smoke_conditions=conditions[:2], task='config00-positive0', seed=17,
        CPU_only=True, actual_GPU_calls=0, no_automatic_retry=True,
        manuscript_text_written=False, goal_complete=False)
    dump(base/'release.json', release)
    pid = os.getpid()
    dump(base/'worker_process.json', dict(pid=pid,
        start_ticks=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[19],
        CUDA_VISIBLE_DEVICES='', source_sha256=digest(__file__)))
    runtime = OUTPUT/'source_bundle/software_render_runtime_001'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', LP_NUM_THREADS='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONNOUSERSITE='1',
        PYOPENGL_PLATFORM='osmesa', PYGLET_SHADOW_WINDOW='false',
        PYTHONPATH=str(PROJECT)+':'+str(runtime/'python_deps'),
        LD_LIBRARY_PATH=str(runtime/'extracted/usr/lib/x86_64-linux-gnu')+
            ':/root/autodl-fs/third_party/graphics_runtime/lib:/root/miniconda3/envs/conceptgraph/lib')
    started, results = time.monotonic(), []
    try:
        for index, condition in enumerate(conditions):
            for name, expected in source.items():
                assert digest(PROJECT/name) == expected, name
            assert digest(PROJECT/'.git/index') == original_index
            remaining = 2400-(time.monotonic()-started)
            assert remaining > 310 and shutil.disk_usage(OUTPUT).free > 6_000_000_000
            assert sum(p.stat().st_size for p in base.rglob('*') if p.is_file()) < 2_000_000_000
            for role, command in [('worker', [sys.executable, str(OUTPUT/'source_bundle/semantic_async_fault_cpu_worker_001.py'),
                    '--release', str(base/'release.json'), '--condition', condition]),
                ('audit', [sys.executable, str(OUTPUT/'source_bundle/semantic_async_fault_cpu_audit_001.py'), '--condition', condition])]:
                with (base/f'{index:02d}_{role}.log').open('x') as log:
                    child = subprocess.run(command, cwd=PROJECT, env=env, stdout=log,
                        stderr=subprocess.STDOUT, timeout=min(300, remaining), check=False)
                dump(base/f'{index:02d}_{role}_exit.json', dict(condition=condition, returncode=child.returncode, goal_complete=False))
                assert child.returncode == 0, condition+' '+role+'失败保留，无自动重试'
            report = read(base/condition/'independent_audit_001/report.json')
            assert report['passed']
            results.append(report)
            dump(base/f'{index:02d}_result.json', report)
            if index == 1:
                dump(base/'smoke_passed.json', dict(passed=True, conditions=conditions[:2],
                    actual_model_calls=sum(r['actual_Qwen_calls'] for r in results),
                    actual_frontend_calls=sum(r['actual_frontend_calls'] for r in results), goal_complete=False))
        dump(base/'complete.json', dict(passed=True, conditions=results,
            actual_Qwen_calls=sum(r['actual_Qwen_calls'] for r in results),
            actual_frontend_calls=sum(r['actual_frontend_calls'] for r in results),
            actual_control_intervals=sum(r['actual_control_intervals'] for r in results),
            CPU_development_only=True, actual_GPU_calls=0, new_independent_tasks=0,
            formal_GPU_fault_gate_complete=False, model_accuracy_credit=False,
            wall_s=time.monotonic()-started, new_test_opened=False, goal_complete=False))
    except BaseException as exc:
        dump(base/'interrupted.json', dict(error=repr(exc), completed_conditions=len(results),
            wall_s=time.monotonic()-started, goal_complete=False))
        raise


if __name__ == '__main__':
    main()
