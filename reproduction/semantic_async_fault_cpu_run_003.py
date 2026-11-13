"""结束既有CPU进程后执行固定观察窗；未暴露负结果保留并继续独立条件。"""
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
    base = OUTPUT/'faults/semantic_async_cpu_003'
    base.mkdir(exist_ok=False)
    parents = [read(OUTPUT/f'faults/semantic_async_cpu_{v:03d}/worker_process.json') for v in [1, 2]]
    assert all(ticks(p['pid']) != p['start_ticks'] for p in parents), '原CPU进程还活跃，不能并发'
    sources = dict(read(OUTPUT/'source_bundle/source_checkpoint_036/checkpoint.json')['sources'])
    for name in ['semantic_async_fault_cpu_worker_003.py', 'semantic_async_fault_cpu_audit_004.py',
        'semantic_async_fault_cpu_run_003.py', 'semantic_minus_execution_audit_001.py']:
        p = OUTPUT/'source_bundle'/name
        sources[str(p.relative_to(PROJECT))] = digest(p)
    protocol = OUTPUT/'task_protocol/semantic_async_fault_observation_cpu_003.md'
    sources[str(protocol.relative_to(PROJECT))] = digest(protocol)
    conditions = ['empty_output', 'malformed_output', 'backend_timeout', 'late_and_queue_expiry',
        'binding_changed', 'expired_before_delivery']
    model = read(OUTPUT/'faults/semantic_async_cpu_001/release.json')['Qwen_model_sha256']
    dump(base/'release.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), conditions=conditions,
        source_sha256=sources, Qwen_model_sha256=model, original_parents=parents,
        navigation_wall_budget_s=180, child_wall_budget_s=300, total_wall_budget_s=1800,
        minimum_free_bytes=6_000_000_000, max_saved_bytes=2_000_000_000,
        baseline_evidence_sha256=digest(OUTPUT/'faults/semantic_async_cpu_001/baseline/independent_audit_004/report.json'),
        exception_evidence_sha256=digest(OUTPUT/'faults/semantic_async_cpu_001/backend_exception/independent_audit_004/report.json'),
        original_002_negative_preserved=True, original_fault_terminal_policy_changed_and_separate=True,
        actual_GPU_calls=0, no_automatic_retry=True, goal_complete=False))
    pid = os.getpid()
    dump(base/'worker_process.json', dict(pid=pid, start_ticks=ticks(pid), CUDA_VISIBLE_DEVICES='', source_sha256=digest(__file__)))
    runtime = OUTPUT/'source_bundle/software_render_runtime_001'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', LP_NUM_THREADS='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONNOUSERSITE='1',
        PYOPENGL_PLATFORM='osmesa', PYGLET_SHADOW_WINDOW='false',
        PYTHONPATH=str(PROJECT)+':'+str(runtime/'python_deps'),
        LD_LIBRARY_PATH=str(runtime/'extracted/usr/lib/x86_64-linux-gnu')+
            ':/root/autodl-fs/third_party/graphics_runtime/lib:/root/miniconda3/envs/conceptgraph/lib')
    started, reports = time.monotonic(), []
    try:
        for index, condition in enumerate(conditions):
            assert all(digest(PROJECT/p) == expected for p, expected in sources.items())
            assert shutil.disk_usage(OUTPUT).free > 6_000_000_000
            assert sum(p.stat().st_size for p in base.rglob('*') if p.is_file()) < 2_000_000_000
            remaining = 1800-(time.monotonic()-started)
            assert remaining > 300
            for role in ['worker', 'audit']:
                if role == 'worker':
                    command = [sys.executable, str(OUTPUT/'source_bundle/semantic_async_fault_cpu_worker_003.py'),
                        '--release', str(base/'release.json'), '--condition', condition]
                else:
                    command = [sys.executable, str(OUTPUT/'source_bundle/semantic_async_fault_cpu_audit_004.py'),
                        '--run-id', 'semantic_async_cpu_003', '--condition', condition]
                with (base/f'{index:02d}_{role}.log').open('x') as log:
                    child = subprocess.run(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT,
                        timeout=min(300, remaining), check=False)
                dump(base/f'{index:02d}_{role}_exit.json', dict(returncode=child.returncode, condition=condition, goal_complete=False))
                assert child.returncode == 0, '结构/控制审核失败，保留，不自动重试'
            report = read(base/condition/'independent_audit_004/report.json')
            reports.append(report)
            dump(base/f'{index:02d}_result.json', report)
        dump(base/'complete.json', dict(execution_complete=True, all_fault_exposures_passed=all(r['passed'] for r in reports),
            reports=reports, actual_Qwen_calls=sum(r['actual_Qwen_calls'] for r in reports),
            actual_frontend_calls=sum(r['actual_frontend_calls'] for r in reports),
            actual_control_intervals=sum(r['actual_control_intervals'] for r in reports),
            unresolved_faults=[r['condition'] for r in reports if not r['passed']], actual_GPU_calls=0,
            model_accuracy_credit=False, new_independent_tasks=0, formal_GPU_fault_gate=False, goal_complete=False))
    except BaseException as exc:
        dump(base/'interrupted.json', dict(error=repr(exc), completed_conditions=len(reports), goal_complete=False))
        raise


if __name__ == '__main__':
    main()
