"""等待既有实际冷/热进程结束，随后独立审核并整理交付；失败不重试。"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


PROJECT = Path('/root/autodl-tmp/lizichong')
OUTPUT = PROJECT/'outputs/project_v2/thesis_completion_v1'


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def ticks(pid):
    try:
        row = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
        return row[19] if row[0] != 'Z' else None
    except OSError:
        return None


def main():
    base = OUTPUT/'reproduction/three_task_fixedlogical_001'
    destination = base/'postrun_001'
    destination.mkdir(exist_ok=False)
    original = json.loads((base/'launcher_process.json').read_text())
    scripts = [OUTPUT/'source_bundle'/name for name in
        ['three_task_fixedlogical_audit_001.py', 'finalize_three_task_cache_delivery_001.py']]
    frozen = {str(p):sha(p) for p in scripts}
    write(destination/'release.json', dict(source_sha256=frozen, parent=original,
        wait_budget_s=2400, work_budget_s=600, minimum_free_bytes=6_000_000_000,
        actual_model_calls=0, no_retry=True, goal_complete=False))
    write(destination/'worker_process.json', dict(pid=os.getpid(), start_ticks=ticks(os.getpid()),
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__)))
    begun = time.monotonic()
    try:
        while ticks(original['pid']) == original['start_ticks']:
            assert time.monotonic()-begun < 2400, '等待预算耗尽，保留原进程，不重复启动'
            time.sleep(5)
        assert (base/'execution_finished.json').is_file() and not (base/'interrupted.json').exists()
        started = time.monotonic()
        for i, path in enumerate(scripts):
            assert sha(path) == frozen[str(path)], '等待期间辅助源变化，停止，不使用修改后代码'
            assert shutil.disk_usage(OUTPUT).free > 6_000_000_000
            with (destination/f'{i:02d}.log').open('x') as log:
                result = subprocess.run([sys.executable, str(path)], cwd=PROJECT,
                    env=dict(os.environ, PYTHONPATH=str(PROJECT), CUDA_VISIBLE_DEVICES='',
                        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                        LD_LIBRARY_PATH='/root/miniconda3/envs/conceptgraph/lib'),
                    stdout=log, stderr=subprocess.STDOUT,
                    timeout=max(1, 600-(time.monotonic()-started)), check=False)
            write(destination/f'{i:02d}_exit.json', dict(returncode=result.returncode, source_sha256=frozen[str(path)]))
            assert result.returncode == 0, '独立审核或交付失败，原样保存，不自动重试'
            if i == 0:
                report = json.loads((base/'independent_audit_001/report.json').read_text())
                assert report['passed'] and report['full_three_task_fixedlogical_small_task_gate']
        write(destination/'complete.json', dict(passed=True, actual_model_calls=0,
            actual_navigation_runs_added_by_postrun=0, wait_wall_s=started-begun,
            work_wall_s=time.monotonic()-started, goal_complete=False))
    except BaseException as exc:
        write(destination/'interrupted.json', dict(error=repr(exc), goal_complete=False))
        raise


if __name__ == '__main__':
    main()
