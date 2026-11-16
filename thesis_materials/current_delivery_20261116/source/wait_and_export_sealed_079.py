"""等既有075监督进程完成后导出图表；不启动/重启任何模型、训练或CARLA。"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from sealed_resource_guard_074 import ResourceGuard

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT/'main_experiments/sealed_execution_075'
OUT = ROOT/'figures/sealed_public_078'
SUPERVISOR = 148297


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    with (OUT/name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def state(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None, None


def main():
    begun = time.monotonic()
    status, identity = state(SUPERVISOR)
    assert status is not None and status != 'Z'
    assert b'run_sealed_resource_guard_075.py' in Path(f'/proc/{SUPERVISOR}/cmdline').read_bytes()
    paths = [Path(__file__), ROOT/'source/export_sealed_public_078.py',
             ROOT/'source/plot_public_evidence_004.py', OUT/'figure_contract.json']
    sources = {str(p): sha(p) for p in paths}
    write('export_wait_registration_079.json', dict(
        UTC=datetime.now(timezone.utc).isoformat(), existing_supervisor_pid=SUPERVISOR,
        existing_supervisor_start_ticks=identity, sources_sha256=sources,
        watch_deadline_s=12*3600, interval_s=10, export_timeout_s=1200,
        new_GPU_calls=0, new_CARLA_calls=0, no_model_or_supervisor_restart=True,
        full_five_model_evaluation_required=True, visual_QA_after_export_required=True,
        metric_edge_tests_passed=8, goal_complete=False))
    try:
        while True:
            status, current_identity = state(SUPERVISOR)
            if status is None or status == 'Z' or current_identity != identity:
                break
            if time.monotonic()-begun > 12*3600:
                raise TimeoutError('等待截止，仅停止导出等待器，不中断实验')
            time.sleep(10)
        stages = []
        for index in range(7):
            path = INPUT/f'stage_{index:02d}_receipt.json'
            receipt = json.loads(path.read_text())
            assert receipt['exit_code'] == 0
            stages.append(dict(index=index, receipt_sha256=sha(path)))
        assert (INPUT/'evaluation/report.json').is_file()
        assert all(sha(Path(p)) == h for p, h in sources.items())
        guard = ResourceGuard(ROOT, OUT)
        tick = time.monotonic()
        guard(tick, 1200)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            LD_LIBRARY_PATH='/root/miniconda3/envs/conceptgraph/lib',
            PYTHONPATH='/root/autodl-tmp/lizichong:'+str(ROOT/'source'))
        with (OUT/'export_actual_079.log').open('x') as log:
            process = subprocess.run([sys.executable, '-B', str(ROOT/'source/export_sealed_public_078.py')],
                env=env, cwd='/root/autodl-tmp/lizichong', stdout=log,
                stderr=subprocess.STDOUT, timeout=1200)
        assert process.returncode == 0, '图表导出失败，保留全部前缀，不自动覆盖重试'
        guard(tick, 1200, force=True)
        qa = json.loads((OUT/'automated_qa.json').read_text())
        assert not any(item['text_outside_canvas'] for item in qa['figures']), '发现文字越界，需要新版本修正'
        write('export_wait_receipt_079.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
            predecessor_stages=stages, total_wait_and_export_s=time.monotonic()-begun,
            export_s=time.monotonic()-tick, export_exit_code=process.returncode,
            automated_qa_sha256=sha(OUT/'automated_qa.json'),
            visual_inspection_pending=True, whole_goal_complete=False,
            actual_new_GPU_calls=0, actual_new_CARLA_calls=0, goal_complete=False))
        print('sealed figure export finished; visual review still pending', flush=True)
    except BaseException as exc:
        write(f'failure_wait_export_{time.time_ns()}.json', dict(
            UTC=datetime.now(timezone.utc).isoformat(), error=repr(exc),
            traceback=traceback.format_exc(), original_experiments_not_restarted=True,
            goal_complete=False))
        raise


if __name__ == '__main__':
    main()
