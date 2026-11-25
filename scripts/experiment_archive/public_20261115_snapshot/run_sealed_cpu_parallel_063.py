"""只调整CPU任务调度，冻结058算法不变；接管前保留正在运行的full，不重复启动。"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'main_experiments/sealed_public_058'
OUT = DEST / 'cpu_parallel_063'
RUNNER = ROOT / 'source/sealed_public_benchmark_058.py'
OLD_PARENT = 143176
OLD_FULL = 144080


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    with path.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def process_state(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return raw[0], int(raw[49]) if len(raw) > 49 else None
    except FileNotFoundError:
        return None, None


def verify_report(condition):
    folder = DEST / 'mapping' / condition
    report = json.loads((folder / 'report.json').read_text())
    assert report['frames'] == 1196 and report['scenes'] == 30
    assert report['gt_in_inference'] is False
    for item in report['reports']:
        assert sha(folder / item['scene'] / 'frames.jsonl.gz') == item['frames_sha256']
    return sha(folder / 'report.json')


def main():
    OUT.mkdir(exist_ok=False)
    release = json.loads((DEST / 'release.json').read_text())
    assert len(release['conditions']) == 10 and len(release['variants']) == 5
    assert 'run_sealed_public_series_059.py' in Path(f'/proc/{OLD_PARENT}/cmdline').read_bytes().decode()
    assert 'map' in Path(f'/proc/{OLD_FULL}/cmdline').read_bytes().decode()
    assert f'PPid:\t{OLD_PARENT}\n' in Path(f'/proc/{OLD_FULL}/status').read_text()
    assert not (DEST / 'mapping/full/report.json').exists()
    assert not (DEST / 'requests').exists() and not (DEST / 'models').exists()
    for condition in release['conditions'][1:]:
        assert not (DEST / 'mapping' / condition).exists()
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', LD_LIBRARY_PATH='/root/miniconda3/envs/conceptgraph/lib',
        PYTHONPATH='/root/autodl-tmp/lizichong:' + str(ROOT/'source'))
    # 在任何新配置启动前保存执行策略变化；没有读取GT或模型成绩进行选择。
    protocol = dict(UTC=datetime.now(timezone.utc).isoformat(),
        scope='CPU调度补充，不改变已冻结方法、输入、模型、提示、阈值、种子、预算、评价和样本分母',
        original_release_sha256=sha(DEST/'release.json'), original_runner_sha256=sha(RUNNER),
        original_supervisor_source_sha256=sha(ROOT/'source/run_sealed_public_series_059.py'),
        new_supervisor_sha256=sha(Path(__file__)), previous_events_sha256=sha(DEST/'series_059/events.jsonl'),
        legacy_parent_pid=OLD_PARENT, continuing_full_worker_pid=OLD_FULL,
        CPU_concurrency=4, each_worker_threads=1, GPU_models_concurrency=1,
        old_prefix_preserved=True, full_not_restarted=True, method_outputs_inspected=False,
        performance_boundary='CPU建图为共享资源耗时，不声称独占速度或实时性；GPU模型仍逐进程串行',
        budgets=release['budgets'], conditions=release['conditions'], model_order=list(release['variants']),
        operating_system_cpu_quota='16 cores (cpu.max=1600000/100000)',
        operating_system_memory_limit_GiB=120, goal_complete=False)
    write(OUT/'protocol.json', protocol)
    os.kill(OLD_PARENT, signal.SIGSTOP)
    event_file = (OUT/'events.jsonl').open('x')
    def event(kind, **fields):
        row = dict(UTC=datetime.now(timezone.utc).isoformat(), kind=kind, **fields)
        event_file.write(json.dumps(row, ensure_ascii=False)+'\n')
        event_file.flush()
        print(json.dumps(row, ensure_ascii=False), flush=True)
    event('old_supervisor_stopped_only', pid=OLD_PARENT, child_continues=OLD_FULL)
    pending = list(release['conditions'][1:])
    active = {}
    finished = []
    failures = []
    legacy_done = False
    try:
        while pending or active or not legacy_done:
            state, exit_code = process_state(OLD_FULL)
            if not legacy_done and state in {None, 'Z'}:
                if (DEST/'mapping/full/report.json').exists() and exit_code in {None, 0}:
                    fingerprint = verify_report('full')
                    finished.append(dict(condition='full', report_sha256=fingerprint, original_worker=True))
                    event('original_full_completed', pid=OLD_FULL, exit_code=exit_code,
                          report_sha256=fingerprint)
                    # full已终止且输出通过核验，原调度器无需再启动下一个条件。
                    os.kill(OLD_PARENT, signal.SIGTERM)
                    os.kill(OLD_PARENT, signal.SIGCONT)
                    event('old_supervisor_retired_after_worker_complete', pid=OLD_PARENT)
                else:
                    failures.append('original_full_failed')
                    # 让原调度器回收真实退出码和失败回执，不自动重试。
                    os.kill(OLD_PARENT, signal.SIGCONT)
                    event('original_full_failed_resume_supervisor_for_receipt', pid=OLD_FULL, exit_code=exit_code)
                legacy_done = True
            for condition, (worker, log, started) in list(active.items()):
                code = worker.poll()
                if code is None and time.monotonic()-started > release['budgets']['mapping_each_condition_s']+120:
                    worker.kill()
                    code = worker.wait()
                if code is None:
                    continue
                log.close()
                receipt = dict(condition=condition, pid=worker.pid, exit_code=code,
                    elapsed_s=time.monotonic()-started, goal_complete=False)
                if code == 0:
                    receipt['report_sha256'] = verify_report(condition)
                    finished.append(receipt)
                else:
                    failures.append(condition)
                write(OUT/f'{condition}_receipt.json', receipt)
                event('mapping_completed' if code == 0 else 'mapping_failed', **receipt)
                del active[condition]
            slots = 4-len(active)-(0 if legacy_done else 1)
            while slots > 0 and pending and not failures:
                condition = pending.pop(0)
                log = (OUT/f'{condition}.log').open('x')
                started = time.monotonic()
                command = [sys.executable, str(RUNNER), 'map', '--condition', condition]
                worker = subprocess.Popen(command, env=env, cwd='/root/autodl-tmp/lizichong',
                                          stdout=log, stderr=subprocess.STDOUT)
                active[condition] = (worker, log, started)
                event('mapping_started', condition=condition, pid=worker.pid)
                slots -= 1
            if failures and not active and legacy_done:
                break
            time.sleep(1)
        write(OUT/'mapping_phase_receipt.json', dict(finished=finished, failures=failures,
            not_started=pending, old_full_not_restarted=True, CPU_shared_performance=True, goal_complete=False))
        if failures or pending:
            return 1
        assert len(finished) == 10
        jobs = [(['requests'], release['budgets']['requests_s'])]
        jobs += [(['gpu', '--variant', v], release['budgets']['GPU_each_model_s']) for v in release['variants']]
        jobs += [(['evaluate'], release['budgets']['evaluate_s'])]
        for index, (args, budget) in enumerate(jobs):
            if args[0] == 'gpu':
                busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True, check=True).stdout.strip()
                if busy:
                    write(OUT/f'GPU_busy_{index:02d}.json', dict(pids=busy, goal_complete=False))
                    event('stopped_GPU_busy', pids=busy)
                    return 1
            event('serial_stage_started', index=index, args=args)
            t = time.monotonic()
            with (OUT/f'serial_{index:02d}.log').open('x') as log:
                try:
                    process = subprocess.run([sys.executable, str(RUNNER), *args], env=env,
                        cwd='/root/autodl-tmp/lizichong', stdout=log, stderr=subprocess.STDOUT, timeout=budget+120)
                    code = process.returncode
                except subprocess.TimeoutExpired:
                    code = -9
            receipt = dict(index=index, args=args, exit_code=code, elapsed_s=time.monotonic()-t, goal_complete=False)
            write(OUT/f'serial_{index:02d}.json', receipt)
            event('serial_stage_completed' if code == 0 else 'serial_stage_failed', **receipt)
            if code != 0:
                return 1
        return 0
    finally:
        event_file.close()


if __name__ == '__main__':
    raise SystemExit(main())
