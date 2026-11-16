"""058执行分支：只改变容量检查调度，复用完成的前端和十组建图。"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from sealed_resource_guard_074 import ResourceGuard

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT/'main_experiments/sealed_public_058'
DEST = ROOT/'main_experiments/sealed_execution_075'
RUNNER = ROOT/'source/sealed_public_benchmark_058.py'
OLD_WORKER = 147822


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def load_runner():
    spec = importlib.util.spec_from_file_location('frozen_benchmark_058_execution', RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DEST = DEST
    return module


def child(action, variant):
    module = load_runner()
    guard = ResourceGuard(ROOT, DEST)
    module.guard = guard
    release = module.frozen()
    budget_key = {'requests': 'requests_s', 'gpu': 'GPU_each_model_s',
                  'evaluate': 'evaluate_s'}[action]
    begun = time.monotonic()
    try:
        guard(begun, release['budgets'][budget_key])
        module.gpu(variant) if action == 'gpu' else getattr(module, action)()
        guard(begun, release['budgets'][budget_key], force=True)
        write(DEST/f'guard_{action}_{variant or "none"}.json', dict(
            scans=guard.scans, global_bytes=guard.global_bytes,
            interval_s=guard.interval_s, reserve_bytes=guard.reserve_bytes,
            additional_execution_output_limit_bytes=guard.stage_limit,
            original_budget_s=release['budgets'][budget_key], goal_complete=False))
    except BaseException as exc:
        write(DEST/f'failure_{action}_{time.time_ns()}.json', dict(
            error=repr(exc), traceback=traceback.format_exc(), goal_complete=False))
        raise


def main():
    DEST.mkdir(exist_ok=False)
    # 先登记，再停止未完成的请求准备；不读取封存测试分数。
    release = json.loads((ORIGINAL/'release.json').read_text())
    assert not (ORIGINAL/'requests/report.json').exists()
    assert not (ORIGINAL/'models').exists()
    command = Path(f'/proc/{OLD_WORKER}/cmdline').read_bytes().split(b'\0')
    assert str(RUNNER).encode() in command and b'requests' in command
    for condition in release['conditions']:
        report = json.loads((ORIGINAL/'mapping'/condition/'report.json').read_text())
        assert report['frames'] == 1196 and report['scenes'] == 30
        for item in report['reports']:
            assert sha(ORIGINAL/'mapping'/condition/item['scene']/'frames.jsonl.gz') == item['frames_sha256']
    write(DEST/'execution_amendment.json', dict(
        UTC=datetime.now(timezone.utc).isoformat(), scope='只修改资源检查调度，不修改冻结算法',
        old_worker_pid=OLD_WORKER, original_release_sha256=sha(ORIGINAL/'release.json'),
        original_runner_sha256=sha(RUNNER), wrapper_sha256=sha(Path(__file__)),
        guard_sha256=sha(ROOT/'source/sealed_resource_guard_074.py'),
        frozen_files=release['frozen_files_sha256'], budgets=release['budgets'],
        conditions=release['conditions'], model_order=list(release['variants']),
        resource_checks=dict(physical_free_min_GiB=20, global_max_GiB=25,
            global_full_scan_interval_s=30, global_reserve_MiB=64,
            additional_execution_output_cap_MiB=64,
            each_call_checks=['time_budget', 'physical_free', 'new_execution_output_bytes']),
        observed_previous_census=dict(files=143752, elapsed_s=4.325,
            requests_frame_checks=1196, estimated_census_only_s=5172.7,
            original_requests_budget_s=release['budgets']['requests_s']),
        original_incomplete_prefix_preserved=True, no_frontend_or_mapping_rerun=True,
        no_model_score_used_to_select_methods=True, CARLA_restart=False, goal_complete=False))
    for name in ['release.json', 'agent_frames.jsonl', 'query_inputs.jsonl', 'frontend', 'mapping']:
        (DEST/name).symlink_to(ORIGINAL/name, target_is_directory=(ORIGINAL/name).is_dir())
    # 指纹和输入清单验证仍调用冻结程序原实现。
    load_runner().frozen()
    ResourceGuard(ROOT, DEST)(time.monotonic(), 1200)
    os.kill(OLD_WORKER, signal.SIGTERM)
    deadline = time.monotonic()+30
    while Path(f'/proc/{OLD_WORKER}').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('原请求进程未退出，不启动重复任务')
        time.sleep(.2)
    write(DEST/'original_requests_retired.json', dict(
        UTC=datetime.now(timezone.utc).isoformat(), pid=OLD_WORKER,
        original_requests_prefix=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
            for p in (ORIGINAL/'requests').iterdir() if p.is_file()],
        reason='资源检查调度导致预算不足，非模型成绩失败', goal_complete=False))
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', LD_LIBRARY_PATH='/root/miniconda3/envs/conceptgraph/lib',
        PYTHONPATH='/root/autodl-tmp/lizichong:'+str(ROOT/'source'))
    jobs = [('requests', None, release['budgets']['requests_s'])]
    jobs += [('gpu', v, release['budgets']['GPU_each_model_s']) for v in release['variants']]
    jobs += [('evaluate', None, release['budgets']['evaluate_s'])]
    with (DEST/'events.jsonl').open('x') as event_file:
        for index, (action, variant, budget) in enumerate(jobs):
            if action == 'gpu':
                busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout.strip()
                if busy:
                    write(DEST/f'GPU_busy_{index}.json', dict(pids=busy, goal_complete=False))
                    return 1
            args = [sys.executable, str(Path(__file__)), '--child', action]
            if variant:
                args += ['--variant', variant]
            tick = time.monotonic()
            with (DEST/f'stage_{index:02d}.log').open('x') as log:
                process = subprocess.Popen(args, env=env, cwd='/root/autodl-tmp/lizichong',
                    stdout=log, stderr=subprocess.STDOUT)
                started = dict(UTC=datetime.now(timezone.utc).isoformat(), event='started',
                    index=index, action=action, variant=variant, pid=process.pid)
                event_file.write(json.dumps(started)+'\n')
                event_file.flush()
                print(json.dumps(started), flush=True)
                try:
                    code = process.wait(timeout=budget+120)
                except subprocess.TimeoutExpired:
                    process.kill()
                    code = process.wait()
                receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), event='finished',
                    index=index, action=action, variant=variant, exit_code=code,
                    elapsed_s=time.monotonic()-tick, goal_complete=False)
                write(DEST/f'stage_{index:02d}_receipt.json', receipt)
                event_file.write(json.dumps(receipt)+'\n')
                event_file.flush()
                print(json.dumps(receipt), flush=True)
                if code != 0:
                    return 1
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', choices=['requests', 'gpu', 'evaluate'])
    parser.add_argument('--variant')
    args = parser.parse_args()
    if args.child:
        child(args.child, args.variant)
    else:
        raise SystemExit(main())
