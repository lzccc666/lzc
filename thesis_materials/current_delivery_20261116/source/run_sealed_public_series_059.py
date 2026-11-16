"""固定顺序串行运行封存前端、十条件建图和五模型；失败即停，不重跑挑优。"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'main_experiments/sealed_public_058'
RUNNER = ROOT / 'source/sealed_public_benchmark_058.py'


def main():
    release = json.loads((DEST / 'release.json').read_text())
    jobs = [(['frontend'], release['budgets']['frontend_s'])]
    jobs += [(['map', '--condition', c], release['budgets']['mapping_each_condition_s']) for c in release['conditions']]
    jobs += [(['requests'], release['budgets']['requests_s'])]
    jobs += [(['gpu', '--variant', v], release['budgets']['GPU_each_model_s']) for v in release['variants']]
    jobs += [(['evaluate'], release['budgets']['evaluate_s'])]
    dest = DEST / 'series_059'
    dest.mkdir(exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',
        LD_LIBRARY_PATH='/root/miniconda3/envs/conceptgraph/lib',
        PYTHONPATH='/root/autodl-tmp/lizichong:' + str(ROOT / 'source'))
    with (dest / 'events.jsonl').open('x') as events:
        for index, (args, budget) in enumerate(jobs):
            command = [sys.executable, str(RUNNER), *args]
            started = time.monotonic()
            event = dict(index=index, args=args, UTC=datetime.now(timezone.utc).isoformat(), kind='started')
            events.write(json.dumps(event)+'\n')
            events.flush()
            with (dest / f'job_{index:02d}.log').open('x') as log:
                try:
                    result = subprocess.run(command, cwd='/root/autodl-tmp/lizichong', env=env,
                                            stdout=log, stderr=subprocess.STDOUT, timeout=budget+120)
                    code, failure = result.returncode, None
                except subprocess.TimeoutExpired:
                    code, failure = -9, 'subprocess_timeout_all_prefix_retained'
            receipt = dict(index=index, args=args, exit_code=code, failure=failure,
                           elapsed_s=time.monotonic()-started, UTC=datetime.now(timezone.utc).isoformat(),
                           raw_log=str(dest / f'job_{index:02d}.log'), goal_complete=False)
            with (dest / f'job_{index:02d}.json').open('x') as f:
                json.dump(receipt, f, ensure_ascii=False, indent=2)
            events.write(json.dumps(dict(receipt, kind='completed' if code == 0 else 'failed'))+'\n')
            events.flush()
            print(json.dumps(receipt, ensure_ascii=False), flush=True)
            if code != 0:
                return code if code > 0 else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
