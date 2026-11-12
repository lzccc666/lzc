"""准备公开轻量夹具；显式execute才运行真实GPU规则闭环，不保证到达。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--run-id', default='public_rule_demo_001')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    fixture = root/'reproduction/fixtures/synthetic_tasks_v1.json'
    asset = root/'reproduction/assets/ToyCar_002.glb'
    if hashlib.sha256(asset.read_bytes()).hexdigest() != '01a60862de55cd4b9f3acfab0b0def86451800f9c42467fcd61052c16cb9838c':
        raise ValueError('CC0三维资产SHA不匹配')
    release = root/'outputs/project_v2/thesis_completion_v1/interactive_simulation/synthetic_tasks_v1.json'
    if release.exists():
        if release.read_bytes() != fixture.read_bytes():
            raise ValueError('已有任务release不同，拒绝覆盖')
    else:
        release.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fixture, release)
    print(json.dumps(dict(prepared=True, asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
        task_release_sha256=hashlib.sha256(release.read_bytes()).hexdigest(), actual_navigation_executed=False)))
    if args.prepare_only:
        return
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True)
    if result.stdout.strip():
        raise RuntimeError('GPU存在其他进程，拒绝并发；不自动停止或杀死其他任务')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    # 同一已核验资产，仅在此新演示进程中适配路径；原冻结业务文件不修改。
    from conceptgraph.completion import interactive_environment
    interactive_environment.ASSET = asset
    sys.argv = ['realtime_navigation', '--run-id', args.run_id, '--model', 'rule',
                '--task', 'config00-positive0', '--seed', '17', '--maximum-wall-s', '90',
                '--semantic-tasks', '', '--candidate-plan-budget-s', '10']
    runpy.run_module('conceptgraph.completion.realtime_navigation', run_name='__main__')


if __name__ == '__main__':
    main()
