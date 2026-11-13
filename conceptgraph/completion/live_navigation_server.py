"""单活动CPU会话的可交互导航服务；既有审核与GPU队列不受修改。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from urllib.parse import urlsplit, quote
import uuid

from conceptgraph.completion.realtime_batch import pipeline_sources
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump

BUDGET = dict(wall_budget_s=900, max_sim_s=30, max_frontend_calls=80,
    storage_budget_bytes=500_000_000, minimum_free_bytes=6_000_000_000)
COMMANDS = {'run', 'pause', 'step', 'stop'}


def process_ticks(pid):
    try:
        fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except (OSError, IndexError):
        return None


def checked_start(record, tasks):
    if not isinstance(record, dict) or set(record) != {'task_id', 'seed'}:
        raise ValueError('只接受冻结task_id和seed')
    if not isinstance(record['task_id'], str) or record['task_id'] not in {t['task_id'] for t in tasks['tasks']}:
        raise ValueError('任务不在冻结50任务中')
    if type(record['seed']) is not int or record['seed'] not in tasks['seeds']:
        raise ValueError('种子不在冻结清单中')
    return dict(record)


class SessionManager:
    def __init__(self, output=OUTPUT):
        self.output = Path(output)
        self.root = self.output/'interactive_simulation/live_navigation_sessions_001'
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.child = None
        self.active = None
        self.requests = {}
        self.task_path = self.output/'interactive_simulation/synthetic_tasks_v1.json'
        self.tasks = json.loads(self.task_path.read_text())
        live = []
        for path in self.root.glob('*/worker_process.json'):
            row = json.loads(path.read_text())
            if process_ticks(row['pid']) == row['start_ticks']:
                live.append((path.parent, row))
        if len(live) > 1:
            raise RuntimeError('存在多个实际worker，请保留证据并先核查')
        if live:
            self.active, self.worker = live[0]

    def alive(self):
        if self.child is not None:
            self.child.poll()
        return self.active is not None and process_ticks(self.worker['pid']) == self.worker['start_ticks']

    def start(self, record):
        request = checked_start(record, self.tasks)
        with self.lock:
            if self.alive():
                raise RuntimeError('已有实际导航会话；先暂停或停车结束')
            if shutil.disk_usage(self.output).free < BUDGET['minimum_free_bytes']:
                raise RuntimeError('磁盘余量不足，保留旧运行，不启动新会话')
            runtime = self.output/'source_bundle/software_render_runtime_001'
            if 'llvmpipe' not in (runtime/'probe_004.log').read_text():
                raise RuntimeError('软件渲染依赖尚未核验')
            name = 'live_cpu_'+str(time.time_ns())+'_'+uuid.uuid4().hex[:8]
            destination = self.root/name
            destination.mkdir(exist_ok=False)
            sources = sorted(set(pipeline_sources()+[PROJECT/'conceptgraph/completion'/n for n in
                ['live_navigation_server.py', 'live_navigation_worker.py', 'live_navigation_page.py']]))
            release = dict(**request, **BUDGET, source_sha256={str(p.relative_to(PROJECT)):digest(p) for p in sources},
                task_release_sha256=digest(self.task_path), created_utc=datetime.now(timezone.utc).isoformat(),
                protocol_sha256=digest(self.output/'task_protocol/live_navigation_console_001.md'),
                frontend_device='cpu', renderer='OSMesa llvmpipe', Qwen_calls=0,
                simulation_clock='fixed_dt_0.1s_user_controlled_not_realtime',
                independent_test=False, performance_credit=False, frozen_main_table_credit=False,
                default_mode='PAUSED', user_stop_is_not_arrival=True, goal_complete=False)
            dump(destination/'release.json', release)
            with tarfile.open(destination/'source_snapshot.tar.gz', 'x:gz') as archive:
                for source in sources:
                    archive.add(source, arcname=str(source.relative_to(PROJECT)))
            (destination/'commands.jsonl').touch(exist_ok=False)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                LP_NUM_THREADS='1', PYOPENGL_PLATFORM='osmesa', PYGLET_SHADOW_WINDOW='false',
                PYTHONPATH=str(runtime/'python_deps'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                LD_LIBRARY_PATH=f'{runtime}/extracted/usr/lib/x86_64-linux-gnu:/root/autodl-fs/third_party/graphics_runtime/lib:/root/miniconda3/envs/conceptgraph/lib')
            with (destination/'worker.log').open('x') as log:
                self.child = subprocess.Popen([sys.executable, '-m', 'conceptgraph.completion.live_navigation_worker',
                    '--session', name], cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
            self.active = destination
            self.worker = dict(pid=self.child.pid, start_ticks=process_ticks(self.child.pid))
            dump(destination/'worker_process.json', self.worker)
            self.requests = {}
            return dict(run_id=name, worker=self.worker, accepted=True)

    def control(self, record):
        if not isinstance(record, dict) or set(record) != {'run_id', 'command', 'command_id'}:
            raise ValueError('控制请求字段不完整')
        command = record['command']
        command_id = record['command_id']
        if not isinstance(command, str) or command not in COMMANDS:
            raise ValueError('只允许单步、运行、暂停或停车')
        if not isinstance(command_id, str) or not 1 <= len(command_id) <= 80:
            raise ValueError('控制请求需要唯一标识')
        with self.lock:
            if not self.alive() or record['run_id'] != self.active.name:
                raise RuntimeError('会话不是当前实际活动worker')
            state = self.active/'state.json'
            if state.exists() and json.loads(state.read_text())['phase'] in {'FINISHED', 'FAILED'}:
                raise RuntimeError('会话已终止，控制命令不再应用')
            for row in (json.loads(line) for line in (self.active/'commands.jsonl').read_text().splitlines()):
                if row['command_id'] == command_id:
                    if row['command'] != command:
                        raise ValueError('同一控制标识不能改成其他命令')
                    return dict(accepted=True, duplicate=True, command_id=command_id)
            row = dict(**record, received_utc=datetime.now(timezone.utc).isoformat(),
                origin='browser_control_not_human_annotation')
            with (self.active/'commands.jsonl').open('a') as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            return dict(accepted=True, duplicate=False, command_id=command_id)

    def snapshot(self):
        with self.lock:
            if self.active is None:
                return dict(phase='IDLE', worker_alive=False, run_id=None)
            path = self.active/'state.json'
            state = json.loads(path.read_text()) if path.exists() else dict(phase='STARTING')
            alive = self.alive()
            state.update(run_id=self.active.name, worker_alive=alive, worker=self.worker,
                session_url='/'+quote(str(self.active.relative_to(self.output)), safe='/')+'/')
            if not alive and state['phase'] not in {'FINISHED', 'FAILED'}:
                state['phase'] = 'WORKER_EXITED_WITHOUT_RESULT'
            frame = state.get('frame')
            if frame:
                def url(path):
                    return '/'+quote(str(Path(path).resolve().relative_to(self.output.resolve())), safe='/')
                frame['rgb_url'] = url(frame['sensor']['rgb'])
                frame['depth_url'] = url(frame['sensor']['depth'])
                frame['detection_url'] = url(frame['detection_vis'])
                frame['depth_view_url'] = url(frame['depth_vis'])
            return state


def reject_constant(value):
    raise ValueError('非法JSON数值：'+value)


class Handler(SimpleHTTPRequestHandler):
    manager = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(OUTPUT), **kwargs)

    def respond(self, status, value):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == '/api/live/tasks':
            # 不向推理代理传GT；浏览器任务选择只显示指令与配置身份。
            self.respond(200, dict(tasks=[{k:t[k] for k in ['task_id', 'config_id', 'kind', 'query_text']}
                for t in self.manager.tasks['tasks']], seeds=self.manager.tasks['seeds'], budget=BUDGET))
        elif self.path == '/api/live/state':
            self.respond(200, self.manager.snapshot())
        elif self.path in {'/', '/index.html'}:
            from conceptgraph.completion.live_navigation_page import HTML
            payload = HTML.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            super().do_GET()

    def do_POST(self):
        if self.path not in {'/api/live/start', '/api/live/control'}:
            self.respond(404, dict(error='未知接口'))
            return
        try:
            origin = self.headers.get('Origin')
            if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                raise ValueError('控制请求需来自同一页面服务')
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 4096:
                raise ValueError('控制请求大小非法')
            record = json.loads(self.rfile.read(size), parse_constant=reject_constant)
            result = self.manager.start(record) if self.path.endswith('/start') else self.manager.control(record)
        except RuntimeError as exc:
            self.respond(409, dict(error=str(exc)))
        except (ValueError, TypeError, OSError) as exc:
            self.respond(400, dict(error=str(exc)))
        else:
            self.respond(202, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8769)
    args = parser.parse_args()
    root = OUTPUT/'interactive_simulation/live_navigation_sessions_001'
    root.mkdir(parents=True, exist_ok=True)
    with (root/'service.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Handler.manager = SessionManager()
        ThreadingHTTPServer(('0.0.0.0', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
