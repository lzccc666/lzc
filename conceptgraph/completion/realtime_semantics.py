"""真实单调时钟的有界语义队列；后台输出仅交回主链核验，不直接操纵车辆。"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import queue
import threading
import time


@dataclass(frozen=True)
class SemanticRequest:
    request_id: int
    task: str
    payload: dict
    binding: dict
    submitted_wall_s: float
    deadline_wall_s: float
    payload_sha256: str


class RealtimeSemanticQueue:
    """一个模型线程、有界等待队列、真实到达时刻及过期结果，支持注入CPU测试后端。"""

    def __init__(self, backend, *, maximum_queued=3, clock=time.monotonic):
        if maximum_queued < 1:
            raise ValueError('queue capacity must be positive')
        self.backend = backend
        self.clock = clock
        self.pending = queue.Queue(maxsize=maximum_queued)
        self.completed = queue.Queue()
        self.events = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._next_id = 0
        self._thread = threading.Thread(target=self._run, name='real-clock-semantic-worker', daemon=True)
        self._thread.start()

    def _record(self, kind, request, **extra):
        event = dict(kind=kind, request_id=request.request_id, task=request.task,
            wall_s=self.clock(), submitted_wall_s=request.submitted_wall_s,
            deadline_wall_s=request.deadline_wall_s, payload_sha256=request.payload_sha256,
            binding=copy.deepcopy(request.binding), **extra)
        with self._lock:
            self.events.append(event)
        return event

    def submit(self, task, payload, binding, *, ttl_s):
        if task not in {'query_parse', 'object_attributes', 'candidate_rerank'}:
            raise ValueError('unknown semantic task')
        if not math.isfinite(ttl_s) or ttl_s <= 0:
            raise ValueError('request deadline must be finite and positive')
        if self._stop.is_set():
            raise RuntimeError('semantic queue already closed')
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        with self._lock:
            identity = self._next_id
            self._next_id += 1
        now = self.clock()
        request = SemanticRequest(identity, task, json.loads(serialized), copy.deepcopy(binding),
            now, now + ttl_s, hashlib.sha256(serialized.encode()).hexdigest())
        try:
            self.pending.put_nowait(request)
        except queue.Full:
            self._record('queue_full_rejected', request)
            return None
        self._record('submitted', request, queued=self.pending.qsize())
        return identity

    def _run(self):
        while not self._stop.is_set() or not self.pending.empty():
            try:
                request = self.pending.get(timeout=.05)
            except queue.Empty:
                continue
            try:
                now = self.clock()
                if self._stop.is_set():
                    self._record('cancelled_before_backend', request)
                    continue
                if now > request.deadline_wall_s:
                    self._record('expired_in_queue', request, queue_wait_s=now-request.submitted_wall_s)
                    continue
                self._record('backend_started', request, queue_wait_s=now-request.submitted_wall_s)
                try:
                    output = self.backend(request.task, copy.deepcopy(request.payload))
                except Exception as exc:
                    self._record('backend_exception', request, error=repr(exc))
                    self.completed.put((request, None, 'backend_exception'))
                    continue
                finished = self.clock()
                status = 'late_expired' if finished > request.deadline_wall_s else 'ready'
                self._record('backend_completed', request, response_status=status,
                    backend_s=finished-now, total_s=finished-request.submitted_wall_s)
                self.completed.put((request, output, status))
            finally:
                self.pending.task_done()

    def poll(self, current_binding):
        """投递时再核验deadline和版本；慢主链不能重新激活已经过期的输出。"""
        accepted = []
        while True:
            try:
                request, output, status = self.completed.get_nowait()
            except queue.Empty:
                break
            if status != 'ready':
                self._record('discarded', request, reason=status)
            elif self.clock() > request.deadline_wall_s:
                self._record('discarded', request, reason='expired_before_delivery')
            elif any(current_binding.get(key) != value for key, value in request.binding.items()):
                self._record('discarded', request, reason='binding_changed')
            else:
                self._record('delivered', request)
                accepted.append((request, output))
            self.completed.task_done()
        return accepted

    def close(self, *, timeout_s=1.):
        self._stop.set()
        self._thread.join(timeout=timeout_s)
        return dict(worker_stopped=not self._thread.is_alive(),
            backend_call_cannot_be_forcibly_cancelled=self._thread.is_alive())

    def snapshot_events(self):
        with self._lock:
            return copy.deepcopy(self.events)
