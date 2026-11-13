"""明确的单请求任务预算与下一感知帧投递；同步逻辑验收，不伪称真实异步。"""
import copy
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

from conceptgraph.completion import navigation_semantics
from conceptgraph.completion.navigation_semantics import NavigationSemanticBridge
from conceptgraph.completion.realtime_semantics import SemanticRequest
from conceptgraph.completion.stage9 import dump


class LogicalClock:
    def __init__(self):
        self.frame = -1
        self.time_s = 0.

    def monotonic(self):
        return self.time_s

    def set(self, index, state):
        self.frame, self.time_s = index, state.time_s


class OnePerTaskNextFrameQueue:
    def __init__(self, backend, clock, destination):
        self.backend, self.clock, self.destination = backend, clock, destination
        self.used, self.pending, self.events = set(), [], []
        self.requests = []
        self.rejected_budget = dict.fromkeys(['query_parse', 'object_attributes', 'candidate_rerank'], 0)

    def submit(self, task, payload, binding, *, ttl_s):
        if task in self.used:
            self.rejected_budget[task] += 1
            return None
        if task not in self.rejected_budget or ttl_s <= 0:
            raise ValueError('非法逻辑语义任务或期限')
        payload = copy.deepcopy(payload)
        body = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
        request = SemanticRequest(len(self.requests), task, payload, copy.deepcopy(binding),
            self.clock.time_s, self.clock.time_s+ttl_s, hashlib.sha256(body.encode()).hexdigest())
        self.used.add(task)
        response = self.backend(task, copy.deepcopy(payload))
        row = dict(request=asdict(request), response=copy.deepcopy(response),
            submitted_frame=self.clock.frame, delivery_frame=self.clock.frame+1,
            synchronous_logical_evaluation=True, real_clock_async=False)
        self.requests.append(row)
        dump(self.destination/'logical_requests'/f'{request.request_id:04d}.json', row)
        self.pending.append((request, response, self.clock.frame+1))
        self.events.append(dict(kind='computed', task=task, request_id=request.request_id,
            frame=self.clock.frame, delivery_frame=self.clock.frame+1))
        return request.request_id

    def poll(self, current_binding):
        accepted, waiting = [], []
        for request, output, delivery_frame in self.pending:
            if self.clock.frame < delivery_frame:
                waiting.append((request, output, delivery_frame))
                continue
            valid = (self.clock.time_s <= request.deadline_wall_s
                and all(current_binding.get(key) == value for key, value in request.binding.items()))
            self.events.append(dict(kind='delivered' if valid else 'discarded', task=request.task,
                request_id=request.request_id, frame=self.clock.frame))
            if valid:
                accepted.append((request, copy.deepcopy(output)))
        self.pending = waiting
        return accepted

    def snapshot_events(self):
        return copy.deepcopy(self.events)

    def close(self, **kwargs):
        return dict(worker_stopped=True, thread_used=False,
            remaining_not_delivered=len(self.pending), synchronous_logical_evaluation=True)


class FixedLogicalSemanticBridge(NavigationSemanticBridge):
    def __init__(self, query, destination, delegate, clock):
        # 只重绑定这个独立进程中桥接模块的时钟，不修改全局time模块或执行预算。
        navigation_semantics.time = SimpleNamespace(monotonic=clock.monotonic)
        super().__init__(query, destination, None, ttl_s=60., maximum_candidates=3,
            candidate_plan_budget_s=20., use_plan_reuse=True)
        self.delegate, self.logical_clock = delegate, clock
        self.queue = OnePerTaskNextFrameQueue(self._backend, clock, destination)
        self.queue.submit('query_parse', dict(query_text=query), dict(query_version=1), ttl_s=60.)

    def close(self, **kwargs):
        return dict(events=self.queue.snapshot_events(), application_events=copy.deepcopy(self.events),
            applied_counts=copy.deepcopy(self.applied), actual_CPU_backend_calls=self.delegate.backend_call_count,
            actual_CUDA_backend=False, real_clock_async=False,
            task_request_budget=1, rejected_after_budget=copy.deepcopy(self.queue.rejected_budget),
            worker=self.queue.close())
