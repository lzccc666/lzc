"""原真实时钟导航链＋真实CPU模型后的显式语义故障；不修改业务源码。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--release', required=True, type=Path)
    args = parser.parse_args()
    release = json.loads(args.release.read_text())
    assert args.condition in release['conditions'] and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import inspect
    import torch
    import ultralytics.utils
    import ultralytics.utils.torch_utils
    from omegaconf import OmegaConf
    from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump
    from conceptgraph.completion import navigation_semantics, realtime_navigation
    from conceptgraph.completion.realtime_semantics import RealtimeSemanticQueue
    from conceptgraph.completion.semantic_evaluation import base_revision
    from conceptgraph.scripts.streamlined_detections import load_detection_models
    from conceptgraph.scripts import streamlined_detections
    from conceptgraph.vlm.qwen_semantic_delegate import QwenDelegateConfig, QwenSemanticDelegate, _invoke_backend
    for name, expected in release['source_sha256'].items():
        assert digest(PROJECT/name) == expected, name
    for name, expected in release['Qwen_model_sha256'].items():
        assert digest(name) == expected, name
    destination = OUTPUT/'faults/semantic_async_cpu_003'/args.condition
    destination.mkdir(exist_ok=False)
    case = destination/'navigation'
    case.mkdir()
    pid = os.getpid()
    dump(destination/'worker_process.json', dict(pid=pid,
        start_ticks=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[19],
        CUDA_VISIBLE_DEVICES='', source_sha256=digest(__file__)))
    torch.set_num_threads(1)
    ultralytics.utils.NUM_THREADS = ultralytics.utils.torch_utils.NUM_THREADS = 1
    condition = args.condition
    started = time.monotonic()
    backend_events, frontend_events, instances = [], [], []
    actual_lock = threading.Lock()
    event_lock = threading.Lock()
    counts = dict(actual_model_started=0, actual_model_finished=0, active_model_calls=0, maximum_model_concurrency=0)

    def event(kind, **extra):
        with event_lock:
            row = dict(kind=kind, wall_s=time.monotonic(), elapsed_s=time.monotonic()-started, **extra)
            backend_events.append(row)
            with (destination/'model_events.jsonl').open('a') as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')

    class FaultBackend:
        def __init__(self, actual):
            self.actual = actual

        def generate(self, task, prompt, images=None, generation_config=None):
            with actual_lock:
                with event_lock:
                    identity = counts['actual_model_started']
                    counts['actual_model_started'] += 1
                    counts['active_model_calls'] += 1
                    counts['maximum_model_concurrency'] = max(counts['maximum_model_concurrency'], counts['active_model_calls'])
                image_proof = [dict(size=list(image.size), mode=image.mode,
                    pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest()) for image in images or []]
                event('actual_model_started', identity=identity, task=task,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), images=image_proof,
                    thread=threading.current_thread().name, torch_threads=torch.get_num_threads())
                try:
                    raw = _invoke_backend(self.actual, task, prompt, images or [], generation_config or {})
                    event('actual_model_finished', identity=identity, task=task,
                        raw_sha256=hashlib.sha256(raw.encode()).hexdigest())
                    dump(destination/'actual_raw'/f'{identity:03d}.json', dict(task=task, raw_output=raw,
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), images=image_proof))
                    with event_lock:
                        counts['actual_model_finished'] += 1
                    if condition == 'late_and_queue_expiry' and task == 'query_parse':
                        event('injected_post_delay_started', identity=identity, delay_s=45)
                        time.sleep(45)
                        event('injected_post_delay_finished', identity=identity)
                    if condition == 'backend_exception':
                        event('injected_exception', identity=identity, task=task)
                        raise RuntimeError('明确注入：真实生成后后端异常')
                    output = '' if condition == 'empty_output' else '{"schema_version":' if condition == 'malformed_output' else raw
                    event('backend_returned', identity=identity, task=task,
                        injected=condition in {'empty_output', 'malformed_output'},
                        output_sha256=hashlib.sha256(output.encode()).hexdigest(), output_chars=len(output))
                    return output
                finally:
                    with event_lock:
                        counts['active_model_calls'] -= 1

    class OneEachRealtimeQueue(RealtimeSemanticQueue):
        def __init__(self, backend, **kwargs):
            self.used, self.rejected = set(), dict.fromkeys(['query_parse', 'object_attributes', 'candidate_rerank'], 0)
            self.created_at = time.monotonic()
            super().__init__(backend, **kwargs)
            instances.append(self)

        def submit(self, task, payload, binding, *, ttl_s):
            if task in self.used:
                self.rejected[task] += 1
                return None
            identity = super().submit(task, payload, binding, ttl_s=ttl_s)
            if identity is not None:
                self.used.add(task)
                dump(destination/'submitted_requests'/f'{identity:03d}.json',
                    dict(task=task, payload=payload, binding=binding, requested_ttl_s=ttl_s,
                        request_id=identity, wall_s=time.monotonic()))
            return identity

        def poll(self, current_binding):
            if condition == 'expired_before_delivery' and time.monotonic()-self.created_at < 140:
                return []
            if condition == 'binding_changed':
                current_binding = dict(current_binding, query_version=2)
            return super().poll(current_binding)

    navigation_semantics.RealtimeSemanticQueue = OneEachRealtimeQueue
    original_init = realtime_navigation.WallClockVehicle.__init__

    def saved_init(driver, state, path, **kwargs):
        dump(case/'initial_state.json', asdict(state))
        original_init(driver, state, path, **kwargs)
        dump(case/'driver_clock.json', dict(started_wall_s=driver.started, sensor_ttl_s=driver.sensor_ttl_s,
            actual_thread=driver._thread.name, time_source='time.monotonic'))

    realtime_navigation.WallClockVehicle.__init__ = saved_init
    original_detect = streamlined_detections.detect_rgb_file

    def checked_detect(*a, **kw):
        before = torch.get_num_threads()
        result = original_detect(*a, **kw)
        row = dict(index=len(frontend_events), before_threads=before,
            after_threads=torch.get_num_threads(), CUDA_initialized=torch.cuda.is_initialized())
        assert before == row['after_threads'] == 1 and not row['CUDA_initialized']
        frontend_events.append(row)
        return result

    streamlined_detections.detect_rgb_file = checked_detect
    # 只改变本辅助进程的故障观察终止策略；原控制、守卫和业务文件保持冻结。
    function_text = inspect.getsource(realtime_navigation.run_task)
    old_terminal = "if final in {'ARRIVED','EXECUTION_FAILED'}:"
    old_guard = "final = 'SAFETY_STOP'\n                    break"
    assert function_text.count(old_terminal) == function_text.count(old_guard) == 1
    function_text = function_text.replace(old_terminal, "if final == 'EXECUTION_FAILED':")
    function_text = function_text.replace(old_guard, "final = 'SAFETY_STOP'")
    generated = destination/'run_task_observation_source.py'
    generated.write_text(function_text)
    exec(compile(function_text, str(generated), 'exec'), realtime_navigation.__dict__)
    dump(destination/'observation_policy.json', dict(fixed_wall_window_s=180,
        continue_after_ARRIVED_or_SAFETY_STOP=True, stop_on_execution_failure=True,
        source_sha256=digest(generated), original_file_sha256=digest(Path(realtime_navigation.__file__)),
        controller_or_guard_changed=False, GT_control_feedback_added=False, goal_complete=False))
    delegate = None
    try:
        cfg = QwenDelegateConfig(enabled=True,
            model_name_or_path=str(PROJECT/'checkpoint/Qwen2-VL-2B-Instruct'), model_revision=base_revision(),
            device='cpu', prompt_profile='completion_v3', torch_dtype='bfloat16', max_new_tokens=256, max_image_side=448,
            use_qwen_query_parse=True, use_qwen_object_attributes=True, use_qwen_rerank=True,
            timeout_s=.05 if condition == 'backend_timeout' else 120,
            cache_dir=None, log_dir=case/'qwen_audit')
        delegate = QwenSemanticDelegate(cfg)
        backend = delegate._load_backend()
        devices = sorted({str(p.device) for p in backend.model.parameters()})
        assert devices == ['cpu'] and not torch.cuda.is_initialized()
        os.environ['OMP_NUM_THREADS'] = '1'
        torch.set_num_threads(1)
        delegate._backend = FaultBackend(backend)
        dump(destination/'actual_model.json', dict(devices=devices,
            parameter_count=sum(p.numel() for p in backend.model.parameters()), revision=base_revision(),
            dtype=sorted({str(p.dtype) for p in backend.model.parameters()}), CUDA_initialized=False,
            timeout_s=cfg.timeout_s, cache_enabled=False, torch_threads=1))
        frontend_cfg = OmegaConf.create(dict(device='cpu'))
        models = load_detection_models(frontend_cfg)
        classes = [s.strip() for s in (PROJECT/'conceptgraph/vehicle_classes.txt').read_text().splitlines() if s.strip()]
        tasks = json.loads((OUTPUT/'interactive_simulation/synthetic_tasks_v1.json').read_text())
        task = next(t for t in tasks['tasks'] if t['task_id'] == 'config00-positive0')
        config = next(c for c in tasks['configs'] if c['config_id'] == 'config00')
        result = realtime_navigation.run_task(config, task, 17, case, models, frontend_cfg, classes, delegate,
            maximum_wall_s=180, semantic_ttl_s=5 if condition == 'late_and_queue_expiry' else 120 if condition == 'expired_before_delivery' else 90,
            semantic_tasks={'query_parse', 'object_attributes', 'candidate_rerank'},
            candidate_plan_budget_s=20, sensor_ttl_s=15, verified_candidate_chain=True)
        delegate.close()
        assert counts['active_model_calls'] == 0 and counts['maximum_model_concurrency'] <= 1
        dump(destination/'complete.json', dict(condition=condition, navigation_result=result,
            actual_CPU_Qwen=counts, delegate_backend_attempts=delegate.backend_call_count,
            frontend_calls=len(frontend_events), frontend_threads=frontend_events,
            submitted_tasks=sorted(instances[0].used), per_task_request_budget=1,
            rejected_after_budget=instances[0].rejected, all_backend_threads_joined=True,
            real_clock_async=True, actual_GPU_calls=0, model_accuracy_credit=False,
            new_independent_tasks=0, new_test_opened=False, wall_s=time.monotonic()-started, goal_complete=False))
        print(json.dumps(dict(condition=condition, status=result['status'], frontend=len(frontend_events), actual=counts)), flush=True)
    except BaseException as exc:
        dump(destination/'interrupted.json', dict(error=repr(exc), counts=counts, goal_complete=False))
        raise
    finally:
        if delegate is not None:
            delegate.close()
        dump(destination/'actual_imports.json', dict(
            realtime_navigation_sha256=digest(Path(realtime_navigation.__file__)),
            navigation_semantics_sha256=digest(Path(navigation_semantics.__file__)),
            worker_sha256=digest(__file__), CUDA_initialized=torch.cuda.is_initialized(),
            global_torch_threads=torch.get_num_threads(), counts=counts, goal_complete=False))


if __name__ == '__main__':
    main()
