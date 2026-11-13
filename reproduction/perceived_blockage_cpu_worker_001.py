"""真实前端观察新屏障；延后一次实际CPU排序，不修改agent障碍或控制。"""
import argparse
from dataclasses import asdict
import inspect
import json
import math
import os
from pathlib import Path
import threading
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', required=True)
    parser.add_argument('--release', type=Path, required=True)
    args = parser.parse_args()
    release = json.loads(args.release.read_text())
    assert args.condition in release['conditions'] and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import numpy as np
    from omegaconf import OmegaConf
    from shapely.geometry import LineString, Polygon
    import torch
    import ultralytics.utils
    import ultralytics.utils.torch_utils
    from conceptgraph.completion import interactive_environment, navigation_semantics, realtime_navigation
    from conceptgraph.completion.kinematic_benchmark import pose_on
    from conceptgraph.completion.realtime_semantics import RealtimeSemanticQueue
    from conceptgraph.completion.semantic_evaluation import base_revision
    from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump
    from conceptgraph.scripts import streamlined_detections
    from conceptgraph.vlm.qwen_semantic_delegate import QwenDelegateConfig, QwenSemanticDelegate, _invoke_backend
    for name, expected in release['source_sha256'].items():
        assert digest(PROJECT/name) == expected, name
    case = OUTPUT/'faults/perceived_blockage_cpu_001'/args.condition
    case.mkdir(exist_ok=False)
    nav = case/'navigation'
    nav.mkdir()
    torch.set_num_threads(1)
    ultralytics.utils.NUM_THREADS = ultralytics.utils.torch_utils.NUM_THREADS = 1
    dump(case/'worker_process.json', dict(pid=os.getpid(), start_ticks=(Path('/proc')/str(os.getpid())/'stat').read_text().rsplit(')',1)[1].split()[19]))
    fixtures = json.loads((OUTPUT/'interactive_simulation/synthetic_tasks_v1.json').read_text())
    task = next(t for t in fixtures['tasks'] if t['task_id'] == 'config00-positive2')
    config = next(c for c in fixtures['configs'] if c['config_id'] == 'config00')
    line = LineString(config['centerline_xy'])
    dump(case/'config_and_task.json', dict(config=config, task=task, seed=17))
    shared = dict(driver=None, gap=False, ready=False, barrier=None, frontend=0,
                  requests=0, model_calls=0, model_finished=0, suppressions=0)
    observed = threading.Event()
    event_lock = threading.Lock()

    def event(kind, **kwargs):
        with event_lock:
            with (case/'events.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(kind=kind, wall_s=time.monotonic(), **kwargs), ensure_ascii=False, allow_nan=False)+'\n')

    original_vehicle = realtime_navigation.WallClockVehicle

    class BlockageVehicle(original_vehicle):
        def __init__(self, state, path, **kwargs):
            dump(nav/'initial_state.json', asdict(state))
            self.initial_xy = np.array([state.x, state.y])
            super().__init__(state, path, **kwargs)
            shared['driver'] = self
            dump(nav/'driver_clock.json', dict(started_wall_s=self.started, sensor_ttl_s=self.sensor_ttl_s))

        def snapshot_after(self, previous_time_s, **kwargs):
            state = super().snapshot_after(previous_time_s, **kwargs)
            if state is None or shared['gap']:
                return state
            with self._lock:
                packet = self.packet
            armed = packet is not None and packet[0].path_purpose == 'target_approach' and len(packet[0].path)>1
            armed &= args.condition == 'rule_blockage' or shared['requests'] == 1
            if armed and state.speed_mps >= .2 and np.linalg.norm(np.array([state.x,state.y])-self.initial_xy) >= .5:
                shared['gap'] = True
                dump(case/'old_target_path.json', dict(path=packet[0].path, goal=packet[0].goal.tolist(),
                    selected=packet[0].selected, state=asdict(state), packet_sensed_at=packet[1]))
                event('observation_gap_begin', state=asdict(state), controls=len(self.records))
                deadline = time.monotonic()+24
                while time.monotonic() < deadline:
                    current = self.snapshot()
                    with self._lock:
                        stale = bool(self.records and self.records[-1]['stale_sensor_braking'])
                    if stale and current.speed_mps == 0:
                        shared['ready'] = True
                        event('gap_original_ttl_braking_stopped', state=asdict(current), controls=len(self.records))
                        return current
                    time.sleep(.05)
                event('gap_stop_not_exposed', state=asdict(self.snapshot()))
                return self.snapshot()
            return state

        def publish(self, agent, sensed_at):
            super().publish(agent, sensed_at)
            barrier = shared['barrier']
            if barrier is not None and not observed.is_set():
                hit = [list(p.exterior.coords) for p in agent.obstacles if p.intersects(barrier.buffer(.3))]
                if hit:
                    event('actual_lidar_obstacle_published', sensed_at=sensed_at, state=asdict(self.snapshot()),
                        measured_obstacles=hit, old_path_still_present=bool(agent.path),
                        control_count=len(self.records), agent_guard=agent.guard_latched)
                    observed.set()

    realtime_navigation.WallClockVehicle = BlockageVehicle
    original_env = interactive_environment.InteractiveRoadEnvironment

    class BlockageEnvironment(original_env):
        def observe(self, dest, index):
            if shared['ready'] and shared['barrier'] is None:
                from shapely.geometry import Point
                progress = line.project(Point(self.state.x, self.state.y))
                xy, yaw = pose_on(line, progress+6)
                normal = np.array([-math.sin(yaw), math.cos(yaw)])
                tangent = np.array([math.cos(yaw), math.sin(yaw)])
                corners = [xy+along*tangent+side*normal for along, side in [(-1,-50),(1,-50),(1,50),(-1,50)]]
                shared['barrier'] = Polygon(corners)
                self._add_box([2,100,2.5], [*xy,1.25], [175,90,30,255], yaw)
                evidence = dict(birth_wall_s=time.monotonic(), frame_idx=index, state=asdict(self.state),
                    center_xy=xy.tolist(), yaw_rad=yaw, polygon_xy=list(shared['barrier'].exterior.coords),
                    extents_m=[2,100,2.5], source='environment-only physical mesh; not agent input')
                dump(case/'evaluation_only_dynamic_obstacle.json', evidence)
                event('physical_barrier_created', **evidence)
            result = super().observe(dest, index)
            event('actual_sensor_frame', frame_idx=index, state=asdict(self.state),
                  rgb_sha256=result[-1]['rgb_sha256'], depth_sha256=result[-1]['depth_sha256'])
            return result

    interactive_environment.InteractiveRoadEnvironment = BlockageEnvironment
    original_detect = streamlined_detections.detect_rgb_file

    def detect(*a, **kwargs):
        begun = time.monotonic()
        result = original_detect(*a, **kwargs)
        shared['frontend'] += 1
        assert torch.get_num_threads() == 1 and not torch.cuda.is_initialized()
        event('actual_frontend_complete', index=shared['frontend']-1, duration_s=time.monotonic()-begun)
        return result

    streamlined_detections.detect_rgb_file = detect

    class OneRankQueue(RealtimeSemanticQueue):
        def submit(self, task_name, payload, binding, *, ttl_s):
            assert task_name == 'candidate_rerank'
            if shared['requests']:
                shared['suppressions'] += 1
                return None
            identity = super().submit(task_name, payload, binding, ttl_s=ttl_s)
            if identity is not None:
                shared['requests'] += 1
                dump(case/'submitted_request.json', dict(request_id=identity, task=task_name,
                    payload=payload, binding=binding, ttl_s=ttl_s, wall_s=time.monotonic()))
                event('rank_submitted', request_id=identity)
            return identity

    navigation_semantics.RealtimeSemanticQueue = OneRankQueue
    text = inspect.getsource(realtime_navigation.run_task)
    old_terminal = "if final in {'ARRIVED','EXECUTION_FAILED'}:"
    old_guard = "final = 'SAFETY_STOP'\n                    break"
    assert text.count(old_terminal) == text.count(old_guard) == 1
    text = text.replace(old_terminal, "if final == 'EXECUTION_FAILED':").replace(old_guard, "final = 'SAFETY_STOP'")
    generated = case/'run_task_observation_source.py'
    generated.write_text(text)
    exec(compile(text, str(generated), 'exec'), realtime_navigation.__dict__)
    delegate = None
    try:
        if args.condition == 'delayed_actual_rank':
            for name, expected in release['Qwen_model_sha256'].items():
                assert digest(name) == expected
            cfg = QwenDelegateConfig(enabled=True, model_name_or_path=str(PROJECT/'checkpoint/Qwen2-VL-2B-Instruct'),
                model_revision=base_revision(), device='cpu', torch_dtype='bfloat16', prompt_profile='completion_v3',
                max_new_tokens=256, max_image_side=448, timeout_s=200, use_qwen_query_parse=False,
                use_qwen_object_attributes=False, use_qwen_rerank=True, cache_dir=None, log_dir=nav/'qwen_audit')
            delegate = QwenSemanticDelegate(cfg)
            actual = delegate._load_backend()
            assert {str(p.device) for p in actual.model.parameters()} == {'cpu'}
            dump(case/'actual_model.json', dict(devices=['cpu'], parameter_count=sum(p.numel() for p in actual.model.parameters()),
                dtype=sorted({str(p.dtype) for p in actual.model.parameters()}), revision=base_revision(), cache_enabled=False))

            class DelayedActualRank:
                def generate(self, task, prompt, images=None, generation_config=None):
                    shared['model_calls'] += 1
                    event('actual_model_started', task=task)
                    raw = _invoke_backend(actual, task, prompt, images or [], generation_config or {})
                    shared['model_finished'] += 1
                    dump(case/'actual_model_text.json', dict(task=task, prompt=prompt, raw_output=raw))
                    event('actual_model_finished', task=task)
                    seen = observed.wait(timeout=160)
                    if seen:
                        time.sleep(2)
                    event('actual_model_result_returned', barrier_observed=seen)
                    return raw

            delegate._backend = DelayedActualRank()
        frontend_cfg = OmegaConf.create(dict(device='cpu'))
        models = streamlined_detections.load_detection_models(frontend_cfg)
        classes = [s.strip() for s in (PROJECT/'conceptgraph/vehicle_classes.txt').read_text().splitlines() if s.strip()]
        result = realtime_navigation.run_task(config, task, 17, nav, models, frontend_cfg, classes, delegate,
            maximum_wall_s=240, semantic_ttl_s=240,
            semantic_tasks={'candidate_rerank'} if delegate is not None else set(),
            sensor_ttl_s=15, candidate_plan_budget_s=5, verified_candidate_chain=True)
        if delegate is not None:
            delegate.close()
        dump(case/'complete.json', dict(condition=args.condition, result=result,
            actual_frontend_calls=shared['frontend'], actual_CPU_Qwen_calls=shared['model_calls'],
            actual_CPU_Qwen_finished=shared['model_finished'], rank_requests=shared['requests'],
            further_requests_suppressed=shared['suppressions'], gap_triggered=shared['gap'],
            original_ttl_stopped=shared['ready'], physical_barrier_created=shared['barrier'] is not None,
            actual_barrier_obstacle_published=observed.is_set(),
            final_driver_guard_latched=shared['driver'].guard_latched,
            all_backend_threads_joined=True, actual_GPU_calls=0, new_independent_tasks=0,
            navigation_accuracy_comparison_credit=False, new_test_opened=False, goal_complete=False))
        print(json.dumps(dict(condition=args.condition, status=result['status'],
            actual_frontend=shared['frontend'], CPU_Qwen=shared['model_calls'], observed=observed.is_set())), flush=True)
    except BaseException as error:
        dump(case/'interrupted.json', dict(error=repr(error), actual_frontend=shared['frontend'],
            actual_CPU_Qwen_calls=shared['model_calls'], goal_complete=False))
        raise
    finally:
        if delegate is not None:
            delegate.close()
        dump(case/'actual_imports.json', dict(worker_sha256=digest(__file__),
            business_sources_unchanged=all(digest(PROJECT/p) == h for p,h in release['source_sha256'].items()),
            CUDA_initialized=torch.cuda.is_initialized(), torch_threads=torch.get_num_threads(), goal_complete=False))


if __name__ == '__main__':
    main()
