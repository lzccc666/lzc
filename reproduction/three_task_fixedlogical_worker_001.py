"""三任务冷/热完整逻辑导航；全部实际重新感知，热运行后端禁止调用。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_value(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--experiment-release', type=Path, required=True)
    parser.add_argument('--phase', choices=['cold', 'warm'], required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    release = json.loads(args.experiment_release.read_text())
    root, source = Path(manifest['isolated_root']), Path(manifest['original_project'])
    sys.path[:] = [str(root)]+[value for value in sys.path if value and Path(value).resolve() not in {root, source}]
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not (source/'outputs').exists() and not (source/'.git').exists()
    for name, expected in manifest['source_sha256'].items():
        assert sha(root/name) == expected
    for name, expected in release['auxiliary_sha256'].items():
        assert sha(name) == expected
    for name, expected in release['Qwen_model_sha256'].items():
        assert sha(name) == expected
    import gzip
    import pickle
    from dataclasses import asdict
    import cv2
    import numpy as np
    from omegaconf import OmegaConf
    import torch
    import ultralytics.utils
    import ultralytics.utils.torch_utils
    from conceptgraph.completion.stage9 import OUTPUT, PROJECT, dump
    from conceptgraph.completion import interactive_environment
    from conceptgraph.completion.interactive_navigation import evaluation_identity
    from conceptgraph.completion.kinematic_benchmark import SPEC, setup
    from conceptgraph.completion.protocol import GoalTruth, ArrivalEvaluator
    from conceptgraph.navigation.vehicle_geometry import VehicleState
    from conceptgraph.scripts.streamlined_detections import load_detection_models, detect_rgb_file
    from conceptgraph.vlm.qwen_semantic_delegate import QwenDelegateConfig, QwenSemanticDelegate
    assert PROJECT == root
    ultralytics.utils.NUM_THREADS = ultralytics.utils.torch_utils.NUM_THREADS = 1
    torch.set_num_threads(1)
    bridge_module = load_module(Path(release['bridge_source']), 'logical_semantic_bridge')
    extension = load_module(Path(manifest['extension_path']), 'perceived_stop_extension')
    interactive_environment.ASSET = Path(manifest['asset_path'])
    assert sha(interactive_environment.ASSET) == interactive_environment.ASSET_SHA
    task_path = OUTPUT/'interactive_simulation/synthetic_tasks_v1.json'
    assert sha(task_path) == manifest['task_release_sha256']
    frozen = json.loads(task_path.read_text())
    task = next(t for t in frozen['tasks'] if t['task_id'] == 'config00-positive0')
    config = next(c for c in frozen['configs'] if c['config_id'] == 'config00')
    destination = OUTPUT/'interactive_simulation/three_task_fixedlogical_001'/args.phase
    destination.mkdir(parents=True, exist_ok=False)
    cache = OUTPUT/'cache_audit/three_task_fixedlogical_001/cache'
    if args.phase == 'cold':
        assert not cache.exists()
    else:
        assert cache.is_dir()
    cache_before = {str(p.relative_to(cache)):sha(p) for p in cache.rglob('*.json')} if cache.exists() else {}
    process = Path('/proc')/str(os.getpid())
    save(destination/'worker_process.json', dict(pid=os.getpid(), start_ticks=(process/'stat').read_text().rsplit(')', 1)[1].split()[19],
        CUDA_VISIBLE_DEVICES='', cwd=str(Path.cwd()), source_root=str(PROJECT), phase=args.phase))
    begun = time.monotonic()
    env, delegate, bridge = None, None, None
    counts = dict(actual_frontend_calls=0, actual_Qwen_backend_calls=0)

    def publish(phase, **extra):
        value = dict(phase=phase, cache_phase=args.phase, wall_s=time.monotonic()-begun, counts=counts, **extra)
        temporary = destination/'progress.json.tmp'
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
        temporary.replace(destination/'progress.json')

    class CPUDelegate(QwenSemanticDelegate):
        def _load_backend(self):
            backend = super()._load_backend()
            model = getattr(backend, 'model', None)
            if model is not None:
                devices = sorted({str(parameter.device) for parameter in model.parameters()})
                assert devices == ['cpu'] and not torch.cuda.is_initialized()
                proof = destination/'qwen_loaded_model.json'
                if not proof.exists():
                    save(proof, dict(actual_parameter_devices=devices,
                        parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                        parameter_count=sum(p.numel() for p in model.parameters()),
                        CUDA_initialized=False, actual_backend_class=type(backend).__name__,
                        model_revision=release['Qwen_model_revision'], model_source=str(root/'checkpoint/Qwen2-VL-2B-Instruct')))
            return backend

        def _execute(self, **kwargs):
            envelope = super()._execute(**kwargs)
            if args.phase == 'warm' and not envelope['cache']['hit']:
                raise RuntimeError('热运行输入未命中原缓存；保留新请求与失败，不继续伪造一致')
            return envelope

        def _call_backend(self, *call_args, **call_kwargs):
            torch.set_num_threads(2)
            publish('SEMANTIC_BACKEND', task=call_args[0])
            try:
                return super()._call_backend(*call_args, **call_kwargs)
            finally:
                torch.set_num_threads(1)
                os.environ['OMP_NUM_THREADS'] = '1'
                counts['actual_Qwen_backend_calls'] = self.backend_call_count

    def forbidden_backend(*backend_args, **backend_kwargs):
        raise RuntimeError('热运行发生缓存缺失；禁止加载或重新调用Qwen')

    try:
        publish('INITIALIZING')
        delegate = CPUDelegate(QwenDelegateConfig(enabled=True,
            model_name_or_path=str(root/'checkpoint/Qwen2-VL-2B-Instruct'),
            model_revision=release['Qwen_model_revision'], device='cpu',
            use_qwen_query_parse=True, use_qwen_object_attributes=True, use_qwen_rerank=True,
            prompt_profile='completion_v3', timeout_s=600, max_new_tokens=256,
            max_images=1, torch_dtype='bfloat16', max_image_side=448,
            cache_dir=cache, log_dir=destination/'qwen_audit'),
            backend=forbidden_backend if args.phase == 'warm' else None)
        clock = bridge_module.LogicalClock()
        frontend_cfg = OmegaConf.create(dict(device='cpu'))
        models = load_detection_models(frontend_cfg)
        classes = [line.strip() for line in (PROJECT/'conceptgraph/vehicle_classes.txt').read_text().splitlines() if line.strip()]
        np.random.seed(17)
        rng = np.random.default_rng(17)
        xy = np.array(config['start_xy'])+rng.normal(0, .03, 2)
        state = VehicleState(*xy, config['start_yaw_rad']+rng.normal(0, math.radians(.5)))
        dump(destination/'initial_state.json', asdict(state))
        bridge = bridge_module.FixedLogicalSemanticBridge(task['query_text'], destination, delegate, clock)
        agent = extension.PerceivedStopCommitmentAgent(task['query_text'],
            {key:config[key] for key in ['centerline_xy', 'road_polygon_xy']}, classes,
            destination, semantic_bridge=bridge)
        env = interactive_environment.InteractiveRoadEnvironment(config, task, 17, state)
        truth = GoalTruth(task['task_id'], tuple(task['accepted_instance_ids']), tuple(task['regions']),
            config['config_id'], (0, 90), '冻结独立评价，不传给Agent或Qwen', reachability='reachable')
        evaluator = ArrivalEvaluator(truth)
        _, road, obstacles, _ = setup(config, task)
        route, correct, locked, streak = False, False, False, 0
        selected_identity, pending, terminal, frame = None, None, None, 0
        states = [state]
        with (destination/'perception.jsonl').open('x') as perception, (destination/'execution.jsonl').open('x') as execution:
            for tick in range(330):
                if tick % 5 == 0 and pending is None:
                    publish('PERCEIVING', sim_time_s=state.time_s, frame_idx=frame)
                    rgb, depth, pose, color, sensor = env.observe(destination, frame)
                    assert torch.get_num_threads() == 1 and not torch.cuda.is_initialized()
                    gobs, vis = detect_rgb_file(frontend_cfg, models, color, classes)
                    assert torch.get_num_threads() == 1 and not torch.cuda.is_initialized()
                    counts['actual_frontend_calls'] += 1
                    detection = destination/'detections'/f'{frame:06d}.pkl.gz'
                    detection.parent.mkdir(exist_ok=True)
                    with gzip.open(detection, 'xb') as handle:
                        pickle.dump(gobs, handle)
                    visual = destination/'detection_vis'/f'{frame:06d}.jpg'
                    visual.parent.mkdir(exist_ok=True)
                    assert cv2.imwrite(str(visual), vis)
                    clock.set(frame, state)
                    record = agent.perceive(frame, state, rgb, depth, pose, color, gobs)
                    _, matches = evaluation_identity(agent.nodes, config['targets'],
                        agent.selected if agent.selected is not None else next(iter(agent.nodes), None))
                    selected_identity = matches.get(agent.selected)
                    correct |= bool(record['candidates'] and matches.get(record['candidates'][0]['node_id']) in task['accepted_instance_ids'])
                    streak = streak+1 if selected_identity in task['accepted_instance_ids'] else 0
                    locked |= streak >= 3
                    route |= bool(record['planning'] and record['path_world'] and record['path_purpose'] == 'target_approach')
                    record.update(sensor=sensor, detector_sha256=sha(detection),
                        evaluation_only_matches=matches, evaluation_only_identity=selected_identity,
                        frontend_torch_threads=1, CUDA_initialized=False)
                    perception.write(json.dumps(record, ensure_ascii=False, allow_nan=False,
                        default=json_value)+'\n')
                    perception.flush()
                    frame += 1
                if (time.monotonic()-begun > (3600 if args.phase == 'cold' else 1200)
                        or state.time_s >= 30 or counts['actual_frontend_calls'] >= 80
                        or sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()) >= 1_000_000_000):
                    pending = 'BUDGET_STOP'
                if pending:
                    acceleration, steering, tracking = -SPEC.max_braking_mps2, state.steering_rad, dict(reason=pending)
                else:
                    acceleration, steering, tracking = agent.control(state)
                state = env.act(acceleration, steering)
                states.append(state)
                from conceptgraph.navigation.vehicle_geometry import swept_body_polygon
                sweep = swept_body_polygon(states[-2], state, SPEC)
                collision, violation = any(sweep.intersects(obstacle) for obstacle in obstacles), not road.covers(sweep)
                arrival = evaluator.update(time_s=state.time_s, xy=[state.x, state.y], yaw_rad=state.yaw_rad,
                    speed_mps=state.speed_mps, evaluated_instance_id=selected_identity,
                    body_collision=collision, lane_violation=violation)
                row = dict(state=asdict(state), control=dict(acceleration_mps2=acceleration,
                    steering_command_rad=steering), tracking=tracking, independent_arrival=arrival,
                    body_collision=collision, lane_violation=violation)
                execution.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                execution.flush()
                terminal = ('EXECUTION_FAILED' if collision or violation else pending if pending and state.speed_mps == 0
                    else 'ARRIVED' if arrival['actually_arrived'] and pending is None
                    else 'SAFETY_STOP' if agent.guard_latched and state.speed_mps == 0 else None)
                if terminal:
                    break
            terminal = terminal or 'CONTROL_BUDGET_STOP'
        semantic = bridge.close()
        dump(destination/'semantic_events.json', semantic)
        cache_after = {str(p.relative_to(cache)):sha(p) for p in cache.rglob('*.json')}
        requests = bridge.queue.requests
        assert {r['request']['task'] for r in requests} == {'query_parse', 'object_attributes', 'candidate_rerank'}
        assert len(requests) == 3
        for request in requests:
            assert request['response']['raw_output'] is not None, '后端未产生原文，不能获得真实三任务调用证据'
        if args.phase == 'cold':
            assert delegate.backend_call_count == 3 and not any(r['response']['cache']['hit'] for r in requests)
        else:
            assert delegate.backend_call_count == 0 and all(r['response']['cache']['hit'] for r in requests)
            assert cache_before == cache_after
        dump(destination/'cache_files.json', dict(before=cache_before, after=cache_after))
        dump(destination/'result.json', dict(task_id=task['task_id'], seed=17, status=terminal,
            route_generated=route, selected_object_evaluated_correct=correct,
            stable_correct_lock=locked, actual_full_perception_arrival=terminal == 'ARRIVED',
            final_state=asdict(state), actual_frontend_calls=counts['actual_frontend_calls'],
            saved_control_intervals=len(states)-1, actual_Qwen_backend_calls=delegate.backend_call_count,
            Qwen_calls=delegate.backend_call_count, Qwen_tasks=sorted(bridge.queue.used),
            semantic_applied_counts=semantic['applied_counts'], cache_all_hits=all(r['response']['cache']['hit'] for r in requests),
            actual_Qwen_device='cpu', gt_in_agent=False, CUDA_initialized=torch.cuda.is_initialized(),
            torch_threads_final=torch.get_num_threads(), real_clock_async=False,
            wall_s=time.monotonic()-begun, performance_credit=False, independent_scenes_added=0,
            frozen_main_table_credit=False, goal_complete=False))
        publish('FINISHED', status=terminal)
        print(json.dumps(json.loads((destination/'result.json').read_text()), ensure_ascii=False), flush=True)
    except BaseException as exc:
        save(destination/'interrupted.json', dict(error=repr(exc), counts=counts, goal_complete=False))
        raise
    finally:
        if env is not None:
            env.close()
        if delegate is not None:
            delegate.close()
        modules = {}
        for name, module in list(sys.modules.items()):
            location = getattr(module, '__file__', None)
            if location and (name == 'conceptgraph' or name.startswith('conceptgraph.')):
                path = Path(location).resolve()
                assert path.is_relative_to(root), str(path)
                modules[name] = dict(path=str(path), sha256=sha(path))
        save(destination/'import_audit.json', dict(all_conceptgraph_modules_from_isolated_root=True,
            modules=modules, CUDA_initialized=torch.cuda.is_initialized(), source_root=str(PROJECT),
            worker_sha256=sha(__file__), phase=args.phase, goal_complete=False))


if __name__ == '__main__':
    main()
