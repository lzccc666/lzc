"""固定逻辑时钟的独立交互运行；复用真实前端、对象图和车辆执行链。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import json
import math
import os
import pickle
import shutil
import time
import traceback

import numpy as np

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, default=json_value))
    temporary.replace(path)


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class Commands:
    """暂停冻结时钟；停车另行执行有界制动，二者不能混称。"""
    def __init__(self):
        self.mode = 'PAUSED'
        self.pending_ticks = 0
        self.processed = 0

    def accept(self, command):
        if self.mode == 'STOPPING':
            return
        if command == 'run':
            self.mode, self.pending_ticks = 'RUNNING', 0
        elif command == 'pause':
            self.mode, self.pending_ticks = 'PAUSED', 0
        elif command == 'step':
            self.mode = 'PAUSED'
            self.pending_ticks = min(50, self.pending_ticks + 5)
        elif command == 'stop':
            self.mode, self.pending_ticks = 'STOPPING', 0
        else:
            raise ValueError('未知控制命令')

    def poll(self, path):
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        new = rows[self.processed:]
        for row in new:
            self.accept(row['command'])
        self.processed = len(rows)
        return new


def run(destination):
    # 该进程只跑CPU；不得因缺CUDA配置回落成占用正在运行的正式队列。
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('交互入口要求显式隐藏CUDA')
    import cv2
    from omegaconf import OmegaConf
    from conceptgraph.completion.interactive_environment import InteractiveRoadEnvironment
    from conceptgraph.completion.interactive_navigation import PerceptionVehicleAgent, evaluation_identity
    from conceptgraph.completion.kinematic_benchmark import SPEC, setup
    from conceptgraph.completion.protocol import GoalTruth, ArrivalEvaluator
    from conceptgraph.navigation.vehicle_geometry import VehicleState, swept_body_polygon, audit_executed_states
    from conceptgraph.scripts.streamlined_detections import load_detection_models, detect_rgb_file

    release = json.loads((destination/'release.json').read_text())
    for name, expected in release['source_sha256'].items():
        if digest(PROJECT/name) != expected:
            raise ValueError('交互运行源码与release不一致：'+name)
    task_path = OUTPUT/'interactive_simulation/synthetic_tasks_v1.json'
    if digest(task_path) != release['task_release_sha256']:
        raise ValueError('冻结任务已改变')
    tasks = json.loads(task_path.read_text())
    task = next(t for t in tasks['tasks'] if t['task_id'] == release['task_id'])
    config = next(c for c in tasks['configs'] if c['config_id'] == task['config_id'])
    np.random.seed(release['seed'])
    rng = np.random.default_rng(release['seed'])
    xy = np.asarray(config['start_xy']) + rng.normal(0, .03, 2)
    state = VehicleState(*xy, config['start_yaw_rad'] + rng.normal(0, math.radians(.5)))
    dump(destination/'initial_state.json', asdict(state))
    states, commands = [state], Commands()
    env = None
    begun = time.monotonic()
    frame, tick, routes, streak, max_streak = 0, 0, 0, 0, 0
    correct_top, correct_lock = False, False
    evaluated_id, current = None, None
    rgb_hashes, depth_hashes = set(), set()
    pending, final = None, None
    counts = dict(actual_frontend_calls=0, gt_in_agent=False, model_device='cpu', Qwen_calls=0,
        new_independent_scenes=0, performance_credit=False, frozen_main_table_credit=False)

    def publish(phase):
        atomic_json(destination/'state.json', dict(run_id=destination.name, phase=phase,
            mode=commands.mode, pending_ticks=commands.pending_ticks, processed_commands=commands.processed,
            state=asdict(state), wall_elapsed_s=time.monotonic()-begun, query_text=task['query_text'],
            task_id=task['task_id'], config_id=config['config_id'], frame=current,
            road_polygon_xy=config['road_polygon_xy'], centerline_xy=config['centerline_xy'],
            execution_xy=[[s.x, s.y] for s in states],
            vehicle_spec=asdict(SPEC), counts=counts, result=None, clock='fixed_simulation_not_realtime'))

    publish('INITIALIZING')
    try:
        cfg = OmegaConf.create(dict(device='cpu'))
        models = load_detection_models(cfg)
        classes = [s.strip() for s in (PROJECT/'conceptgraph/vehicle_classes.txt').read_text().splitlines() if s.strip()]
        env = InteractiveRoadEnvironment(config, task, release['seed'], state)
        agent = PerceptionVehicleAgent(task['query_text'], {k:config[k] for k in
            ['centerline_xy', 'road_polygon_xy']}, classes, destination)
        _, road, obstacles, proof = setup(config, task)
        truth = GoalTruth(task['task_id'], tuple(task['accepted_instance_ids']), tuple(task['regions']),
            config['config_id'], (0, 90), '冻结合成任务；仅隔离评价端使用',
            reachability='reachable' if task['kind'] == 'reachable_positive' else
                'inaccessible' if task['kind'] == 'inaccessible' else 'unknown')
        evaluator = ArrivalEvaluator(truth)
        with (destination/'perception.jsonl').open('x') as perception, \
                (destination/'execution.jsonl').open('x') as execution, \
                (destination/'processed_commands.jsonl').open('x') as applied:
            def observe():
                nonlocal frame, current, evaluated_id, correct_top, streak, max_streak, correct_lock, routes
                publish('PERCEIVING')
                rgb, depth, pose, color, sensor = env.observe(destination, frame)
                if frame == 0:
                    from OpenGL.GL import glGetString, GL_RENDERER
                    env.renderer._platform.make_current()
                    try:
                        renderer_name = glGetString(GL_RENDERER).decode()
                    finally:
                        env.renderer._platform.make_uncurrent()
                    if 'llvmpipe' not in renderer_name.lower():
                        raise RuntimeError('交互演示未使用声明的软件渲染器')
                    counts['actual_renderer'] = renderer_name
                gobs, vis = detect_rgb_file(cfg, models, color, classes)
                counts['actual_frontend_calls'] += 1
                det = destination/'detections'/f'{frame:06d}.pkl.gz'
                det.parent.mkdir(exist_ok=True)
                with gzip.open(det, 'xb') as handle:
                    pickle.dump(gobs, handle)
                overlay = destination/'detection_vis'/f'{frame:06d}.jpg'
                overlay.parent.mkdir(exist_ok=True)
                if not cv2.imwrite(str(overlay), vis):
                    raise OSError('检测图保存失败')
                depth_view = destination/'depth_vis'/f'{frame:06d}.png'
                depth_view.parent.mkdir(exist_ok=True)
                display = cv2.applyColorMap(np.clip(depth/60*255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                display[depth <= 0] = 0
                if not cv2.imwrite(str(depth_view), display):
                    raise OSError('有效深度显示副本保存失败')
                record = agent.perceive(frame, state, rgb, depth, pose, color, gobs)
                # 仅评价用：selected为空也保存全部可关联节点，不改agent或旧记录。
                _, matches = evaluation_identity(agent.nodes, config['targets'],
                    agent.selected if agent.selected is not None else next(iter(agent.nodes), None))
                evaluated_id = matches.get(agent.selected)
                ids = [c['node_id'] for c in record['candidates']]
                correct_top |= bool(ids and matches.get(ids[0]) in task['accepted_instance_ids'])
                streak = streak+1 if evaluated_id in task['accepted_instance_ids'] else 0
                max_streak = max(max_streak, streak)
                correct_lock |= streak >= 3
                routes += int(bool(record['planning'] and record['path_world'] and
                    record['planning']['purpose'] == 'target_approach'))
                record.update(sensor=sensor, detector_sha256=digest(det),
                    evaluation_only_identity=evaluated_id, evaluation_only_matches=matches,
                    detection_vis=str(overlay), depth_vis=str(depth_view),
                    completed_wall_elapsed_s=time.monotonic()-begun)
                perception.write(json.dumps(record, ensure_ascii=False, allow_nan=False, default=json_value)+'\n')
                perception.flush()
                current = record
                frame += 1
                rgb_hashes.add(sensor['rgb_sha256'])
                depth_hashes.add(sensor['depth_sha256'])
                publish('READY')

            observe()
            while final is None:
                for row in commands.poll(destination/'commands.jsonl'):
                    applied.write(json.dumps(dict(**row, applied_at_sim_s=state.time_s,
                        applied_at_wall_s=time.monotonic()-begun), ensure_ascii=False)+'\n')
                    applied.flush()
                elapsed = time.monotonic()-begun
                over_budget = (elapsed >= release['wall_budget_s'] or state.time_s >= release['max_sim_s'] or
                    counts['actual_frontend_calls'] >= release['max_frontend_calls'] or
                    shutil.disk_usage(destination).free < release['minimum_free_bytes'])
                if tick % 5 == 0:
                    over_budget |= sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()) >= release['storage_budget_bytes']
                if pending is None and over_budget:
                    pending = 'BUDGET_STOP'
                if pending is None and commands.mode == 'STOPPING':
                    pending = 'USER_STOPPED'
                if pending is None and commands.mode == 'PAUSED' and not commands.pending_ticks:
                    publish('PAUSED')
                    time.sleep(.1)
                    continue
                if pending:
                    acceleration, steering = -SPEC.max_braking_mps2, state.steering_rad
                    tracking = dict(reason=pending, command_source='bounded_terminal_braking')
                else:
                    acceleration, steering, tracking = agent.control(state)
                old = state
                state = env.act(acceleration, steering)
                states.append(state)
                tick += 1
                commands.pending_ticks = max(0, commands.pending_ticks-1)
                sweep = swept_body_polygon(old, state, SPEC)
                collision = any(sweep.intersects(o) for o in obstacles)
                violation = not road.covers(sweep)
                arrival = evaluator.update(time_s=state.time_s, xy=[state.x, state.y], yaw_rad=state.yaw_rad,
                    speed_mps=state.speed_mps, evaluated_instance_id=evaluated_id,
                    body_collision=collision, lane_violation=violation)
                execution.write(json.dumps(dict(state=asdict(state), control=dict(acceleration_mps2=acceleration,
                    steering_command_rad=steering), tracking=tracking, independent_arrival=arrival,
                    body_collision=collision, lane_violation=violation,
                    processed_commands=commands.processed), ensure_ascii=False, allow_nan=False)+'\n')
                execution.flush()
                if collision or violation:
                    final = 'EXECUTION_FAILED'
                elif pending and state.speed_mps == 0:
                    final = pending
                elif arrival['actually_arrived'] and not pending:
                    final = 'ARRIVED'
                elif agent.guard_latched and state.speed_mps == 0:
                    final = 'SAFETY_STOP'
                if tick % 5 == 0 and final is None:
                    observe()
                publish('READY')
        audit = audit_executed_states(states, road, obstacles, SPEC)
        if not audit['passed']:
            final = 'EXECUTION_FAILED'
        dump(destination/'independent_execution_audit.json', audit)
        result = dict(task_id=task['task_id'], config_id=config['config_id'], kind=task['kind'],
            seed=release['seed'], status=final, route_generated=routes > 0,
            selected_object_evaluated_correct=correct_top, stable_correct_lock=correct_lock,
            max_consecutive_independently_correct_locked_observations=max_streak,
            actual_full_perception_arrival=final == 'ARRIVED', initial_state_observed=True,
            final_state=asdict(state), terminal_zero_speed=state.speed_mps == 0,
            saved_control_intervals=tick, saved_perception_frames=frame,
            distinct_RGB=len(rgb_hashes), distinct_depth=len(depth_hashes),
            independent_execution_audit=audit, independent_disconnection_proof=proof,
            wall_duration_s=time.monotonic()-begun, **counts, goal_complete=False)
        dump(destination/'result.json', result)
        publish('FINISHED')
        snapshot = json.loads((destination/'state.json').read_text())
        snapshot['result'] = result
        atomic_json(destination/'state.json', snapshot)
    except BaseException as exc:
        failure = dict(error=repr(exc), traceback=traceback.format_exc(), goal_complete=False)
        dump(destination/'failure.json', failure)
        publish('FAILED')
        raise
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    args = parser.parse_args()
    path = (OUTPUT/'interactive_simulation/live_navigation_sessions_001'/args.session).resolve()
    if path.parent != (OUTPUT/'interactive_simulation/live_navigation_sessions_001').resolve():
        raise ValueError('非法交互会话路径')
    run(path)


if __name__ == '__main__':
    main()
