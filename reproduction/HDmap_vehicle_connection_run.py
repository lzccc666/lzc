"""实际运行真实开发地图的车辆连接诊断；参考位姿不计语义到达。"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from shapely.geometry import LineString, mapping, shape

from conceptgraph.completion.independent_continuous_body import audit_trajectory
from conceptgraph.completion.kinematic_benchmark import point_astar
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump
from conceptgraph.completion.topology_vehicle import DirectedVehicleCorridor, execute_corridor
from conceptgraph.navigation.vehicle_geometry import VehicleSpec, VehicleState
from conceptgraph.navigation.vehicle_planner import hybrid_astar


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--mode', choices=['smoke', 'full'], required=True)
    args = parser.parse_args()
    started = time.monotonic()
    tasks_root = OUTPUT/'vehicle_execution/HDmap_connection_tasks_001'
    metadata = read(tasks_root/'manifest.json')
    tasks = read(tasks_root/'tasks.json')
    assert digest(tasks_root/'tasks.json') == metadata['tasks_sha256']
    spec = VehicleSpec()
    plans = [dict(task=t, seed=seed, method=method) for t in tasks for seed in metadata['seeds'] for method in metadata['methods']]
    if args.mode == 'smoke':
        first_ready = next(t for t in tasks if t['status'] == 'ready_given_map_route')
        plans = [dict(task=t, seed=17, method='hybrid_astar') for t in [tasks[0], first_ready]]
    control_smoke = OUTPUT/'vehicle_execution/HDmap_connection_control_smoke_001/report.json'
    if args.mode == 'full':
        checked = read(control_smoke)
        assert checked['actual_motion'] and checked['independent_body_passed']
    root = OUTPUT/'vehicle_execution'/args.run_id
    root.mkdir(exist_ok=False)
    sources = [Path(__file__).resolve()]+[PROJECT/p for p in [
        'conceptgraph/completion/topology_vehicle.py', 'conceptgraph/completion/execution_direction_audit.py',
        'conceptgraph/completion/independent_continuous_body.py', 'conceptgraph/completion/kinematic_benchmark.py',
        'conceptgraph/navigation/vehicle_geometry.py', 'conceptgraph/navigation/vehicle_planner.py']]
    frozen = {str(p.relative_to(PROJECT)): digest(p) for p in sources}
    dump(root/'release.json', dict(source_sha256=frozen, tasks_sha256=digest(tasks_root/'tasks.json'),
        tasks_manifest_sha256=digest(tasks_root/'manifest.json'),
        protocol_sha256=digest(OUTPUT/'task_protocol/HDmap_vehicle_connection_001.md'),
        smoke_addendum_sha256=digest(OUTPUT/'task_protocol/HDmap_vehicle_connection_smoke_addendum_001.md'),
        control_smoke_sha256=digest(control_smoke),
        control_smoke_protocol_sha256=digest(OUTPUT/'task_protocol/HDmap_vehicle_connection_control_smoke_001.md'),
        mode=args.mode, expected_trials=len(plans), vehicle=asdict(spec),
        plans=[dict(task_id=p['task']['task_id'], seed=p['seed'], method=p['method']) for p in plans],
        wall_budget_s=3600, storage_budget_bytes=1024**3, minimum_free_bytes=6*1024**3,
        planner_wall_budget_s=3, planner_max_expansions=2000, maximum_control_steps=900,
        CPU_shared=True, CUDA_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gt_assisted_map_route=True, actual_perception_run=False, new_test_opened=False,
        actual_semantic_arrival_credit=0, goal_complete=False))
    events = OUTPUT/'interactive_simulation/realtime_three_model_full_001/resource_events.jsonl'

    def event(state):
        with events.open('a') as handle:
            handle.write(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                event=args.run_id, status=state, shared_CPU=True, CUDA_visible=os.environ.get('CUDA_VISIBLE_DEVICES')))+'\n')

    results = []
    event('start')
    try:
        for index, plan in enumerate(plans):
            assert time.monotonic()-started < 3600 and shutil.disk_usage(OUTPUT).free > 6*1024**3
            assert all(digest(PROJECT/path) == value for path, value in frozen.items())
            task, seed, method = plan['task'], plan['seed'], plan['method']
            key = f"{index:03d}_{task['scene']}_seed{seed}_{method}"
            case = root/key
            record = dict(index=index, task_id=task['task_id'], scene=task['scene'], log_token=task['log_token'],
                seed=seed, method=method, task_preparation_status=task['status'],
                context_path=task['context_path'], context_sha256=task['context_sha256'],
                path_generated=False, actual_vehicle_motion=False, reference_pose_reached=False,
                selected_correct_object=None, stable_correct_lock=None, actual_semantic_arrival_credit=0,
                reference_route_given=True, recorded_vehicle_rear_axle_reproduced=False,
                actual_perception_run=False, actual_model_calls=0)
            if task['status'] != 'ready_given_map_route':
                record.update(status=task['status'], saved_control_records=0, independent_continuous_body=None)
            else:
                assert digest(task['context_path']) == task['context_sha256']
                context = read(task['context_path'])
                lanes = [context['lane_geometries'][token] for token in task['lane_tokens']]
                try:
                    corridor = DirectedVehicleCorridor(lanes, context['transitions'], shape(context['drivable_geojson']),
                        [shape(g) for g in context['obstacles_geojson']])
                except ValueError as exc:
                    record.update(status='UNKNOWN_INVALID_DECLARED_CHAIN', diagnostic=str(exc), saved_control_records=0,
                                  independent_continuous_body=None)
                else:
                    rng = np.random.default_rng(seed)
                    xy = np.asarray(task['start_xy'])+rng.normal(0, .03, 2)
                    start = VehicleState(*xy, task['start_yaw_rad']+rng.normal(0, math.radians(.5)))
                    reference = LineString(task['reference_centerline_xy'])
                    if method == 'hybrid_astar':
                        route = hybrid_astar(start, task['goal_xy'], task['goal_yaw_rad'], corridor.road,
                            corridor.obstacles, reference, spec, budget_s=3, max_expanded=2000)
                        path = [[s.x, s.y] for s in route.states]
                        planning = dict(expanded=route.expanded, failure=route.failure, duration_s=route.duration_s)
                    else:
                        route = point_astar(start, task['goal_xy'], corridor.road, corridor.obstacles, reference)
                        path = route.pop('points')
                        planning = route
                    dump(case/'plan.json', dict(path=path, initial_state=asdict(start), goal_xy=task['goal_xy'],
                        goal_yaw_rad=task['goal_yaw_rad'], lane_tokens=task['lane_tokens'], **planning))
                    execution = execute_corridor(start, path, task['goal_xy'], task['goal_yaw_rad'], corridor,
                        spec=spec, maximum_steps=900)
                    rows = execution.pop('records')
                    states = execution.pop('states')
                    dump(case/'states.json', states)
                    dump(case/'geometry.json', dict(road=mapping(corridor.road),
                        obstacles=[mapping(o) for o in corridor.obstacles],
                        lane_tokens=task['lane_tokens'], initial_state=asdict(start)))
                    with (case/'execution.jsonl').open('x') as handle:
                        for row in rows:
                            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    if rows:
                        independent = audit_trajectory(asdict(start), rows, asdict(spec), corridor.road, corridor.obstacles)
                        dump(case/'independent_continuous_body.json', independent)
                        independent_result = {k: v for k, v in independent.items() if k != 'intervals'}
                    else:
                        independent_result = None
                    record.update(execution, path_generated=bool(path), planning=planning, saved_control_records=len(rows),
                        actual_vehicle_motion=any(s['speed_mps'] > 1e-6 for s in states),
                        initial_state=asdict(start), independent_continuous_body=independent_result,
                        final_state=states[-1], source_execution_sha256=digest(case/'execution.jsonl'))
            dump(case/'result.json', record)
            results.append(record)
            with (root/'trials.jsonl').open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
            print(json.dumps(dict(trial=index+1, scene=task['scene'], method=method, status=record['status'],
                                 motion=record['actual_vehicle_motion']), ensure_ascii=False), flush=True)
        summary = dict(mode=args.mode, attempted=len(results), expected=len(plans),
            status_counts=dict(Counter(r['status'] for r in results)),
            planned_routes=sum(r['path_generated'] for r in results), executed_trials=sum(r['actual_vehicle_motion'] for r in results),
            reference_pose_reached=sum(r['reference_pose_reached'] for r in results),
            actual_semantic_arrival_credit=0, actual_model_calls=0, actual_perception_run=False,
            by_method={method: dict(trials=sum(r['method'] == method for r in results),
                routes=sum(r['method'] == method and r['path_generated'] for r in results),
                actual_motion=sum(r['method'] == method and r['actual_vehicle_motion'] for r in results),
                reference_pose_reached=sum(r['method'] == method and r['reference_pose_reached'] for r in results))
                for method in sorted({r['method'] for r in results})},
            trial_records_sha256=digest(root/'trials.jsonl'), wall_CPU_shared_s=time.monotonic()-started,
            limitations=['给定地图路线的人为后轴起始条件，不是录制车辆后轴重建或真实感知闭环',
                        '当前预测障碍冻结为静态；不评价动态交通安全',
                        '参考位姿到达不是正确对象选择、锁定或语义到达'], goal_complete=False)
        assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) < 1024**3
        dump(root/'summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except BaseException as exc:
        dump(root/'interrupted.json', dict(error=repr(exc), completed_trials=len(results), goal_complete=False))
        raise
    finally:
        event('end')


if __name__ == '__main__':
    main()
