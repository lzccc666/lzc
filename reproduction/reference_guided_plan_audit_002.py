"""核对保存的规划积分与0.05米车身余量；不重新规划或执行。"""
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import time

import numpy as np
from shapely.geometry import Point, Polygon, shape

from conceptgraph.completion.independent_continuous_body import audit_trajectory
from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def read(path):
    return json.loads(path.read_text())


def main():
    begun = time.monotonic()
    source = OUTPUT/'vehicle_execution/HDmap_reference_guided_full_002'
    root = OUTPUT/'vehicle_execution/HDmap_reference_guided_plan_audit_002'
    root.mkdir(exist_ok=False)
    release, summary = read(source/'release.json'), read(source/'summary.json')
    assert summary['attempted'] == summary['expected'] == 228
    assert digest(source/'trials.jsonl') == summary['trial_records_sha256']
    dump(root/'release.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=digest(__file__), original_release_sha256=digest(source/'release.json'),
        original_trials_sha256=digest(source/'trials.jsonl'), padded_footprint_m=.05,
        budget_wall_s=900, budget_bytes=100000000, new_executions=0, new_model_calls=0, goal_complete=False))
    files, cases, values = {}, [], defaultdict(list)
    planned_intervals = 0
    for line in (source/'trials.jsonl').open():
        case = json.loads(line)
        folder = source/f"{case['index']:03d}_{case['scene']}_seed{case['seed']}_{case['method']}"
        path = folder/'plan_states.json'
        record = dict(task_id=case['task_id'], scene=case['scene'], log_token=case['log_token'],
            method=case['method'], seed=case['seed'], planned_route=case['path_generated'],
            body_certificate=None, planned_intervals=0, semantic_arrival_credit=0)
        if not path.exists():
            assert not case['path_generated']
        else:
            states, plan = read(path), read(folder/'plan.json')
            assert [[s['x'], s['y']] for s in states] == plan['path']
            assert bool(states) == case['path_generated']
            files[str(path)] = digest(path)
            files[str(folder/'plan.json')] = digest(folder/'plan.json')
            if states:
                assert states[0] == case['initial_state'] and len(states) >= 2
                assert math.dist([states[-1]['x'], states[-1]['y']], plan['goal_xy']) <= .65+1e-9
                error = math.atan2(math.sin(states[-1]['yaw_rad']-plan['goal_yaw_rad']),
                                   math.cos(states[-1]['yaw_rad']-plan['goal_yaw_rad']))
                assert abs(error) <= math.radians(12)+1e-9
                context_path = OUTPUT/case['context_path'].split('/thesis_completion_v1/', 1)[1]
                context, geometry = read(context_path), read(folder/'geometry.json')
                assert Polygon(context['lane_geometries'][plan['lane_tokens'][-1]]['polygon']).covers(
                    Point(states[-1]['x'], states[-1]['y']))
                files[str(context_path)] = digest(context_path)
                files[str(folder/'geometry.json')] = digest(folder/'geometry.json')
                # 只构造能产生同一计划状态的有界控制见证，不冒充保存了原始建议转角。
                records, curvature, steering_rates, accelerations = [], [], [], []
                for i, (before, after) in enumerate(zip(states, states[1:])):
                    dt = after['time_s']-before['time_s']
                    assert abs(dt-release['vehicle']['dt_s']) < 1e-7
                    acceleration = (after['speed_mps']-before['speed_mps'])/release['vehicle']['dt_s']
                    records.append(dict(index=i, state=after,
                        control=dict(acceleration_mps2=acceleration, steering_command_rad=after['steering_rad'])))
                    travel = .5*(before['speed_mps']+after['speed_mps'])*dt
                    turn = math.atan2(math.sin(after['yaw_rad']-before['yaw_rad']),
                                      math.cos(after['yaw_rad']-before['yaw_rad']))
                    if travel > 1e-12:
                        curvature.append(abs(turn)/travel)
                    steering_rates.append(abs(after['steering_rad']-before['steering_rad'])/dt)
                    accelerations.append(acceleration)
                padded = dict(release['vehicle'])
                padded['front_from_rear_axle_m'] += .05
                padded['rear_overhang_m'] += .05
                padded['width_m'] += .1
                body = audit_trajectory(states[0], records, padded, shape(geometry['road']),
                                        [shape(g) for g in geometry['obstacles']])
                report_path = root/f"{case['index']:03d}_{case['scene']}_seed{case['seed']}_{case['method']}.json"
                dump(report_path, body)
                record.update(body_certificate=body['status'], body_passed=body['passed'],
                    planned_intervals=len(records), planned_length_m=sum(math.dist([a['x'], a['y']], [b['x'], b['y']])
                        for a, b in zip(states, states[1:])),
                    max_curvature_per_m=max(curvature, default=0),
                    max_steering_rad=max(abs(s['steering_rad']) for s in states),
                    max_steering_rate_radps=max(steering_rates), max_acceleration_mps2=max(accelerations),
                    inferred_admissible_controls_are_not_original_planner_commands=True,
                    certificate_sha256=digest(report_path))
                assert record['max_curvature_per_m'] <= math.tan(release['vehicle']['max_steering_rad'])/release['vehicle']['wheelbase_m']+1e-7
                assert record['max_steering_rate_radps'] <= release['vehicle']['max_steering_rate_radps']+1e-7
                assert record['max_acceleration_mps2'] <= release['vehicle']['max_acceleration_mps2']+1e-7
                planned_intervals += len(records)
                for key in ['planned_length_m', 'max_curvature_per_m', 'max_steering_rad', 'max_steering_rate_radps']:
                    values[case['method']+'::'+key].append(record[key])
        cases.append(record)
        assert time.monotonic()-begun < 900
    assert len(cases) == 228 and sum(c['planned_route'] for c in cases) == summary['planned_routes']
    dump(root/'cases.json', cases)
    dump(root/'files.json', files)
    report = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(), trials=len(cases),
        planned_routes=sum(c['planned_route'] for c in cases), planned_intervals=planned_intervals,
        planned_routes_with_padded_body_certificate=sum(c.get('body_passed', False) for c in cases),
        body_unknown_or_failed_routes=sum(c['planned_route'] and not c.get('body_passed', False) for c in cases),
        padded_rectangular_body_margin_m=.05, independent_input_files=len(files),
        source_files_sha256=digest(root/'files.json'),
        descriptive_metrics={key: dict(count=len(v), median=float(np.median(v)), maximum=max(v)) for key, v in values.items()},
        original_planner_zero_routes_has_no_paired_curvature_or_tracking_difference=not any(c['method'] == 'hybrid_astar' and c['planned_route'] for c in cases),
        controls_are_inferred_admissible_witness_not_original_proposal_commands=True,
        actual_executed_control_audit_is_separate=True, new_executions=0, new_model_calls=0,
        semantic_arrival_credit=0, wall_CPU_shared_s=time.monotonic()-begun,
        limits=['本检查复用已独立于执行器的连续车身实现，属于规划状态输入/余量核验，不是第三种车身算法',
                '未知证书不计通过；规划几何与实际控制跟踪、方向和连接分开评价',
                '开发诊断、人工地图起点和静态预测障碍，不是新封存、语义到达或动态交通安全'], goal_complete=False)
    dump(root/'report.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
