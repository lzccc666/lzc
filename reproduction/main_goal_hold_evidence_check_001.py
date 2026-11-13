"""保存观测重建身份、四项业务前三项与保守真实时钟终点保持证据。"""
import bisect
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, Polygon


PROJECT = Path('/root/autodl-tmp/lizichong')
OUT = PROJECT/'outputs/project_v2/thesis_completion_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def identities(frame, targets):
    """从保存类别与中心重新构造代价；同一Hungarian后端，不称独立身份真值。"""
    nodes = sorted(frame['nodes'], key=lambda node: node['node_id'])
    if not nodes:
        return {}
    costs = np.asarray([[math.dist(node['center'][:2], target['center_xy'])
                         if node['category'] == target['category'] else 1e6
                         for target in targets] for node in nodes])
    rows, columns = linear_sum_assignment(costs)
    return {nodes[int(row)]['node_id']: targets[int(column)]['instance_id']
            for row, column in zip(rows, columns) if costs[row, column] <= 2.5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--body-run', default='independent_body_main_full_001')
    parser.add_argument('--run-id', default='main_goal_hold_evidence_check_001')
    args = parser.parse_args()
    started = time.monotonic()
    source = OUT/'interactive_simulation/realtime_three_model_full_001'
    summary, release = read(source/'summary.json'), read(source/'release.json')
    assert summary['attempted'] == summary['expected'] == 450
    assert sha(source/'trial_records.jsonl') == summary['trial_records_sha256']
    task_path = OUT/'interactive_simulation/synthetic_tasks_v1.json'
    assert sha(task_path) == release['task_release_sha256']
    data = read(task_path)
    tasks = {task['task_id']: task for task in data['tasks']}
    configs = {config['config_id']: config for config in data['configs']}
    body_root = OUT/'vehicle_execution'/args.body_run
    body = {(row['method'], row['task'], row['seed']): row
            for row in map(json.loads, (body_root/'cases.jsonl').open())}
    assert read(body_root/'report.json')['cases'] == len(body) == 450
    raw = [json.loads(line) for line in (source/'trial_records.jsonl').open()]
    destination = OUT/'vehicle_execution'/args.run_id
    destination.mkdir(exist_ok=False)
    cases, files, counts = [], {}, Counter()
    for record in raw:
        assert time.monotonic()-started < 900
        case = Path(record['raw_case'])
        result = record['result']
        output = {field: record[field] for field in ['variant', 'task', 'config_id', 'seed', 'status', 'raw_case']}
        output['reported_arrival'] = record['actual_arrival']
        if result is None:
            assert not record['actual_arrival']
            output.update(status='unknown_missing_complete_result', hold_certificate=None)
            counts['missing_complete_results'] += 1
            cases.append(output)
            continue
        assert read(case/'result.json') == result and sha(case/'result.json') == record['result_sha256']
        frames = [json.loads(line) for line in (case/'perception.jsonl').open()]
        execution = [json.loads(line) for line in (case/'execution.jsonl').open()]
        task, config = tasks[record['task']], configs[record['config_id']]
        accepted = set(task['accepted_instance_ids'])
        independent_ids, completion_times = [], []
        correct_selection, correct_lock, streak, routes = False, False, 0, 0
        for frame in frames:
            mapping = identities(frame, config['targets'])
            selected_id = mapping.get(frame['selected_node_id'])
            assert frame['evaluation_only_identity'] == selected_id
            saved = {int(key): value for key, value in frame['evaluation_only_matches'].items()}
            assert saved == (mapping if frame['selected_node_id'] is not None else {})
            top_id = mapping.get(frame['candidates'][0]['node_id']) if frame['candidates'] else None
            correct_selection |= top_id in accepted
            streak = streak+1 if selected_id in accepted else 0
            correct_lock |= streak >= 3
            routes += bool(frame['planning'] and frame['path_world'] and frame['path_purpose'] == 'target_approach')
            independent_ids.append(selected_id)
            completion_times.append(frame['completed_wall_elapsed_s'])
            assert not frame['gt_sent_to_agent']
            assert frame['sensor']['state']['time_s'] == frame['sensor']['timestamp_s']
        assert all(a < b for a, b in zip(completion_times, completion_times[1:]))
        assert result['selected_object_evaluated_correct'] == correct_selection
        assert result['stable_correct_lock'] == correct_lock
        assert result['route_generated'] == (routes > 0)
        body_record = body[record['variant'], record['task'], record['seed']]
        assert body_record['raw_case'] == record['raw_case']
        longest, hold_start, previous_valid = 0., None, False
        witness, longest_witness = [], []
        previous_wall = 0.
        regions = [(region, Polygon(region['polygon_xy'])) for region in task['regions']]
        for index, row in enumerate(execution):
            state, wall = row['state'], row['wall_elapsed_s']
            assert wall > previous_wall
            # 不复现原评价器隐式批次截止点。整个心跳两端及其间所有已完成
            # 感知身份均须是可接受对象，得到更保守的保存时间轴证据。
            before = bisect.bisect_right(completion_times, previous_wall)-1
            after = bisect.bisect_right(completion_times, wall)-1
            ids = independent_ids[max(0, before):after+1] if before >= 0 else []
            identity_ok = bool(ids) and all(identity in accepted for identity in ids)
            geometry_ok = False
            if identity_ok:
                for region, polygon in regions:
                    if region['instance_id'] not in ids:
                        continue
                    heading = abs(math.remainder(state['yaw_rad']-region['heading_rad'], 2*math.pi))
                    geometry_ok |= polygon.covers(Point(state['x'], state['y'])) and heading <= math.pi/12
            valid = bool(identity_ok and geometry_ok and abs(state['speed_mps']) <= .15
                         and task['kind'] == 'reachable_positive' and body_record['passed'])
            if valid:
                if not previous_valid:
                    hold_start, witness = state['time_s'], []
                witness.append(index)
                held = state['time_s']-hold_start
                if held > longest:
                    longest, longest_witness = held, witness.copy()
            else:
                hold_start, witness = None, []
            previous_valid, previous_wall = valid, wall
        certified = longest >= 2. and bool(body_record['passed'])
        output.update(reconstructed_route=routes > 0, reconstructed_selection=correct_selection,
                      reconstructed_lock=correct_lock, hold_certificate=certified,
                      longest_conservative_hold_s=longest, hold_control_indices=longest_witness,
                      body_certificate=body_record['passed'],
                      reported_arrival_without_hold_certificate=record['actual_arrival'] and not certified,
                      uncredited_posthoc_hold=certified and not record['actual_arrival'])
        counts['complete_results'] += 1
        counts['perception_frames'] += len(frames)
        counts['control_records'] += len(execution)
        counts['reported_arrivals'] += record['actual_arrival']
        counts['reported_arrivals_with_conservative_hold'] += record['actual_arrival'] and certified
        counts['reported_arrivals_without_conservative_hold'] += record['actual_arrival'] and not certified
        counts['uncredited_posthoc_holds'] += certified and not record['actual_arrival']
        cases.append(output)
        for path in [case/'result.json', case/'perception.jsonl', case/'execution.jsonl']:
            files[str(path)] = sha(path)
    assert len(cases) == 450 and counts['complete_results']+counts['missing_complete_results'] == 450
    for path in [task_path, source/'release.json', source/'summary.json', source/'trial_records.jsonl',
                 body_root/'cases.jsonl', body_root/'report.json']:
        files[str(path)] = sha(path)
    with (destination/'cases.jsonl').open('x') as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, allow_nan=False)+'\n')
    save(destination/'files.json', files)
    save(destination/'report.json', dict(passed=counts['reported_arrivals_without_conservative_hold'] == 0,
         created_utc=datetime.now(timezone.utc).isoformat(), cases=450, counts=dict(counts), body_run=args.body_run,
         source_sha256=sha(__file__), files_manifest_sha256=sha(destination/'files.json'), SHA_files=len(files),
         unresolved_arrivals=[case for case in cases if case.get('reported_arrival_without_hold_certificate')],
         wall_CPU_shared_s=time.monotonic()-started, actual_model_calls=0, new_executions=0, new_arrivals=0,
         limits=['重算保存类别/中心的2.5米一对一关联，沿用Hungarian后端，不是新人工实例真值',
                 '每0.1秒保存状态核对冻结终点区域、15度航向、0.15米每秒和2秒保持；车身连续证书另绑定',
                 '采用完成时间重建保守身份区间，不推测原异步评价器未记录的精确批次截止点',
                 '未通过只记未决，不删除原ARRIVED或强制补成功；额外事后保持不计到达信用',
                 '合成颜色任务、静态地图；不声称自然指令、动态交通或新封存验收完成'], goal_complete=False))
    print(json.dumps(read(destination/'report.json'), ensure_ascii=False))


if __name__ == '__main__':
    main()
