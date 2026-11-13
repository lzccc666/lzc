"""完整冷/热导航三任务、输入/输出、连续车辆及业务内容独立核验。"""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, Polygon

from conceptgraph.completion.independent_continuous_body import audit_trajectory, rectangle
from conceptgraph.completion.execution_direction_audit import audit_execution_direction
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(row) for row in Path(path).open()]


def signature(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def normalize(value, case):
    if isinstance(value, dict):
        return {key:normalize(item, case) for key, item in value.items()
            if key not in {'duration_s', 'latency'}}
    if isinstance(value, list):
        return [normalize(item, case) for item in value]
    if isinstance(value, str) and value.startswith(str(case)):
        return '<CASE>'+value[len(str(case)):]
    return value


def matches(frame, targets):
    nodes = sorted(frame['nodes'], key=lambda n:n['node_id'])
    if not nodes:
        return {}
    costs = np.array([[math.dist(n['center'][:2], t['center_xy']) if n['category'] == t['category'] else 1e6
        for t in targets] for n in nodes])
    rows, cols = linear_sum_assignment(costs)
    return {nodes[int(a)]['node_id']:targets[int(b)]['instance_id'] for a, b in zip(rows, cols) if costs[a, b] <= 2.5}


def audit_phase(case, task, config, vehicle, destination):
    result = read(case/'result.json')
    frames, execution, initial = lines(case/'perception.jsonl'), lines(case/'execution.jsonl'), read(case/'initial_state.json')
    assert initial['time_s'] == 0 and frames[0]['sensor']['state'] == initial
    assert not any(Polygon(r['polygon_xy']).covers(Point(initial['x'], initial['y'])) for r in task['regions'])
    assert len(execution) == result['saved_control_intervals'] and len(frames) == result['actual_frontend_calls']
    assert result['final_state'] == execution[-1]['state'] and not result['gt_in_agent']
    road = Polygon(config['road_polygon_xy'])
    obstacles = [rectangle(t['center_xy'], t['yaw_rad'], t['length_m']/2, t['length_m']/2, t['width_m']) for t in config['targets']]
    body = audit_trajectory(initial, execution, vehicle, road, obstacles)
    direction = audit_execution_direction(initial, execution, config['centerline_xy'], vehicle)
    ids, sensor_hashes, detections, files = [], [], [], {}
    correct, stable, streak, route = False, False, 0, False
    for i, frame in enumerate(frames):
        assert frame['frame_idx'] == i and abs(frame['sim_time_s']-i*.5) < 1e-8
        assert not frame['gt_sent_to_agent'] and frame['frontend_torch_threads'] == 1 and not frame['CUDA_initialized']
        sensor = frame['sensor']
        assert sensor['state'] == (initial if i == 0 else execution[i*5-1]['state'])
        for key in ['rgb', 'depth']:
            assert digest(sensor[key]) == sensor[key+'_sha256']
            files[sensor[key]] = sensor[key+'_sha256']
        z = np.array(Image.open(sensor['depth']))
        assert z.shape == (360, 640) and int((z > 0).sum()) == sensor['lidar_pixels']
        path = case/'detections'/f'{i:06d}.pkl.gz'
        assert digest(path) == frame['detector_sha256']
        payload = gzip.decompress(path.read_bytes())
        gob = pickle.loads(payload)
        assert gob['frame_provenance']['rgb_sha256'] == sensor['rgb_sha256']
        mapping = matches(frame, config['targets'])
        assert {str(k):v for k, v in mapping.items()} == frame['evaluation_only_matches']
        identity = mapping.get(frame['selected_node_id'])
        assert identity == frame['evaluation_only_identity']
        ids.append(identity)
        correct |= bool(frame['candidates'] and mapping.get(frame['candidates'][0]['node_id']) in task['accepted_instance_ids'])
        streak = streak+1 if identity in task['accepted_instance_ids'] else 0
        stable |= streak >= 3
        route |= bool(frame['planning'] and frame['path_world'] and frame['path_purpose'] == 'target_approach')
        sensor_hashes.append([sensor['rgb_sha256'], sensor['depth_sha256']])
        detections.append(hashlib.sha256(payload).hexdigest())
        files[str(path)] = digest(path)
    old, index, beginning, longest, intervals = initial, 0, None, 0., []
    for row in execution:
        new = row['state']
        while index+1 < len(frames) and frames[index+1]['sim_time_s'] <= old['time_s']+1e-9:
            index += 1
        dt = new['time_s']-old['time_s']
        assert abs(dt-.1) < 1e-8
        travel = (old['speed_mps']+new['speed_mps'])/2*dt
        valid = False
        for region in task['regions']:
            if region['instance_id'] != ids[index] or ids[index] not in task['accepted_instance_ids']:
                continue
            polygon, start = Polygon(region['polygon_xy']), Point(old['x'], old['y'])
            position = polygon.covers(start) and polygon.boundary.distance(start) >= travel+1e-9
            heading = max(abs(math.remainder(s['yaw_rad']-region['heading_rad'], 2*math.pi)) for s in [old, new]) <= math.pi/12
            turn = abs(math.remainder(new['yaw_rad']-old['yaw_rad'], 2*math.pi)) < math.pi/2
            valid |= position and heading and turn and max(old['speed_mps'], new['speed_mps']) <= .15 and new['time_s'] <= 90
        if valid:
            if beginning is None:
                beginning = old['time_s']
            longest = max(longest, new['time_s']-beginning)
        else:
            beginning = None
        intervals.append(dict(start_s=old['time_s'], end_s=new['time_s'], identity=ids[index],
            valid_whole_interval=bool(valid), held_s=0 if beginning is None else new['time_s']-beginning))
        old = new
    arrived = longest >= 2-1e-8 and body['passed'] and direction['passed'] is True
    four = dict(route_generated=route, selected_object_evaluated_correct=correct,
        stable_correct_lock=stable, actual_full_perception_arrival=arrived)
    assert all(result[k] == value for k, value in four.items())
    destination.mkdir()
    dump(destination/'body.json', body)
    dump(destination/'direction.json', direction)
    dump(destination/'goal_hold_intervals.json', intervals)
    dump(destination/'files.json', files)
    imports = read(case/'import_audit.json')
    assert imports['all_conceptgraph_modules_from_isolated_root'] and not imports['CUDA_initialized']
    for record in imports['modules'].values():
        assert digest(record['path']) == record['sha256']
        assert Path(record['path']).is_relative_to(Path(imports['source_root']))
    requests = [read(path) for path in sorted((case/'logical_requests').glob('*.json'))]
    assert len(requests) == 3 and Counter(r['request']['task'] for r in requests) == dict(query_parse=1, object_attributes=1, candidate_rerank=1)
    events = read(case/'semantic_events.json')
    delivered = {row['request_id']:row for row in events['events'] if row['kind'] == 'delivered'}
    assert len(delivered) == 3 and events['worker']['remaining_not_delivered'] == 0
    assert not any(row['kind'] == 'discarded' for row in events['events'])
    assert not events['real_clock_async'] and not events['actual_CUDA_backend']
    normalized_requests = []
    for request in requests:
        assert request['delivery_frame'] == request['submitted_frame']+1
        assert delivered[request['request']['request_id']]['frame'] == request['delivery_frame']
        assert request['synchronous_logical_evaluation'] and not request['real_clock_async']
        payload = request['request']['payload']
        assert signature(payload) == request['request']['payload_sha256'] or hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest() == request['request']['payload_sha256']
        value = normalize(request, case)
        # 真实提交路径SHA保存在原请求与files；比较模型实际输入时规范化角色路径后重算。
        value['request']['payload_sha256'] = signature(value['request']['payload'])
        info = value['response']['cache']
        assert info['enabled'] and info['key']
        value['response']['cache'] = dict(enabled=info['enabled'], key=info['key'])
        normalized_requests.append(value)
    canonical_frames = normalize(frames, case)
    for frame, payload_sha in zip(canonical_frames, detections):
        frame['detector_sha256'] = dict(identity='entire_uncompressed_pickle', sha256=payload_sha)
    dump(destination/'report.json', dict(passed=body['passed'] and (direction['passed'] is True or direction['moving_intervals'] == 0),
        four_outcomes=four, controls=len(execution), perception_frames=len(frames),
        actual_reported_Qwen_calls=result['actual_Qwen_backend_calls'], independent_hold_s=longest,
        task_schema_valid_counts=dict(Counter(r['request']['task'] for r in requests if r['response']['output_schema_valid'])),
        fallback_counts=dict(Counter(r['request']['task'] for r in requests if r['response']['fallback_used'])),
        actual_source_imports_verified=len(imports['modules']), input_files_checked=len(files),
        identity_criterion='原粗类别XY2.5米Hungarian重算，非新人工实例真值', goal_complete=False))
    return dict(initial=initial, frames=canonical_frames, execution=normalize(execution, case),
        requests=normalized_requests, sensors=sensor_hashes, detection_payloads=detections,
        four=four, final=result['final_state'], cache=read(case/'cache_files.json'),
        independent_safety_passed=body['passed'] and (direction['passed'] is True or direction['moving_intervals'] == 0))


def main():
    base = OUTPUT/'reproduction/three_task_fixedlogical_001'
    assert (base/'execution_finished.json').exists()
    manifest, release = read(base/'manifest.json'), read(base/'experiment_release.json')
    for name, expected in release['source_sha256'].items():
        assert digest(PROJECT/name) == digest(Path(manifest['isolated_root'])/name) == expected
    for name, expected in release['auxiliary_sha256'].items():
        assert digest(name) == expected
    root = Path(manifest['isolated_root'])
    data = read(root/'outputs/project_v2/thesis_completion_v1/interactive_simulation/synthetic_tasks_v1.json')
    task = next(t for t in data['tasks'] if t['task_id'] == 'config00-positive0')
    config = next(c for c in data['configs'] if c['config_id'] == 'config00')
    destination = base/'independent_audit_001'
    destination.mkdir(exist_ok=False)
    try:
        cases = {phase:root/'outputs/project_v2/thesis_completion_v1/interactive_simulation/three_task_fixedlogical_001'/phase for phase in ['cold', 'warm']}
        model = read(cases['cold']/'qwen_loaded_model.json')
        assert model['actual_parameter_devices'] == ['cpu'] and not model['CUDA_initialized']
        assert model['parameter_count'] > 1_000_000_000 and not (cases['warm']/'qwen_loaded_model.json').exists()
        cold, warm = [audit_phase(cases[phase], task, config, data['vehicle'], destination/phase) for phase in ['cold', 'warm']]
        assert cold['cache']['before'] == {} and cold['cache']['after'] == warm['cache']['before'] == warm['cache']['after']
        assert read(cases['cold']/'result.json')['actual_Qwen_backend_calls'] == 3
        assert read(cases['warm']/'result.json')['actual_Qwen_backend_calls'] == 0
        for phase in ['cold', 'warm']:
            requests = [read(p) for p in sorted((cases[phase]/'logical_requests').glob('*.json'))]
            assert all(r['response']['raw_output'] is not None and bool(r['response']['cache']['hit']) == (phase == 'warm') for r in requests)
        containers = []
        cold_files = sorted((cases['cold']/'detections').glob('*.pkl.gz'))
        warm_files = sorted((cases['warm']/'detections').glob('*.pkl.gz'))
        assert [p.name for p in cold_files] == [p.name for p in warm_files]
        for a, b in zip(cold_files, warm_files):
            left, right = a.read_bytes(), b.read_bytes()
            assert left[:3] == right[:3] == b'\x1f\x8b\x08'
            outside_mtime_equal = left[:4]+left[8:] == right[:4]+right[8:]
            containers.append(dict(name=a.name, cold_sha256=digest(a), warm_sha256=digest(b),
                entire_uncompressed_sha256=hashlib.sha256(gzip.decompress(left)).hexdigest(),
                cold_mtime=int.from_bytes(left[4:8], 'little'), warm_mtime=int.from_bytes(right[4:8], 'little'),
                all_bytes_outside_gzip_mtime_equal=outside_mtime_equal))
            assert outside_mtime_equal and gzip.decompress(left) == gzip.decompress(right)
        dump(destination/'detector_containers.json', containers)
        checks = {key:dict(equal=cold[key] == warm[key], cold_sha256=signature(cold[key]), warm_sha256=signature(warm[key]))
            for key in ['initial', 'frames', 'execution', 'requests', 'sensors', 'detection_payloads', 'four', 'final']}
        differences = {key:dict(cold=cold[key], warm=warm[key]) for key, check in checks.items() if not check['equal']}
        dump(destination/'differences.json', differences)
        passed = all(check['equal'] for check in checks.values()) and cold['independent_safety_passed'] and warm['independent_safety_passed']
        dump(destination/'report.json', dict(passed=passed,
            updated_utc=datetime.now(timezone.utc).isoformat(), comparisons=checks,
            actual_cold_CPU_Qwen_calls=3, actual_warm_Qwen_calls=0, real_frontend_both_runs=True,
            exact_cache_file_set_and_content_unchanged=True, full_three_task_fixedlogical_small_task_gate=passed,
            detector_containers_compared=len(containers), gzip_only_mtime_difference_verified=True,
            original_raw_and_normalized_model_outputs_compared=True, removed_business_fields=[], numerical_tolerance_used=False,
            nonbusiness_normalizations=['实测duration_s/latency', 'case角色路径及依赖路径SHA重算',
                '已核验的完整检测pickle内容代替gzip容器mtime', '缓存hit/status/diagnostic单列核验，比较同一cache key'],
            grid_business_state=None, grid_reason='本链车辆规划使用冻结连续道路/障碍几何；无另外保存的占用栅格。Hybrid搜索离散化源码相同。',
            real_clock_async=False, shared_CPU=True, GPU_performance_credit=False,
            new_independent_tasks=0, frozen_main_table_credit=False, human_labels_added=0,
            source_sha256=digest(__file__), goal_complete=False))
        print(json.dumps(read(destination/'report.json'), ensure_ascii=False))
    except BaseException as exc:
        dump(destination/'interrupted.json', dict(error=repr(exc), goal_complete=False))
        raise


if __name__ == '__main__':
    main()
