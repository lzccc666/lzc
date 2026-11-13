"""审核原语义缺一450的已完成执行；严格初态与显式重建口径分列。"""
import argparse
import bisect
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tarfile
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, Polygon

from conceptgraph.completion.independent_continuous_body import audit_trajectory, rectangle
from conceptgraph.completion.execution_direction_audit import audit_execution_direction
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(row) for row in Path(path).open()]


def ticks(pid):
    try:
        fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except OSError:
        return None


def stable_prefix(path):
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raw = raw[:raw.rfind(b'\n')+1]
    return [json.loads(row) for row in raw.splitlines()], hashlib.sha256(raw).hexdigest()


def interpolate(points, progress):
    remaining = progress
    for a, b in zip(points, points[1:]):
        distance = math.dist(a, b)
        if remaining <= distance:
            return [a[j]+(b[j]-a[j])*remaining/distance for j in range(2)]
        remaining -= distance
    return points[-1]


def geometry(config, task):
    obstacles = [rectangle(t['center_xy'], t['yaw_rad'], t['length_m']/2,
        t['length_m']/2, t['width_m']) for t in config['targets']]
    if task['kind'] == 'inaccessible':
        a, b = interpolate(config['centerline_xy'], 17.9), interpolate(config['centerline_xy'], 18.1)
        xy = interpolate(config['centerline_xy'], 18.)
        obstacles.append(rectangle(xy, math.atan2(b[1]-a[1], b[0]-a[0]), 1., 1., 100.))
    return Polygon(config['road_polygon_xy']), obstacles


def association(frame, targets):
    nodes = sorted(frame['nodes'], key=lambda node: node['node_id'])
    if not nodes:
        return {}
    costs = np.array([[math.dist(node['center'][:2], target['center_xy'])
        if node['category'] == target['category'] else 1e6 for target in targets] for node in nodes])
    rows, columns = linear_sum_assignment(costs)
    return {nodes[int(a)]['node_id']:targets[int(b)]['instance_id']
        for a, b in zip(rows, columns) if costs[a, b] <= 2.5}


def initial(config, seed, frames, execution):
    first = frames[0]['sensor']['state']
    if first['time_s'] == 0:
        return first, dict(strict_observed_time_zero=True, reconstructed=False)
    rng = np.random.default_rng(seed)
    xy = np.array(config['start_xy'])+rng.normal(0, .03, 2)
    value = dict(x=float(xy[0]), y=float(xy[1]),
        yaw_rad=float(config['start_yaw_rad']+rng.normal(0, math.radians(.5))),
        speed_mps=0., steering_rad=0., time_s=0.)
    prefix = [row for row in execution if row['state']['time_s'] <= first['time_s']]
    supported = bool(prefix and prefix[-1]['state'] == first)
    for row in prefix:
        state = row['state']
        supported &= (row['stale_sensor_braking'] and not row['stop_requested']
            and row['sensor_age_s'] is None and row['control']['acceleration_mps2'] == -2.
            and state['speed_mps'] == state['steering_rad'] == 0.
            and math.dist([state['x'], state['y']], [value['x'], value['y']]) < 1e-12
            and abs(math.remainder(state['yaw_rad']-value['yaw_rad'], 2*math.pi)) < 1e-12)
    if not supported:
        return None, dict(strict_observed_time_zero=False, reconstructed=False,
            status='unknown_missing_initial_observation_and_unsupported_prefix')
    return value, dict(strict_observed_time_zero=False, reconstructed=True,
        first_observed_state=first, stationary_prefix_records=len(prefix),
        reconstructed_initial=value, reconstructed_is_not_observed=True,
        RNG='NumPy default_rng/PCG64 archived initialization order')


def hold(frames, execution, first, task, vehicle):
    completed = [frame['completed_wall_elapsed_s'] for frame in frames]
    assert all(a < b for a, b in zip(completed, completed[1:]))
    identities = [frame['_reconstructed_identity'] for frame in frames]
    accepted = set(task['accepted_instance_ids'])
    regions = [(region, Polygon(region['polygon_xy'])) for region in task['regions']]
    old, previous_wall, beginning, longest = first, 0., None, 0.
    witnesses = []
    for i, row in enumerate(execution):
        state, wall = row['state'], row['wall_elapsed_s']
        assert wall > previous_wall
        before = bisect.bisect_right(completed, previous_wall)-1
        after = bisect.bisect_right(completed, wall)-1
        ids = identities[before:after+1] if before >= 0 else []
        identity_ok = bool(ids) and all(identity in accepted for identity in ids)
        dt = state['time_s']-old['time_s']
        assert abs(dt-vehicle['dt_s']) < 1e-7
        travel = (old['speed_mps']+state['speed_mps'])/2*dt
        geometry_ok = False
        for region, polygon in regions:
            if not identity_ok or region['instance_id'] not in ids:
                continue
            start = Point(old['x'], old['y'])
            position = polygon.covers(start) and polygon.boundary.distance(start) >= travel+1e-9
            heading = max(abs(math.remainder(s['yaw_rad']-region['heading_rad'], 2*math.pi))
                for s in [old, state]) <= math.pi/12
            turn = abs(math.remainder(state['yaw_rad']-old['yaw_rad'], 2*math.pi))
            geometry_ok |= position and heading and turn < math.pi/2
        valid = bool(identity_ok and geometry_ok and max(old['speed_mps'], state['speed_mps']) <= .15
            and task['kind'] == 'reachable_positive' and state['time_s'] <= 90)
        if valid:
            if beginning is None:
                beginning = old['time_s']
            longest = max(longest, state['time_s']-beginning)
        else:
            beginning = None
        if valid:
            witnesses.append(dict(control_index=i, start_s=old['time_s'], end_s=state['time_s'],
                identities_across_interval=ids, held_s=state['time_s']-beginning))
        old, previous_wall = state, wall
    return dict(certified=longest >= 2.-1e-8, longest_whole_interval_hold_s=longest,
        valid_control_intervals=witnesses, uses_result_arrival_as_truth=False)


def case(record, index, destination, tasks, configs, vehicle):
    output = {key:record[key] for key in ['variant', 'task', 'config_id', 'seed', 'status', 'raw_case']}
    output.update(index=index, reported_arrival=record['actual_arrival'], new_arrival_credit=0)
    result = record['result']
    path = Path(record['raw_case'])
    if result is None:
        assert not record['actual_arrival'] and record['result_sha256'] is None
        output.update(audit_status='unknown_missing_complete_result', strict_initial_certificate=False,
            augmented_body_certificate=False, moving_direction_certificate=False,
            conservative_hold_certificate=False, saved_control_records=0, perception_records=0)
        return output
    assert read(path/'result.json') == result and digest(path/'result.json') == record['result_sha256']
    inputs = {str(path/name):digest(path/name) for name in ['result.json', 'perception.jsonl', 'execution.jsonl']}
    frames, execution = lines(path/'perception.jsonl'), lines(path/'execution.jsonl')
    assert frames and execution
    assert execution[-1]['state'] == result['final_state']
    assert len(execution) == result['actual_control_ticks']
    task, config = tasks[record['task']], configs[record['config_id']]
    selected, locked, streak, routed = False, False, 0, False
    for frame in frames:
        assert not frame['gt_sent_to_agent']
        matches = association(frame, config['targets'])
        identity = matches.get(frame['selected_node_id'])
        assert identity == frame['evaluation_only_identity']
        saved = {int(key):value for key, value in frame['evaluation_only_matches'].items()}
        assert saved == (matches if frame['selected_node_id'] is not None else {})
        frame['_reconstructed_identity'] = identity
        candidates = frame['candidates']
        selected |= bool(candidates and matches.get(candidates[0]['node_id']) in task['accepted_instance_ids'])
        streak = streak+1 if identity in task['accepted_instance_ids'] else 0
        locked |= streak >= 3
        routed |= bool(frame['planning'] and frame['path_world'] and frame['path_purpose'] == 'target_approach')
    assert (routed, selected, locked) == tuple(result[key] for key in
        ['route_generated', 'selected_object_evaluated_correct', 'stable_correct_lock'])
    first, proof = initial(config, record['seed'], frames, execution)
    target = destination/f'{index:04d}'
    target.mkdir()
    dump(target/'initial_state_proof.json', proof)
    if first is None:
        output.update(audit_status='unknown_initial_state', strict_initial_certificate=False,
            augmented_body_certificate=False, moving_direction_certificate=False,
            conservative_hold_certificate=False)
    else:
        road, obstacles = geometry(config, task)
        body = audit_trajectory(first, execution, vehicle, road, obstacles)
        direction = audit_execution_direction(first, execution, config['centerline_xy'], vehicle)
        held = hold(frames, execution, first, task, vehicle)
        certificates = dict(body=body, direction=direction, hold=held)
        with gzip.open(target/'certificates.json.gz', 'xt', encoding='utf-8') as handle:
            json.dump(certificates, handle, ensure_ascii=False, allow_nan=False)
        dump(target/'geometry.json', dict(road_polygon_xy=list(road.exterior.coords),
            obstacles_xy=[list(p.exterior.coords) for p in obstacles], source='independent scalar reconstruction'))
        output.update(audit_status=body['status'], strict_initial_certificate=proof['strict_observed_time_zero'] and body['passed'],
            strict_initial_observed=proof['strict_observed_time_zero'], explicit_initial_reconstructed=proof['reconstructed'],
            augmented_body_certificate=body['passed'], moving_direction_certificate=direction['passed'] is True,
            direction_status=direction['status'], moving_intervals=direction['moving_intervals'],
            conservative_hold_certificate=held['certified'] and body['passed'] and direction['passed'] is True,
            longest_whole_interval_hold_s=held['longest_whole_interval_hold_s'],
            audit_intervals=body['audited_intervals'], certificates_sha256=digest(target/'certificates.json.gz'))
    output.update(saved_control_records=len(execution), perception_records=len(frames), source_sha256=inputs,
        reconstructed_route=routed, reconstructed_selection=selected, reconstructed_lock=locked)
    output['reported_arrival_without_full_certificate'] = record['actual_arrival'] and not output['conservative_hold_certificate']
    output['uncredited_posthoc_hold'] = output['conservative_hold_certificate'] and not record['actual_arrival']
    assert all(digest(name) == value for name, value in inputs.items())
    dump(target/'report.json', output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert args.limit is None or args.limit > 0
    assert not (args.follow and args.limit is not None)
    source = OUTPUT/'interactive_simulation/realtime_semantic_minus_full_001'
    original = read(source/'release.json')
    task_path = OUTPUT/'interactive_simulation/synthetic_tasks_v1.json'
    assert digest(task_path) == original['task_release_sha256']
    data = read(task_path)
    tasks, configs = {t['task_id']:t for t in data['tasks']}, {c['config_id']:c for c in data['configs']}
    assert len(original['trials']) == 450
    with tarfile.open(source/'pipeline_snapshot.tar.gz') as archive:
        initializer = archive.extractfile('conceptgraph/completion/realtime_navigation.py').read()
    assert hashlib.sha256(initializer).hexdigest() == original['source_sha256']['conceptgraph/completion/realtime_navigation.py']
    assert 'rng = np.random.default_rng(seed)' in initializer.decode()
    destination = OUTPUT/'vehicle_execution'/args.run_id
    destination.mkdir(exist_ok=False)
    frozen = {str(p.relative_to(PROJECT)):digest(p) for p in [Path(__file__).resolve(),
        PROJECT/'conceptgraph/completion/independent_continuous_body.py',
        PROJECT/'conceptgraph/completion/execution_direction_audit.py',
        OUTPUT/'task_protocol/semantic_minus_execution_audit_001.md']}
    tick = ticks(268029)
    release = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=frozen,
        original_release_sha256=digest(source/'release.json'), task_release_sha256=digest(task_path),
        expected_trials=450, limit=args.limit, follow=args.follow, producer_pid=268029, producer_start_ticks=tick,
        budget_execution_wall_s=900 if args.limit else 3600, budget_wait_s=90000, budget_bytes=2_000_000_000,
        minimum_free_bytes=6_000_000_000, CUDA_visible='', actual_model_calls=0, new_executions=0, goal_complete=False)
    dump(destination/'release.json', release)
    started, computing, outputs = time.monotonic(), 0., []
    proc = Path('/proc')/str(os.getpid())
    dump(destination/'worker_process.json', dict(pid=os.getpid(), start_ticks=(proc/'stat').read_text().rsplit(')', 1)[1].split()[19],
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'), source_sha256=digest(__file__)))
    try:
        while True:
            assert all(digest(PROJECT/name) == value for name, value in frozen.items())
            assert digest(source/'release.json') == release['original_release_sha256']
            assert shutil.disk_usage(OUTPUT).free > release['minimum_free_bytes']
            records, prefix_sha = stable_prefix(source/'trial_records.jsonl')
            assert len(records) <= 450
            for index, row in enumerate(records):
                assert all(row[key] == value for key, value in original['trials'][index].items())
            available = records[:args.limit] if args.limit else records
            for index in range(len(outputs), len(available)):
                before = time.monotonic()
                output = case(available[index], index, destination, tasks, configs, data['vehicle'])
                computing += time.monotonic()-before
                outputs.append(output)
                with (destination/'cases.jsonl').open('a') as handle:
                    handle.write(json.dumps(output, ensure_ascii=False, allow_nan=False)+'\n')
                assert computing < release['budget_execution_wall_s']
                if index % 20 == 0:
                    assert sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()) < release['budget_bytes']
                print(json.dumps(dict(audited=len(outputs), available=len(records), status=output['audit_status'],
                    original_arrival=output['reported_arrival'], full_certificate=output['conservative_hold_certificate'])), flush=True)
            full = len(outputs) == len(records) == 450 and (source/'summary.json').exists()
            counts = Counter()
            for row in outputs:
                counts['body_'+row['audit_status']] += 1
                for key in ['strict_initial_certificate', 'explicit_initial_reconstructed', 'augmented_body_certificate',
                    'moving_direction_certificate', 'reported_arrival', 'conservative_hold_certificate',
                    'reported_arrival_without_full_certificate', 'uncredited_posthoc_hold']:
                    counts[key] += bool(row.get(key))
            report = dict(updated_utc=datetime.now(timezone.utc).isoformat(), complete=full,
                audited_cases=len(outputs), expected_cases=450, is_smoke=args.limit is not None,
                frozen_prefix_SHA=prefix_sha, original_available_records=len(records), counts=dict(counts),
                control_intervals=sum(row['saved_control_records'] for row in outputs),
                moving_intervals=sum(row.get('moving_intervals', 0) for row in outputs),
                perception_records=sum(row['perception_records'] for row in outputs),
                unresolved_arrivals=[row for row in outputs if row.get('reported_arrival_without_full_certificate')],
                source_sha256=frozen, wait_and_execution_elapsed_s=time.monotonic()-started, computing_wall_s=computing,
                actual_model_calls=0, new_executions=0, new_arrivals=0, new_test_opened=False,
                independent_identity_ground_truth=False, initial_reconstruction_is_not_observed=True,
                single_lane_connection_accuracy=None, goal_complete=False)
            temporary = destination/'progress.json.tmp'
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
            temporary.replace(destination/'progress.json')
            if full:
                summary = read(source/'summary.json')
                assert summary['attempted'] == summary['expected'] == 450
                assert summary['trial_records_sha256'] == digest(source/'trial_records.jsonl')
                dump(destination/'report.json', report)
                dump(destination/'complete.json', dict(complete_cohort=True, cases=450,
                    unresolved_original_arrivals=counts['reported_arrival_without_full_certificate'],
                    report_sha256=digest(destination/'report.json'), goal_complete=False))
                break
            if not args.follow:
                dump(destination/'report.json', report)
                break
            assert time.monotonic()-started < release['budget_wait_s']
            assert ticks(268029) == tick and tick is not None, '原进程已终止但完整summary缺失，不重启模型'
            time.sleep(5)
    except BaseException as exc:
        dump(destination/'interrupted.json', dict(error=repr(exc), audited_cases=len(outputs), goal_complete=False))
        raise


if __name__ == '__main__':
    main()
