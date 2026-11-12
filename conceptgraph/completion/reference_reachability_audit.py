"""不调用参考生成器或待评测方法，重放控制并审计静态可达见证。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from conceptgraph.completion.reference_region_audit import independent_map
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump
from conceptgraph.scripts.build_lane_constrained_bev import _sample_arcline


def angular_error(a, b):
    return (a-b+math.pi) % (2*math.pi)-math.pi


def audit_control_rows(rows, vehicle, initial_pose):
    """独立写出离散自行车方程；不信任保存的位置、success或规划器标志。"""
    errors = []
    if len(rows) < 11:
        return ['insufficient_control_or_hold_rows']
    first = rows[0]['state']
    expected_initial = [initial_pose[0, 3], initial_pose[1, 3],
        math.atan2(initial_pose[1, 0], initial_pose[0, 0]), 0., 0., 0.]
    names = ['x', 'y', 'yaw_rad', 'speed_mps', 'steering_rad', 'time_s']
    if not np.allclose([first[k] for k in names], expected_initial, rtol=0, atol=1e-8):
        errors.append('initial_state_does_not_match_known_rear_axle_at_rest')
    for i, row in enumerate(rows):
        values = [row['state'][k] for k in names]
        if not np.isfinite(values).all():
            errors.append('nonfinite_state')
            continue
        if not i:
            continue
        a, b, control = rows[i-1]['state'], row['state'], row['control']
        if not np.isfinite([control['acceleration_mps2'], control['steering_command_rad']]).all():
            errors.append('nonfinite_control')
            continue
        dt = vehicle['dt_s']
        acceleration = float(np.clip(control['acceleration_mps2'],
            -vehicle['max_braking_mps2'], vehicle['max_acceleration_mps2']))
        v = float(np.clip(a['speed_mps']+acceleration*dt, 0, vehicle['max_speed_mps']))
        command = float(np.clip(control['steering_command_rad'],
            -vehicle['max_steering_rad'], vehicle['max_steering_rad']))
        steering = a['steering_rad']+float(np.clip(command-a['steering_rad'],
            -vehicle['max_steering_rate_radps']*dt, vehicle['max_steering_rate_radps']*dt))
        distance = (a['speed_mps']+v)/2*dt
        theta = distance*math.tan((a['steering_rad']+steering)/2)/vehicle['wheelbase_m']
        chord = distance*(math.sin(theta/2)/(theta/2) if abs(theta) > 1e-10 else 1.)
        expected = [a['x']+chord*math.cos(a['yaw_rad']+theta/2),
            a['y']+chord*math.sin(a['yaw_rad']+theta/2), angular_error(a['yaw_rad']+theta, 0.),
            v, steering, a['time_s']+dt]
        if not np.allclose(values, expected, rtol=0, atol=1e-6):
            errors.append('saved_state_not_obtained_by_control_integration')
        if not 0 <= b['speed_mps'] <= vehicle['max_speed_mps']+1e-8:
            errors.append('speed_limit')
        if abs(b['steering_rad']) > vehicle['max_steering_rad']+1e-8:
            errors.append('steering_limit')
    return sorted(set(errors))


def rectangle(state, vehicle):
    local = np.array([[-vehicle['rear_overhang_m'], -vehicle['width_m']/2],
        [vehicle['front_from_rear_axle_m'], -vehicle['width_m']/2],
        [vehicle['front_from_rear_axle_m'], vehicle['width_m']/2],
        [-vehicle['rear_overhang_m'], vehicle['width_m']/2]])
    c, s = math.cos(state['yaw_rad']), math.sin(state['yaw_rad'])
    return Polygon(local@np.array([[c, s], [-s, c]])+[state['x'], state['y']])


def audit_body_rows(rows, vehicle, road_polygon, lane_polygon, obstacle_polygon):
    road, lanes, obstacles = [prep(p) for p in [road_polygon, lane_polygon, obstacle_polygon]]
    errors = []
    previous = None
    for row in rows:
        state = row['state']
        body = rectangle(state, vehicle)
        if not road.covers(body) or not lanes.covers(body) or obstacles.intersects(body):
            errors.append('body_outside_road_lanes_or_hits_official_box')
        if previous is not None:
            before = previous['state']
            angle = abs(angular_error(state['yaw_rad'], before['yaw_rad']))
            if angle > math.pi/2:
                errors.append('integration_interval_excessive_rotation')
                previous = row
                continue
            chord = math.hypot(state['x']-before['x'], state['y']-before['y'])
            radius = chord/(2*math.sin(angle/2)) if angle > 1e-10 else 0.
            corner = math.hypot(max(vehicle['front_from_rear_axle_m'], vehicle['rear_overhang_m']),
                                vehicle['width_m']/2)
            bulge = (radius+corner)*(1-math.cos(angle/2))+1e-8
            envelope = unary_union([rectangle(before, vehicle), body]).convex_hull.buffer(
                bulge/math.cos(math.pi/64), quad_segs=16)
            if not road.covers(envelope) or not lanes.covers(envelope) or obstacles.intersects(envelope):
                errors.append('continuous_sweep_outside_road_lanes_or_hits_official_box')
        previous = row
    return sorted(set(errors))


def audit_directed_rows(rows, lane_tokens, lane_records, connectivity, heading_tolerance):
    """逐积分点有向lane分配动态规划；禁止跳过未声明连接。"""
    if not lane_tokens or any(t not in lane_records for t in lane_tokens):
        return ['missing_official_reference_lane']
    if any(b not in connectivity.get(a, {}).get('outgoing', []) for a, b in zip(lane_tokens, lane_tokens[1:])):
        return ['undeclared_official_lane_connection']
    active = {0}
    for n, row in enumerate(rows):
        s = row['state']
        xy = [s['x'], s['y']]
        point = Point(xy)
        valid = set()
        for index in active | {i+1 for i in active if i+1 < len(lane_tokens)}:
            lane = lane_records[lane_tokens[index]]
            if not lane['polygon'].covers(point):
                continue
            samples = lane['samples']
            k = int(np.argmin(np.linalg.norm(samples[:, :2]-xy, axis=1)))
            if abs(angular_error(s['yaw_rad'], float(samples[k, 2]))) > heading_tolerance:
                continue
            if not n and index != 0:
                continue
            if n:
                before = rows[n-1]['state']
                segment = LineString([[before['x'], before['y']], xy])
                possible = [i for i in active if index in (i, i+1)]
                if not any(unary_union([lane_records[lane_tokens[i]]['polygon'], lane['polygon']]).covers(segment)
                           for i in possible):
                    continue
            valid.add(index)
        active = valid
        if not active:
            return ['no_continuous_directed_lane_assignment']
    return [] if len(lane_tokens)-1 in active else ['witness_did_not_reach_declared_target_lane']


def audit_terminal_rows(rows, region, policy):
    errors = []
    hold = rows[-policy['hold_steps']:]
    if len(hold) != policy['hold_steps']:
        return ['insufficient_terminal_hold_rows']
    elapsed = hold[-1]['state']['time_s']-hold[0]['state']['time_s']
    if elapsed < policy.get('minimum_hold_duration_s', 0.)-1e-8:
        errors.append('terminal_hold_duration_too_short')
    if rows[-1]['state']['time_s'] > policy.get('maximum_reference_time_s', math.inf)+1e-8:
        errors.append('reference_exceeds_declared_horizon')
    endpoint = Polygon(region['polygon_xy'])
    for row in hold:
        s = row['state']
        if (not endpoint.covers(Point(s['x'], s['y'])) or s['speed_mps'] > .02
                or abs(angular_error(s['yaw_rad'], region['heading_rad'])) > region['max_heading_error_rad']):
            errors.append('endpoint_pose_speed_or_hold_failed')
    return sorted(set(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--audit-id', default='audit_001')
    args = parser.parse_args()
    started = time.monotonic()
    source = OUTPUT/'annotation'/args.run_id
    release, summary = [json.loads((source/f'{name}.json').read_text()) for name in ['release', 'summary']]
    dest = source/args.audit_id
    dest.mkdir(exist_ok=False)
    shutil.copy2(__file__, dest/'auditor_source.py')
    for group in ['source_sha256', 'input_sha256']:
        for name, sha in release[group].items():
            if digest(name) != sha:
                raise ValueError('changed reference input or source')
    for name, sha in summary['files_sha256'].items():
        if digest(source/name) != sha:
            raise ValueError('changed generated reference artifact')
    regions = OUTPUT/'annotation'/release['region_run']
    static_release = json.loads((regions/'release.json').read_text())
    if release['vehicle'] != static_release['vehicle']:
        raise ValueError('reference vehicle differs from accepted region vehicle')
    targets = {(r['scene'], r['frame_idx'], r['instance_token']): r for r in
               map(json.loads, (regions/'targets.jsonl').open())}
    locations = {r['scene']: r['strata']['location'] for r in json.loads(
        (OUTPUT/'splits/reserved_test_scenes_v2.json').read_text())['scenes']}
    maps, results, accepted = {}, [], set()
    total_states = 0
    for target in map(json.loads, (source/'targets.jsonl').open()):
        if not target['reference']:
            continue
        path = source/target['reference']['path']
        if digest(path) != target['reference']['sha256']:
            raise ValueError('changed witness sequence')
        witness = json.loads(path.read_text())
        key = target['scene'], target['frame_idx'], target['instance_token']
        official = targets[key]
        region = official['regions'][witness['region_index']]
        failures = []
        if region != witness['region'] or witness['lane_tokens'][-1] != region['lane_token']:
            failures.append('target_region_or_lane_mismatch')
        input_path = Path(witness['known_initial_pose']['path'])
        if digest(input_path) != witness['known_initial_pose']['sha256']:
            raise ValueError('changed known initial pose')
        initial = json.loads(input_path.read_text())
        if (initial['scene'], initial['frame_idx']) != key[:2]:
            failures.append('wrong_initial_scene_or_frame')
        context_path = regions/official['context']['path']
        if digest(context_path) != official['context']['sha256']:
            raise ValueError('changed accepted static context')
        context = json.loads(context_path.read_text())
        location = locations[key[0]]
        if location not in maps:
            raw_path = PROJECT/'dataset/nuscenes_map_expansion/expansion'/f'{location}.json'
            if digest(raw_path) != static_release['map_sha256'][location]:
                raise ValueError('changed official map')
            raw = json.loads(raw_path.read_text())
            geometry = independent_map(raw)
            lanes = {token: dict(polygon=p, samples=np.concatenate([
                _sample_arcline(pose, .125) for pose in raw['arcline_path_3'].get(token, [])]))
                for token, p in geometry['lanes'].items() if raw['arcline_path_3'].get(token)}
            maps[location] = dict(**geometry, records=lanes, connectivity=raw['connectivity'],
                                 lane_union=unary_union(list(geometry['lanes'].values())))
        prior = maps[location]
        rows = witness['rows']
        failures.extend(audit_control_rows(rows, release['vehicle'], np.asarray(initial['rear_axle_reference_world_pose'])))
        # 静态标签上游已独立核对全部原始官方盒；这里逐步使用其完整集合。
        obstacles = unary_union([shape(o['polygon_geojson']) for o in context['objects']])
        failures.extend(audit_body_rows(rows, release['vehicle'], prior['road'],
            prior['lane_union'], obstacles))
        failures.extend(audit_directed_rows(rows, witness['lane_tokens'], prior['records'],
            prior['connectivity'], release['policy']['maximum_reference_heading_error_rad']))
        failures.extend(audit_terminal_rows(rows, region, release['policy']))
        passed = not failures
        if passed:
            accepted.add(key)
        total_states += len(rows)
        results.append(dict(scene=key[0], frame_idx=key[1], instance_token=key[2], passed=passed,
            failures=sorted(set(failures)), witness_sha256=digest(path), checked_states=len(rows),
            checked_continuous_intervals=len(rows)-1))
    dump(dest/'witness_checks.json', results)
    query_rows = []
    for q in map(json.loads, (regions/'queries.jsonl').open()):
        matches = [token for token in q['accepted_instance_tokens'] if (q['scene'], q['frame_idx'], token) in accepted]
        query_rows.append(dict(query_id=q['query_id'], scene=q['scene'], frame_idx=q['frame_idx'], cohort=q['cohort'],
            accepted_instance_tokens=q['accepted_instance_tokens'], certified_instance_tokens=matches,
            reachability='conditionally_reachable_static_from_rest' if matches else 'unknown',
            reachable_label=True if matches else None, inaccessible_label=None,
            conditional_scope=release['policy']['initial_state_scope']+'；'+release['policy']['static_obstacle_scope'],
            source_query_unchanged=True, actual_arrival=None, full_perception_arrival_credit=False,
            human_verified=False, method_inference_executed=False, new_test_opened=False))
    with (dest/'queries.jsonl').open('x') as handle:
        for row in query_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    dump(dest/'report.json', dict(passed=all(r['passed'] for r in results),
        proposed_witnesses=len(results), accepted_static_from_rest_witnesses=len(accepted),
        failed_witnesses=sum(not r['passed'] for r in results), independently_checked_states=total_states,
        queries=len(query_rows), query_status=dict(Counter(r['reachability'] for r in query_rows)),
        certified_scenes=len({q['scene'] for q in query_rows if q['reachable_label']}),
        source_sha256=digest(Path(__file__)), generator_release_sha256=digest(source/'release.json'),
        elapsed_s=time.monotonic()-started, conditional_static_reference_only=True,
        minimum_hold_duration_s=release['policy'].get('minimum_hold_duration_s'),
        maximum_reference_time_s=release['policy'].get('maximum_reference_time_s'),
        missing_certificates_are_not_inaccessible=True, actual_vehicle_executions=0,
        full_perception_arrivals=0, model_calls=0, new_test_opened=False, goal_complete=False))


if __name__ == '__main__':
    main()
