"""隔离标签端用官方有向车道和独立控制见证准备静态可达参考。"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from conceptgraph.completion.reference_regions import load_reference_prior
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump
from conceptgraph.navigation.vehicle_geometry import (
    VehicleSpec, VehicleState, angle_difference, footprint_polygon, step_vehicle, swept_body_polygon,
)


POLICY = dict(maximum_lane_edges=20, maximum_route_length_m=150.,
    maximum_start_heading_error_rad=math.radians(20), maximum_reference_heading_error_rad=math.radians(30),
    maximum_connector_gap_m=.75, maximum_steps=3500, cruise_speed_mps=.5,
    lookahead_m=1., hold_steps=10, start_speed_mps=0., start_steering_rad=0.,
    static_obstacle_scope='冻结官方sample时刻所有类别/所有运动状态盒；不保证未来交通',
    initial_state_scope='已知相机时刻后轴位置/航向，从静止、零转角起步；不是原录制车辆实测速率',
    unsuccessful_reference='unknown；绝不由有限参考搜索失败标记不可达')


def lane_paths(prior, start, end_token):
    """仅枚举官方有向连接，未找到有限参考保持未知。"""
    lanes = {r['token']: r for r in prior['lanes']}
    candidates = []
    point = Point(start.x, start.y)
    for token, lane in lanes.items():
        if not Polygon(lane['polygon']).covers(point):
            continue
        samples = lane['centerline']
        k = int(np.argmin(np.linalg.norm(samples[:, :2]-[start.x, start.y], axis=1)))
        if abs(angle_difference(float(samples[k, 2]), start.yaw_rad)) <= POLICY['maximum_start_heading_error_rad']:
            candidates.append(token)
    queue = deque((token,) for token in sorted(candidates))
    found = []
    expansions = 0
    while queue and expansions < 4000 and len(found) < 4:
        path = queue.popleft()
        expansions += 1
        if path[-1] == end_token:
            found.append(path)
            continue
        if len(path) >= POLICY['maximum_lane_edges']:
            continue
        for token in sorted(lanes[path[-1]]['outgoing']):
            if token in lanes and token not in path:
                queue.append((*path, token))
    return found


def reference_line(prior, path, start, region):
    lanes = {r['token']: r for r in prior['lanes']}
    points = []
    for token in path:
        samples = np.asarray(lanes[token]['centerline'])[:, :2]
        if points and math.dist(points[-1], samples[0]) > POLICY['maximum_connector_gap_m']:
            return None, 'official_centerline_gap'
        for xy in samples:
            if not points or math.dist(points[-1], xy) > 1e-5:
                points.append(xy.tolist())
    if len(points) < 2:
        return None, 'short_reference'
    original = LineString(points)
    begin = original.project(Point(start.x, start.y))
    end = original.project(Point(region['nominal_rear_axle_xy']))
    if end-begin < .25 or end-begin > POLICY['maximum_route_length_m']:
        return None, 'reference_not_forward_or_beyond_budget'
    # 首尾为真实后轴与区域名义点，不把相机或车道投影冒充车辆初始位置。
    distances = np.arange(min(end, begin+1.), end, .25)
    xy = [[start.x, start.y], *[[original.interpolate(s).x, original.interpolate(s).y] for s in distances],
          region['nominal_rear_axle_xy']]
    line = LineString(xy)
    return line, None


def reference_control(state, line, region, held):
    """独立低速参考跟随器；不调用待评测规划器、排序或现有导航控制器。"""
    spec = VehicleSpec()
    point = Point(state.x, state.y)
    progress = line.project(point)
    remaining = max(0., line.length-progress)
    target = line.interpolate(min(line.length, progress+POLICY['lookahead_m']))
    alpha = angle_difference(math.atan2(target.y-state.y, target.x-state.x), state.yaw_rad)
    steering = math.atan2(2*spec.wheelbase_m*math.sin(alpha), max(point.distance(target), .05))
    inside = (Polygon(region['polygon_xy']).covers(point)
        and abs(angle_difference(state.yaw_rad, region['heading_rad'])) <= region['max_heading_error_rad'])
    desired = min(POLICY['cruise_speed_mps'], max(0., .9*(remaining-.025)))
    if held or inside and state.speed_mps <= .08:
        desired = 0.
    acceleration = (desired-state.speed_mps)*3.
    return acceleration, steering, inside


def integrate_reference(start, line, region, context, deadline):
    road, lanes = [prep(shape(context[k])) for k in ['drivable_geojson', 'lane_union_geojson']]
    obstacles = prep(unary_union([shape(o['polygon_geojson']) for o in context['objects']]))
    def legal(body):
        return road.covers(body) and lanes.covers(body) and not obstacles.intersects(body)
    if not legal(footprint_polygon(start)):
        return None, dict(reason='initial_body_not_certified', steps=0)
    state = start
    held, hold_count = False, 0
    rows = [dict(state=asdict(state), control=None)]
    for step in range(POLICY['maximum_steps']):
        if time.monotonic() > deadline:
            return None, dict(reason='reference_time_budget', steps=step)
        accel, steer, inside = reference_control(state, line, region, held)
        held = held or inside and state.speed_mps <= .08
        after = step_vehicle(state, accel, steer)
        sweep = swept_body_polygon(state, after)
        if not legal(sweep) or not legal(footprint_polygon(after)):
            return None, dict(reason='reference_body_or_sweep_not_certified', steps=step)
        rows.append(dict(state=asdict(after), control=dict(acceleration_mps2=accel, steering_command_rad=steer)))
        state = after
        point = Point(state.x, state.y)
        valid_end = (Polygon(region['polygon_xy']).covers(point) and state.speed_mps <= .02
            and abs(angle_difference(state.yaw_rad, region['heading_rad'])) <= region['max_heading_error_rad'])
        hold_count = hold_count+1 if valid_end else 0
        if hold_count >= POLICY['hold_steps']:
            return rows, dict(reason=None, steps=step+1, held_steps=hold_count,
                              simulation_time_s=state.time_s, endpoint=asdict(state))
    return None, dict(reason='finite_reference_did_not_certify_endpoint', steps=POLICY['maximum_steps'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--region-run', default='reference_regions_reserved_full_003')
    parser.add_argument('--input-run', default='reserved_causal_inputs_full_001')
    parser.add_argument('--target-limit', type=int, default=0)
    parser.add_argument('--case-budget-s', type=float, default=3.)
    parser.add_argument('--wall-budget-s', type=float, default=3600.)
    args = parser.parse_args()
    if args.target_limit < 0 or args.case_budget_s <= 0 or args.wall_budget_s <= 0:
        raise ValueError('invalid reference budget')
    started = time.monotonic()
    source = OUTPUT/'annotation'/args.region_run
    inputs = OUTPUT/'data_manifest'/args.input_run
    acceptance = json.loads((source/'acceptance_001.json').read_text())
    if not acceptance['static_region_certificates_accepted']:
        raise ValueError('requires accepted static endpoint labels')
    if not json.loads((inputs/'audit_001/report.json').read_text())['passed']:
        raise ValueError('requires independently audited known rear axle inputs')
    targets = [json.loads(line) for line in (source/'targets.jsonl').open()]
    selected = targets[:args.target_limit] if args.target_limit else targets
    locations = {r['scene']: r['strata']['location'] for r in json.loads(
        (OUTPUT/'splits/reserved_test_scenes_v2.json').read_text())['scenes']}
    dest = OUTPUT/'annotation'/args.run_id
    dest.mkdir(exist_ok=False)
    (dest/'witnesses').mkdir()
    dependencies = [Path(__file__), PROJECT/'conceptgraph/completion/reference_regions.py',
        PROJECT/'conceptgraph/navigation/vehicle_geometry.py', PROJECT/'conceptgraph/scripts/build_lane_constrained_bev.py']
    release = dict(version='independent-static-directed-lane-control-reference-v1', policy=POLICY,
        vehicle=asdict(VehicleSpec()), region_run=args.region_run, input_run=args.input_run,
        targets=len(selected), source_target_records=len(targets), target_limit=args.target_limit,
        source_sha256={str(p): digest(p) for p in dependencies},
        input_sha256={str(p): digest(p) for p in [source/'targets.jsonl', source/'queries.jsonl',
            source/'acceptance_001.json', source/'audit_001/report.json', inputs/'complete.json',
            inputs/'audit_001/report.json', OUTPUT/'splits/reserved_test_scenes_v2.json']},
        case_budget_s=args.case_budget_s, wall_budget_s=args.wall_budget_s,
        storage_budget_bytes=500000000, minimum_free_bytes=6000000000,
        evaluated_method_called=False, evaluated_planner_called=False, model_calls=0,
        evaluation_label_preparation_only=True, new_test_opened=False, actual_arrival_credit=False,
        independently_audited=False, goal_complete=False)
    dump(dest/'release.json', release)
    shutil.copy2(__file__, dest/'generator_source.py')
    priors, contexts, results = {}, {}, []
    for index, target in enumerate(selected):
        before = time.monotonic()
        result = dict(scene=target['scene'], frame_idx=target['frame_idx'], instance_token=target['instance_token'],
            reachability='unknown', reason='no_static_endpoint_certificate', reference=None, attempts=[])
        key = target['scene'], target['frame_idx']
        input_path = inputs/'converted'/key[0]/'inputs'/f'{key[1]:06d}.json'
        result['known_initial_pose'] = dict(path=str(input_path), sha256=digest(input_path))
        agent = json.loads(input_path.read_text())
        pose = np.asarray(agent['rear_axle_reference_world_pose'])
        start = VehicleState(float(pose[0, 3]), float(pose[1, 3]), agent['rear_axle_yaw_rad'])
        if target['regions']:
            result['reason'] = 'no_finite_reference_certificate'
            if time.monotonic()-started >= args.wall_budget_s:
                result['reason'] = 'global_budget_not_attempted'
            else:
                location = locations[key[0]]
                if location not in priors:
                    priors[location] = load_reference_prior(location)
                if key not in contexts:
                    path = source/target['context']['path']
                    if digest(path) != target['context']['sha256']:
                        raise ValueError('changed static context')
                    contexts[key] = json.loads(path.read_text())
                deadline = min(started+args.wall_budget_s, time.monotonic()+args.case_budget_s)
                for region_index, region in enumerate(target['regions']):
                    paths = lane_paths(priors[location], start, region['lane_token'])
                    if not paths:
                        result['attempts'].append(dict(region_index=region_index, reason='no_finite_official_directed_reference'))
                    for lane_tokens in paths:
                        line, reason = reference_line(priors[location], lane_tokens, start, region)
                        rows, detail = (integrate_reference(start, line, region, contexts[key], deadline)
                            if line is not None else (None, dict(reason=reason)))
                        result['attempts'].append(dict(region_index=region_index, lane_tokens=list(lane_tokens), **detail))
                        if rows is not None:
                            name = f'witnesses/target_{index:04d}.json'
                            dump(dest/name, dict(scene=key[0], frame_idx=key[1], instance_token=target['instance_token'],
                                region_index=region_index, region=region, context=target['context'],
                                known_initial_pose=result['known_initial_pose'], lane_tokens=list(lane_tokens),
                                reference_line_xy=list(line.coords), rows=rows,
                                static_reference_only=True, actual_vehicle_execution=False, evaluated_method_called=False))
                            result.update(reachability='pending_independent_witness_audit', reason=None,
                                          reference=dict(path=name, sha256=digest(dest/name)))
                            break
                    if result['reference'] or time.monotonic() >= deadline:
                        break
        result['elapsed_s'] = time.monotonic()-before
        results.append(result)
        with (dest/'targets.jsonl').open('a') as handle:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
        if shutil.disk_usage(dest).free < 6000000000:
            raise OSError('reference disk reserve; partial result preserved')
        if index % 10 == 0 or result['reference']:
            print(json.dumps(dict(completed=index+1, total=len(selected),
                proposed_witnesses=sum(r['reference'] is not None for r in results),
                elapsed_s=time.monotonic()-started), ensure_ascii=False), flush=True)
    if any(digest(name) != sha for name, sha in release['source_sha256'].items()):
        raise ValueError('reference source changed')
    files = {str(p.relative_to(dest)): digest(p) for p in dest.rglob('*') if p.is_file()}
    if sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()) > release['storage_budget_bytes']:
        raise OSError('reference storage budget exceeded; no acceptance')
    dump(dest/'summary.json', dict(target_records=len(results),
        proposed_witnesses=sum(r['reference'] is not None for r in results),
        status_counts=dict(Counter(r['reason'] or 'pending_independent_witness_audit' for r in results)),
        elapsed_s=time.monotonic()-started, files_sha256=files, independently_audited=False,
        actual_arrival_credit=False, full_perception_credit=False, model_calls=0,
        new_test_opened=False, goal_complete=False))


if __name__ == '__main__':
    main()
