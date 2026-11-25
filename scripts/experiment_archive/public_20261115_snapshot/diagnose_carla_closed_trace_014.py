"""从已闭合真实记录定位停车/候选消失事件；只读离线诊断，不重新授予到达。"""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

from shapely.geometry import Point, Polygon

from conceptgraph.navigation.vehicle_geometry import VehicleSpec, VehicleState, footprint_polygon
from summarize_carla_expansion_013 import digest, read, validate_closed

R = Path(__file__).resolve().parents[1]


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', required=True, help='carla_official下已闭合目录')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    episode = (R / 'carla_official' / args.episode).resolve()
    assert episode.is_relative_to(R / 'carla_official')
    assert Path(args.run_id).name == args.run_id
    out = R / 'carla_official' / args.run_id
    out.mkdir(exist_ok=False)
    inputs = {}
    release = read(episode / 'release.json', inputs)
    summary = read(episode / 'summary.json', inputs)
    audit = read(episode / 'independent_expansion_audit_009/report.json', inputs)
    guard = read(episode / 'independent_motion_guard_audit_002/report.json', inputs)
    validate_closed(summary, audit, guard)
    for path, expected in audit['inputs_sha256'].items():
        assert digest(path) == expected, path
    states = rows(episode / 'execution.jsonl.gz')
    observations = rows(episode / 'perception.jsonl.gz')
    spec = VehicleSpec(**release['vehicle_spec'])
    road = Polygon(release['prior']['road_polygon_xy'])
    cursor = 0
    trace = []
    for execution in states:
        while cursor + 1 < len(observations) and observations[cursor + 1]['sensor_frame'] <= execution['frame']:
            cursor += 1
        obs = observations[cursor]
        state = VehicleState(**execution['state'])
        body = footprint_polygon(state, spec)
        padded = footprint_polygon(state, spec, .05)
        nominal = execution['tracking']['always_on_motion_guard']
        semantic = obs['semantic_audit']
        plan = semantic.get('plan_check') or {}
        coasting = nominal.get('current_steering_coasting_forecast') or {}
        following = nominal.get('route_following_forecast') or {}
        center_goal = (obs.get('planning') or {}).get('goal_xy')
        row = dict(tick=execution['tick'], time_s=state.time_s,
                   observation_frame_idx=obs['frame_idx'],
                   speed_mps=state.speed_mps, steering_rad=state.steering_rad,
                   evidence_passed=semantic.get('current_evidence_passed'),
                   semantic_proposed_node_id=semantic.get('proposed_node_id'),
                   authorized_node_id=semantic.get('authorized_node_id'),
                   selected_node_id=obs['selected_node_id'],
                   lock_state=obs['lock']['state_after'],
                   candidate_count=len(obs['candidates']),
                   planned_path_points=len(obs['path_world']),
                   plan_failure=plan.get('failure'), plan_reason=plan.get('reason'),
                   motion_guard_stop=nominal['stop_required'],
                   following_forecast_reason=following.get('reason'),
                   coasting_forecast_reason=coasting.get('reason'),
                   coasting_first_bad_prediction_tick=(coasting.get('violations') or [{}])[0].get('tick'),
                   nominal_unpadded_body_inside_unbuffered_road=road.covers(body),
                   nominal_padded_body_inside_unbuffered_road=road.covers(padded),
                   nominal_padded_body_outside_road_area_m2=padded.difference(road).area,
                   nominal_unpadded_body_inside_audit_road_tolerance=road.buffer(.05).covers(body),
                   observed_goal_distance_m=(Point(state.x, state.y).distance(Point(center_goal))
                                             if center_goal else None),
                   applied_throttle=execution['applied_control']['throttle'],
                   applied_brake=execution['applied_control']['brake'],
                   next_throttle=execution['next_control']['throttle'],
                   next_brake=execution['next_control']['brake'])
        trace.append(row)
    with (out / 'trace.csv').open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    first = lambda predicate: next((r for r in trace if predicate(r)), None)
    relevant = lambda r: {k: r[k] for k in ['tick', 'time_s', 'speed_mps', 'selected_node_id',
                                           'evidence_passed', 'plan_failure',
                                           'motion_guard_stop', 'coasting_forecast_reason',
                                           'nominal_padded_body_outside_road_area_m2']}
    events = {}
    for name, predicate in [
        ('first_motion_guard_stop', lambda r: r['motion_guard_stop']),
        ('first_current_padded_body_outside_road', lambda r: not r['nominal_padded_body_inside_unbuffered_road']),
        ('first_invalid_initial_body_plan', lambda r: r['plan_failure'] == 'INVALID_INITIAL_BODY')]:
        match = first(predicate)
        events[name] = relevant(match) if match else None
    report = dict(episode=str(episode), physical_ticks=len(trace),
                  frontend_frames=len(observations), actual_status=summary['status'],
                  actual_strict_safe_arrival=audit['strict_safe_arrival'],
                  correct_proposal=audit['correct_first_proposal'], correct_lock=audit['correct_lock'],
                  wrong_lock=audit['wrong_lock'],
                  motion_guard_stop_ticks=sum(r['motion_guard_stop'] for r in trace),
                  evidence_pass_but_no_authorized_route_ticks=sum(
                      bool(r['evidence_passed']) and r['authorized_node_id'] is None for r in trace),
                  plan_failure_ticks=dict(Counter(r['plan_failure'] for r in trace if r['plan_failure'])),
                  current_padded_body_outside_road_ticks=sum(
                      not r['nominal_padded_body_inside_unbuffered_road'] for r in trace),
                  body_inside_evaluator_tolerance_all_ticks=all(
                      r['nominal_unpadded_body_inside_audit_road_tolerance'] for r in trace),
                  first_events=events,
                  interpretation='逐帧观察到的事件关联；未进行改变保护/规划/控制的干预，不宣称单一根因或反事实必然成功。保护将车身再扩0.05m后限制在原道路内，独立越界核验将道路外扩0.05m后检查原车身；两者口径不同，不能据此取消安全余量。',
                  frozen_execution_sources_changed=False, actual_new_model_calls=0,
                  actual_new_simulator_calls=0, inputs_sha256=inputs,
                  source_sha256=digest(__file__), goal_complete=False)
    with (out / 'report.json').open('x') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['actual_status', 'motion_guard_stop_ticks',
          'plan_failure_ticks', 'current_padded_body_outside_road_ticks', 'first_events']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
