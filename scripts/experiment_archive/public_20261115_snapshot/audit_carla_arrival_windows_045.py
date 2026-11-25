"""分解真实到达条件的时间重合；不改变目标区、速度门限或到达成绩。"""
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon

from diagnose_carla_ten_bottlenecks_043 import sha

R = Path(__file__).resolve().parents[1]


def main():
    out = R / 'carla_official/arrival_windows_045'
    out.mkdir(exist_ok=False)
    inputs = {}

    def read(p):
        inputs[str(p)] = sha(p)
        return json.loads(p.read_text())

    rows = []
    details = []
    for number in (1, 2, 5, 6, 10):
        job = read(R / f'json/expansion_jobs_011_primary/job_{number:03d}.json')
        d = Path(job['episode_directory'])
        gt = read(d / 'evaluation_only/truth.json')['arrival_truth']
        region = Polygon(gt['regions'][0]['polygon_xy'])
        ep, pp = d / 'execution.jsonl.gz', d / 'perception.jsonl.gz'
        inputs[str(ep)] = sha(ep)
        inputs[str(pp)] = sha(pp)
        audit = read(d / 'independent_expansion_audit_009/report.json')
        inside = []
        overlap = []
        events = []
        last_e = None
        max_hold = 0.
        with gzip.open(ep, 'rt') as f:
            for line in f:
                e = json.loads(line)
                v = e['evaluation']
                max_hold = max(max_hold, v['held_s'])
                simple = dict(tick=e['tick'], time_s=e['state']['time_s'],
                    speed_mps=e['state']['speed_mps'], correct_identity=v['correct_identity'],
                    inside=v['inside_independent_region'], heading_ok=v['heading_ok'],
                    stopped=v['stopped'], held_s=v['held_s'],
                    region_distance_m=region.distance(Point(e['state']['x'], e['state']['y'])),
                    applied_throttle=e['applied_control']['throttle'],
                    applied_brake=e['applied_control']['brake'],
                    next_brake=e['next_control']['brake'])
                if v['inside_independent_region']:
                    inside.append(simple)
                if all(v[k] for k in ('correct_identity', 'inside_independent_region',
                                      'heading_ok', 'stopped', 'safe_so_far')):
                    overlap.append(simple)
                if v['inside_independent_region'] and (not events or simple['tick'] > events[-1]['tick'] + 1):
                    events.append(simple)
                elif v['inside_independent_region']:
                    events[-1] = simple
                last_e = simple
        goals = []
        missed = []
        planchecks = 0
        with gzip.open(pp, 'rt') as f:
            for line in f:
                p = json.loads(line)
                if p['semantic_audit']['plan_check'].get('passed') and p['selected_node_id'] is not None:
                    planchecks += 1
                    if p['planning'] is None:
                        missed.append(p['frame_idx'])
                if p['planning'] and p['planning'].get('goal_xy'):
                    goal = p['planning']['goal_xy']
                    if p['path_purpose'] == 'target_approach':
                        goals.append(dict(frame_idx=p['frame_idx'], goal_xy=goal,
                            distance_to_independent_region_m=region.distance(Point(goal)),
                            distance_to_region_center_m=float(np.linalg.norm(np.asarray(goal) -
                                [region.centroid.x, region.centroid.y]))))
        row = dict(job=number, episode=job['episode'], strict_safe_arrival=audit['strict_safe_arrival'],
            inside_region_physical_ticks=len(inside),
            min_speed_inside_region_mps=min((v['speed_mps'] for v in inside), default=None),
            first_inside_time_s=inside[0]['time_s'] if inside else None,
            last_inside_time_s=inside[-1]['time_s'] if inside else None,
            inside_but_moving_ticks=sum(not v['stopped'] for v in inside),
            inside_but_wrong_or_unknown_identity_ticks=sum(not v['correct_identity'] for v in inside),
            simultaneous_valid_ticks=len(overlap), max_contiguous_hold_s=max_hold,
            valid_selected_plan_checks=planchecks, plans_not_installed_same_frame=len(missed),
            first_observed_goal_outside_region_m=goals[0]['distance_to_independent_region_m'] if goals else None,
            last_observed_goal_outside_region_m=goals[-1]['distance_to_independent_region_m'] if goals else None,
            final_speed_mps=last_e['speed_mps'], final_region_distance_m=last_e['region_distance_m'])
        rows.append(row)
        details.append(dict(**row, first_inside=inside[:1], last_inside=inside[-1:],
            first_jointly_valid=overlap[:1], final_state=last_e,
            observed_goals=goals, valid_plan_without_install_frames=missed))
    with (out / 'episodes.csv').open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report = dict(episodes=details,
        supersedes_hypothesis_in='ten_bottlenecks_043/report.json#code_contract_issue',
        hypothesis_disposition='本批所有锁定目标的有效计划均已在同帧安装；原分支潜在风险不是本批停滞原因。044仅为未部署的防御候选，不计入系统改善或完成验收。',
        actual_new_CARLA_calls=0, actual_new_GPU_calls=0, changed_arrival_thresholds=False,
        source_sha256=sha(__file__), inputs_sha256=inputs, goal_complete=False)
    with (out / 'report.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == '__main__':
    main()
