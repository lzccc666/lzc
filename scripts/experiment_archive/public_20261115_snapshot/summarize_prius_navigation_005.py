"""关闭原厂Prius三任务开发证据；到达、身份、控制约束分别列。"""
import csv
import datetime
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import beta
from shapely.geometry import Polygon, Point

from conceptgraph.completion.stage9 import digest, dump

R = Path(__file__).resolve().parents[1]


def main():
    out = R/'carla_official/prius_navigation_summary_005'
    out.mkdir(exist_ok=False)
    rows, sources = [], {}
    for name in ['car20','car30_curve','no_match_bus20']:
        e = R/'carla_official/prius_controller_navigation_005'/name/'full_rule_geometry'
        paths = [e/'summary.json', e/'independent_audit_003/report.json',
            e/'independent_choice_audit_005/report.json', e/'release.json',
            e/'execution.jsonl', e/'perception.jsonl', e/'evaluation_only/truth.json']
        sources.update({str(p): digest(p) for p in paths})
        summary, arrival, choice = [json.loads(p.read_text()) for p in paths[:3]]
        rows.append(dict(episode=name, positive=choice['positive_task'],
            execution_ticks=summary['frames'], perception_calls=summary['perception_calls'],
            first_proposal_correct=choice['first_proposal_correct_ever'],
            correct_stable_lock=choice['correct_stable_lock_ever'],
            wrong_stable_lock=choice['wrong_stable_lock_ever'],
            route_generated=choice['route_generated'],
            geometric_identity_stop_arrival=arrival['geometric_identity_stop_arrival'],
            strict_safe_arrival=arrival['strict_safe_arrival'],
            no_match_correct_refusal=choice['no_match_correct_refusal'],
            no_collision=choice['no_collision'], no_lane_violation=choice['no_lane_violation'],
            max_acceleration_mps2=arrival['maximum_acceleration_mps2'],
            min_acceleration_mps2=arrival['minimum_acceleration_mps2'],
            actuation_violation_count=sum(arrival['actuation_limits'].values()),
            maximum_contiguous_hold_s=arrival['maximum_contiguous_hold_s'],
            status=summary['status'], wall_s=summary['wall_s'], sim_s=summary['sim_s'],
            mean_perception_update_s=choice['mean_frontend_mapping_selection_plan_s'],
            p95_perception_update_s=choice['p95_frontend_mapping_selection_plan_s']))
    with (out/'episodes.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    positive = [r for r in rows if r['positive']]
    k, n = sum(r['strict_safe_arrival'] for r in positive), len(positive)
    ci = [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),
          1. if k==n else float(beta.ppf(.975,k+1,n-k))]
    e = R/'carla_official/prius_controller_navigation_005/car30_curve/full_rule_geometry'
    perceptions = list(map(json.loads,(e/'perception.jsonl').open()))
    execution = list(map(json.loads,(e/'execution.jsonl').open()))
    truth = json.loads((e/'evaluation_only/truth.json').read_text())
    region = Polygon(truth['arrival_truth']['regions'][0]['polygon_xy'])
    distances = [float(region.distance(Point(p['state']['x'],p['state']['y']))) for p in execution]
    switches = sum(a['selected_node_id']!=b['selected_node_id']
        for a,b in zip(perceptions,perceptions[1:]))
    diagnosis = dict(episode='car30_curve', status='TIMEOUT',
        full_simulated_time_s=execution[-1]['state']['time_s'],
        minimum_distance_to_independent_arrival_region_m=min(distances),
        final_distance_to_independent_arrival_region_m=distances[-1],
        perception_frames_without_authorized_candidate=sum(not p['candidates'] for p in perceptions),
        perception_frames_without_path=sum(not p['path_world'] for p in perceptions),
        lock_identity_or_missing_state_switches=switches,
        emergency_braking_frames=sum(bool(p['tracking'].get('emergency_braking')) for p in execution),
        maximum_actual_speed_mps=max(p['state']['speed_mps'] for p in execution),
        interpretation='真实感知证据/锁定多次中断，未到达；未发现控制加减速超限。这里只做关联诊断，未证明单一根因。',
        goal_complete=False)
    dump(out/'car30_failure_diagnosis.json', diagnosis)
    used = sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
    report = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        episodes=rows, actual_new_episodes=3, unique_configuration_count=1, unique_seed_count=1,
        actual_execution_ticks=sum(r['execution_ticks'] for r in rows),
        actual_frontend_mapping_calls=sum(r['perception_calls'] for r in rows),
        strict_positive_arrival_numerator=k, strict_positive_arrival_denominator=n,
        strict_positive_arrival_rate=k/n, descriptive_binomial_95CI=ci,
        CI_limitation='固定开发场景和单种子，二项区间仅描述两个任务；不是跨场景泛化置信度',
        no_match_correct_refusal_count=sum(r['no_match_correct_refusal'] is True for r in rows),
        no_match_task_count=sum(not r['positive'] for r in rows),
        all_episodes_actuation_limits_pass=all(r['actuation_violation_count']==0 for r in rows),
        actual_Qwen_calls=0, source_sha256=sources, evaluator_sha256=digest(__file__),
        project_artifact_GiB=used/2**30, physical_free_GiB=shutil.disk_usage(R).free/2**30,
        historical_results_modified=False, real_time_claim=False,
        all_configurations_all_seeds_complete=False, whole_project_sealed_test=False,
        goal_complete=False)
    dump(out/'report.json', report)
    print(json.dumps({k:report[k] for k in ['actual_execution_ticks','actual_frontend_mapping_calls',
        'strict_positive_arrival_numerator','strict_positive_arrival_denominator',
        'no_match_correct_refusal_count','project_artifact_GiB','physical_free_GiB']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
