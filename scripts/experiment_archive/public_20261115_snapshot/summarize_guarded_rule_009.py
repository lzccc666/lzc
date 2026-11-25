"""独立到达/选择核验之后分层解释008规则回合，不把选对或锁定记成到达。"""
import datetime
import json
from pathlib import Path

import numpy as np
from conceptgraph.completion.stage9 import digest, dump

R = Path(__file__).resolve().parents[1]


def main():
    root = R/'carla_official/guarded_complex_navigation_008/complex_second_nearest20/rule'
    previous = R/'carla_official/complex_semantic_navigation_007/complex_second_nearest20/rule'
    out = R/'carla_official/guarded_rule_diagnosis_009'
    out.mkdir(exist_ok=False)
    summary = json.loads((root/'summary.json').read_text())
    arrival = json.loads((root/'independent_audit_003/report.json').read_text())
    choices = json.loads((root/'independent_choice_audit_008/report.json').read_text())
    old = json.loads((previous/'independent_choice_audit_008/report.json').read_text())
    perception = list(map(json.loads, (root/'perception.jsonl').open()))
    execution = list(map(json.loads, (root/'execution.jsonl').open()))
    for path, sha in choices['source_sha256'].items():
        assert digest(path) == sha
    assert len(perception) == summary['perception_calls'] == choices['observations']
    assert len(execution) == summary['frames']
    elapsed_frontend = np.array([r['measured_wall_s'] for r in perception])
    planning_timeouts = [r['frame_idx'] for r in perception
        if r['semantic_audit']['plan_check'].get('failure') == 'PLANNING_TIME_BUDGET']
    evidence_gaps = [r['frame_idx'] for r in perception
        if not r['semantic_audit']['current_evidence_passed']]
    moving = sum(r['state']['speed_mps'] > .15 for r in execution)
    no_path = sum(r['tracking']['always_on_motion_guard']['route_following_forecast']['checked_without_route']
        for r in execution)
    rows = [dict(frame_idx=r['frame_idx'], sim_time_s=r['sim_time_s'],
        actual_frontend_mapping_planning_wall_s=r['measured_wall_s'],
        current_evidence_passed=r['semantic_audit']['current_evidence_passed'],
        authorized_node_id=r['semantic_audit']['authorized_node_id'],
        committed_identity=r['semantic_audit'].get('intent_committed', False),
        plan_failure=r['semantic_audit']['plan_check'].get('failure'),
        path_purpose=r['path_purpose'], path_states=len(r['path_world'])) for r in perception]
    with (out/'frame_causes.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    value = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_new_physical_runs_by_this_analysis=0, completed_new_rule_episode=1,
        physics_ticks=len(execution), perception_updates=len(perception), final_status=summary['status'],
        route_generated=summary['route_generated'], actual_first_proposal_correct=choices['authorized_first_proposal_correct_ever'],
        correctly_stably_locked=choices['correct_stable_lock_ever'], wrong_stable_lock=choices['wrong_stable_lock_ever'],
        strict_safe_arrival=arrival['strict_safe_arrival'], collision_count=len(summary['collisions']),
        lane_violation=summary['lane_violation'], actuation_violations=arrival['actuation_limits'],
        simulation_time_s=summary['sim_s'], actual_episode_wall_s=summary['wall_s'],
        frontend_mapping_and_planning_wall_sum_s=float(elapsed_frontend.sum()),
        frontend_mapping_and_planning_median_s=float(np.median(elapsed_frontend)),
        frontend_mapping_and_planning_p95_s=float(np.quantile(elapsed_frontend, .95)),
        planning_10s_budget_failures=len(planning_timeouts), planning_failure_frames=planning_timeouts,
        current_evidence_gap_frames=evidence_gaps, current_evidence_gap_count=len(evidence_gaps),
        physics_ticks_without_route=no_path, physics_ticks_moving_above_point15=moving,
        total_guard_stop_requests=sum(r['tracking']['always_on_motion_guard']['stop_required'] for r in execution),
        interpretation='本回合瓶颈主要在55次有界规划超时、证据中断后的反复减速和重起步；始终保护未发出stop，不能把低速停滞归因于新增保护过于保守。耗时统计包含感知/建图/规划，不能全部称为检测器耗时。',
        descriptive_previous_same_task=dict(old_status=old['final_status'], old_wrong_lock=old['wrong_stable_lock_ever'],
            new_status=summary['status'], new_wrong_lock=choices['wrong_stable_lock_ever'],
            old_strict_safe_arrival=old['strict_safe_arrival'], new_strict_safe_arrival=arrival['strict_safe_arrival'],
            scope='相同一个开发任务的多项共同修复前后描述，不是单因素消融或独立泛化统计'),
        comparison_task_denominator=1, independent_configurations=1, seeds=[17],
        full_motion_guard_readback_pending=True, Qwen_gain_claim=False,
        source_sha256={str(p): digest(p) for p in [Path(__file__), root/'summary.json', root/'perception.jsonl',
            root/'execution.jsonl', root/'independent_audit_003/report.json',
            root/'independent_choice_audit_008/report.json', previous/'independent_choice_audit_008/report.json']},
        whole_project_sealed_test=False, goal_complete=False)
    dump(out/'report.json', value)
    print(json.dumps({k:v for k,v in value.items() if k not in
        ['source_sha256','planning_failure_frames','current_evidence_gap_frames']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
