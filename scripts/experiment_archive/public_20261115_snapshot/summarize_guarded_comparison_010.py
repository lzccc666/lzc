"""三方法008真实闭环全部负结果同表；调用分母与任务分母分开。"""
import csv
import datetime
import json
from pathlib import Path

from conceptgraph.completion.stage9 import digest, dump

R=Path(__file__).resolve().parents[1]


def main():
    out=R/'carla_official/guarded_comparison_summary_010';out.mkdir(exist_ok=False)
    rows=[];inputs={}
    for mode in ['rule','base','adapter']:
        p=R/'carla_official/guarded_complex_navigation_008/complex_second_nearest20'/mode
        s=json.loads((p/'summary.json').read_text());a=json.loads((p/'independent_choice_audit_008/report.json').read_text())
        g=json.loads((p/'independent_motion_guard_audit_001/report.json').read_text())
        arrival=json.loads((p/'independent_audit_003/report.json').read_text())
        calls=a['actual_GPU_model_calls'];valid=a['valid_model_outputs']
        row=dict(mode=mode,distinct_task_denominator=1,seeds=1,physics_ticks=s['frames'],frontend_calls=s['perception_calls'],
            actual_GPU_calls=calls,strict_schema_valid_outputs=valid,schema_valid_fraction=valid/calls if calls else None,
            semantic_proposal_correct_ever=a['semantic_proposal_correct_ever'],
            authorized_first_proposal_correct_ever=a['authorized_first_proposal_correct_ever'],
            correct_lock_ever=a['correct_stable_lock_ever'],wrong_lock_ever=a['wrong_stable_lock_ever'],
            target_route_generated=a['target_route_generated'],strict_safe_arrival=arrival['strict_safe_arrival'],
            collision=bool(s['collisions']),lane_violation=s['lane_violation'],status=s['status'],
            wall_s=s['wall_s'],sim_s=s['sim_s'],maximum_acceleration_mps2=a['maximum_acceleration_mps2'],
            minimum_acceleration_mps2=a['minimum_acceleration_mps2'],
            nominal_guard_all_ticks_recomputed=g['every_tick_guard_executed'] and g['every_tick_obstacles_reconstructed'],
            nominal_guard_stop_requests=g['actual_stop_commands_verified'],backend_applied_frames=a['backend_application_frames'],
            independently_audited_source=str(p/'independent_choice_audit_008/report.json'))
        assert s['frames']==g['physical_ticks'];rows.append(row)
        for file in ['release.json','summary.json','independent_choice_audit_008/report.json','independent_audit_003/report.json','independent_motion_guard_audit_001/report.json']:
            inputs[str(p/file)]=digest(p/file)
    with (out/'methods.csv').open('x',newline='',encoding='utf-8-sig') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,
        distinct_configurations=1,distinct_tasks=1,seeds_per_method=1,actual_episodes=3,
        actual_physics_ticks=sum(r['physics_ticks'] for r in rows),actual_frontend_calls=sum(r['frontend_calls'] for r in rows),
        actual_GPU_calls=sum(r['actual_GPU_calls'] for r in rows),strict_arrivals=sum(r['strict_safe_arrival'] for r in rows),
        adapter_minus_base_schema_valid_percentage_points=100*(rows[2]['schema_valid_fraction']-rows[1]['schema_valid_fraction']),
        Qwen_positive_target_or_arrival_improvement_demonstrated=False,
        result='本开发任务：Adapter格式有效38/41，基础Qwen为0/45；两者均未选对并锁定目标，未实际到达。规则选对并锁定过目标，但最终也超时。不能用格式改善宣称目标任务改善。',
        statistics='一配置一任务一种子，缺乏独立任务重复；不把1519物理帧、761前端帧或86模型调用视为独立场景，也不计算总体置信区间。',
        zero_path_or_stationary_is_not_task_safety_success=True,
        source_sha256=inputs,evaluator_sha256=digest(__file__),whole_project_sealed_test=False,goal_complete=False)
    dump(out/'report.json',report);print(json.dumps({k:report[k] for k in ['actual_episodes','actual_physics_ticks','actual_frontend_calls','actual_GPU_calls','strict_arrivals','result']},ensure_ascii=False))


if __name__=='__main__':main()
