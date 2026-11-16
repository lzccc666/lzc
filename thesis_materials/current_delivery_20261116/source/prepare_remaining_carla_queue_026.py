"""修正重复使用预检目录的问题，冻结后续210回合，派生独占官方服务器等待器。"""
import copy
import datetime
import json
from pathlib import Path

from summarize_carla_expansion_013 import digest

R = Path(__file__).resolve().parents[1]


def derive(source, dest, changes):
    value = source.read_text()
    for before, after, count in changes:
        assert value.count(before) == count, before
        value = value.replace(before, after)
    if dest.exists():
        assert dest.read_text() == value
    else:
        with dest.open('x') as stream:
            stream.write(value)
    return dict(original=str(source), original_sha256=digest(source),
                derived=str(dest), derived_sha256=digest(dest),
                changes=[dict(before=a, after=b, occurrences=c) for a, b, c in changes])


def main():
    records = []
    records.append(derive(R/'source/preflight_carla_expansion_011.py',
        R/'source/preflight_carla_remaining_014.py',
        [('expansion_spawn_preflight_011', 'expansion_spawn_preflight_014', 1)]))
    records.append(derive(R/'source/run_official_expansion_019.py',
        R/'source/run_official_remaining_025.py', [
            ('official_expansion_execution_019', 'official_remaining_execution_025', 1),
            ('run_carla_expansion_batch_011.py', 'run_carla_remaining_comparison_024.py', 1),
            ('guarded_motion_audit_jobs_001/report.json', 'expansion_jobs_011_primary/report.json', 1),
            ('expansion_registration_011.json', 'remaining_comparison_registration_024.json', 1),
            ("completed['all_three_complete_and_guard_audited']", "completed['all_execution_and_audits_closed'] and completed['actual_launched_episodes']==150", 1),
            ('>10800', '>604800', 1),
            ('等待原作业超过3h', '等待原作业超过7天', 1),
            ('stdout_graphics_expansion_021.log', 'stdout_graphics_remaining_025.log', 2),
            ('carla_graphics_expansion_021.log', 'carla_graphics_remaining_025.log', 2),
            ("report=R/'json/expansion_jobs_011_primary/report.json'", "report=R/'json/remaining_comparison_jobs_024/report.json'", 1),
            ("actual['actual_launched_episodes']==150", "actual['actual_launched_episodes']==210", 1),
            ('registered expansion primary episodes and audits complete', 'registered remaining comparison episodes and audits complete', 1)]))
    bankfile = R/'carla_official/expansion_registration_011.json'
    historyfile = R/'carla_official/expansion_registration_012.json'
    bank = json.loads(bankfile.read_text())
    history = json.loads(historyfile.read_text())
    phases = {k: copy.deepcopy(bank['execution_phases'][k]) for k in ['methods', 'ablations']}
    phases['history_maintenance'] = copy.deepcopy(history['execution_phases']['history_maintenance'])
    sources = [R/'source'/name for name in [
        'run_carla_remaining_comparison_024.py', 'run_official_remaining_025.py',
        'preflight_carla_remaining_014.py', 'prepare_remaining_carla_queue_026.py',
        'audit_carla_history_execution_023.py']]
    sha = {**bank['source_sha256'], **history['source_sha256'], **{str(p): digest(p) for p in sources}}
    value = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 scope='保留主系统150和原方法/消融清单；追加完整时间生命周期消融30。所有实际回合共360，当前已完成数量另读真实执行报告。',
                 primary_required_closed_episodes=150,
                 execution_phases=phases, registered_additional_episodes=210,
                 total_primary_and_additional_episodes=360,
                 original_primary_results_not_reexecuted_or_overwritten=True,
                 same_queries_tasks_sensors_seeds_planner_controller_and_budgets=True,
                 registration_sha256={str(p): digest(p) for p in [bankfile, historyfile]},
                 source_sha256=sha, derivations=records,
                 immediate_GPU_or_world_calls=0, goal_complete=False)
    assert sum(map(len, phases.values())) == 210
    with (R/'carla_official/remaining_comparison_registration_024.json').open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(additional_registered=210, total_registered=360, actual_new_physical_episodes=0)))


if __name__ == '__main__':
    main()
