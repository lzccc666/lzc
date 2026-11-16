"""由冻结010精确派生历史维护消融；登记全部差异，不改旧批次或物理参数。"""
import copy
import datetime
import json
from pathlib import Path

from carla_history_lifecycle_017 import CONDITION, DEFINITION, MODULE_PARAMETERS
from summarize_carla_expansion_013 import digest

R = Path(__file__).resolve().parents[1]


def main():
    original = R / 'source/carla_expansion_episode_010.py'
    source = original.read_text()
    changes = [
        ('from carla_ablation_agent_001 import Agent as PreviousAgent,CONDITIONS',
         'from carla_history_lifecycle_017 import Agent as PreviousAgent,CONDITIONS', 1),
        ('expansion_registration_010.json', 'expansion_registration_012.json', 2),
        ('expansion_navigation_010', 'expansion_history_navigation_012', 1),
        ("condition=args.condition;assert args.seed in bank['seeds']",
         "condition=args.condition;assert condition in bank['runtime_allowed_conditions'];assert args.seed in bank['seeds']", 1),
        ("short_term_expiry_enabled=condition!='without_history_expiry'",
         "short_term_expiry_enabled=condition not in ('without_history_expiry','without_history_maintenance'),temporal_history_maintenance_enabled=condition!='without_history_maintenance'", 1)]
    for before, after, count in changes:
        assert source.count(before) == count, before
        source = source.replace(before, after)
    runtime = R / 'source/carla_history_episode_012.py'
    with runtime.open('x') as stream:
        stream.write(source)
    original_bank = R / 'carla_official/expansion_registration_010.json'
    bank = copy.deepcopy(json.loads(original_bank.read_text()))
    old_expiry = [j for j in bank['execution_phases']['ablations']
                  if j['condition'] == 'without_history_expiry']
    assert len(old_expiry) == 30
    jobs = [dict(j, condition=CONDITION) for j in old_expiry]
    references = [dict(j, condition='full_rule_geometry') for j in old_expiry]
    assert all(j in bank['execution_phases']['primary'] for j in references)
    bank.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                scope='完整时间生命周期维护子系统消融；开发测试，不是封存新测试',
                original_frozen_registration=str(original_bank),
                original_frozen_registration_sha256=digest(original_bank),
                original_episode_runtime=str(original), original_runtime_sha256=digest(original),
                derived_episode_runtime=str(runtime),
                runtime_allowed_conditions=[CONDITION],
                execution_phases={'history_maintenance': jobs},
                full_reference_jobs_in_primary=references,
                full_reference_location='carla_official/expansion_navigation_010',
                bulk_sensor_root=str(Path(bank['bulk_sensor_root']).parent/'runtime012_history'),
                logical_removed_module='temporal_object_lifecycle',
                actual_changed_mapper_parameters=MODULE_PARAMETERS,
                history_map_and_assignment_retained=True,
                control_planner_geometry_and_budget_unchanged=True,
                observations_without_motion_history_remain_unknown=True,
                module_removal_not_individual_internal_knob_causality=True,
                condition_definitions={CONDITION: DEFINITION},
                physical_episodes_executed_at_registration=0,
                whole_project_sealed_test=False, goal_complete=False)
    for path in [runtime, Path(__file__).resolve(), R/'source/carla_history_lifecycle_017.py']:
        bank['source_sha256'][str(path)] = digest(path)
    for filename, value in [('expansion_registration_012.json', bank),
                            ('history_runtime_derivation_012.json', dict(
                                source=str(original), source_sha256=digest(original),
                                derived=str(runtime), derived_sha256=digest(runtime),
                                exact_replacements=[dict(before=a, after=b, occurrences=c)
                                                    for a, b, c in changes],
                                source_changed_elsewhere=False,
                                goal_complete=False))]:
        with (R / 'carla_official' / filename).open('x') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(new_registered_history_episodes=len(jobs),
                         actual_physical_episodes=0, full_references=len(references)), ensure_ascii=False))


if __name__ == '__main__':
    main()
