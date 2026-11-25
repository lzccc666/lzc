"""回读真实独立CPU环境完整导航与已结束依赖审计，不重复启动实验。"""
import datetime
import json
from pathlib import Path

import numpy as np
from conceptgraph.completion.stage9 import digest, dump

R = Path(__file__).resolve().parents[1]


def main():
    dest = R/'reproduction/clean_cpu_navigation_readback_003'
    dest.mkdir(exist_ok=False)
    root = R/'carla_official/clean_cpu_navigation_002/car20/full_rule_geometry'
    batch = R/'json/clean_cpu_navigation_jobs_002/report.json'
    launcher = R/'json/official_clean_cpu_navigation_execution_014/report.json'
    names = ['summary.json', 'release.json', 'fresh_environment_modules.json',
        'independent_audit_003/report.json', 'independent_choice_audit_005/report.json']
    values = {n: json.loads((root/n).read_text()) for n in names}
    summary, audit = values['summary.json'], values['independent_audit_003/report.json']
    environment = values['fresh_environment_modules.json']
    batch_data = json.loads(batch.read_text())
    assert batch_data['subprocesses_and_audits_complete']
    assert batch_data['dependencies_unchanged_during_execution']
    assert batch_data['actual_started_navigation_episodes'] == 1
    assert audit['strict_safe_arrival'] and audit['geometric_identity_stop_arrival']
    assert summary['correct_object_selected'] and summary['correctly_locked']
    assert summary['actually_arrived'] and not summary['collisions']
    assert not summary['Qwen_evaluated'] and not environment['actual_torch_CUDA']
    assert len(environment['modules']) == 19
    allowed = [Path(environment['prefix']).resolve(),
               Path(environment['explicit_new_dependency_overlay']).resolve()]
    for module in environment['modules']:
        assert any(Path(module['path']).resolve().is_relative_to(p) for p in allowed)
    assert Path(environment['original_core_module']).resolve().is_relative_to(
        Path(environment['isolated_checkout']).resolve())
    for path, sha in audit['inputs_sha256'].items():
        assert digest(path) == sha
    rows = list(map(json.loads, (root/'execution.jsonl').open()))
    fronts = list(map(json.loads, (root/'perception.jsonl').open()))
    assert len(rows) == summary['frames'] == 561
    assert len(fronts) == summary['perception_calls'] == 281
    assert all(r['snapshot_frame'] == r['frame']
        and all(f == r['frame'] for f in r['sensor_frames'].values()) for r in rows)
    original = R/'carla_official/prius_controller_navigation_005/car20/full_rule_geometry/execution.jsonl'
    old = list(map(json.loads, original.open()))
    keys = ['x', 'y', 'yaw_rad', 'speed_mps', 'steering_rad', 'time_s']
    # 同任务同种子重跑，只作为复现；不得把重复开发样本计入独立泛化分母。
    difference = None
    if len(old) == len(rows):
        difference = {k: float(np.max(np.abs(np.array([r['state'][k] for r in rows])
            - np.array([r['state'][k] for r in old])))) for k in keys}
    report = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_new_official_navigation_episodes=1, actual_physics_ticks=len(rows),
        actual_CPU_frontend_and_mapping_calls=len(fronts), isolated_third_party_modules=19,
        third_party_module_paths_checked=True, original_core_from_isolated_snapshot=True,
        dependencies_unchanged_during_execution=True, actual_CPU_Qwen_generations=0,
        actual_GPU_inference_calls=0, official_server_graphics_GPU=True,
        route_generated=summary['route_generated'], correct_first_proposal=True,
        correctly_locked=True, strict_safe_arrival=True,
        maximum_contiguous_hold_s=audit['maximum_contiguous_hold_s'],
        actuation_limits=audit['actuation_limits'], maximum_acceleration_mps2=audit['maximum_acceleration_mps2'],
        minimum_acceleration_mps2=audit['minimum_acceleration_mps2'],
        same_seed_original_development_trace_max_absolute_differences=difference,
        performance=dict(sim_s=summary['sim_s'], actual_episode_wall_s=summary['wall_s'],
            real_time=False, limitation='CPU前端平均慢于仿真步进；证明可复现链路，不证明实时'),
        scope='Town10HD_Opt/晴天/空交通/seed17/清晰规则单任务，重复任务不是独立泛化样本',
        source_sha256={str(p): digest(p) for p in [Path(__file__), batch, launcher, original,
            *[root/n for n in names]]}, whole_project_sealed_test=False, goal_complete=False)
    dump(dest/'report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
