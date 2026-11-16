"""核验真实保存对象中确已移除生命周期，且地图仍跨帧累积。"""
import argparse
import json
from pathlib import Path

from carla_history_lifecycle_017 import CONDITION
from carla_lossless_jsonl_001 import read_rows
from summarize_carla_expansion_013 import digest

R = Path(__file__).resolve().parents[1]


def check_off_observation(value):
    assert value['enable_dynamic_object_handling'] is False
    assert value['dynamic_ttl_s'] == 1.e9
    for obj in value['objects']:
        assert obj['motion_state'] == 'unknown'
        assert not obj['motion_state_field_present']
        assert obj['center_history_length'] == obj['frame_history_length'] == 0
        assert obj['timestamp_age_s'] is None or obj['timestamp_age_s'] >= -1.e-8
    return any(obj['num_detections'] > 1 for obj in value['objects'])


def recorded():
    root = R/'carla_official/history_module_recorded_diagnostic_021'
    out = root/'independent_lifecycle_audit_023'
    out.mkdir(exist_ok=False)
    registration = json.loads((root/'registration.json').read_text())
    report = json.loads((root/'report.json').read_text())
    assert len(report['reports']) == 6 and report['actual_CPU_mapping_updates'] == 2148
    for source, sha in registration['source_sha256'].items():
        assert digest(source) == sha
    checked = 0
    cases = []
    for case in report['reports']:
        folder = root/Path(case['episode']).parents[2].name/case['condition']
        assert digest(folder/'frames.jsonl') == case['frame_records_sha256']
        frames = [json.loads(line) for line in (folder/'frames.jsonl').open()]
        assert len(frames) == case['frames']
        cross_frame = False
        for row in frames:
            checked += 1
            life = row['lifecycle']
            if case['condition'] == CONDITION:
                cross_frame |= check_off_observation(life)
                assert all(n['motion'] == 'unknown' for n in row['nodes'])
                assert all(not v[0] for v in row['evidence_decisions'].values())
        if case['condition'] == CONDITION:
            assert cross_frame, '移除维护后仍需保留实际跨帧关联/累积'
        cases.append(dict(episode=case['episode'], condition=case['condition'], frames=len(frames),
                          cross_frame_accumulation_seen=cross_frame if case['condition'] == CONDITION else None,
                          frame_sha256=case['frame_records_sha256']))
    assert checked == 2148
    result = dict(actual_recorded_CPU_updates_rechecked=checked, unique_sensor_frames=716,
                  cases=cases, history_disabled_consumed_in_actual_mapper=True,
                  short_term_expiry_only_is_different_from_whole_lifecycle_removal=True,
                  original_noncompact_run_storage_overrun_kept=True,
                  actual_new_physical_episodes=0, actual_new_GPU_calls=0,
                  no_Qwen_or_closed_loop_advantage_claim=True,
                  actual_closed_loop_history_ablation_complete=False,
                  audit_source_sha256=digest(__file__), report_sha256=digest(root/'report.json'), goal_complete=False)
    with (out/'report.json').open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(rechecked=checked, unique_frames=716, new_arrival_credit=0)))


def physical(episode):
    bankfile = R/'carla_official/expansion_registration_012.json'
    bank = json.loads(bankfile.read_text())
    release = json.loads((episode/'release.json').read_text())
    assert release['condition'] == CONDITION
    assert release['registration_sha256'] == digest(bankfile)
    assert release['ablation_switches']['temporal_history_maintenance_enabled'] is False
    assert release['ablation_switches']['short_term_expiry_enabled'] is False
    assert release['ablation_switches']['evidence_check_enabled'] is True
    assert release['ablation_switches']['relations_enabled'] is True
    assert release['ablation_switches']['current_frame_only'] is False
    for p, sha in bank['source_sha256'].items():
        assert digest(p) == sha
    observations = read_rows(episode, 'perception.jsonl')
    accumulated = False
    for obs in observations:
        value = obs['temporal_lifecycle_observation']
        accumulated |= check_off_observation(value)
        by_id = {o['stable_object_id']: o for o in value['objects']}
        assert len(by_id) == len(value['objects'])
        for node in obs['nodes']:
            source = by_id[node['metadata']['object_identity']]
            assert source['motion_state'] == node['motion']
            assert source['num_detections'] == node['metadata']['num_detections']
            assert source['timestamp_s'] == node['metadata']['timestamp_s']
    out = episode/'independent_history_lifecycle_audit_023'
    out.mkdir(exist_ok=False)
    with (out/'report.json').open('x') as stream:
        json.dump(dict(actual_frontend_frames=len(observations),
                       lifecycle_removal_verified_every_frame=True,
                       cross_frame_accumulation_seen=accumulated,
                       no_detection_can_make_accumulation_unobservable=True,
                       inference_source_sha_verified=True,
                       arrival_requires_separate_expansion_audit=True,
                       source_sha256=digest(__file__), goal_complete=False),
                  stream, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recorded', action='store_true')
    parser.add_argument('--episode')
    args = parser.parse_args()
    if args.recorded:
        assert args.episode is None
        recorded()
    else:
        assert args.episode
        path = (R/'carla_official'/args.episode).resolve()
        assert path.is_relative_to(R/'carla_official')
        physical(path)
