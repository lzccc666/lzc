"""按原始指令和不同相机分别核对两条真实过去LiDAR转换链的重合。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    dest = OUTPUT/'data_manifest'/args.run_id
    dest.mkdir(exist_ok=False)
    shutil.copy2(__file__, dest/'auditor_source.py')
    new = OUTPUT/'data_manifest/reserved_causal_inputs_full_001'
    natural = OUTPUT/'data_manifest/natural_original_inputs_full_001'
    raw = OUTPUT/'data_manifest/natural_distribution_audit_001/evaluation_only_natural_commands.jsonl'
    original_report = json.loads((raw.parent/'report.json').read_text())
    if digest(raw) != original_report['records_sha256']:
        raise ValueError('original natural command population changed')
    commands = [json.loads(line) for line in raw.open()]
    expected_camera_tokens = {r['camera_sample_data']['token'] for r in commands
                             if r['camera_sample_data'] and r['camera_sample_data']['is_key_frame']}
    new_rows = {r['camera_token']: r for r in map(json.loads, (new/'sensor_provenance.jsonl').open())
                if r['channel'] == 'LIDAR_TOP'}
    natural_rows = {r['camera_token']: r for r in map(json.loads, (natural/'sensor_provenance.jsonl').open())
                    if r['channel'] == 'LIDAR_TOP'}
    shared = set(new_rows) & set(natural_rows)
    if shared != expected_camera_tokens:
        raise ValueError('overlap differs from original distinct keyframe camera tokens')
    checks = []
    for token in sorted(shared):
        a, b = new_rows[token], natural_rows[token]
        checks.append(dict(camera_token=token, same_source_member=a['member'] == b['member'],
            same_raw_source_sha256=a['member_sha256'] == b['member_sha256'],
            same_actual_projected_depth_sha256=digest(a['output']) == digest(b['output']),
            fingerprints_match_saved=digest(a['output']) == a['output_sha256'] == b['output_sha256']))
    passed = all(all(r[k] for k in ['same_source_member', 'same_raw_source_sha256',
        'same_actual_projected_depth_sha256', 'fingerprints_match_saved']) for r in checks)
    report = dict(passed=passed, overlapping_distinct_camera_tokens=len(shared),
        original_natural_commands_on_these_cameras=sum(r['camera_sample_data'] is not None
            and r['camera_sample_data']['token'] in shared for r in commands),
        original_natural_commands=len(commands), unique_natural_cameras=len(natural_rows),
        all_shared_past_inputs_exact=passed, independent_scenes_added=0, comparisons=checks,
        source_sha256=digest(Path(__file__)), raw_commands_sha256=digest(raw),
        new_complete_sha256=digest(new/'complete.json'), natural_complete_sha256=digest(natural/'complete.json'),
        explanation='47条关键帧原始指令对应44个不同相机帧；重复指令不增加相机或场景数',
        model_calls=0, new_test_opened=False, goal_complete=False)
    dump(dest/'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'comparisons'}, ensure_ascii=False))
    if not passed:
        raise AssertionError('shared past inputs differ; no correction or substitution performed')


if __name__ == '__main__':
    main()
