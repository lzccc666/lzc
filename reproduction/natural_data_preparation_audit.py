"""从原始表重算覆盖，并独立校验采集页全部RGB绑定；不提交人工记录。"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from PIL import Image

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, dump
from conceptgraph.nuscenes.expansion_benchmark import iter_records


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(4*1024*1024):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    started = time.monotonic()
    root = OUTPUT/'data_manifest/natural_data_preparation_audit_002'
    root.mkdir(exist_ok=False)
    coverage = OUTPUT/'data_manifest/natural_candidate_coverage_001'
    campaign = OUTPUT/'annotation/human_natural_collection_001'
    report = read(coverage/'report.json')
    release = read(coverage/'release.json')
    inputs = dict(release['source_sha256'])
    inputs[str(coverage/'report.json')] = sha(coverage/'report.json')
    inputs[str(coverage/'command_motion_metadata.jsonl')] = sha(coverage/'command_motion_metadata.jsonl')
    inputs[str(coverage/'scene_coverage.json')] = sha(coverage/'scene_coverage.json')
    inputs[str(campaign/'manifest.json')] = sha(campaign/'manifest.json')
    dump(root/'release.json', dict(source_sha256=sha(__file__), coverage_release_sha256=sha(coverage/'release.json'),
        campaign_manifest_sha256=sha(campaign/'manifest.json'), budget_wall_s=300, budget_bytes=5000000,
        CUDA_visible=os.environ.get('CUDA_VISIBLE_DEVICES'), shared_CPU=True,
        method_inference_executed=False, new_test_opened=False, goal_complete=False))
    events = OUTPUT/'interactive_simulation/realtime_three_model_full_001/resource_events.jsonl'

    def event(state):
        with events.open('a') as handle:
            handle.write(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                event='natural_data_preparation_audit_002', status=state, shared_CPU=True,
                CUDA_visible=os.environ.get('CUDA_VISIBLE_DEVICES')))+'\n')

    event('start')
    try:
        assert all(sha(path) == value for path, value in inputs.items())
        assert sha(coverage/'command_motion_metadata.jsonl') == report['command_motion_metadata_sha256']
        assert sha(coverage/'scene_coverage.json') == report['scene_coverage_sha256']
        assert sha(coverage/'release.json') == report['source_release_sha256']
        candidates = read(OUTPUT/'splits/geographic_candidate_audit_001.json')['candidates']
        by_scene = {s['scene_token']: s for s in candidates}
        commands = []
        for split in ['train', 'val', 'test']:
            commands.extend((c, split) for c in read(OUTPUT/f'data_manifest/talk2car_source_001/data/commands/{split}_commands.json')['commands']
                            if c['scene_token'] in by_scene)
        wanted = {c['box_token'] for c, _ in commands}
        metadata = PROJECT/'dataset/nuscenes_trainval_raw/v1.0-trainval'
        attributes = {a['token']: a['name'] for a in read(metadata/'attribute.json')}
        annotations = {}
        for a in iter_records(metadata/'sample_annotation.json'):
            if a['token'] in wanted:
                annotations[a['token']] = a
            assert time.monotonic()-started < 300
        recorded = {r['command_token']: r for r in map(json.loads, (coverage/'command_motion_metadata.jsonl').open())}
        assert set(recorded) == {c['command_token'] for c, _ in commands}
        raw, stationary = Counter(), Counter()
        motion_counts = Counter()
        for command, split in commands:
            annotation = annotations[command['box_token']]
            attrs = [attributes[a] for a in annotation['attribute_tokens']]
            states = {name.rsplit('.', 1)[-1] for name in attrs}
            motion = 'dynamic' if 'moving' in states else 'static' if states & {'parked', 'stopped', 'standing', 'sitting_lying_down'} else 'unknown'
            row = recorded[command['command_token']]
            assert (row['box_token'], row['upstream_split'], row['official_attributes'], row['official_motion']) == (command['box_token'], split, attrs, motion)
            assert row['scene_token'] == command['scene_token'] and row['annotation_found']
            raw[command['scene_token']] += 1
            stationary[command['scene_token']] += motion == 'static'
            motion_counts[motion] += 1
        assert dict(motion_counts) == report['official_motion_counts']
        for row in report['thresholds']:
            selected = {s['scene_token']: s for s in candidates if s['min_historical_camera_distance_m'] >= row['minimum_distance_m']}
            raw20 = {s: selected[s]['log_token'] for s in selected if raw[s] >= 20}
            static20 = {s: selected[s]['log_token'] for s in selected if stationary[s] >= 20}
            expected = dict(eligible_scenes=len(selected), raw_original_commands=sum(raw[s] for s in selected),
                officially_static_commands=sum(stationary[s] for s in selected), scenes_with_20_raw=len(raw20),
                logs_with_20_raw=len(set(raw20.values())), scenes_with_20_officially_static=len(static20),
                logs_with_20_officially_static=len(set(static20.values())))
            assert all(row[k] == value for k, value in expected.items())
        manifest = read(campaign/'manifest.json')
        assert sha(campaign/'cards.json') == manifest['cards_sha256']
        assert sha(campaign/'index.html') == manifest['page_sha256']
        assert sha(campaign/'generator_source.py') == manifest['source_sha256']
        assert (campaign/'manifest.sha256').read_text().strip() == sha(campaign/'manifest.json')
        selected = {s['scene']: s for s in read(OUTPUT/'splits/reserved_test_scenes_v2.json')['scenes']}
        cards = read(campaign/'cards.json')
        assert {s['scene'] for s in cards} == set(selected) and len(cards) == 30
        source = read(OUTPUT/'data_manifest/new_scene_sources_03_04.json')
        provenance = {r['camera_token']: r for r in map(json.loads, (OUTPUT/'data_manifest/reserved_causal_inputs_full_001/sensor_provenance.jsonl').open())
                      if r['channel'] == 'CAM_FRONT'}
        checked = 0
        for scene in cards:
            spec = selected[scene['scene']]
            assert (scene['scene_token'], scene['log_token']) == (spec['scene_token'], spec['log_token'])
            sequence = {s['sample_token']: s['sensors']['CAM_FRONT'] for s in source[scene['scene']]['sequence']}
            assert len(scene['frames']) == spec['frame_count']
            assert [f['frame_idx'] for f in scene['frames']] == list(range(spec['frame_count']))
            for frame in scene['frames']:
                camera = sequence[frame['sample_token']]
                assert (frame['camera_token'], frame['camera_timestamp']) == (camera['token'], camera['timestamp'])
                p = (OUTPUT/frame['rgb']).resolve()
                assert p.is_relative_to(OUTPUT.resolve())
                entry = provenance[camera['token']]
                value = sha(p)
                assert value == frame['rgb_sha256'] == entry['output_sha256'] == entry['member_sha256']
                assert p == Path(entry['output']).resolve()
                with Image.open(p) as image:
                    assert image.size == (camera['width'], camera['height']) == (1600, 900)
                    image.verify()
                inputs[str(p)] = value
                checked += 1
        assert checked == manifest['frames'] == 1196
        for path, value in manifest['alignment_sha256'].items():
            assert sha(OUTPUT/path) == value
            inputs[str(OUTPUT/path)] = value
        frozen = {}
        for rel in ['interactive_simulation/realtime_three_model_full_001', 'source_bundle/causal_ablations_full_001',
                    'source_bundle/causal_sensitivity_full_001', 'faults/realtime_fault_after_semantics_001',
                    'performance/runtime_profile_after_faults_001', 'vehicle_execution/independent_body_after_main_001']:
            sources = read(OUTPUT/rel/'release.json')['source_sha256']
            assert all(sha(Path(p) if Path(p).is_absolute() else PROJECT/p) == value for p, value in sources.items())
            frozen[rel] = len(sources)
        index_sha = sha(PROJECT/'.git/index')
        assert index_sha == 'bd7eef11d88c9d208692b152369d1dfbde6ba583193fd666c44e577968cc1338'
        dump(root/'files.json', inputs)
        dump(root/'report.json', dict(passed=True, original_command_metadata_checked=len(commands),
            thresholds_recomputed=len(report['thresholds']), actual_original_RGB_checked=checked,
            static_attributes_not_human_truth=True, source_file_checks=len(inputs),
            files_sha256=sha(root/'files.json'), frozen_sources_unchanged=frozen, root_git_index_sha256=index_sha,
            actual_new_human_submissions=0, new_qualified_labels=0, new_model_calls=0,
            method_inference_executed=False, new_test_opened=False, goal_complete=False,
            wall_s=time.monotonic()-started, created_utc=datetime.now(timezone.utc).isoformat()))
        assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) < 5000000
        event('complete')
        print(json.dumps(dict(passed=True, commands=len(commands), original_RGB=checked, model_calls=0)))
    except BaseException:
        event('failed_preserved')
        raise


if __name__ == '__main__':
    main()
