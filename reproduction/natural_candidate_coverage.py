"""固定语料与既有候选的元数据覆盖审计；不运行封存方法、不改选场。"""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump
from conceptgraph.nuscenes.expansion_benchmark import iter_records


def main():
    started = time.monotonic()
    root = OUTPUT/'data_manifest/natural_candidate_coverage_001'
    root.mkdir(exist_ok=False)
    source = OUTPUT/'data_manifest/talk2car_source_001'
    candidates_path = OUTPUT/'splits/geographic_candidate_audit_001.json'
    selection_path = OUTPUT/'splits/reserved_test_scenes_v2.json'
    metadata = PROJECT/'dataset/nuscenes_trainval_raw/v1.0-trainval'
    files = [candidates_path, selection_path, source/'source_manifest.json',
             metadata/'sample_annotation.json', metadata/'attribute.json', Path(__file__).resolve()]
    manifest = json.loads((source/'source_manifest.json').read_text())
    for row in manifest['downloaded']:
        path = source/row['path']
        assert digest(path) == row['sha256']
        files.append(path)
    dump(root/'release.json', dict(source_sha256={str(p): digest(p) for p in files},
         purpose='只审计固定原始自然语料的可用规模；静态属性不是人工语义或长期运动真值',
         thresholds_m=[0, 20, 50, 100], retained_policy_m=50,
         budget_wall_s=300, budget_bytes=15000000, CUDA_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
         shared_CPU=True, method_inference_executed=False, new_test_opened=False,
         selection_modified=False, goal_complete=False))
    events = OUTPUT/'interactive_simulation/realtime_three_model_full_001/resource_events.jsonl'

    def event(state):
        with events.open('a') as handle:
            handle.write(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                         event='natural_candidate_coverage_001', status=state, shared_CPU=True,
                         CUDA_visible=os.environ.get('CUDA_VISIBLE_DEVICES')))+'\n')

    event('start')
    try:
        candidates = json.loads(candidates_path.read_text())['candidates']
        by_token = {c['scene_token']: c for c in candidates}
        commands = []
        for split in ['train', 'val', 'test']:
            data = json.loads((source/f'data/commands/{split}_commands.json').read_text())
            commands.extend(dict(c, upstream_split=split) for c in data['commands']
                            if c['scene_token'] in by_token)
        assert len({c['command_token'] for c in commands}) == len(commands)
        wanted = {c['box_token'] for c in commands}
        attributes = {a['token']: a['name'] for a in iter_records(metadata/'attribute.json')}
        annotations = {}
        for row in iter_records(metadata/'sample_annotation.json'):
            if row['token'] in wanted:
                annotations[row['token']] = row
            assert time.monotonic()-started < 300
        rows = []
        for command in commands:
            annotation = annotations.get(command['box_token'])
            attrs = [attributes[a] for a in annotation['attribute_tokens']] if annotation else []
            motion = ('dynamic' if any(a.endswith('.moving') for a in attrs) else
                      'static' if any(a.endswith(('.parked', '.stopped', '.standing', '.sitting_lying_down'))
                                      for a in attrs) else 'unknown')
            candidate = by_token[command['scene_token']]
            rows.append(dict(scene=candidate['scene'], scene_token=command['scene_token'],
                        command_token=command['command_token'], upstream_split=command['upstream_split'],
                        box_token=command['box_token'], annotation_found=annotation is not None,
                        official_attributes=attrs, official_motion=motion))
        with (root/'command_motion_metadata.jsonl').open('x') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        counts = Counter(row['scene_token'] for row in rows)
        static = Counter(row['scene_token'] for row in rows if row['official_motion'] == 'static')
        scene_rows = [dict(scene=c['scene'], scene_token=c['scene_token'], log_token=c['log_token'],
                      historical_distance_m=c['min_historical_camera_distance_m'],
                      raw_original_commands=counts[c['scene_token']],
                      officially_static_commands=static[c['scene_token']]) for c in candidates]
        dump(root/'scene_coverage.json', scene_rows)
        summaries = []
        for distance in [0, 20, 50, 100]:
            eligible = [s for s in scene_rows if s['historical_distance_m'] >= distance]
            raw20 = [s for s in eligible if s['raw_original_commands'] >= 20]
            static20 = [s for s in eligible if s['officially_static_commands'] >= 20]
            summaries.append(dict(minimum_distance_m=distance, eligible_scenes=len(eligible),
                raw_original_commands=sum(s['raw_original_commands'] for s in eligible),
                officially_static_commands=sum(s['officially_static_commands'] for s in eligible),
                scenes_with_20_raw=len(raw20), logs_with_20_raw=len({s['log_token'] for s in raw20}),
                scenes_with_20_officially_static=len(static20),
                logs_with_20_officially_static=len({s['log_token'] for s in static20})))
        selected = {s['scene_token'] for s in json.loads(selection_path.read_text())['scenes']}
        selected_rows = [r for r in scene_rows if r['scene_token'] in selected]
        retained = next(s for s in summaries if s['minimum_distance_m'] == 50)
        assert retained['scenes_with_20_raw'] == 15 and retained['logs_with_20_raw'] == 6
        report = dict(created_utc=datetime.now(timezone.utc).isoformat(), candidates=len(candidates),
            original_commands=len(rows), annotations_found=sum(r['annotation_found'] for r in rows),
            official_motion_counts=dict(Counter(r['official_motion'] for r in rows)), thresholds=summaries,
            selected_scenes=len(selected_rows), selected_original_commands=sum(s['raw_original_commands'] for s in selected_rows),
            selected_scenes_with_20_raw=sum(s['raw_original_commands'] >= 20 for s in selected_rows),
            selected_scenes_with_20_officially_static=sum(s['officially_static_commands'] >= 20 for s in selected_rows),
            source_release_sha256=digest(root/'release.json'),
            scene_coverage_sha256=digest(root/'scene_coverage.json'),
            command_motion_metadata_sha256=digest(root/'command_motion_metadata.jsonl'),
            selection_only_can_satisfy_20_scenes_10_logs=False,
            limitations=['原文规模只是上限，尚未审核静止目标接近语义、歧义、颜色、可达性与表达家族',
                        '官方单帧运动属性不是持续静止的人工证据',
                        '不同距离阈值只作诊断；既定50米隔离与30场景选场保持不变'],
            new_human_annotations=0, model_calls=0, method_inference_executed=False,
            new_test_opened=False, goal_complete=False, wall_s=time.monotonic()-started)
        dump(root/'report.json', report)
        lines = ['# 自然语料候选覆盖审计', '',
                 '审计固定Talk2Car原文与既有146个历史隔离候选，仅用于数据准备。未运行新测试方法。', '',
                 '|历史轨迹距离/m|可用场景|原文数|≥20条原文的场景/log|≥20条官方静态属性的场景/log|',
                 '|---:|---:|---:|---:|---:|']
        for s in summaries:
            lines.append(f"|{s['minimum_distance_m']}|{s['eligible_scenes']}|{s['raw_original_commands']}|"
                         f"{s['scenes_with_20_raw']}/{s['logs_with_20_raw']}|"
                         f"{s['scenes_with_20_officially_static']}/{s['logs_with_20_officially_static']}|")
        lines += ['', '保持50米历史轨迹隔离时，仅15场景、6个log拥有至少20条原文。仅重新选场无法满足至少20场景的要求。',
                  '原文上限不等于有效静止目标指令数量；需要真实新表达采集与标签审核。模板、自动改写和重复帧不能补作独立自然指令。',
                  '', '当前30场景保持封存。人工新表达另存，合格数量仍须语义、目标身份及其他标签核验后才能确定。']
        (root/'report.md').write_text('\n'.join(lines)+'\n')
        assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) < 15000000
        assert shutil.disk_usage(OUTPUT).free > 6*1024**3
        event('complete')
        print(json.dumps(report, ensure_ascii=False))
    except BaseException:
        event('failed_preserved')
        raise


if __name__ == '__main__':
    main()
