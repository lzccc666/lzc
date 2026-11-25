"""仅核查封存材料容量/来源，不运行方法、不把模板指令称为人工自然表达。"""
from collections import Counter
import csv
import json
from pathlib import Path

from summarize_carla_expansion_013 import digest, read

R = Path(__file__).resolve().parents[1]
OLD = R.parent


def main():
    out = R / 'annotation/sealed_inventory_readback_015'
    out.mkdir(exist_ok=False)
    inputs = {}
    root = OLD / 'annotation'
    collection = read(root / 'human_natural_collection_001/cards.json', inputs)
    natural = read(root / 'natural_command_review_001/cards.json', inputs)
    manifest = read(root / 'reserved_official_annotation_001/manifest.json', inputs)
    def load(name):
        path = root / 'reserved_official_annotation_001' / name
        inputs[str(path)] = digest(path)
        return [json.loads(line) for line in path.open()]
    main_rows = load('main_queries.jsonl')
    challenge = load('challenge_queries.jsonl')
    assert len({r['query_id'] for r in main_rows + challenge}) == len(main_rows + challenge)
    assert len({r['command_token'] for r in natural}) == len(natural)
    natural_count = Counter(r['scene'] for r in natural)
    main_count = Counter(r['scene'] for r in main_rows)
    challenge_count = Counter(r['scene'] for r in challenge)
    replay = Counter(r['scene'] for r in main_rows + challenge if r['online_replay_selected'])
    collection_scenes = {r['scene'] for r in collection}
    assert set(main_count) == set(natural_count) == collection_scenes
    assert len(collection_scenes) == manifest['scenes']
    assert sum(len(r['frames']) for r in collection) == manifest['frames']
    assert len(main_rows) == manifest['main_queries']
    assert len(challenge) == manifest['challenge_queries']
    assert sum(replay.values()) == manifest['replay_queries']
    rows = [dict(scene=r['scene'], log_token=r['log_token'],
                 frames=len(r['frames']), original_Talk2Car_commands=natural_count[r['scene']],
                 original_natural_gap_to_20=max(0, 20-natural_count[r['scene']]),
                 template_main_queries=main_count[r['scene']],
                 template_challenge_queries=challenge_count[r['scene']],
                 preselected_template_replay_queries=replay[r['scene']],
                 query_count_is_not_manual_review_or_label_validity=True)
            for r in collection]
    with (out / 'per_scene.csv').open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(scenes=len(collection), logs=len({r['log_token'] for r in collection}),
                  frames=sum(r['frames'] for r in rows), original_natural_commands=len(natural),
                  original_natural_scenes_already_at_least20=sum(n >= 20 for n in natural_count.values()),
                  original_natural_extra_needed_if_all30_require20=sum(r['original_natural_gap_to_20'] for r in rows),
                  template_main_queries=len(main_rows), template_challenge_queries=len(challenge),
                  template_all_scenes_at_least20=all(n >= 20 for n in main_count.values()),
                  preselected_template_replay_queries=sum(replay.values()),
                  all_scenes_have_at_least6_preselected_replay=all(n >= 6 for n in replay.values()),
                  official_label_statuses=dict(Counter(r['label_status'] for r in main_rows + challenge)),
                  count_does_not_certify_manual_labels_or_independence=True,
                  existing_600_templates_not_original_natural_queries=True,
                  original_natural_not_promoted_to_reviewed_by_attribute_cards=True,
                  task='封存材料数量与来源核对；不选择子集、不生成或修改标签、不读取方法输出。模板官方身份标签可独立审计，不能因未做人审就一概称其标签不存在；颜色/歧义人工审核和可达性另有验收。',
                  actual_new_method_inference=0, new_sealed_test_opened=False,
                  goal_complete=False, inputs_sha256=inputs, source_sha256=digest(__file__))
    with (out / 'report.json').open('x') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['inputs_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
