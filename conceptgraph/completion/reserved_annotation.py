"""准备预留测试的隔离官方标签与声明语法指令；不运行任何待评测方法。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump
from conceptgraph.nuscenes.benchmark_ground_truth import (
    CATEGORY_TEXT, accepted_targets, official_category,
)
from conceptgraph.nuscenes.expansion_benchmark import iter_records

STATIC = {'vehicle.parked', 'vehicle.stopped', 'pedestrian.standing',
          'pedestrian.sitting_lying_down', 'cycle.without_rider'}
DYNAMIC = {'vehicle.moving', 'pedestrian.moving'}
RELATIONS = dict(left_of='左侧', right_of='右侧', in_front_of='前方', behind='后方')
PATTERNS = (
    '将车辆驶向观察范围内{modifier}{target}旁的合法区域',
    '观察范围内若有{modifier}{target}，请安全接近',
    '本次接近观察范围内{modifier}{target}，而不是{exclude}',
    '目标是在{anchor}{relation}的静止{target}，请接近它',
)


def scoped_objects(objects, pose, intrinsic, width, height):
    """与算法检测无关：官方中心在相机内、深度与可见性满足预先定义范围。"""
    visible = []
    for obj in objects:
        xyz = pose[:3, :3].T @ (np.asarray(obj['center']) - pose[:3, 3])
        if not 1 < xyz[2] < 65 or int(obj['visibility_token']) < 2:
            continue
        uvz = intrinsic @ xyz
        if 0 <= uvz[0] / uvz[2] < width and 0 <= uvz[1] / uvz[2] < height:
            visible.append(obj)
    return sorted(visible, key=lambda o: o['instance_token'])


def derive_query(frame, pose, *, category, nearest, family, query_id,
                 anchor=None, relation=None, excluded=None, cohort='distribution_preserving'):
    static = [o for o in frame['objects'] if o['official_motion'] == 'static']
    # 锚点允许是同一时刻的动态对象；静止约束只施加到目标。
    if anchor:
        all_tokens = accepted_targets(frame['objects'], category, pose[:2, 3],
            anchor_category=anchor, relation=relation, forward_xy=pose[:2, 2])
        candidates = [o for o in static if o['instance_token'] in all_tokens]
    else:
        candidates = static
    tokens = accepted_targets(candidates, category, pose[:2, 3], nearest=nearest)
    modifier = '最近的静止' if nearest else '静止'
    target = CATEGORY_TEXT[category]
    if anchor:
        text = PATTERNS[3].format(anchor=CATEGORY_TEXT[anchor],
            relation=RELATIONS[relation], target=target)
        family = 3
    else:
        text = PATTERNS[family].format(modifier=modifier, target=target,
            exclude=CATEGORY_TEXT[excluded] if excluded else '')
    return dict(query_id=query_id, scene=frame['scene'], frame_idx=frame['frame_idx'],
        sample_token=frame['sample_token'], timestamp_s=frame['camera_timestamp_s'],
        query_text=text, cohort=cohort, expression_family=f'reserved_v3_{family}',
        semantic_truth=dict(target_categories=[category], target_colors=[],
            anchor_categories=[anchor] if anchor else [], anchor_colors=[], relation=relation,
            prefer_nearest=nearest, prefer_front=False,
            excluded_categories=[excluded] if excluded else [], excluded_colors=[],
            requires_static=True),
        accepted_instance_tokens=tokens, no_matching_target=not tokens,
        target_existence_scope='当前指定相机观测、中心可见、1<z<65m、visibility>=2且官方静止属性；不代表整个道路',
        static_labels_unknown_excluded_from_target_truth=True,
        query_source='事前声明语法构造；不是人工自然语言指令',
        label_status='official_instance_attribute_and_independent_geometry',
        human_verified=False, color_ground_truth='unknown',
        reachable_label=None, reachability_status='未做独立道路拓扑和整车可达性核验',
        online_replay_selected=False)


def make_scene_queries(frames, poses, seed=17):
    """主集保留采样场景；按官方实例抽类别，不按方法结果选题。挑战集另列。"""
    rng = random.Random(seed + int(frames[0]['scene'].split('-')[-1]))
    categories = list(CATEGORY_TEXT)
    main = []
    anchors = np.linspace(0, (len(frames) - 1) // 2, 10, dtype=int)
    for j in range(20):
        frame_idx = int(anchors[j // 2])
        frame = frames[frame_idx]
        static = [o for o in frame['objects'] if o['official_motion'] == 'static']
        category = rng.choice(static)['category'] if j % 2 == 0 and static else categories[(j // 2) % len(categories)]
        family = j % 3
        excluded = categories[(categories.index(category) + 1) % len(categories)] if family == 2 else None
        row = derive_query(frame, poses[frame_idx], category=category, nearest=j % 2 == 1,
            family=family, excluded=excluded, query_id=f"{frame['scene']}-main-{j:02d}")
        row['online_replay_selected'] = j in {0, 2, 4, 6, 8, 10}
        main.append(row)
    # 每场景挑战候选全集按声明优先序产生；不使用模型或前端输出。
    challenge = []
    seen = set()
    for frame_idx in sorted(set(int(i) for i in anchors)):
        frame = frames[frame_idx]
        static = [o for o in frame['objects'] if o['official_motion'] == 'static']
        counts = Counter(o['category'] for o in static)
        specifications = []
        for cat in categories:
            if counts[cat] >= 2:
                specifications.append(dict(category=cat, nearest=True, family=1,
                    challenge_type='same_category_ambiguity'))
            if not counts[cat]:
                specifications.append(dict(category=cat, nearest=False, family=0,
                    challenge_type='no_static_match_in_observed_scope'))
            for anchor in categories:
                for relation in RELATIONS:
                    if any(o['category'] == cat for o in static):
                        specifications.append(dict(category=cat, nearest=False, family=3,
                            anchor=anchor, relation=relation, challenge_type='positive_relation'))
        for specification in specifications:
            kind = specification.pop('challenge_type')
            row = derive_query(frame, poses[frame_idx], **specification,
                query_id=f"{frame['scene']}-challenge-{len(challenge):02d}", cohort='challenge')
            if kind == 'positive_relation' and not row['accepted_instance_tokens']:
                continue
            key = (row['query_text'], frame_idx)
            if key in seen:
                continue
            row['challenge_type'] = kind
            seen.add(key)
            challenge.append(row)
    # 每类最多4条，保留实际存在类型；不以填齐缺失关系或类别为目的造标签。
    selected = []
    types = ['positive_relation', 'same_category_ambiguity', 'no_static_match_in_observed_scope']
    for kind in types:
        selected.extend([q for q in challenge if q['challenge_type'] == kind][:4])
    for j, row in enumerate(selected):
        row['query_id'] = f"{frames[0]['scene']}-challenge-{j:02d}"
        row['online_replay_selected'] = j == 0 and row['challenge_type'] == 'positive_relation'
    return main, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--inputs', default='reserved_inputs_002')
    parser.add_argument('--selection', default='reserved_test_scenes_v2.json')
    args = parser.parse_args()
    dest = OUTPUT / 'annotation' / args.run_id
    dest.mkdir(parents=True, exist_ok=False)
    selection_path = OUTPUT / 'splits' / args.selection
    selection = json.loads(selection_path.read_text())
    sensor_root = OUTPUT / 'data_manifest' / args.inputs
    if not (sensor_root / 'complete.json').exists():
        raise ValueError('传感器准备尚未完成')
    meta = PROJECT / 'dataset/nuscenes_trainval_raw/v1.0-trainval'
    categories = {r['token']: r['name'] for r in iter_records(meta / 'category.json')}
    instances = {r['token']: categories[r['category_token']] for r in iter_records(meta / 'instance.json')}
    attributes = {r['token']: r['name'] for r in iter_records(meta / 'attribute.json')}
    inputs, samples = {}, {}
    for scene in selection['scenes']:
        source = sensor_root / 'converted' / scene['scene']
        alignment = json.loads((source / 'frame_alignment.json').read_text())
        poses = np.loadtxt(source / 'poses.txt').reshape(-1, 4, 4)
        if len(alignment) != len(poses):
            raise ValueError('标定和位姿长度不一致')
        inputs[scene['scene']] = (source, alignment, poses)
        samples.update({row['sample_token']: [] for row in alignment})
    for annotation in iter_records(meta / 'sample_annotation.json'):
        if annotation['sample_token'] not in samples:
            continue
        category = official_category(instances[annotation['instance_token']])
        if category is None:
            continue
        names = [attributes[token] for token in annotation['attribute_tokens']]
        motion = 'dynamic' if set(names) & DYNAMIC else 'static' if set(names) & STATIC else 'unknown'
        samples[annotation['sample_token']].append(dict(annotation_token=annotation['token'],
            instance_token=annotation['instance_token'], category=category,
            official_category=instances[annotation['instance_token']], center=annotation['translation'],
            size_wlh=annotation['size'], rotation_wxyz=annotation['rotation'],
            visibility_token=annotation['visibility_token'], num_lidar_pts=annotation['num_lidar_pts'],
            official_attributes=names, official_motion=motion, color='unknown',
            body_type='unknown', human_verified=False))
    all_frames, main_queries, challenge_queries, per_scene = [], [], [], []
    for scene in selection['scenes']:
        name = scene['scene']
        source, alignment, poses = inputs[name]
        frames = []
        for i, (row, pose) in enumerate(zip(alignment, poses)):
            with Image.open(source / 'color' / f'{i:06d}.jpg') as rgb:
                width, height = rgb.size
            objects = scoped_objects(samples[row['sample_token']], pose,
                np.asarray(row['intrinsic']), width, height)
            frames.append(dict(scene=name, log_token=scene['log_token'], frame_idx=i,
                sample_token=row['sample_token'], camera_timestamp_s=row['camera_timestamp'] / 1e6,
                objects=objects, rgb=str(source / 'color' / f'{i:06d}.jpg'),
                camera_pose=pose.tolist(), intrinsic=row['intrinsic'], width=width, height=height))
        main_q, challenge = make_scene_queries(frames, poses)
        all_frames.extend(frames)
        main_queries.extend(main_q)
        challenge_queries.extend(challenge)
        per_scene.append(dict(scene=name, log_token=scene['log_token'], frames=len(frames),
            main_queries=len(main_q), challenge_queries=len(challenge),
            replay_queries=sum(q['online_replay_selected'] for q in main_q + challenge),
            positive_queries=sum(bool(q['accepted_instance_tokens']) for q in main_q),
            query_categories=dict(Counter(q['semantic_truth']['target_categories'][0] for q in main_q)),
            challenge_types=dict(Counter(q['challenge_type'] for q in challenge))))
    for name, rows in [('evaluation_only_frames.jsonl', all_frames),
                       ('main_queries.jsonl', main_queries), ('challenge_queries.jsonl', challenge_queries)]:
        with (dest / name).open('x') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    training_texts = set()
    for path in (OUTPUT / 'semantic_dataset/structured_supervision_002').glob('*.jsonl'):
        for row in map(json.loads, path.open()):
            if row['task'] == 'query_parse':
                training_texts.add(row['query_text'])
    overlap = sorted(training_texts & {q['query_text'] for q in main_queries + challenge_queries})
    if overlap:
        raise ValueError('发现训练/验证与新指令完全重合；保留当前run并重新设计新版本')
    dump(dest / 'manifest.json', dict(version='reserved-official-annotation-v1',
        scenes=len(per_scene), logs=len({r['log_token'] for r in selection['scenes']}),
        frames=len(all_frames), main_queries=len(main_queries), challenge_queries=len(challenge_queries),
        replay_queries=sum(q['online_replay_selected'] for q in main_queries + challenge_queries),
        per_scene=per_scene, expression_patterns=PATTERNS,
        train_validation_exact_expression_overlap=overlap,
        files={p.name: digest(p) for p in dest.glob('*.jsonl')},
        source_sha256=digest(Path(__file__)), selection_sha256=digest(selection_path),
        source_tables_sha256={p.name: digest(p) for p in [meta / f'{name}.json' for name in
            ['category', 'instance', 'attribute', 'sample_annotation']]},
        inference_executed=False, methods_frozen=False, new_test_opened=False,
        human_review_count=0, goal_complete=False,
        limitations=['场景保留预留采样分布，但指令为构造语法，不能声称人工自然指令分布',
            '每场景20条不代表20个独立场景；统计必须按场景/log聚类',
            '挑战集只在真实存在相应官方关系/歧义时设置，不代表颜色已核验',
            '静态取官方属性；unknown不当静态，停着也不等于道路可达',
            '新测试模型结果尚未访问；方法/提示/指标等冻结后方可开启评测']))
    print(json.dumps(dict(scenes=len(per_scene), frames=len(all_frames), main_queries=len(main_queries),
        challenge_queries=len(challenge_queries), replay_queries=sum(q['online_replay_selected']
            for q in main_queries + challenge_queries), new_test_opened=False)), flush=True)


if __name__ == '__main__':
    main()
