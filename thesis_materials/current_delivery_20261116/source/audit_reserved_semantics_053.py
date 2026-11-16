"""只读官方表，独立复核预留标签；不导入标签生成器或任何待评测方法。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

PROJECT = Path('/root/autodl-tmp/lizichong')
sys.path.insert(0, str(PROJECT))
# 仅复用JSON流读取器，几何、类别、属性和答案计算均在本文件独立实现。
from conceptgraph.nuscenes.expansion_benchmark import iter_records  # noqa: E402

OLD = PROJECT / 'outputs/project_v2/thesis_completion_v1'
ROOT = OLD / 'public_experiment_completion_20261115'
META = PROJECT / 'dataset/nuscenes_trainval_raw/v1.0-trainval'
TEXT = dict(car='小汽车', truck='卡车', bus='公交车', motorcycle='摩托车',
            bicycle='自行车', pedestrian='行人')
STATIC = {'vehicle.parked', 'vehicle.stopped', 'pedestrian.standing',
          'pedestrian.sitting_lying_down', 'cycle.without_rider'}
DYNAMIC = {'vehicle.moving', 'pedestrian.moving'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def category(name):
    if name.startswith('human.pedestrian.'):
        return 'pedestrian'
    if name in {'vehicle.bus.rigid', 'vehicle.bus.bendy'}:
        return 'bus'
    return {'vehicle.' + k: k for k in TEXT if k not in {'bus', 'pedestrian'}}.get(name)


def transform(record):
    w, x, y, z = record['rotation']
    norm = w*w + x*x + y*y + z*z
    if abs(norm - 1) > 1e-6:
        raise ValueError('官方四元数不是单位四元数')
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    matrix[:3, 3] = record['translation']
    return matrix


def in_scope(obj, pose, intrinsic, width, height):
    point = np.linalg.solve(pose, np.r_[obj['center'], 1.0])[:3]
    if point[2] <= 1 or point[2] >= 65 or int(obj['visibility_token']) < 2:
        return False
    projected = intrinsic @ point
    u, v = projected[:2] / projected[2]
    return bool(0 <= u < width and 0 <= v < height)


def independent_answer(objects, semantics, pose):
    """按已声明的相机坐标方向/距离公差复核；不推断可达或颜色。"""
    targets = [o for o in objects if o['category'] in semantics['target_categories']
               and o['official_motion'] == 'static'
               and o['category'] not in semantics['excluded_categories']]
    if semantics['anchor_categories']:
        forward = np.asarray(pose[:2, 2], dtype=float)
        forward /= np.linalg.norm(forward)
        # 列向量[forward,left]把局部前左坐标变换到世界坐标。
        basis = np.column_stack((forward, [-forward[1], forward[0]]))
        anchors = [o for o in objects if o['category'] in semantics['anchor_categories']]
        retained = []
        for target in targets:
            for anchor in anchors:
                if target['instance_token'] == anchor['instance_token']:
                    continue
                delta = np.subtract(target['center'][:2], anchor['center'][:2])
                front, left = np.linalg.solve(basis, delta)
                relation = semantics['relation']
                satisfied = dict(left_of=left > 1, right_of=left < -1,
                                 in_front_of=front > 1, behind=front < -1)[relation]
                if satisfied and float(delta @ delta) <= 225:
                    retained.append(target)
                    break
        targets = retained
    if semantics['prefer_nearest'] and targets:
        distances = {o['instance_token']: float(np.linalg.norm(
            np.subtract(o['center'][:2], pose[:2, 3]))) for o in targets}
        targets = [o for o in targets
                   if distances[o['instance_token']] <= min(distances.values()) + .5]
    return sorted(o['instance_token'] for o in targets)


def declared_text(semantics, family):
    target = TEXT[semantics['target_categories'][0]]
    if family == 3:
        direction = dict(left_of='左侧', right_of='右侧', in_front_of='前方', behind='后方')
        return ('目标是在' + TEXT[semantics['anchor_categories'][0]]
                + direction[semantics['relation']] + '的静止' + target + '，请接近它')
    modifier = '最近的静止' if semantics['prefer_nearest'] else '静止'
    if family == 0:
        return '将车辆驶向观察范围内' + modifier + target + '旁的合法区域'
    if family == 1:
        return '观察范围内若有' + modifier + target + '，请安全接近'
    return ('本次接近观察范围内' + modifier + target + '，而不是'
            + TEXT[semantics['excluded_categories'][0]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='reserved_semantic_audit_053')
    args = parser.parse_args()
    dest = ROOT / 'annotation' / args.run_id
    dest.mkdir(exist_ok=False)
    started = time.monotonic()
    source = OLD / 'annotation/reserved_official_annotation_001'
    selection_path = OLD / 'splits/reserved_test_scenes_v2.json'
    card_path = OLD / 'annotation/human_natural_collection_001/cards.json'
    train_manifest = ROOT / 'qwen_adapter_fair_comparison/real_graph_dataset_003/manifest.json'
    tables = ['category', 'instance', 'attribute', 'scene', 'sample', 'sample_annotation',
              'sample_data', 'ego_pose', 'calibrated_sensor', 'sensor']
    protected = [source / n for n in ['manifest.json', 'evaluation_only_frames.jsonl',
                                     'main_queries.jsonl', 'challenge_queries.jsonl']]
    protected += [selection_path, card_path, train_manifest]
    protected += [META / (n + '.json') for n in tables]
    protocol = dict(UTC=datetime.now(timezone.utc).isoformat(), source_sha256=sha(Path(__file__)),
                    scope='仅官方标签/标定/输入图像/划分；不访问方法预测结果，不开启模型测试',
                    tolerance=dict(pose_absolute=2e-7, intrinsic_absolute=1e-9,
                                   timestamp_s=1e-6, relation_axis_m=1, relation_radius_m=15,
                                   nearest_tie_m=.5, min_camera_z_exclusive_m=1,
                                   max_camera_z_exclusive_m=65, min_visibility=2),
                    shared_code='仅JSON流读取器，无标签生成器/匹配器/模型导入',
                    protected_inputs_sha256={str(p): sha(p) for p in protected},
                    actual_new_GPU_calls=0, actual_new_CARLA_calls=0,
                    new_test_opened=False, goal_complete=False)
    write(dest / 'protocol.json', protocol)
    errors = []
    def check(ok, name, key):
        if not bool(ok):
            errors.append(dict(check=name, key=str(key)))
        return bool(ok)

    manifest = json.loads((source / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        check(sha(source / name) == expected, 'annotation_file_hash', name)
    for name, expected in manifest['source_tables_sha256'].items():
        check(sha(META / name) == expected, 'official_table_hash', name)
    check(sha(selection_path) == manifest['selection_sha256'], 'selection_hash', selection_path)
    frames = [json.loads(s) for s in (source / 'evaluation_only_frames.jsonl').open()]
    queries = [json.loads(s) for name in ['main_queries.jsonl', 'challenge_queries.jsonl']
               for s in (source / name).open()]
    cards = json.loads(card_path.read_text())
    card_map = {(c['scene'], f['frame_idx']): f for c in cards for f in c['frames']}
    selections = {s['scene']: s for s in json.loads(selection_path.read_text())['scenes']}
    frame_map = {(f['scene'], f['frame_idx']): f for f in frames}
    check(len(frame_map) == len(frames) == manifest['frames'], 'frame_population', len(frames))
    check(set(frame_map) == set(card_map), 'card_frame_population', len(card_map))
    check(len({q['query_id'] for q in queries}) == len(queries), 'unique_query_ids', len(queries))
    check(len(queries) == manifest['main_queries'] + manifest['challenge_queries'],
          'query_population', len(queries))
    needed_samples = {f['sample_token'] for f in frames}
    cat_names = {r['token']: r['name'] for r in iter_records(META / 'category.json')}
    instances = {r['token']: cat_names[r['category_token']] for r in iter_records(META / 'instance.json')}
    attrs = {r['token']: r['name'] for r in iter_records(META / 'attribute.json')}
    scenes = {r['token']: r for r in iter_records(META / 'scene.json')}
    samples = {r['token']: r for r in iter_records(META / 'sample.json') if r['token'] in needed_samples}
    official_objects = defaultdict(list)
    for row in iter_records(META / 'sample_annotation.json'):
        if row['sample_token'] not in needed_samples:
            continue
        raw_cat = instances[row['instance_token']]
        mapped = category(raw_cat)
        if mapped is None:
            continue
        attributes = [attrs[t] for t in row['attribute_tokens']]
        motion = ('dynamic' if DYNAMIC.intersection(attributes)
                  else 'static' if STATIC.intersection(attributes) else 'unknown')
        official_objects[row['sample_token']].append(dict(
            annotation_token=row['token'], instance_token=row['instance_token'], category=mapped,
            official_category=raw_cat, center=row['translation'], size_wlh=row['size'],
            rotation_wxyz=row['rotation'], visibility_token=row['visibility_token'],
            num_lidar_pts=row['num_lidar_pts'], official_attributes=attributes, official_motion=motion))
    print('official annotations loaded', flush=True)
    needed_cameras = {c['camera_token'] for c in card_map.values()}
    cameras = {r['token']: r for r in iter_records(META / 'sample_data.json')
               if r['token'] in needed_cameras}
    needed_poses = {r['ego_pose_token'] for r in cameras.values()}
    ego = {r['token']: r for r in iter_records(META / 'ego_pose.json') if r['token'] in needed_poses}
    calibration = {r['token']: r for r in iter_records(META / 'calibrated_sensor.json')}
    sensors = {r['token']: r for r in iter_records(META / 'sensor.json')}
    rgb_fingerprints = {}
    validated = {}
    object_counts = Counter()
    time_offsets = []
    with (dest / 'frame_checks.jsonl').open('x') as handle:
        for key, frame in frame_map.items():
            before = len(errors)
            card = card_map[key]
            camera = cameras[card['camera_token']]
            sample = samples[frame['sample_token']]
            scene = scenes[sample['scene_token']]
            selection = selections[frame['scene']]
            cal = calibration[camera['calibrated_sensor_token']]
            pose = transform(ego[camera['ego_pose_token']]) @ transform(cal)
            intrinsic = np.asarray(cal['camera_intrinsic'])
            check(scene['name'] == frame['scene'] and scene['log_token'] == frame['log_token']
                  == selection['log_token'], 'official_scene_log', key)
            check(camera['sample_token'] == card['sample_token'] == frame['sample_token'],
                  'official_sample_camera_binding', key)
            check(camera['is_key_frame'] and sensors[cal['sensor_token']]['channel'] == 'CAM_FRONT',
                  'official_front_key_camera', key)
            check(camera['timestamp'] == card['camera_timestamp']
                  and abs(camera['timestamp']/1e6-frame['camera_timestamp_s']) < 1e-6,
                  'camera_timestamp', key)
            check(np.allclose(pose, frame['camera_pose'], atol=2e-7, rtol=0), 'official_camera_pose', key)
            check(np.allclose(intrinsic, frame['intrinsic'], atol=1e-9, rtol=0), 'official_intrinsic', key)
            check((camera['width'], camera['height']) == (frame['width'], frame['height']),
                  'official_image_dimensions', key)
            for rgb in {Path(frame['rgb']), OLD / card['rgb']}:
                rgb_fingerprints[str(rgb)] = sha(rgb)
                check(rgb_fingerprints[str(rgb)] == card['rgb_sha256'], 'rgb_card_sha256', rgb)
                with Image.open(rgb) as image:
                    check(image.size == (frame['width'], frame['height']), 'decoded_image_dimensions', rgb)
            objects = [o for o in official_objects[frame['sample_token']]
                       if in_scope(o, pose, intrinsic, frame['width'], frame['height'])]
            expected = {o['annotation_token']: o for o in objects}
            actual = {o['annotation_token']: o for o in frame['objects']}
            check(len(actual) == len(frame['objects']) and actual.keys() == expected.keys(),
                  'scoped_official_object_population', key)
            for token in expected.keys() & actual.keys():
                check(all(actual[token][k] == v for k, v in expected[token].items()),
                      'official_object_fields', token)
                check(actual[token].get('human_verified') is False
                      and actual[token].get('color') == actual[token].get('body_type') == 'unknown',
                      'unknown_attributes_not_fabricated', token)
            validated[key] = (objects, pose)
            object_counts.update(o['official_motion'] for o in objects)
            offset = (camera['timestamp'] - sample['timestamp']) / 1e6
            time_offsets.append(offset)
            handle.write(json.dumps(dict(scene=key[0], frame_idx=key[1], sample_token=frame['sample_token'],
                official_scoped_objects=len(objects), camera_minus_annotation_sample_s=offset,
                passed=len(errors) == before), ensure_ascii=False) + '\n')
    for scene_name in selections:
        ordered = sorted((f for f in frames if f['scene'] == scene_name), key=lambda f: f['frame_idx'])
        check([f['frame_idx'] for f in ordered] == list(range(len(ordered))), 'continuous_indices', scene_name)
        for a, b in zip(ordered, ordered[1:]):
            check(samples[a['sample_token']]['next'] == b['sample_token']
                  and a['camera_timestamp_s'] < b['camera_timestamp_s'], 'official_continuous_sample_order', scene_name)
    cohorts = defaultdict(Counter)
    with (dest / 'query_checks.jsonl').open('x') as handle:
        for query in queries:
            before = len(errors)
            key = (query['scene'], query['frame_idx'])
            frame = frame_map[key]
            objects, pose = validated[key]
            semantics = query['semantic_truth']
            check(query['sample_token'] == frame['sample_token']
                  and abs(query['timestamp_s'] - frame['camera_timestamp_s']) < 1e-6,
                  'query_time_binding', query['query_id'])
            check(semantics['requires_static'] is True and not semantics['prefer_front']
                  and not any(semantics[k] for k in ['target_colors', 'anchor_colors', 'excluded_colors']),
                  'supported_semantic_scope', query['query_id'])
            expected = independent_answer(objects, semantics, pose)
            check(expected == sorted(query['accepted_instance_tokens'])
                  and len(set(query['accepted_instance_tokens'])) == len(query['accepted_instance_tokens']),
                  'independent_accepted_instances', query['query_id'])
            check(query['no_matching_target'] == (not expected), 'no_match_label', query['query_id'])
            family = int(query['expression_family'].rsplit('_', 1)[1])
            check(declared_text(semantics, family) == query['query_text'], 'text_semantics_binding', query['query_id'])
            check(query['human_verified'] is False and query['color_ground_truth'] == 'unknown'
                  and query['reachable_label'] is None, 'label_evidence_boundary', query['query_id'])
            cohorts[query['cohort']].update(queries=1, positive=bool(expected), no_match=not expected,
                relation=bool(semantics['relation']), nearest=semantics['prefer_nearest'],
                replay_selected=query['online_replay_selected'])
            unknown = sum(o['category'] in semantics['target_categories']
                          and o['official_motion'] == 'unknown' for o in objects)
            handle.write(json.dumps(dict(query_id=query['query_id'], passed=len(errors) == before,
                expected_instance_tokens=expected, same_class_unknown_motion_objects=unknown,
                no_match_scope='仅官方属性已知的静止对象，不代表现实不存在或不可达'), ensure_ascii=False) + '\n')
    sealed_logs = {f['log_token'] for f in frames}
    split_logs = json.loads(train_manifest.read_text())['split_logs']
    log_overlaps = {k: sorted(sealed_logs & set(v)) for k, v in split_logs.items()}
    check(not any(log_overlaps.values()), 'new_adapter_train_val_dev_log_isolation', log_overlaps)
    # 与本轮真实图训练/验证/开发文本逐字比较；不把“不同文字”当作无语义相似。
    query_texts = {q['query_text'] for q in queries}
    text_overlaps = {}
    for split in split_logs:
        path = train_manifest.parent / (split + '.jsonl')
        protected.append(path)
        protocol['protected_inputs_sha256'][str(path)] = sha(path)
        texts = {json.loads(line)['query_text'] for line in path.open()}
        text_overlaps[split] = sorted(texts & query_texts)
    # 后读训练文本只作隔离检查，不改变任何预留指令、标签或评测方案。
    write(dest / 'supplemental_input_hashes.json', {
        str(p): protocol['protected_inputs_sha256'][str(p)] for p in protected if p.parent == train_manifest.parent})
    preserved = all(sha(p) == protocol['protected_inputs_sha256'][str(p)] for p in protected)
    check(preserved, 'protected_inputs_unchanged', len(protected))
    write(dest / 'rgb_hashes.json', rgb_fingerprints)
    write(dest / 'errors.json', errors)
    report = dict(UTC=datetime.now(timezone.utc).isoformat(), passed=not errors, error_count=len(errors),
        error_kinds=dict(Counter(e['check'] for e in errors)), scenes=len(selections), logs=len(sealed_logs),
        frames=len(frames), queries=len(queries), cohort_counts={k: dict(v) for k, v in cohorts.items()},
        scoped_object_motion_counts=dict(object_counts), rgb_paths_verified=len(rgb_fingerprints),
        camera_sample_time_offset_s=dict(min=min(time_offsets), max=max(time_offsets),
                                       max_abs=max(map(abs, time_offsets))),
        adapter_log_overlap=log_overlaps, adapter_exact_text_overlap=text_overlaps,
        protected_inputs_rehashed_unchanged=preserved, source_sha256=sha(Path(__file__)),
        files_sha256={p.name: sha(p) for p in dest.iterdir() if p.is_file()},
        elapsed_s=time.monotonic()-started, actual_new_GPU_calls=0, actual_new_CARLA_calls=0,
        method_evaluation_executed=False, new_test_opened=False, human_review_added=False,
        limitations=['仅模板语义和官方标注一致性；不是人工自然指令验证',
                    '官方可见度与中心入画不能保证整个物体清楚可辨',
                    '官方标注为sample时刻，未对camera时间差插值；时间差另列',
                    '无匹配仅指当前相机范围且静止属性已知的官方对象',
                    '不验证颜色细属性、遮挡下视觉歧义、实际可达或实际到达',
                    '日志隔离仅针对本轮real_graph_dataset_003；不替代全部历史泄漏审计',
                    '文本无逐字重合不代表无语义相似'], goal_complete=False)
    write(dest / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
