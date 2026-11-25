"""Independent evaluation-side reconstruction of all 1200 frozen labels."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from conceptgraph.completion import relation_nav_unified_20261125 as impl
from conceptgraph.nuscenes.expansion_benchmark import iter_records


def official_category(name):
    if name.startswith('human.pedestrian.'):
        return 'pedestrian'
    if name.startswith('vehicle.bus.'):
        return 'bus'
    return {'vehicle.car': 'car', 'vehicle.truck': 'truck', 'vehicle.bicycle': 'bicycle',
            'vehicle.motorcycle': 'motorcycle'}.get(name)


def motion(attributes):
    static = {'vehicle.parked', 'vehicle.stopped', 'pedestrian.standing', 'pedestrian.sitting_lying_down'}
    dynamic = {'vehicle.moving', 'pedestrian.moving'}
    if set(attributes) & static:
        return 'static'
    if set(attributes) & dynamic:
        return 'dynamic'
    return 'unknown'


def main():
    begun = time.monotonic()
    frames = {(r['scene'], r['frame_idx']): r for r in impl.rows(impl.DATA / 'method_frames.jsonl') if r['cohort'] == 'S2'}
    truth = {(r['scene'], r['frame_idx']): r['truth'] for r in impl.rows(impl.DATA / 'evaluation_only/frames.jsonl') if r['cohort'] == 'S2'}
    wanted = {o['annotation_token'] for group in truth.values() for o in group}
    attrs = {r['token']: r['name'] for r in impl.read(impl.RAW / 'attribute.json')}
    categories = {r['token']: r['name'] for r in impl.read(impl.RAW / 'category.json')}
    ann = {}
    for record in iter_records(impl.RAW / 'sample_annotation.json'):
        if record['token'] in wanted:
            ann[record['token']] = record
        if len(ann) == len(wanted):
            break
    wanted_instances = {r['instance_token'] for r in ann.values()}
    instances = {}
    for record in iter_records(impl.RAW / 'instance.json'):
        if record['token'] in wanted_instances:
            instances[record['token']] = record
        if len(instances) == len(wanted_instances):
            break
    if set(ann) != wanted or set(instances) != wanted_instances:
        raise AssertionError('official_annotation_or_instance_missing')
    source_mismatches = []; rebuilt_truth = {}
    for key, group in truth.items():
        rebuilt_truth[key] = []
        for obj in group:
            official = ann[obj['annotation_token']]
            attribute_names = sorted(attrs[t] for t in official['attribute_tokens'])
            category = official_category(categories[instances[official['instance_token']]['category_token']])
            if (not np.allclose(obj['center'], official['translation'], atol=1e-10, rtol=0)
                    or obj['instance_token'] != official['instance_token']
                    or obj['category'] != category
                    or sorted(obj['official_attributes']) != attribute_names
                    or obj['evaluation_motion'] != motion(attribute_names)):
                source_mismatches.append(dict(scene=key[0], frame_idx=key[1], annotation_token=obj['annotation_token']))
            rebuilt_truth[key].append(dict(id=official['instance_token'], category=category,
                center=np.asarray(official['translation']), motion=motion(attribute_names)))
    registered = {r['task_id']: r for r in impl.rows(impl.DATA / 'evaluation_only/task_labels.jsonl')}
    differences = []; statuses = Counter(); classifications = Counter(); checked = []
    names = {'小汽车': 'car', '卡车': 'truck', '公交车': 'bus', '行人': 'pedestrian', '自行车': 'bicycle', '摩托车': 'motorcycle'}
    relation_words = {'左侧': 'left_of', '右侧': 'right_of', '前方': 'in_front_of', '后方': 'behind'}
    for task in impl.rows(impl.DATA / 'method_tasks.jsonl'):
        key = task['scene'], task['frame_idx']; frame = frames[key]; group = rebuilt_truth[key]
        query = task['query_text']; relation = next((value for word, value in relation_words.items() if word in query), None)
        target_word = query.split('停着的', 1)[1].removesuffix('附近')
        category = names[target_word]
        target_class = [o for o in group if o['category'] == category]
        candidates = [o for o in target_class if o['motion'] == 'static']
        status = 'pending_motion_evidence' if any(o['motion'] == 'unknown' for o in target_class) else 'scorable'
        references = [o for o in group if o['category'] == 'truck'] if relation else []
        if relation:
            if len(references) > 1:
                status = 'pending_reference_ambiguity'
            elif not references:
                candidates = []
            else:
                forward = np.asarray(frame['camera_pose'])[:2, 2]
                forward /= np.linalg.norm(forward)
                left = np.array([-forward[1], forward[0]])
                axis = {'left_of': left, 'right_of': -left,
                        'in_front_of': forward, 'behind': -forward}[relation]
                anchor = references[0]
                # Fully independent direct pair geometry; no graph builder or
                # original label helper is called in this evaluation audit.
                candidates = [o for o in candidates if o['id'] != anchor['id']
                    and np.linalg.norm(o['center'][:2] - anchor['center'][:2]) <= 12.
                    and float((o['center'][:2] - anchor['center'][:2]) @ axis) > 1.]
        origin = np.asarray(frame['camera_pose'])[:2, 3]
        distances = {o['id']: float(np.linalg.norm(o['center'][:2] - origin)) for o in candidates}
        acceptable = sorted(token for token, distance in distances.items()
                            if abs(distance - min(distances.values())) <= 1e-9) if distances else []
        positive = bool(acceptable) if status == 'scorable' else None
        expected = registered[task['task_id']]
        if (expected['label_status'], expected['positive'], expected['acceptable_instances']) != (status, positive, acceptable):
            differences.append(dict(task_id=task['task_id'], actual=expected,
                reconstructed=dict(label_status=status, positive=positive, acceptable_instances=acceptable)))
        statuses[status] += 1
        classifications['positive' if positive else 'negative' if positive is False else 'pending'] += 1
        checked.append(dict(task_id=task['task_id'], label_status=status,
            positive=positive, acceptable_instances=acceptable))
    report = dict(passed=not source_mismatches and not differences, created_utc=datetime.now(timezone.utc).isoformat(),
        source_official_annotations_checked=len(ann), official_instances_checked=len(instances),
        source_mismatches=source_mismatches, task_labels_checked=len(checked), label_differences=differences,
        statuses=dict(statuses), classifications=dict(classifications),
        GT_in_method=False, evaluation_only=True, configuration_or_label_changes=False,
        view_scope='same frozen current annotated front3 visibility set; no re-render or extra visible GT added',
        source_sha256={str(Path(__file__)): impl.sha(__file__),
            str(impl.ROOT / 'scripts/relation_nav_label_audit_20261125.py'): impl.sha(impl.ROOT / 'scripts/relation_nav_label_audit_20261125.py')},
        elapsed_s=time.monotonic() - begun)
    impl.write(impl.REPORT / 'label_independent_audit_v1.json', report)
    if not report['passed']:
        raise AssertionError('independent_label_reconstruction_mismatch')
    print(json.dumps({k: report[k] for k in ['passed', 'task_labels_checked', 'statuses', 'classifications', 'elapsed_s']}))


if __name__ == '__main__':
    main()
