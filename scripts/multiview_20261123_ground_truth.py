"""隔离官方GT：同一实例跨相机只计一次；不向融合传入标注。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conceptgraph.completion.stage9 import PROJECT, digest, dump
from conceptgraph.nuscenes.benchmark_ground_truth import official_category
from conceptgraph.nuscenes.expansion_benchmark import iter_records
from conceptgraph.slam.multiview_20261123_fusion import in_view


P = 'multiview_20261123_'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    groups = [(scene, group) for scene in manifest['scenes'] for group in scene['groups']]
    samples = {g['sample_token']: [] for _, g in groups}
    meta = PROJECT/'dataset/raw/nuscenes_trainval/v1.0-trainval'
    inputs = {str(args.manifest): digest(args.manifest)}
    for name in ['category', 'instance', 'sample_annotation']:
        inputs[str(meta/(name+'.json'))] = digest(meta/(name+'.json'))
    categories = {r['token']: r['name'] for r in iter_records(meta/'category.json')}
    instances = {r['token']: categories[r['category_token']] for r in iter_records(meta/'instance.json')}
    for row in iter_records(meta/'sample_annotation.json'):
        if row['sample_token'] not in samples:
            continue
        category = official_category(instances[row['instance_token']])
        if category and int(row['visibility_token']) >= 2:
            samples[row['sample_token']].append(dict(annotation_token=row['token'],
                instance_token=row['instance_token'], category=category, center=row['translation'],
                size_wlh=row['size'], rotation_wxyz=row['rotation'], visibility_token=row['visibility_token'],
                num_lidar_pts=row['num_lidar_pts']))
    target = args.manifest.parent/(P+'evaluation_ground_truth.jsonl')
    counts = dict(groups=len(groups), union_GT=0, front_GT=0, side_only_GT=0)
    with target.open('x') as stream:
        for scene, group in groups:
            objects = []
            for obj in samples[group['sample_token']]:
                cameras = [c for c, v in group['views'].items() if in_view(obj['center'], v)]
                if cameras:
                    objects.append(dict(obj, visible_cameras=cameras))
            objects.sort(key=lambda o: o['instance_token'])
            assert len({o['instance_token'] for o in objects}) == len(objects)
            front = [o for o in objects if 'CAM_FRONT' in o['visible_cameras']]
            counts['union_GT'] += len(objects)
            counts['front_GT'] += len(front)
            counts['side_only_GT'] += len(objects)-len(front)
            stream.write(json.dumps(dict(scene=scene['scene'], frame_idx=group['frame_idx'],
                sample_token=group['sample_token'], official_sample_timestamp_s=group['official_sample_timestamp_s'],
                objects=objects), allow_nan=False)+'\n')
    dump(args.manifest.parent/(P+'ground_truth_release.json'), dict(counts=counts,
        input_sha256=inputs, output_sha256={str(target): digest(target)}, source_sha256=digest(__file__),
        criteria='official six classes; visibility>=2; center FOV and 1<camera_z<65 in at least one camera; instance deduplicated',
        time='official sample-time annotation; not motion-compensated to camera/group reference',
        GT_in_inference=False))
    print(counts)


if __name__ == '__main__':
    main()
