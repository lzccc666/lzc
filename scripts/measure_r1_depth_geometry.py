"""用现存深度的非零点核对GT投影框/官方3D框/原mask支持，不评价分割精度。"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
from PIL import Image

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.r1_failure_attribution import inside_official_box


BASE = PROJECT / 'outputs/results/nuscenes/validation/iterations'


def process_scene(task):
    scene, destination, evidence_run = task
    name = scene['scene']
    source = OUTPUT / 'mapping_evaluation/causal_mapping_full_001' / name / 'frames.jsonl'
    path = BASE / evidence_run / name / 'gt_evidence.jsonl.gz'
    with gzip.open(path, 'rt') as stream:
        evidence = {(r['frame_idx'], r['truth_index']): r for r in map(json.loads, stream)}
    result = []
    for row in map(json.loads, source.open()):
        if shutil.disk_usage(destination).free < 20 * 2**30:
            raise RuntimeError('磁盘保护线')
        if not row['evaluation_only_official']:
            continue
        first = evidence[(row['frame_idx'], 0)]
        path = Path(first['source_records']['detections'])
        if digest(path) != first['source_records']['detections_sha256']:
            raise ValueError('检测缓存改变')
        with gzip.open(path, 'rb') as stream:
            det = pickle.load(stream)
        depth = np.asarray(Image.open(row['depth']), dtype=float) / 1000
        v, u = np.nonzero((depth > 1) & (depth < 65) & np.isfinite(depth))
        z = depth[v, u]
        intrinsic, pose = np.asarray(row['intrinsic']), np.asarray(row['camera_pose'])
        camera = np.column_stack(((u - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (v - intrinsic[1, 2]) * z / intrinsic[1, 1], z))
        points = camera @ pose[:3, :3].T + pose[:3, 3]
        for ti, obj in enumerate(row['evaluation_only_official']):
            e = evidence[(row['frame_idx'], ti)]
            box = e['gt_projected_box']
            in_rect = ((u >= box[0]) & (u <= box[2]) & (v >= box[1]) & (v <= box[3])) if box else np.zeros(len(u), dtype=bool)
            exact = inside_official_box(points, obj, 0.)
            margin = inside_official_box(points, obj, .5)
            candidates = []
            for c in e['candidates']:
                idx = c['raw_index']
                mask = det['mask'][idx]
                if mask.shape != depth.shape:
                    import cv2
                    mask = cv2.resize(mask.astype(np.uint8), depth.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
                selected = mask[v, u]
                candidates.append(dict(raw_index=idx,
                    raw_mask_valid_points=int(selected.sum()),
                    raw_mask_points_inside_gt_3d_box=int(np.sum(selected & exact)),
                    raw_mask_points_inside_gt_3d_box_margin05=int(np.sum(selected & margin)),
                    raw_mask_points_in_gt_projected_rect=int(np.sum(selected & in_rect))))
            result.append(dict(scene=name, frame_idx=row['frame_idx'], truth_index=ti,
                matched=e['matched'], depth_bin=e['depth_bin'], iou025_detection_count=e['iou025_detection_count'],
                gt_projected_rect_valid_depth_points=int(in_rect.sum()),
                current_points_inside_official_3d_box=int(exact.sum()),
                current_points_inside_official_3d_box_margin05=int(margin.sum()),
                candidates=candidates))
    target = Path(destination) / name
    target.mkdir()
    with gzip.open(target / 'depth_geometry.jsonl.gz', 'wt') as stream:
        for row in result:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    return len(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--evidence-run', default='r1_failure_attribution_20261123_v1')
    args = parser.parse_args()
    start = time.monotonic()
    release = json.loads((BASE / args.evidence_run / 'release.json').read_text())
    dest = experiment_directory('nuscenes', 'validation', args.run_id, create=True)
    sources = [Path(__file__).resolve(), PROJECT / 'conceptgraph/slam/r1_failure_attribution.py']
    for path in sources:
        target = dest / 'source_snapshot' / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    dump(dest / 'release.json', dict(mode='depth_geometry_support_proxy', evidence_run=args.evidence_run,
        sources_sha256={str(p): digest(p) for p in sources}, diagnostic_3d_box_margins_m=[0., .5],
        main_threshold_m=3., GT_in_inference=False, new_frontend_calls=0, test_accessed=False,
        limitations=['3D框内点不是实例像素真值；动态时差会混淆', 'GT投影框内点可能属于背景或遮挡物']))
    with ProcessPoolExecutor(max_workers=4) as pool:
        n = sum(pool.map(process_scene, [(s, str(dest), args.evidence_run) for s in release['scenes']]))
    if n != 5723:
        raise AssertionError('范围不完整')
    cross = Counter()
    for path in dest.glob('scene-*/*.gz'):
        for row in map(json.loads, gzip.open(path, 'rt')):
            if row['matched'] or not row['iou025_detection_count'] or row['depth_bin'] == '5_plus':
                continue
            prefix = row['depth_bin']
            for key in ['gt_projected_rect_valid_depth_points', 'current_points_inside_official_3d_box', 'current_points_inside_official_3d_box_margin05']:
                cross[prefix + ':' + key + ':' + ('0' if row[key] == 0 else '1_to_4' if row[key] < 5 else '5_plus')] += 1
            best = max((c['raw_mask_points_inside_gt_3d_box_margin05'] for c in row['candidates']), default=0)
            cross[prefix + ':raw_mask_gt_margin05:' + ('0' if best == 0 else '1_to_4' if best < 5 else '5_plus')] += 1
    dump(dest / 'report.json', dict(complete=True, truth=n, cross=dict(cross), wall_s=time.monotonic()-start,
        output_sha256={str(p.relative_to(dest)): digest(p) for p in dest.glob('scene-*/*.gz')}))
    print(json.dumps(dict(truth=n, cross=cross, wall_s=time.monotonic()-start), ensure_ascii=False))


if __name__ == '__main__':
    main()
