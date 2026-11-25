"""完整 R1 的冻结缓存漏失归因；不执行前端，不改推理或评价协议。"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import pickle
import shutil
import time

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.r1_failure_attribution import frame_evidence, summarize


R1 = PROJECT / 'outputs/results/nuscenes/validation/iterations/recall_refinement_R1_full_20261120_v2'


def scene_evidence(task):
    scene, destination = task
    name = scene['scene']
    source = OUTPUT / 'mapping_evaluation/causal_mapping_full_001' / name / 'frames.jsonl'
    frozen = R1 / name / 'frames.jsonl.gz'
    result, frames, seen = [], 0, set()
    with source.open() as a, gzip.open(frozen, 'rt') as b:
        aa, bb = list(map(json.loads, a)), list(map(json.loads, b))
    if len(aa) != len(bb):
        raise ValueError('帧数变化')
    for raw, saved in zip(aa, bb):
        if shutil.disk_usage(destination).free < 20 * 2**30:
            raise OSError('磁盘保护线')
        if raw['frame_idx'] != saved['frame_idx']:
            raise ValueError('帧次序变化')
        path = Path(saved['input']['detections'])
        if digest(path) != saved['input']['detections_sha256']:
            raise ValueError('检测缓存指纹变化')
        with gzip.open(path, 'rb') as handle:
            det = pickle.load(handle)
        raw['_path'], saved['_path'] = str(source), str(frozen)
        result.extend(frame_evidence(raw, saved, det, log_token=scene['log_token'], first_seen=seen))
        frames += 1
    target = Path(destination) / name
    target.mkdir()
    with gzip.open(target / 'gt_evidence.jsonl.gz', 'wt') as handle:
        for row in result:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    return dict(scene=name, frames=frames, truth=len(result), matched=sum(r['matched'] for r in result),
        input_sha256={str(source): digest(source), str(frozen): digest(frozen)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    start = time.monotonic()
    release = json.loads((R1 / 'release.json').read_text())
    dest = experiment_directory('nuscenes', 'validation', args.run_id, create=True)
    sources = [Path(__file__).resolve(), PROJECT / 'conceptgraph/slam/r1_failure_attribution.py',
        PROJECT / 'conceptgraph/completion/mapping_recall_diagnosis.py']
    for path in sources:
        target = dest / 'source_snapshot' / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    dump(dest / 'release.json', dict(mode='R1_frozen_failure_diagnostic', baseline=str(R1),
        original_release_sha256=digest(R1 / 'release.json'), original_report_sha256=digest(R1 / 'report.json'),
        sources_sha256={str(p): digest(p) for p in sources}, scenes=release['scenes'], seed=17,
        diagnostic_box_iou=.25, main_threshold_m=3., max_new_bytes=2 * 2**30, minimum_free_bytes=20 * 2**30,
        new_frontend_calls=0, parameter_changes=False, GT_in_inference=False, test_accessed=False,
        limitations=['框投影重叠是支持代理，不是pixel GT', '缺少实例mask GT，不能独立量化分割质量',
                    '阶段计数不等于纯因果责任；稀疏1—4点分支保留']))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        checks = list(pool.map(scene_evidence, [(s, str(dest)) for s in release['scenes']]))
    rows = []
    for scene in release['scenes']:
        with gzip.open(dest / scene['scene'] / 'gt_evidence.jsonl.gz', 'rt') as handle:
            rows.extend(map(json.loads, handle))
    summary = summarize(rows)
    if (len(checks), sum(x['frames'] for x in checks), len(rows), sum(x['matched'] for x in rows)) != (38, 1508, 5723, 2769):
        raise AssertionError('完整 R1 分母或 TP 未复现')
    dump(dest / 'preliminary_report.json', dict(summary, complete_scope=True, scenes=38, frames=1508,
        truth=5723, tp=2769, fn=2954, stage='cached_2d_and_real_R1_depth_audit',
        internal_state_attribution_pending=True, scene_checks=checks, wall_s=time.monotonic() - start,
        output_sha256={str(p.relative_to(dest)): digest(p) for p in dest.glob('scene-*/*.gz')}))
    print(json.dumps(dict(run=str(dest), counts=summary['counts'], cross=summary['cross'], wall_s=time.monotonic()-start), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
