"""R系列完整核验：相同输入重放，逐字段核对预测、身份和内部证据。"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path
import pickle
import resource
import shutil
import time
import traceback

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump, predict
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.evidence_mapper import confirmed_observation
from conceptgraph.slam.dense_refinement import RefinedEvidenceMapper
from conceptgraph.slam.streamlined_mapping import MappingFrame


def verify_source_snapshots(run, release):
    for path, sha in release['sources_sha256'].items():
        frozen = run / 'source_snapshot' / Path(path).relative_to(PROJECT)
        if digest(frozen) != sha:
            raise ValueError('冻结源码损坏：' + path)
        # 原runner仅负责首次运行，核验由本脚本逐帧执行；其冻结实体仍须完整。
        if Path(path).name not in {'run_mapping_refinement.py', 'run_uncertainty_refinement.py'} and digest(Path(path)) != sha:
            raise ValueError('推理/评价源码已变化，不能原样核验：' + path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--wall-budget-s', type=float, default=900)
    args = parser.parse_args()
    begun = time.monotonic()
    original = PROJECT / 'outputs/results/nuscenes/validation/iterations' / args.input_run
    release = json.loads((original / 'release.json').read_text())
    report = json.loads((original / 'report.json').read_text())
    if not report['full_validation_scope'] or report['frames'] != 1508:
        raise ValueError('仅核验完整配置')
    verify_source_snapshots(original, release)
    dest = experiment_directory('nuscenes', 'validation', args.run_id, create=True)
    snapshot = dest / 'source_snapshot/scripts'
    snapshot.mkdir(parents=True)
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    dump(dest / 'release.json', dict(mode='verification_only', input_run=str(original),
        original_release_sha256=digest(original / 'release.json'),
        verifier_sha256=digest(Path(__file__)), original_report_sha256=digest(original / 'report.json'),
        new_candidate=False, parameter_changes=False, new_statistical_test=False,
        wall_budget_s=args.wall_budget_s, full_scope_required=True))
    frames, audit_rows, scenes = 0, 0, 0
    method = report['condition']
    try:
        for scene in release['scenes']:
            name = scene['scene']
            cfg = OmegaConf.create(release['config'])
            settings = release['experimental_settings']
            if method == 'R1':
                mapper = RefinedEvidenceMapper(cfg, CLASSES, **settings)
            elif method in {'R1_unknown_ttl', 'R3_unknown_ttl'}:
                from conceptgraph.slam.unknown_expiry_refinement import create_unknown_expiry_mapper
                mapper = create_unknown_expiry_mapper(cfg, CLASSES, **settings)
            else:
                from conceptgraph.slam.uncertainty_refinement import create_refinement_mapper
                mapper = create_refinement_mapper(cfg, CLASSES, **settings)
            source_path = OUTPUT / 'mapping_evaluation/causal_mapping_full_001' / name / 'frames.jsonl'
            case = dest / name
            case.mkdir()
            with source_path.open() as source, gzip.open(original / name / 'frames.jsonl.gz', 'rt') as reference, gzip.open(case / 'association_confirmation.jsonl.gz', 'wt') as out:
                rows, references = list(map(json.loads, source)), list(map(json.loads, reference))
                if len(rows) != len(references):
                    raise AssertionError('核验范围不同')
                for row, expected in zip(rows, references):
                    if time.monotonic() - begun > args.wall_budget_s or shutil.disk_usage(dest).free < 20 * 1024**3:
                        raise RuntimeError('达到核验预算或磁盘保护线')
                    idx = row['frame_idx']
                    if idx != expected['frame_idx']:
                        raise AssertionError('帧错位')
                    path = Path(expected['input']['detections'])
                    if digest(path) != expected['input']['detections_sha256']:
                        raise ValueError('实际检测缓存变化')
                    with gzip.open(path, 'rb') as stream:
                        gobs = pickle.load(stream)
                    before = {obj['stable_object_id'] for obj in mapper.objects}
                    frame = MappingFrame(idx, np.asarray(Image.open(row['rgb']).convert('RGB')),
                        np.asarray(Image.open(row['depth']), dtype=float) / 1000,
                        np.asarray(row['intrinsic']), np.asarray(row['camera_pose']), copy.deepcopy(gobs),
                        row['rgb'], timestamp_s=row['camera_timestamp_s'])
                    np.random.seed(release['seed'] + idx)
                    mapper.update(frame)
                    output = predict(mapper.confirmed_objects(), CLASSES)
                    output = mapper.position_predictions(output, frame)
                    # JSON精确往返后逐字段一致，不用仅匹配TP掩盖中心或身份变化。
                    if json.loads(json.dumps(output, allow_nan=False)) != expected['predictions'][method]:
                        raise AssertionError(f'同配置预测不确定：{name}/{idx}')
                    associations, confirmations = [], []
                    for obj in mapper.objects:
                        node_id = obj['stable_object_id']
                        if obj['image_idx'][-1] == idx:
                            associations.append(dict(node_id=node_id, filtered_mask_index=int(obj['mask_idx'][-1]),
                                reason='matched_existing' if node_id in before else 'created',
                                observations=int(obj['num_detections'])))
                        confirmed = confirmed_observation(obj,
                            strong_confidence=float(cfg.strong_confidence),
                            minimum_observations=int(cfg.weak_confirmation_observations))
                        confirmations.append(dict(node_id=node_id, confirmed=confirmed,
                            reason='strong_or_multiframe_depth_supported' if confirmed else 'insufficient_confirmation',
                            observations=int(obj['num_detections']),
                            distinct_frames=len(set(obj['image_idx']))))
                    after = {obj['stable_object_id'] for obj in mapper.objects}
                    record = dict(frame_idx=idx, predictions_exactly_reproduced=True,
                        associations=associations, confirmations=confirmations,
                        expired_node_ids=sorted(before - after), output_count=len(output),
                        original_frame_audit=mapper.frame_audit)
                    if mapper.module_a:
                        record['sparse_hypotheses'] = [dict(node_id=t['node_id'], category=t['category'],
                            independent_observations=t['hits'], measured_points_total=t['total_points'],
                            timestamp_s=t['timestamp'], position_covariance_diag=t['variance'].tolist())
                            for t in mapper.position_layer.tracks]
                    out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                    frames += 1
                    audit_rows += len(associations) + len(confirmations)
            scenes += 1
            del mapper
            print(name, 'exactly reproduced', flush=True)
        verify_source_snapshots(original, release)
        if frames != 1508 or scenes != 38:
            raise AssertionError('核验范围不完整')
        dump(dest / 'report.json', dict(mode='verification_only', condition=method,
            frames=frames, scenes=scenes, predictions_exactly_reproduced_every_frame=True,
            association_confirmation_records=audit_rows, wall_s=time.monotonic() - begun,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            original_run=str(original), new_candidate=False, parameter_changes=False,
            new_statistical_test=False, test_accessed=False,
            output_sha256={str(p.relative_to(dest)): digest(p)
                           for p in dest.glob('scene-*/*.gz')}))
    except BaseException as exc:
        dump(dest / 'interrupted.json', dict(complete=False, frames=frames, scenes=scenes,
            error=repr(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
