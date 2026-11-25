"""R1基点上的固定关联候选与机制消融；同输入及原评价协议。"""
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

from conceptgraph.completion.config_comparison import ratios, summarize
from conceptgraph.completion.fault_robustness import mapping_config
from conceptgraph.completion.quality import object_frame_metrics
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, available_development, digest, dump, predict
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.project_paths import experiment_directory
from conceptgraph.slam.streamlined_mapping import MappingFrame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--condition', choices=['R2', 'R3', 'R3_single_stage'], required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--wall-budget-s', type=float, default=1500)
    args = parser.parse_args()
    begun = time.monotonic()
    dest = experiment_directory('nuscenes', 'validation', args.run_id, create=True)
    scenes = available_development()[:2] if args.pilot else available_development()
    reference = PROJECT / 'outputs/results/nuscenes/validation/iterations/vehicle_frontend_full_20261118_v1'
    sources = [Path(__file__).resolve(), PROJECT / 'conceptgraph/completion/config_comparison.py',
               PROJECT / 'conceptgraph/completion/fault_robustness.py',
               PROJECT / 'conceptgraph/completion/stage9.py', PROJECT / 'conceptgraph/project_paths.py',
               PROJECT / 'conceptgraph/completion/quality.py',
               PROJECT / 'conceptgraph/completion/temporal_identity_evaluation.py',
               PROJECT / 'conceptgraph/completion/association.py', PROJECT / 'conceptgraph/completion/depth.py',
               PROJECT / 'conceptgraph/slam/audited_mapper.py', PROJECT / 'conceptgraph/slam/evidence_mapper.py',
               PROJECT / 'conceptgraph/slam/streamlined_mapping.py', PROJECT / 'conceptgraph/slam/utils.py',
               PROJECT / 'conceptgraph/slam/mapping.py', PROJECT / 'conceptgraph/hydra_configs/base_mapping.yaml']
    from conceptgraph.slam.dense_refinement import DenseRefinementSettings
    from conceptgraph.slam.uncertainty_refinement import create_refinement_mapper, UncertaintyAssociationSettings
    from conceptgraph.slam.sparse_observations import SparseQualitySettings
    cls, settings = create_refinement_mapper, dict(condition=args.condition)
    sources += [PROJECT / 'conceptgraph/slam/quality_mapper.py',
                PROJECT / 'conceptgraph/slam/sparse_observations.py',
                PROJECT / 'conceptgraph/slam/dense_refinement.py',
                PROJECT / 'conceptgraph/slam/uncertainty_refinement.py',
                PROJECT / 'conceptgraph/slam/motion_association_mapper.py']
    algorithm_settings = dict(sparse=SparseQualitySettings().to_dict(),
                              dense=DenseRefinementSettings().to_dict(),
                              reliable_dense_protection=args.condition in {'R3', 'R3_single_stage'},
                              assignment_mode=('single_stage' if args.condition == 'R3_single_stage'
                                               else 'two_stage'),
                              uncertainty=UncertaintyAssociationSettings().to_dict())
    cfg = mapping_config()
    cfg.merge_with(dict(mask_conf_threshold=.1, strong_confidence=.2, weak_confirmation_observations=2))
    source_sha = {str(p): digest(p) for p in sources}
    for source in sources:
        target = dest / 'source_snapshot' / source.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    input_sha = {}
    for scene in scenes:
        name = scene['scene']
        for p in [OUTPUT / 'mapping_evaluation/causal_mapping_full_001' / name / 'frames.jsonl',
                  reference / name / 'frames.jsonl.gz']:
            input_sha[str(p)] = digest(p)
    protected = [PROJECT / 'MEMORY.md', PROJECT / 'PAPER_MEMORY.md', reference / 'report.json']
    dump(dest / 'release.json', dict(condition=args.condition, baseline='R1', pilot=args.pilot, scenes=scenes,
        config=OmegaConf.to_container(cfg), experimental_settings=settings,
        algorithm_settings=algorithm_settings,
        sources_sha256=source_sha, inputs_sha256=input_sha,
        protected_sha256={str(p): digest(p) for p in protected}, seed=17,
        wall_budget_s=args.wall_budget_s, new_storage_limit_gib=2, minimum_free_gib=20,
        ground_truth_in_inference=False, test_accessed=False, frontend_calls=0,
        camera_shape_wh=[1600, 900], threshold_m=3., paired_bootstrap_replicates=10000,
        comparisons_registered=3, bootstrap_seed=20261123, primary_metric='IDF1',
        comparison_family='association_optimization_20261123'))
    reports = []
    files_bytes = 0
    try:
        for scene in scenes:
            name = scene['scene']
            with gzip.open(reference / name / 'frames.jsonl.gz', 'rt') as stream:
                cached = {r['frame_idx']: r for r in map(json.loads, stream)}
            mapper = cls(copy.deepcopy(cfg), CLASSES, **settings) if settings else cls(copy.deepcopy(cfg), CLASSES)
            case = dest / name
            case.mkdir()
            records = []
            input_path = OUTPUT / 'mapping_evaluation/causal_mapping_full_001' / name / 'frames.jsonl'
            with input_path.open() as source, gzip.open(case / 'frames.jsonl.gz', 'wt') as out:
                for row in map(json.loads, source):
                    if time.monotonic() - begun > args.wall_budget_s:
                        raise TimeoutError('达到事前预算，保留前缀')
                    if shutil.disk_usage(dest).free < 20 * 1024**3 or files_bytes > 2 * 1024**3:
                        raise OSError('达到磁盘保护线')
                    idx = row['frame_idx']
                    detection = Path(cached[idx]['detections'])
                    if digest(detection) != cached[idx]['detection_sha256']:
                        raise ValueError('检测缓存指纹不一致')
                    with gzip.open(detection, 'rb') as stream:
                        gobs = pickle.load(stream)
                    for p in [Path(row['rgb']), Path(row['depth']), detection]:
                        input_sha[str(p)] = digest(p)
                    frame = MappingFrame(idx, np.asarray(Image.open(row['rgb']).convert('RGB')),
                        np.asarray(Image.open(row['depth']), dtype=float) / 1000,
                        np.asarray(row['intrinsic']), np.asarray(row['camera_pose']),
                        gobs, row['rgb'], timestamp_s=row['camera_timestamp_s'])
                    np.random.seed(17 + idx)
                    then = time.monotonic()
                    mapper.update(frame)
                    # 与B1相同，在最终位置估计后应用同一视锥。
                    objects = predict(mapper.confirmed_objects(), CLASSES)
                    objects = mapper.position_predictions(objects, frame)
                    elapsed = time.monotonic() - then
                    # 从此处起才读取评价真值；mapper从未收到整条row。
                    truth = row['evaluation_only_official']
                    metrics = object_frame_metrics(objects, truth)
                    record = dict(scene=name, frame_idx=idx, camera_timestamp_s=row['camera_timestamp_s'],
                        predictions={args.condition: objects}, metrics={args.condition: metrics},
                        evaluation_only_official=truth, gt_in_inference=False,
                        input=dict(rgb=row['rgb'], depth=row['depth'], detections=str(detection),
                                   detections_sha256=cached[idx]['detection_sha256']),
                        audit=mapper.frame_audit, mapping_s=elapsed)
                    out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                    records.append(record)
            result = summarize(records, args.condition)
            report = dict(scene=name, log=scene['log_token'], frames=len(records), result=result,
                          mapping_s=sum(r['mapping_s'] for r in records),
                          records_sha256=digest(case / 'frames.jsonl.gz'))
            dump(case / 'report.json', report)
            reports.append(report)
            files_bytes += (case / 'frames.jsonl.gz').stat().st_size
            del mapper
            print(json.dumps(dict(scene=name, condition=args.condition, **result['metrics'])), flush=True)
        counts = {k: sum(r['result']['counts'][k] for r in reports) for k in reports[0]['result']['counts']}
        if any(digest(Path(p)) != sha for p, sha in source_sha.items()):
            raise ValueError('运行期间源码变化')
        if any(digest(Path(p)) != sha for p, sha in input_sha.items()):
            raise ValueError('运行期间输入变化')
        dump(dest / 'input_fingerprints.json', input_sha)
        dump(dest / 'report.json', dict(condition=args.condition, scenes=len(reports),
            logs=len({r['log'] for r in reports}), frames=sum(r['frames'] for r in reports),
            result=dict(counts=counts, metrics=ratios(counts)),
            full_validation_scope=not args.pilot, comparison_baseline='R1',
            mapping_s=sum(r['mapping_s'] for r in reports), wall_s=time.monotonic() - begun,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            serialized_bytes=files_bytes, frontend_calls=0, test_accessed=False,
            actual_arrival_evaluated=False, threshold_m=3.))
        print(json.dumps(dict(condition=args.condition, counts=counts, metrics=ratios(counts))), flush=True)
    except BaseException as exc:
        dump(dest / 'interrupted.json', dict(complete=False, error=repr(exc),
             traceback=traceback.format_exc(), completed_scenes=len(reports), wall_s=time.monotonic() - begun))
        raise


if __name__ == '__main__':
    main()
