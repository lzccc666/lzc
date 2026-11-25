"""串行配置重建：同缓存、完整分母、逐帧证据与原身份评价。"""
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
from conceptgraph.slam.audited_mapper import AuditedEvidenceMapper
from conceptgraph.slam.streamlined_mapping import MappingFrame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--condition', choices=['B0', 'B1', 'B2', 'B3'], required=True)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--wall-budget-s', type=float, default=1500)
    args = parser.parse_args()
    if args.condition in {'B2', 'B3'}:
        parser.error('B1已满足提前停止的数值条件；本轮B2/B3未实施，不能将注册配置当成已实现算法')
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
    cls = AuditedEvidenceMapper
    settings = {}
    algorithm_settings = {}
    if args.condition != 'B0':
        from conceptgraph.slam.quality_mapper import QualityEvidenceMapper, SETTINGS
        cls, settings = QualityEvidenceMapper, SETTINGS[args.condition]
        sources += [PROJECT / 'conceptgraph/slam/quality_mapper.py',
                    PROJECT / 'conceptgraph/slam/sparse_observations.py']
        from conceptgraph.slam.sparse_observations import SparseQualitySettings
        algorithm_settings = SparseQualitySettings().to_dict()
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
    dump(dest / 'release.json', dict(condition=args.condition, pilot=args.pilot, scenes=scenes,
        config=OmegaConf.to_container(cfg), experimental_settings=settings,
        algorithm_settings=algorithm_settings,
        sources_sha256=source_sha, inputs_sha256=input_sha,
        protected_sha256={str(p): digest(p) for p in protected}, seed=17,
        wall_budget_s=args.wall_budget_s, new_storage_limit_gib=2, minimum_free_gib=20,
        ground_truth_in_inference=False, test_accessed=False, frontend_calls=0,
        camera_shape_wh=[1600, 900], threshold_m=3., paired_bootstrap_replicates=10000,
        comparisons_registered=3, bootstrap_seed=20261120))
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
                    # 模块A最终位置估计之后才应用相同视锥；B0/B2保持原入口。
                    if settings.get('module_a'):
                        objects = predict(mapper.confirmed_objects(), CLASSES)
                    else:
                        objects = predict(mapper.confirmed_objects(), CLASSES,
                                          pose=frame.camera_pose, intrinsic=frame.cam_K)
                    if args.condition != 'B0':
                        objects = mapper.position_predictions(objects, frame)
                    elapsed = time.monotonic() - then
                    # 从此处起才读取评价真值；mapper从未收到整条row。
                    truth = row['evaluation_only_official']
                    metrics = object_frame_metrics(objects, truth)
                    if args.condition == 'B0':
                        for k in ('true_positive', 'truth', 'predictions', 'duplicate_nodes'):
                            if metrics[k] != cached[idx]['metrics']['frontend_v2_evidence'][k]:
                                raise AssertionError(f'基线未复现：{name}/{idx}/{k}')
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
            full_validation_scope=not args.pilot, baseline_counts_reproduced=args.condition == 'B0',
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
