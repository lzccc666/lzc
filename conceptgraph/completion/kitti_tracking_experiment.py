"""KITTI连续车载开发诊断：实际CPU前端、无GT建图、隔离身份评价。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import gzip
import json
import os
from pathlib import Path
import pickle
import shutil
import tarfile
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

from conceptgraph.completion.quality import object_frame_metrics, identity_metrics
from conceptgraph.completion.realtime_batch import pipeline_sources
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump, predict
from conceptgraph.completion.temporal_identity_evaluation import (
    SUPPORTS, aggregate_statistics, identity_counts, prepare_sequence, return_diagnostics,
)

METHODS = ('single_frame', 'original_fusion', 'one_to_one_depth_robust_motion')
CATEGORIES = {'Car': 'car', 'Truck': 'truck', 'Pedestrian': 'pedestrian'}


def source_hashes():
    files = set(pipeline_sources())
    files.update(PROJECT/f'conceptgraph/completion/{name}.py' for name in (
        'kitti_tracking_experiment', 'kitti_tracking_geometry', 'kitti_geometry',
        'temporal_identity_evaluation'))
    files.update([PROJECT/'conceptgraph/hydra_configs/base_mapping.yaml',
                  PROJECT/'conceptgraph/vehicle_classes.txt',
                  OUTPUT/'task_protocol/kitti_tracking_001.md',
                  OUTPUT/'task_protocol/kitti_tracking_time_addendum_001.md'])
    return {str(p.relative_to(PROJECT)): digest(p) for p in sorted(files)}


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError('输入/输出SHA不一致：'+str(path))


def guard(started, args, destination):
    if time.monotonic()-started > args.wall_budget_s:
        raise TimeoutError('达到冻结墙钟预算')
    if shutil.disk_usage(OUTPUT).free < args.minimum_free_gb*1024**3:
        raise OSError('磁盘余量低于保护预算')
    if sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()) > args.storage_budget_gb*1024**3:
        raise OSError('达到新增存储预算')


def load_agent(path):
    path = Path(path).resolve()
    if path.name != 'agent_manifest.json':
        raise ValueError('观测链只接受agent_manifest.json')
    agent = json.loads(path.read_text())
    if set(agent) != {'frames', 'classes', 'selection_sha256'}:
        raise ValueError('agent清单含未声明字段')
    for row in agent['frames']:
        if any('evaluation_only' in str(value) for value in (row['rgb'], row['depth'], row['calib'], row['lidar'])):
            raise ValueError('评价路径不得进入观测链')
    return path, agent


def freeze(destination, release):
    dump(destination/'release.json', release)
    with tarfile.open(destination/'sources.tar.gz', 'x:gz') as archive:
        for name in release['source_sha256']:
            archive.add(PROJECT/name, arcname=name, recursive=False)


def front(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU旁路必须显式隐藏CUDA')
    import cv2
    import torch
    from conceptgraph.scripts.streamlined_detections import detect_rgb_file, load_detection_models

    started = time.monotonic()
    path, agent = load_agent(args.agent_manifest)
    frames = agent['frames'][:args.limit] if args.limit else agent['frames']
    destination = OUTPUT/'mapping_evaluation'/args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    checkpoint = PROJECT/'checkpoint'
    weights = [checkpoint/'yolov8l-world.pt', checkpoint/'mobile_sam.pt',
        checkpoint/'hf_models/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_model.safetensors']
    release = dict(agent_manifest=str(path), agent_manifest_sha256=digest(path),
        source_sha256=source_hashes(), model_sha256={str(p): digest(p) for p in weights},
        classes=agent['classes'], selected=[(r['sequence'], r['original_frame_id']) for r in frames],
        gt_read=False, device='cpu', threads=1, performance_credit=False, budgets=vars(args))
    freeze(destination, release)
    try:
        reused = {}
        if args.reuse_frontend:
            old = OUTPUT/'mapping_evaluation'/args.reuse_frontend
            prior = json.loads((old/'release.json').read_text())
            for field in ['agent_manifest_sha256', 'source_sha256', 'model_sha256', 'classes', 'device', 'threads']:
                if release[field] != prior[field]:
                    raise ValueError('续跑前端指纹不一致：'+field)
            summary = json.loads((old/'summary.json').read_text())
            if summary['gt_read'] or summary['cuda_initialized']:
                raise ValueError('既有前端隔离失效')
            reused = {(r['sequence'], r['original_frame_id']): r for r in summary['frames']}
        torch.set_num_threads(1)
        cfg = SimpleNamespace(device='cpu')
        models = load_detection_models(cfg)
        for model in models[:2]:
            model.to('cpu')
            model.overrides['device'] = 'cpu'
        cold = time.monotonic()-started
        rows, new_calls = [], 0
        for row in frames:
            guard(started, args, destination)
            verify_hashes(row['input_sha256'])
            key = row['sequence'], row['original_frame_id']
            if key in reused:
                saved = copy.deepcopy(reused[key])
                verify_hashes(saved['output_sha256'])
                if saved['input_sha256'] != row['input_sha256']:
                    raise ValueError('复用观测输入不同')
                saved['reused_from'] = args.reuse_frontend
            else:
                tick = time.monotonic()
                result, annotated = detect_rgb_file(cfg, models, row['rgb'], agent['classes'])
                if torch.cuda.is_initialized() or any(str(next(m.model.parameters()).device) != 'cpu' for m in models[:2]):
                    raise RuntimeError('CPU隔离失败')
                if result['frame_provenance']['rgb_sha256'] != row['input_sha256'][row['rgb']]:
                    raise ValueError('前端RGB来源不同')
                folder = destination/'frames'/row['sequence']/f"{row['original_frame_id']:06d}"
                folder.mkdir(parents=True)
                detection = folder/'detections.pkl.gz'
                with gzip.open(detection, 'xb') as handle:
                    pickle.dump(result, handle)
                visual = folder/'detections.jpg'
                if not cv2.imwrite(str(visual), annotated):
                    raise OSError('检测可视化保存失败')
                saved = dict(sequence=key[0], original_frame_id=key[1], frame_idx=row['frame_idx'],
                    detections=str(detection), visualization=str(visual), input_sha256=row['input_sha256'],
                    output_sha256={str(p): digest(p) for p in [detection, visual]},
                    elapsed_s=time.monotonic()-tick, boxes=len(result['xyxy']),
                    provenance=result['frame_provenance'], reused_from=None)
                new_calls += 1
                dump(folder/'complete.json', saved)
            rows.append(saved)
            print(json.dumps(dict(sequence=key[0], frame=key[1], new_calls=new_calls,
                boxes=saved['boxes'], reused=saved['reused_from'] is not None)), flush=True)
        if source_hashes() != release['source_sha256']:
            raise ValueError('实验期间源码变化')
        dump(destination/'summary.json', dict(frames=rows, frame_count=len(rows),
            actual_new_frontend_calls=new_calls, reused_frames=len(rows)-new_calls,
            gt_read=False, cuda_initialized=torch.cuda.is_initialized(), cold_start_s=cold,
            shared_CPU_wall_s=time.monotonic()-started, performance_credit=False,
            navigation_arrivals=0, release_sha256=digest(destination/'release.json')))
    except BaseException as exc:
        dump(destination/'interrupted.json', dict(error=repr(exc), wall_s=time.monotonic()-started))
        raise


def in_view(predictions, row):
    pose, intrinsic = np.asarray(row['pose']), np.asarray(row['intrinsic'])
    width, height = row['image_wh']
    selected = []
    for obj in predictions:
        center = pose[:3, :3].T@(np.asarray(obj['center'])-pose[:3, 3])
        uv = intrinsic@center
        if obj['category'] in CATEGORIES.values() and 1 < center[2] < 65 and 0 <= uv[0]/uv[2] < width and 0 <= uv[1]/uv[2] < height:
            selected.append(obj | {'center_camera2': center.tolist()})
    return selected


def build_map(args):
    from omegaconf import OmegaConf
    from conceptgraph.slam.streamlined_mapping import MappingFrame, StreamlinedMapper

    started = time.monotonic()
    path, agent = load_agent(args.agent_manifest)
    frontend = OUTPUT/'mapping_evaluation'/args.frontend_run
    front_release = json.loads((frontend/'release.json').read_text())
    if front_release['agent_manifest_sha256'] != digest(path):
        raise ValueError('前端agent清单不同')
    front_summary = json.loads((frontend/'summary.json').read_text())
    if front_summary['gt_read'] or front_summary['cuda_initialized']:
        raise ValueError('前端隔离失败')
    inputs = {(r['sequence'], r['original_frame_id']): r for r in agent['frames']}
    destination = OUTPUT/'mapping_evaluation'/args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(PROJECT/'conceptgraph/hydra_configs/base_mapping.yaml')
    cfg.merge_with(dict(device='cpu', min_points_threshold=4, obj_min_detections=1,
        dbscan_remove_noise=False, match_method='sim_sum', sim_threshold=.35,
        phys_bias=.5, spatial_sim_type='iou', frame_dt=.1, skip_bg=True))
    improved_cfg = copy.deepcopy(cfg)
    improved_cfg.merge_with(dict(association_policy='one_to_one', dynamic_ttl_s=2.,
        initial_motion_unknown=True, static_confirmation_observations=3,
        depth_policy='largest_supported_cluster', motion_estimator='robust_xy'))
    release = dict(source_sha256=source_hashes(), agent_manifest=str(path),
        agent_manifest_sha256=digest(path), frontend_release_sha256=digest(frontend/'release.json'),
        frontend_summary_sha256=digest(frontend/'summary.json'), frontend_run=args.frontend_run,
        configs={METHODS[1]: OmegaConf.to_container(cfg), METHODS[2]: OmegaConf.to_container(improved_cfg)},
        methods=METHODS, gt_read=False, nominal_dt_s=.1, measured_sensor_timestamps=False,
        single_frame_source='original_fusion.last_observations', budgets=vars(args))
    if release['source_sha256'] != front_release['source_sha256']:
        raise ValueError('前端与建图代码未冻结在同一版本')
    freeze(destination, release)
    sequence = None
    complete = []
    try:
        with (destination/'frames.jsonl').open('x') as handle:
            for saved in front_summary['frames']:
                guard(started, args, destination)
                row = inputs[saved['sequence'], saved['original_frame_id']]
                verify_hashes(row['input_sha256'])
                verify_hashes(saved['output_sha256'])
                if row['sequence'] != sequence:
                    sequence = row['sequence']
                    mappers = {METHODS[1]: StreamlinedMapper(copy.deepcopy(cfg), agent['classes']),
                        METHODS[2]: StreamlinedMapper(copy.deepcopy(improved_cfg), agent['classes'])}
                with gzip.open(saved['detections'], 'rb') as detection:
                    gobs = pickle.load(detection)
                image = np.asarray(Image.open(row['rgb']).convert('RGB'))
                depth = np.asarray(Image.open(row['depth']), dtype=float)/1000
                all_predictions, timings = {}, {}
                for method, mapper in mappers.items():
                    tick = time.monotonic()
                    np.random.seed(17+row['frame_idx'])
                    mapper.update(MappingFrame(row['frame_idx'], image, depth, np.asarray(row['intrinsic']),
                        np.asarray(row['pose']), copy.deepcopy(gobs), row['rgb'],
                        timestamp_s=row['nominal_timestamp_s']))
                    all_predictions[method] = predict(mapper.objects, agent['classes'])
                    timings[method] = time.monotonic()-tick
                all_predictions[METHODS[0]] = predict(mappers[METHODS[1]].last_observations, agent['classes'])
                predictions = {m: in_view(all_predictions[m], row) for m in METHODS}
                record = dict(sequence=sequence, frame_idx=row['frame_idx'], original_frame_id=row['original_frame_id'],
                    camera_timestamp_s=row['nominal_timestamp_s'], sensor_timestamp_measured=False,
                    rgb=row['rgb'], depth=row['depth'], detections=saved['detections'],
                    detection_visualization=saved['visualization'], input_sha256=row['input_sha256'],
                    frontend_output_sha256=saved['output_sha256'], pose=row['pose'], intrinsic=row['intrinsic'],
                    image_wh=row['image_wh'], predictions=predictions, all_predictions=all_predictions,
                    mapping_shared_CPU_s=timings, gt_read=False)
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                handle.flush()
                complete.append(dict(sequence=sequence, frame_idx=row['frame_idx'],
                    objects={m: len(v) for m, v in predictions.items()}))
                print(json.dumps(complete[-1]), flush=True)
        if source_hashes() != release['source_sha256']:
            raise ValueError('建图期间源码变化')
        dump(destination/'summary.json', dict(frames=complete, frame_count=len(complete),
            frames_sha256=digest(destination/'frames.jsonl'), gt_read=False,
            shared_CPU_wall_s=time.monotonic()-started, nominal_dt_s=.1,
            actual_sensor_timestamps=False, navigation_arrivals=0))
    except BaseException as exc:
        dump(destination/'interrupted.json', dict(error=repr(exc), wall_s=time.monotonic()-started))
        raise


def eligible_truth(objects, image_wh):
    width, height = image_wh
    eligible, excluded = [], []
    for obj in objects:
        box = obj['bbox']
        reason = None
        if obj['category'] not in CATEGORIES:
            reason = 'unsupported_category_or_DontCare'
        elif obj['track_id'] < 0 or min(obj['dimensions_hwl']) <= 0:
            reason = 'no_valid_official_3D_identity'
        elif not 1 < obj['center_camera2'][2] < 65:
            reason = 'outside_fixed_depth_scope'
        elif min(box[2], width) <= max(box[0], 0) or min(box[3], height) <= max(box[1], 0):
            reason = 'label_bbox_outside_image'
        if reason:
            excluded.append(obj | {'exclusion_reason': reason})
        else:
            eligible.append(obj | {'original_category': obj['category'], 'category': CATEGORIES[obj['category']]})
    return eligible, excluded


def object_totals(rows):
    keys = ['predictions', 'truth', 'true_positive', 'unmatched_predictions', 'false_negative', 'duplicate_nodes']
    total = {key: sum(row[key] for row in rows) for key in keys}
    total['center_error_sum_m'] = sum(sum(m['distance_m'] for m in r['matches']) for r in rows)
    return total


def object_rates(total):
    def ratio(a, b):
        return total[a]/total[b] if total[b] else None
    return dict(recall=ratio('true_positive', 'truth'),
        matched_prediction_fraction=ratio('true_positive', 'predictions'),
        center_error_m=ratio('center_error_sum_m', 'true_positive'),
        duplicate_fraction=ratio('duplicate_nodes', 'predictions'))


def object_statistics(by_sequence):
    sequences = sorted(by_sequence)
    draws = np.random.default_rng(17).integers(len(sequences), size=(4000, len(sequences)))
    results, boots, differences = {}, {}, []
    for method in METHODS:
        totals = [object_totals([f['metrics'][method] for f in by_sequence[seq]]) for seq in sequences]
        total = {key: sum(x[key] for x in totals) for key in totals[0]}
        results[method] = dict(counts=total, metrics=object_rates(total))
        values = defaultdict(list)
        for draw in draws:
            aggregate = {key: sum(totals[i][key] for i in draw) for key in total}
            for metric, value in object_rates(aggregate).items():
                values[metric].append(np.nan if value is None else value)
        boots[method] = {k: np.array(v) for k, v in values.items()}
    for method in METHODS:
        if method == 'original_fusion':
            continue
        for metric, value in results[method]['metrics'].items():
            base = results['original_fusion']['metrics'][metric]
            boot = boots[method][metric]-boots['original_fusion'][metric]
            finite = boot[np.isfinite(boot)]
            differences.append(dict(method=method, baseline='original_fusion', metric=metric,
                difference=value-base if value is not None and base is not None else None,
                ci95=np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                valid_resamples=len(finite)))
    return dict(methods=results, paired_differences=differences,
        bootstrap=dict(unit='sequence', repetitions=4000, seed=17, paired=True,
                       limit='仅三个序列，区间不证明广泛泛化'))


def evaluate(args):
    started = time.monotonic()
    mapping = OUTPUT/'mapping_evaluation'/args.mapping_run
    data = OUTPUT/'data_manifest'/args.input_run
    complete = json.loads((data/'complete.json').read_text())
    if digest(data/'evaluation_only/truth.json') != complete['truth_sha256']:
        raise ValueError('隔离GT指纹不一致')
    release = json.loads((mapping/'release.json').read_text())
    summary = json.loads((mapping/'summary.json').read_text())
    if release['agent_manifest_sha256'] != complete['agent_manifest_sha256'] or summary['gt_read']:
        raise ValueError('隔离推理/建图不匹配')
    if digest(mapping/'frames.jsonl') != summary['frames_sha256']:
        raise ValueError('建图记录指纹不一致')
    truth = {(r['sequence'], r['original_frame_id']): r for r in json.loads((data/'evaluation_only/truth.json').read_text())['frames']}
    destination = OUTPUT/'statistics'/args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    dump(destination/'release.json', dict(source_sha256=source_hashes(), mapping_run=args.mapping_run,
        mapping_release_sha256=digest(mapping/'release.json'), mapping_summary_sha256=digest(mapping/'summary.json'),
        input_complete_sha256=digest(data/'complete.json'), truth_sha256=complete['truth_sha256'],
        gate='同粗类别，初始IMU世界XY<=3m，最大基数', official_KITTI_HOTA_AP=False,
        budgets=vars(args), goal_complete=False))
    by_sequence = defaultdict(list)
    for line in (mapping/'frames.jsonl').open():
        guard(started, args, destination)
        frame = json.loads(line)
        raw_truth = truth[frame['sequence'], frame['original_frame_id']]['objects']
        eligible, excluded = eligible_truth(raw_truth, frame['image_wh'])
        frame.update(evaluation_only_official=eligible, evaluation_only_excluded=excluded,
            metrics={m: object_frame_metrics(frame['predictions'][m], eligible) for m in METHODS})
        by_sequence[frame['sequence']].append(frame)
    rows, scene_reports = [], {}
    for sequence, frames in by_sequence.items():
        with (destination/f'{sequence}_frames.jsonl').open('x') as handle:
            for frame in frames:
                handle.write(json.dumps(frame, ensure_ascii=False, allow_nan=False)+'\n')
        scene_reports[sequence] = {}
        for method in METHODS:
            records = prepare_sequence(frames, method)
            returned = return_diagnostics(records)
            pairs = [(f['frame_idx'], [(f['predictions'][method][m['prediction_index']]['node_id'],
                f['evaluation_only_official'][m['truth_index']]['instance_token']) for m in f['metrics'][method]['matches']]) for f in frames]
            scene_reports[sequence][method] = dict(object=object_totals([f['metrics'][method] for f in frames]),
                legacy_identity=identity_metrics(pairs), return_diagnostics=returned)
            for support in SUPPORTS:
                identity = identity_counts(records, support)
                dump(destination/f'{sequence}_{method}_{support}.json', identity)
                rows.append(dict(log_token=sequence, sequence=sequence, method=method, support=support,
                    counts=identity['counts'] | returned['counts']))
    identity, by_log = aggregate_statistics(rows)
    dump(destination/'scene_reports.json', scene_reports)
    dump(destination/'identity_rows.json', rows)
    dump(destination/'identity_by_sequence.json', by_log)
    report = dict(dataset='KITTI Tracking training', role='external_development_diagnostic',
        frame_count=sum(map(len, by_sequence.values())), sequence_count=len(by_sequence),
        primary_categories=list(CATEGORIES.values()), object=object_statistics(by_sequence), identity=identity,
        excluded_label_reasons=dict(Counter(obj['exclusion_reason'] for frames in by_sequence.values() for f in frames for obj in f['evaluation_only_excluded'])),
        source_evaluation_frames_sha256={str(destination/f'{seq}_frames.jsonl'): digest(destination/f'{seq}_frames.jsonl') for seq in by_sequence},
        nominal_dt_s=.1, measured_sensor_timestamps=False, official_KITTI_HOTA_AP=False,
        navigation_arrivals=0, human_annotation_credit=False, GPU_performance_credit=False,
        limitations=['三个3秒中心窗口，非完整序列；三个序列不等于三个独立采集log',
            '预训练数据与KITTI重合未知；开发诊断不能替代新的封存自然语言测试',
            'GT框中心与可见表面点云中心存在定义差异；XY中心门限关联不等于逐点人工身份真值',
            '未匹配预测包含DontCare/未支持类别混淆，matched_prediction_fraction不是完整误检率',
            'original_fusion已统一正确名义0.1秒；改进条件是组合改动',
            '无原始传感器采集时间戳，不声明实测同步或真实时钟速度真值'],
        shared_CPU_wall_s=time.monotonic()-started, goal_complete=False)
    dump(destination/'report.json', report)
    print(json.dumps(dict(frames=report['frame_count'], sequences=report['sequence_count'],
        object=report['object']['methods']), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['frontend', 'map', 'evaluate']:
        p = sub.add_parser(name)
        p.add_argument('--run-id', required=True)
        p.add_argument('--wall-budget-s', type=float, default=3600 if name == 'frontend' else 600)
        p.add_argument('--storage-budget-gb', type=float, default=3 if name == 'frontend' else .5)
        p.add_argument('--minimum-free-gb', type=float, default=6)
        if name in ['frontend', 'map']:
            p.add_argument('--agent-manifest', required=True)
        if name == 'frontend':
            p.add_argument('--limit', type=int, default=0)
            p.add_argument('--reuse-frontend')
        elif name == 'map':
            p.add_argument('--frontend-run', required=True)
        else:
            p.add_argument('--input-run', required=True)
            p.add_argument('--mapping-run', required=True)
    args = parser.parse_args()
    {'frontend': front, 'map': build_map, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
