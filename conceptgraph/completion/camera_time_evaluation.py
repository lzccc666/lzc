"""固定开发对象集合的评价中心时间敏感性；只重评分，不调用感知或导航。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np

from conceptgraph.completion.quality import identity_metrics, object_frame_metrics
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, available_development, digest, dump
from conceptgraph.nuscenes.benchmark_ground_truth import official_category
from conceptgraph.nuscenes.expansion_benchmark import iter_records


CONDITIONS = ['official_sample_proxy', 'camera_center_interpolation_proxy']
RATIOS = {'recall': (0, 1), 'matched_prediction_precision': (0, 2),
    'center_error_m_mean': (3, 0), 'identity_switch_rate': (4, 5),
    'duplicate_nodes_per_GT': (6, 1)}
GROUPS = ['interpolated', 'exact', 'fallback', 'motion_static', 'motion_dynamic', 'motion_unknown']


def interpolate_center(obj, current, previous, samples, camera_us, maximum_gap_s=1.):
    """只对被两个同实例官方中心包围的时刻插值；不外推或改变对象集合。"""
    if (obj['annotation_token'] != current['token']
            or obj['instance_token'] != current['instance_token']
            or not np.allclose(obj['center'], current['translation'], atol=1e-12, rtol=0)):
        raise ValueError('当前官方对象与原始标注不一致')
    current_sample = samples[current['sample_token']]
    current_us = current_sample['timestamp']
    result = copy.deepcopy(obj)
    detail = dict(annotation_token=current['token'], instance_token=current['instance_token'],
        camera_timestamp_us=camera_us, current_sample_timestamp_us=current_us,
        current_annotation_lag_ms=(current_us-camera_us)/1000.,
        previous_annotation_token=current['prev'] or None, previous_sample_timestamp_us=None,
        interpolation_weight=None, center_only=True, evaluation_only=True)
    if camera_us == current_us:
        state, reason = 'exact', 'camera_equals_official_sample'
    elif previous is None:
        state, reason = 'fallback', 'no_previous_annotation'
    else:
        if (previous['token'] != current['prev']
                or previous['instance_token'] != current['instance_token']):
            raise ValueError('不能使用其他实例或非紧邻前驱插值')
        previous_sample = samples[previous['sample_token']]
        if previous_sample['scene_token'] != current_sample['scene_token']:
            raise ValueError('前后标注不在同一场景')
        previous_us = previous_sample['timestamp']
        detail['previous_sample_timestamp_us'] = previous_us
        if current_us <= previous_us:
            raise ValueError('官方前后时间不递增')
        if current_us-previous_us > maximum_gap_s*1e6:
            state, reason = 'fallback', 'annotation_gap_exceeds_budget'
        elif not previous_us <= camera_us <= current_us:
            state, reason = 'fallback', 'camera_not_bracketed_no_extrapolation'
        else:
            weight = (camera_us-previous_us)/(current_us-previous_us)
            result['center'] = ((1-weight)*np.asarray(previous['translation'], dtype=float)
                + weight*np.asarray(current['translation'], dtype=float)).tolist()
            detail['interpolation_weight'] = weight
            state, reason = 'interpolated', 'linear_centers_between_adjacent_official_annotations'
    detail.update(state=state, reason=reason,
        center_shift_m=float(np.linalg.norm(np.asarray(result['center'])-obj['center'])))
    return result, detail


def ratio_values(counts):
    return np.stack([np.divide(counts[..., i], counts[..., j],
        out=np.full(counts.shape[:-1], np.nan), where=counts[..., j] > 0)
        for i, j in RATIOS.values()], axis=-1)


def estimate(point, samples):
    valid = samples[np.isfinite(samples)]
    return dict(value=float(point) if np.isfinite(point) else None,
        ci95=np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
        valid_resamples=len(valid))


def write_statistics(destination, counts, cohorts, logs, methods):
    draws = np.random.default_rng(17).integers(0, len(logs), (4000, len(logs)))
    points = ratio_values(counts.sum(axis=0))
    samples = ratio_values(counts[draws].sum(axis=1))
    conditions, temporal_differences, contrasts = {}, [], []
    for c, condition in enumerate(CONDITIONS):
        conditions[condition] = {}
        for m, method in enumerate(methods):
            conditions[condition][method] = {name: estimate(points[c, m, k], samples[:, c, m, k])
                for k, name in enumerate(RATIOS)}
            if c:
                for k, name in enumerate(RATIOS):
                    temporal_differences.append(dict(method=method, metric=name,
                        condition=condition, baseline=CONDITIONS[0],
                        **estimate(points[c, m, k]-points[0, m, k],
                            samples[:, c, m, k]-samples[:, 0, m, k])))
        for method, baseline in [('one_to_one_depth_quality', 'original_fusion'),
                ('one_to_one_depth_robust_motion', 'one_to_one_depth_quality')]:
            a, b = methods.index(method), methods.index(baseline)
            for k, name in enumerate(RATIOS):
                contrasts.append(dict(condition=condition, method=method, baseline=baseline,
                    metric=name, **estimate(points[c, a, k]-points[c, b, k],
                        samples[:, c, a, k]-samples[:, c, b, k])))
    cohort_points, cohort_samples = cohorts.sum(axis=0), cohorts[draws].sum(axis=1)
    grouped = {}
    for c, condition in enumerate(CONDITIONS):
        grouped[condition] = {}
        for m, method in enumerate(methods):
            grouped[condition][method] = {}
            for g, group in enumerate(GROUPS):
                truth, matched, error = cohort_points[c, m, g]
                sample = cohort_samples[:, c, m, g]
                def ratio(numerator, denominator):
                    return np.divide(numerator, denominator,
                        out=np.full(np.shape(numerator), np.nan), where=denominator > 0)
                grouped[condition][method][group] = dict(truth=int(truth), matched=int(matched),
                    recall=estimate(matched/truth if truth else np.nan, ratio(sample[:, 1], sample[:, 0])),
                    center_error_m_mean=estimate(error/matched if matched else np.nan,
                        ratio(sample[:, 2], sample[:, 1])))
    fields = ['true_positive', 'truth', 'predictions', 'sum_center_error_m',
        'identity_switches', 'adjacent_matched_transitions', 'duplicate_nodes']
    with (destination/'counts_by_log.csv').open('x') as handle:
        writer = csv.writer(handle)
        writer.writerow(['log_token', 'condition', 'method', *fields])
        for i, log in enumerate(logs):
            for c, condition in enumerate(CONDITIONS):
                for m, method in enumerate(methods):
                    writer.writerow([log, condition, method, *counts[i, c, m]])
    with (destination/'cohorts_by_log.csv').open('x') as handle:
        writer = csv.writer(handle)
        writer.writerow(['log_token', 'condition', 'method', 'group', 'truth', 'matched', 'sum_center_error_m'])
        for i, log in enumerate(logs):
            for c, condition in enumerate(CONDITIONS):
                for m, method in enumerate(methods):
                    for g, group in enumerate(GROUPS):
                        writer.writerow([log, condition, method, group, *cohorts[i, c, m, g]])
    return dict(conditions=conditions, temporal_differences=temporal_differences,
        method_contrasts=contrasts, cohorts=grouped,
        cohort_precision='不定义；匹配继承完整集合指派，不能把其他组预测算作本组误检',
        bootstrap=dict(unit='采集log', clusters=len(logs), repetitions=4000, seed=17, paired=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--scene', action='append')
    parser.add_argument('--wall-budget-s', type=float, default=600.)
    args = parser.parse_args()
    begun = time.monotonic()
    known = {row['scene']: row for row in available_development()}
    scenes = args.scene or sorted(known)
    if len(scenes) != len(set(scenes)) or not set(scenes) <= known.keys():
        raise ValueError('只允许不重复的既定开发场景')
    destination = OUTPUT/'mapping_evaluation'/args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    events = OUTPUT/'interactive_simulation/realtime_three_model_full_001/resource_events.jsonl'
    def resource(event):
        with events.open('a') as handle:
            handle.write(json.dumps(dict(event=event, run_id=args.run_id, time_unix_s=time.time(),
                device='cpu', GPU_calls=0, performance_credit=False))+'\n')
    resource('camera_time_evaluation_start')
    def budget():
        if time.monotonic()-begun > args.wall_budget_s or shutil.disk_usage(OUTPUT).free < 6*1024**3:
            raise RuntimeError('达到时间或磁盘保护线')
        if sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()) > 200*1024**2:
            raise RuntimeError('超过200MiB存储预算')
    try:
        original = OUTPUT/'mapping_evaluation/causal_mapping_full_001'
        input_paths = [original/scene/'frames.jsonl' for scene in scenes]
        causal = OUTPUT/'data_manifest/development_causal_inputs_full_001'
        plans = json.loads((causal/'alignment_manifest.json').read_text())['plans']
        plans = {(p['camera']['scene'], p['camera']['frame_idx']): p['camera'] for p in plans}
        metadata = PROJECT/'dataset/nuscenes_trainval_raw/v1.0-trainval'
        tables = [metadata/(name+'.json') for name in ['sample_annotation', 'sample', 'instance', 'category', 'scene']]
        motion_path = OUTPUT/'annotation/official_motion_dev_001/evaluation_only_motion.jsonl'
        sources = [Path(__file__), PROJECT/'conceptgraph/completion/quality.py',
            PROJECT/'conceptgraph/completion/association.py', PROJECT/'conceptgraph/completion/stage9.py',
            PROJECT/'conceptgraph/nuscenes/benchmark_ground_truth.py', PROJECT/'conceptgraph/nuscenes/expansion_benchmark.py',
            OUTPUT/'task_protocol/camera_time_gt_sensitivity_001.md']
        input_sha = {str(p): digest(p) for p in [*input_paths, *tables, motion_path,
            causal/'alignment_manifest.json', causal/'audit_001/report.json']}
        frozen = {str(p): digest(p) for p in sources}
        dump(destination/'release.json', dict(source_sha256=frozen, input_sha256=input_sha, scenes=scenes,
            original_mapping_run='causal_mapping_full_001', maximum_annotation_gap_s=1.,
            conditions=CONDITIONS, center_only=True, original_GT_cohort_preserved=True,
            wall_budget_s=args.wall_budget_s, storage_budget_bytes=200*1024**2,
            minimum_free_bytes=6*1024**3, model_calls=0, independent_test=False,
            methods_frozen_before_condition_results=True, new_test_opened=False, goal_complete=False))
        shutil.copy2(__file__, destination/'generator_source.py')
        frames = {path.parent.name: [json.loads(line) for line in path.open()] for path in input_paths}
        desired = {obj['annotation_token'] for rows in frames.values()
            for frame in rows for obj in frame['evaluation_only_official']}
        annotation_path = metadata/'sample_annotation.json'
        current = {row['token']: row for row in iter_records(annotation_path) if row['token'] in desired}
        if set(current) != desired:
            raise ValueError('缺少原始官方标注')
        previous_tokens = {row['prev'] for row in current.values() if row['prev']}
        previous = {row['token']: row for row in iter_records(annotation_path) if row['token'] in previous_tokens}
        if set(previous) != previous_tokens:
            raise ValueError('标注指定的前驱缺失，不静默降级')
        sample_tokens = {row['sample_token'] for row in [*current.values(), *previous.values()]}
        samples = {row['token']: row for row in iter_records(metadata/'sample.json') if row['token'] in sample_tokens}
        required_instances = {row['instance_token'] for row in current.values()}
        instances = {row['token']: row for row in iter_records(metadata/'instance.json')
            if row['token'] in required_instances}
        categories = {row['token']: row['name'] for row in iter_records(metadata/'category.json')}
        scene_tokens = {row['name']: row['token'] for row in iter_records(metadata/'scene.json')}
        motion = {(row['scene'], row['frame_idx']): row['objects']
            for row in map(json.loads, motion_path.open()) if row['scene'] in scenes}
        methods = list(next(iter(frames.values()))[0]['predictions'])
        logs = sorted({known[scene]['log_token'] for scene in scenes})
        counts = np.zeros((len(logs), 2, len(methods), 7))
        cohorts = np.zeros((len(logs), 2, len(methods), len(GROUPS), 3))
        states, reasons = Counter(), Counter()
        shifts, lags = [], []
        changed_matches = Counter()
        total_frames = total_truth = 0
        with (destination/'frames.jsonl').open('x') as handle:
            for scene, rows in frames.items():
                identities = defaultdict(list)
                log_index = logs.index(known[scene]['log_token'])
                for frame in rows:
                    budget()
                    key = (scene, frame['frame_idx'])
                    camera = plans[key]
                    camera_us = camera['timestamp']
                    if abs(camera_us/1e6-frame['camera_timestamp_s']) > 1e-7:
                        raise ValueError('预测时间与冻结相机时间不一致')
                    old = frame['evaluation_only_official']
                    if [g['annotation_token'] for g in old] != [g['annotation_token'] for g in motion[key]]:
                        raise ValueError('独立运动标签与原GT顺序不一致')
                    aligned, details = [], []
                    for obj in old:
                        annotation = current[obj['annotation_token']]
                        sample = samples[annotation['sample_token']]
                        category = categories[instances[obj['instance_token']]['category_token']]
                        if (sample['token'] != camera['sample_token']
                                or sample['scene_token'] != scene_tokens[scene]
                                or official_category(category) != obj['category']
                                or not np.allclose(annotation['size'], obj['size_wlh'], atol=1e-12, rtol=0)):
                            raise ValueError('原始样本/场景/类别/尺寸与固定对象集合不一致')
                        obj_aligned, detail = interpolate_center(obj, annotation,
                            previous.get(annotation['prev']), samples, camera_us)
                        aligned.append(obj_aligned)
                        details.append(detail)
                        states[detail['state']] += 1
                        reasons[detail['reason']] += 1
                        shifts.append(detail['center_shift_m'])
                        lags.append(detail['current_annotation_lag_ms'])
                    results = {}
                    for c, (condition, truth) in enumerate(zip(CONDITIONS, [old, aligned])):
                        results[condition] = {}
                        for m, method in enumerate(methods):
                            pred = frame['predictions'][method]
                            metric = object_frame_metrics(pred, truth)
                            if c == 0 and metric != frame['metrics'][method]:
                                raise ValueError('原始逐帧指标未精确复现')
                            results[condition][method] = metric
                            pairs = metric['matches']
                            counts[log_index, c, m] += [len(pairs), len(truth), len(pred),
                                sum(p['distance_m'] for p in pairs), 0, 0, metric['duplicate_nodes']]
                            identities[(c, m)].append((frame['frame_idx'],
                                [(pred[p['prediction_index']]['node_id'], truth[p['truth_index']]['instance_token']) for p in pairs]))
                            matched = {p['truth_index']: p['distance_m'] for p in pairs}
                            for t, detail in enumerate(details):
                                groups = [detail['state'], 'motion_'+motion[key][t]['official_motion']]
                                for group in groups:
                                    cohorts[log_index, c, m, GROUPS.index(group)] += [1, int(t in matched), matched.get(t, 0)]
                            if c:
                                old_pairs = results[CONDITIONS[0]][method]['matches']
                                changed_matches[method] += {(p['prediction_index'], p['truth_index']) for p in pairs} != {
                                    (p['prediction_index'], p['truth_index']) for p in old_pairs}
                    handle.write(json.dumps(dict(scene=scene, frame_idx=frame['frame_idx'],
                        source_frame_file=str(original/scene/'frames.jsonl'), camera_timestamp_us=camera_us,
                        annotation_center_alignment=details, aligned_evaluation_only=aligned,
                        methods=results, GT_sent_to_inference=False), ensure_ascii=False, allow_nan=False)+'\n')
                    total_frames += 1
                    total_truth += len(old)
                for (c, m), pairs in identities.items():
                    ids = identity_metrics(pairs)
                    counts[log_index, c, m, 4:6] += [ids['identity_switches'], ids['adjacent_matched_transitions']]
        for name, expected in [*input_sha.items(), *frozen.items()]:
            if digest(name) != expected:
                raise ValueError('重评分期间输入或源码变化')
        result = write_statistics(destination, counts, cohorts, logs, methods)
        result.update(scenes=len(scenes), frames=total_frames, fixed_GT_observations=total_truth,
            original_predictions_unchanged=True, original_frame_metrics_exactly_reproduced=True,
            alignment_states=dict(states), fallback_reasons=dict(reasons),
            center_shift_m=dict(zip(['minimum', 'median', 'p95', 'maximum'],
                np.quantile(shifts, [0, .5, .95, 1]).tolist() if shifts else [None]*4)),
            current_sample_lag_ms=[min(lags), max(lags)] if lags else [None, None],
            changed_assignment_frames=dict(changed_matches),
            raw_records_sha256=digest(destination/'frames.jsonl'), release_sha256=digest(destination/'release.json'),
            limitations=['固定原可见对象清单，只改变评价中心；不是完整相机时刻GT或官方新榜单',
                '插值端点只供离线评价，未来标注从未进入模型或mapper',
                '缺前驱或时间未包围时保留原中心及分母；混合参考中仍有时间近似',
                '线性运动只是参考模型，不能替代真实人工身份、精确轨迹或全套3D盒姿态真值',
                '中心误差条件于代理匹配，分组不报告未定义精确率；开发集已访问',
                '无新前端/训练/导航或GPU调用，不获得锁定、路线或实际到达信用'],
            elapsed_s=time.monotonic()-begun, model_calls=0, new_test_opened=False,
            stage9_complete=False, stage12_complete=False, goal_complete=False)
        dump(destination/'report.json', result)
        dump(destination/'complete.json', dict(frames=total_frames, fixed_GT_observations=total_truth,
            report_sha256=digest(destination/'report.json'), elapsed_s=time.monotonic()-begun, goal_complete=False))
        print(json.dumps(dict(scenes=len(scenes), frames=total_frames, fixed_GT_observations=total_truth,
            states=dict(states), elapsed_s=time.monotonic()-begun, goal_complete=False), ensure_ascii=False))
    except BaseException as error:
        dump(destination/'interrupted.json', dict(error=repr(error), goal_complete=False))
        raise
    finally:
        resource('camera_time_evaluation_end')


if __name__ == '__main__':
    main()
