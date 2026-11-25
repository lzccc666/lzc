"""过去LiDAR开发重建统计；缺测帧不排除，按原采集log配对重采样。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import shutil

import numpy as np

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


COUNT_KEYS = ['true_positive', 'truth', 'predictions', 'duplicate_nodes',
    'identity_switches', 'adjacent_matched_transitions', 'nodes_associated_with_multiple_instances',
    'matched_center_error_sum_m']
METRIC_NAMES = ['recall', 'matched_prediction_precision', 'center_error_m_mean', 'identity_switch_rate']


def metric_values(counts):
    counts = np.asarray(counts, dtype=float)
    pairs = [(0, 1), (0, 2), (7, 0), (4, 5)]
    return np.stack([np.divide(counts[..., a], counts[..., b],
        out=np.full_like(counts[..., a], np.nan), where=counts[..., b] > 0) for a, b in pairs], axis=-1)


def finite_value(number):
    return float(number) if np.isfinite(number) else None


def interval(samples):
    finite = np.asarray(samples)[np.isfinite(samples)]
    return dict(ci95=np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
        valid_resamples=len(finite))


def summarize(input_run, mapping_run, run_id):
    inputs = OUTPUT/'data_manifest'/input_run
    mapping = OUTPUT/'mapping_evaluation'/mapping_run
    complete = json.loads((inputs/'complete.json').read_text())
    converted_release = json.loads((inputs/'release.json').read_text())
    summary = json.loads((mapping/'summary.json').read_text())
    release = json.loads((mapping/'release.json').read_text())
    destination = OUTPUT/'statistics'/run_id
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, destination/'statistics_source.py')
    frozen = dict(mapping_summary_sha256=digest(mapping/'summary.json'),
        mapping_release_sha256=digest(mapping/'release.json'), input_complete_sha256=digest(inputs/'complete.json'),
        source_sha256=digest(__file__), bootstrap=dict(unit='采集log', paired=True, repetitions=4000, seed=17),
        performance_credit=False, new_test_opened=False, goal_complete=False)
    dump(destination/'release.json', frozen)
    camera_frames = {}
    plans = json.loads((inputs/'alignment_manifest.json').read_text())['plans']
    for plan in plans:
        camera_frames[camera_key(plan)] = plan
    scene_logs = {row['scene']: row['log_token'] for row in release['scenes']}
    if ({k[0] for k in camera_frames} != set(scene_logs)
            or summary['scene_count'] != complete['scenes'] or summary['frames'] != complete['cameras']):
        raise ValueError('转换与建图总体不一致，不排除缺测或失败场景')
    count_rows = defaultdict(lambda: np.zeros(len(COUNT_KEYS)))
    per_frame = []
    methods, seen_frames = None, set()
    report_hashes, frame_hashes = {}, {}
    for name, log in scene_logs.items():
        path = mapping/name/'report.json'
        report = json.loads(path.read_text())
        report_hashes[str(path)] = digest(path)
        current_methods = sorted(report['methods'])
        if methods is None:
            methods = current_methods
        if current_methods != methods or report['log_token'] != log:
            raise ValueError('缺失方法或log变化')
        for method, result in report['methods'].items():
            row = np.array([result[k] if k != 'matched_center_error_sum_m'
                else (result['center_error_m_mean'] or 0)*result['true_positive'] for k in COUNT_KEYS], dtype=float)
            count_rows[(log, method, 'all')]+=row
        frames_path = mapping/name/'frames.jsonl'
        frame_hashes[str(frames_path)] = digest(frames_path)
        count = 0
        for frame in map(json.loads, frames_path.open()):
            count += 1
            key = name, frame['frame_idx']
            if key in seen_frames or key not in camera_frames:
                raise ValueError('缺失或重复的帧')
            seen_frames.add(key)
            plan = camera_frames[key]
            group = 'real_past_lidar' if plan['lidar'] else 'missing_past_lidar'
            for method, result in frame['metrics'].items():
                if sorted(frame['metrics']) != methods:
                    raise ValueError('帧方法不完整')
                counts = [result[k] for k in COUNT_KEYS[:4]]+[0, 0, 0,
                    sum(p['distance_m'] for p in result['matches'])]
                count_rows[(log, method, group)] += np.array(counts)
                per_frame.append(dict(scene=name, log_token=log, frame_idx=key[1],
                    method=method, group=group, true_positive=result['true_positive'],
                    truth=result['truth'], predictions=result['predictions'],
                    duplicate_nodes=result['duplicate_nodes'], matched_error_sum_m=counts[-1],
                    GT_sample_minus_camera_s=(plan['camera']['official_sample_timestamp']-plan['camera']['timestamp'])/1e6))
        if count != report['frame_count']:
            raise ValueError('帧数量与场景汇总不同')
    if seen_frames != set(camera_frames):
        raise ValueError('总体不完整，不生成选择性统计')
    logs = sorted(set(scene_logs.values()))
    groups = ['all', 'real_past_lidar', 'missing_past_lidar']
    counts = np.array([[[count_rows[(log, method, group)] for group in groups] for method in methods] for log in logs])
    totals = counts.sum(axis=0)
    rates = metric_values(totals)
    draws = np.random.default_rng(17).integers(len(logs), size=(4000, len(logs)))
    sampled_rates = metric_values(counts[draws].sum(axis=1))
    estimates, differences = {}, []
    baseline = methods.index('original_fusion')
    for j, method in enumerate(methods):
        estimates[method] = {}
        for k, group in enumerate(groups):
            result = {name: (float(totals[j, k, i]) if name.endswith('_sum_m') else int(totals[j, k, i]))
                      for i, name in enumerate(COUNT_KEYS)}
            if group != 'all':
                for name in ['identity_switches', 'adjacent_matched_transitions', 'nodes_associated_with_multiple_instances']:
                    result[name] = None
            for metric_index, metric in enumerate(METRIC_NAMES):
                result[metric] = dict(value=finite_value(rates[j, k, metric_index]),
                    **interval(sampled_rates[:, j, k, metric_index]))
                if j != baseline:
                    difference = rates[j, k, metric_index]-rates[baseline, k, metric_index]
                    differences.append(dict(method=method, baseline='original_fusion', group=group, metric=metric,
                        difference=finite_value(difference),
                        **interval(sampled_rates[:, j, k, metric_index]-sampled_rates[:, baseline, k, metric_index])))
            estimates[method][group] = result
        # 全帧计数应严格等于两种深度状态分组之和；身份统计仅全序列定义。
        if not np.allclose(totals[j, 0, [0, 1, 2, 3, 7]], totals[j, 1:, :].sum(axis=0)[[0, 1, 2, 3, 7]], atol=1e-8, rtol=0):
            raise ValueError('实测/缺测分组没有保留全体分母')
    output = dict(scenes=len(scene_logs), logs=len(logs), frames=len(seen_frames),
        real_past_lidar_frames=sum(p['lidar'] is not None for p in plans),
        missing_past_lidar_frames=sum(p['lidar'] is None for p in plans),
        all_frames_retained=True, methods=estimates, paired_differences=differences,
        source_report_sha256=report_hashes, source_frames_sha256=frame_hashes,
        source_input_release_sha256=digest(inputs/'release.json'), bootstrap=frozen['bootstrap'],
        actual_new_frontend_calls=0, evaluation_limit=converted_release['evaluation_time_limit'],
        group_identity_limit='身份指标仅全连续序列定义，不为按深度可用性筛选后的不连续帧重算身份信用',
        unmatched_prediction_limit='范围内未匹配预测含评测关联不足，不全部等同真实误检',
        actual_arrivals=0, navigation_evaluated=False, independent_test=False, goal_complete=False)
    dump(destination/'report.json', output)
    with (destination/'per_frame_method.csv').open('x') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_frame[0]))
        writer.writeheader()
        writer.writerows(per_frame)
    text = ['# 开发集过去LiDAR重建统计', '',
        f'{len(scene_logs)}场景、{len(logs)}log、{len(seen_frames)}帧；缺测帧留在全部GT分母。', '',
        '|条件|匹配数/GT|预测数|召回|匹配精确率|中心误差(m)|身份变化/连续匹配|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method in methods:
        r = estimates[method]['all']
        def formatted(field, scale=1):
            value = r[field]['value']
            return f'{value*scale:.4f}' if value is not None else 'unknown'
        text.append(f'|{method}|{r["true_positive"]}/{r["truth"]}|{r["predictions"]}|{formatted("recall")}|{formatted("matched_prediction_precision")}|{formatted("center_error_m_mean")}|{r["identity_switches"]}/{r["adjacent_matched_transitions"]}|')
    text += ['', converted_release['evaluation_time_limit'], '',
        '仅开发集对象建图，不提供检索、锁定、路径、正确到达或独占实时性能信用。配对log聚类区间、缺测分组及全部原始计数见report.json。goal_complete=false。', '']
    (destination/'report.md').write_text('\n'.join(text), encoding='utf-8')
    print(json.dumps({k:output[k] for k in ['scenes', 'logs', 'frames', 'real_past_lidar_frames', 'missing_past_lidar_frames', 'goal_complete']}, ensure_ascii=False), flush=True)


def camera_key(plan):
    return plan['camera']['scene'], plan['camera']['frame_idx']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run', required=True)
    parser.add_argument('--mapping-run', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    summarize(args.input_run, args.mapping_run, args.run_id)


if __name__ == '__main__':
    main()
