"""开发集全序列身份与间隔返回诊断；完整GT仅用于隔离评价。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import shutil
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from conceptgraph.completion.stage9 import OUTPUT, digest, dump

SUPPORTS = ('all_gated', 'mutually_unique_gated')
ID_COUNTS = ('IDTP', 'IDFN', 'IDFP', 'GT_observations', 'prediction_observations',
             'ambiguous_GT_observations', 'ambiguous_prediction_observations',
             'gated_edges', 'supported_edges')
RETURN_COUNTS = ('official_return_opportunities', 'official_return_both_matched',
                 'official_return_same_id', 'official_return_changed_id',
                 'official_return_unknown_id', 'adjacent_matched_transitions',
                 'adjacent_identity_changes', 'method_gap_returns', 'method_gap_same_id',
                 'method_gap_changed_id', 'GT_instances', 'never_matched_instances',
                 'never_matched_GT_observations', 'unreturned_visible_tail_observations',
                 'censored_tails', 'tails_GT_absent_at_scene_end')


def stable_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value == '':
        raise ValueError('身份必须是非空字符串或整数；未知身份不能擅自连接')
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def prediction_id(obj, method, frame_idx):
    value = stable_id(obj['node_id'])
    return f'frame={frame_idx};id={value}' if method == 'single_frame' else value


def center(obj):
    value = np.asarray(obj['center'], dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all() or not isinstance(obj['category'], str):
        raise ValueError('中心/类别无效，不删除样本')
    return value[:2]


def prepare_sequence(frames, method):
    """不读取或修改任何推理状态；保存的GT、预测和原逐帧匹配均完整保留。"""
    records, previous_time, previous_index = [], -math.inf, -1
    for frame in frames:
        index, timestamp = frame['frame_idx'], float(frame['camera_timestamp_s'])
        if not math.isfinite(timestamp) or timestamp <= previous_time or index != previous_index + 1:
            raise ValueError('帧编号不连续或相机时间不严格递增')
        previous_time, previous_index = timestamp, index
        gt, pred = frame['evaluation_only_official'], frame['predictions'][method]
        gids = [stable_id(obj['instance_token']) for obj in gt]
        pids = [prediction_id(obj, method, index) for obj in pred]
        if len(set(gids)) != len(gids) or len(set(pids)) != len(pids):
            raise ValueError('同帧身份重复；不能重复计数潜在支持')
        xy_gt = np.array([center(obj) for obj in gt]).reshape(-1, 2)
        xy_pred = np.array([center(obj) for obj in pred]).reshape(-1, 2)
        distance = np.linalg.norm(xy_gt[:, None] - xy_pred[None, :], axis=-1)
        categories = np.array([a['category'] for a in gt], dtype=object)[:, None] == np.array(
            [a['category'] for a in pred], dtype=object)[None, :]
        gate = (distance <= 3.0) & categories
        pairs, legacy_pairs = {}, {}
        used_predictions = set()
        for match in frame['metrics'][method]['matches']:
            gi, pi = match['truth_index'], match['prediction_index']
            if not 0 <= gi < len(gt) or not 0 <= pi < len(pred) or not gate[gi, pi]:
                raise ValueError('保存的逐帧匹配超出固定类别/距离门限')
            if gids[gi] in pairs or pi in used_predictions:
                raise ValueError('保存的逐帧匹配不是一对一')
            pairs[gids[gi]] = pids[pi]
            legacy_pairs[gids[gi]] = stable_id(pred[pi]['node_id'])
            used_predictions.add(pi)
        records.append(dict(frame_idx=index, timestamp_s=timestamp, gt_ids=gids, prediction_ids=pids,
                            gate=gate, matched_ids=pairs, legacy_matched_ids=legacy_pairs))
    return records


def identity_counts(records, support):
    """一对一最大累计门限重合；全观测分母不随保守支持条件改变。"""
    if support not in SUPPORTS:
        raise ValueError(support)
    gids = sorted({x for r in records for x in r['gt_ids']})
    pids = sorted({x for r in records for x in r['prediction_ids']})
    gi, pi = {v: i for i, v in enumerate(gids)}, {v: i for i, v in enumerate(pids)}
    potential = np.zeros((len(gids), len(pids)), dtype=np.int64)
    counts = dict.fromkeys(ID_COUNTS, 0)
    for record in records:
        gate = record['gate']
        gdegree, pdegree = gate.sum(axis=1), gate.sum(axis=0)
        counts['GT_observations'] += len(record['gt_ids'])
        counts['prediction_observations'] += len(record['prediction_ids'])
        counts['ambiguous_GT_observations'] += int((gdegree > 1).sum())
        counts['ambiguous_prediction_observations'] += int((pdegree > 1).sum())
        counts['gated_edges'] += int(gate.sum())
        supported = gate if support == SUPPORTS[0] else gate & (gdegree[:, None] == 1) & (pdegree[None, :] == 1)
        counts['supported_edges'] += int(supported.sum())
        for g, p in zip(*np.nonzero(supported)):
            potential[gi[record['gt_ids'][g]], pi[record['prediction_ids'][p]]] += 1
    rows, cols = linear_sum_assignment(potential, maximize=True)
    counts['IDTP'] = int(potential[rows, cols].sum())
    counts['IDFN'] = counts['GT_observations'] - counts['IDTP']
    counts['IDFP'] = counts['prediction_observations'] - counts['IDTP']
    if min(counts['IDFN'], counts['IDFP']) < 0:
        raise AssertionError('身份重合超过全体观测')
    assignments = [dict(GT_instance=gids[g], prediction_identity=pids[p], overlap_observations=int(potential[g, p]))
                   for g, p in zip(rows, cols) if potential[g, p] > 0]
    return dict(counts=counts, assignments=assignments)


def return_diagnostics(records):
    """返回事件依赖原逐帧关联，不用全序列GT分配修补推理身份。"""
    counts = dict.fromkeys(RETURN_COUNTS, 0)
    present = defaultdict(list)
    for i, record in enumerate(records):
        for gt in record['gt_ids']:
            present[gt].append(i)
    events, tails = [], []
    for gt, visible in sorted(present.items()):
        counts['GT_instances'] += 1
        matched = [i for i in visible if gt in records[i]['matched_ids']]
        for left, right in zip(visible, visible[1:]):
            if right - left > 1:
                a, b = records[left]['matched_ids'].get(gt), records[right]['matched_ids'].get(gt)
                status = 'unknown' if a is None or b is None else ('same' if a == b else 'changed')
                elapsed = records[right]['timestamp_s'] - records[left]['timestamp_s']
                events.append(dict(kind='official_list_return', GT_instance=gt, left_frame=left,
                                   right_frame=right, intervening_frames=right-left-1, endpoint_gap_s=elapsed,
                                   gap_bucket='over_2s' if elapsed > 2 else 'at_most_2s',
                                   previous_matched=a is not None, returned_matched=b is not None,
                                   previous_prediction_identity=a, returned_prediction_identity=b, identity_status=status))
                counts['official_return_opportunities'] += 1
                counts['official_return_both_matched'] += int(status != 'unknown')
                counts[f'official_return_{status}_id'] += 1
        for left, right in zip(matched, matched[1:]):
            a, b = records[left]['matched_ids'][gt], records[right]['matched_ids'][gt]
            if right == left+1:
                counts['adjacent_matched_transitions'] += 1
                counts['adjacent_identity_changes'] += int(records[left]['legacy_matched_ids'][gt] !=
                                                          records[right]['legacy_matched_ids'][gt])
                continue
            counts['method_gap_returns'] += 1
            counts['method_gap_same_id' if a == b else 'method_gap_changed_id'] += 1
            absent = sum(gt not in records[i]['gt_ids'] for i in range(left+1, right))
            events.append(dict(kind='method_matched_gap_return', GT_instance=gt, left_frame=left, right_frame=right,
                               intervening_frames=right-left-1, endpoint_gap_s=records[right]['timestamp_s']-records[left]['timestamp_s'],
                               intervening_GT_absent_frames=absent,
                               gap_cause='GT_list_absence_included' if absent else 'GT_list_continuously_present',
                               identity_status='same' if a == b else 'changed',
                               previous_prediction_identity=a, returned_prediction_identity=b))
        if not matched:
            counts['never_matched_instances'] += 1
            counts['never_matched_GT_observations'] += len(visible)
        elif matched[-1] < len(records)-1:
            last = matched[-1]
            later_visible = sum(i > last for i in visible)
            absent_end = visible[-1] < len(records)-1
            counts['censored_tails'] += 1
            counts['tails_GT_absent_at_scene_end'] += int(absent_end)
            counts['unreturned_visible_tail_observations'] += later_visible
            tails.append(dict(GT_instance=gt, last_matched_frame=last, last_GT_visible_frame=visible[-1],
                              scene_end_frame=len(records)-1, unmatched_visible_observations=later_visible,
                              GT_absent_at_scene_end=absent_end,
                              limit='无后续匹配的尾段；场景边界/官方列表缺席不证明永久丢失'))
    return dict(counts=counts, events=events, censored_tails=tails)


def metric_values(counts):
    def ratio(a, b):
        return a / b if b else None
    return dict(IDR=ratio(counts['IDTP'], counts['GT_observations']),
                IDP=ratio(counts['IDTP'], counts['prediction_observations']),
                IDF1=ratio(2*counts['IDTP'], counts['GT_observations']+counts['prediction_observations']))


def rates(counts):
    result = metric_values(counts)
    for name, numerator, denominator in [
        ('adjacent_identity_change_rate', 'adjacent_identity_changes', 'adjacent_matched_transitions'),
        ('official_return_both_matched_rate', 'official_return_both_matched', 'official_return_opportunities'),
        ('official_return_same_id_rate', 'official_return_same_id', 'official_return_opportunities'),
        ('official_return_changed_id_rate', 'official_return_changed_id', 'official_return_opportunities'),
        ('official_return_unknown_id_rate', 'official_return_unknown_id', 'official_return_opportunities')]:
        result[name] = counts[numerator]/counts[denominator] if counts[denominator] else None
    return result


def aggregate_statistics(rows, repetitions=4000, seed=17):
    logs = sorted({r['log_token'] for r in rows})
    methods = sorted({r['method'] for r in rows})
    grouped = defaultdict(Counter)
    for row in rows:
        grouped[row['log_token'], row['method'], row['support']].update(row['counts'])
    keys = ID_COUNTS + RETURN_COUNTS
    draws = np.random.default_rng(seed).integers(len(logs), size=(repetitions, len(logs)))
    results, differences, samples = {}, [], {}
    for method in methods:
        results[method] = {}
        for support in SUPPORTS:
            values = np.array([[grouped[log, method, support][k] for k in keys] for log in logs])
            total = dict(zip(keys, map(int, values.sum(axis=0))))
            estimates = rates(total)
            boot = defaultdict(list)
            for sample in values[draws].sum(axis=1):
                for metric, value in rates(dict(zip(keys, sample))).items():
                    boot[metric].append(np.nan if value is None else value)
            samples[method, support] = {metric: np.array(v) for metric, v in boot.items()}
            results[method][support] = dict(counts=total, metrics={metric: dict(value=value,
                **interval(samples[method, support][metric])) for metric, value in estimates.items()})
    for method in methods:
        if method == 'original_fusion':
            continue
        for support in SUPPORTS:
            for metric, estimate in results[method][support]['metrics'].items():
                base = results['original_fusion'][support]['metrics'][metric]['value']
                differences.append(dict(method=method, baseline='original_fusion', support=support, metric=metric,
                    difference=estimate['value']-base if estimate['value'] is not None and base is not None else None,
                    **interval(samples[method, support][metric]-samples['original_fusion', support][metric])))
    by_log = [dict(log_token=log, method=method, support=support,
                   **{k: grouped[log, method, support][k] for k in keys})
              for log in logs for method in methods for support in SUPPORTS]
    return dict(logs=len(logs), methods=results, paired_differences=differences,
                bootstrap=dict(unit='采集log', repetitions=repetitions, seed=seed, paired=True)), by_log


def interval(samples):
    valid = samples[np.isfinite(samples)]
    return dict(ci95=np.quantile(valid, [.025, .975]).tolist() if len(valid) else None, valid_resamples=len(valid))


def run(mapping_run, run_id, scene_limit=None):
    started = time.monotonic()
    mapping = OUTPUT/'mapping_evaluation'/mapping_run
    release = json.loads((mapping/'release.json').read_text())
    scenes = release['scenes'][:scene_limit]
    destination = OUTPUT/'statistics'/run_id
    if shutil.disk_usage(OUTPUT).free < 6*1024**3:
        raise RuntimeError('磁盘余量低于6GiB预算')
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, destination/'evaluation_source.py')
    dump(destination/'release.json', dict(mapping_run=mapping_run, source_sha256=digest(__file__),
        mapping_release_sha256=digest(mapping/'release.json'), mapping_summary_sha256=digest(mapping/'summary.json'),
        protocol_sha256=digest(OUTPUT/'task_protocol/temporal_identity_001.md'),
        scene_limit=scene_limit, budget=dict(cpu_wall_s=900, bytes=1024**3, minimum_free_bytes=6*1024**3),
        independent_test=False, new_test_opened=False, ground_truth_in_inference=False, goal_complete=False))
    rows, hashes, methods, frames_total = [], {}, None, 0
    gap_groups = defaultdict(Counter)
    with (destination/'scene_method_support.jsonl').open('x') as output:
        for scene in scenes:
            if time.monotonic()-started > 900:
                raise TimeoutError('全序列身份诊断超出900秒预算，保留部分结果')
            path = mapping/scene['scene']/'frames.jsonl'
            hashes[str(path)] = digest(path)
            frames = [json.loads(line) for line in path.open()]
            current_methods = sorted(frames[0]['predictions'])
            methods = current_methods if methods is None else methods
            if any(sorted(f['predictions']) != methods or f['scene'] != scene['scene'] for f in frames):
                raise ValueError('方法/场景不完整')
            frames_total += len(frames)
            for method in methods:
                records = prepare_sequence(frames, method)
                returns = return_diagnostics(records)
                for event in returns['events']:
                    if event['kind'] == 'official_list_return':
                        group = gap_groups[method, event['gap_bucket']]
                        group['opportunities'] += 1
                        group[event['identity_status']] += 1
                dump(destination/scene['scene']/f'{method}_returns.json', returns)
                for support in SUPPORTS:
                    result = identity_counts(records, support)
                    dump(destination/scene['scene']/f'{method}_{support}.json', result)
                    row = dict(scene=scene['scene'], log_token=scene['log_token'], method=method, support=support,
                               counts=dict(**result['counts'], **returns['counts']))
                    rows.append(row)
                    output.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            if digest(path) != hashes[str(path)]:
                raise ValueError('运行期间输入帧文件发生变化')
    summary = json.loads((mapping/'summary.json').read_text())
    if scene_limit is None and (frames_total != summary['frames'] or len(scenes) != summary['scene_count']):
        raise ValueError('全量总体与原建图不一致')
    statistics, by_log = aggregate_statistics(rows)
    for method in methods:
        a, b = (statistics['methods'][method][s]['counts'] for s in SUPPORTS)
        if a['GT_observations'] != b['GT_observations'] or a['prediction_observations'] != b['prediction_observations']:
            raise AssertionError('保守支持条件改变了分母')
        legacy = Counter()
        for scene in scenes:
            original = json.loads((mapping/scene['scene']/'report.json').read_text())['methods'][method]
            legacy.update({key: original[key] for key in ('identity_switches', 'adjacent_matched_transitions')})
        if (a['adjacent_identity_changes'] != legacy['identity_switches'] or
                a['adjacent_matched_transitions'] != legacy['adjacent_matched_transitions']):
            raise ValueError('旧相邻匹配身份指标未严格复现')
    report = dict(**statistics, scenes=len(scenes), frames=frames_total, source_frames_sha256=hashes,
        elapsed_cpu_shared_wall_s=time.monotonic()-started, actual_new_frontend_calls=0, model_calls=0,
        return_matching_source='固定保存的逐帧3米同类别一对一关联；不由全序列GT分配替换',
        official_return_gap_groups=[dict(method=m, gap_bucket=b, **dict(c)) for (m, b), c in sorted(gap_groups.items())],
        reference='TrackEval Identity框架的XY中心3米门限适配；不是官方nuScenes/MOT成绩',
        limitations=['开发后验诊断；IDF1受检测覆盖影响，不能独自证明身份稳定性提升',
            '官方GT采样比相机晚约33–42毫秒，中心评价保留时间近似',
            '列表缺席不等于确定遮挡；自动中心关联不等于人工确认实例/错合并',
            '单帧基线节点按帧隔离；未知返回不冒充错误或成功'],
        independent_test=False, ground_truth_in_inference=False, actual_arrivals=0, goal_complete=False)
    dump(destination/'report.json', report)
    with (destination/'by_log.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(by_log[0]))
        writer.writeheader()
        writer.writerows(by_log)
    dump(destination/'complete.json', dict(scenes=len(scenes), frames=frames_total, methods=len(methods),
        identity_cases=len(rows), report_sha256=digest(destination/'report.json'), goal_complete=False))
    print(json.dumps(dict(scenes=len(scenes), frames=frames_total, cases=len(rows), wall_s=time.monotonic()-started,
                          goal_complete=False), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping-run', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--scene-limit', type=int)
    args = parser.parse_args()
    run(args.mapping_run, args.run_id, args.scene_limit)


if __name__ == '__main__':
    main()
