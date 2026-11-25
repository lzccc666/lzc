"""隔离评价KITTI预测；自定义中心诊断不是官方3D AP。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json

import numpy as np

from conceptgraph.completion.kitti_geometry import assignment, box_overlap
from conceptgraph.completion.kitti_transfer import METHODS
from conceptgraph.completion.stage9 import OUTPUT, digest, dump

CATEGORIES = {'Car': 'car', 'Truck': 'truck', 'Pedestrian': 'pedestrian', 'Cyclist': 'bicycle'}
PRIMARY = ['car', 'truck', 'pedestrian']


def evaluate_frame(predictions, truth, category, method):
    eligible, ignored = [], []
    for obj in truth:
        center = obj['center_camera0']
        if CATEGORIES.get(obj['category']) == category and 1 < center[2] < 65:
            eligible.append(obj | {'evaluation_category': category})
        elif obj['category'] == 'DontCare' or CATEGORIES.get(obj['category']) == category or (
                (category == 'car' and obj['category'] == 'Van') or
                (category == 'pedestrian' and obj['category'] == 'Person_sitting')):
            ignored.append(obj)
    pred = [r | {'evaluation_category': category} for r in predictions if r['category'] == category]
    pairs = assignment(pred, eligible)
    matched = {i for i, _ in pairs}
    # 仅忽略未被正确GT匹配且与忽略框IoA>=0.5的预测，不能隐藏有效GT上的重复。
    ignore_pred = {i for i, row in enumerate(pred) if i not in matched and
                   any(box_overlap(row['bbox'], g['bbox'], ioa=True) >= .5 for g in ignored)
                   and not any(box_overlap(row['bbox'], g['bbox']) >= .5 for g in eligible)}
    valid_pred = [p for i, p in enumerate(pred) if i not in ignore_pred]
    localized = [p | p['methods'][method] for p in valid_pred if p['methods'][method]['center_camera0'] is not None]
    pairs3d = assignment(localized, eligible, metric='distance', gate=3)
    conditional = []
    for i, j in pairs:
        center = pred[i]['methods'][method]['center_camera0']
        if center is not None:
            conditional.append(dict(prediction_index=pred[i]['prediction_index'],
                label_index=eligible[j]['label_index'], error_m=float(np.linalg.norm(
                    np.asarray(center)-eligible[j]['center_camera0'])),
                depth_evidence=pred[i]['methods'][method]['depth_evidence']))
    return dict(category=category, method=method, truth=len(eligible), predictions=len(valid_pred),
        ignored_predictions=len(ignore_pred), matches_2d=len(pairs), localized_predictions=len(localized),
        matches_3d=len(pairs3d), conditional_center_errors=conditional,
        conditional_error_sum=sum(r['error_m'] for r in conditional),
        conditional_localized=len(conditional),
        matched_center_error_sum=sum(float(np.linalg.norm(np.asarray(localized[i]['center_camera0'])-
                                    eligible[j]['center_camera0'])) for i, j in pairs3d))


def aggregate(rows):
    keys = ['truth', 'predictions', 'ignored_predictions', 'matches_2d', 'localized_predictions',
            'matches_3d', 'conditional_error_sum', 'conditional_localized', 'matched_center_error_sum']
    sums = {k: sum(r[k] for r in rows) for k in keys}
    def ratio(a, b):
        return sums[a]/sums[b] if sums[b] else None
    return sums | dict(recall_2d=ratio('matches_2d', 'truth'),
                       precision_2d=ratio('matches_2d', 'predictions'),
                       recall_3d=ratio('matches_3d', 'truth'),
                       precision_3d_all_detections=ratio('matches_3d', 'predictions'),
                       precision_3d_localized=ratio('matches_3d', 'localized_predictions'),
                       conditional_center_error_m=ratio('conditional_error_sum', 'conditional_localized'),
                       matched_center_error_m=ratio('matched_center_error_sum', 'matches_3d'),
                       unknown_depth_predictions=sums['predictions']-sums['localized_predictions'])


def common_supported_errors(records):
    """条件位置差异只在两方法均支持的同一个2D匹配上配对。"""
    groups = defaultdict(dict)
    for row in records:
        if row['category'] not in PRIMARY:
            continue
        for error in row['conditional_center_errors']:
            key = (row['drive'], row['frame_id'], row['category'],
                   error['prediction_index'], error['label_index'])
            groups[key][row['method']] = error['error_m']
    return [dict(drive=k[0], frame_id=k[1], category=k[2], prediction_index=k[3],
                 label_index=k[4], errors_m=v) for k, v in groups.items() if all(m in v for m in METHODS)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run', required=True)
    parser.add_argument('--inference-run', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    source = OUTPUT/'data_manifest'/args.input_run
    inference = OUTPUT/'mapping_evaluation'/args.inference_run
    manifest = json.loads((source/'agent_manifest.json').read_text())
    complete = json.loads((source/'complete.json').read_text())
    if digest(source/'evaluation_only/truth.json') != complete['truth_sha256']:
        raise ValueError('GT SHA校验失败')
    release = json.loads((inference/'release.json').read_text())
    summary = json.loads((inference/'summary.json').read_text())
    if release['agent_manifest_sha256'] != digest(source/'agent_manifest.json'):
        raise ValueError('推理输入不匹配')
    if summary['gt_read'] or summary['cuda_initialized'] or summary['actual_frontend_calls'] != summary['frame_count']:
        raise ValueError('实际推理调用证据不一致')
    truth = {r['frame_id']: r for r in json.loads((source/'evaluation_only/truth.json').read_text())['frames']}
    frames = {r['frame_id']: r for r in manifest['frames']}
    records, evidence_hash = [], {}
    for saved in summary['frames']:
        path = saved['observation']
        if digest(path) != saved['observation_sha256']:
            raise ValueError('推理输出SHA不匹配')
        evidence_hash[path] = saved['observation_sha256']
        row = json.loads(open(path).read())
        gt = truth[row['frame_id']]
        if row['drive'] != gt['drive'] or row['drive'] != frames[row['frame_id']]['drive']:
            raise ValueError('官方drive聚类映射不匹配')
        for category in PRIMARY + ['bicycle']:
            for method in METHODS:
                records.append(dict(frame_id=row['frame_id'], drive=row['drive'],
                    **evaluate_frame(row['predictions'], gt['objects'], category, method)))
    scores = {m: aggregate([r for r in records if r['method'] == m and r['category'] in PRIMARY]) for m in METHODS}
    common = common_supported_errors(records)
    by_drive = defaultdict(list)
    for row in records:
        if row['category'] in PRIMARY:
            by_drive[row['drive']].append(row)
    drives = sorted(by_drive)
    rng = np.random.default_rng(17)
    values = defaultdict(list)
    common_differences = []
    for _ in range(4000):
        sample_drives = rng.choice(drives, len(drives), replace=True)
        sampled = [r for drive in sample_drives for r in by_drive[drive]]
        group = {m: aggregate([r for r in sampled if r['method'] == m]) for m in METHODS}
        for key in ['recall_3d', 'precision_3d_all_detections', 'conditional_center_error_m']:
            a, b = group[METHODS[0]][key], group[METHODS[1]][key]
            if a is not None and b is not None:
                values[key].append(b-a)
        shared = [r for drive in sample_drives for r in common if r['drive'] == drive]
        if shared:
            common_differences.append(float(np.mean([r['errors_m'][METHODS[1]]-r['errors_m'][METHODS[0]] for r in shared])))
    intervals = {k: dict(difference=scores[METHODS[1]][k]-scores[METHODS[0]][k],
        ci95=np.quantile(v, [.025, .975]).tolist(), valid_resamples=len(v)) for k, v in values.items()}
    dest = OUTPUT/'statistics'/args.run_id
    dest.mkdir(parents=True, exist_ok=False)
    dump(dest/'records.json', records)
    dump(dest/'common_supported_pairs.json', common)
    report = dict(dataset='KITTI Object training', role='外部跨数据集开发诊断，非封存测试',
        frame_count=len(summary['frames']), raw_drive_count=len(drives),
        primary_categories=PRIMARY, metrics=scores,
        per_class={c: {m: aggregate([r for r in records if r['category'] == c and r['method'] == m])
                      for m in METHODS} for c in PRIMARY+['bicycle']},
        paired_drive_bootstrap=intervals, bootstrap_seed=17, bootstrap_iterations=4000,
        common_supported_pairs=dict(count=len(common),
            mean_error_m={m: float(np.mean([r['errors_m'][m] for r in common])) if common else None for m in METHODS},
            paired_difference_ci95_m=np.quantile(common_differences, [.025,.975]).tolist() if common_differences else None,
            valid_resamples=len(common_differences), scope='两条件都有深度的相同二维正确匹配，不能代替全GT召回'),
        matching='同类别最大基数一对一；2D IoU>=0.5；3D相机0中心距离<=3m；不是官方KITTI AP',
        inference_sha256=digest(inference/'summary.json'), release_sha256=digest(inference/'release.json'),
        source_complete_sha256=digest(source/'complete.json'), evidence_sha256=evidence_hash,
        source_sha256=digest(__file__), gt_used_for_inference=False,
        pretrained_dataset_overlap='unknown; 基础检测模型预训练历史未证明完全不含KITTI',
        actual_navigation_arrivals=0, actual_scenegraph_fusions=0,
        complete_navigation_test=False, new_sealed_test_opened=False,
        limitations=['单帧，无连续世界位姿、语言目标、车道和控制闭环证据',
                     '稀疏可见表面点均值不等于官方完整三维框中心，须保留定位偏差',
                     '各方法条件误差的支持分母可能不同；共同支持子集另给配对误差，主召回保留全部GT',
                     '所有范围内GT保留，不以漏检、遮挡、低支持或无深度删分母',
                     'Cyclist与bicycle为框语义不完全一致的代理，单列不并入主要三类',
                     'CPU与主矩阵同时运行，不计独占性能结果',
                     '几何标定与像素投影未证明逐点严格早于相机，不作严格因果在线结论'])
    dump(dest/'report.json', report)
    with (dest/'per_frame_class_method.csv').open('x') as handle:
        keys = [k for k in records[0] if k != 'conditional_center_errors']
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: r[k] for k in keys} for r in records)
    lines = ['# KITTI 真实前端跨数据集诊断', '',
        f"{report['frame_count']}帧，{len(drives)}个官方原始采集drive；主要类别为car/truck/pedestrian。", '',
        '|方法|GT|预测|2D匹配|3D匹配|3D召回|3D正确/全部检测|2D匹配且有深度的中心误差(m)|深度未知|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):
        return 'unknown' if x is None else f'{x:.4f}'
    for method, row in scores.items():
        lines.append(f"|{method}|{row['truth']}|{row['predictions']}|{row['matches_2d']}|{row['matches_3d']}|"
            f"{fmt(row['recall_3d'])}|{fmt(row['precision_3d_all_detections'])}|"
            f"{fmt(row['conditional_center_error_m'])}|{row['unknown_depth_predictions']}|")
    lines += ['', '方法差异采用4000次drive配对聚类bootstrap，详见report.json。所有失败、未检出和深度未知保留。',
              f"两方法共同有支持的正确二维匹配共{len(common)}对，配对中心误差与区间另存common_supported_pairs；条件误差不替代端到端召回。",
              '', '实际场景图跨帧融合、规划、目标锁定和到达次数均为0，不能作为KITTI完整导航结果。',
              '', *[f'- {v}' for v in report['limitations']], '',
              '数据来源：https://www.cvlibs.net/datasets/kitti/ ；数据许可证CC BY-NC-SA 3.0。']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(metrics=scores, paired_drive_bootstrap=intervals), ensure_ascii=False))


if __name__ == '__main__':
    main()
