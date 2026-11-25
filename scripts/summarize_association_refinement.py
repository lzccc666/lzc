"""冻结关联候选的完整配对核对；未完成全范围时拒绝统计。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from conceptgraph.completion.config_comparison import ratios, summarize  # noqa: E402
from conceptgraph.completion.metric_acceptance import paired_metric_uncertainty  # noqa: E402
from conceptgraph.completion.quality import object_frame_metrics  # noqa: E402

REPORT_ROOT = PROJECT / 'outputs/reports/metric_optimization_20261123/association_optimization'
RUN_ROOT = PROJECT / 'outputs/results/nuscenes/validation/iterations'
CLASSES = {'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'pedestrian'}
DEFAULT_RUNS = {
    'R1': 'recall_refinement_R1_full_20261120_v2',
    'R3': 'association_R3_full_20261123_v1',
    'R3_single_stage': 'association_R3_single_stage_full_20261123_v1',
}
COMPARISONS = [['R1', 'R3'], ['R1', 'R3_single_stage'], ['R3_single_stage', 'R3']]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_numbers(actual, expected, label):
    if actual.keys() != expected.keys():
        raise ValueError(f'{label}字段不同')
    for key, value in actual.items():
        other = expected[key]
        if value is None or other is None:
            equal = value is other
        elif isinstance(value, float) or isinstance(other, float):
            equal = math.isclose(value, other, rel_tol=1e-12, abs_tol=1e-9)
        else:
            equal = value == other
        if not equal:
            raise ValueError(f'{label}/{key}未复现：{value!r} != {other!r}')


def validate_scope(report, release, method):
    if (report['condition'] != method or report['full_validation_scope'] is not True
            or (report['scenes'], report['logs'], report['frames']) != (38, 11, 1508)
            or report['result']['counts']['truth'] != 5723
            or release['threshold_m'] != 3 or release['seed'] != 17
            or release['camera_shape_wh'] != [1600, 900] or release['pilot'] is not False
            or release['ground_truth_in_inference'] is not False
            or release['test_accessed'] is not False):
        raise ValueError(f'{method}全范围、种子、相机或原协议不完整')
    scope = [(row['scene'], row['log_token']) for row in release['scenes']]
    if len(scope) != 38 or len({s for s, _ in scope}) != 38 or len({log for _, log in scope}) != 11:
        raise ValueError(f'{method}场景或日志清单不完整/重复')
    return scope


def validate_manifest(manifest):
    if (manifest['registered_comparisons'] != COMPARISONS
            or manifest['primary_metric'] != 'IDF1'
            or manifest['bootstrap_seed'] != 20261123
            or manifest['bootstrap_replicates'] != 10000
            or manifest['ground_truth_in_inference'] is not False
            or manifest['test_accessed'] is not False):
        raise ValueError('预登记主比较/随机种子/评价隔离不对应')


def nearest(predictions, truth):
    matches = [(float(np.linalg.norm(np.asarray(p['center'][:2]) - truth['center'][:2])), i)
               for i, p in enumerate(predictions) if p['category'] == truth['category']]
    if not matches:
        return None
    distance, index = min(matches)
    return dict(distance_m=distance, prediction_index=index,
                prediction=predictions[index], caveat='最近同类不保证对应此GT身份')


def threshold_summary(counter):
    tp, gt, pred = (counter[k] for k in ('true_positive', 'truth', 'predictions'))
    return dict(TP=tp, FP=pred-tp, FN=gt-tp, GT=gt, predictions=pred,
                precision=tp/pred if pred else None, recall=tp/gt if gt else None,
                F1=2*tp/(pred+gt) if pred+gt else None)


def describe_predictions(by_scene, method):
    """冻结输出的分布诊断；输出首次出现不冒充内部轨迹出生。"""
    statuses = Counter()
    slices = defaultdict(lambda: defaultdict(Counter))
    node_instances, instance_nodes = defaultdict(set), defaultdict(set)
    unique_nodes = set()
    stale_single_unknown = defaultdict(Counter)
    maximum_age_frames = maximum_age_s = 0.
    for scene, rows in by_scene.items():
        seen = set()
        for row in rows:
            truth, predictions = row['evaluation_only_official'], row['predictions'][method]
            matches = {m['prediction_index']: m for m in row['metrics'][method]['matches']}
            for index, prediction in enumerate(predictions):
                node = json.dumps(prediction['node_id'], sort_keys=True)
                current_key = (scene, node)
                unique_nodes.add(current_key)
                if index in matches:
                    status = 'matched'
                    instance = truth[matches[index]['truth_index']]['instance_token']
                    node_instances[current_key].add(instance)
                    instance_nodes[scene, instance].add(node)
                else:
                    within_gate = any(p['category'] == prediction['category'] and
                        np.linalg.norm(np.asarray(prediction['center'][:2])-p['center'][:2]) <= 3.
                        for p in truth)
                    status = 'unmatched_duplicate_within_3m' if within_gate else 'unmatched_outside_same_class_gates'
                statuses[status] += 1
                raw_timestamp = prediction.get('timestamp_s')
                if (raw_timestamp is not None and math.isfinite(raw_timestamp)
                        and row['camera_timestamp_s']-raw_timestamp > 2.
                        and prediction.get('num_detections') == 1
                        and prediction.get('motion_state') == 'unknown'):
                    sparse_only = (prediction.get('shape_available') is False
                                   or prediction.get('position_source') == 'confirmed_sparse_lidar')
                    origin = 'sparse_position_only' if sparse_only else 'dense_pointcloud_backed'
                    stale_single_unknown[origin][status] += 1
                recency = 'unknown_last_seen_frame'
                if 'last_seen_frame' in prediction:
                    age = row['frame_idx']-prediction['last_seen_frame']
                    maximum_age_frames = max(maximum_age_frames, age)
                    recency = 'measured_current_frame' if age == 0 else 'older_measurement' if age > 0 else 'invalid_future_frame'
                age_s = prediction.get('prediction_age_s')
                if age_s is None and 'timestamp_s' in prediction:
                    age_s = row['camera_timestamp_s']-prediction['timestamp_s']
                age_bucket = 'unknown_age'
                if age_s is not None and math.isfinite(age_s):
                    maximum_age_s = max(maximum_age_s, age_s)
                    age_bucket = ('current_le_1ms' if 0 <= age_s <= .001 else
                                  'older_le_0.6s' if 0 < age_s <= .6 else
                                  'older_le_2s' if .6 < age_s <= 2 else
                                  'older_over_2s' if age_s > 2 else 'invalid_negative_age')
                observations = prediction.get('num_detections')
                observation_bucket = ('unknown' if observations is None else 'one'
                                      if observations == 1 else 'two' if observations == 2 else 'three_plus')
                fields = dict(position_source=prediction.get('position_source', 'original_geometry'),
                    motion_state=prediction.get('motion_state', 'not_recorded'),
                    measurement_recency=recency, prediction_age=age_bucket,
                    detection_observations=observation_bucket,
                    output_history='first_output_of_node' if node not in seen else 'previously_output_node')
                for key, value in fields.items():
                    slices[status][key][value] += 1
                seen.add(node)
    return dict(output_observations=sum(statuses.values()), by_output_status=dict(statuses),
        by_status_and_field={status: {key: dict(c) for key, c in fields.items()}
                             for status, fields in slices.items()},
        unique_scene_scoped_output_nodes=len(unique_nodes),
        nodes_matching_multiple_official_instances=sum(len(s) > 1 for s in node_instances.values()),
        matched_official_instances_with_multiple_output_nodes=sum(len(s) > 1 for s in instance_nodes.values()),
        matched_official_instances=len(instance_nodes),
        total_output_nodes_over_matched_instances=sum(len(s) for s in instance_nodes.values()),
        single_unknown_timestamp_over_2s_by_source_and_status={
            origin: dict(c) for origin, c in stale_single_unknown.items()},
        maximum_recorded_measurement_age_frames=maximum_age_frames,
        maximum_prediction_age_s=maximum_age_s,
        caveat='均为已输出预测观测及自动3米匹配诊断；首次输出不等于内部新建，'
               '多节点/多实例映射不等于人工确认碎片化/误合并。按shape_available/position_source分稀疏位置与稠密几何，'
               '不能按node_id前缀，因为稳定身份交接可给稠密对象分配sparse前缀。未记录字段保持unknown，不补造。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run', default=DEFAULT_RUNS['R1'])
    parser.add_argument('--candidate-run', default=DEFAULT_RUNS['R3'])
    parser.add_argument('--ablation-run', default=DEFAULT_RUNS['R3_single_stage'])
    parser.add_argument('--manifest', type=Path, default=REPORT_ROOT / 'start_manifest.json')
    parser.add_argument('--output', type=Path, default=REPORT_ROOT / 'comparison_v1')
    parser.add_argument('--wall-budget-s', type=float, default=900)
    args = parser.parse_args()
    begun = time.monotonic()
    manifest = read(args.manifest)
    validate_manifest(manifest)
    runs = dict(R1=args.baseline_run, R3=args.candidate_run, R3_single_stage=args.ablation_run)
    if runs['R1'] != manifest['baseline_run']:
        raise ValueError('R1基点与预登记不同')
    paths = {m: RUN_ROOT / run for m, run in runs.items()}
    reports, releases, scope, input_list = {}, {}, None, None
    # 三组完整结束后才创建报告目录；部分输出不进入统计。
    for method, path in paths.items():
        if not (path / 'report.json').is_file() or (path / 'interrupted.json').exists():
            raise ValueError(f'{method}尚未完成全范围：{path}')
        reports[method], releases[method] = read(path / 'report.json'), read(path / 'release.json')
        current_scope = validate_scope(reports[method], releases[method], method)
        if method != 'R1' and any(releases[method].get(key) != value for key, value in (
                ('bootstrap_seed', 20261123), ('paired_bootstrap_replicates', 10000),
                ('comparisons_registered', 3), ('primary_metric', 'IDF1'),
                ('comparison_family', 'association_optimization_20261123'))):
            raise ValueError('新候选release与本轮预登记不一致：' + method)
        current_inputs = read(path / 'input_fingerprints.json')
        if len(current_inputs) != 4600:
            raise ValueError(f'{method}完整4600条输入指纹清单缺失')
        if scope is None:
            scope, input_list = current_scope, current_inputs
        elif scope != current_scope or input_list != current_inputs:
            raise ValueError('三组场景/日志或逐帧输入清单不同')
        for key in ('inputs_sha256', 'seed', 'threshold_m', 'camera_shape_wh', 'config'):
            if releases[method][key] != releases['R1'][key]:
                raise ValueError('冻结主配置或评价输入不同：' + key)
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT / 'outputs/reports'):
        raise ValueError('跨实验报告必须位于outputs/reports')
    if shutil.disk_usage(PROJECT).free < 20 * 1024**3:
        raise OSError('磁盘低于20GiB，停止新增报告')
    output.mkdir(parents=True, exist_ok=False)
    fingerprints = {}

    def freeze(path, expected=None):
        value = sha(path)
        if expected is not None and value != expected:
            raise ValueError('来源指纹不对应：' + str(path))
        fingerprints[str(Path(path).resolve())] = value
        return value

    def guard():
        if time.monotonic() - begun > args.wall_budget_s:
            raise TimeoutError('达到汇总墙钟预算')
        if shutil.disk_usage(PROJECT).free < 20 * 1024**3:
            raise OSError('达到20GiB磁盘保护线')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > 63*1024**2:
            raise OSError('达到64MiB报告预算')

    try:
        freeze(args.manifest)
        freeze(args.manifest.parent / '执行计划.md')
        source_files = [Path(__file__).resolve()] + [PROJECT / p for p in (
            'conceptgraph/completion/quality.py', 'conceptgraph/completion/association.py',
            'conceptgraph/completion/config_comparison.py',
            'conceptgraph/completion/metric_acceptance.py',
            'conceptgraph/completion/temporal_identity_evaluation.py')]
        for source in source_files:
            freeze(source)
            target = output / 'source_snapshot' / source.relative_to(PROJECT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        write(output / 'start_manifest.json', dict(
            runs=runs, primary_metric='IDF1', comparisons=COMPARISONS,
            bootstrap_seed=20261123, replicates=10000, correction_comparisons=3,
            wall_budget_s=args.wall_budget_s, storage_limit_bytes=64*1024**2,
            sensors_read=False, new_inference=False, predictions_modified=False))
        records, scene_reports = defaultdict(dict), defaultdict(list)
        reproduction, table, group_counts = [], [], defaultdict(Counter)
        audit_counts, sensitivity = defaultdict(Counter), defaultdict(Counter)
        for method, path in paths.items():
            for filename in ('report.json', 'release.json', 'input_fingerprints.json'):
                freeze(path / filename)
            for relative in ('quality.py', 'association.py', 'config_comparison.py',
                             'temporal_identity_evaluation.py'):
                source = PROJECT / 'conceptgraph/completion' / relative
                frozen = path / 'source_snapshot' / source.relative_to(PROJECT)
                freeze(frozen, releases[method]['sources_sha256'][str(source)])
                if sha(source) != sha(frozen):
                    raise ValueError('当前匹配/身份汇总实现与原冻结源码不同')
            for scene, log in scope:
                guard()
                saved = read(path / scene / 'report.json')
                freeze(path / scene / 'report.json')
                frame_path = path / scene / 'frames.jsonl.gz'
                freeze(frame_path, saved['records_sha256'])
                with gzip.open(frame_path, 'rt') as stream:
                    frames = list(map(json.loads, stream))
                if (len(frames) != saved['frames'] or saved['scene'] != scene or saved['log'] != log
                        or len({r['frame_idx'] for r in frames}) != len(frames)):
                    raise ValueError('逐场景帧数/日志/帧索引不同')
                for row in frames:
                    if row['gt_in_inference'] is not False or row['scene'] != scene:
                        raise ValueError('推理隔离或场景标识无效')
                    predictions, truth = row['predictions'][method], row['evaluation_only_official']
                    if any(obj['category'] not in CLASSES for obj in predictions + truth):
                        raise ValueError('原六类之外对象，不能删样本')
                    measured = object_frame_metrics(predictions, truth, threshold_m=3.)
                    if measured != row['metrics'][method]:
                        raise ValueError(f'原3米逐帧匹配未复现：{method}/{scene}/{row["frame_idx"]}')
                    for event in row['audit']:
                        audit_counts[method][event['stage'] + ':' + event['reason']] += 1
                    for threshold in (1., 2.):
                        values = object_frame_metrics(predictions, truth, threshold_m=threshold)
                        sensitivity[method, threshold].update({k: values[k] for k in
                            ('true_positive', 'truth', 'predictions')})
                recomputed = summarize(frames, method)
                exact_numbers(recomputed['counts'], saved['result']['counts'], f'{method}/{scene}/counts')
                exact_numbers(recomputed['metrics'], saved['result']['metrics'], f'{method}/{scene}/metrics')
                if not math.isclose(sum(r['mapping_s'] for r in frames), saved['mapping_s'], abs_tol=1e-8):
                    raise ValueError('逐帧与场景mapping_s不同')
                records[method][scene] = frames
                scene_reports[method].append(saved)
            totals = {key: sum(s['result']['counts'][key] for s in scene_reports[method])
                      for key in scene_reports[method][0]['result']['counts']}
            exact_numbers(totals, reports[method]['result']['counts'], method + '/total counts')
            exact_numbers(ratios(totals), reports[method]['result']['metrics'], method + '/total metrics')
            if sum(s['frames'] for s in scene_reports[method]) != 1508:
                raise ValueError('完整帧分母变化')
            if not math.isclose(sum(s['mapping_s'] for s in scene_reports[method]),
                                reports[method]['mapping_s'], abs_tol=1e-8):
                raise ValueError('总运行时间与逐场景不同')
            reproduction.append(dict(method=method, frames=1508, scenes=38,
                exact_frame_metrics=True, scene_counts_and_IDF1=True, total_counts_and_IDF1=True))
            table.append(dict(method=method, scenes=38, logs=11, frames=1508,
                TP=totals['true_positive'], GT=totals['truth'], predictions=totals['predictions'],
                FP=totals['predictions']-totals['true_positive'], FN=totals['truth']-totals['true_positive'],
                **ratios(totals), duplicates=totals['duplicate_nodes'],
                mapping_s=reports[method]['mapping_s'], wall_s=reports[method]['wall_s'],
                peak_rss_mib=reports[method]['peak_rss_mib']))
        changes = {f'{b}-{a}': [] for a, b in COMPARISONS}
        change_counts = {key: Counter() for key in changes}
        observation_audit = {key: dict(input_filter_event_difference_frames=[],
            geometry_accepted_count_difference_frames=[], geometry_accepted_event_difference_frames=[])
            for key in changes}
        for scene, _ in scope:
            guard()
            meta_paths = [Path(p) for p in releases['R1']['inputs_sha256']
                          if Path(p).name == 'frames.jsonl' and Path(p).parent.name == scene]
            if len(meta_paths) != 1:
                raise ValueError('原场景位姿元数据不唯一')
            meta = meta_paths[0]
            freeze(meta, releases['R1']['inputs_sha256'][str(meta)])
            with meta.open() as stream:
                metadata = list(map(json.loads, stream))
            original = {r['frame_idx']: r for r in metadata}
            if len(original) != len(metadata) or len(metadata) != len(records['R1'][scene]):
                raise ValueError('原元数据场景帧范围不同')
            for grouped in zip(*(records[m][scene] for m in paths), strict=True):
                rows = dict(zip(paths, grouped, strict=True))
                base = rows['R1']
                index, truth = base['frame_idx'], base['evaluation_only_official']
                for row in rows.values():
                    for key in ('scene', 'frame_idx', 'camera_timestamp_s', 'evaluation_only_official', 'input'):
                        if row[key] != base[key]:
                            raise ValueError(f'三方法逐帧输入/GT不一致：{scene}/{index}/{key}')
                for left, right in COMPARISONS:
                    key = f'{right}-{left}'
                    selected = {}
                    for method in (left, right):
                        selected[method] = dict(
                            filtered=[e for e in rows[method]['audit'] if e['stage'] == 'input_filter'],
                            accepted=[e for e in rows[method]['audit'] if
                                      (e['stage'], e['reason']) == ('geometry', 'accepted')])
                    loc = dict(scene=scene, frame_idx=index)
                    if selected[left]['filtered'] != selected[right]['filtered']:
                        observation_audit[key]['input_filter_event_difference_frames'].append(loc)
                    if len(selected[left]['accepted']) != len(selected[right]['accepted']):
                        observation_audit[key]['geometry_accepted_count_difference_frames'].append(loc)
                    if selected[left]['accepted'] != selected[right]['accepted']:
                        observation_audit[key]['geometry_accepted_event_difference_frames'].append(loc)
                if truth != original[index]['evaluation_only_official']:
                    raise ValueError('保存评价GT与原输入不同')
                pose = np.asarray(original[index]['camera_pose'])
                matched = {m: {x['truth_index']: x for x in row['metrics'][m]['matches']}
                           for m, row in rows.items()}
                for i, obj in enumerate(truth):
                    distance = float(np.linalg.norm(np.asarray(obj['center'][:2]) - pose[:2, 3]))
                    bucket = '0_20m' if distance < 20 else '20_40m' if distance < 40 else '40m_plus'
                    for method in paths:
                        for group in ('category:' + obj['category'], 'range:' + bucket):
                            group_counts[method, group]['truth'] += 1
                            group_counts[method, group]['matched'] += i in matched[method]
                    for left, right in COMPARISONS:
                        key = f'{right}-{left}'
                        a, b = i in matched[left], i in matched[right]
                        kind = 'both' if a and b else 'gain' if b else 'loss' if a else 'neither'
                        change_counts[key][kind] += 1
                        if kind in {'gain', 'loss'}:
                            changes[key].append(dict(kind=kind, scene=scene, frame_idx=index,
                                truth_index=i, category=obj['category'], instance=obj['instance_token'],
                                official_center=obj['center'], range_m=distance, input=base['input'],
                                source_records={m: str(paths[m] / scene / 'frames.jsonl.gz') for m in paths},
                                saved_matches={m: matched[m].get(i) for m in paths},
                                nearest_same_class={m: nearest(row['predictions'][m], obj)
                                                    for m, row in rows.items()}))
        comparisons = {}
        for left, right in COMPARISONS:
            uncertainty = paired_metric_uncertainty(scene_reports[left], scene_reports[right],
                seed=manifest['bootstrap_seed'], replicates=manifest['bootstrap_replicates'])
            uncertainty['comparison_family'] = 'association_optimization_20261123'
            uncertainty['interpretation'] = (
                '本轮三项关联比较独立预登记，IDF1为主；附指标为描述性区间，'
                '不构成六指标联合错误率控制；不更改旧R1-B1三配置统计。'
                '中心误差的匹配样本组成可不同。')
            a, b = reports[left]['result'], reports[right]['result']
            delta = {k: 100*(b['metrics'][k]-a['metrics'][k]) for k in
                     ('recall', 'precision', 'F1', 'IDF1')}
            ci = uncertainty['intervals']['IDF1']['ci_bonferroni']
            comparisons[f'{right}-{left}'] = dict(baseline=left, candidate=right,
                primary_metric='IDF1', statistics=uncertainty,
                primary_corrected_IDF1_positive=ci is not None and ci[0] > 0,
                point_delta_percentage_points=delta,
                point_metrics_non_decreasing=all(v >= 0 for v in delta.values()),
                duplicates_non_increasing=b['counts']['duplicate_nodes'] <= a['counts']['duplicate_nodes'],
                duplicate_delta=b['counts']['duplicate_nodes']-a['counts']['duplicate_nodes'],
                center_error_delta_m=b['metrics']['center_error_m']-a['metrics']['center_error_m'],
                paired_observation_changes=dict(change_counts[f'{right}-{left}']),
                adoption_decision='仅报告显著性、点估计和护栏，不自动采用或切换默认')
        groups = [dict(method=m, group=g, matched=c['matched'], truth=c['truth'],
                       recall=c['matched']/c['truth']) for (m, g), c in sorted(group_counts.items())]
        for method in paths:
            for prefix in ('category:', 'range:'):
                if sum(r['truth'] for r in groups if r['method'] == method and r['group'].startswith(prefix)) != 5723:
                    raise ValueError('类别/距离分组未保留完整GT')
        supplementary = [dict(method=m, threshold_m=t, protocol='补充位置评价，不重算身份',
            **threshold_summary(sensitivity[m, t])) for t in (1., 2.) for m in paths]
        prediction_diagnostics = {method: describe_predictions(records[method], method) for method in paths}
        for method, diagnostics in prediction_diagnostics.items():
            c = reports[method]['result']['counts']
            if (diagnostics['output_observations'] != c['predictions']
                    or diagnostics['by_output_status'].get('matched', 0) != c['true_positive']
                    or diagnostics['by_output_status'].get('unmatched_duplicate_within_3m', 0)
                    != c['duplicate_nodes']):
                raise ValueError('预测分布诊断未复现原输出分母/匹配/重复节点')
        write_csv(output / 'main_metrics.csv', table)
        write_csv(output / 'group_recall.csv', groups)
        write_csv(output / 'supplementary_1m_2m.csv', supplementary)
        write(output / 'reproduction_3m.json', dict(passed=True, methods=reproduction))
        write(output / 'all_gain_loss_cases.json', changes)
        write(output / 'audit_event_counts.json', {m: dict(c) for m, c in audit_counts.items()})
        write(output / 'prediction_failure_diagnostics.json', prediction_diagnostics)
        write(output / 'observation_audit_comparisons.json', dict(comparisons=observation_audit,
            per_method_input_filter_and_geometry_counts={m: {k: v for k, v in counts.items()
                if k.startswith(('input_filter:', 'geometry:'))} for m, counts in audit_counts.items()},
            caveat='输入缓存一致不保证历史深度分簇后观测相同；这里同时比较接受事件计数与完整事件内容，'
                   '事件内容差异本身不等于已证明观测中心变化。'))
        caveats = ['38场景是验证/调试，不是独立测试或车辆闭环',
            '3米是原水平XY中心同类一对一评价，不证明高度、轮廓或到达',
            '1/2米所有方法同阈值，仅补充位置敏感性，不用于调参或重定义主检验',
            'IDF1为预登记主指标；其他指标区间描述性，不控制六指标联合错误率',
            '中心误差仅针对各方法已匹配对象，匹配组成不同',
            '运行时间来自历史独立运行，非本次随机交错硬件基准',
            '逐帧重新计算指标一致，不等于新候选已完成推理确定性复现',
            'audit与输出分布只能定位关联拒配/新建/确认/陈旧输出的线索，不能单独作因果归因',
            '保留原R1-B1三配置校正，本次三关联比较独立登记，不改旧结论']
        write(output / 'comparison.json', dict(primary_metric='IDF1', main_threshold_m=3,
            registered_manifest=str(args.manifest.resolve()), runs={m: str(p) for m, p in paths.items()},
            scope=dict(scenes=38, logs=11, frames=1508, GT=5723, input_fingerprints=4600),
            main_metrics=table, comparisons=comparisons, supplementary=supplementary,
            default_changed=False, test_accessed=False, new_inference=False, caveats=caveats))
        lines = ['# 固定关联候选与机制消融', '',
            '范围：38验证/调试场景、11日志、1508帧、5723次GT，4600项输入清单一致。',
            '三组原3米逐帧匹配、逐场景IDF1及全量汇总均原样复现。', '',
            '| 方法 | Precision | Recall | F1 | IDF1 | 重复节点观测 | 匹配中心误差(m) | 建图(s) |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in table:
            lines.append(f'| {row["method"]} | {row["precision"]:.2%} | {row["recall"]:.2%} | '
                f'{row["F1"]:.2%} | {row["IDF1"]:.2%} | {row["duplicates"]} | '
                f'{row["center_error_m"]:.4f} | {row["mapping_s"]:.2f} |')
        lines += ['', '主指标为IDF1；11日志配对bootstrap 10000次、种子20261123，三比较Bonferroni。', '']
        for key, item in comparisons.items():
            lo, hi = item['statistics']['intervals']['IDF1']['ci_bonferroni']
            lines.append(f'- {key}：IDF1增量{item["point_delta_percentage_points"]["IDF1"]:+.3f}个百分点，'
                f'校正区间[{lo:+.3f},{hi:+.3f}]；四主要点估计均不降={item["point_metrics_non_decreasing"]}，'
                f'重复节点不增={item["duplicates_non_increasing"]}。')
        lines += ['', '类别/距离完整分母见group_recall.csv；全部得失案例和audit计数均保存，未筛选成功样本。',
                  '原1米/2米补充表见supplementary_1m_2m.csv，不改变3米主表或IDF1主检验。', '']
        for method, diagnostics in prediction_diagnostics.items():
            stale = diagnostics['single_unknown_timestamp_over_2s_by_source_and_status']
            stale_total = sum(sum(c.values()) for c in stale.values())
            stale_unmatched = sum(v for c in stale.values() for status, v in c.items() if status != 'matched')
            lines.append(f'- {method}输出观测{diagnostics["output_observations"]}次，'
                f'场景内不同输出节点{diagnostics["unique_scene_scoped_output_nodes"]}个；'
                f'被多个输出节点匹配的官方实例{diagnostics["matched_official_instances_with_multiple_output_nodes"]}个；'
                f'单次观测unknown且测量时间戳落后超过2秒的输出{stale_total}次，其中未匹配{stale_unmatched}次。')
        lines += ['', '预测的测量新旧、观测次数、motion_state、位置来源和未匹配分布见',
                  'prediction_failure_diagnostics.json；与audit并读，未记录属性保持unknown。',
                  '输入过滤/几何接受事件的计数及逐帧内容差异见observation_audit_comparisons.json。',
                  '相同缓存不保证历史深度分簇后观测相同；R1不记录同名uncertainty_association事件，不能把缺项当成0次新建。', '']
        lines += [f'- {text}' for text in caveats]
        (output / '关联对照结论.md').write_text('\n'.join(lines) + '\n')
        guard()
        changed = [p for p, value in fingerprints.items() if sha(p) != value]
        if changed:
            raise ValueError('汇总期间来源变化：' + repr(changed))
        write(output / 'source_manifest.json', fingerprints)
        write(output / 'complete.json', dict(complete=True, source_files=len(fingerprints),
            comparison_sha256=sha(output / 'comparison.json'), wall_s=time.monotonic()-begun,
            storage_bytes_before_completion=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
            default_changed=False, test_accessed=False, sensors_read=False, new_inference=False))
        print(json.dumps(dict(output=str(output), comparisons={k: dict(
            delta_IDF1_pp=v['point_delta_percentage_points']['IDF1'],
            corrected_CI=v['statistics']['intervals']['IDF1']['ci_bonferroni'])
            for k, v in comparisons.items()}), ensure_ascii=False), flush=True)
    except BaseException as exc:
        write(output / 'interrupted.json', dict(complete=False, error=repr(exc),
            traceback=traceback.format_exc(), wall_s=time.monotonic()-begun))
        raise


if __name__ == '__main__':
    main()
