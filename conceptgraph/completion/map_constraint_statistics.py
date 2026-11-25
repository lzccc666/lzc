"""六地图条件的完整配对开发统计：零路径无安全信用，离线结果无到达信用。"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import itertools
import json
from pathlib import Path
import shutil
import time

import numpy as np

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


METRICS = ('route_generated', 'full_lane_legal_all_queries', 'full_road_legal_all_queries',
    'full_lane_legal_among_routes', 'full_road_legal_among_routes', 'point_drivable_among_routes',
    'direction_violation_among_routes', 'connection_violation_among_routes', 'reverse_among_routes',
    'out_of_lane_among_routes', 'static_rectangle_legal_among_routes', 'tangent_curvature_exceeds_bound_among_routes',
    'planning_budget_all_queries', 'common_projection_failure_all_queries')


def measurements(row):
    evaluation = row['evaluation']
    route = bool(evaluation['path_generated'])
    lane = evaluation.get('independent_lane_audit', {})
    legal = evaluation['full_lane_legal'] is True
    road = legal and evaluation.get('continuous_point_drivable') is True
    values = [route, legal, road, legal, road, evaluation.get('continuous_point_drivable') is True,
        lane.get('direction_violation_count', 0) > 0, bool(lane.get('illegal_transitions')),
        lane.get('reverse_count', 0) > 0, bool(lane.get('invalid_cells')),
        evaluation.get('body_pose_legal') is True, evaluation.get('curvature_exceeds_vehicle_bound') is True,
        row['planner']['failure'] in ('PLANNING_WALL_BUDGET', 'PLANNING_EXPANSION_BUDGET'),
        row['planner']['failure'] == 'COMMON_ENDPOINT_PROJECTION_FAILED']
    denominators = [True]*3+[route]*9+[True]*2
    return np.array([values, denominators], dtype=int).T


def ratios(counts):
    return np.divide(counts[..., 0], counts[..., 1], out=np.full(counts.shape[:-1], np.nan),
                     where=counts[..., 1] > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-run', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--wait-budget-s', type=float, default=0.)
    args = parser.parse_args()
    source = OUTPUT/'ablations'/args.input_run
    begun = time.monotonic()
    own_sha = digest(__file__)
    while not (source/'summary.json').exists():
        if (source/'interrupted.json').exists():
            raise RuntimeError('map run interrupted; prefix is not a full experiment')
        if time.monotonic()-begun > args.wait_budget_s:
            raise TimeoutError('map ablation has not completed')
        time.sleep(5)
    if digest(__file__) != own_sha:
        raise ValueError('statistics source changed while waiting')
    release = json.loads((source/'release.json').read_text())
    summary = json.loads((source/'summary.json').read_text())
    if digest(source/'cases.jsonl') != summary['records_sha256']:
        raise ValueError('map records changed')
    rows = [json.loads(s) for s in (source/'cases.jsonl').open()]
    methods = release['conditions']
    expected = {(q['scene'], q['frame_idx'], q['query_index'], c)
                for q in release['selected_queries'] for c in methods}
    if len(rows) != len(expected) or {(r['scene'], r['frame_idx'], r['query_index'], r['condition']) for r in rows} != expected:
        raise ValueError('missing or duplicated frozen map case')
    lookup = {(r['scene'], r['frame_idx'], r['query_index'], r['condition']): r for r in rows}
    checked_contexts = set()
    for row in rows:
        other = lookup[(row['scene'], row['frame_idx'], row['query_index'], methods[0])]
        if any(row[field] != other[field] for field in ('common_start', 'common_goal', 'target_instance', 'context')):
            raise ValueError('map conditions use different inputs or endpoints')
        context = row['context']
        if context['path'] not in checked_contexts:
            path = Path(context['path'])
            if digest(path/'context.json') != context['context_sha256'] or digest(path/'rasters.npz') != context['rasters_sha256']:
                raise ValueError('map context changed')
            checked_contexts.add(context['path'])
        if row['vehicle_executed'] or row['actual_arrival_evaluated'] or row['evaluation']['actual_arrival'] is not None:
            raise ValueError('offline map diagnostic cannot claim vehicle arrival')
    logs = sorted(set(release['scene_logs'].values()))
    counts = np.zeros((len(logs), len(methods), len(METRICS), 2), dtype=int)
    for row in rows:
        counts[logs.index(row['log_token']), methods.index(row['condition'])] += measurements(row)
    point = ratios(counts.sum(axis=0))
    rng = np.random.default_rng(17)
    boot = ratios(counts[rng.integers(0, len(logs), (4000, len(logs)))].sum(axis=1))
    results, comparisons = {}, []
    for j, condition in enumerate(methods):
        results[condition] = {}
        for k, name in enumerate(METRICS):
            valid = boot[:, j, k]
            valid = valid[np.isfinite(valid)]
            totals = counts[:, j, k].sum(axis=0)
            results[condition][name] = dict(estimate=float(point[j, k]) if totals[1] else None,
                ci95=np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
                count=int(totals[0]), denominator=int(totals[1]), valid_resamples=len(valid))
    for a, b in itertools.combinations(range(len(methods)), 2):
        for k, name in enumerate(METRICS):
            delta = boot[:, a, k]-boot[:, b, k]
            valid = delta[np.isfinite(delta)]
            comparisons.append(dict(a=methods[a], b=methods[b], metric=name,
                difference=float(point[a, k]-point[b, k]) if np.isfinite(point[[a, b], k]).all() else None,
                ci95=np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
                valid_resamples=len(valid)))
    dest = OUTPUT/'statistics'/args.run_id
    dest.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, dest/'generator_source.py')
    with (dest/'source_data_by_log.csv').open('x') as handle:
        writer = csv.writer(handle)
        writer.writerow(['log', 'condition', 'metric', 'count', 'denominator'])
        for i, log in enumerate(logs):
            for j, method in enumerate(methods):
                for k, metric in enumerate(METRICS):
                    writer.writerow([log, method, metric, *counts[i, j, k].tolist()])
    fields = ['scene', 'log_token', 'frame_idx', 'query_index', 'condition', 'target_instance',
              'route_generated', 'full_lane_legal', 'point_drivable', 'static_rectangle_legal',
              'failure', 'path_length_m', 'actual_arrival', 'selected_correct_object', 'stable_correct_lock']
    with (dest/'queries.csv').open('x') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            e = row['evaluation']
            writer.writerow(dict(**{f: row[f] for f in fields[:6]}, route_generated=e['path_generated'],
                full_lane_legal=e['full_lane_legal'], point_drivable=e.get('continuous_point_drivable'),
                static_rectangle_legal=e['body_pose_legal'], failure=row['planner']['failure'],
                path_length_m=e.get('independent_lane_audit', {}).get('path_length_m'),
                actual_arrival=None, selected_correct_object=None, stable_correct_lock=None))
    dump(dest/'report.json', dict(source_run=args.input_run, source_release_sha256=digest(source/'release.json'),
        source_summary_sha256=digest(source/'summary.json'), source_records_sha256=digest(source/'cases.jsonl'),
        statistics_source_sha256=own_sha, queries=release['queries'], cases=len(rows),
        unique_geometric_queries=len({(r['scene'], r['frame_idx'], r['target_instance']) for r in rows}),
        scenes=len(release['scene_logs']), logs=len(logs), unique_observation_windows=len(checked_contexts),
        conditions=results, paired_differences=comparisons,
        statuses={c: dict(Counter(r['planner']['failure'] or 'PATH_GENERATED' for r in rows if r['condition'] == c)) for c in methods},
        bootstrap=dict(unit='采集log', repetitions=4000, seed=17, paired=True),
        limitations=['当前正确对象由隔离GT给定，参考点不是已核验安全接近区域；不覆盖真实语义选择/锁定/到达',
            '开发查询按场景固定散列预选；重复语义查询不增加独立几何任务或场景，按log聚类',
            '共同端点和已观测预测障碍；未观测障碍与动态交通风险不在此安全结论内',
            '规划失败/预算耗尽不自证不可达，所有冻结查询保留；零路径的条件安全率为null',
            '完整车道审计与官方可行驶多边形检查分别报告，地图先验不一致不隐去',
            '矩形车身只作路径切线静态姿态检查，未积分控制或审计连续车身扫掠；不声称运动学可执行',
            '共享CPU且含当前窗口边缓存，不作为独占规划性能或新封存主表'],
        performance_credit=False, independent_test=False, actual_arrival_evaluated=False, goal_complete=False))
    print(json.dumps(dict(queries=release['queries'], cases=len(rows), conditions=results), ensure_ascii=False))


if __name__ == '__main__':
    main()
