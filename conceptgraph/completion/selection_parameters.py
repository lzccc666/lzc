"""固定开发图与查询的选择端单因素参数检查；不修改在运行的导航版本。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np

from conceptgraph.completion.navigation_diagnostics import to_nodes
from conceptgraph.completion.retrieval_statistics import interval, nullable_number
from conceptgraph.completion.stage9 import OUTPUT, available_development, digest, dump
from conceptgraph.navigation.semantic import parse_navigation_query, rank_candidate_targets
from conceptgraph.scenegraph.relations import build_relations


BASE = dict(top_k=3, minimum_observations=3, minimum_depth_support_fraction=.4, rejection_score=0.)
SETTINGS = {'baseline': BASE}
for field, values in [('top_k', [1, 5]), ('minimum_observations', [1, 5]),
                      ('minimum_depth_support_fraction', [.2, .6]), ('rejection_score', [.5, .7])]:
    for value in values:
        SETTINGS[f'{field}_{value}'] = dict(BASE, **{field: value})


def filtered_candidates(parsed, nodes, edges, pose, setting):
    """只访问解析、预测图与当前位姿；独立标签仅在本函数返回后使用。"""
    parsed.vehicle_heading_xy = pose[:2, 2]
    ranked = rank_candidate_targets(parsed, nodes, edges, pose[:2, 3], .55, .3, .25,
        relation_coordinate_frame='ego_heading', relation_coordinate_frame_fallback=None,
        evidence_policy=dict(enabled=True, minimum_lidar_pixels=5,
            minimum_observations=setting['minimum_observations'],
            minimum_depth_support_fraction=setting['minimum_depth_support_fraction']))
    return [c for c in ranked if c.score >= setting['rejection_score']][:setting['top_k']]


def counts(rows):
    return np.array([sum(r['positive'] and r['correct'] for r in rows), sum(r['positive'] for r in rows),
        sum(not r['positive'] and r['selected'] for r in rows), sum(not r['positive'] for r in rows),
        sum(r['positive'] and r['correct_in_candidates'] for r in rows)], dtype=int)


def estimates(values):
    denominator = values[..., [1, 3, 1]]
    return np.divide(values[..., [0, 2, 4]], denominator,
        out=np.full(denominator.shape, np.nan), where=denominator > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--retrieval-run', default='retrieval_baselines_dev_full_001')
    args = parser.parse_args()
    begun = time.monotonic()
    source = OUTPUT/'main_experiments'/args.retrieval_run
    frozen = json.loads((source/'release.json').read_text())['source_frames_sha256']
    queries = defaultdict(list)
    for row in map(json.loads, (source/'queries.jsonl').open()):
        if row['method'] == 'existing_object_joint_rule':
            queries[(row['scene'], row['frame_idx'])].append(row)
    keys = [(r['scene'], r['frame_idx'], r['query_index']) for group in queries.values() for r in group]
    expected_queries = len(keys)
    if not expected_queries or len(set(keys)) != expected_queries:
        raise ValueError('开发查询为空或重复')
    dest = OUTPUT/'performance'/args.run_id
    dest.mkdir(parents=True, exist_ok=False)
    dump(dest/'release.json', dict(settings=SETTINGS, retrieval_run=args.retrieval_run,
        source_queries_sha256=digest(source/'queries.jsonl'),
        source_frames_sha256=frozen, source_sha256=digest(__file__), wall_budget_s=600,
        storage_budget_bytes=20000000, paired_queries=expected_queries, expected_rows=expected_queries*len(SETTINGS),
        design='选择端单因素检查；Top-K只改变候选覆盖与后续排序上限，未假设它会改变首选',
        grouping='官方可接受目标类别、可见性最差值与距自车最近距离；无匹配单列；道路分组尚缺',
        association_threshold_evaluated=False, mapper_rebuilt=False, parameter_applied_to_running_navigation=False,
        independent_test=False, goal_complete=False))
    shutil.copy2(__file__, dest/'generator_source.py')
    rows = []
    baseline_checked = 0
    for name, sha in frozen.items():
        path = Path(name)
        if digest(path) != sha:
            raise ValueError('frozen development graph changed')
        for frame in map(json.loads, path.open()):
            qs = queries.get((frame['scene'], frame['frame_idx']), [])
            if not qs:
                continue
            pose = np.asarray(frame['camera_pose'])
            nodes = to_nodes(frame['predictions']['one_to_one_depth_quality'])
            edges = build_relations(nodes, dict(coordinate_frame='ego_heading', directional_mode='all_supported_axes'),
                frame_index=frame['frame_idx'], ego_heading=pose[:2, 2]).edges
            for q in qs:
                for condition, setting in SETTINGS.items():
                    ranked = filtered_candidates(parse_navigation_query(q['query_text'], 8.), nodes, edges, pose, setting)
                    # 至此推理结束。以下官方身份与可见性均只用于隔离评价。
                    truth = frame['evaluation_only_official']
                    association = {m['prediction_index']: truth[m['truth_index']]['instance_token']
                        for m in frame['metrics']['one_to_one_depth_quality']['matches']}
                    selected = ranked[0].node_id if ranked else None
                    identity = association.get(selected)
                    if condition == 'baseline':
                        if (selected is not None) != q['selected'] or identity != q['evaluation_identity']:
                            raise ValueError('baseline does not reproduce all frozen choices')
                        baseline_checked += 1
                    accepted = set(q['acceptable_instances'])
                    objects = [o for o in truth if o['instance_token'] in accepted]
                    visibility = [int(o['visibility_token']) for o in objects if str(o.get('visibility_token', '')).isdigit()]
                    distance = min((float(np.linalg.norm(np.asarray(o['center'])[:2]-pose[:2, 3]))
                                    for o in objects), default=None)
                    rows.append(dict(scene=q['scene'], frame_idx=q['frame_idx'], query_index=q['query_index'],
                        condition=condition, query_type=q['query_type'], positive=bool(q['positive']),
                        selected=selected is not None, selected_node_id=selected, evaluation_identity=identity,
                        correct=bool(accepted and identity in accepted),
                        correct_in_candidates=bool(accepted and any(association.get(c.node_id) in accepted for c in ranked)),
                        candidate_ids=[c.node_id for c in ranked], association_unknown=selected is not None and identity is None,
                        target_category='+'.join(sorted({o['category'] for o in objects})) or 'no_match',
                        visibility_group=str(min(visibility)) if visibility else 'unknown',
                        distance_group='no_match' if distance is None else '1–15m' if distance < 15 else '15–30m' if distance < 30 else '30–65m',
                        gt_in_inference=False, arrival_evaluated=False))
            if time.monotonic()-begun > 600:
                raise TimeoutError('frozen selection parameter wall budget')
    if baseline_checked != expected_queries or len(rows) != expected_queries*len(SETTINGS):
        raise ValueError('incomplete paired development sweep')
    with (dest/'queries.jsonl').open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    scene_logs = {r['scene']: r['log_token'] for r in available_development()
                  if r['scene'] in {q['scene'] for q in rows}}
    logs = sorted(set(scene_logs.values()))
    methods = list(SETTINGS)
    grouped = defaultdict(list)
    subgroups = defaultdict(list)
    for row in rows:
        grouped[(scene_logs[row['scene']], row['condition'])].append(row)
        for field in ['target_category', 'visibility_group', 'distance_group', 'query_type']:
            subgroups[(row['condition'], field, row[field])].append(row)
    totals = np.array([[counts(grouped[(log, method)]) for method in methods] for log in logs])
    rng = np.random.default_rng(17)
    rates = estimates(totals[rng.integers(0, len(logs), size=(4000, len(logs)))].sum(axis=1))
    points = estimates(totals.sum(axis=0))
    results, paired = {}, []
    for j, method in enumerate(methods):
        results[method] = {}
        for m, metric in enumerate(['positive_selection', 'no_match_false_selection', 'positive_topk_coverage']):
            values = rates[:, j, m]
            results[method][metric] = dict(estimate=nullable_number(points[j, m]), **interval(values),
                count=int(totals[:, j, [0, 2, 4][m]].sum()),
                denominator=int(totals[:, j, [1, 3, 1][m]].sum()))
            if j:
                difference = rates[:, j, m]-rates[:, 0, m]
                paired.append(dict(condition=method, baseline='baseline', metric=metric,
                    difference=nullable_number(points[j, m]-points[0, m]), **interval(difference)))
    with (dest/'source_data_by_log.csv').open('x') as handle:
        writer = csv.writer(handle)
        writer.writerow(['log', 'condition', 'positive_correct', 'positive_queries', 'negative_false', 'negative_queries', 'positive_topk'])
        for i, log in enumerate(logs):
            for j, method in enumerate(methods):
                writer.writerow([log, method, *totals[i, j].tolist()])
    with (dest/'exploratory_subgroups.csv').open('x') as handle:
        writer = csv.writer(handle)
        writer.writerow(['condition', 'grouping', 'group', 'positive_correct', 'positive_queries', 'negative_false', 'negative_queries', 'positive_topk'])
        for group, members in sorted(subgroups.items()):
            writer.writerow([*group, *counts(members).tolist()])
    dump(dest/'report.json', dict(conditions=results, paired_differences=paired, paired_queries=expected_queries,
        retrieval_run=args.retrieval_run,
        scenes=len(scene_logs), logs=len(logs), raw_sha256=digest(dest/'queries.jsonl'), baseline_all_queries_verified=True,
        bootstrap=dict(unit='采集log', paired=True, repetitions=4000, seed=17),
        elapsed_s=time.monotonic()-begun, parameter_applied_to_running_navigation=False,
        limitations=['仅选择端；未改变关联阈值/重新建图，不覆盖车辆闭环参数',
            'Top-K改变候选包含率，不赋予oracle重排或正确到达信用',
            '类别/遮挡/距离分组为探索性计数，未补齐道路泛化；unknown保持独立',
            '未按开发最佳设置修改已冻结的450次导航版本；无封存方法调用'],
        independent_test=False, stage12_complete=False, goal_complete=False))
    print(json.dumps(dict(conditions=results, elapsed_s=time.monotonic()-begun), ensure_ascii=False))


if __name__ == '__main__':
    main()
