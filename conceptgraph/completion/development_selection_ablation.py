"""显式绑定开发检索输入的关系/证据消融；保留正在运行的旧生成器。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np

from conceptgraph.completion.retrieval_statistics import aggregate, bootstrap, interval, nullable_number
from conceptgraph.completion.selection_ablation import select
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, available_development, digest, dump


CONDITIONS = ['full_joint_rule', 'without_relation_edges', 'without_depth_evidence_gate']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--retrieval-run', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    source = OUTPUT/'main_experiments'/args.retrieval_run
    source_release = json.loads((source/'release.json').read_text())
    query_sha = digest(source/'queries.jsonl')
    frozen = defaultdict(list)
    for row in map(json.loads, (source/'queries.jsonl').open()):
        if row['method'] == 'existing_object_joint_rule':
            frozen[(row['scene'], row['frame_idx'])].append(row)
    keys = [(r['scene'], r['frame_idx'], r['query_index']) for group in frozen.values() for r in group]
    known = {r['scene']: r['log_token'] for r in available_development()}
    if not keys or len(keys) != len(set(keys)) or not {key[0] for key in keys} <= known.keys():
        raise ValueError('只接受固定开发查询；不允许空输入、重复或测试场景')
    destination = OUTPUT/'ablations'/args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__)]
    sources += [Path(__file__).with_name(name+'.py') for name in
        ['selection_ablation', 'navigation_diagnostics', 'retrieval_statistics', 'stage9']]
    sources += [PROJECT/'conceptgraph/navigation'/name for name in ['semantic.py', 'types.py']]
    sources += [PROJECT/'conceptgraph/scenegraph/relations.py']
    source_hashes = {str(p): digest(p) for p in sources}
    dump(destination/'release.json', dict(retrieval_run=args.retrieval_run, conditions=CONDITIONS,
        source_queries_sha256=query_sha, source_frames_sha256=source_release['source_frames_sha256'],
        source_release_sha256=digest(source/'release.json'), source_report_sha256=digest(source/'report.json'),
        source_sha256=source_hashes, expected_queries=len(keys), expected_rows=len(keys)*len(CONDITIONS),
        wall_budget_s=600, storage_budget_bytes=15000000, CPU_threads=2, model_calls=0,
        design='全部源开发查询；关系条件只移除预测边，证据门限关闭只作离线评价',
        independent_test=False, gt_in_inference=False, actual_arrival_evaluated=False, goal_complete=False))
    for path in sources:
        target = destination/'frozen_source'/path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    rows, counts = [], defaultdict(Counter)
    by_kind = defaultdict(lambda: defaultdict(Counter))
    baseline_checked = 0
    for name, sha in source_release['source_frames_sha256'].items():
        if digest(name) != sha:
            raise ValueError('开发预测图指纹改变')
        for frame in map(json.loads, Path(name).open()):
            for query in frozen.get((frame['scene'], frame['frame_idx']), []):
                for condition in CONDITIONS:
                    ranked = select(query['query_text'], frame['predictions']['one_to_one_depth_quality'],
                        np.asarray(frame['camera_pose']), frame['frame_idx'], condition)
                    selected = ranked[0].node_id if ranked else None
                    # 推理到此结束，官方身份仅在隔离评价中使用。
                    truth = frame['evaluation_only_official']
                    association = {m['prediction_index']: truth[m['truth_index']]['instance_token']
                        for m in frame['metrics']['one_to_one_depth_quality']['matches']}
                    identity = association.get(selected)
                    if condition == 'full_joint_rule':
                        if (selected is not None) != query['selected'] or identity != query['evaluation_identity']:
                            raise ValueError('完整条件未复现冻结选择')
                        baseline_checked += 1
                    positive = bool(query['positive'])
                    correct = identity in query['acceptable_instances'] if positive else None
                    false = selected is not None if not positive else None
                    row = {key: query[key] for key in ['scene', 'frame_idx', 'query_index', 'query_text',
                        'query_type', 'positive', 'acceptable_instances']}
                    row.update(method=condition, selected=selected is not None, selected_node_id=selected,
                        evaluation_identity=identity, candidate_ids=[c.node_id for c in ranked],
                        selected_correct=correct, false_selection_on_no_match=false,
                        gt_in_inference=False, actual_arrival_evaluated=False)
                    rows.append(row)
                    increment = dict(queries=1, positive_queries=int(positive), negative_queries=int(not positive),
                        positive_selected_correct=int(positive and correct), no_match_false_selection=int(bool(false)),
                        no_selection=int(selected is None), association_unknown=int(selected is not None and identity is None))
                    counts[condition].update(increment)
                    by_kind[query['query_type']][condition].update(increment)
            if time.monotonic()-started > 600:
                raise TimeoutError('开发选择消融达到600秒预算，保留已写文件')
    if baseline_checked != len(keys) or len(rows) != len(keys)*len(CONDITIONS):
        raise ValueError('冻结查询缺失，不能删除失败行')
    with (destination/'queries.jsonl').open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    if (destination/'queries.jsonl').stat().st_size > 15000000:
        raise OSError('开发选择消融超过存储预算')
    scene_logs = {scene: known[scene] for scene in {key[0] for key in keys}}
    methods, logs, cluster_counts = aggregate(rows, scene_logs)
    totals, estimates, rates = bootstrap(cluster_counts)
    baseline = methods.index('full_joint_rule')
    results, paired = {}, []
    for j, method in enumerate(methods):
        results[method] = dict(counts[method])
        for m, metric in enumerate(['positive_selection', 'no_match_false_selection']):
            results[method][metric] = dict(estimate=nullable_number(estimates[j, m]),
                numerator=int(totals[j, m*2]), denominator=int(totals[j, m*2+1]), **interval(rates[:, j, m]))
            if j != baseline:
                paired.append(dict(method=method, baseline='full_joint_rule', metric=metric,
                    difference=nullable_number(estimates[j, m]-estimates[baseline, m]),
                    **interval(rates[:, j, m]-rates[:, baseline, m])))
    if digest(source/'queries.jsonl') != query_sha or any(digest(p) != sha for p, sha in source_hashes.items()):
        raise ValueError('源码或查询在运行中改变')
    with (destination/'source_data_by_log.csv').open('x') as handle:
        writer = csv.writer(handle)
        writer.writerow(['log', 'condition', 'positive_correct', 'positive_queries', 'negative_false_selection', 'negative_queries'])
        for i, log in enumerate(logs):
            for j, method in enumerate(methods):
                writer.writerow([log, method, *cluster_counts[i, j].tolist()])
    dump(destination/'report.json', dict(retrieval_run=args.retrieval_run, conditions=results, by_query_kind=by_kind,
        paired_differences=paired, scenes=len(scene_logs), logs=len(logs), paired_queries=len(keys),
        bootstrap=dict(unit='采集log', paired=True, repetitions=4000, seed=17), baseline_all_queries_equivalent=True,
        raw_sha256=digest(destination/'queries.jsonl'), elapsed_s=time.monotonic()-started,
        limitations=['固定已有预测对象图，未知关联与目标缺失保留为正查询失败',
            '构造开发指令；未开启封存测试，也不获得正确锁定、路线或实际到达信用',
            '关闭证据门限仅离线；未修改车辆安全检查、正在运行的导航或旧实验生成器',
            '官方sample GT仍较相机晚约33–42ms，保留评价时间近似'],
        independent_test=False, gt_in_inference=False, actual_arrival_evaluated=False, goal_complete=False))
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
