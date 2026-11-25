"""同一冻结1115验证任务上的单一关系支持范围修正。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from conceptgraph.completion.retrieval_statistics import aggregate, bootstrap, interval  # noqa: E402
from conceptgraph.completion.selection_ablation import select  # noqa: E402
from conceptgraph.completion.selection_consistency import select_consistent_relations  # noqa: E402
from conceptgraph.completion.stage9 import available_development  # noqa: E402
from diagnose_selection_validation import ABLATION, RETRIEVAL, dump, jsonl, read, sha  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination
    started = time.monotonic()
    preregistration = read(destination/'preregistration.json')
    if preregistration['status'] != 'preregistered_before_new_method_results':
        raise ValueError('需要先固定方案')
    release = read(RETRIEVAL/'release.json')
    sources = [Path(__file__), PROJECT/'scripts/diagnose_selection_validation.py',
        PROJECT/'conceptgraph/navigation/semantic.py', PROJECT/'conceptgraph/navigation/types.py',
        PROJECT/'conceptgraph/completion/selection_consistency.py',
        PROJECT/'conceptgraph/completion/selection_ablation.py',
        PROJECT/'conceptgraph/completion/navigation_diagnostics.py',
        PROJECT/'conceptgraph/completion/retrieval_statistics.py',
        PROJECT/'conceptgraph/completion/evidence.py', PROJECT/'conceptgraph/scenegraph/relations.py',
        destination/'preregistration.json', RETRIEVAL/'queries.jsonl', ABLATION/'queries.jsonl']
    fingerprints = {str(path): sha(path) for path in sources}
    dump(destination/'release.json', dict(preregistration_sha256=sha(destination/'preregistration.json'),
        source_sha256=fingerprints, source_frames_sha256=release['source_frames_sha256'],
        model_calls=0, independent_test=False, actual_arrival_evaluated=False,
        seed=17, bootstrap_seed=17, bootstrap_repetitions=10000,
        inference_uses_gt=False, default_changed=False))
    for path in sources:
        if path.suffix == '.py':
            target = destination/'source_snapshot'/path.relative_to(PROJECT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    original_queries = list(jsonl(RETRIEVAL/'queries.jsonl'))
    ablations = list(jsonl(ABLATION/'queries.jsonl'))
    groups = defaultdict(list)
    for row in original_queries:
        if row['method'] == 'existing_object_joint_rule':
            groups[(row['scene'], row['frame_idx'])].append(row)
    frozen = {(r['scene'],r['frame_idx'],r['query_index']): r for r in ablations
              if r['method']=='full_joint_rule'}
    records, cases = [], []
    for filename, expected in release['source_frames_sha256'].items():
        if sha(filename) != expected:
            raise ValueError('冻结对象图改变')
        for frame in jsonl(filename):
            queries = groups.get((frame['scene'],frame['frame_idx']), [])
            if not queries:
                continue
            predictions = frame['predictions']['one_to_one_depth_quality']
            pose = np.asarray(frame['camera_pose'])
            # 函数仅接收预测/位姿/文本；GT映射在函数返回后只用于评分。
            for query in queries:
                new_records = []
                for method in ('full_joint_rule','relation_support_consistent'):
                    if method=='full_joint_rule':
                        ranked = select(query['query_text'],predictions,pose,frame['frame_idx'],method)
                    else:
                        ranked = select_consistent_relations(query['query_text'],predictions,pose,frame['frame_idx'])
                    selected = ranked[0].node_id if ranked else None
                    truth = frame['evaluation_only_official']
                    association = {m['prediction_index']:truth[m['truth_index']]['instance_token']
                        for m in frame['metrics']['one_to_one_depth_quality']['matches']}
                    identity = association.get(selected)
                    previous = frozen[(frame['scene'],frame['frame_idx'],query['query_index'])]
                    if method=='full_joint_rule' and (selected!=previous['selected_node_id']
                            or identity!=previous['evaluation_identity']
                            or [c.node_id for c in ranked]!=previous['candidate_ids']):
                        raise ValueError('新增可选策略改变了原默认选择')
                    positive = bool(query['positive'])
                    record = {key:query[key] for key in ('scene','frame_idx','query_index','query_text',
                        'query_type','positive','acceptable_instances')}
                    record.update(method=method,selected=selected is not None,selected_node_id=selected,
                        evaluation_identity=identity,candidate_ids=[c.node_id for c in ranked],
                        selected_correct=identity in query['acceptable_instances'] if positive else None,
                        false_selection_on_no_match=selected is not None if not positive else None,
                        selected_debug=ranked[0].debug if ranked else None,
                        gt_in_inference=False,actual_arrival_evaluated=False)
                    new_records.append(record)
                records.extend(new_records)
                if new_records[0]['selected_node_id']!=new_records[1]['selected_node_id']:
                    cases.append(dict(baseline=new_records[0],new_method=new_records[1]))
            if time.monotonic()-started>1200 or shutil.disk_usage(PROJECT).free<20*1024**3:
                raise RuntimeError('预算或磁盘边界，保留已写产物')
    # 已保存的公平参照使用原同任务记录，不重新执行CLIP，也不降低任何基线。
    records += [r for r in original_queries if r['method']!='existing_object_joint_rule']
    records += [r for r in ablations if r['method']!='full_joint_rule']
    with (destination/'queries.jsonl').open('x') as handle:
        for row in records:
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    dump(destination/'changed_cases.json',cases)
    scene_logs = {r['scene']:r['log_token'] for r in available_development()
                  if r['scene'] in {row['scene'] for row in records}}
    methods, logs, cluster_counts = aggregate(records,scene_logs)
    totals, estimates, rates = bootstrap(cluster_counts,repetitions=10000,seed=17)
    counts = defaultdict(Counter)
    by_kind = defaultdict(lambda:defaultdict(Counter))
    for row in records:
        positive = bool(row['positive'])
        increments = dict(queries=1,positive_queries=int(positive),negative_queries=int(not positive),
            positive_selected_correct=int(positive and row['selected_correct']),
            no_match_false_selection=int(not positive and row['false_selection_on_no_match']),
            no_selection=int(not row['selected']),
            unknown_identity=int(row['selected'] and row['evaluation_identity'] is None))
        counts[row['method']].update(increments)
        by_kind[row['query_type']][row['method']].update(increments)
    results = {}
    for j,method in enumerate(methods):
        results[method] = dict(counts[method])
        for m,metric in enumerate(('positive_selection','no_match_false_selection')):
            results[method][metric] = dict(estimate=float(estimates[j,m]),
                numerator=int(totals[j,m*2]),denominator=int(totals[j,m*2+1]),**interval(rates[:,j,m]))
    base, new = methods.index('full_joint_rule'),methods.index('relation_support_consistent')
    paired = [dict(method='relation_support_consistent',baseline='full_joint_rule',metric=metric,
        difference=float(estimates[new,m]-estimates[base,m]),**interval(rates[:,new,m]-rates[:,base,m]))
        for m,metric in enumerate(('positive_selection','no_match_false_selection'))]
    wins = sum(bool(c['new_method']['selected_correct']) and not bool(c['baseline']['selected_correct']) for c in cases)
    losses = sum(bool(c['baseline']['selected_correct']) and not bool(c['new_method']['selected_correct']) for c in cases)
    if any(sha(path)!=expected for path,expected in fingerprints.items()):
        raise ValueError('输入/源码在运行中改变')
    dump(destination/'report.json',dict(methods=results,by_query_kind=by_kind,paired_differences=paired,
        queries_per_method=1115,positive=733,negative=382,scenes=len(scene_logs),logs=len(logs),
        full_baseline_reproduced=True,changed_choices=len(cases),positive_wins=wins,positive_losses=losses,
        bootstrap=dict(unit='acquisition_log',paired=True,repetitions=10000,seed=17,primary_comparisons=1),
        elapsed_s=time.monotonic()-started,gt_in_inference=False,independent_test=False,
        actual_arrival_evaluated=False,default_changed=False,raw_sha256=sha(destination/'queries.jsonl'),
        limitations=['诊断后固定的验证试验；不是新的独立测试或自然指令泛化成绩',
            '733个正任务全部保留，目标未建图/未知关联仍作失败',
            '仅修正关系支持范围，不放宽静止/观测次数/深度或几何授权门',
            '规则检索与Qwen模型推理分开；本轮模型调用为0',
            '三个旧检测/CLIP参照与对象图的身份关联规则不完全相同，详见原release']))
    with (destination/'source_data_by_log.csv').open('x') as handle:
        writer = csv.writer(handle)
        writer.writerow(['log','method','positive_correct','positive','negative_false_selection','negative'])
        for i,log in enumerate(logs):
            for j,method in enumerate(methods):
                writer.writerow([log,method,*cluster_counts[i,j].tolist()])
    print(json.dumps(dict(methods=results,paired_differences=paired,changed_choices=len(cases),
        positive_wins=wins,positive_losses=losses,elapsed_s=time.monotonic()-started),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
