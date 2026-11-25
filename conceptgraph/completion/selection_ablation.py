"""固定开发查询及对象图上的关系/证据门限消融，仅离线检索，不执行车辆。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np

from conceptgraph.completion.navigation_diagnostics import to_nodes
from conceptgraph.completion.retrieval_statistics import aggregate, bootstrap, interval
from conceptgraph.completion.stage9 import OUTPUT, available_development, digest, dump
from conceptgraph.navigation.semantic import parse_navigation_query, rank_candidate_targets
from conceptgraph.scenegraph.relations import build_relations


def select(query, predictions, pose, frame_index, condition):
    """推理接口不接收官方类别、身份、可接受集合或正确解析。"""
    nodes = to_nodes(predictions)
    forward = pose[:2,2]
    edges = [] if condition == 'without_relation_edges' else build_relations(nodes,
        dict(coordinate_frame='ego_heading',directional_mode='all_supported_axes'),
        frame_index=frame_index,ego_heading=forward).edges
    parsed = parse_navigation_query(query,8.)
    parsed.vehicle_heading_xy = forward
    policy = None if condition == 'without_depth_evidence_gate' else dict(enabled=True,
        minimum_observations=3,minimum_lidar_pixels=5,minimum_depth_support_fraction=.4)
    ranked = rank_candidate_targets(parsed,nodes,edges,pose[:2,3],.55,.3,.25,
        relation_coordinate_frame='ego_heading',relation_coordinate_frame_fallback=None,
        evidence_policy=policy)
    return ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    args = parser.parse_args()
    started = time.monotonic()
    source = OUTPUT/'main_experiments/retrieval_baselines_dev_full_001'
    source_release = json.loads((source/'release.json').read_text())
    paths = [Path(p) for p in source_release['source_frames_sha256']]
    dest = OUTPUT/'ablations'/args.run_id
    dest.mkdir(exist_ok=False)
    conditions = ['full_joint_rule','without_relation_edges','without_depth_evidence_gate']
    dump(dest/'release.json',dict(conditions=conditions,source_sha256=digest(Path(__file__)),
        source_queries_sha256=digest(source/'queries.jsonl'),
        source_frames_sha256=source_release['source_frames_sha256'],
        source_report_sha256=digest(source/'report.json'),
        wall_budget_s=600,storage_budget_bytes=15000000,
        query_selection='全部原1115开发查询，未按消融结果选择',
        relation_ablation='只去掉预测关系边，保留原指令关系要求；保守拒绝属于真实结果',
        evidence_ablation='关闭选择端深度/观测次数门限，仅离线，不进入执行器',
        independent_test=False,gt_in_inference=False,arrival_evaluated=False,goal_complete=False))
    shutil.copy2(__file__,dest/'generator_source.py')
    frozen = defaultdict(list)
    for row in map(json.loads,(source/'queries.jsonl').open()):
        if row['method']=='existing_object_joint_rule':
            frozen[(row['scene'],row['frame_idx'])].append(row)
    rows = []
    counts = defaultdict(Counter)
    by_kind = defaultdict(lambda:defaultdict(Counter))
    for path in paths:
        assert digest(path)==source_release['source_frames_sha256'][str(path)]
        for frame in map(json.loads,path.open()):
            for q in frozen.get((frame['scene'],frame['frame_idx']),[]):
                # 只有预测、位姿与文字进入选择；关联真值在选择函数返回后访问。
                predictions = frame['predictions']['one_to_one_depth_quality']
                for condition in conditions:
                    ranked = select(q['query_text'],predictions,np.asarray(frame['camera_pose']),
                                    frame['frame_idx'],condition)
                    selected = ranked[0].node_id if ranked else None
                    truth = frame['evaluation_only_official']
                    association = {m['prediction_index']:truth[m['truth_index']]['instance_token']
                        for m in frame['metrics']['one_to_one_depth_quality']['matches']}
                    identity = association.get(selected)
                    if condition=='full_joint_rule':
                        assert (selected is not None)==q['selected'] and identity==q['evaluation_identity']
                    positive = bool(q['positive'])
                    correct = identity in q['acceptable_instances'] if positive else None
                    false = selected is not None if not positive else None
                    row = {k:q[k] for k in ['scene','frame_idx','query_index','query_text','query_type',
                                            'positive','acceptable_instances']}
                    row.update(method=condition,selected=selected is not None,selected_node_id=selected,
                        candidate_ids=[c.node_id for c in ranked],evaluation_identity=identity,
                        selected_correct=correct,false_selection_on_no_match=false,
                        gt_in_inference=False,actual_arrival_evaluated=False)
                    rows.append(row)
                    increments = dict(queries=1,positive_queries=int(positive),negative_queries=int(not positive),
                        positive_selected_correct=int(positive and correct),no_match_false_selection=int(bool(false)),
                        no_selection=int(selected is None),association_unknown=int(selected is not None and identity is None))
                    counts[condition].update(increments)
                    by_kind[q['query_type']][condition].update(increments)
            if time.monotonic()-started>600:
                raise TimeoutError('selection ablation wall budget')
    with (dest/'queries.jsonl').open('x') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    scene_logs = {r['scene']:r['log_token'] for r in available_development()
                  if r['scene'] in {x['scene'] for x in rows}}
    methods,logs,cluster_counts = aggregate(rows,scene_logs)
    totals,estimates,rates = bootstrap(cluster_counts)
    baseline = methods.index('full_joint_rule')
    estimates_out,paired = {},[]
    for j,method in enumerate(methods):
        estimates_out[method] = dict(counts[method])
        for m,metric in enumerate(['positive_selection','no_match_false_selection']):
            estimates_out[method][metric] = dict(estimate=float(estimates[j,m]),
                numerator=int(totals[j,m*2]),denominator=int(totals[j,m*2+1]),**interval(rates[:,j,m]))
            if j!=baseline:
                paired.append(dict(method=method,baseline='full_joint_rule',metric=metric,
                    difference=float(estimates[j,m]-estimates[baseline,m]),
                    **interval(rates[:,j,m]-rates[:,baseline,m])))
    with (dest/'source_data_by_log.csv').open('x') as f:
        writer=csv.writer(f)
        writer.writerow(['log','condition','positive_correct','positive_queries','negative_false_selection','negative_queries'])
        for i,log in enumerate(logs):
            for j,method in enumerate(methods):
                writer.writerow([log,method,*cluster_counts[i,j].tolist()])
    dump(dest/'report.json',dict(conditions=estimates_out,by_query_kind=by_kind,paired_differences=paired,
        scenes=len(scene_logs),logs=len(logs),paired_queries=len(rows)//len(methods),
        bootstrap=dict(unit='采集log',repetitions=4000,seed=17,paired=True),
        baseline_all_queries_equivalent=True,raw_sha256=digest(dest/'queries.jsonl'),elapsed_s=time.monotonic()-started,
        independent_test=False,gt_in_inference=False,arrival_evaluated=False,goal_complete=False,
        limitations=['固定已有融合图，不是重新运行关闭动态维护的建图消融',
          '官方中心/类别一对一关联未知仍作正任务失败；未增加人工实例信用',
          '原有构造开发指令，不是自然分布或封存测试；仅检索，不评价路径/锁定/到达',
          '关闭证据门限仅离线；未关闭车辆安全保护或实际执行']))
    print(json.dumps(estimates_out,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
