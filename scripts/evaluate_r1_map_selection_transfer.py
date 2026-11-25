"""仅替换冻结对象地图的完整1115任务验证，不叠加关系修正。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))

from conceptgraph.completion.evidence import target_evidence_decision  # noqa: E402
from conceptgraph.completion.quality import object_frame_metrics  # noqa: E402
from conceptgraph.completion.retrieval_statistics import aggregate, bootstrap, interval  # noqa: E402
from conceptgraph.completion.selection_map_transfer import (  # noqa: E402
    EVIDENCE_POLICY, nodes_from_frozen_predictions, select_frozen_map,
)
from conceptgraph.completion.stage9 import available_development  # noqa: E402
from diagnose_selection_validation import ABLATION, RETRIEVAL, dump, jsonl, read, sha  # noqa: E402
from audit_r1_selection_interface import R1  # noqa: E402


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    destination=args.destination
    started=time.monotonic()
    prereg=read(destination/'preregistration.json')
    if not prereg['before_new_results']:
        raise ValueError('需先冻结地图传递对照')
    sources=[Path(__file__),PROJECT/'scripts/diagnose_selection_validation.py',
        PROJECT/'scripts/audit_r1_selection_interface.py',PROJECT/'conceptgraph/completion/selection_map_transfer.py',
        PROJECT/'conceptgraph/completion/quality.py',PROJECT/'conceptgraph/completion/association.py',
        PROJECT/'conceptgraph/completion/navigation_diagnostics.py',PROJECT/'conceptgraph/navigation/semantic.py',
        PROJECT/'conceptgraph/navigation/types.py',PROJECT/'conceptgraph/scenegraph/relations.py',
        PROJECT/'conceptgraph/completion/evidence.py',PROJECT/'conceptgraph/completion/retrieval_statistics.py',
        destination/'preregistration.json',RETRIEVAL/'queries.jsonl',ABLATION/'queries.jsonl']
    fingerprints={str(path):sha(path) for path in sources}
    release=read(RETRIEVAL/'release.json')
    r1_paths=sorted(R1.glob('scene-*/frames.jsonl.gz'))
    r1_fingerprints={str(path):sha(path) for path in r1_paths}
    dump(destination/'release.json',dict(source_sha256=fingerprints,
        original_frames_sha256=release['source_frames_sha256'],R1_frames_sha256=r1_fingerprints,
        R1_release_sha256=sha(R1/'release.json'),evidence_policy=EVIDENCE_POLICY,
        changed_map_only=True,relation_support_policy_enabled=False,
        inference_uses_gt=False,new_model_calls=0,independent_test=False,arrival_evaluated=False))
    for path in sources:
        if path.suffix=='.py':
            target=destination/'source_snapshot'/path.relative_to(PROJECT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
    queries=defaultdict(list)
    for row in jsonl(RETRIEVAL/'queries.jsonl'):
        if row['method']=='existing_object_joint_rule':
            queries[(row['scene'],row['frame_idx'])].append(row)
    original={}
    for filename,expected in release['source_frames_sha256'].items():
        if sha(filename)!=expected:
            raise ValueError('原图输入指纹改变')
        for frame in jsonl(filename):
            key=(frame['scene'],frame['frame_idx'])
            if key in queries:
                original[key]=frame
    frozen={(r['scene'],r['frame_idx'],r['query_index']):r for r in jsonl(ABLATION/'queries.jsonl')
            if r['method']=='full_joint_rule'}
    r1={}
    for path in r1_paths:
        with gzip.open(path,'rt') as handle:
            for line in handle:
                frame=json.loads(line)
                key=(frame['scene'],frame['frame_idx'])
                if key in queries:
                    r1[key]=frame
    if set(queries)!=set(original) or set(queries)!=set(r1) or len(queries)!=382:
        raise ValueError('原382任务画面必须全部保留')
    rows,cases=[],[]
    summaries=defaultdict(Counter)
    by_kind=defaultdict(lambda:defaultdict(Counter))
    frame_checks=[]
    for key,group in queries.items():
        old,new=original[key],r1[key]
        if (old['camera_timestamp_s']!=new['camera_timestamp_s']
                or old['evaluation_only_official']!=new['evaluation_only_official']
                or Path(old['rgb']).resolve()!=Path(new['input']['rgb']).resolve()):
            raise ValueError('地图比较的时间/RGB/GT不一致')
        pose=np.asarray(old['camera_pose'])
        method_inputs={'old_map_original_rule':old['predictions']['one_to_one_depth_quality'],
                       'R1_map_original_rule':new['predictions']['R1']}
        for query in group:
            pair=[]
            for method,predictions in method_inputs.items():
                # 下面两次函数仅看到预测地图、文本和位姿。官方信息在返回后才用于评价。
                nodes=nodes_from_frozen_predictions(predictions)
                ranked=select_frozen_map(query['query_text'],predictions,pose,key[1])
                selected=ranked[0].node_id if ranked else None
                truth=old['evaluation_only_official']
                metrics=object_frame_metrics(predictions,truth,threshold_m=3.)
                expected=(old['metrics']['one_to_one_depth_quality'] if method=='old_map_original_rule'
                          else new['metrics']['R1'])
                if metrics['matches']!=expected['matches']:
                    raise ValueError('原3m逐对象匹配未复现')
                association={m['prediction_index']:truth[m['truth_index']]['instance_token'] for m in metrics['matches']}
                identity=association.get(selected)
                previous=frozen[(key[0],key[1],query['query_index'])]
                if method=='old_map_original_rule' and (selected!=previous['selected_node_id']
                        or [c.node_id for c in ranked]!=previous['candidate_ids']
                        or identity!=previous['evaluation_identity']):
                    raise ValueError('旧图原规则逐条结果改变')
                positive=bool(query['positive'])
                acceptable=set(query['acceptable_instances'])
                raw_covered=any(value in acceptable for value in association.values())
                candidate_covered=any(association.get(c.node_id) in acceptable for c in ranked)
                evidence_eligible=bool(selected is not None and target_evidence_decision(nodes[selected],EVIDENCE_POLICY)[0])
                if selected is not None and not evidence_eligible:
                    raise AssertionError('选择越过原证据门')
                sparse_selected=bool(selected is not None and predictions[selected].get('shape_available') is False)
                if sparse_selected:
                    raise AssertionError('缺运动/深度的稀疏节点不应取得选择授权')
                record={k:query[k] for k in ('scene','frame_idx','query_index','query_text','query_type',
                                           'positive','acceptable_instances')}
                record.update(method=method,selected=selected is not None,selected_node_id=selected,
                    selected_frozen_object_id=predictions[selected]['node_id'] if selected is not None else None,
                    evaluation_identity=identity,candidate_ids=[c.node_id for c in ranked],
                    raw_node_target_covered=raw_covered if positive else None,
                    ranked_candidate_target_covered=candidate_covered if positive else None,
                    selected_correct=identity in acceptable if positive else None,
                    false_selection_on_no_match=selected is not None if not positive else None,
                    selection_evidence_eligible=evidence_eligible,sparse_selected=sparse_selected,
                    official_info_in_inference=False,actual_vehicle_authorization_evaluated=False,
                    actual_arrival_evaluated=False)
                pair.append(record)
                increments=dict(queries=1,positive_queries=int(positive),negative_queries=int(not positive),
                    positive_selected_correct=int(positive and identity in acceptable),
                    positive_raw_node_covered=int(positive and raw_covered),
                    positive_ranked_candidate_covered=int(positive and candidate_covered),
                    no_match_false_selection=int(not positive and selected is not None),
                    no_selection=int(selected is None),unknown_identity=int(selected is not None and identity is None),
                    selection_evidence_passed=int(evidence_eligible),sparse_selected=int(sparse_selected))
                summaries[method].update(increments)
                by_kind[query['query_type']][method].update(increments)
            rows.extend(pair)
            if (pair[0]['evaluation_identity']!=pair[1]['evaluation_identity']
                    or pair[0]['selected']!=pair[1]['selected']):
                cases.append(dict(old_map=pair[0],R1_map=pair[1]))
        frame_checks.append(dict(scene=key[0],frame_idx=key[1],timestamp_same=True,GT_same=True,
            RGB_same=True,old_predictions=len(method_inputs['old_map_original_rule']),
            R1_predictions=len(method_inputs['R1_map_original_rule']),
            R1_sparse_unknown=sum(p.get('shape_available') is False for p in method_inputs['R1_map_original_rule'])))
        if time.monotonic()-started>2700 or shutil.disk_usage(PROJECT).free<20*1024**3:
            raise RuntimeError('达到地图传递运行预算或20GiB停写线')
    with (destination/'queries.jsonl').open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    dump(destination/'changed_cases.json',cases)
    dump(destination/'frame_checks.json',frame_checks)
    scene_logs={r['scene']:r['log_token'] for r in available_development() if r['scene'] in {k[0] for k in queries}}
    methods,logs,clusters=aggregate(rows,scene_logs)
    totals,estimates,rates=bootstrap(clusters,repetitions=10000,seed=17)
    results={}
    for j,method in enumerate(methods):
        results[method]=dict(summaries[method])
        for m,metric in enumerate(('positive_selection','no_match_false_selection')):
            results[method][metric]=dict(estimate=float(estimates[j,m]),numerator=int(totals[j,m*2]),
                denominator=int(totals[j,m*2+1]),**interval(rates[:,j,m]))
    base,new_index=methods.index('old_map_original_rule'),methods.index('R1_map_original_rule')
    paired=[dict(method='R1_map_original_rule',baseline='old_map_original_rule',metric=metric,
        difference=float(estimates[new_index,m]-estimates[base,m]),**interval(rates[:,new_index,m]-rates[:,base,m]))
        for m,metric in enumerate(('positive_selection','no_match_false_selection'))]
    if any(sha(path)!=expected for path,expected in {**fingerprints,**r1_fingerprints}.items()):
        raise ValueError('冻结输入或源码运行中改变')
    dump(destination/'report.json',dict(methods=results,by_query_kind=by_kind,paired_differences=paired,
        queries_per_method=len(rows)//2,query_frames=len(frame_checks),scenes=len(scene_logs),logs=len(logs),
        all_original_choices_and_matchings_reproduced=True,changed_map_only=True,
        relation_support_policy_enabled=False,sparse_unknown_retained=True,new_model_calls=0,
        inference_uses_gt=False,independent_test=False,default_changed=False,
        actual_vehicle_authorization_evaluated=False,actual_arrival_evaluated=False,
        elapsed_s=time.monotonic()-started,raw_sha256=sha(destination/'queries.jsonl'),
        limitations=['旧图与R1来自不同前端及建图版本；这是地图版本全链路传递，不是单一关联模块因果贡献',
            '38场景已有调试历史，不是新的未见测试；733正和382负任务不随新地图改写',
            '稀疏节点缺尺寸/运动/深度证据，保留unknown；不伪造静止或安全授权',
            '通过原选择证据门不等于道路/车辆检查通过；本轮不评价路线、锁定或到达',
            'CLIP特征未保存，此处仅同一规则选择，不能归因给Qwen']))
    with (destination/'source_data_by_log.csv').open('x') as handle:
        writer=csv.writer(handle)
        writer.writerow(['log','method','positive_correct','positive','negative_false_selection','negative'])
        for i,log in enumerate(logs):
            for j,method in enumerate(methods):
                writer.writerow([log,method,*clusters[i,j].tolist()])
    print(json.dumps(dict(methods=results,paired_differences=paired,elapsed_s=time.monotonic()-started),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
