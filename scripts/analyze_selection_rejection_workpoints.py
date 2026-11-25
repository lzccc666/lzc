"""相同原连续评分上的共享拒选阈值；阈值冻结先于读取评价正确性。"""
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

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))

from conceptgraph.completion.selection_map_transfer import select_frozen_map  # noqa: E402
from conceptgraph.navigation.semantic import FAR_WORDS, normalize_text  # noqa: E402
from audit_r1_selection_interface import R1  # noqa: E402
from diagnose_selection_validation import RETRIEVAL, dump, jsonl, read, sha  # noqa: E402

TRANSFER=PROJECT/'outputs/results/qwen/validation/iterations/selection_r1_map_transfer_20261123_v1'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    destination=args.destination
    started=time.monotonic()
    prereg=read(destination/'preregistration.json')
    if prereg['quantiles']!=[.1,.3,.5,.7,.9]:
        raise ValueError('不增加或改选分位数')
    sources=[Path(__file__),PROJECT/'conceptgraph/completion/selection_map_transfer.py',
        PROJECT/'conceptgraph/navigation/semantic.py',PROJECT/'conceptgraph/completion/evidence.py',
        PROJECT/'conceptgraph/scenegraph/relations.py',RETRIEVAL/'queries.jsonl',TRANSFER/'queries.jsonl',
        TRANSFER/'report.json',destination/'preregistration.json']
    fingerprints={str(p):sha(p) for p in sources}
    release=read(RETRIEVAL/'release.json')
    r1_paths=sorted(R1.glob('scene-*/frames.jsonl.gz'))
    inputs={**release['source_frames_sha256'],**{str(p):sha(p) for p in r1_paths}}
    dump(destination/'release.json',dict(source_sha256=fingerprints,input_frames_sha256=inputs,
        new_inference_only_scores=True,new_model_calls=0,gt_in_threshold_definition=False,
        default_changed=False,original_report_changed=False))
    for path in sources:
        if path.suffix=='.py':
            target=destination/'source_snapshot'/path.relative_to(PROJECT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
    queries=defaultdict(list)
    for row in jsonl(RETRIEVAL/'queries.jsonl'):
        if row['method']=='existing_object_joint_rule':
            # 不把positive、正确性、实例或可接受集合传递到评分收集阶段。
            queries[(row['scene'],row['frame_idx'])].append(
                dict(query_index=row['query_index'],query_text=row['query_text']))
    old_frames={}
    for filename in release['source_frames_sha256']:
        for frame in jsonl(filename):
            key=(frame['scene'],frame['frame_idx'])
            if key in queries:
                old_frames[key]=frame
    new_frames={}
    for path in r1_paths:
        with gzip.open(path,'rt') as handle:
            for line in handle:
                frame=json.loads(line)
                key=(frame['scene'],frame['frame_idx'])
                if key in queries:
                    new_frames[key]=frame
    scores=[]
    for key,group in queries.items():
        old,new=old_frames[key],new_frames[key]
        pose=np.asarray(old['camera_pose'])
        for query in group:
            text=query['query_text']
            if any(word in text or word in normalize_text(text) for word in FAR_WORDS):
                raise ValueError('出现图内归一化far分支，本固定评分比较需要重新解释，停止')
            for method,predictions in [('old_map_original_rule',old['predictions']['one_to_one_depth_quality']),
                                       ('R1_map_original_rule',new['predictions']['R1'])]:
                ranked=select_frozen_map(text,predictions,pose,key[1])
                value=float(ranked[0].score) if ranked else None
                if value is not None and not np.isfinite(value):
                    raise ValueError('评分不有限')
                scores.append(dict(scene=key[0],frame_idx=key[1],query_index=query['query_index'],
                    method=method,selected_node_id=ranked[0].node_id if ranked else None,score=value))
        if time.monotonic()-started>1200 or shutil.disk_usage(PROJECT).free<20*1024**3:
            raise RuntimeError('达到20分钟预算或20GiB停写线')
    with (destination/'inference_only_scores.jsonl').open('x') as handle:
        for row in scores:
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    pool=np.array([r['score'] for r in scores if r['score'] is not None])
    if len(pool)==0:
        raise ValueError('无原始连续评分，停止工作点分析')
    thresholds=np.quantile(pool,prereg['quantiles']).tolist()
    frozen=dict(quantiles=prereg['quantiles'],thresholds=thresholds,score_count=len(pool),
        unique_score_count=len(set(pool.tolist())),score_sha256=sha(destination/'inference_only_scores.jsonl'),
        gt_in_threshold_definition=False,algorithm='numpy.quantile default linear; shared pooled scores',
        score_is_calibrated_probability=False)
    dump(destination/'thresholds_frozen_before_evaluation.json',frozen)
    # 到此阈值已落盘，之后才读取正负、身份和正确性作评价。
    evaluation=list(jsonl(TRANSFER/'queries.jsonl'))
    lookup={(r['scene'],r['frame_idx'],r['query_index'],r['method']):r for r in scores}
    totals=defaultdict(Counter)
    records=[]
    points=[('origin',None)]+[(f'q{int(q*100):02d}',t) for q,t in zip(prereg['quantiles'],thresholds)]
    for row in evaluation:
        score=lookup[(row['scene'],row['frame_idx'],row['query_index'],row['method'])]
        if score['selected_node_id']!=row['selected_node_id']:
            raise ValueError('原top候选改变，不能声称只增加拒选')
        for point,threshold in points:
            selected=bool(row['selected'] and (threshold is None or score['score']>=threshold))
            correct=bool(row['positive'] and selected and row['selected_correct'])
            false=bool(not row['positive'] and selected)
            totals[(row['method'],point)].update(queries=1,positive=int(row['positive']),
                negative=int(not row['positive']),correct=int(correct),false_selection=int(false),
                rejected=int(not selected),selected=int(selected))
            records.append(dict(scene=row['scene'],frame_idx=row['frame_idx'],query_index=row['query_index'],
                method=row['method'],point=point,threshold=threshold,original_score=score['score'],
                selected=selected,selected_correct=correct if row['positive'] else None,
                false_selection_on_no_match=false if not row['positive'] else None,
                positive=row['positive'],new_vehicle_authorization_evaluated=False))
    original=read(TRANSFER/'report.json')['methods']
    table=[]
    methods=prereg['methods']
    for method in methods:
        previous_selected=1115
        for point,threshold in points:
            c=totals[(method,point)]
            if (c['queries'],c['positive'],c['negative'])!=(1115,733,382):
                raise ValueError('阈值表分母变化')
            if point=='origin':
                expected=original[method]
                if [c[k] for k in ('correct','false_selection','rejected')]!=[
                        expected['positive_selected_correct'],expected['no_match_false_selection']['numerator'],
                        expected['no_selection']]:
                    raise ValueError('原点未复现')
            if c['selected']>previous_selected:
                raise ValueError('阈值升高却增加选择，不是单一拒选策略')
            previous_selected=c['selected']
            table.append(dict(method=method,point=point,threshold=threshold,**dict(c),
                positive_correct_rate=c['correct']/733,negative_false_rate=c['false_selection']/382,
                rejection_rate=c['rejected']/1115))
    with (destination/'all_workpoint_decisions.jsonl').open('x') as handle:
        for row in records:
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    with (destination/'workpoints.csv').open('x') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    risk_choices={}
    for method in methods:
        eligible=[r for r in table if r['method']==method and r['false_selection']<=8]
        risk_choices[method]=max(eligible,key=lambda r:(r['correct'],-r['false_selection'])) if eligible else None
    if any(sha(path)!=expected for path,expected in {**fingerprints,**inputs}.items()):
        raise ValueError('原输入/源码变化')
    dump(destination/'report.json',dict(all_shared_workpoints=table,development_risk_reference=8,
        development_selected_workpoints=risk_choices,thresholds_frozen_before_GT=True,
        all_original_points_reproduced=True,same_thresholds_applied_to_both_maps=True,
        only_existing_rule_score_used=True,primary_evaluation_3m_unchanged=True,
        original_evidence_gates_unchanged=True,new_model_calls=0,independent_test=False,
        default_changed=False,elapsed_s=time.monotonic()-started,
        limitation='开发集上的显式系统拒选策略；不是模型本身提升或独立测试/显著泛化收益',
        decision_rows=len(records),score_sha256=frozen['score_sha256'],
        decisions_sha256=sha(destination/'all_workpoint_decisions.jsonl')))
    print(json.dumps(dict(workpoints=table,development_risk_choices=risk_choices,
        elapsed_s=time.monotonic()-started),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
