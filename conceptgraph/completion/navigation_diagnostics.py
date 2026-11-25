"""开发范围的对象选择上限诊断；GT条件单列，绝不写入推理缓存。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from conceptgraph.completion.quality import relations_from_centers
from conceptgraph.completion.semantic_dataset import CATEGORIES, RELATIONS, query_response
from conceptgraph.completion.stage9 import OUTPUT, dump, digest
from conceptgraph.navigation.semantic import parse_navigation_query, apply_qwen_query_parse, rank_candidate_targets
from conceptgraph.navigation.types import SceneGraphNode
from conceptgraph.scenegraph.relations import build_relations


def queries_for_frame(frame,motion):
    truth=frame['evaluation_only_official']
    static=[i for i,o in enumerate(truth) if motion.get(o['annotation_token'])=='static']
    pose=np.asarray(frame['camera_pose'])
    xy=pose[:2,3]
    forward=pose[:2,2]
    queries=[]
    for cat in sorted({truth[i]['category'] for i in static}):
        options=[i for i in static if truth[i]['category']==cat]
        distance={i:np.linalg.norm(np.asarray(truth[i]['center'][:2])-xy) for i in options}
        acceptable=[truth[i]['instance_token'] for i in options if distance[i]<=min(distance.values())+.5]
        queries.append(dict(query_text=f'去最近的停着的{CATEGORIES[cat]}附近',
            response=query_response(cat,nearest=True,static=True),acceptable_instances=acceptable,
            query_type='nearest_static_positive'))
    # 无匹配只在该帧被官方标注覆盖的相机范围内成立，不声明整条道路不存在。
    absent=next((c for c in CATEGORIES if not any(truth[i]['category']==c for i in static)),None)
    if absent:
        queries.append(dict(query_text=f'去最近的停着的{CATEGORIES[absent]}附近',
            response=query_response(absent,nearest=True,static=True),acceptable_instances=[],
            query_type='no_static_match_in_current_annotated_view'))
    relations=relations_from_centers([dict(id=i,center=o['center']) for i,o in enumerate(truth)],
        forward_xy=forward,max_distance_m=12,component_margin_m=1,near_m=6)
    chosen=set()
    for i,rel,j in sorted(relations):
        if i not in static or rel=='near':
            continue
        target,anchor=truth[i]['category'],truth[j]['category']
        key=(target,anchor,rel)
        if key in chosen:
            continue
        chosen.add(key)
        accepted={truth[a]['instance_token'] for a,r,b in relations if r==rel and a in static
            and truth[a]['category']==target and truth[b]['category']==anchor}
        queries.append(dict(query_text=f'去{CATEGORIES[anchor]}{RELATIONS[rel]}停着的{CATEGORIES[target]}旁边',
            response=query_response(target,anchor=anchor,relation=rel,static=True),
            acceptable_instances=sorted(accepted),query_type='relation_static_positive'))
        if len(chosen)>=2:
            break
    return queries


def to_nodes(predictions):
    return {i:SceneGraphNode(i,o['category'],[o['category']],o['category'],
        np.asarray(o['center']),np.asarray(o['extent']),motion_state=o['motion_state'],
        is_short_term=o['motion_state']=='dynamic',
        metadata=dict(num_detections=o['num_detections'],depth_evidence=o['depth_evidence']))
        for i,o in enumerate(predictions)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mapping-run',default='robust_motion_full_001')
    parser.add_argument('--method',default='one_to_one_depth_quality')
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    dest=OUTPUT/'replay'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    labelpath=OUTPUT/'annotation/official_motion_dev_001/evaluation_only_motion.jsonl'
    labels={(r['scene'],r['frame_idx']):{o['annotation_token']:o['official_motion'] for o in r['objects']}
        for r in map(json.loads,labelpath.open())}
    totals=defaultdict(Counter)
    scene_totals=defaultdict(lambda:defaultdict(Counter))
    policy=dict(enabled=True,minimum_observations=3,minimum_lidar_pixels=5,minimum_depth_support_fraction=.4)
    records=[]
    for path in sorted((OUTPUT/'mapping_evaluation'/args.mapping_run).glob('scene-*/frames.jsonl')):
        for frame in map(json.loads,path.open()):
            if frame['frame_idx']%4:
                continue
            predictions=frame['predictions'][args.method]
            truth=frame['evaluation_only_official']
            mapping={m['prediction_index']:truth[m['truth_index']]['instance_token'] for m in frame['metrics'][args.method]['matches']}
            nodes=to_nodes(predictions)
            pose=np.asarray(frame['camera_pose'])
            forward=pose[:2,2]
            edges=build_relations(nodes,dict(coordinate_frame='ego_heading',
                directional_mode='all_supported_axes'),frame_index=frame['frame_idx'],ego_heading=forward).edges
            for qidx,q in enumerate(queries_for_frame(frame,labels[(frame['scene'],frame['frame_idx'])])):
                acceptable=set(q['acceptable_instances'])
                rule=parse_navigation_query(q['query_text'],8)
                correct=apply_qwen_query_parse(rule,q['response'],8)
                conditions=[('rule_parse_no_evidence_gate',rule,None,False),
                    ('correct_parse_no_evidence_gate',correct,None,True),
                    ('correct_parse_depth_gate',correct,policy,True)]
                for condition,parsed,evidence,oracle in conditions:
                    parsed.vehicle_heading_xy=forward
                    ranked=rank_candidate_targets(parsed,nodes,edges,pose[:2,3],.55,.3,.25,
                        relation_coordinate_frame='ego_heading',relation_coordinate_frame_fallback=None,
                        evidence_policy=evidence)
                    selected=ranked[0].node_id if ranked else None
                    identity=mapping.get(selected)
                    positive=bool(acceptable)
                    row=dict(scene=frame['scene'],frame_idx=frame['frame_idx'],query_index=qidx,
                        query_text=q['query_text'],query_type=q['query_type'],condition=condition,
                        ground_truth_assisted=oracle,acceptable_instances=sorted(acceptable),
                        candidate_ids=[c.node_id for c in ranked],selected_node_id=selected,
                        evaluation_associated_instance=identity,
                        selected_correct_by_evaluation_association=identity in acceptable if positive else None,
                        false_selection_on_no_match=selected is not None if not positive else None,
                        correct_target_in_candidates=any(mapping.get(c.node_id) in acceptable for c in ranked) if positive else None,
                        unknown_selection_association=selected is not None and identity is None)
                    counts=dict(queries=1,positive_queries=int(positive),negative_queries=int(not positive),
                        positive_selected_correct=int(positive and identity in acceptable),
                        positive_contains_correct=int(positive and any(mapping.get(c.node_id) in acceptable for c in ranked)),
                        no_match_false_selection=int(not positive and selected is not None),
                        unknown_selection_association=int(selected is not None and identity is None),
                        no_selection=int(selected is None))
                    for k,v in counts.items():
                        totals[condition][k]+=v
                        scene_totals[frame['scene']][condition][k]+=v
                    records.append(row)
    report={}
    for name,c in totals.items():
        report[name]=dict(c,positive_selection_rate=c['positive_selected_correct']/c['positive_queries'] if c['positive_queries'] else None,
            positive_candidate_coverage=c['positive_contains_correct']/c['positive_queries'] if c['positive_queries'] else None,
            no_match_false_selection_rate=c['no_match_false_selection']/c['negative_queries'] if c['negative_queries'] else None)
    with (dest/'queries.jsonl').open('x') as f:
        for row in records:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    dump(dest/'report.json',dict(method=args.method,mapping_run=args.mapping_run,conditions=report,
        scenes=scene_totals,source_sha256=digest(Path(__file__)),motion_label_sha256=digest(labelpath),
        evidence_policy=policy,scope='每4帧，当前CAM_FRONT官方标注范围，1—65m，限定六类；非新测试',
        correct_parse_conditions='诊断上限，直接提供正确解析；不计实际系统成绩，不写推理缓存',
        target_containment_condition='正确对象在候选内的数量用于候选上限；保留所有候选缺失的正查询',
        correct_object_condition='尚未运行运动学执行；给定正确对象的车辆规划/到达上限仍未完成',
        selection_identity_limit='使用独立类别/中心门限关联，未知关联单列；尚无人工实例审核，非已证明精确身份',
        routes_generated=0,stable_locks_evaluated=False,actual_arrivals=0,
        stage9_complete=False,goal_complete=False))
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
