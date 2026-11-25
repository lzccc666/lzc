"""选择证据的任务互斥/节点累计诊断；不改变预测、标签或门限。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from conceptgraph.completion.evidence import target_evidence_decision
from conceptgraph.completion.selection_map_transfer import (
    EVIDENCE_POLICY, nodes_from_frozen_predictions, select_frozen_map)
from conceptgraph.navigation import semantic as sem
from conceptgraph.project_paths import PROJECT_ROOT as ROOT, experiment_directory
from conceptgraph.scenegraph.relations import build_relations

BASE=ROOT/'outputs/project_v2/thesis_completion_v1'
TRANSFER=ROOT/'outputs/results/qwen/validation/iterations/selection_r1_map_transfer_20261123_v1'
FOLDER=ROOT/'dataset/processed/navstage2_repair_20261124_v1'
RUN='navstage2_repair_20261124_diagnosis_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def lines(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as f:
        for s in f:yield json.loads(s)
def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def gate_trace(query,predictions,pose,frame_idx):
    """镜像现有rank顺序，额外记录其全部可观察失败门；必须与实际候选逐节点一致。"""
    nodes=nodes_from_frozen_predictions(predictions)
    forward=pose[:2,2];vehicle_xy=pose[:2,3]
    edges=build_relations(nodes,dict(coordinate_frame='ego_heading',directional_mode='all_supported_axes'),
        frame_index=frame_idx,ego_heading=forward).edges
    parsed=sem.parse_navigation_query(query,8.);parsed.vehicle_heading_xy=forward
    relation_index=sem.build_relation_index(edges)
    anchors=({i:sem.score_text_similarity(parsed.anchor_text,n) for i,n in nodes.items()}
        if parsed.relation is not None and parsed.anchor_text is not None else None)
    ranked=select_frozen_map(query,predictions,pose,frame_idx);ranked_ids={c.node_id for c in ranked}
    result=[]
    for i,node in nodes.items():
        obj=predictions[i];fail=[]
        if parsed.requires_static and (node.motion_state!='static' or node.is_short_term):fail.append('query_requires_confirmed_static')
        terms=sem.node_category_terms(node)
        if any(terms & sem.CLASS_COMPATIBILITY.get(c,{c}) for c in parsed.excluded_categories):fail.append('excluded_category')
        if parsed.excluded_colors:
            colors=node.metadata.get('attributes',{}).get('colors',[])
            if not colors or 'unknown' in colors or any(sem.node_has_color(node,[c]) for c in parsed.excluded_colors):fail.append('excluded_color_unknown_or_present')
        evidence,reason=target_evidence_decision(node,EVIDENCE_POLICY)
        if not evidence:fail.append('evidence:'+reason)
        compatible,compatibility=sem.is_node_compatible_with_query(parsed,node)
        if not compatible:fail.append('compatibility:'+compatibility)
        text_score=sem.score_text_similarity(parsed.target_text,node)
        if not parsed.target_categories and parsed.query_mode!='nearest_static' and text_score<=0:fail.append('no_text_similarity')
        score,debug=sem.score_relation_constraint(node,parsed,nodes,relation_index,anchor_text_scores=anchors,
            minimum_edge_confidence=0.,fallback_coordinate_frame='ego_heading',fallback_coordinate_frame_fallback=None,
            fallback_ego_heading=forward,fallback_trajectory=None,fallback_ego_position=vehicle_xy,
            relation_support_policy=None)
        if parsed.relation is not None and score<=.05:fail.append('relation_constraint')
        distance=float(np.linalg.norm(node.bbox_center[:2]-vehicle_xy))
        if parsed.min_target_distance_m is not None and distance<parsed.min_target_distance_m:fail.append('minimum_target_distance')
        if parsed.max_target_distance_m is not None and distance>parsed.max_target_distance_m:fail.append('maximum_target_distance')
        if parsed.prefer_front and sem._front_score(node.bbox_center[:2],vehicle_xy,forward)<=0:fail.append('front_constraint')
        actual=i in ranked_ids
        if actual != (not fail):raise AssertionError(f'gate_mirror_mismatch:{query}:{obj["node_id"]}:{fail}')
        missing=[k for k in ['category','center','num_detections','motion_state','depth_evidence'] if k not in obj]
        latest=obj.get('depth_evidence',[])[-1] if obj.get('depth_evidence') else {}
        result.append(dict(prediction_index=i,node_id=obj['node_id'],category=obj['category'],motion_state=node.motion_state,
            num_detections=obj['num_detections'],velocity_mps=obj.get('velocity_mps'),shape_available=obj.get('shape_available',True),
            first_failed_gate=fail[0] if fail else None,all_failed_gates=fail,candidate=actual,
            evidence_reason=reason,compatibility=compatibility,relation_score=score,relation_debug=debug,
            missing_fields=missing,depth_latest={k:latest.get(k) for k in ['source','units','retained_pixels','support_fraction']},
            parsed=dict(target_text=parsed.target_text,target_categories=parsed.target_categories,
                anchor_text=parsed.anchor_text,relation=parsed.relation,requires_static=parsed.requires_static)))
    return ranked,result


def prepare():
    FOLDER.mkdir(exist_ok=False)
    source=BASE/'main_experiments/causal_retrieval_full_001';release=read(source/'release.json')
    transfer=read(TRANSFER/'release.json')
    queries=[r for r in lines(source/'queries.jsonl') if r['method']=='existing_object_joint_rule']
    assert len(queries)==1115 and sum(r['positive'] for r in queries)==733
    scene_logs={r['name']:r['log_token'] for r in read(ROOT/'dataset/nuscenes_trainval_raw/v1.0-trainval/scene.json')}
    scenes=sorted({r['scene'] for r in queries});assert len(scenes)==38
    historical=[r for r in lines(ROOT/'dataset/processed/navstage2_20261124_v2/method_inputs.jsonl')]
    hist_scenes=sorted({r['scene'] for r in historical})
    overlap_logs=sorted({scene_logs[s] for s in scenes}&{scene_logs[s] for s in hist_scenes})
    overlap_scenes=sorted(set(scenes)&set(hist_scenes));assert not overlap_scenes and not overlap_logs
    qpath=FOLDER/'validation_queries.jsonl';qpath.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in queries))
    source_paths=[Path(__file__),ROOT/'scripts/navstage2_repair_20261124_diagnose.py',
        ROOT/'conceptgraph/completion/selection_map_transfer.py',ROOT/'conceptgraph/completion/evidence.py',
        ROOT/'conceptgraph/navigation/semantic.py',ROOT/'conceptgraph/navigation/types.py',
        ROOT/'conceptgraph/scenegraph/relations.py',ROOT/'conceptgraph/project_paths.py']
    inputs={str(qpath):sha(qpath),str(source/'release.json'):sha(source/'release.json'),
        str(TRANSFER/'queries.jsonl'):sha(TRANSFER/'queries.jsonl'),str(TRANSFER/'release.json'):sha(TRANSFER/'release.json')}
    inputs.update(release['source_frames_sha256']);inputs.update(transfer['R1_frames_sha256'])
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=17,
        purpose='完整验证诊断；若发现可靠bug再另行冻结修复；无可靠bug不盲改',
        validation_tasks=1115,positive=733,negative=382,scenes=scenes,scene_logs={s:scene_logs[s] for s in scenes},
        validation_role='38 already seen debugging/validation scenes; not training or unseen test',
        historical81_role='separate explanation ledger only; never choose method, thresholds or branches from it',
        historical81_scenes=hist_scenes,scene_overlap=overlap_scenes,log_overlap=overlap_logs,
        methods=['old_map_original_rule','R1_map_original_rule'],policy=EVIDENCE_POLICY,
        budget=dict(wall_s=1800,new_bytes=128*1024**2,GPU_calls=0,minimum_disk_free_bytes=20*1024**3),
        source_sha256={str(p):sha(p) for p in source_paths},input_sha256=inputs,
        grouping='任务：正例按图无正确对象/有图无候选/有候选未选对/选对互斥；图有对象无候选子组=所有正确节点首拒门的去重排序组合；节点累计与任务分开，all_failed_gates非互斥',
        no_parameter_or_threshold_change=True)
    dump(FOLDER/'manifest.json',manifest);print(str(FOLDER/'manifest.json'),sha(FOLDER/'manifest.json'))


def frozen_verify(manifest):
    for group in ['source_sha256','input_sha256']:
        for p,expected in manifest[group].items():
            if sha(p)!=expected:raise ValueError('frozen_changed:'+p)


def run_validation():
    begun=time.monotonic();manifest=read(FOLDER/'manifest.json');frozen_verify(manifest)
    dest=experiment_directory('nuscenes','validation',RUN,create=True);shutil.copy2(FOLDER/'manifest.json',dest/'manifest.json')
    for p in manifest['source_sha256']:
        target=dest/'source_snapshot'/Path(p).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    queries=list(lines(FOLDER/'validation_queries.jsonl'));keys={(r['scene'],r['frame_idx']) for r in queries}
    old={};new={}
    release=read(BASE/'main_experiments/causal_retrieval_full_001/release.json')
    for p in release['source_frames_sha256']:
        for f in lines(p):
            if (f['scene'],f['frame_idx']) in keys:old[f['scene'],f['frame_idx']]=f
    for p in read(TRANSFER/'release.json')['R1_frames_sha256']:
        for f in lines(p):
            if (f['scene'],f['frame_idx']) in keys:new[f['scene'],f['frame_idx']]=f
    previous={(r['method'],r['scene'],r['frame_idx'],r['query_index']):r for r in lines(TRANSFER/'queries.jsonl')}
    tasks=[];summary={};allnodes=defaultdict(Counter);acceptednodes=defaultdict(Counter)
    with gzip.open(dest/'nodes.jsonl.gz','wt') as nodeout:
        for method in manifest['methods']:
            counts=Counter();subgroups=Counter();allgates=Counter();fields=Counter();bylog=defaultdict(Counter)
            for query in queries:
                key=query['scene'],query['frame_idx'];frame=old[key];alt=new[key]
                predictions=(frame['predictions']['one_to_one_depth_quality'] if method=='old_map_original_rule' else alt['predictions']['R1'])
                ranked,traces=gate_trace(query['query_text'],predictions,np.asarray(frame['camera_pose']),key[1])
                metrics=(frame['metrics']['one_to_one_depth_quality'] if method=='old_map_original_rule' else alt['metrics']['R1'])
                truth=frame['evaluation_only_official'];binding={m['prediction_index']:truth[m['truth_index']]['instance_token'] for m in metrics['matches']}
                acceptable=set(query['acceptable_instances']);correct_nodes=[t for t in traces if binding.get(t['prediction_index']) in acceptable]
                selected=ranked[0].node_id if ranked else None;identity=binding.get(selected)
                prev=previous[method,key[0],key[1],query['query_index']]
                assert prev['candidate_ids']==[c.node_id for c in ranked] and prev['evaluation_identity']==identity and prev['selected_node_id']==selected
                covered=bool(correct_nodes);candidate=any(t['candidate'] for t in correct_nodes);correct=identity in acceptable
                if not query['positive']:stage='negative_false_selection' if selected is not None else 'negative_correct_rejection'
                elif not covered:stage='no_accepted_object_in_graph'
                elif not candidate:stage='accepted_object_in_graph_but_no_candidate'
                elif not correct:stage='accepted_candidate_but_not_selected_correct'
                else:stage='selected_correct'
                counts[stage]+=1;counts['tasks']+=1;bylog[manifest['scene_logs'][key[0]]][stage]+=1
                first_set=sorted({t['first_failed_gate'] for t in correct_nodes if t['first_failed_gate']})
                if stage=='accepted_object_in_graph_but_no_candidate':subgroups[' | '.join(first_set)]+=1
                for gate in {g for t in correct_nodes for g in t['all_failed_gates']}:allgates[gate]+=1
                for t in traces:
                    is_correct=binding.get(t['prediction_index']) in acceptable
                    allnodes[method]['nodes']+=1;allnodes[method]['first:'+(t['first_failed_gate'] or 'passed')]+=1
                    for gate in t['all_failed_gates']:allnodes[method]['all:'+gate]+=1
                    for field in t['missing_fields']:fields[field]+=1
                    if is_correct:
                        acceptednodes[method]['nodes']+=1;acceptednodes[method]['first:'+(t['first_failed_gate'] or 'passed')]+=1
                        acceptednodes[method]['motion:'+t['motion_state']]+=1
                        for gate in t['all_failed_gates']:acceptednodes[method]['all:'+gate]+=1
                    nodeout.write(json.dumps(dict(method=method,scene=key[0],frame_idx=key[1],query_index=query['query_index'],
                        evaluation_accepted_node=is_correct,**t),ensure_ascii=False,allow_nan=False)+'\n')
                tasks.append(dict(method=method,scene=key[0],frame_idx=key[1],query_index=query['query_index'],
                    query_text=query['query_text'],stage=stage,positive=query['positive'],
                    correct_node_count=len(correct_nodes),correct_candidate_count=sum(t['candidate'] for t in correct_nodes),
                    selected_correct=correct,selected_index=selected,selected_identity=identity,
                    task_first_gate_set=first_set,correct_node_traces=correct_nodes))
                if time.monotonic()-begun>1800 or shutil.disk_usage(ROOT).free<20*1024**3:raise RuntimeError('budget')
            summary[method]=dict(tasks=dict(counts),blocked_task_mutually_exclusive_subgroups=dict(subgroups),
                accepted_node_task_all_gate_counts_nonexclusive=dict(allgates),missing_field_node_counts=dict(fields),
                by_log={k:dict(v) for k,v in bylog.items()})
    (dest/'tasks.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in tasks))
    frozen_verify(manifest)
    dump(dest/'summary.json',dict(methods=summary,all_nodes={k:dict(v) for k,v in allnodes.items()},
        accepted_nodes={k:dict(v) for k,v in acceptednodes.items()},queries_per_method=1115,
        baseline_choices_and_candidates_exact=True,gate_trace_mirrors_actual_candidates=True,
        changed_policy=False,new_model_calls=0,elapsed_s=time.monotonic()-begun,
        source_and_inputs_unchanged=True))
    print(json.dumps(summary,ensure_ascii=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','validation']);args=p.parse_args()
    {'prepare':prepare,'validation':run_validation}[args.action]()
