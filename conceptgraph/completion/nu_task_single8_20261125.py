"""New single8 task protocol over all2315 historical source instructions.

Old tasks/labels/results remain intact. Only directional acceptance is rebuilt
from independent official centers, original reference IDs, motion and camera
pose. No predictions or method results participate in GT preparation.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import numpy as np

from conceptgraph.completion import nu_task_replay_20261125 as base
from conceptgraph.navigation.single8_task_20261125 import parse_query,pair_matches,select
from conceptgraph.scenegraph.relation_single8_20261125 import build_edges_from_centers

ROOT=base.ROOT
OLD_DATA=ROOT/'dataset/processed/nu_task_replay_20261125_v1'
DATA=ROOT/'dataset/processed/nu_task_single8_20261125_v1'
REPORT=ROOT/'outputs/reports/nu_task_single8_20261125'
PRODUCER=ROOT/'outputs/results/nuscenes/validation/iterations/nuscenes_full_20261125_single8_main_v1'
CHANNELS=('retrieval','confirmation')
MOTION=ROOT/'outputs/project_v2/thesis_completion_v1/annotation/official_motion_dev_001/evaluation_only_motion.jsonl'
POLICY=dict(seed=17,configuration_count=1,split='seen_validation_debug_regression',all_original_tasks=2315,
    direction='shared single8 45deg half-open; source relative target; zero front, CCW left',
    relation_matching='front_left satisfies left, front and front_left; inverse computed without storing extra edges',
    graph_adjacency_m=12.,near_query_default_m=6.,explicit_near_distance='literal instruction meters',
    frame='CAM_FRONT optical forward projected worldXY, not object heading',
    affected_GT='directional queries only, original reference protocol and static target annotation unchanged',
    unchanged_GT='all class/nearest acceptance lists copied exactly; all original pending labels preserved',
    labels_scope='legacy1115 current front; unified1200 current front3; no side-view label extension',
    reference_protocol='legacy exists-any matching category; unified multiple references remains pending/clarification',
    legacy_nearest_tolerance_m=.5,unified_nearest_tolerance_m=1e-9,
    object_retrieval='common category/geometry/distance, all motion states participate; static unverified',
    static_confirmation='separate frozen raw evidence gate, unsupported baseline not zero retrieval improvement',
    GT_in_method=False,Qwen_calls=0,GPU_calls=0,trajectories_generated=0)


def contexts(tasks,truth):
    needed={(t['scene'],t['original_frame_idx']) for t in tasks if t['protocol']=='legacy1115'}
    poses={};inputs=[MOTION]
    for scene in sorted({k[0] for k in needed}):
        path=base.OLD_MAP/scene/'frames.jsonl';inputs.append(path)
        for frame in base.rows(path):
            if (scene,frame['frame_idx']) in needed:
                poses['legacy1115',scene,frame['frame_idx']]=frame['camera_pose']
    s2=base.S2/'method_frames.jsonl';inputs.append(s2)
    for frame in base.rows(s2):
        if frame['cohort']=='S2':poses['unified1200',frame['scene'],frame['frame_idx']]=frame['camera_pose']
    motion={(r['scene'],r['frame_idx']):{o['annotation_token']:o['official_motion'] for o in r['objects']} for r in base.rows(MOTION)}
    result={}
    for key,frame in truth.items():
        objects=copy.deepcopy(frame['objects']);pose=np.asarray(poses[key],dtype=float)
        assert pose.shape==(4,4) and np.isfinite(pose).all()
        for obj in objects:
            assert np.isfinite(np.asarray(obj['center'][:2],dtype=float)).all()
            assert obj['instance_token'] and obj['annotation_token']
            if key[0]=='legacy1115':
                obj['evaluation_motion']=motion[key[1],key[2]][obj['annotation_token']]
            else:
                assert obj['evaluation_motion'] in ('static','dynamic','unknown')
        assert len({o['instance_token'] for o in objects})==len(objects)
        result[key]=dict(**{k:v for k,v in frame.items() if k!='objects'},objects=objects,camera_pose=pose.tolist())
    return result,inputs


def directional_answer(task,old,context):
    if old['label_status']!='scorable':return copy.deepcopy(old)
    parsed=parse_query(task['query_text']);assert parsed['supported'] and parsed['relation']
    truth=context['objects'];pose=np.asarray(context['camera_pose']);forward=pose[:2,2]
    edges=build_edges_from_centers([dict(id=o['instance_token'],center=o['center']) for o in truth],forward,max_distance_m=12.)
    references=[o for o in truth if o['category'] in parsed['anchor_categories']]
    if task['protocol']=='unified1200':
        assert len(references)<=1,'scorable original reference ambiguity cannot be reinterpreted'
    positions={o['instance_token']:o['center'] for o in truth};eligible=[]
    for obj in truth:
        if obj['category'] not in parsed['target_categories'] or obj['evaluation_motion']!='static':continue
        if any(pair_matches(edges,obj['instance_token'],anchor['instance_token'],parsed['relation'],
                            near_m=parsed['near_m'],positions=positions)[0] for anchor in references):
            eligible.append(obj)
    if task['protocol']=='unified1200' and eligible:
        distances={o['instance_token']:float(np.linalg.norm(np.asarray(o['center'][:2])-pose[:2,3])) for o in eligible}
        closest=min(distances.values());accepted=sorted(k for k,d in distances.items() if abs(d-closest)<=1e-9)
    else:
        accepted=sorted(o['instance_token'] for o in eligible)
    result=copy.deepcopy(old);result.update(positive=bool(accepted),acceptable_instances=accepted)
    if 'reason' in result:
        result['reason']='single8_directional_predicate_recomputed_from_official_geometry'
    return result


def prepare():
    if DATA.exists():raise FileExistsError(DATA)
    old_manifest=base.verify();tasks=list(base.rows(OLD_DATA/'method_tasks.jsonl'))
    labels={r['task_id']:r for r in base.rows(OLD_DATA/'evaluation_only/task_labels.jsonl')}
    truth={(r['protocol'],r['scene'],r['original_frame_idx']):r for r in base.rows(OLD_DATA/'evaluation_only/frame_truth.jsonl')}
    assert len(tasks)==len(labels)==2315 and len(truth)==502
    assert all(parse_query(t['query_text'])['supported'] for t in tasks)
    context,input_paths=contexts(tasks,truth)
    sources=[Path(__file__),ROOT/'scripts/nu_task_single8_20261125.py',
             ROOT/'conceptgraph/navigation/single8_task_20261125.py',ROOT/'conceptgraph/scenegraph/relation_single8_20261125.py']
    source_sha={**old_manifest['source_sha256'],**{str(p):base.sha(p) for p in sources}}
    input_sha={str(p):base.sha(p) for p in [OLD_DATA/'manifest.json',OLD_DATA/'method_tasks.jsonl',
        OLD_DATA/'evaluation_only/task_labels.jsonl',OLD_DATA/'evaluation_only/frame_truth.jsonl',*input_paths]}
    DATA.mkdir(parents=True);(DATA/'evaluation_only').mkdir();REPORT.mkdir(parents=True,exist_ok=True);base.guard()
    base.write(DATA/'preregistration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),policy=POLICY,
        source_sha256=source_sha,input_sha256=input_sha,method_results_read=False,
        expected_original_tasks=2315,expected_original_pending=60,expected_directional_queries=827,expected_class_queries=1488))
    output=[];ledger=[];counts=defaultdict(Counter)
    for task in tasks:
        old=labels[task['task_id']];parsed=parse_query(task['query_text']);key=task['protocol'],task['scene'],task['original_frame_idx']
        new=directional_answer(task,old,context[key]) if parsed['relation'] else copy.deepcopy(old)
        if not parsed['relation'] or old['label_status']!='scorable':assert new==old
        output.append(new);scorable=new['label_status']=='scorable';c=counts[task['protocol']]
        c['all']+=1;c['pending']+=int(not scorable);c['positive']+=int(scorable and new['positive']);c['negative']+=int(scorable and not new['positive'])
        ledger.append(dict(task_id=task['task_id'],directional=bool(parsed['relation']),
            old_label=old,new_label=new,acceptance_changed=old['acceptable_instances']!=new['acceptable_instances'],
            unchanged_class_or_pending_exact=(not parsed['relation'] or old['label_status']!='scorable') and new==old,
            source='independent official geometry, IDs, official motion and original camera pose; no predicted nodes'))
    assert sum(c['pending'] for c in counts.values())==60
    shutil.copy2(OLD_DATA/'method_tasks.jsonl',DATA/'method_tasks.jsonl')
    shutil.copy2(OLD_DATA/'evaluation_only/frame_truth.jsonl',DATA/'evaluation_only/frame_truth.jsonl')
    base.jsonl(DATA/'evaluation_only/task_labels.jsonl',output)
    base.jsonl(DATA/'evaluation_only/GT_context.jsonl',context.values())
    base.jsonl(DATA/'evaluation_only/label_change_ledger.jsonl',ledger)
    for path,digest in {**source_sha,**input_sha}.items():assert base.sha(path)==digest
    own=[DATA/'method_tasks.jsonl',DATA/'evaluation_only/task_labels.jsonl',DATA/'evaluation_only/frame_truth.jsonl',
         DATA/'evaluation_only/GT_context.jsonl',DATA/'evaluation_only/label_change_ledger.jsonl',DATA/'preregistration.json']
    base.write(DATA/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),policy=POLICY,
        task_counts={p:dict(c) for p,c in counts.items()},source_sha256=source_sha,
        frozen_inputs_sha256={**input_sha,**{str(p):base.sha(p) for p in own}},
        original_source_manifest_sha256=base.sha(OLD_DATA/'manifest.json'),scene_count=len({t['scene'] for t in tasks}),
        frame_tasks=len(truth),method_results_read=False,label_changes=sum(r['acceptance_changed'] for r in ledger),
        unaffected_class_queries=1488,original_pending_exact=60,
        incoming_main_run=str(PRODUCER),old_protocol_and_scores_preserved=True))
    base.write(REPORT/'GT_freeze_summary_v1.json',dict(task_counts={p:dict(c) for p,c in counts.items()},
        acceptance_changes=sum(r['acceptance_changed'] for r in ledger),all2315_retained=True,GT_frames=502,
        unaffected_class_queries_exact=1488,original_pending_exact=60,method_scores_read=False,
        manifest_sha256=base.sha(DATA/'manifest.json'),preregistration_sha256=base.sha(DATA/'preregistration.json')))
    print(json.dumps(dict(task_counts={p:dict(c) for p,c in counts.items()},label_changes=sum(r['acceptance_changed'] for r in ledger))))


def configure(channel):
    if channel not in CHANNELS:raise ValueError(channel)
    base.DATA=DATA;base.REPORT=REPORT;base.RUN_ID='nu_task_single8_20261125_'+channel
    base.select=lambda *args:select(*args,confirmed=channel=='confirmation')


def infer(scene,frame_file):
    canonical=PRODUCER/scene/'frames.jsonl.gz'
    if Path(frame_file).absolute()!=canonical.absolute():raise AssertionError('only_new_single8_formal_canonical_input_allowed')
    for channel in CHANNELS:
        configure(channel);base.infer(scene,frame_file)


def evaluate(scene):
    for channel in CHANNELS:
        configure(channel);base.evaluate(scene)


def contract():
    for phrase,expected in [('左前方','front_left'),('右前方','front_right'),('左后方','back_left'),('右后方','back_right')]:
        parsed=parse_query('去卡车'+phrase+'最近停着的小汽车附近')
        assert parsed['supported'] and parsed['relation']==expected and parsed['target_categories']==['car'] and parsed['anchor_categories']==['truck']
    edges=build_edges_from_centers([dict(id='car',center=[2,2,0]),dict(id='truck',center=[0,0,0])],[1,0])
    assert len(edges)==1
    for query in ['front','left','front_left']:
        assert pair_matches(edges,'car','truck',query)[0]
    assert pair_matches(edges,'truck','car','back_right')[0] and len(edges)==1
    assert not pair_matches(edges,'car','truck','right')[0]
    assert pair_matches(edges,'car','truck','near',near_m=3.)[0]
    assert not pair_matches(edges,'car','truck','near',near_m=2.)[0]
    assert parse_query('去卡车3米附近的小汽车旁边')['near_m']==3.
    print(json.dumps(dict(compound_parse=True,single_edge_query_inclusion=True,inverse_without_new_edges=True,near_distance=True,GPU_calls=0)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','contract','infer','evaluate'])
    parser.add_argument('--scene');parser.add_argument('--frame-file');args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='contract':contract()
    elif args.action=='infer':infer(args.scene,args.frame_file)
    else:evaluate(args.scene)


if __name__=='__main__':main()
