"""Frozen CPU validation of current relation/cache semantics and near6 subgoals."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import copy
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from conceptgraph.project_paths import experiment_directory
from conceptgraph.completion.quality import object_frame_metrics,relations_from_centers
from conceptgraph.completion.reference_regions import load_reference_prior
from conceptgraph.nuscenes.expansion_benchmark import iter_records
from conceptgraph.scenegraph.incremental_graph import IncrementalSceneGraph
from conceptgraph.scenegraph.relations import RelationNode
from conceptgraph.scenegraph.relation_nav_unified_20261125 import (
    POLICY,FRAME_SOURCE,CurrentRelationCache,full_current,canonical_payload,expanded_facts,reference_query_status)
from conceptgraph.navigation.relation_nav_unified_20261125 import adapt_nodes,select_task,engineering_approach_region,POLICY as NAV_POLICY

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'dataset/processed/relation_nav_unified_20261125_v1'
REPORT=ROOT/'outputs/reports/relation_nav_unified_20261125'
RUN_ID='relation_nav_unified_20261125_v1'
R1=ROOT/'outputs/results/nuscenes/validation/iterations/recall_refinement_R1_full_20261120_v2'
S2=ROOT/'outputs/results/nuscenes/validation/iterations/mapnext_20261124_S2_full_v1'
S2INPUT=ROOT/'dataset/processed/mapnext_20261124_inputs_v1'
RAW=ROOT/'dataset/nuscenes_trainval_raw/v1.0-trainval'
CATEGORIES={'car':'小汽车','truck':'卡车','bus':'公交车','pedestrian':'行人','bicycle':'自行车','motorcycle':'摩托车'}
DIRECTIONS={'left_of':'左侧','right_of':'右侧','in_front_of':'前方','behind':'后方'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def rows(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as stream:
        for line in stream:
            if line.strip():yield json.loads(line)
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
def jsonl(path,records):
    with Path(path).open('x') as stream:
        for value in records:stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def guard(begun):
    if time.monotonic()-begun>3600:raise TimeoutError('bounded_CPU_stage')
    if shutil.disk_usage(ROOT).free<20*1024**3:raise OSError('disk_below_20GiB')
    folders=[DATA,REPORT,experiment_directory('nuscenes','validation',RUN_ID)]
    if sum(p.stat().st_size for folder in folders if folder.exists() for p in folder.rglob('*') if p.is_file())>128*1024**2:
        raise OSError('relation_lane_storage_above_128MiB')


def contracts():
    cache=CurrentRelationCache();tests=[]
    sequence=[([5.99,0,0],[1,0],False,1),([6.01,0,0],[1,0],False,1),
        ([6.01,0,0],[1,0],True,2),([6.01,0,0],[-1,0],True,2),
        ([12.01,1.01,0],[-1,0],False,3),([5,1.01,0],[-1,0],False,3)]
    for idx,(center,heading,dynamic,observations) in enumerate(sequence):
        nodes={0:RelationNode(0,[0,0,0],last_seen_frame=idx,metadata={'timestamp_s':float(idx+1)}),
            1:RelationNode(1,center,dynamic=dynamic,observation_count=observations,last_seen_frame=idx,
                metadata={'timestamp_s':float(idx+1),'source_node_id':'a'})}
        if idx==5:nodes[2]=nodes.pop(1);nodes[2].node_id=2
        full=full_current(nodes,frame_index=idx,heading=heading,timestamp_s=idx+1.)
        inc=cache.update_snapshot(nodes,frame_index=idx,heading=heading,timestamp_s=idx+1.)
        tests.append(dict(case=idx,passed=canonical_payload(full.edges)==canonical_payload(inc.edges),statistics=inc.update_statistics))
    empty=cache.update_snapshot({},frame_index=6,heading=[1,0],timestamp_s=7.)
    tests.append(dict(case='empty_snapshot_expires_current_facts',passed=not empty.edges and not cache.pairs))
    tests.append(dict(case='unknown_object_heading_not_supported',passed=not reference_query_status('物体自身右侧','object_relative')['supported']))
    if not all(t['passed'] for t in tests):raise AssertionError('relation_cache_contract')
    return tests


def motion_from_attributes(attributes):
    if any(x in {'vehicle.parked','vehicle.stopped','pedestrian.standing','pedestrian.sitting_lying_down'} for x in attributes):return 'static'
    if any(x in {'vehicle.moving','pedestrian.moving'} for x in attributes):return 'dynamic'
    return 'unknown'


def prepare():
    begun=time.monotonic();DATA.mkdir(parents=True,exist_ok=False);REPORT.mkdir(parents=True,exist_ok=True)
    checks=contracts();frames=[];evaluation=[];input_paths=set();release=read(R1/'release.json')
    official_scenes={r['name']:r for r in read(RAW/'scene.json')};official_logs={r['token']:r for r in read(RAW/'log.json')}
    for scene in release['scenes']:
        name=scene['scene'];predpath=R1/name/'frames.jsonl.gz'
        rawpath=ROOT/'outputs/project_v2/thesis_completion_v1/mapping_evaluation/causal_mapping_full_001'/name/'frames.jsonl'
        raw={r['frame_idx']:r for r in rows(rawpath)};input_paths.update([predpath,rawpath])
        for row in rows(predpath):
            source=raw[row['frame_idx']];pose=np.asarray(source['camera_pose']);log=official_scenes[name]['log_token']
            frames.append(dict(cohort='R1',scene=name,log=log,frame_idx=row['frame_idx'],timestamp_s=source['camera_timestamp_s'],
                camera_pose=source['camera_pose'],heading=pose[:2,2].tolist(),frame_source=FRAME_SOURCE,
                location=official_logs[log]['location'],predictions=row['predictions']['R1']))
            evaluation.append(dict(cohort='R1',scene=name,frame_idx=row['frame_idx'],truth=row['evaluation_only_official']))
    s2manifest=read(S2INPUT/'all_manifest.json');s2gt={(r['scene'],r['frame_idx']):r for r in rows(S2INPUT/'all_GT.jsonl')}
    for scene in s2manifest['scenes']:
        name=scene['scene'];predpath=S2/name/'mapnext_20261124_predictions.jsonl.gz'
        predictions={r['frame_idx']:r for r in rows(predpath)};input_paths.add(predpath)
        for group in scene['groups']:
            row=predictions[group['frame_idx']];pose=np.asarray(group['views']['CAM_FRONT']['camera_pose']);log=scene['log_token']
            frames.append(dict(cohort='S2',scene=name,log=log,frame_idx=group['frame_idx'],timestamp_s=row['reference_timestamp_s'],
                camera_pose=pose.tolist(),heading=pose[:2,2].tolist(),frame_source=FRAME_SOURCE,
                location=official_logs[log]['location'],predictions=row['predictions']['S2_three']))
            truth=[o for o in s2gt[name,group['frame_idx']]['objects'] if set(o['visible_cameras'])&{'CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT'}]
            evaluation.append(dict(cohort='S2',scene=name,frame_idx=group['frame_idx'],truth=truth))
    assert sum(f['cohort']=='R1' for f in frames)==1508 and sum(f['cohort']=='S2' for f in frames)==120
    attrs={r['token']:r['name'] for r in read(RAW/'attribute.json')}
    wanted={o['annotation_token'] for r in evaluation if r['cohort']=='S2' for o in r['truth']};annotation_attrs={}
    for annotation in iter_records(RAW/'sample_annotation.json'):
        if annotation['token'] in wanted:
            annotation_attrs[annotation['token']]=[attrs[t] for t in annotation['attribute_tokens']]
        if len(annotation_attrs)==len(wanted):break
    for row in evaluation:
        if row['cohort']=='S2':
            for obj in row['truth']:
                obj['official_attributes']=annotation_attrs.get(obj['annotation_token'],[])
                obj['evaluation_motion']=motion_from_attributes(obj['official_attributes'])
    (DATA/'evaluation_only').mkdir()
    jsonl(DATA/'method_frames.jsonl',frames);jsonl(DATA/'evaluation_only/frames.jsonl',evaluation)
    task_inputs=[];task_labels=[]
    for frame,evaluation_frame in zip(frames,evaluation):
        if frame['cohort']!='S2':continue
        truth=evaluation_frame['truth'];xy=np.asarray(frame['camera_pose'])[:2,3]
        facts=relations_from_centers([dict(id=o['instance_token'],center=o['center']) for o in truth],forward_xy=frame['heading'],max_distance_m=12.,near_m=6.,component_margin_m=1.)
        specifications=[(cat,None) for cat in CATEGORIES]+[('car',rel) for rel in DIRECTIONS]
        for index,(cat,relation) in enumerate(specifications):
            text=(f'去最近停着的{CATEGORIES[cat]}附近' if relation is None else f'去卡车{DIRECTIONS[relation]}最近停着的小汽车附近')
            task_id=f"{frame['scene']}:{frame['frame_idx']}:{index}"
            task_inputs.append(dict(task_id=task_id,scene=frame['scene'],frame_idx=frame['frame_idx'],query_text=text,reference_frame='ego_heading'))
            category=[o for o in truth if o['category']==cat];anchors=[o for o in truth if o['category']=='truck'] if relation else []
            status='scorable';reason=None;eligible=[o for o in category if o['evaluation_motion']=='static']
            if any(o['evaluation_motion']=='unknown' for o in category):status='pending_motion_evidence';reason='official motion attribute absent/ambiguous; not a negative'
            if relation:
                if len(anchors)>1:status='pending_reference_ambiguity';reason='more than one official truck; no unique reference specified'
                elif not anchors:eligible=[];reason='no_reference_in_annotated_front3'
                else:eligible=[o for o in eligible if (o['instance_token'],relation,anchors[0]['instance_token']) in facts]
            if eligible:
                distance={o['instance_token']:float(np.linalg.norm(np.asarray(o['center'][:2])-xy)) for o in eligible};minimum=min(distance.values())
                acceptable=sorted(k for k,v in distance.items() if abs(v-minimum)<=1e-9)
            else:acceptable=[]
            task_labels.append(dict(task_id=task_id,label_status=status,reason=reason,positive=bool(acceptable) if status=='scorable' else None,
                acceptable_instances=acceptable,GT_reference_count=len(anchors),task_kind='static_object_approach',query_family='nearest_static' if relation is None else relation,
                no_match_scope='current official annotated three front camera view only'))
    assert len(task_inputs)==1200
    jsonl(DATA/'method_tasks.jsonl',task_inputs);jsonl(DATA/'evaluation_only/task_labels.jsonl',task_labels)
    paths=[Path(__file__),ROOT/'scripts/relation_nav_unified_20261125.py',ROOT/'conceptgraph/scenegraph/relation_nav_unified_20261125.py',ROOT/'conceptgraph/navigation/relation_nav_unified_20261125.py',
        ROOT/'conceptgraph/scenegraph/relations.py',ROOT/'conceptgraph/scenegraph/incremental_graph.py',ROOT/'conceptgraph/navigation/semantic.py',ROOT/'conceptgraph/completion/quality.py',ROOT/'conceptgraph/completion/evidence.py',ROOT/'conceptgraph/project_paths.py']
    initial=read(ROOT/'doc/交接/20261125/integration_manifest_v1.json')
    old_audit=read(ROOT/'outputs/reports/navcandidate_20261125/independent/baseline_and_git_sync_v1.json')
    protected=[dict(path=r['path'],sha256=sha(r['path'])) for r in old_audit['protected']]
    protected.extend(dict(path=str(ROOT/p),sha256=sha(ROOT/p)) for p in ['MEMORY.md','PAPER_MEMORY.md','conceptgraph/slam/utils.py','conceptgraph/slam/streamlined_mapping.py','conceptgraph/navigation/semantic.py','conceptgraph/scenegraph/relations.py','conceptgraph/scenegraph/incremental_graph.py'])
    input_paths.update([R1/'release.json',S2INPUT/'all_manifest.json',S2INPUT/'all_GT.jsonl',DATA/'method_frames.jsonl',DATA/'method_tasks.jsonl',RAW/'scene.json',RAW/'log.json'])
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        policy=POLICY,navigation_policy=NAV_POLICY,source_sha256={str(p):sha(p) for p in paths},input_sha256={str(p):sha(p) for p in input_paths},
        evaluation_only_sha256={str(p):sha(p) for p in [DATA/'evaluation_only/frames.jsonl',DATA/'evaluation_only/task_labels.jsonl',RAW/'attribute.json',RAW/'sample_annotation.json']},
        protection=protected,scope=dict(R1_scenes=38,R1_frames=1508,S2_scenes=6,S2_frames=120,S2_constructed_tasks=1200),
        no_test_opened=True,configuration_count=1,seed=17,GT_in_method=False,Qwen_calls=0,GPU_calls=0,planner_calls=0,
        contracts=checks,task_labels=dict(static='official current parked/stopped/standing/sitting attribute; others unknown or moving',
            unknown_motion='pending, never relabelled as no-match',multiple_reference='pending ambiguity, no arbitrary unique GT',nearest='unique ego-distance minimum among static eligible objects; only exact distance ties share credit'),
        current_facts='no old dynamic memory in current facts; upstream current snapshot owns node lifetime; empty/removed snapshots expire cached pairs',
        storage_budget_bytes=128*1024**2,shared_budget_bytes=256*1024**2,wall_budget_s=5400,minimum_free_bytes=20*1024**3,
        legacy_near8_unchanged=True,region_evaluation='engineering road annulus2..8m independent of relation near6; no independent region GT, parking/curb or path success',integration_manifest_sha256=sha(ROOT/'doc/交接/20261125/integration_manifest_v1.json'))
    write(DATA/'manifest.json',manifest);write(REPORT/'registration.json',dict(manifest=str(DATA/'manifest.json'),sha256=sha(DATA/'manifest.json'),elapsed_s=time.monotonic()-begun))
    guard(begun);print(json.dumps(dict(prepared=True,frames=len(frames),tasks=len(task_inputs),labels=dict(Counter(r['label_status'] for r in task_labels)),elapsed_s=time.monotonic()-begun)))


def relation_nodes(frame,registry,first_seen):
    result={}
    for obj in frame['predictions']:
        stable=obj['node_id'];registry.setdefault(stable,len(registry));first_seen.setdefault(stable,frame['frame_idx']);local=registry[stable]
        last=obj.get('last_seen_frame',frame['frame_idx'])
        result[local]=RelationNode(local,obj['center'],dynamic=obj.get('motion_state','unknown')=='dynamic',
            observation_count=obj['num_detections'],first_seen_frame=first_seen[stable],last_seen_frame=last,
            metadata=dict(timestamp_s=obj.get('timestamp_s'),motion_state=obj.get('motion_state','unknown'),
                shape_available=obj.get('shape_available',True),source_node_id=stable))
    return result


def verify(manifest):
    for group in ['source_sha256','input_sha256','evaluation_only_sha256']:
        for path,expected in manifest[group].items():
            if sha(path)!=expected:raise AssertionError('frozen_file_changed:'+path)


def relations():
    begun=time.monotonic();manifest=read(DATA/'manifest.json');verify(manifest)
    dest=experiment_directory('nuscenes','validation',RUN_ID,create=True)
    for path in manifest['source_sha256']:
        target=dest/'source_snapshot'/Path(path).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    counts=defaultdict(Counter);per_scene=defaultdict(Counter);state={}
    with gzip.open(dest/'relation_frames.jsonl.gz','wt') as out:
        for frame in rows(DATA/'method_frames.jsonl'):
            guard(begun);key=frame['cohort'],frame['scene']
            if key not in state:state[key]=(CurrentRelationCache(),IncrementalSceneGraph(POLICY),{}, {})
            cache,legacy,registry,first_seen=state[key];nodes=relation_nodes(frame,registry,first_seen)
            start=time.perf_counter();full=full_current(nodes,frame_index=frame['frame_idx'],heading=frame['heading'],timestamp_s=frame['timestamp_s']);full_s=time.perf_counter()-start
            inc=cache.update_snapshot(nodes,frame_index=frame['frame_idx'],heading=frame['heading'],timestamp_s=frame['timestamp_s'])
            old=legacy.update_snapshot(nodes,frame_index=frame['frame_idx'],ego_heading=frame['heading'])
            same=canonical_payload(full.edges)==canonical_payload(inc.edges)
            row=dict(cohort=key[0],scene=key[1],frame_idx=frame['frame_idx'],timestamp_s=frame['timestamp_s'],nodes=len(nodes),
                all_current_fields_equal=same,legacy_fact_equal=expanded_facts(old.edges)==expanded_facts(full.edges),
                full_edges=[e.to_dict() for e in full.edges],incremental_edges=[e.to_dict() for e in inc.edges],
                incremental_statistics=inc.update_statistics,full_s=full_s,incremental_s=inc.timing['total_s'],GT_in_method=False)
            out.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            increments=dict(frames=1,all_current_fields_equal=int(same),legacy_fact_equal=int(row['legacy_fact_equal']),
                geometry_recomputed_pairs=inc.update_statistics['geometry_recomputed_pairs'],geometry_cache_hits=inc.update_statistics['geometry_cache_hits'],
                expired_pair_cache=inc.update_statistics['expired_pair_cache'],axis_changed=int(inc.update_statistics['coordinate_axes_changed']))
            counts[key[0]].update(increments);per_scene[key].update(increments)
            if not same:raise AssertionError('full_incremental_current_payload_mismatch:'+str(key)+str(frame['frame_idx']))
    verify(manifest)
    write(dest/'relation_summary.json',dict(passed=True,cohorts={k:dict(v) for k,v in counts.items()},by_scene=[dict(cohort=k[0],scene=k[1],**v) for k,v in per_scene.items()],
        no_speedup_claim=True,geometry_cache='exact unchanged coordinates+frame axes only; no displacement tolerance',
        legacy_results_preserved=True,elapsed_s=time.monotonic()-begun,GT_in_method=False,GPU_calls=0))
    print(json.dumps({k:dict(v) for k,v in counts.items()}))


def navigation():
    parser=argparse.ArgumentParser();parser.add_argument('--node-run',required=True);args=parser.parse_args()
    begun=time.monotonic();manifest=read(DATA/'manifest.json');verify(manifest);dest=experiment_directory('nuscenes','validation',RUN_ID)
    nodepath=ROOT/'outputs/results/nuscenes/validation/iterations'/args.node_run/'S2_frames.jsonl.gz'
    unified={(r['scene'],r['frame_idx']):r for r in rows(nodepath)}
    frames={(r['scene'],r['frame_idx']):r for r in rows(DATA/'method_frames.jsonl') if r['cohort']=='S2'}
    assert set(unified)==set(frames) and len(frames)==120
    labels={r['task_id']:r for r in rows(DATA/'evaluation_only/task_labels.jsonl')};truth={(r['scene'],r['frame_idx']):r['truth'] for r in rows(DATA/'evaluation_only/frames.jsonl') if r['cohort']=='S2'}
    taskgroups=defaultdict(list)
    for query in rows(DATA/'method_tasks.jsonl'):taskgroups[query['scene'],query['frame_idx']].append(query)
    write(DATA/'node_input_amendment_v1.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),path=str(nodepath),sha256=sha(nodepath),
        task_policy_unchanged=True,addition='unified node schema from independent node/motion lane; same prediction geometry/IDs and same120 frames'))
    totals=Counter();by_log=defaultdict(Counter);per_family=defaultdict(Counter);maps={};registries={};output=[]
    for key,frame in frames.items():
        guard(begun);incoming=unified[key]
        assert incoming['predictions']==frame['predictions']
        records=incoming['nodes'];registry=registries.setdefault(key[0],{})
        nodes,eligible,indices=adapt_nodes(records,registry)
        relation_input={i:RelationNode(i,n.bbox_center,dynamic=n.motion_state=='dynamic',observation_count=n.metadata.get('num_detections',1),last_seen_frame=key[1],
            metadata=dict(timestamp_s=frame['timestamp_s'],source_node_id=n.metadata['frozen_object_id'],motion_state=n.motion_state)) for i,n in nodes.items()}
        edges=full_current(relation_input,frame_index=key[1],heading=frame['heading'],timestamp_s=frame['timestamp_s']).edges
        heading=np.asarray(frame['heading']);heading=heading/np.linalg.norm(heading);origin=np.asarray(frame['camera_pose'])[:2,3]
        matches=object_frame_metrics(frame['predictions'],truth[key],threshold_m=3.)['matches']
        binding={m['prediction_index']:truth[key][m['truth_index']]['instance_token'] for m in matches}
        for query in taskgroups[key]:
            result=select_task(query['query_text'],nodes,eligible,edges,heading=heading,origin=origin,reference_frame=query['reference_frame'])
            selected=result['selected_id'];region=None
            if selected is not None:
                if frame['location'] not in maps:maps[frame['location']]=load_reference_prior(frame['location'])
                region=engineering_approach_region(nodes[selected],maps[frame['location']],origin=origin,heading=heading)
            # Official identity and label first enter after inference/subgoal return.
            label=labels[query['task_id']];identity=binding.get(indices.get(selected));scorable=label['label_status']=='scorable';positive=label['positive'];accepted=set(label['acceptable_instances'])
            correct=identity in accepted if scorable and positive else None
            count=dict(tasks=1,scorable=int(scorable),positive=int(scorable and positive),negative=int(scorable and not positive),pending=int(not scorable),
                positive_selected_correct=int(bool(correct)),negative_false_selection=int(scorable and not positive and selected is not None),
                rejected=int(selected is None),engineering_subgoal_emitted=int(bool(region and region['status']=='ENGINEERING_ROAD_SUBGOAL')),
                pending_selected=int(not scorable and selected is not None),vehicle_route_success=0,parking_success=0)
            totals.update(count);by_log[frame['log']].update(count);per_family[label['query_family']].update(count)
            output.append(dict(**query,log=frame['log'],inference=result,engineering_region=region,
                evaluation_only=dict(label_status=label['label_status'],positive=positive,acceptable_instances=label['acceptable_instances'],selected_identity=identity,selected_correct=correct),
                GT_in_method=False,Qwen_calls=0,planner_calls=0))
    jsonl(dest/'navigation_tasks.jsonl',output)
    write(dest/'navigation_summary.json',dict(counts=dict(totals),by_log={k:dict(v) for k,v in by_log.items()},by_family={k:dict(v) for k,v in per_family.items()},
        label_status=dict(Counter(r['label_status'] for r in labels.values())),frames=120,empty_graph_frames=sum(not r['predictions'] for r in frames.values()),
        task_source='1200 fixed constructed static-object tasks, not16 reviewed sentences or natural language generalization',
        region_correctness_evaluated=False,region_truth='none; engineering geometry only',configuration_count=1,parameter_search=False,GT_in_method=False,
        Qwen_calls=0,GPU_calls=0,planner_calls=0,elapsed_s=time.monotonic()-begun,node_input_sha256=sha(nodepath)))
    print(json.dumps(dict(totals)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','relations','navigation']);args,extra=parser.parse_known_args()
    if args.action=='navigation':
        import sys
        sys.argv=[sys.argv[0],*extra]
    {'prepare':prepare,'relations':relations,'navigation':navigation}[args.action]()


if __name__=='__main__':main()
