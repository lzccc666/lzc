"""固定配置CPU旁路：统一R1/S2节点、真实运动证据和隔离官方运动评价。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,copy,gzip,hashlib,json,pickle,shutil,subprocess,time
from collections import Counter,defaultdict
from datetime import datetime,timezone
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from conceptgraph.project_paths import PROJECT_ROOT as ROOT, experiment_directory
from conceptgraph.completion.stage9 import predict
from conceptgraph.completion.vehicle_frontend import CLASSES
from conceptgraph.completion.association import gated_distance_match
from conceptgraph.nuscenes.expansion_benchmark import iter_records
from conceptgraph.slam.streamlined_mapping import MappingFrame
from conceptgraph.scenegraph.node_motion_unified_20261125 import (
    UnifiedR1Mapper,UnifiedS2Mapper,SCHEMA_VERSION,FRAME_SOURCE,unified_nodes,task_predictions,motion_evidence)
from scripts.multiview_20261123_run_graph import load_frame

DATA=ROOT/'dataset/processed/node_motion_unified_20261125_v1'
REPORT=ROOT/'outputs/reports/node_motion_unified_20261125'
RUN=experiment_directory('nuscenes','validation','node_motion_unified_20261125_v1')
OLD=Path('/root/autodl-tmp/lizichong')
FRONT3=['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']
DEADLINE=datetime.fromisoformat('2026-11-25T09:09:57+00:00')


def read(p):return json.loads(Path(p).read_text())
def lines(p):
    op=gzip.open if str(p).endswith('.gz') else open
    with op(p,'rt') as f:
        for l in f:yield json.loads(l)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def dump(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def clean(v):return json.loads(json.dumps(v,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()))
def guard():
    if datetime.now(timezone.utc)>=DEADLINE:raise TimeoutError('frozen_wall_deadline')
    if shutil.disk_usage(ROOT).free<20*1024**3:raise OSError('disk_below20GiB')
    size=sum(p.stat().st_size for d in [DATA,REPORT,RUN] if d.exists() for p in d.rglob('*') if p.is_file())
    if size>128*1024**2:raise OSError('node_line_storage_budget128MiB')
def verify(m):
    for group in ['source_sha256','input_sha256']:
        for p,h in m[group].items():assert sha(p)==h,(group,p)


def prepare():
    mfr=OLD/'dataset/processed/mf_position_20261125_v2'
    r1=read(mfr/'manifest.json')
    s2=read(OLD/'dataset/processed/mapnext_20261124_inputs_v1/all_manifest.json')
    cfg2=read(OLD/'outputs/results/nuscenes/validation/iterations/mvrepair_20261123_T3_pilot_v2/mvrepair_20261123_release.json')['config']
    inputs={str(p):sha(p) for p in [mfr/'manifest.json',mfr/'method_frames.jsonl',DATA/'schema_contract.json',OLD/'dataset/processed/mapnext_20261124_inputs_v1/all_manifest.json']}
    for p,h in r1['input_sha256'].items():inputs[p]=h
    for scene in s2['scenes']:
        for group in scene['groups']:
            for cam in FRONT3:
                for key in ['rgb','depth','detections','projection_evidence']:
                    p=group['views'][cam][key];inputs[p]=sha(p)
    sources=set()
    for p in r1['source_sha256']:
        rel=Path(p).relative_to(OLD);target=ROOT/rel
        if target.exists():sources.add(target)
    t3=read(OLD/'outputs/results/nuscenes/validation/iterations/mvrepair_20261123_T3_pilot_v2/mvrepair_20261123_release.json')
    for p in t3['source_sha256']:
        rel=Path(p).relative_to(OLD);target=ROOT/rel
        if target.is_file():sources.add(target)
    sources.update([Path(__file__).resolve(),ROOT/'conceptgraph/scenegraph/node_motion_unified_20261125.py',
        ROOT/'conceptgraph/slam/mapnext_20261124_sparse_separation_v1.py',ROOT/'conceptgraph/slam/mapnext_20261124_sparse_separation_v2.py',ROOT/'conceptgraph/slam/mapnext_20261124_mapper.py'])
    expected_r1=OLD/'outputs/results/nuscenes/validation/iterations/mf_position_20261125_v2/frames.jsonl.gz';inputs[str(expected_r1)]=sha(expected_r1)
    for scene in s2['scenes']:
        p=OLD/'outputs/results/nuscenes/validation/iterations/mapnext_20261124_S2_full_v1'/scene['scene']/'mapnext_20261124_predictions.jsonl.gz';inputs[str(p)]=sha(p)
    m=dict(schema_version=SCHEMA_VERSION,created_utc=datetime.now(timezone.utc).isoformat(),seed=17,configurations=1,parameter_search=False,
        git_base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        R1_config=r1['config'],R1_scenes=r1['scenes'],R1_method_frames=str(mfr/'method_frames.jsonl'),R1_expected=str(expected_r1),
        S2_config=cfg2,S2_manifest=s2,R1_frames=1508,S2_frames=120,
        R1_GT_objects=5723,S2_front3_GT_objects=687,scope='previously_used_validation_debugging_only',
        input_sha256=inputs,source_sha256={str(p):sha(p) for p in sorted(sources)},
        evaluation_only_sha256={str(OLD/'dataset/raw/nuscenes_trainval/v1.0-trainval'/n):sha(OLD/'dataset/raw/nuscenes_trainval/v1.0-trainval'/n) for n in ['attribute.json','sample_annotation.json','sample.json']},
        evaluation_only_inputs=[str(OLD/'dataset/processed/mapnext_20261124_inputs_v1/all_GT.jsonl')],
        motion_policy=read(DATA/'schema_contract.json')['motion_policy'],GT_in_method=False,GPU_calls=0,
        budget=dict(deadline_utc=DEADLINE.isoformat(),new_bytes=128*1024**2),
        scope_note='Full original geometrical predictions preserved, new evidence states are a separate conservative task view; no default change')
    dump(DATA/'manifest.json',m)
    experiment_directory('nuscenes','validation','node_motion_unified_20261125_v1',create=True)
    shutil.copy2(DATA/'manifest.json',RUN/'manifest.json')
    for p in sources:
        q=RUN/'source_snapshot'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    dump(REPORT/'frozen_inputs_and_sources.json',dict(manifest_sha256=sha(DATA/'manifest.json'),source_count=len(sources),input_count=len(inputs),git_base=m['git_base']))
    print(json.dumps(dict(prepared=True,run=str(RUN),sources=len(sources),inputs=len(inputs))),flush=True)


def frame_record(scene,log,index,now,pose,predictions,mapper):
    states=mapper.evidence_states();nodes=unified_nodes(predictions,states,mapper.resolve_evidence_id,now)
    assert len(nodes)==len(predictions)
    for i,n in enumerate(nodes):
        assert n['original_index']==i and n['node_id']==predictions[i]['node_id']
        assert n['center_world_m']==predictions[i]['center']
        assert n['motion_state'] in ['static','dynamic','unknown']
        assert n['body_heading_world_rad'] is None and not n['vehicle_safety_authorized']
        if n['motion_state']!='unknown':assert n['measurement']['source_closed']
    return dict(schema_version=SCHEMA_VERSION,scene=scene,log=log,frame_idx=index,render_timestamp_s=now,
        ego_pose_world=np.asarray(pose).tolist(),ego_heading_world_xy=np.asarray(pose)[:2,2].tolist(),frame_source=FRAME_SOURCE,
        predictions=predictions,nodes=nodes,task_predictions=task_predictions(predictions,nodes),
        evidence_states=states,raw_observations=list(mapper._nm_raw.values()),lineage_events=mapper._nm_events,
        original_prediction_exact=True,GT_in_method=False,vehicle_safety_authorized=False)


def export(scope):
    m=read(DATA/'manifest.json');verify(m);start=time.monotonic();counter=Counter();by_scene={};samples=[]
    path=RUN/(scope+'_frames.jsonl.gz')
    assert not path.exists()
    with gzip.open(path,'wt',compresslevel=6) as out:
        if scope=='R1':
            tasks=list(lines(m['R1_method_frames']));expected={(r['scene'],r['frame_idx']):r for r in lines(m['R1_expected'])}
            scenes=m['R1_scenes']
        else:scenes=m['S2_manifest']['scenes']
        for scene in scenes:
            name=scene['scene'];mapper=(UnifiedR1Mapper if scope=='R1' else UnifiedS2Mapper)(OmegaConf.create(m[scope+'_config']),CLASSES)
            if scope=='R1':items=[r for r in tasks if r['scene']==name]
            else:
                items=scene['groups'];ep=OLD/'outputs/results/nuscenes/validation/iterations/mapnext_20261124_S2_full_v1'/name/'mapnext_20261124_predictions.jsonl.gz'
                expected={r['frame_idx']:r for r in lines(ep)}
            cc=Counter()
            for item in items:
                guard();index=item['frame_idx'];mapper.begin_evidence(name,index);np.random.seed(17+index)
                if scope=='R1':
                    with gzip.open(item['detections'],'rb') as f:gobs=pickle.load(f)
                    pose=np.asarray(item['camera_pose']);now=item['timestamp_s']
                    frame=MappingFrame(index,np.asarray(Image.open(item['rgb']).convert('RGB')),np.asarray(Image.open(item['depth']),float)/1000.,np.asarray(item['intrinsic']),pose,copy.deepcopy(gobs),item['rgb'],timestamp_s=now)
                    mapper.update(frame);predictions=mapper.position_predictions(predict(mapper.confirmed_objects(),CLASSES),frame)
                    original=expected[name,index]['predictions']
                else:
                    frames=[(c,load_frame(item,c)) for c in FRONT3]
                    projection={}
                    for c in FRONT3:
                        with np.load(item['views'][c]['projection_evidence']) as f:projection[c]={k:f[k].copy() for k in f.files}
                        projection[c]['lidar_token']=item['views'][c].get('lidar_token')
                    captures={c:item['views'][c]['camera_timestamp_s'] for c in FRONT3}
                    predictions=mapper.update_group(frames,captures,projection)
                    pose=np.asarray(item['views']['CAM_FRONT']['camera_pose']);now=item['reference_timestamp_s'];original=expected[index]['predictions']['S2_three']
                predictions=clean(predictions)
                assert predictions==original,('original_prediction_changed',scope,name,index)
                rec=frame_record(name,scene['log_token'],index,now,pose,predictions,mapper)
                out.write(json.dumps(clean(rec),ensure_ascii=False,allow_nan=False)+'\n')
                counter['frames']+=1;counter['predictions']+=len(predictions);counter['raw_observations']+=len(rec['raw_observations'])
                cc['frames']+=1
                for n in rec['nodes']:
                    counter['motion:'+n['motion_state']]+=1;counter['shape:'+str(n['shape_available'])]+=1
                    counter['legacy_motion:'+n['legacy_motion_state']]+=1
                    counter['semantic_position_candidates']+=int(n['semantic_position_candidate'])
                    for reason in n['measurement']['unknown_reasons']:counter['unknown:'+reason]+=1
                if rec['nodes'] and len(samples)<3:samples.append(rec)
            by_scene[name]=dict(cc);out.flush();print(json.dumps(dict(scope=scope,scene=name,frames=counter['frames'],elapsed_s=time.monotonic()-start)),flush=True)
    assert counter['frames']==m[scope+'_frames']
    verify(m)
    dump(RUN/(scope+'_export_summary.json'),dict(passed=True,scope=scope,counts=dict(counter),by_scene=by_scene,all_original_predictions_exact=True,elapsed_s=time.monotonic()-start,frames_sha256=sha(path),GT_in_method=False,GPU_calls=0))
    dump(RUN/(scope+'_schema_samples.json'),samples)
    print(json.dumps(dict(scope=scope,complete=True,counts=dict(counter))),flush=True)


def _attribute_motion(names):
    dynamic={'vehicle.moving','pedestrian.moving'}
    static={'vehicle.stopped','vehicle.parked','pedestrian.standing','pedestrian.sitting_lying_down'}
    labels=({'dynamic'} if set(names)&dynamic else set())|({'static'} if set(names)&static else set())
    return next(iter(labels)) if len(labels)==1 else 'unknown'


def evaluate():
    """Only this command opens official labels, after both exported scopes exist."""
    start=time.monotonic();m=read(DATA/'manifest.json');verify(m)
    for p,h in m['evaluation_only_sha256'].items():assert sha(p)==h,('evaluation_label_changed',p)
    assert read(RUN/'R1_export_summary.json')['passed'] and read(RUN/'S2_export_summary.json')['passed']
    gt={};selected=set()
    for scene in m['R1_scenes']:
        p=OLD/'outputs/project_v2/thesis_completion_v1/mapping_evaluation/causal_mapping_full_001'/scene['scene']/'frames.jsonl'
        for r in lines(p):
            objs=r['evaluation_only_official'];gt['R1',r['scene'],r['frame_idx']]=objs
            selected.update(o['annotation_token'] for o in objs)
    for r in lines(OLD/'dataset/processed/mapnext_20261124_inputs_v1/all_GT.jsonl'):
        objs=[o for o in r['objects'] if set(o['visible_cameras'])&set(FRONT3)]
        gt['S2',r['scene'],r['frame_idx']]=objs;selected.update(o['annotation_token'] for o in objs)
    assert sum(len(v) for k,v in gt.items() if k[0]=='R1')==5723
    assert sum(len(v) for k,v in gt.items() if k[0]=='S2')==687
    meta=OLD/'dataset/raw/nuscenes_trainval/v1.0-trainval';attributes={r['token']:r['name'] for r in iter_records(meta/'attribute.json')}
    anns={r['token']:r for r in iter_records(meta/'sample_annotation.json') if r['token'] in selected}
    prev={r['prev'] for r in anns.values() if r['prev']};previous={r['token']:r for r in iter_records(meta/'sample_annotation.json') if r['token'] in prev}
    sampletokens={r['sample_token'] for r in [*anns.values(),*previous.values()]}
    timestamps={r['token']:r['timestamp']/1e6 for r in iter_records(meta/'sample.json') if r['token'] in sampletokens}
    freeze={str(meta/n):sha(meta/n) for n in ['attribute.json','sample_annotation.json','sample.json']}
    dump(DATA/'evaluation_only_sources.json',dict(input_sha256=freeze,method_outputs_frozen={s:sha(RUN/(s+'_frames.jsonl.gz')) for s in ['R1','S2']},labels_role='evaluation_only_not_method',attribute_semantics='stopped vs parked intent retained; cycle rider status not motion',displacement_semantics='past sample annotation interval ending at current official sample; not exact instantaneous/camera-time speed'))
    summaries={};bylog={};allrows=[]
    for scope in ['R1','S2']:
        counts=Counter();confusion=defaultdict(Counter);logc=defaultdict(Counter);last_id={};speederrors=[];stableerrors=[];changederrors=[]
        for frame in lines(RUN/(scope+'_frames.jsonl.gz')):
            truth=gt[scope,frame['scene'],frame['frame_idx']];pred=frame['predictions'];nodes=frame['nodes'];log=frame['log']
            matches=gated_distance_match(pred,truth,3.);bygt={x['truth_index']:x for x in matches};matchedpred={x['prediction_index'] for x in matches}
            counts['frames']+=1;counts['GT']+=len(truth);counts['predictions']+=len(pred);counts['matched_3m']+=len(matches)
            counts['unmatched_predictions']+=len(pred)-len(matches);logc[log]['GT']+=len(truth)
            for i,n in enumerate(nodes):
                if i not in matchedpred:counts['unmatched_prediction_motion:'+n['motion_state']]+=1
            for j,obj in enumerate(truth):
                ann=anns[obj['annotation_token']];names=[attributes[t] for t in ann['attribute_tokens'] if t in attributes];attr=_attribute_motion(names)
                prior=previous.get(ann['prev']);speed=None;dt=None
                if prior and prior['instance_token']==ann['instance_token']:
                    dt=timestamps[ann['sample_token']]-timestamps[prior['sample_token']]
                    if dt>0:speed=float(np.linalg.norm((np.asarray(ann['translation'])-prior['translation'])[:2])/dt)
                displacement='unknown' if speed is None else 'static' if speed<=.5 else 'dynamic'
                counts['attribute_label:'+attr]+=1;counts['displacement_label:'+displacement]+=1
                match=bygt.get(j);node=nodes[match['prediction_index']] if match else None
                state=node['motion_state'] if node else 'missing_node';legacy=node['legacy_motion_state'] if node else 'missing_node'
                status='no_assigned_identity_history';error=None
                if node:
                    key=(frame['scene'],node['node_id']);old=last_id.get(key)
                    status='same_assigned_identity' if old and old[0]==obj['instance_token'] else 'changed_assigned_identity' if old else 'no_assigned_identity_history'
                    last_id[key]=(obj['instance_token'],frame['render_timestamp_s']);counts['matched_history:'+status]+=1
                    if speed is not None and node['motion_speed_xy_mps'] is not None:
                        error=abs(node['motion_speed_xy_mps']-speed);speederrors.append(error)
                        (stableerrors if status=='same_assigned_identity' else changederrors if status=='changed_assigned_identity' else []).append(error)
                for kind,label in [('attribute',attr),('past_displacement',displacement)]:
                    confusion[kind+':'+label][state]+=1
                    confusion['legacy_'+kind+':'+label][legacy]+=1
                    if label!='unknown':
                        counts[kind+':all_labeled']+=1;counts[kind+':known_prediction']+=int(state in ['static','dynamic'])
                        counts[kind+':correct_over_all_labeled']+=int(state==label);counts[kind+':unknown_or_missing']+=int(state not in ['static','dynamic'])
                        logc[log][kind+':all_labeled']+=1;logc[log][kind+':correct_over_all_labeled']+=int(state==label)
                if attr!='unknown' and displacement!='unknown':counts['attribute_displacement_conflict']+=int(attr!=displacement)
                allrows.append(dict(scope=scope,scene=frame['scene'],log=log,frame_idx=frame['frame_idx'],instance_token=obj['instance_token'],annotation_token=ann['token'],
                    official_attributes=names,attribute_motion=attr,past_displacement_motion=displacement,past_displacement_speed_mps=speed,past_interval_s=dt,
                    current_official_sample_timestamp_s=timestamps[ann['sample_token']],render_timestamp_s=frame['render_timestamp_s'],
                    matched_node_id=node['node_id'] if node else None,predicted_motion=state,legacy_predicted_motion=legacy,
                    prediction_speed_mps=node['motion_speed_xy_mps'] if node else None,speed_abs_error_mps=error,assigned_identity_history=status,
                    GT_in_method=False))
        def avg(xs):return sum(xs)/len(xs) if xs else None
        summaries[scope]=dict(counts=dict(counts),confusion={k:dict(v) for k,v in confusion.items()},speed=dict(matched_known_N=len(speederrors),MAE_mps=avg(speederrors),same_assigned_identity_N=len(stableerrors),same_assigned_identity_MAE_mps=avg(stableerrors),changed_assigned_identity_N=len(changederrors),changed_assigned_identity_MAE_mps=avg(changederrors)))
        bylog[scope]={k:dict(v) for k,v in logc.items()}
    with gzip.open(RUN/'motion_evaluation_rows.jsonl.gz','wt',compresslevel=6) as f:
        for r in allrows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    verify(m)
    dump(RUN/'motion_evaluation_summary.json',dict(passed=True,scopes=summaries,bylog=bylog,rows=len(allrows),evaluation_only_source_sha256=freeze,GT_in_method=False,elapsed_s=time.monotonic()-start,
        limits=['Official attributes and past interval displacement are different references, not exact instantaneous ground truth','Unknown and missing-node cases remain in all labeled denominators','Same3m matching may confuse nearby objects; identity changes are automatic diagnosis, not human audited switches','Body heading is unknown; no GT orientation enters method','No frontend/model/GPU/planning call']))
    print(json.dumps(dict(evaluation_complete=True,scopes={k:v['counts'] for k,v in summaries.items()})),flush=True)


def contracts():
    base=[dict(scene='synthetic',frame_idx=i,timestamp_s=i*.5,render_timestamp_s=i*.5,raw_center_world_m=[i*.05,0,0],current_points=1,source_closed=True,depth_evidence=dict(source='current_projected_lidar',units='m',retained_pixels=1,support_fraction=.5)) for i in range(3)]
    checks=[]
    def test(name,trace,var,now,wanted):
        r=motion_evidence(trace,var,now);assert r['state']==wanted,(name,r);checks.append(dict(name=name,result=r))
    test('three_past_independent_single_point_static',base,[1,1,1],1,'static')
    moving=copy.deepcopy(base)
    for i,r in enumerate(moving):r['raw_center_world_m']=[i,0,0]
    test('real_raw_motion_dynamic',moving,[1,1,1],1,'dynamic')
    test('two_measurements_unknown',base[:2],[1,1,1],.5,'unknown')
    test('stale_current_unknown',base,[1,1,1],2,'unknown')
    ambiguous=copy.deepcopy(base);ambiguous[-1]['source_closed']=False
    test('identity_competition_unknown',ambiguous,[1,1,1],1,'unknown')
    future=copy.deepcopy(base);future[-1]['timestamp_s']=2
    test('future_raw_unknown',future,[1,1,1],1,'unknown')
    duplicate=copy.deepcopy(base);duplicate[-1]['frame_idx']=1
    test('same_group_not_multiple_independent',duplicate,[1,1,1],1,'unknown')
    test('uncertainty_guard',base,[4.1,1,1],1,'unknown')
    dump(REPORT/'motion_contracts.json',dict(passed=True,checks=checks,synthetic_contract_only=True,GPU_calls=0))
    print(json.dumps(dict(contracts_passed=len(checks))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','contracts','R1','S2','evaluate']);args=p.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='contracts':contracts()
    elif args.command=='evaluate':evaluate()
    else:export(args.command)
