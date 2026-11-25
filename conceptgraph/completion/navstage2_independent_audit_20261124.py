"""CPU-only independent replay/accounting; never import or run the tested planner."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import difflib
import gzip
import hashlib
import json
import math
from pathlib import Path
import tarfile
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from conceptgraph.completion.navbench_stage1_independent_audit import ROOT, REPORT, digest, dump, verify_protection
from conceptgraph.scripts.build_lane_constrained_bev import _sample_arcline

DATA=ROOT/'dataset/processed/navstage2_20261124_v2'
PACKAGE=ROOT/'dataset/processed/navbench_stage1_20261124_v1'
RUN=ROOT/'outputs/results/nuscenes/test/iterations/navstage2_20261124_v2'
OLD=RUN.parent/'navstage2_20261124_v1'
BASE=ROOT/'outputs/project_v2/thesis_completion_v1'
SEALED=BASE/'public_experiment_completion_20261115/main_experiments/sealed_public_058'


def rows(path):return [json.loads(s) for s in Path(path).open() if s.strip()]
def read(path):return json.loads(Path(path).read_text())
def wrap(a):return (a+math.pi)%(2*math.pi)-math.pi
def clip(a,lo,hi):return max(lo,min(hi,a))
def scrub_times(value):
    if isinstance(value,dict):return {k:scrub_times(v) for k,v in value.items() if k!='duration_s'}
    if isinstance(value,list):return [scrub_times(v) for v in value]
    return value


def association(nodes,objects):
    if not nodes or not objects:return {}
    cost=np.full((len(nodes),len(objects)),1e9)
    for i,n in enumerate(nodes):
        for j,o in enumerate(objects):
            if n['category']==o['category']:cost[i,j]=math.dist(n['center'][:2],o['center'][:2])
    ii,jj=linear_sum_assignment(cost); result={}
    for i,j in zip(ii,jj):
        d=cost[i,j]
        if d>3.:continue
        alternatives=np.delete(cost[i],j);competitors=np.delete(cost[:,j],i)
        if len(alternatives) and min(alternatives)-d<.5:continue
        if len(competitors) and min(competitors)-d<.5:continue
        result[nodes[i]['node_id']]=objects[j]['instance_token']
    return result


def body(state,vehicle):
    x,y,yaw=state['x'],state['y'],state['yaw_rad'];c,s=math.cos(yaw),math.sin(yaw)
    return Polygon([(x+c*a-s*b,y+s*a+c*b) for a,b in [
        (-vehicle['rear_overhang_m'],-vehicle['width_m']/2),
        (vehicle['front_from_rear_axle_m'],-vehicle['width_m']/2),
        (vehicle['front_from_rear_axle_m'],vehicle['width_m']/2),
        (-vehicle['rear_overhang_m'],vehicle['width_m']/2)]])


def control_replay(saved,vehicle,task):
    names=['x','y','yaw_rad','speed_mps','steering_rad','time_s']
    pose=task['rear_axle_reference_world_pose'];expected=[pose[0][3],pose[1][3],math.atan2(pose[1][0],pose[0][0]),0.,0.,0.]
    errors=[];maximum=0.
    if max(abs(saved[0]['state'][n]-v) for n,v in zip(names,expected))>1e-8:errors.append('initial_state')
    for previous,current in zip(saved,saved[1:]):
        a,b=previous['state'],current['state'];u=current['control'];dt=vehicle['dt_s']
        speed=clip(a['speed_mps']+clip(u['acceleration_mps2'],-vehicle['max_braking_mps2'],vehicle['max_acceleration_mps2'])*dt,0.,vehicle['max_speed_mps'])
        command=clip(u['steering_command_rad'],-vehicle['max_steering_rad'],vehicle['max_steering_rad'])
        steering=a['steering_rad']+clip(command-a['steering_rad'],-vehicle['max_steering_rate_radps']*dt,vehicle['max_steering_rate_radps']*dt)
        length=(a['speed_mps']+speed)/2*dt;angle=length*math.tan((a['steering_rad']+steering)/2)/vehicle['wheelbase_m']
        chord=length*(math.sin(angle/2)/(angle/2) if abs(angle)>1e-10 else 1.)
        expected=[a['x']+chord*math.cos(a['yaw_rad']+angle/2),a['y']+chord*math.sin(a['yaw_rad']+angle/2),wrap(a['yaw_rad']+angle),speed,steering,a['time_s']+dt]
        if not all(math.isfinite(b[n]) for n in names):errors.append('nonfinite')
        residual=max(abs(b[n]-v) for n,v in zip(names,expected));maximum=max(maximum,residual)
        if residual>1e-6:errors.append('control_replay')
        if abs(u['acceleration_mps2'])>max(vehicle['max_acceleration_mps2'],vehicle['max_braking_mps2'])+1e-8:errors.append('command_acceleration_range')
        if abs(b['steering_rad']-a['steering_rad'])>vehicle['max_steering_rate_radps']*dt+1e-8:errors.append('steering_rate')
    return dict(passed=not errors,errors=sorted(set(errors)),maximum_state_residual=maximum,states=len(saved))


def geometry_replay(saved,vehicle,context):
    road=prep(shape(context['drivable_geojson']));lanes=prep(shape(context['lane_union_geojson']))
    obstacles=prep(unary_union([shape(o['polygon_geojson']) for o in context['objects']]))
    errors=[];previous=None
    for index,row in enumerate(saved):
        state=row['state'];foot=body(state,vehicle)
        if not road.covers(foot):errors.append('body_outside_road')
        if not lanes.covers(foot):errors.append('body_outside_lane_union')
        if obstacles.intersects(foot):errors.append('body_official_obstacle')
        if previous is not None:
            angle=abs(wrap(state['yaw_rad']-previous['yaw_rad']))
            chord=math.dist([state['x'],state['y']],[previous['x'],previous['y']])
            radius=chord/(2*math.sin(angle/2)) if angle>1e-10 else 0.
            corner=math.hypot(max(vehicle['front_from_rear_axle_m'],vehicle['rear_overhang_m']),vehicle['width_m']/2)
            bulge=(radius+corner)*(1-math.cos(angle/2))+1e-8
            sweep=unary_union([body(previous,vehicle),foot]).convex_hull.buffer(bulge/math.cos(math.pi/64),quad_segs=16)
            if not road.covers(sweep):errors.append('sweep_outside_road')
            if not lanes.covers(sweep):errors.append('sweep_outside_lane_union')
            if obstacles.intersects(sweep):errors.append('sweep_official_obstacle')
        previous=state
    return dict(passed=not errors,errors=sorted(set(errors)),official_obstacles=len(context['objects']))


def lane_records(location):
    data=read(ROOT/'dataset/nuscenes_map_expansion/expansion'/f'{location}.json')
    nodes={r['token']:(r['x'],r['y']) for r in data['node']};polygons={r['token']:r for r in data['polygon']};lanes={}
    for kind in ['lane','lane_connector']:
        for row in data[kind]:
            paths=data['arcline_path_3'].get(row['token'],[])
            if not paths:continue
            poly=polygons[row['polygon_token']]
            lanes[row['token']]=dict(polygon=Polygon([nodes[t] for t in poly['exterior_node_tokens']],
                [[nodes[t] for t in h['node_tokens']] for h in poly.get('holes',[])]),samples=np.concatenate([_sample_arcline(p,.25) for p in paths]))
    return lanes,data['connectivity']


def lane_heading(state,lane):
    samples=lane['samples'];k=int(np.argmin(np.sum((samples[:,:2]-[state['x'],state['y']])**2,axis=1)))
    return abs(wrap(state['yaw_rad']-float(samples[k,2])))


def directed_replay(saved,tokens,lanes,connectivity):
    if not tokens or any(t not in lanes for t in tokens):return False
    if any(b not in connectivity.get(a,{}).get('outgoing',[]) for a,b in zip(tokens,tokens[1:])):return False
    possible={0}
    for index,row in enumerate(saved):
        s=row['state'];point=Point(s['x'],s['y']);next_possible=set()
        for j in possible|{j+1 for j in possible if j+1<len(tokens)}:
            lane=lanes[tokens[j]]
            if not lane['polygon'].covers(point) or lane_heading(s,lane)>math.pi/3:continue
            if index==0:
                if j==0:next_possible.add(j)
                continue
            before=saved[index-1]['state'];segment=LineString([(before['x'],before['y']),(s['x'],s['y'])])
            if any(j in (i,i+1) and unary_union([lanes[tokens[i]]['polygon'],lane['polygon']]).covers(segment) for i in possible):next_possible.add(j)
        possible=next_possible
        if not possible:return False
    return len(tokens)-1 in possible


def terminals(saved,accepted,context,regions,tokens,lanes):
    hold=saved[-21:];hold_ok=len(hold)==21 and hold[-1]['state']['time_s']-hold[0]['state']['time_s']>=2.-1e-8
    evidence=[];broad=False
    for obj in context['objects']:
        if obj['instance_token'] not in accepted:continue
        polygon=shape(obj['polygon_geojson']);distances=[];headings=[]
        for row in hold:
            s=row['state'];point=Point(s['x'],s['y']);distances.append(polygon.distance(point))
            headings.append(min([lane_heading(s,lanes[t]) for t in tokens if lanes[t]['polygon'].covers(point)] or [math.inf]))
        okay=hold_ok and all(2.<=d<=8. for d in distances) and max(headings)<=math.pi/36 and all(r['state']['speed_mps']<=.02 and r['state']['time_s']<=90. for r in hold)
        broad|=okay;evidence.append(dict(instance_token=obj['instance_token'],distance_min_m=min(distances),distance_max_m=max(distances),maximum_heading_error_rad=max(headings),broad_terminal_passed=okay))
    region_checks=[]
    for target in regions:
        if target['instance_token'] not in accepted:continue
        for index,region in enumerate(target.get('regions',[])):
            poly=Polygon(region['polygon_xy']);okay=hold_ok and saved[-1]['state']['time_s']<=90.+1e-8 and all(poly.covers(Point(r['state']['x'],r['state']['y'])) and r['state']['speed_mps']<=.02 and abs(wrap(r['state']['yaw_rad']-region['heading_rad']))<=region['max_heading_error_rad'] for r in hold)
            region_checks.append(dict(instance_token=target['instance_token'],region_index=index,passed=okay))
    return dict(broad=bool(broad),region=any(r['passed'] for r in region_checks),hold_ok=hold_ok,targets=evidence,regions=region_checks)


def main():
    begun=time.monotonic();manifest=read(DATA/'manifest.json');old=read(OLD/'manifest.json');failures=[]
    checks=[]
    for group in ['source_sha256','input_sha256','evaluation_only_sha256']:
        for path,expected in manifest[group].items():checks.append(dict(group=group,path=path,passed=digest(path)==expected))
    snapshots=[];contents=[]
    for directory,m in [(OLD,old),(RUN,manifest)]:
        with tarfile.open(directory/'source_snapshot.tar.gz') as archive:
            files={p.name:archive.extractfile(p).read() for p in archive.getmembers() if p.isfile()}
        snapshots.append(dict(run=directory.name,files=len(files),passed=all(hashlib.sha256(files[str(Path(p).relative_to(ROOT))]).hexdigest()==sha for p,sha in m['source_sha256'].items())))
        contents.append(files)
    source_diffs={k:''.join(difflib.unified_diff(contents[0].get(k,b'').decode().splitlines(True),contents[1].get(k,b'').decode().splitlines(True))) for k in set(contents[0])|set(contents[1]) if contents[0].get(k)!=contents[1].get(k)}
    tasks=rows(DATA/'method_inputs.jsonl');predictions=rows(RUN/'method_predictions.jsonl');prior=rows(OLD/'method_predictions.jsonl')
    freeze=read(PACKAGE/'denominator_freeze.json');ids=freeze['conditional_historical_reference_ids']
    for name,actual in [('manifest',manifest['task_ids']),('inputs',[r['task_id'] for r in tasks]),('predictions',[r['task_id'] for r in predictions])]:
        if actual!=ids:failures.append(name+'_81_denominator_or_order')
    if len(ids)!=81 or len(set(ids))!=81:failures.append('frozen_ids')
    if (DATA/'method_inputs.jsonl').read_bytes()!=(DATA.parent/'navstage2_20261124_v1/method_inputs.jsonl').read_bytes():failures.append('v1_v2_input_changed')
    for key in ['policy','vehicle','task_ids','denominator','evaluation','evaluation_only_sha256']:
        if old[key]!=manifest[key]:failures.append('v1_v2_'+key)
    if not all(r['passed'] for r in checks+snapshots):failures.append('frozen_source_or_input_hash')
    expected_changes={'conceptgraph/navigation/navstage2_20261124.py','conceptgraph/navigation/navstage2_runner_20261124.py'}
    if set(source_diffs)!=expected_changes:failures.append('unexpected_source_change')
    prefix_checks=[dict(task_id=a['task_id'],passed=scrub_times(a)==scrub_times(b)) for a,b in zip(prior,predictions)]
    if not all(r['passed'] for r in prefix_checks):failures.append('v1_completed_prefix_difference')
    candidates={r['task_id']:r for r in rows(PACKAGE/'method_inputs/candidate_tasks.jsonl')};agents={(r['scene'],r['frame_idx']):r for r in rows(SEALED/'agent_frames.jsonl')}
    graph_cache={};input_checks=[]
    allowed={'task_id','instruction','scene','issue_frame_idx','map_location','rear_axle_reference_world_pose','rear_axle_yaw_rad','initial_speed_mps','initial_steering_rad','sample_token','camera_token','prediction_frame'}
    frame_keys={'scene','frame_idx','camera_timestamp_s','camera_pose','intrinsic','objects','edges','gt_in_inference'}
    for task in tasks:
        errors=[];candidate=candidates[task['task_id']];key=task['scene'],task['issue_frame_idx'];agent=agents[key]
        if set(task)!=allowed or set(task['prediction_frame'])!=frame_keys:errors.append('unexpected_input_field')
        if task['instruction']!=candidate['instruction']:errors.append('instruction_change')
        path=candidate['historical_prediction_path']
        if path not in graph_cache:
            with gzip.open(path,'rt') as stream:graph_cache[path]={r['frame_idx']:r for r in map(json.loads,stream)}
        frame=graph_cache[path][task['issue_frame_idx']]
        if task['prediction_frame']!={k:frame[k] for k in frame_keys}:errors.append('prediction_graph_not_original')
        for name in ['rear_axle_reference_world_pose','rear_axle_yaw_rad','sample_token','camera_token']:
            if task[name]!=agent[name]:errors.append('initial_binding_'+name)
        if frame['camera_timestamp_s']!=agent['camera_timestamp_s'] or frame['gt_in_inference'] is not False:errors.append('causal_frame')
        if any(o['last_seen_frame']>task['issue_frame_idx'] or o['timestamp_s']>frame['camera_timestamp_s'] for o in frame['objects']):errors.append('future_object')
        input_checks.append(dict(task_id=task['task_id'],passed=not errors,errors=errors,objects=len(frame['objects'])))
    if not all(r['passed'] for r in input_checks):failures.append('method_input_provenance_or_isolation')
    labels={r['task_id']:r for r in rows(PACKAGE/'evaluation_only/conditional_reference_labels.jsonl')};certificates={r['certificate_id']:r for r in rows(PACKAGE/'evaluation_only/certificates.jsonl')}
    official={(r['scene'],r['frame_idx']):r for r in rows(BASE/'annotation/reserved_official_annotation_001/evaluation_only_frames.jsonl')}
    all_regions=rows(BASE/'annotation/reference_regions_reserved_full_003/targets.jsonl');saved_eval={r['task_id']:r for r in rows(RUN/'evaluation_only/queries.jsonl')}
    task_lookup={r['task_id']:r for r in tasks};evaluations=[];totals=Counter();map_cache={}
    for prediction in predictions:
        task=task_lookup[prediction['task_id']];label=labels[prediction['task_id']];accepted=label['source_query']['accepted_instance_tokens'];key=task['scene'],task['issue_frame_idx']
        identities=association(task['prediction_frame']['objects'],official[key]['objects']);selected=identities.get(prediction['selected_node_id']);correct=selected in accepted
        entry=dict(task_id=prediction['task_id'],candidate_target_covered=any(identities.get(c['node_id']) in accepted for c in prediction['semantic_candidates']),selected_correct=correct,predicted_route=bool(prediction['predicted_rows']),official_body_safe=False,control_valid=False,directed_route_valid=False,broad_terminal_geometry=False,reference_region_membership=False,frozen_region_contract_success=False,broad_semantic_geometry_success=False)
        if prediction['predicted_rows']:
            cert=certificates[label['certificate_ids'][0]];artifact=cert['artifacts']['static_context']
            if digest(artifact['path'])!=artifact['sha256']:failures.append('context_hash')
            context=read(artifact['path']);saved=prediction['predicted_rows'];vehicle=manifest['vehicle']
            control=control_replay(saved,vehicle,task);geometry=geometry_replay(saved,vehicle,context)
            if task['map_location'] not in map_cache:map_cache[task['map_location']]=lane_records(task['map_location'])
            lanes,connectivity=map_cache[task['map_location']];directed=directed_replay(saved,prediction['selected_lane_tokens'],lanes,connectivity)
            terminals_result=terminals(saved,accepted,context,[r for r in all_regions if (r['scene'],r['frame_idx'])==key],prediction['selected_lane_tokens'],lanes)
            entry.update(official_body_safe=geometry['passed'],control_valid=control['passed'],directed_route_valid=directed,broad_terminal_geometry=terminals_result['broad'],reference_region_membership=terminals_result['region'],frozen_region_contract_success=bool(correct and control['passed'] and geometry['passed'] and directed and terminals_result['region']),broad_semantic_geometry_success=bool(correct and control['passed'] and geometry['passed'] and directed and terminals_result['broad']),control_replay=control,geometry_replay=geometry,terminal_evidence=terminals_result,selected_instance=selected)
        metrics=['candidate_target_covered','selected_correct','predicted_route','official_body_safe','control_valid','directed_route_valid','broad_terminal_geometry','reference_region_membership','frozen_region_contract_success','broad_semantic_geometry_success']
        for k in metrics:
            totals[k]+=int(entry[k])
            if entry[k]!=saved_eval[prediction['task_id']][k]:failures.append('evaluation_difference:'+prediction['task_id']+':'+k)
        evaluations.append(entry)
    protection=verify_protection()
    if not protection['passed']:failures.append('protected_history_changed')
    status=Counter(r['status'] for r in predictions);maximum_query=max(r['duration_s'] for r in predictions)
    if status['EMPTY_PREDICTED_GRAPH']!=23 or sum(not t['prediction_frame']['objects'] for t in tasks)!=23:failures.append('empty_graphs_not_retained')
    if maximum_query>manifest['policy']['query_planning_budget_s']+.05:failures.append('query_budget_exceeded')
    if sum(r['duration_s'] for r in predictions)>300.:failures.append('total_budget_exceeded')
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=not failures,failures=failures,denominator=81,status_counts=dict(status),metrics=dict(totals),
        input_and_gt_isolation=dict(passed=all(r['passed'] for r in input_checks),input_checks=input_checks,method_consumes_only_predicted_objects_known_pose_instruction_common_HD=True,evaluation_only_loaded_after_frozen_predictions=True,prepare_uses_reference_membership_only_to_freeze_independent_feasible_denominator=True),
        source_and_input_hash_checks=checks,snapshot_checks=snapshots,v1_v2_source_diff=source_diffs,v1_completed_prefix_checks=prefix_checks,
        budget=dict(maximum_query_s=maximum_query,total_method_s=sum(r['duration_s'] for r in predictions),per_subplanner_s=.5,subplanner_budget_registration='Explicit in both pre-run frozen source snapshots; not duplicated as separate manifest field. No post-run change.',planner_calls_by_auditor=0,model_calls=0,GPU_calls=0),
        independent_evaluation=evaluations,protection=protection,elapsed_s=time.monotonic()-begun,
        limitations=['Historical front-camera object-approach regression only; no new pull-over/parking/following scores.', 'Known rear-axle pose and from-rest vehicle assumptions; static frozen boxes, no dynamic traffic or real arrival claim.', 'Finite goal rectangles are a sufficient subset; membership is reported separately from broad 2–8m geometry.', 'Independent control/body/direction/terminal replay uses frozen official geometry only after prediction generation; common HD arcline sampler is reused.'])
    dump(REPORT/'stage2_independent_v1.json',result)
    print(json.dumps({k:result[k] for k in ['passed','failures','denominator','status_counts','metrics','budget','elapsed_s']},ensure_ascii=False))


if __name__=='__main__':main()
