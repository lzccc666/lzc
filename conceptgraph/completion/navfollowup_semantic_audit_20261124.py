"""Independently inspect future-data isolation and replay recorded follow witnesses."""
from collections import Counter
from datetime import datetime,timezone
import difflib
import itertools
import json
import math

import numpy as np
from shapely.geometry import LineString,MultiPoint,Point,Polygon,shape
from shapely.ops import unary_union
from shapely.prepared import prep

from conceptgraph.completion.navfollowup_independent_20261124 import ROOT,REPORT,read,sha,write,protection
from conceptgraph.completion.reference_region_audit import independent_map
from conceptgraph.completion.navstage2_independent_audit_20261124 import body


def rows(path):return [json.loads(s) for s in path.open() if s.strip()]
def wrap(x):return (x+math.pi)%(2*math.pi)-math.pi
def yaw(record):
    w,x,y,z=record['rotation'];return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
def box_polygon(record):
    q=np.asarray(record['rotation'],float);q=q/np.linalg.norm(q);w=q[0];v=q[1:]
    cross=np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])
    rotation=np.eye(3)+2*w*cross+2*cross@cross
    corners=np.asarray(list(itertools.product([-.5,.5],repeat=3)))*np.asarray(record['size'])[[1,0,2]]
    return MultiPoint((corners@rotation.T+record['translation'])[:,:2]).convex_hull


def sweep(a,b,vehicle):
    angle=abs(wrap(b['yaw_rad']-a['yaw_rad']));corner=math.hypot(vehicle['front_from_rear_axle_m'],vehicle['width_m']/2)
    # The declared trajectory interpolates translation and yaw. Endpoint hull
    # plus rotation sagitta covers this interpolation, with outward GEOS correction.
    bulge=corner*(1-math.cos(angle/2))+1e-8
    return unary_union([body(a,vehicle),body(b,vehicle)]).convex_hull.buffer(bulge/math.cos(math.pi/64),quad_segs=16)


def replay(record,witness,road):
    contract=record['contract'];vehicle=contract['vehicle'];states=witness['reference_states'];anchors=witness['future_official_anchors'];target=witness['chosen_initial_official_instance']
    routes=[dict(tokens=r['tokens'],line=LineString(r['line_xy']),polygon=shape(r['polygon_geojson'])) for r in witness['allowed_successor_routes']]
    failures=[];checks=[];residual=0.;minimum_gap=math.inf;maximum_gap=-math.inf
    if not states or abs(states[0]['time_s'])>1e-8 or abs(states[-1]['time_s']-3.)>1e-8:failures.append('time_domain')
    saved_checks={round(r['time_s'],8):r for r in witness['independent_audit']['checks']}
    for index,s in enumerate(states):
        t=s['time_s'];anchor_pair=next(((a,b) for a,b in zip(anchors,anchors[1:]) if a['time_s']-1e-8<=t<=b['time_s']+1e-8),None)
        if anchor_pair is None:failures.append('missing_anchor_bracket');continue
        a,b=anchor_pair;alpha=(t-a['time_s'])/(b['time_s']-a['time_s']);p,q=a['official_ego_pose'],b['official_ego_pose']
        expected_xy=(1-alpha)*np.asarray(p['translation'][:2])+alpha*np.asarray(q['translation'][:2]);expected_yaw=yaw(p)+alpha*wrap(yaw(q)-yaw(p))
        residual=max(residual,math.dist(expected_xy,[s['x'],s['y']]),abs(wrap(s['yaw_rad']-expected_yaw)))
        if residual>1e-8:failures.append('recorded_ego_interpolation')
        left={o['instance_token']:o for o in a['objects']};right={o['instance_token']:o for o in b['objects']}
        if target not in left or target not in right:failures.append('missing_target');continue
        target_xy=(1-alpha)*np.asarray(left[target]['translation'][:2])+alpha*np.asarray(right[target]['translation'][:2])
        target_yaw=yaw(left[target])+alpha*wrap(yaw(right[target])-yaw(left[target]))
        choices=[]
        for route in routes:
            if route['polygon'].covers(Point(s['x'],s['y'])) and route['polygon'].covers(Point(target_xy)):
                line=route['line'];choices.append((line.distance(Point(s['x'],s['y']))+line.distance(Point(target_xy)),route['tokens'],line.project(Point(target_xy))-line.project(Point(s['x'],s['y']))))
        same=bool(choices)
        distance=min(choices,key=lambda r:r[:2])[2] if same else math.nan
        target_length=max(left[target]['size'][1],right[target]['size'][1]);gap=distance-vehicle['front_from_rear_axle_m']-target_length/2
        if same:minimum_gap=min(minimum_gap,gap);maximum_gap=max(maximum_gap,gap)
        gap_ok=same and 5.<=gap<=60.;heading_ok=abs(wrap(s['yaw_rad']-target_yaw))<=math.pi/6
        if index==0:
            envelope=body(s,vehicle);obstacles=unary_union([box_polygon(o) for o in a['objects']])
        else:
            before=states[index-1];mid=(before['time_s']+t)/2
            segment=next((z for z in zip(anchors,anchors[1:]) if z[0]['time_s']-1e-8<=mid<=z[1]['time_s']+1e-8))
            lo={o['instance_token']:o for o in segment[0]['objects']};ro={o['instance_token']:o for o in segment[1]['objects']}
            obstacles=unary_union([unary_union([box_polygon(v[token]) for v in [lo,ro] if token in v]).convex_hull for token in lo.keys()|ro.keys()])
            envelope=sweep(before,s,vehicle)
        road_ok=road.covers(envelope);obstacle_ok=not obstacles.intersects(envelope)
        for name,okay in [('road_sweep',road_ok),('obstacle_sweep',obstacle_ok),('same_directed_route',same),('heading',heading_ok),('gap',gap_ok)]:
            if not okay:failures.append(('initial_' if not index else '')+name)
        checks.append(dict(time_s=t,road_sweep=road_ok,obstacle_sweep=obstacle_ok,same_directed_route=same,heading_compatible=heading_ok,gap_accepted=gap_ok,longitudinal_body_gap_m=gap if same else None))
        original=saved_checks.get(round(t,8))
        if original:
            for name in ['road_sweep','obstacle_sweep','same_directed_route','heading_compatible','gap_accepted']:
                if bool(original[name])!=checks[-1][name]:failures.append('saved_check_difference:'+name)
    start,end=states[0],states[-1]
    base=LineString(witness['route_line_xy']);progress=base.project(Point(end['x'],end['y']))-base.project(Point(start['x'],start['y']))
    if progress<.5:failures.append('no_forward_follow_progress')
    # The initial target-selected route may end before later legal successors;
    # progress is diagnostic here, and the saved contract uses the same frozen definition.
    passed=not failures
    return dict(passed=passed,failures=sorted(set(failures)),states=len(states),maximum_interpolation_residual=residual,minimum_body_gap_m=minimum_gap,maximum_body_gap_m=maximum_gap,checks=checks,source_audit_passed=witness['independent_audit']['passed'])


def main():
    folder=ROOT/'dataset/processed/navsemantic_tasks_20261124_v3';old=folder.parent/'navsemantic_tasks_20261124_v2';manifest=read(folder/'manifest.json');prior_manifest=read(old/'manifest.json')
    errors=[];hashes=[]
    for name,expected in manifest['files'].items():hashes.append(dict(path=str(folder/name),passed=sha(folder/name)==expected))
    for group in ['source_sha256','input_sha256']:
        for path,expected in manifest[group].items():hashes.append(dict(path=path,passed=sha(path)==expected))
    candidates=rows(folder/'evaluation_only/task_candidates.jsonl');previous=rows(old/'evaluation_only/task_candidates.jsonl');inputs=rows(folder/'method_inputs/follow_tasks.jsonl')
    if [r['task_id'] for r in candidates]!=[r['task_id'] for r in previous] or len(candidates)!=12:errors.append('candidate_denominator_changed')
    if (folder/'method_inputs/follow_tasks.jsonl').read_bytes()!=(old/'method_inputs/follow_tasks.jsonl').read_bytes():errors.append('method_input_changed')
    if read(folder/'preregistration.json')['policy']!=read(old/'preregistration.json')['policy']:errors.append('policy_threshold_changed')
    for a,b in zip(candidates,previous):
        if a['contract']!=b['contract'] or a.get('target_instance_token')!=b.get('target_instance_token'):errors.append('changed_contract_or_initial_target')
    sensor=read(ROOT/'dataset/processed/mv136_20261123_sixscene_v1/manifest.json');groups={(s['scene'],g['frame_idx']):g for s in sensor['scenes'] for g in s['groups']}
    whitelist={'gt_in_inference','initial_rear_axle_world_pose','initial_yaw_rad','instruction','issue_frame_idx','issue_time_s','past_and_current_sensor_references','scene','task_id','task_scope'}
    input_checks=[]
    for record in inputs:
        failures=[]
        if set(record)!=whitelist or record['gt_in_inference'] is not False:failures.append('input_field_or_gt')
        if [r['frame_idx'] for r in record['past_and_current_sensor_references']]!=list(range(record['issue_frame_idx']+1)):failures.append('past_frame_domain')
        for frame in record['past_and_current_sensor_references']:
            original=groups[record['scene'],frame['frame_idx']]
            for channel,view in frame['views'].items():
                if channel not in ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT']:failures.append('camera_channel')
                if view!={k:original['views'][channel][k] for k in view}:failures.append('sensor_not_original')
                if view['camera_timestamp_s']>record['issue_time_s']:failures.append('future_sensor')
                if any(token in str(view).lower() for token in ['future_official','target_instance','evaluation_only','reference_states']):failures.append('oracle_sensor_pointer')
        input_checks.append(dict(task_id=record['task_id'],passed=not failures,failures=failures))
    if not all(r['passed'] for r in input_checks):errors.append('method_input_isolation')
    map_path=next(path for path in manifest['input_sha256'] if '/expansion/' in path);raw_map=read(map_path);road=independent_map(raw_map)['road'];replays=[]
    for record in candidates:
        if not record.get('witness'):continue
        witness=read(record['witness']['path'])
        if sha(record['witness']['path'])!=record['witness']['sha256']:errors.append('witness_hash')
        for route in witness['allowed_successor_routes']:
            if any(b not in raw_map['connectivity'].get(a,{}).get('outgoing',[]) for a,b in zip(route['tokens'],route['tokens'][1:])):errors.append('unofficial_successor_edge')
            if route['tokens'][:len(witness['route_tokens'])]!=witness['route_tokens']:errors.append('nonuniform_successor_rule')
        result=replay(record,witness,road);result['task_id']=record['task_id'];replays.append(result)
        if result['passed']!=witness['independent_audit']['passed']:errors.append('witness_outcome_difference:'+record['task_id'])
    if sum(r['passed'] for r in replays)!=3:errors.append('feasible_count')
    if any(r['feasibility_status']=='infeasible_proven' for r in candidates):errors.append('unknown_promoted_to_infeasible')
    if any(r['main_table_eligible'] for r in candidates):errors.append('legacy_main_changed')
    snapshot_diffs={}
    for a,b in [('v1','v2'),('v2','v3')]:
        prefix=ROOT/'outputs/results/nuscenes/validation/iterations';tail='source_snapshot/conceptgraph/completion/navsemantic_tasks_20261124.py'
        left=prefix/f'navsemantic_tasks_20261124_{a}'/tail;right=prefix/f'navsemantic_tasks_20261124_{b}'/tail
        snapshot_diffs[a+'_'+b]=''.join(difflib.unified_diff(left.read_text().splitlines(True),right.read_text().splitlines(True)))
    preserved=protection()
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=not errors and all(r['passed'] for r in hashes) and preserved['unchanged'],failures=errors,hash_checks=hashes,
        fixed_candidates=12,target_bound=4,no_current_matching_target=8,geometry_time_feasible=3,pending=9,method_evaluated=False,
        inference_checks=input_checks,witness_replays=replays,snapshot_diffs=snapshot_diffs,protection=preserved,
        scope='3-second timed XY/yaw geometry under declared linear anchor/box model; no controls, forced from-rest, stopping hold, real execution or unobserved traffic guarantee.',
        successor_revision='v2 negative reference outcome preserved; v3 generic legal successors changed reference checking before any method evaluation, not original v1 certification. No candidate/target/threshold or method input change.',
        limitations=['Three feasible candidates come from two scenes, with scene0013 frames0/10 sharing one physical episode; not three independent scenes.', '8 no-current-target cases are source no-match/not-applicable, not proven unreachable.', 'Default speed/steering fields in saved VehicleState are unused placeholders; only timed XY/yaw is certified.'])
    write(REPORT/'semantic_follow_package_v1.json',result)
    print(json.dumps({k:result[k] for k in ['passed','failures','fixed_candidates','target_bound','geometry_time_feasible','pending']},ensure_ascii=False))


if __name__=='__main__':main()
