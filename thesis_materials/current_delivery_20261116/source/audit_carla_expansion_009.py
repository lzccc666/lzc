"""从原始同步位姿、实际轮角、官方对象真值和当前预测重算；不信任ARRIVED。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, Polygon

from carla_lossless_jsonl_001 import actual_log, read_rows
from conceptgraph.completion.stage9 import digest, dump

R=Path(__file__).resolve().parents[1]


def match_identities(nodes, targets):
    if not nodes:
        return {}
    costs=np.asarray([[np.linalg.norm(np.asarray(n['center'][:2])-t['center_xy'])
        if n['category']==t['category'] else 1.e6 for t in targets] for n in nodes])
    ri,ci=linear_sum_assignment(costs)
    return {nodes[i]['node_id']:targets[j]['instance_id'] for i,j in zip(ri,ci) if costs[i,j]<=2.5}


def audit(episode):
    dest=episode/'independent_expansion_audit_009';dest.mkdir(exist_ok=False)
    release=json.loads((episode/'release.json').read_text())
    truth=json.loads((episode/'evaluation_only/truth.json').read_text())
    summary=json.loads((episode/'summary.json').read_text())
    states=read_rows(episode,'execution.jsonl');observations=read_rows(episode,'perception.jsonl')
    assert states and observations
    target=truth['target_actor_id'];T=truth['arrival_truth'];spec=release['vehicle_spec']
    targets=[dict(instance_id=target,center_xy=truth['center_xy'],category=truth['category'])]+truth['other_controlled_actors']
    road=Polygon(release['prior']['road_polygon_xy']);accepted=set(T['accepted_instance_ids'])
    mapping={p['frame_idx']:match_identities(p['nodes'],targets) for p in observations}
    cursor=0;held=None;maxheld=0.;ever_arrived=False;details=[];lane=[];errors=[];previous=None
    for row in states:
        while cursor+1<len(observations) and observations[cursor+1]['sensor_frame']<=row['frame']:
            cursor+=1
        obs=observations[cursor];identity=mapping[obs['frame_idx']].get(obs['selected_node_id'])
        matrix=np.asarray(row['ego_transform_carla']);yaw=-math.atan2(matrix[1,0],matrix[0,0])
        center=matrix[:2,3]*[1,-1];xy=center-release['rear_axle_offset_from_actor_center_m']*np.array([math.cos(yaw),math.sin(yaw)])
        state=row['state'];time_s=state['time_s'];speed=state['speed_mps'];errors.append(float(np.linalg.norm(xy-[state['x'],state['y']])))
        assert row['snapshot_frame']==row['frame'] and all(f==row['frame'] for f in row['sensor_frames'].values())
        assert abs(np.linalg.norm(row['actual_velocity_carla'][:2])-speed)<1.e-8
        assert abs(-math.atan(float(np.mean(np.tan(np.radians(row['actual_front_wheel_deg'])))))-state['steering_rad'])<1.e-8
        if previous is not None:
            assert all(abs(previous['next_control'][k]-row['applied_control'][k])<1.e-7 for k in ['throttle','steer','brake'])
        previous=row
        ex=(spec['front_from_rear_axle_m']+spec['rear_overhang_m'])/2;ey=spec['width_m']/2
        local=np.array([[-ex,-ey],[ex,-ey],[ex,ey],[-ex,ey]]);c,s=math.cos(yaw),math.sin(yaw)
        body=Polygon(local@np.array([[c,s],[-s,c]])+center);legal=road.buffer(.05).covers(body)
        if not legal:lane.append(row['tick'])
        inside=heading=False
        for region in T['regions']:
            if region['instance_id']==identity and Polygon(region['polygon_xy']).covers(Point(xy)):
                inside=True;heading=abs(math.atan2(math.sin(yaw-region['heading_rad']),math.cos(yaw-region['heading_rad'])))<=T['max_heading_error_rad']
        valid=identity in accepted and inside and heading and speed<=T['max_speed_mps'] and not row['collisions'] and legal and T['reachability']=='reachable'
        if valid and held is None:held=time_s
        if not valid:held=None
        duration=0. if held is None else time_s-held;maxheld=max(maxheld,duration)
        arrived=duration>=T['hold_s'] and time_s<=T['timeout_s'];ever_arrived|=arrived
        details.append(dict(tick=row['tick'],time_s=time_s,matched_instance_id=identity,correct_identity=identity in accepted,inside=inside,heading_ok=heading,stopped=speed<=T['max_speed_mps'],body_legal=legal,held_s=duration,geometric_arrival=arrived))
    assert max(errors)<1.e-6
    t=np.array([r['state']['time_s'] for r in states]);v=np.array([r['state']['speed_mps'] for r in states]);steer=np.array([r['state']['steering_rad'] for r in states])
    assert np.all(np.diff(t)>0)
    acc=np.diff(v)/np.diff(t);rate=np.abs(np.diff(steer)/np.diff(t))
    limits=dict(speed_violations=int((v>spec['max_speed_mps']+1.e-5).sum()),acceleration_violations=int((acc>spec['max_acceleration_mps2']+1.e-5).sum()),braking_violations=int((acc < -spec['max_braking_mps2']-1.e-5).sum()),steering_rate_violations=int((rate>spec['max_steering_rate_radps']+1.e-5).sum()),steering_limit_violations=int((np.abs(steer)>spec['max_steering_rad']+1.e-5).sum()))
    choices=[];submitted={}
    from real_graph_contract_003 import build_completion_prompt
    from carla_real_graph_bridge_003 import fingerprint
    for obs in observations:
        matches=mapping[obs['frame_idx']];a=obs['semantic_audit'];first=obs['candidates'][0]['node_id'] if obs['candidates'] else None
        assert a['authorized_node_id']==first
        correct=lambda node:node is not None and matches.get(node) in accepted
        locked=obs['lock']['state_after']=='LOCKED';sid=obs['selected_node_id']
        choices.append(dict(frame_idx=obs['frame_idx'],semantic_proposal_correct=correct(a['proposed_node_id']),authorized_first_correct=correct(first),candidate_coverage=any(correct(c['node_id']) for c in obs['candidates']),correct_lock=locked and correct(sid),wrong_lock=locked and sid is not None and not correct(sid),target_route_generated=bool(obs['path_world'] and obs['path_purpose']=='target_approach'),matched_identities=matches))
        switches=release['ablation_switches'];assert a['evidence_check_enabled']==switches['evidence_check_enabled']
        if not switches['relations_enabled']:assert obs['relations']==[] and a['input_row']['relation_metadata']==[]
        for event in a['events']:
            if event['kind']=='submitted':
                payload=a['input_row']
                assert hashlib.sha256(build_completion_prompt('candidate_rerank',payload).encode()).hexdigest()==event['prompt_sha256']
                queued=json.loads(json.dumps(payload,ensure_ascii=False,sort_keys=True,allow_nan=False))
                submitted[hashlib.sha256(build_completion_prompt('candidate_rerank',queued).encode()).hexdigest()]=queued
    assert summary['correct_object_selected']==any(c['authorized_first_correct'] for c in choices)
    assert summary['correctly_locked']==any(c['correct_lock'] for c in choices)
    assert summary['candidate_target_coverage']==any(c['candidate_coverage'] for c in choices)
    callsfile=episode/'semantic_backend/calls.jsonl'
    calls=list(map(json.loads,callsfile.open())) if callsfile.exists() else []
    final=json.loads((episode/'semantic_queue_final.json').read_text())
    assert len(calls)==(final['actual_model_calls'] or 0)
    assert final['worker'] is None or final['worker']['worker_stopped']
    assert sum(e['kind']=='backend_completed' for e in final['events'])==len(calls)
    for call in calls:
        assert call['prompt_sha256'] in submitted and call['payload']==submitted[call['prompt_sha256']]
        assert call['payload_sha256']==fingerprint(call['payload'])
        assert call['actual_CUDA'] and not call['fallback_used']
        assert call['input_tokens']<=2048 and call['output_tokens']<=256
        assert set(call['payload'])=={'task','sample_id','query_text','candidate_nodes','candidate_ids','relation_metadata'}
    manifest=json.loads((episode/'sensor_files_manifest.json').read_text());bulk=Path(manifest['bulk_root']).resolve()
    authorization=R/'json/sensor_bulk_authorization_003.json';policy=json.loads(authorization.read_text())
    assert digest(authorization)==manifest['authorization_sha256']==release['storage_authorization_sha256']
    assert bulk.is_relative_to(Path(policy['authorized_sensor_root']).resolve())
    for name,value in manifest['files'].items():
        p=(bulk/name).resolve();assert p.is_relative_to(bulk) and p.stat().st_size==value['bytes'] and digest(p)==value['sha256']
    strict=bool(ever_arrived and not any(limits.values()) and not lane and not any(r['collisions'] for r in states))
    report=dict(episode=release['episode']['episode'],seed=release['seed'],mode=release['semantic_mode'],condition=summary['condition'],task_kind=truth['task_kind'],physical_ticks=len(states),frontend_calls=len(observations),status=summary['status'],strict_safe_arrival=strict,geometric_identity_stop_arrival=bool(ever_arrived),maximum_contiguous_hold_s=maxheld,correct_first_proposal=any(c['authorized_first_correct'] for c in choices),correct_lock=any(c['correct_lock'] for c in choices),wrong_lock=any(c['wrong_lock'] for c in choices),target_route_generated=any(c['target_route_generated'] for c in choices),pose_reconstruction_max_error_m=max(errors),lane_violation_ticks=lane,actuation_limits=limits,minimum_acceleration_mps2=float(acc.min()) if len(acc) else None,maximum_acceleration_mps2=float(acc.max()) if len(acc) else None,independent_reachability=T['reachability'],no_match_scope_verified=truth.get('no_match_scope_certificate',{}).get('verified_no_bus_in_declared_scope'),sensor_files_rehashed=len(manifest['files']),actual_wheel_observations=True,GT_not_sent_to_agent=not any(p['gt_sent_to_agent'] for p in observations),ablation_switches_checked=True,inputs_sha256={str(p):digest(p) for p in [episode/'release.json',episode/'summary.json',actual_log(episode,'execution.jsonl'),actual_log(episode,'perception.jsonl'),episode/'evaluation_only/truth.json',episode/'sensor_files_manifest.json']},evaluator_sha256=digest(__file__),real_time_claim=False,whole_project_sealed_test=False,goal_complete=False)
    report.update(actual_final_GPU_model_calls=len(calls),schema_valid_model_outputs=sum(c['schema_valid'] for c in calls),actual_model_input_hashes_and_queue_checked=True,
        actual_model_application_frames=sum(p['semantic_audit']['background_result_used'] for p in observations),model_queue_final_sha256=digest(episode/'semantic_queue_final.json'))
    dump(dest/'report.json',report)
    for filename,rows in [('frames.jsonl',details),('choices.jsonl',choices)]:
        with (dest/filename).open('x') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(report,ensure_ascii=False));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--episode',required=True);args=parser.parse_args();audit(R/'carla_official'/args.episode)
