"""分开核查真实模型提议、后台失效、当前授权、物理碰撞；只读三个已关闭回合。"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, Polygon

from conceptgraph.completion.stage9 import digest, dump
from conceptgraph.navigation.vehicle_geometry import VehicleSpec, VehicleState, footprint_polygon
from real_graph_contract_003 import validate

R=Path(__file__).resolve().parents[1]


def bind(nodes,targets):
    if not nodes:
        return {}
    cost=np.array([[np.linalg.norm(np.asarray(n['center'][:2])-t['center_xy'])
        if n['category']==t['category'] else 1e6 for t in targets] for n in nodes])
    ri,ci=linear_sum_assignment(cost)
    return {nodes[i]['node_id']:targets[j]['instance_id'] for i,j in zip(ri,ci) if cost[i,j]<=2.5}


def main():
    out=R/'carla_official/complex_semantic_diagnosis_008'
    out.mkdir(exist_ok=False)
    allcalls=[]
    summaries=[]
    inputs={}
    for mode in ['rule','base','adapter']:
        e=R/'carla_official/complex_semantic_navigation_007/complex_second_nearest20'/mode
        truth=json.loads((e/'evaluation_only/truth.json').read_text())
        target=truth['target_actor_id']
        targets=[dict(instance_id=target,category=truth['category'],center_xy=truth['center_xy'],
            polygon_xy=truth['target_polygon_xy'])]+truth['other_controlled_actors']
        perceptions=list(map(json.loads,(e/'perception.jsonl').open()))
        observations={p['frame_idx']:p for p in perceptions}
        execution=list(map(json.loads,(e/'execution.jsonl').open()))
        queue=json.loads((e/'semantic_queue_final.json').read_text())
        report=json.loads((e/'independent_choice_audit_008/report.json').read_text())
        log=e/'semantic_backend/calls.jsonl'
        calls=list(map(json.loads,log.open())) if log.exists() else []
        scores=[]
        for call in calls:
            i=int(call['payload']['sample_id'].rsplit('_',1)[1])
            p=observations[i]
            assert call['payload']==p['semantic_audit']['input_row']
            v=validate('candidate_rerank',call['raw_output'],allowed_candidate_ids=call['payload']['candidate_ids'])
            assert v.valid==call['schema_valid']
            selected=v.normalized_output['selected_node_id'] if v.valid else None
            original=selected-100 if selected is not None else None
            matches=bind(p['nodes'],targets)
            record=dict(mode=mode,call=call['call'],frame_idx=i,sim_time_s=p['sim_time_s'],
                schema_valid=v.valid,selected_semantic_id=selected,
                raw_proposal_correct=bool(original is not None and matches.get(original)==target),
                observed_target_covered=target in matches.values(),GPU_latency_s=call['GPU_latency_s'])
            allcalls.append(record)
            scores.append(record)
        discarded=[x for x in queue['events'] if x['kind']=='discarded']
        final=execution[-1]
        release=json.loads((e/'release.json').read_text())
        body=footprint_polygon(VehicleState(**final['state']),VehicleSpec(**release['vehicle_spec']))
        current=perceptions[-1]
        polygons=[Polygon(p) for p in current['known_obstacle_polygons']]
        lidar=np.fromfile(e/'frames'/f"{current['frame_idx']:06d}"/'lidar.bin',np.float32).reshape(-1,4)
        transform=np.diag([1.,-1.,1.,1.])@np.asarray(current['lidar_pose_carla'])
        world=(transform@np.column_stack([lidar[:,:3],np.ones(len(lidar))]).T).T
        elevated=world[(world[:,2]>.25)&(world[:,2]<4.)]
        optical=(np.linalg.inv(np.asarray(current['camera_pose_cv']))@elevated.T).T[:,:3]
        K=np.array([[320.,0,320],[0,320.,180],[0,0,1.]])
        valid=optical[:,2]>.1
        uv=optical@K.T
        uv[valid,:2]/=uv[valid,2,None]
        visible=valid&(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<360)&(optical[:,2]<65)
        geometry=[]
        for t in targets:
            polygon=Polygon(t['polygon_xy'])
            in_target=np.array([polygon.buffer(.15).covers(Point(xy[:2])) for xy in elevated])
            geometry.append(dict(instance_id=t['instance_id'],body_distance_m=float(body.distance(polygon)),
                full_LiDAR_elevated_points_near_object=int(in_target.sum()),
                front_RGB_FOV_LiDAR_points_near_object=int((in_target&visible).sum())))
        flips=[]
        a,b=np.asarray(targets[0]['center_xy']),np.asarray(targets[1]['center_xy'])
        for row in execution:
            xy=np.array([row['state']['x'],row['state']['y']])
            if np.linalg.norm(xy-a)<np.linalg.norm(xy-b):
                flips.append(row['state']['time_s'])
        item=dict(mode=mode,execution_ticks=len(execution),perception_frames=len(perceptions),
            actual_model_calls=len(calls),schema_valid_outputs=sum(x['schema_valid'] for x in scores),
            raw_model_proposal_correct_calls=sum(x['raw_proposal_correct'] for x in scores),
            backend_result_used_frames=report['backend_application_frames'],
            main_chain_proposal_correct_ever=report['semantic_proposal_correct_ever'],
            authorized_correct_ever=report['authorized_first_proposal_correct_ever'],
            correct_lock_ever=report['correct_stable_lock_ever'],wrong_lock_ever=report['wrong_stable_lock_ever'],
            strict_safe_arrival=report['strict_safe_arrival'],status=report['final_status'],
            actual_collisions=final['collisions'],
            discarded_binding_changed=sum(x.get('reason')=='binding_changed' for x in discarded),
            discarded_late_expired=sum(x.get('reason') in ['late_expired','expired_before_delivery'] for x in discarded),
            backend_delivered=sum(x['kind']=='delivered' for x in queue['events']),
            existing_target_approach_route_frames=sum(bool(p['path_world']) and p['path_purpose']=='target_approach' for p in perceptions),
            guard_braking_ticks=sum(bool(p['tracking'].get('emergency_braking')) for p in execution),
            final_body_minimum_observed_obstacle_distance_m=min((float(body.distance(p)) for p in polygons),default=None),
            final_controlled_object_geometry=geometry,official_second_nearest_order_flip_ticks=len(flips),
            first_order_flip_s=flips[0] if flips else None,
            sensor_diagnostic_scope='最后一次已保存LiDAR帧，可能比终止状态早一物理tick；只作可观测性诊断，不向智能体发送GT')
        summaries.append(item)
        files=[e/'evaluation_only/truth.json',e/'perception.jsonl',e/'execution.jsonl',
            e/'semantic_queue_final.json',e/'independent_choice_audit_008/report.json']
        if log.exists():files.append(log)
        inputs.update({str(p):digest(p) for p in files})
    with (out/'actual_model_calls.csv').open('x') as stream:
        w=csv.DictWriter(stream,fieldnames=list(allcalls[0]))
        w.writeheader()
        w.writerows(allcalls)
    dump(out/'report.json',dict(methods=summaries,actual_GPU_calls=sum(x['actual_model_calls'] for x in summaries),
        actual_closed_physical_episodes=3,source_sha256=inputs,evaluator_sha256=digest(__file__),
        original_failed_audits_retained=True,whole_project_sealed_test=False,goal_complete=False))
    print(json.dumps(summaries,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
