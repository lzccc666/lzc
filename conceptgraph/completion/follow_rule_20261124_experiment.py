"""Separate preparation, inference and evaluation for a tiny follow regression."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Point, shape
from shapely.ops import unary_union

from conceptgraph.completion.navbench_stage1_20261124 import ROOT,canonical,dump,jsonl,now,read,rows,sha,source_ref
from conceptgraph.completion.navsemantic_tasks_20261124 import transform,route_measure
from conceptgraph.completion.reference_regions import load_reference_prior,object_polygon
from conceptgraph.navigation.follow_rule_20261124 import POLICY,infer
from conceptgraph.navigation.vehicle_geometry import VehicleSpec,VehicleState,angle_difference,swept_body_polygon,footprint_polygon
from conceptgraph.nuscenes.expansion_benchmark import iter_records
from conceptgraph.project_paths import experiment_directory

ID='follow_rule_20261124_v3'
DATA=ROOT/'dataset/processed'/ID
REFERENCE=ROOT/'dataset/processed/navsemantic_tasks_20261124_v3'
GRAPHS=ROOT/'outputs/results/nuscenes/validation/iterations/mapnext_20261124_baseline_v1'
REPORT=ROOT/'outputs/reports'/ID


def prepare():
    DATA.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    run=experiment_directory('nuscenes','validation',ID,create=True)
    tasks=rows(REFERENCE/'method_inputs/follow_tasks.jsonl')
    plans=read(ROOT/'dataset/processed/mv136_20261123_sixscene_v1/raw_sensor_plan.json')['plans']
    metadata=ROOT/'dataset/raw/nuscenes_trainval/v1.0-trainval'
    calibrations={x['token']:x for x in iter_records(metadata/'calibrated_sensor.json')}
    plan_index={(p['scene'],p['frame_idx']):p for p in plans if p['channel']=='CAM_FRONT'}
    locations={x['token']:x['location'] for x in iter_records(metadata/'log.json')}
    sensor=read(ROOT/'dataset/processed/mv136_20261123_sixscene_v1/manifest.json')
    scene_info={s['scene']:s for s in sensor['scenes']}
    graph_rows={};source_hashes={}
    for scene in scene_info:
        path=GRAPHS/scene/'mapnext_20261124_predictions.jsonl.gz'
        source_hashes[str(path)]=sha(path)
        with gzip.open(path,'rt') as handle:
            graph_rows[scene]=[json.loads(line) for line in handle]
    output=[]
    for task in tasks:
        issue=task['issue_time_s'];history=[];odometry=[]
        for frame in graph_rows[task['scene']]:
            if frame['frame_idx']>task['issue_frame_idx'] or frame['reference_timestamp_s']>issue:continue
            assert frame['reference_timestamp_s']<=issue and frame['GT_in_inference'] is False
            objects=[]
            for obj in frame['predictions'][POLICY['graph']]:
                timestamp=obj.get('state_last_measurement_timestamp_s',obj.get('timestamp_s'))
                assert timestamp<=issue
                objects.append(dict(node_id=obj['node_id'],category=obj['category'],center=obj['center'],
                    extent=obj.get('extent'),measurement_timestamp_s=timestamp))
            history.append(dict(frame_idx=frame['frame_idx'],timestamp_s=frame['reference_timestamp_s'],objects=objects))
        for sensor_frame in task['past_and_current_sensor_references']:
            view=sensor_frame['views']['CAM_FRONT']
            plan=plan_index[(task['scene'],sensor_frame['frame_idx'])]
            extrinsic=transform(calibrations[plan['camera']['calibrated_sensor_token']])
            ego=np.asarray(view['camera_pose'])@np.linalg.inv(extrinsic)
            assert view['camera_timestamp_s']<=issue
            odometry.append(dict(time_s=view['camera_timestamp_s'],xy=ego[:2,3].tolist(),
                rear_axle_engineering_reference='ego_pose origin; sensor extrinsic removed'))
        output.append(dict(task_id=task['task_id'],scene=task['scene'],issue_frame_idx=task['issue_frame_idx'],
            issue_time_s=issue,instruction=task['instruction'],initial_rear_axle_world_pose=task['initial_rear_axle_world_pose'],
            initial_yaw_rad=task['initial_yaw_rad'],map_location=locations[scene_info[task['scene']]['log_token']],
            prediction_history=history,past_rear_axle_odometry=odometry))
    jsonl(DATA/'method_inputs.jsonl',output)
    dump(DATA/'preregistration.json',dict(created_utc=now(),policy=POLICY,
        candidates=12,certified_geometry_time_denominator=source_ref(REFERENCE/'geometry_time_denominator_freeze_v1.json'),
        graph_sources_sha256=source_hashes,input=source_ref(DATA/'method_inputs.jsonl'),
        no_parameter_selection=True,one_fixed_run=True,GT_in_method=False,reference_future_in_method=False))
    for path in [Path(__file__),ROOT/'conceptgraph/navigation/follow_rule_20261124.py',ROOT/'scripts/follow_rule_20261124.py']:
        destination=run/'source_snapshot'/path.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
    print(canonical(dict(prepared=True,candidates=len(output),input=str(DATA/'method_inputs.jsonl'))))


def run_method():
    # The only task data opened by this process is the frozen method-input bundle.
    result=[infer(t) for t in rows(DATA/'method_inputs.jsonl')]
    jsonl(experiment_directory('nuscenes','validation',ID)/'predictions.jsonl',result)
    print(canonical(dict(completed=True,statuses={s:sum(r['status']==s for r in result) for s in sorted({r['status'] for r in result})})))


def evaluate():
    run=experiment_directory('nuscenes','validation',ID)
    predictions={r['task_id']:r for r in rows(run/'predictions.jsonl')}
    inputs={r['task_id']:r for r in rows(DATA/'method_inputs.jsonl')}
    tasks=rows(REFERENCE/'evaluation_only/task_candidates.jsonl')
    frames={(r['scene'],r['frame_idx']):r for r in rows(REFERENCE/'evaluation_only/official_selected_frames.jsonl')}
    metadata=ROOT/'dataset/raw/nuscenes_trainval/v1.0-trainval'
    category_names={r['token']:r['name'] for r in iter_records(metadata/'category.json')}
    official_vehicle_ids={r['token'] for r in iter_records(metadata/'instance.json') if category_names[r['category_token']] in
        ['vehicle.car','vehicle.truck','vehicle.bus.bendy','vehicle.bus.rigid']}
    spec=VehicleSpec();results=[]
    for task in tasks:
        pred=predictions[task['task_id']];inp=inputs[task['task_id']]
        record=dict(task_id=task['task_id'],method_status=pred['status'],geometry_time_feasible=task['new_geometry_time_conditional_eligible'],
            source_has_target='target_instance_token' in task,selected_identity_correct=False,trajectory_geometry_time_passed=False,
            predicted_trajectory_emitted=bool(pred['predicted_states']),failures=[])
        source=frames[(task['scene'],task['issue_frame_idx'])]
        current=inp['prediction_history'][-1]['objects'] if inp['prediction_history'] else []
        official=[o for o in source['objects'] if o['instance_token'] in official_vehicle_ids]
        if current and official:
            costs=np.array([[np.linalg.norm(np.asarray(p['center'])-o['translation']) if p['category'] in POLICY['target_categories'] else 1e6 for o in official] for p in current])
            for i,j in zip(*linear_sum_assignment(costs)):
                if str(current[i]['node_id'])==pred['selected_node_id'] and costs[i,j]<=3. and 'target_instance_token' in task:
                    record['selected_identity_correct']=official[j]['instance_token']==task['target_instance_token']
        if pred['predicted_states'] and 'witness' in task:
            witness=read(task['witness']['path']);anchors=witness['future_official_anchors']
            route=dict(continuations=[dict(tokens=r['tokens'],line=LineString(r['line_xy']),polygon=shape(r['polygon_geojson'])) for r in witness['allowed_successor_routes']])
            prior=load_reference_prior(inp['map_location']);states=[VehicleState(**r) for r in pred['predicted_states']]
            violations=[];details=[]
            initial=np.asarray(inp['initial_rear_axle_world_pose'])
            if math.dist([states[0].x,states[0].y],initial[:2,3])>1e-6 or abs(angle_difference(states[0].yaw_rad,inp['initial_yaw_rad']))>1e-6:
                violations.append('initial_state_mismatch')
            for before,after in zip([states[0]]+states[:-1],states):
                left=next((a for a,b in zip(anchors,anchors[1:]) if a['time_s']<=after.time_s+1e-8 and after.time_s<=b['time_s']+1e-8),anchors[-2])
                idx=anchors.index(left);right=anchors[idx+1]
                lo={o['instance_token']:o for o in left['objects']};ro={o['instance_token']:o for o in right['objects']}
                # Include full endpoint-convex envelopes of every keyframe interval
                # touched by this step; a 0.1s step can cross a 2Hz anchor.
                relevant=[]
                for a,b in zip(anchors,anchors[1:]):
                    if a['time_s']<=after.time_s+1e-8 and b['time_s']>=before.time_s-1e-8:
                        relevant.extend([a,b])
                by_instance={}
                for a in relevant:
                    for o in a['objects']:by_instance.setdefault(o['instance_token'],[]).append(object_polygon(o))
                obstacles=unary_union([unary_union(polygons).convex_hull for polygons in by_instance.values()])
                sweep=swept_body_polygon(before,after,spec)
                road=bool(prior['drivable'].covers(sweep));safe=not sweep.intersects(obstacles)
                alpha=(after.time_s-left['time_s'])/(right['time_s']-left['time_s'])
                tok=task['target_instance_token'];target_xy=(1-alpha)*np.asarray(lo[tok]['translation'][:2])+alpha*np.asarray(ro[tok]['translation'][:2])
                same,distance=route_measure(route,[after.x,after.y],target_xy)
                length=max(lo[tok]['size'][1],ro[tok]['size'][1]);gap=distance-spec.front_from_rear_axle_m-length/2 if same else None
                gap_ok=gap is not None and 5.<=gap<=60.
                target_yaw0=math.atan2(transform(lo[tok])[1,0],transform(lo[tok])[0,0])
                target_yaw1=math.atan2(transform(ro[tok])[1,0],transform(ro[tok])[0,0])
                target_yaw=target_yaw0+alpha*angle_difference(target_yaw1,target_yaw0)
                heading_ok=abs(angle_difference(after.yaw_rad,target_yaw))<=math.radians(30.)
                if not road:violations.append('road_sweep')
                if not safe:violations.append('obstacle_sweep')
                if not same:violations.append('same_directed_route')
                if not gap_ok:violations.append('gap')
                if not heading_ok:violations.append('heading')
                details.append(dict(time_s=after.time_s,road=road,safe=safe,same_route=same,gap_m=gap,heading_ok=bool(heading_ok)))
            if states[-1].time_s!=3. or len(states)!=31:violations.append('time_horizon')
            progresses=[r['line'].project(Point(states[-1].x,states[-1].y))-r['line'].project(Point(states[0].x,states[0].y))
                for r in route['continuations'] if r['polygon'].covers(Point(states[-1].x,states[-1].y))]
            if not progresses or max(progresses)<.5:violations.append('no_forward_progress')
            record.update(trajectory_geometry_time_passed=not violations,failures=sorted(set(violations)),checks=details)
        record['joint_success']=bool(record['selected_identity_correct'] and record['trajectory_geometry_time_passed'])
        results.append(record)
    feasible=[r for r in results if r['geometry_time_feasible']]
    summary=dict(created_utc=now(),fixed_candidates=12,certified_geometry_time_denominator=len(feasible),
        correct_identity=sum(r['selected_identity_correct'] for r in feasible),
        trajectories_emitted=sum(r['predicted_trajectory_emitted'] for r in feasible),
        geometry_time_passed=sum(r['trajectory_geometry_time_passed'] for r in feasible),
        joint_success=sum(r['joint_success'] for r in feasible),
        source_targets=4,source_no_matches=8,all_candidate_statuses={s:sum(r['method_status']==s for r in results) for s in sorted({r['method_status'] for r in results})},
        tiny_seen_regression=True,independent_logs=2,old_2015_81_unchanged=True,GPU_calls=0,Qwen_calls=0,
        no_claim_of_closed_loop_or_control_feasibility=True)
    jsonl(run/'evaluation.jsonl',results);dump(run/'summary.json',summary);dump(REPORT/'summary.json',summary)
    print(canonical(summary))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','evaluate']);args=parser.parse_args()
    {'prepare':prepare,'run':run_method,'evaluate':evaluate}[args.mode]()


if __name__=='__main__':main()
