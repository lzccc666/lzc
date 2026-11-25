"""Inspect saved sparse evidence; reproject real pixels without running a mapper."""
from collections import Counter
from datetime import datetime,timezone
import gzip
import itertools
import json
import math
import pickle

import numpy as np
from PIL import Image

from conceptgraph.completion.navfollowup_independent_20261124 import ROOT,REPORT,read,sha,write,protection


def rows(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else path.open()) as stream:
        return [json.loads(s) for s in stream if s.strip()]


def selected_points(frame,observation,cfg):
    with gzip.open(frame['detections'],'rb') as stream:gobs=pickle.load(stream)
    depth=np.asarray(Image.open(frame['depth']),float)/1000.;height,width=depth.shape
    masks=np.asarray(gobs['mask'],bool);boxes=np.asarray(gobs['xyxy'],float).copy()
    # Frozen detections in this bounded batch already match the depth image.
    if masks.shape[1:]!=(height,width):raise ValueError('unexpected_mask_resize')
    keep=[]
    for i,mask in enumerate(masks):
        category=gobs['classes'][gobs['class_id'][i]];x1,y1,x2,y2=boxes[i]
        if mask.sum()<max(cfg['mask_area_threshold'],10):continue
        if cfg['skip_bg'] and category in ['wall','floor','ceiling']:continue
        if (x2-x1)*(y2-y1)>cfg['max_bbox_area_ratio']*height*width:continue
        if gobs['confidence'] is not None and gobs['confidence'][i]<cfg['mask_conf_threshold']:continue
        keep.append(i)
    masks=masks[keep].copy();boxes=boxes[keep]
    areas=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            left=np.maximum(boxes[i,:2],boxes[j,:2]);right=np.minimum(boxes[i,2:],boxes[j,2:]);intersection=float(np.prod(np.maximum(right-left,0.)))
            if intersection/areas[i]<.7 and intersection/areas[j]>.8:masks[i]&=~masks[j]
    index=observation['filtered_detection_index'];mask=masks[index]
    v,u=np.nonzero(mask & np.isfinite(depth) & (depth>1.) & (depth<65.));z=depth[v,u]
    order=np.argsort(z,kind='stable');sorted_z=z[order];splits=np.flatnonzero(np.diff(sorted_z)>np.maximum(1.,.03*sorted_z[:-1]))+1
    clusters=np.split(order,splits);selected=clusters[observation['depth_evidence']['selected_cluster']]
    v,u,z=v[selected],u[selected],z[selected];k=np.asarray(frame['intrinsic']);pose=np.asarray(frame['camera_pose'])
    xyz=np.column_stack([(u-k[0,2])*z/k[0,0],(v-k[1,2])*z/k[1,1],z])@pose[:3,:3].T+pose[:3,3]
    category=gobs['classes'][gobs['class_id'][keep[index]]]
    return np.median(xyz,axis=0),len(xyz),category


def main():
    folder=ROOT/'dataset/processed/navstage2_sparse_evidence_20261124_v1';run=ROOT/'outputs/results/nuscenes/validation/iterations/navstage2_sparse_evidence_20261124_v1';manifest=read(folder/'manifest.json');summary=read(run/'summary.json')
    failures=[];hashes=[];snapshots=[]
    for group in ['source_sha256','input_sha256']:
        for path,expected in manifest[group].items():hashes.append(dict(path=path,passed=sha(path)==expected))
    for path,expected in manifest['source_sha256'].items():
        p=run/'source_snapshot'/__import__('pathlib').Path(path).relative_to(ROOT);snapshots.append(dict(path=str(p),passed=sha(p)==expected))
    frames={(r['scene'],r['frame_idx']):r for r in rows(folder/'method_frames.jsonl')};evidence=rows(run/'raw_navigation_evidence.jsonl.gz');counts=Counter();unique={};frame_checks=[];traces=[]
    reference={}
    for scene in manifest['scenes']:
        reference.update({(scene,r['frame_idx']):r['predictions']['R1'] for r in rows(ROOT/'outputs/results/nuscenes/validation/iterations/recall_refinement_R1_full_20261120_v2'/scene/'frames.jsonl.gz')})
    for row in evidence:
        key=row['scene'],row['frame_idx'];frame=frames[key]
        exact=row['predictions']==reference[key];frame_checks.append(dict(scene=key[0],frame_idx=key[1],predictions_exact=exact));counts['frames_exactly_reproduced']+=int(exact)
        if not exact:failures.append('changed_prediction')
        for track in row['sparse_navigation_evidence']:
            counts['sparse_track_frame_records']+=1;trace=track['raw_trace'];times=[x['timestamp_s'] for x in trace]
            causal=bool(trace) and all(t<=frame['timestamp_s'] for t in times) and all(b>a for a,b in zip(times,times[1:]))
            if not causal or track['causal']!=causal:failures.append('future_or_duplicate_trace')
            for observation in trace:
                source=frames[row['scene'],observation['frame_idx']]
                if source['timestamp_s']!=observation['timestamp_s'] or observation['future_used'] or observation['filtered_position_used_as_measurement']:failures.append('provenance_binding')
                obskey=(row['scene'],observation['frame_idx'],observation['filtered_detection_index'],observation['timestamp_s'])
                if obskey in unique and unique[obskey]!=observation:failures.append('duplicate_observation_value_changed')
                unique[obskey]=observation;counts['raw_trace_records']+=1
            speed=None
            if len(trace)>=2:
                vectors=[(np.asarray(b['raw_measured_center'])[:2]-np.asarray(a['raw_measured_center'])[:2])/(b['timestamp_s']-a['timestamp_s']) for a,b in itertools.combinations(trace,2) if b['timestamp_s']-a['timestamp_s']>=.75]
                vector=np.median(vectors,axis=0) if vectors else (np.asarray(trace[-1]['raw_measured_center'])[:2]-np.asarray(trace[-2]['raw_measured_center'])[:2])/(times[-1]-times[-2]);speed=float(np.linalg.norm(vector))
            motion=('static' if speed<=.5 else 'dynamic') if speed is not None and len(set(times))>=3 else 'unknown'
            if motion!=track['diagnostic_motion_from_raw_observations']:failures.append('raw_motion_recompute')
            latest=trace[-1]['depth_evidence'] if trace else {};current=bool(trace) and times[-1]==frame['timestamp_s'];depth=bool(current and latest.get('source')=='current_projected_lidar' and latest.get('units')=='m' and latest.get('retained_pixels',0)>=5 and latest.get('support_fraction',0)>=.4)
            complete=bool(causal and motion=='static' and depth)
            if depth!=track['current_depth_passed_original_gate'] or complete!=track['complete_original_navigation_evidence']:failures.append('navigation_evidence_recompute')
            counts['raw_motion_'+motion]+=1;counts['current_depth_passed']+=int(depth);counts['complete_original_navigation_evidence']+=int(complete)
    point_hist=Counter();reprojections=[]
    for key,observation in unique.items():
        center,points,category=selected_points(frames[key[0],key[1]],observation,manifest['config'])
        residual=float(np.max(np.abs(center-observation['raw_measured_center'])));okay=residual<1e-8 and points==observation['measurement_points']==observation['depth_evidence']['retained_pixels'] and category==observation['category']
        if not okay:failures.append('raw_center_not_measured_pixels')
        point_hist[str(points)]+=1;reprojections.append(dict(scene=key[0],frame_idx=key[1],filtered_detection_index=key[2],points=points,center_maximum_residual_m=residual,passed=okay))
    if dict(counts)!=summary['counts']:failures.append('summary_counts')
    if len(evidence)!=24 or len(unique)!=26 or counts['complete_original_navigation_evidence']!=0:failures.append('bounded_batch_totals')
    sem=ROOT/'dataset/processed/navsemantic_tasks_20261124_v3';append_manifest=read(sem/'effective_additions_manifest_v2.json');append_checks=[dict(path=str(sem/p),passed=sha(sem/p)==h) for p,h in append_manifest['files'].items()]
    frozen=read(sem/'geometry_time_denominator_freeze_v1.json');certified=[r['task_id'] for r in read(REPORT/'semantic_follow_package_v1.json')['witness_replays'] if r['passed']]
    if frozen['ids']!=certified or frozen['denominator']!=3 or frozen['methods_run_at_freeze']!=0:failures.append('semantic_denominator')
    overlays=rows(sem/'evaluation_only/initial_target_binding_overlay_v2.jsonl');no_target=[r for r in overlays if r['official_target_instance_token'] is None]
    if len(overlays)!=12 or len(no_target)!=8 or any(r['navigation_feasibility']!='pending' for r in no_target):failures.append('no_match_overlay')
    projection_checks=[]
    for row in rows(sem/'evaluation_only/reference_timed_xy_yaw_projection_v1.jsonl'):
        source=read(row['source_witness']['path']);expected=[{k:s[k] for k in ['time_s','x','y','yaw_rad']} for s in source['reference_states']]
        projection_checks.append(dict(task_id=row['task_id'],passed=row['states']==expected and sha(row['source_witness']['path'])==row['source_witness']['sha256']))
    preserved=protection()
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=not failures and all(r['passed'] for r in hashes+snapshots+append_checks+projection_checks) and preserved['unchanged'],failures=sorted(set(failures)),
        hash_checks=hashes,snapshot_checks=snapshots,frame_checks=frame_checks,counts=dict(counts),unique_exported_raw_observations=len(unique),unique_measurement_point_histogram=dict(point_hist),raw_pixel_reprojection_checks=reprojections,
        semantic_addition_checks=append_checks,semantic_projection_checks=projection_checks,semantic_frozen_ids=frozen['ids'],protection=preserved,
        new_navigation_authorizations=0,model_calls=0,planner_calls=0,GPU_calls=0,
        scope='26 unique observations are only those retained in the exported sparse-track traces, not all callback observations or dense-state evidence. Instrumentation preserves all 24 predictions and adds no performance gain. Full past/current real-pixel center replay was performed without invoking a mapper.')
    write(REPORT/'sparse_provenance_and_semantic_additions_v1.json',result)
    print(json.dumps({k:result[k] for k in ['passed','failures','counts','unique_exported_raw_observations','unique_measurement_point_histogram','new_navigation_authorizations']},ensure_ascii=False))


if __name__=='__main__':main()
