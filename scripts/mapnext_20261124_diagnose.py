"""Post-hoc failure taxonomy from full GT and real cached mask/depth/observation evidence."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
from pyquaternion import Quaternion

from conceptgraph.completion.mv136_20261123_protocol import FRONT3
from conceptgraph.completion.quality import object_frame_metrics
from conceptgraph.completion.stage9 import PROJECT, dump
from conceptgraph.slam.multiview_20261123_fusion import in_view
from scripts.multiview_20261123_run_graph import load_frame, write_csv

P='mapnext_20261124_'


def projected_box(obj, view):
    w,l,h=obj['size_wlh']
    corners=np.asarray([[x*l/2,y*w/2,z*h/2] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    world=corners@Quaternion(obj['rotation_wxyz']).rotation_matrix.T+obj['center']
    pose=np.asarray(view['camera_pose'])
    camera=(world-pose[:3,3])@pose[:3,:3]
    camera=camera[camera[:,2]>1]
    if not len(camera):
        return None
    uvz=camera@np.asarray(view['intrinsic']).T
    uv=uvz[:,:2]/uvz[:,2:]
    return np.r_[uv.min(0),uv.max(0)]


def iou(a,b):
    intersect=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    union=np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-intersect
    return float(intersect/union) if union else 0.


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--ground-truth',type=Path,required=True)
    ap.add_argument('--report-dir',type=Path,required=True)
    ap.add_argument('--method',default='T3_three')
    args=ap.parse_args()
    args.report_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text())
    gt={(r['scene'],r['frame_idx']):r for r in map(json.loads,args.ground_truth.open())}
    counts,by_scene,by_class=Counter(),defaultdict(Counter),defaultdict(Counter)
    records,fp_records,id_records=[],[],[]
    for scene in manifest['scenes']:
        name=scene['scene']
        seq=list(map(json.loads,gzip.open(args.run/name/(P+'predictions.jsonl.gz'),'rt')))
        last={}
        with gzip.open(args.report_dir/(name+'_all_GT_evidence.jsonl.gz'),'wt') as evidence_stream:
            for row,group in zip(seq,scene['groups']):
                truth=[o for o in gt[name,row['frame_idx']]['objects'] if set(o['visible_cameras'])&set(FRONT3)]
                predictions=[p for p in row['predictions'][args.method] if any(in_view(p['center'],group['views'][c]) for c in FRONT3)]
                metrics=object_frame_metrics(predictions,truth,threshold_m=3.)
                matched={m['truth_index']:m for m in metrics['matches']}
                used={m['prediction_index'] for m in metrics['matches']}
                audit=row['audit'][args.method]
                observations=audit['mapnext_observations']
                frames={c:load_frame(group,c) for c in FRONT3}
                for index,obj in enumerate(truth):
                    potential=[]
                    for camera in set(obj['visible_cameras'])&set(FRONT3):
                        view,frame=group['views'][camera],frames[camera]
                        box=projected_box(obj,view)
                        if box is None:
                            continue
                        det=frame.detections
                        for di,detection_box in enumerate(det['xyxy']):
                            overlap=iou(box,np.asarray(detection_box))
                            if overlap<.1:
                                continue
                            mask=np.asarray(det['mask'][di],bool)
                            depth=np.asarray(frame.depth_array)
                            valid=mask&np.isfinite(depth)&(depth>1)&(depth<65)
                            category=det['classes'][int(det['class_id'][di])]
                            potential.append(dict(camera=camera,detection_index=di,category=category,
                                GT_box_IoU=overlap,confidence=float(det['confidence'][di]),
                                raw_mask_pixels=int(mask.sum()),raw_valid_depth_pixels=int(valid.sum()),
                                source_key=[camera,view['camera_timestamp_s'],di]))
                    same=[p for p in potential if p['category']==obj['category']]
                    keys={tuple(p['source_key']) for p in same}
                    linked=[o for o in observations if o['source_key'] and tuple(o['source_key']) in keys and o['category']==obj['category']]
                    best=min((float(np.linalg.norm(np.asarray(o['center'][:2])-obj['center'][:2])) for o in linked),default=None)
                    if index in matched:
                        kind='matched_3D_map'
                    elif not same:
                        kind='no_same_class_2D_evidence' if not potential else '2D_class_or_overlap_ambiguity'
                    elif max(p['raw_valid_depth_pixels'] for p in same)==0:
                        kind='2D_evidence_but_no_current_depth'
                    elif not linked:
                        kind='depth_exists_but_no_accepted_measurement'
                    elif best>3:
                        kind='accepted_measurement_outside_3m'
                    else:
                        kind='measurement_within_3m_but_output_unmatched'
                    counts[kind]+=1
                    by_scene[name][kind]+=1
                    by_class[obj['category']][kind]+=1
                    record=dict(scene=name,log_token=scene['log_token'],frame_idx=row['frame_idx'],
                        instance_token=obj['instance_token'],category=obj['category'],classification=kind,
                        num_official_lidar_points=obj.get('num_lidar_pts'),visibility_token=obj.get('visibility_token'),
                        projected_2D_evidence=len(potential),same_class_2D_evidence=len(same),
                        maximum_raw_depth_points=max((p['raw_valid_depth_pixels'] for p in same),default=0),
                        accepted_linked_measurements=len(linked),nearest_observation_error_m=best)
                    records.append(record)
                    evidence_stream.write(json.dumps(dict(**record,GT_evaluation_only=obj,
                        possible_2D_evidence=potential,linked_measurements=linked),allow_nan=False)+'\n')
                    if index in matched:
                        node=predictions[matched[index]['prediction_index']]['node_id']
                        previous=last.get(obj['instance_token'])
                        if previous and previous[0]==row['frame_idx']-1 and previous[1]!=node:
                            id_records.append(dict(scene=name,frame_idx=row['frame_idx'],instance_token=obj['instance_token'],
                                category=obj['category'],previous_node=previous[1],current_node=node,
                                status='automatic_3m_pairing_diagnostic_not_human_identity'))
                        last[obj['instance_token']]=(row['frame_idx'],node)
                for i,pred in enumerate(predictions):
                    if i in used:
                        continue
                    candidates=[(float(np.linalg.norm(np.asarray(pred['center'][:2])-obj['center'][:2])),obj) for obj in truth if obj['category']==pred['category']]
                    distance,obj=min(candidates,key=lambda v:v[0]) if candidates else (None,None)
                    fp_records.append(dict(scene=name,log_token=scene['log_token'],frame_idx=row['frame_idx'],
                        node_id=pred['node_id'],category=pred['category'],center=pred['center'],
                        nearest_same_class_distance_m=distance,duplicate_metric=distance is not None and distance<=3,
                        nearest_GT_instance=obj['instance_token'] if obj else None,position_source=pred.get('position_source'),
                        prediction_age_s=pred.get('prediction_age_s')))
    write_csv(args.report_dir/'all_GT_taxonomy.csv',records)
    write_csv(args.report_dir/'all_unmatched_predictions.csv',fp_records)
    write_csv(args.report_dir/'all_automatic_ID_changes.csv',id_records)
    dump(args.report_dir/'failure_taxonomy_summary.json',dict(total_GT=sum(counts.values()),counts=dict(counts),
        by_scene={k:dict(v) for k,v in by_scene.items()},by_category={k:dict(v) for k,v in by_class.items()},
        unmatched_predictions=len(fp_records),duplicate_metric=sum(r['duplicate_metric'] for r in fp_records),
        automatic_ID_changes=len(id_records),GT_in_inference=False,
        interpretation='posthoc evidence taxonomy, projected GT box IoU>=.1; not a 2D detector score or definitive physical attribution',
        projection_limits='official sample-time boxes vs asynchronous exposures, occluded center/frustum objects retained; unknown visual identity not invented'))
    print(json.dumps(dict(counts),ensure_ascii=False))


if __name__=='__main__':
    main()
