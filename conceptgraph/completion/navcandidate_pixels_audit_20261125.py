"""Reproject the real depth pixels supporting every confirmed sparse history."""
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
from PIL import Image

from conceptgraph.completion.navcandidate_independent_20261125 import ROOT,REPORT,read,write


def rows(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def prepare_masks(frame,cfg):
    with gzip.open(frame['detections'],'rb') as stream:gobs=pickle.load(stream)
    depth=np.asarray(Image.open(frame['depth']),float)/1000.;height,width=depth.shape
    masks=np.asarray(gobs['mask'],bool);boxes=np.asarray(gobs['xyxy'],float).copy();keep=[]
    if masks.shape[1:]!=(height,width):raise ValueError('unexpected_mask_resize')
    for i,mask in enumerate(masks):
        cat=gobs['classes'][gobs['class_id'][i]];x1,y1,x2,y2=boxes[i]
        if mask.sum()<max(cfg['mask_area_threshold'],10):continue
        if cfg['skip_bg'] and cat in ['wall','floor','ceiling']:continue
        if (x2-x1)*(y2-y1)>cfg['max_bbox_area_ratio']*height*width:continue
        if gobs['confidence'] is not None and gobs['confidence'][i]<cfg['mask_conf_threshold']:continue
        keep.append(i)
    masks=masks[keep].copy();boxes=boxes[keep];areas=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            lo=np.maximum(boxes[i,:2],boxes[j,:2]);hi=np.minimum(boxes[i,2:],boxes[j,2:]);overlap=float(np.prod(np.maximum(hi-lo,0.)))
            if overlap/areas[i]<.7 and overlap/areas[j]>.8:masks[i]&=~masks[j]
    return depth,masks,keep,gobs


def main():
    begun=time.monotonic();folder=ROOT/'dataset/processed/mf_position_20261125_v2';run=ROOT/'outputs/results/nuscenes/validation/iterations/mf_position_20261125_v2';manifest=read(folder/'manifest.json');core=read(REPORT/'independent_results_v2.json')
    keys={key for r in core['qualified_state_frames'] for key in r['raw_keys']};unique={};frame_inputs={(r['scene'],r['frame_idx']):r for r in rows(folder/'method_frames.jsonl')}
    for frame in rows(run/'frames.jsonl.gz'):
        for raw in frame['ledger']['raw_observations']:
            if raw['key'] in keys:unique[raw['key']]=raw
    byframe={}
    for raw in unique.values():byframe.setdefault((raw['scene'],raw['frame_idx']),[]).append(raw)
    failures=[];details=[];hist=Counter()
    if set(unique)!=keys:failures.append('missing_qualified_raw_key')
    for framekey,observations in byframe.items():
        source=frame_inputs[framekey];depth,masks,original_indices,gobs=prepare_masks(source,manifest['config']);k=np.asarray(source['intrinsic']);pose=np.asarray(source['camera_pose'])
        for raw in observations:
            index=raw['filtered_detection_index'];mask=masks[index];v,u=np.nonzero(mask & np.isfinite(depth) & (depth>1.) & (depth<65.));z=depth[v,u];order=np.argsort(z,kind='stable');sorted_z=z[order]
            splits=np.flatnonzero(np.diff(sorted_z)>np.maximum(1.,.03*sorted_z[:-1]))+1;clusters=np.split(order,splits);chosen=clusters[raw['depth_evidence']['selected_cluster']]
            vv,uu,zz=v[chosen],u[chosen],z[chosen];selected=np.zeros_like(mask);selected[vv,uu]=True
            xyz=np.column_stack([(uu-k[0,2])*zz/k[0,0],(vv-k[1,2])*zz/k[1,1],zz])@pose[:3,:3].T+pose[:3,3]
            center=np.median(xyz,axis=0);residual=float(np.max(np.abs(center-raw['raw_measured_center'])));digest=hashlib.sha256(np.packbits(selected).tobytes()).hexdigest()
            okay=(residual<1e-8 and len(chosen)==raw['measured_points']==raw['depth_evidence']['retained_pixels'] and digest==raw['selected_mask_sha256'] and original_indices[index]==raw['original_detection_index'] and len(chosen)/len(v)==raw['depth_evidence']['support_fraction'])
            if not okay:failures.append(raw['key'])
            hist[str(len(chosen))]+=1;details.append(dict(key=raw['key'],passed=okay,points=len(chosen),center_maximum_residual_m=residual,selected_mask_sha256_matches=digest==raw['selected_mask_sha256']))
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=not failures,failures=failures,qualified_state_frames=len(core['qualified_state_frames']),qualified_unique_measurements=len(unique),qualified_frame_count=len(byframe),point_count_histogram=dict(hist),details=details,elapsed_s=time.monotonic()-begun,
        scope='All unique real observations appearing in any confirmed sparse-state history were reprojected from frozen measured depth PNG and cached filtered masks; no mapper/model was run. Unconfirmed histories are covered by time/key/lineage accounting in the core audit, not pixel reprojected here.',model_calls=0,training_calls=0,planner_calls=0,GPU_calls=0)
    write(REPORT/'qualified_pixels_v2.json',result)
    print(json.dumps({k:result[k] for k in ['passed','failures','qualified_state_frames','qualified_unique_measurements','qualified_frame_count','point_count_histogram','elapsed_s']},ensure_ascii=False))


if __name__=='__main__':main()
