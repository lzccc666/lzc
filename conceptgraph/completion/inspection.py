"""保存真实RGB、crop、掩膜/稀疏深度与独立投影证据，供人工核验。"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def projected_box(obj, pose, k):
    width,length,height=obj['size_wlh']
    corners=np.array([[x,y,z] for x in [-length/2,length/2]
        for y in [-width/2,width/2] for z in [-height/2,height/2]])
    w,x,y,z=obj['rotation_wxyz']
    rotation=Rotation.from_quat([x,y,z,w]).as_matrix()
    world=corners@rotation.T+obj['center']
    camera=(world-pose[:3,3])@pose[:3,:3]
    if np.any(camera[:,2]<=0.1):
        return None
    uv=camera@k.T
    uv=uv[:,:2]/uv[:,2:]
    return np.concatenate([uv.min(axis=0),uv.max(axis=0)]).tolist()


def box_iou(a,b):
    lo=np.maximum(a[:2],b[:2])
    hi=np.minimum(a[2:],b[2:])
    inter=float(np.prod(np.maximum(hi-lo,0)))
    aa=float(np.prod(np.maximum(np.asarray(a[2:])-a[:2],0)))
    bb=float(np.prod(np.maximum(np.asarray(b[2:])-b[:2],0)))
    return inter/max(aa+bb-inter,1e-9)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mapping-run',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--method',default='one_to_one_timestamp')
    args=parser.parse_args()
    source=OUTPUT/'mapping_evaluation'/args.mapping_run
    dest=OUTPUT/'annotation'/args.run_id
    if dest.exists():
        raise FileExistsError('不覆盖核验记录')
    dest.mkdir(parents=True)
    cases=[]
    errors=[]
    for framespath in sorted(source.glob('scene-*/frames.jsonl')):
        frames=list(map(json.loads,framespath.open()))
        indices=sorted({0,len(frames)//2,len(frames)-1})
        for frame_idx in indices:
            frame=frames[frame_idx]
            scene=frame['scene']
            base=dest/f'{scene}_{frame_idx:03d}'
            base.mkdir()
            rgb=Image.open(frame['rgb']).convert('RGB')
            rgb.save(base/'rgb.jpg',quality=90)
            depth=np.array(Image.open(frame['depth']),dtype=float)/1000
            with gzip.open(frame['detections'],'rb') as f:
                gobs=pickle.load(f)
            k=np.asarray(frame['intrinsic'])
            pose=np.asarray(frame['camera_pose'])
            official=[]
            for obj in frame['evaluation_only_official']:
                official.append(dict(obj,projected_bbox_xyxy=projected_box(obj,pose,k)))
            detections=[]
            for i,(bbox,mask) in enumerate(zip(gobs['xyxy'],gobs['mask'])):
                bbox=np.asarray(bbox,dtype=float)
                mask=np.asarray(mask,dtype=bool)
                if mask.shape!=depth.shape:
                    bbox=bbox*np.array([rgb.width/mask.shape[1],rgb.height/mask.shape[0]]*2)
                    mask=np.asarray(Image.fromarray(mask).resize(rgb.size,Image.Resampling.NEAREST),dtype=bool)
                valid=mask & (depth>0) & np.isfinite(depth)
                ys,xs=np.nonzero(valid)
                values=depth[valid]
                cropname=f'crop_{i:03d}.jpg'
                x1,y1,x2,y2=bbox
                cropbox=(max(0,int(x1)),max(0,int(y1)),min(rgb.width,int(x2)+1),min(rgb.height,int(y2)+1))
                if cropbox[2]<=cropbox[0] or cropbox[3]<=cropbox[1]:
                    cropname=None
                else:
                    rgb.crop(cropbox).save(base/cropname,quality=92)
                actual_cropname=None
                if i < len(gobs.get('image_crops', [])):
                    actual_crop=gobs['image_crops'][i]
                    if not isinstance(actual_crop,Image.Image):
                        actual_crop=Image.fromarray(np.asarray(actual_crop,dtype='uint8'))
                    actual_cropname=f'actual_clip_crop_{i:03d}.png'
                    actual_crop.save(base/actual_cropname)
                # 用可视化掩膜边界与有效点证据区分颜色图和真实语义标签。
                maskname=f'mask_{i:03d}.png'
                Image.fromarray((mask*255).astype('uint8')).save(base/maskname)
                cat=gobs['classes'][int(gobs['class_id'][i])]
                candidates=[dict(instance_token=o['instance_token'],
                    bbox_iou=box_iou(bbox,o['projected_bbox_xyxy'])) for o in official
                    if o['category']==cat and o['projected_bbox_xyxy'] is not None]
                candidates.sort(key=lambda x:-x['bbox_iou'])
                sample=np.linspace(0,max(len(xs)-1,0),min(len(xs),200),dtype=int)
                detections.append(dict(detection_index=i,category=cat,bbox_xyxy=bbox.tolist(),
                    confidence=float(gobs['confidence'][i]),mask_area_px=int(mask.sum()),
                    valid_lidar_pixels=int(valid.sum()),depth_quantiles_m=np.quantile(values,[0,.1,.5,.9,1]).tolist() if len(values) else None,
                    valid_depth_uvz=[[int(xs[j]),int(ys[j]),float(values[j])] for j in sample],
                    crop=cropname,mask=maskname,official_projection_candidates=candidates,
                    actual_clip_crop=actual_cropname,
                    actual_clip_crop_file_sha256=digest(base/actual_cropname) if actual_cropname else None,
                    frontend_provenance=gobs.get('frame_provenance'),
                    annotation_status='自动证据，未人工审核',color='unknown'))
            for obj in official:
                identity=obj['instance_token']
                raw=[(d,max((c['bbox_iou'] for c in d['official_projection_candidates'] if c['instance_token']==identity),default=0)) for d in detections]
                raw=[(d,s) for d,s in raw if s>=.1]
                newm=frame['metrics'][args.method]
                matched={official[m['truth_index']]['instance_token'] for m in newm['matches']}
                if identity in matched:
                    code='中心距离关联成功_仍需身份核验'
                elif not raw:
                    code='无同类别二维投影匹配_检测或投影待核验'
                elif max(d['valid_lidar_pixels'] for d,s in raw)<5:
                    code='掩膜内有效深度不足5像素'
                else:
                    code='三维定位或跨帧关联待核验'
                errors.append(dict(scene=scene,frame_idx=frame_idx,instance_token=identity,
                    diagnosis=code,evidence=str(base),human_verified=False))
            case=dict(scene=scene,frame_idx=frame_idx,rgb='rgb.jpg',
                detections=detections,official_evaluation_only=official,
                predictions=frame['predictions'],metrics=frame['metrics'],
                source_frame_file=str(framespath),source_frame_file_sha256=digest(framespath),
                camera_pose=frame['camera_pose'],intrinsic=frame['intrinsic'])
            dump(base/'evidence.json',case)
            cases.append(dict(scene=scene,frame_idx=frame_idx,url=str((base/'evidence.json').relative_to(OUTPUT))))
    dump(dest/'index.json',dict(cases=cases,case_count=len(cases),human_review_count=0,
        diagnostic_method=args.method,
        selection_rule='每开发场景首帧、中间帧、末帧；不按成功选择',
        human_labels_required=['颜色','细属性','歧义实例对应'],source_mapping_run=str(source)))
    with (dest/'error_inventory.jsonl').open('x') as f:
        for e in errors:
            f.write(json.dumps(e,ensure_ascii=False)+'\n')
    print(json.dumps(dict(cases=len(cases),official_object_cases=len(errors),human_review_count=0)))


if __name__=='__main__':
    main()
