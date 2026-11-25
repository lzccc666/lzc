"""L5：当前同曝光同类的原始LiDAR支持包含；不改L1/L3/L4。

同scan原索引+像素包含，保留独立物体证据；不凭共享深度强认相邻实例。
"""
from __future__ import annotations

from dataclasses import asdict,dataclass

import numpy as np
from scipy.spatial import cKDTree

from conceptgraph.completion.stage9 import predict
from conceptgraph.slam.mviter_20261123_diagnostics import SourceAuditedMapper
from conceptgraph.slam.mvrepair_20261123_combined import CanonicalPhysicalFrontMapper
from conceptgraph.slam.slam_classes import DetectionList
from conceptgraph.slam.sparse_observations import cosine
from conceptgraph.slam import utils as slam_utils


@dataclass(frozen=True)
class SameViewSettings:
    minimum_shared_original_points:int=5
    minimum_smaller_support_containment:float=.9
    minimum_feature_cosine:float=.8
    maximum_xy_observation_center_m:float=.5
    maximum_z_observation_center_m:float=.5
    minimum_smaller_mask_containment:float=.5
    independent_dense_support_points:int=5


def sameview_witness(a,b,settings=None):
    s=settings or SameViewSettings()
    result=dict(source_key_a=a['source_key'],source_key_b=b['source_key'],accepted=False,reason='different_class_or_exposure')
    if a['category']!=b['category'] or a['source_key'][:2]!=b['source_key'][:2]:return result
    if not a['lidar_token'] or a['lidar_token']!=b['lidar_token']:
        result['reason']='missing_or_different_original_scan';return result
    ia,ib=set(a['selected_point_indices']),set(b['selected_point_indices'])
    common=len(ia&ib);smaller=min(len(ia),len(ib))
    result.update(lidar_token=a['lidar_token'],shared_original_points=common,smaller_selected_points=smaller,
        selected_support_containment=common/smaller if smaller else 0.,independent_points_a=len(ia-ib),independent_points_b=len(ib-ia))
    if common<s.minimum_shared_original_points or not smaller or common/smaller<s.minimum_smaller_support_containment:
        result['reason']='insufficient_original_point_containment';return result
    if len(ia-ib)>=s.independent_dense_support_points and len(ib-ia)>=s.independent_dense_support_points:
        result['reason']='both_hypotheses_have_independent_dense_support';return result
    delta=np.asarray(a['center'])-b['center'];appearance=cosine(a['feature'],b['feature'])
    result.update(xy_observation_distance_m=float(np.linalg.norm(delta[:2])),z_observation_distance_m=float(abs(delta[2])),feature_cosine=appearance)
    if not np.isfinite([*delta,appearance]).all() or appearance<s.minimum_feature_cosine:
        result['reason']='appearance_or_finite_gate';return result
    if result['xy_observation_distance_m']>s.maximum_xy_observation_center_m or result['z_observation_distance_m']>s.maximum_z_observation_center_m:
        result['reason']='current_observation_geometry_gate';return result
    ma,mb=np.asarray(a['raw_mask'],bool),np.asarray(b['raw_mask'],bool)
    if ma.shape!=mb.shape:result['reason']='pixel_frame_mismatch';return result
    size=min(int(ma.sum()),int(mb.sum()));overlap=int((ma&mb).sum())
    result['smaller_mask_containment']=overlap/size if size else 0.
    if not size or overlap/size<=s.minimum_smaller_mask_containment:
        result['reason']='no_majority_pixel_containment';return result
    ba,bb=np.asarray(a['box'],float),np.asarray(b['box'],float)
    small,large=(ba,bb) if np.prod(ba[2:]-ba[:2])<=np.prod(bb[2:]-bb[:2]) else(bb,ba)
    center=(small[:2]+small[2:])/2
    if not bool(np.all(center>=large[:2]) and np.all(center<=large[2:])):
        result['reason']='smaller_box_center_outside_other_independent_image_risk';return result
    result.update(accepted=True,reason='same_exposure_original_support_and_image_containment')
    return result


def sameview_groups(records,settings=None):
    settings=settings or SameViewSettings()
    pairs={(j,i):sameview_witness(records[j],r,settings) for i,r in enumerate(records) for j in range(i)}
    clusters=[]
    # 更完整的真实支持代表，conf仅作同支持数量平局；不合成点云或计数。
    for i in sorted(range(len(records)),key=lambda i:(-len(records[i]['selected_point_indices']),-records[i]['confidence'],i)):
        compatible=[c for c in clusters if all(pairs[min(i,j),max(i,j)]['accepted'] for j in c)]
        if compatible:compatible[0].append(i)
        else:clusters.append([i])
    return [c[0] for c in clusters],dict(settings=asdict(settings),pairs=list(pairs.values()),
        clusters=[dict(representative=c[0],members=c,source_keys=[records[j]['source_key'] for j in c]) for c in clusters],
        observations=[dict(source_key=r['source_key'],category=r['category'],lidar_token=r['lidar_token'],
            selected_point_indices=r['selected_point_indices'],confidence=r['confidence'],path=r['path'],path_index=r['path_index'],
            actual_retained_points=r.get('actual_retained_points'),point_provenance_status=r.get('point_provenance_status'),
            actual_retained_world_points=r.get('actual_retained_world_points')) for r in records],
        removed=len(records)-len(clusters))


class SameViewSupportMixin:
    def extract_detections(self,frame):
        raw={};frames=dict(self.group_frames)
        for camera,f in self.group_frames:
            d=f.detections
            for i in range(len(d['class_id'])):
                raw[camera,i]=dict(box=np.asarray(d['xyxy'][i]).copy(),raw_mask=np.asarray(d['mask'][i],bool).copy())
        # 只在本独立CPU入口被动记录原process_pcd，原算法原返回值不变。
        # 精确voxel trace将最终保留点追溯到原输入像素，避免拿深度区间13点冒充实际7点。
        processing=[];old_process=slam_utils.process_pcd
        def traced_process(pcd,cfg,*args,**kwargs):
            original_input=np.asarray(pcd.points).copy()
            voxel=float(cfg.downsample_voxel_size)
            traced,_,contributors=pcd.voxel_down_sample_and_trace(voxel,
                pcd.get_min_bound()-voxel*.5,pcd.get_max_bound()+voxel*.5)
            output=old_process(pcd,cfg,*args,**kwargs)
            final=np.asarray(output.points).copy();centers=np.asarray(traced.points)
            valid=False;input_indices=[]
            if len(final) and len(centers):
                distances,indices=cKDTree(centers).query(final)
                valid=bool(np.max(distances)<1e-7)
                if valid:input_indices=sorted({int(j) for i in indices for j in contributors[int(i)]})
            processing.append(dict(input_points=original_input,final_points=final,
                retained_input_indices=input_indices,exact_voxel_trace=valid))
            return output
        slam_utils.process_pcd=traced_process
        try:fg,bg=super().extract_detections(frame)
        finally:slam_utils.process_pcd=old_process
        records=[]
        for path,items in [('dense',fg),('sparse',self.sparse_measurements)]:
            for i,item in enumerate(items):
                if path=='dense':
                    src=self.measurement_sources[item['mask_idx'][-1]];key=src['_observation_key'];mask=np.asarray(item['mask'][-1],bool)
                    center,feature,category,confidence=src['center'],src['feature'],src['category'],src['confidence']
                    actual=np.asarray(src['world_points']);trace=next((p for p in processing
                        if p['final_points'].shape==actual.shape and np.array_equal(p['final_points'],actual)),None)
                    if trace is None or not trace['exact_voxel_trace']:
                        mask=np.zeros_like(mask);trace_status='unavailable_keep_hypothesis'
                    else:
                        y,x=np.nonzero(mask)
                        if len(y)!=len(trace['input_points']):
                            mask=np.zeros_like(mask);trace_status='pixel_input_count_conflict_keep_hypothesis'
                        else:
                            keep=np.asarray(trace['retained_input_indices'],int);mask=np.zeros_like(mask)
                            mask[y[keep],x[keep]]=True;trace_status='exact_voxel_contributor_indices'
                else:
                    key=item['_observation_key'];f=frames[key[0]];points=np.asarray(item['_world_points'])
                    camera=(points-f.camera_pose[:3,3])@f.camera_pose[:3,:3]
                    pixels=camera@f.cam_K.T;uv=np.rint(pixels[:,:2]/pixels[:,2,None]).astype(int)
                    mask=np.zeros(f.depth_array.shape,bool)
                    valid=(uv[:,0]>=0)&(uv[:,0]<mask.shape[1])&(uv[:,1]>=0)&(uv[:,1]<mask.shape[0])
                    mask[uv[valid,1],uv[valid,0]]=True
                    center,feature,category,confidence=item['center'],item['feature'],item['category'],item['confidence']
                    actual=np.asarray(item['_world_points']);trace_status='unperturbed_sparse_pixel_reprojection'
                view=self.sameview_views[key[0]];projection=np.load(view['projection_evidence'])
                selected=mask[projection['pixel_y'],projection['pixel_x']]
                ids=sorted(set(map(int,projection['point_index'][selected])))
                records.append(dict(source_key=key,category=category,center=center,feature=feature,confidence=confidence,
                    selected_point_indices=ids,lidar_token=view.get('lidar_token'),path=path,path_index=i,
                    actual_retained_points=len(actual),point_provenance_status=trace_status,
                    actual_retained_world_points=actual.tolist(),**raw[key[0],key[2]]))
        kept,audit=sameview_groups(records,self.sameview_settings);audit['audit_only']=self.sameview_audit_only
        self.sameview_audit=audit
        if self.sameview_audit_only:return fg,bg
        dr={r['path_index'] for i,r in enumerate(records) if i not in kept and r['path']=='dense'}
        sr={r['path_index'] for i,r in enumerate(records) if i not in kept and r['path']=='sparse'}
        self.sparse_measurements=[x for i,x in enumerate(self.sparse_measurements) if i not in sr]
        return DetectionList([x for i,x in enumerate(fg) if i not in dr]),bg


class SameViewCanonicalMapper(SameViewSupportMixin,CanonicalPhysicalFrontMapper):
    def __init__(self,cfg,classes,*,audit_only=False):
        super().__init__(cfg,classes);self.sameview_settings=SameViewSettings();self.sameview_audit_only=audit_only

    def update_group(self,frames,capture_times,projections=None):
        if projections is None:raise ValueError('同scan原索引见证需要真实projection_evidence和lidar_token')
        self.sameview_views=projections
        output=super().update_group(frames,capture_times,projections)
        self.audit['sameview']=self.sameview_audit
        return output


class SameViewOriginalFrontMapper(SameViewSupportMixin,SourceAuditedMapper):
    def __init__(self,cfg,classes):
        super().__init__(cfg,classes,deduplication=False);self.sameview_settings=SameViewSettings();self.sameview_audit_only=False

    def update_group(self,frames,capture_times,projections=None):
        if [c for c,_ in frames]!=['CAM_FRONT'] or projections is None:raise ValueError('原A同视清理只允许前视和真实projection')
        self.capture_times=capture_times;self.sameview_views=projections
        super().update_group(frames)
        output=self.position_predictions(predict(self.confirmed_objects(),self.classes),frames[0][1])
        self.audit=dict(sameview=self.sameview_audit,mapping=self.frame_audit,crossview=self.crossview_audit)
        return output
