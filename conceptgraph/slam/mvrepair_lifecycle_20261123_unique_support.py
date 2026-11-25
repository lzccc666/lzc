"""O1：同曝光原LiDAR实测支持唯一归属；只重算几何，不归并身份。

先被动追溯原算法真正保留的点，再按原检测置信度分配冲突点。
不同表面/未共享原点不会因原mask重合而被吞并。
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from conceptgraph.slam.mvrepair_lifecycle_20261123_sameview import SameViewCanonicalMapper,SameViewOriginalFrontMapper
from conceptgraph.slam import utils as slam_utils


def allocate_unique_support(observations):
    """scope=(camera, exposure, original scan)；不跨相机/时刻/scan竞争。"""
    claims=defaultdict(list)
    for i,r in enumerate(observations):
        if not r['lidar_token'] or not r['selected_point_indices'] or r['point_provenance_status'] not in ['exact_voxel_contributor_indices','unperturbed_sparse_pixel_reprojection','fixture']:continue
        for point in r['selected_point_indices']:
            claims[(*r['source_key'][:2],r['lidar_token'],int(point))].append(i)
    owners={k:min(v,key=lambda i:(-observations[i]['confidence'],observations[i]['source_key'][2],i)) for k,v in claims.items()}
    decisions=[];allowed={}
    for i,r in enumerate(observations):
        original=set(r['selected_point_indices']);own={p for p in original
            if owners.get((*r['source_key'][:2],r['lidar_token'],int(p)),i)==i}
        if r['lidar_token'] and original and r['point_provenance_status'] in ['exact_voxel_contributor_indices','unperturbed_sparse_pixel_reprojection','fixture']:
            allowed[tuple(r['source_key'])]=own
        decisions.append(dict(source_key=r['source_key'],category=r['category'],lidar_token=r['lidar_token'],
            original_confidence=r['confidence'],actual_retained_points=r['actual_retained_points'],
            provenance=r['point_provenance_status'],original_selected_indices=sorted(original),
            retained_indices=sorted(own),lost_indices=sorted(original-own),
            loss_owners=[dict(point_index=p,owner_source_key=observations[owners[(*r['source_key'][:2],r['lidar_token'],p)]]['source_key']) for p in sorted(original-own)]))
    collisions=[dict(camera=k[0],capture_timestamp_s=k[1],lidar_token=k[2],point_index=k[3],
        claimant_source_keys=[observations[i]['source_key'] for i in v],owner_source_key=observations[owners[k]]['source_key'])
        for k,v in claims.items() if len(v)>1]
    return allowed,dict(rule='original detconf descending, original detection index tie-break; selected actual support only',
        source_decisions=decisions,point_collisions=collisions,
        collision_points=len(collisions),lost_support_points=sum(len(d['lost_indices']) for d in decisions),
        affected_sources=sum(bool(d['lost_indices']) for d in decisions),identity_merge=False,history_state_deleted=False)


class UniqueSupportMixin:
    def observation_policy(self,frame):
        original=super().observation_policy(frame)
        if original is None:raise ValueError('O1 requires frozen original depth-cluster policy')
        camera=next(c for c,f in self.group_frames if f is frame)
        def select(depth,mask,gobs,index):
            key=(camera,self.capture_times[camera],int(gobs['_audit_indices'][index]))
            allowed=getattr(self,'_unique_allowed',{}).get(key)
            if allowed is None:return original(depth,mask,gobs,index)
            # 保留原无深度像素的mask几何，只移除不归本实例的有效深度点。
            projection=self._unique_projection[camera]
            valid=np.isfinite(depth)&(depth>0)
            owned=np.zeros(mask.shape,bool)
            keep=np.isin(projection['point_index'],list(allowed))
            owned[projection['pixel_y'][keep],projection['pixel_x'][keep]]=True
            owned_mask=np.asarray(mask,bool)&(~valid|owned)
            return original(depth,owned_mask,gobs,index)
        return select

    def extract_detections(self,frame):
        self._unique_allowed={};self._unique_projection={}
        state=np.random.get_state()
        traces=[];old_process=slam_utils.process_pcd
        def capture(pcd,cfg,*args,**kwargs):
            points=np.asarray(pcd.points).copy();voxel=float(cfg.downsample_voxel_size)
            down,_,contributors=pcd.voxel_down_sample_and_trace(voxel,
                pcd.get_min_bound()-voxel*.5,pcd.get_max_bound()+voxel*.5)
            output=old_process(pcd,cfg,*args,**kwargs)
            traces.append(dict(original_input_world_points=points.tolist(),voxel_size_m=voxel,
                voxel_world_points=np.asarray(down.points).tolist(),voxel_input_contributors=[list(map(int,c)) for c in contributors],
                final_world_points=np.asarray(output.points).tolist()))
            return output
        slam_utils.process_pcd=capture
        try:fg,bg=super().extract_detections(frame)
        finally:slam_utils.process_pcd=old_process
        initial=self.sameview_audit
        # 新O1额外持久化带jitter的真实输入和完整voxel trace，供独立逐点复核。
        for r in initial['observations']:
            if r['path']!='dense':continue
            trace=next((t for t in traces if t['final_world_points']==r['actual_retained_world_points']),None)
            if trace is not None:
                obj=fg[r['path_index']];y,x=np.nonzero(obj['mask'][-1])
                r['dense_trace']=dict(**trace,input_pixel_y=y.tolist(),input_pixel_x=x.tolist())
        allowed,audit=allocate_unique_support(initial['observations'])
        audit['initial_trace_observations']=initial['observations']
        if audit['collision_points']:
            self._unique_allowed=allowed
            self._unique_projection={c:dict(np.load(v['projection_evidence'])) for c,v in self.sameview_views.items()}
            np.random.set_state(state)
            try:fg,bg=super().extract_detections(frame)
            finally:self._unique_allowed={}
            # 只对有精确来源的原观测限制；不确定来源保留，不伪造可追溯性。
            _,final=allocate_unique_support(self.sameview_audit['observations'])
            audit['final_remaining_collision_points']=final['collision_points']
            audit['final_current_sources']=[dict(source_key=z['source_key'],category=z['category'],
                actual_retained_points=z['actual_retained_points'],selected_point_indices=z['selected_point_indices'],
                point_provenance_status=z['point_provenance_status']) for z in self.sameview_audit['observations']]
        else:
            audit['final_remaining_collision_points']=0
            audit['final_current_sources']=[dict(source_key=z['source_key'],category=z['category'],
                actual_retained_points=z['actual_retained_points'],selected_point_indices=z['selected_point_indices'],
                point_provenance_status=z['point_provenance_status']) for z in initial['observations']]
        self.unique_support_audit=audit
        return fg,bg

    def update_group(self,*args,**kwargs):
        output=super().update_group(*args,**kwargs)
        self.audit['unique_support']=self.unique_support_audit
        return output


class UniqueSupportCanonicalMapper(UniqueSupportMixin,SameViewCanonicalMapper):
    def __init__(self,cfg,classes):
        # 同视身份合并关闭：本候选只有唯一支持归属，不叠L5身份清理。
        super().__init__(cfg,classes,audit_only=True)


class UniqueSupportOriginalFrontMapper(UniqueSupportMixin,SameViewOriginalFrontMapper):
    def __init__(self,cfg,classes):
        super().__init__(cfg,classes)
        self.sameview_audit_only=True
