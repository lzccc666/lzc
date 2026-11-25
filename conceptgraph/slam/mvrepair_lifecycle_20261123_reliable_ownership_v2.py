"""O3：同曝光原LiDAR实测支持唯一归属；只重算几何，不归并身份。

先被动追溯原算法真正保留的点，再按原检测置信度分配冲突点。
不同表面/未共享原点不会因原mask重合而被吞并。
"""
from __future__ import annotations
from collections import defaultdict
import copy,hashlib
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
    all_claims=claims
    claims={k:v for k,v in all_claims.items() if all(observations[i].get('r1_reliable_current_dense',False) for i in v)}
    owners={k:min(v,key=lambda i:(-observations[i]['confidence'],observations[i]['source_key'][2],i)) for k,v in claims.items()}
    decisions=[];allowed={}
    for i,r in enumerate(observations):
        original=set(r['selected_point_indices']);own={p for p in original
            if owners.get((*r['source_key'][:2],r['lidar_token'],int(p)),i)==i}
        if own!=original and r['lidar_token'] and original and r['point_provenance_status'] in ['exact_voxel_contributor_indices','unperturbed_sparse_pixel_reprojection','fixture']:
            allowed[tuple(r['source_key'])]=own
        decisions.append(dict(source_key=r['source_key'],category=r['category'],lidar_token=r['lidar_token'],
            original_confidence=r['confidence'],actual_retained_points=r['actual_retained_points'],
            provenance=r['point_provenance_status'],r1_reliable_current_dense=r.get('r1_reliable_current_dense',False),original_selected_indices=sorted(original),
            retained_indices=sorted(own),lost_indices=sorted(original-own),
            loss_owners=[dict(point_index=p,owner_source_key=observations[owners[(*r['source_key'][:2],r['lidar_token'],p)]]['source_key']) for p in sorted(original-own)]))
    collisions=[dict(camera=k[0],capture_timestamp_s=k[1],lidar_token=k[2],point_index=k[3],
        claimant_source_keys=[observations[i]['source_key'] for i in v],owner_source_key=observations[owners[k]]['source_key'])
        for k,v in claims.items() if len(v)>1]
    return allowed,dict(rule='both original R1 reliable dense (points>=8,conf>=.2,XYvariance<=4); detconf descending/index tiebreak; only losing source rerun; ambiguous weak original update preserved',
        source_decisions=decisions,point_collisions=collisions,
        collision_points=len(collisions),ambiguous_deferred_collision_points=sum(len(v)>1 and not all(observations[i].get('r1_reliable_current_dense',False) for i in v) for v in all_claims.values()),lost_support_points=sum(len(d['lost_indices']) for d in decisions),
        affected_sources=sum(bool(d['lost_indices']) for d in decisions),identity_merge=False,history_state_deleted=False)


def input_signature(value):
    h=hashlib.sha256()
    def put(x):
        if isinstance(x,np.ndarray):
            h.update(str((x.dtype.str,x.shape)).encode());h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x,dict):
            for k in sorted(x):h.update(str(k).encode());put(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(str(len(x)).encode())
            for z in x:put(z)
        else:h.update(repr(x).encode())
    put(value);return h.hexdigest()


class ReliableUniqueSupportMixin:
    def observation_policy(self,frame):
        original=super().observation_policy(frame)
        if original is None:raise ValueError('O3 requires frozen original depth-cluster policy')
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
        original_inputs={c:copy.deepcopy(f.detections) for c,f in self.group_frames}
        original_signatures={c:input_signature(d) for c,d in original_inputs.items()}
        temporary_keys=['measurement_sources','sparse_measurements','frame_audit','crossview_audit','physical_audit','sameview_audit','group_views']
        temporary={k:copy.deepcopy(getattr(self,k)) for k in temporary_keys if hasattr(self,k)}
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
        self._ownership_after_probe_rng=np.random.get_state()
        mutation_signatures={c:input_signature(f.detections) for c,f in self.group_frames}
        # 新O3额外持久化带jitter的真实输入和完整voxel trace，供独立逐点复核。
        for r in initial['observations']:
            if r['path']!='dense':continue
            trace=next((t for t in traces if t['final_world_points']==r['actual_retained_world_points']),None)
            if trace is not None:
                obj=fg[r['path_index']];y,x=np.nonzero(obj['mask'][-1])
                r['dense_trace']=dict(**trace,input_pixel_y=y.tolist(),input_pixel_x=x.tolist())
        sources={tuple(s['_observation_key']):s for s in self.measurement_sources.values()}
        for r in initial['observations']:
            src=sources.get(tuple(r['source_key']),{})
            r['r1_reliable_current_dense']=bool(r['path']=='dense' and src.get('reliable',False))
            r['r1_variance']=np.asarray(src['variance']).tolist() if 'variance' in src else None
        allowed,audit=allocate_unique_support(initial['observations'])
        audit['initial_trace_observations']=initial['observations']
        audit['two_pass_control']=self.ownership_audit_only
        audit['input_signatures_before_probe']=original_signatures
        audit['input_signatures_after_probe']=mutation_signatures
        if audit['collision_points'] or self.ownership_audit_only:
            # 原入口会原地filter/扣mask；第二pass必须逐相机恢复原输入和临时状态。
            for c,f in self.group_frames:
                f.detections.clear();f.detections.update(copy.deepcopy(original_inputs[c]))
            assert {c:input_signature(f.detections) for c,f in self.group_frames}==original_signatures
            for k in temporary_keys:
                if k in temporary:setattr(self,k,copy.deepcopy(temporary[k]))
                elif hasattr(self,k):delattr(self,k)
            self._unique_allowed={} if self.ownership_audit_only else allowed
            self._unique_projection={c:dict(np.load(v['projection_evidence'])) for c,v in self.sameview_views.items()}
            np.random.set_state(state)
            try:fg,bg=super().extract_detections(frame)
            finally:self._unique_allowed={}
            # 只对有精确来源的原观测限制；不确定来源保留，不伪造可追溯性。
            final_sources={tuple(s['_observation_key']):s for s in self.measurement_sources.values()}
            for r in self.sameview_audit['observations']:
                src=final_sources.get(tuple(r['source_key']),{})
                r['r1_reliable_current_dense']=bool(r['path']=='dense' and src.get('reliable',False))
                r['r1_variance']=np.asarray(src['variance']).tolist() if 'variance' in src else None
            _,final=allocate_unique_support(self.sameview_audit['observations'])
            audit['final_remaining_collision_points']=final['collision_points']+final['ambiguous_deferred_collision_points']
            if self.ownership_audit_only:
                audit['probe_and_second_pass_observations_exact']=input_signature([{k:v for k,v in z.items() if k!='dense_trace'} for z in initial['observations']])==input_signature(self.sameview_audit['observations'])
                assert audit['probe_and_second_pass_observations_exact']
            audit['final_current_sources']=[dict(source_key=z['source_key'],category=z['category'],
                actual_retained_points=z['actual_retained_points'],selected_point_indices=z['selected_point_indices'],
                point_provenance_status=z['point_provenance_status']) for z in self.sameview_audit['observations']]
        else:
            audit['final_remaining_collision_points']=audit['ambiguous_deferred_collision_points']
            audit['final_current_sources']=[dict(source_key=z['source_key'],category=z['category'],
                actual_retained_points=z['actual_retained_points'],selected_point_indices=z['selected_point_indices'],
                point_provenance_status=z['point_provenance_status']) for z in initial['observations']]
        self.unique_support_audit=audit
        return fg,bg

    def update_group(self,*args,**kwargs):
        output=super().update_group(*args,**kwargs)
        self.audit['unique_support']=self.unique_support_audit
        return output


class CausalPCDPreservationMixin:
    def observation_policy(self,frame):
        original=super().observation_policy(frame)
        camera=next(c for c,f in self.group_frames if f is frame)
        def select(depth,mask,gobs,index):
            self._ownership_current_key=(camera,self.capture_times[camera],int(gobs['_audit_indices'][index]))
            return original(depth,mask,gobs,index)
        return select

    def extract_detections(self,frame):
        cache={};old_create=slam_utils.create_object_pcd
        def exact_create(*args,**kwargs):
            key=self._ownership_current_key
            if key in cache:
                pcd,before,after=cache[key]
                if key not in self._unique_allowed:
                    np.random.set_state(after);return copy.deepcopy(pcd)
                np.random.set_state(before)
                result=old_create(*args,**kwargs)
                np.random.set_state(after)
                return result
            before=np.random.get_state();result=old_create(*args,**kwargs)
            cache[key]=(copy.deepcopy(result),before,np.random.get_state())
            return result
        slam_utils.create_object_pcd=exact_create
        try:output=super().extract_detections(frame)
        finally:slam_utils.create_object_pcd=old_create
        np.random.set_state(self._ownership_after_probe_rng)
        audit=self.unique_support_audit;affected={tuple(d['source_key']) for d in audit['source_decisions'] if d['lost_indices']}
        before={tuple(z['source_key']):z for z in audit['initial_trace_observations']}
        exact=0
        for z in self.sameview_audit['observations']:
            key=tuple(z['source_key'])
            if key not in affected and key in before:
                assert z['actual_retained_world_points']==before[key]['actual_retained_world_points']
                assert z['selected_point_indices']==before[key]['selected_point_indices']
                exact+=1
        audit['unchanged_source_geometry_exact']=exact
        audit['same_rng_after_probe_and_final']=True
        return output


class ReliableUniqueSupportCanonicalMapper(CausalPCDPreservationMixin,ReliableUniqueSupportMixin,SameViewCanonicalMapper):
    def __init__(self,cfg,classes,*,audit_only=False):
        self.ownership_audit_only=audit_only
        # 同视身份合并关闭：本候选只有唯一支持归属，不叠L5身份清理。
        super().__init__(cfg,classes,audit_only=True)


class ReliableUniqueSupportOriginalFrontMapper(CausalPCDPreservationMixin,ReliableUniqueSupportMixin,SameViewOriginalFrontMapper):
    def __init__(self,cfg,classes):
        super().__init__(cfg,classes)
        self.sameview_audit_only=True
        self.ownership_audit_only=False
