"""公平消融：原A单前视，只加冻结L3当前物理假设归并，不加生命周期/位置层。"""
from __future__ import annotations

import numpy as np

from conceptgraph.slam.mviter_20261123_diagnostics import SourceAuditedMapper
from conceptgraph.slam.mvrepair_lifecycle_20261123_crossclass import PhysicalHypothesisSettings,deduplicate_competing_vehicle_hypotheses
from conceptgraph.slam.slam_classes import DetectionList


class FrontPhysicalOnlyMapper(SourceAuditedMapper):
    def __init__(self,cfg,classes,*,audit_only=False):
        super().__init__(cfg,classes,deduplication=False)
        self.audit_only=audit_only
        self.physical_settings=PhysicalHypothesisSettings()
        self.physical_audit={}

    def update_group(self,frames):
        if [c for c,_ in frames]!=['CAM_FRONT']:raise ValueError('该消融只允许原A前视')
        return super().update_group(frames)

    def extract_detections(self,frame):
        raw={}
        for camera,f in self.group_frames:
            d=f.detections
            for i,cid in enumerate(d['class_id']):
                if d['classes'][int(cid)] in self.physical_settings.allowed_family:
                    raw[camera,i]=dict(box=np.asarray(d['xyxy'][i]).copy(),mask=np.asarray(d['mask'][i],bool).copy())
        fg,bg=super().extract_detections(frame)
        records=[]
        for i,obj in enumerate(fg):
            source=self.measurement_sources[obj['mask_idx'][-1]];key=source['_observation_key']
            if source['category'] not in self.physical_settings.allowed_family:continue
            records.append(dict(source_key=key,category=source['category'],center=source['center'],feature=source['feature'],
                real_points=source['world_points'],confidence=source['confidence'],path='dense',path_index=i,**raw[key[0],key[2]]))
        for i,observation in enumerate(self.sparse_measurements):
            key=observation['_observation_key']
            if observation['category'] not in self.physical_settings.allowed_family:continue
            records.append(dict(source_key=key,category=observation['category'],center=observation['center'],feature=observation['feature'],
                real_points=observation['_world_points'],confidence=observation['confidence'],path='sparse',path_index=i,**raw[key[0],key[2]]))
        kept,audit=deduplicate_competing_vehicle_hypotheses(records,self.physical_settings)
        audit['audit_only']=self.audit_only
        audit['observation_sources']=[dict(source_key=o['source_key'],category=o['category'],confidence=o['confidence'],
            path=o['path'],path_index=o['path_index'],points=len(o['real_points'])) for o in records]
        self.physical_audit=audit
        if self.audit_only:return fg,bg
        dense_removed={o['path_index'] for i,o in enumerate(records) if i not in kept and o['path']=='dense'}
        sparse_removed={o['path_index'] for i,o in enumerate(records) if i not in kept and o['path']=='sparse'}
        self.sparse_measurements=[o for i,o in enumerate(self.sparse_measurements) if i not in sparse_removed]
        return DetectionList([o for i,o in enumerate(fg) if i not in dense_removed]),bg
