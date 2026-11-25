"""L3独立候选：同曝光中车辆互斥子类的高重叠物理实体假设。

只用输入框/掩码、CLIP与真实选后点；保留所有类别候选，不读GT。
不是跨类别全局距离抑制；行人/骑车人不在可合并家族。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree

from conceptgraph.slam.mvrepair_lifecycle_20261123_mapper import LifecycleRetiringMapper
from conceptgraph.slam.slam_classes import DetectionList
from conceptgraph.slam.sparse_observations import cosine


@dataclass(frozen=True)
class PhysicalHypothesisSettings:
    allowed_family: tuple[str, ...] = ('car', 'truck', 'bus')
    minimum_box_iou: float = .9
    minimum_mask_iou: float = .9
    minimum_feature_cosine: float = .95
    minimum_shared_points: int = 2
    minimum_shared_fraction: float = .8
    shared_radius_m: float = .2
    maximum_xy_distance_m: float = .5
    maximum_z_distance_m: float = .5


def competing_hypothesis_witness(a, b, settings=None):
    s=settings or PhysicalHypothesisSettings()
    result=dict(source_key_a=a['source_key'],source_key_b=b['source_key'],
        category_a=a['category'],category_b=b['category'],accepted=False,reason='different_source_or_nonexclusive_family')
    if (a['category']==b['category'] or a['category'] not in s.allowed_family or b['category'] not in s.allowed_family
            or a['source_key'][:2]!=b['source_key'][:2]):return result
    x,y=np.asarray(a['box'],float),np.asarray(b['box'],float)
    aa,bb=np.asarray(a['mask'],bool),np.asarray(b['mask'],bool)
    pa,pb=np.asarray(a['real_points'],float).reshape(-1,3),np.asarray(b['real_points'],float).reshape(-1,3)
    if (x.shape!=(4,) or y.shape!=(4,) or aa.shape!=bb.shape or not np.isfinite([*x,*y]).all()
            or min(len(pa),len(pb))<s.minimum_shared_points or not np.isfinite(pa).all() or not np.isfinite(pb).all()):
        result['reason']='invalid_or_insufficient_real_evidence';return result
    inter=float(np.prod(np.maximum(np.minimum(x[2:],y[2:])-np.maximum(x[:2],y[:2]),0)))
    union=float(np.prod(x[2:]-x[:2])+np.prod(y[2:]-y[:2])-inter)
    mask_union=int((aa|bb).sum())
    box_iou=inter/union if union>0 else 0.;mask_iou=float((aa&bb).sum()/mask_union) if mask_union else 0.
    clip=cosine(a['feature'],b['feature']);delta=np.asarray(a['center'])-b['center']
    result.update(box_iou=box_iou,mask_iou=mask_iou,feature_cosine=clip,
        xy_distance_m=float(np.linalg.norm(delta[:2])),z_distance_m=float(abs(delta[2])))
    if not np.isfinite([clip,*delta]).all():result['reason']='nonfinite_evidence';return result
    if box_iou<s.minimum_box_iou or mask_iou<s.minimum_mask_iou:result['reason']='pixel_overlap_gate';return result
    if clip<s.minimum_feature_cosine:result['reason']='appearance_gate';return result
    if result['xy_distance_m']>s.maximum_xy_distance_m or result['z_distance_m']>s.maximum_z_distance_m:
        result['reason']='selected_geometry_gate';return result
    small,large=(pa,pb) if len(pa)<=len(pb) else(pb,pa)
    distances=cKDTree(large).query(small)[0]
    count=int(np.sum(distances<=s.shared_radius_m))
    result.update(shared_points=count,smaller_points=len(small),shared_fraction=count/len(small),maximum_smaller_nn_m=float(max(distances)))
    result['accepted']=count>=s.minimum_shared_points and count/len(small)>=s.minimum_shared_fraction
    result['reason']='same_exposure_competing_vehicle_hypotheses' if result['accepted'] else 'real_support_gate'
    return result


def deduplicate_competing_vehicle_hypotheses(observations,settings=None):
    """高conf代表，完整链接簇；真实类候选保留，绝不把不同类当两次观测。"""
    settings=settings or PhysicalHypothesisSettings()
    pairs={(j,i):competing_hypothesis_witness(observations[j],a,settings)
        for i,a in enumerate(observations) for j in range(i)}
    clusters=[]
    for i in sorted(range(len(observations)),key=lambda i:(-observations[i]['confidence'],i)):
        compatible=[c for c in clusters if all(pairs[min(i,j),max(i,j)]['accepted'] for j in c)]
        if compatible:compatible[0].append(i)
        else:clusters.append([i])
    return [c[0] for c in clusters],dict(settings=asdict(settings),
        pairs=list(pairs.values()),clusters=[dict(representative=c[0],members=c,
            chosen_category=observations[c[0]]['category'],class_hypotheses=[dict(category=observations[j]['category'],
                confidence=observations[j]['confidence'],source_key=observations[j]['source_key']) for j in c]) for c in clusters],
        input_observations=len(observations),output_observations=len(clusters),removed=len(observations)-len(clusters))


class CrossClassPhysicalMapper(LifecycleRetiringMapper):
    def __init__(self,cfg,classes,*,audit_only=False):
        super().__init__(cfg,classes)
        self.audit_only=audit_only
        self.physical_settings=PhysicalHypothesisSettings()
        self.physical_audit={}

    def extract_detections(self,frame):
        # 上游会原地过滤/改写detections；保留原同曝光像素掩码引用供物理支持核查。
        raw={}
        for camera,f in self.group_frames:
            d=f.detections
            for i,cid in enumerate(d['class_id']):
                category=d['classes'][int(cid)]
                if category in self.physical_settings.allowed_family:
                    raw[camera,i]=dict(box=np.asarray(d['xyxy'][i]).copy(),mask=np.asarray(d['mask'][i],bool).copy())
        fg,bg=super().extract_detections(frame)
        records=[]
        for i,obj in enumerate(fg):
            source=self.measurement_sources[obj['mask_idx'][-1]]
            key=source['_observation_key']
            if source['category'] not in self.physical_settings.allowed_family:continue
            records.append(dict(source_key=key,category=source['category'],center=source['center'],
                feature=source['feature'],real_points=source['world_points'],confidence=source['confidence'],
                path='dense',path_index=i,**raw[key[0],key[2]]))
        for i,observation in enumerate(self.sparse_measurements):
            key=observation['_observation_key']
            if observation['category'] not in self.physical_settings.allowed_family:continue
            records.append(dict(source_key=key,category=observation['category'],center=observation['center'],
                feature=observation['feature'],real_points=observation['_world_points'],confidence=observation['confidence'],
                path='sparse',path_index=i,**raw[key[0],key[2]]))
        kept,audit=deduplicate_competing_vehicle_hypotheses(records,self.physical_settings)
        audit['audit_only']=self.audit_only
        audit['observation_sources']=[dict(source_key=o['source_key'],category=o['category'],confidence=o['confidence'],
            path=o['path'],path_index=o['path_index'],points=len(o['real_points'])) for o in records]
        self.physical_audit=audit
        if self.audit_only:return fg,bg
        removed=[o for i,o in enumerate(records) if i not in kept]
        dense_removed={o['path_index'] for o in removed if o['path']=='dense'}
        sparse_removed={o['path_index'] for o in removed if o['path']=='sparse'}
        self.sparse_measurements=[o for i,o in enumerate(self.sparse_measurements) if i not in sparse_removed]
        return DetectionList([o for i,o in enumerate(fg) if i not in dense_removed]),bg
