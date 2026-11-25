"""唯一S候选：保留R1，严格图像连续性支持下回连稀疏位置假设。"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from conceptgraph.slam.dense_refinement import DenseProtectedPositionLayer, RefinedEvidenceMapper
from conceptgraph.slam.sparse_observations import cosine


@dataclass(frozen=True)
class ImageContinuitySettings:
    minimum_box_iou: float = .35
    minimum_iou_margin: float = .15
    appearance_minimum: float = .8
    maximum_image_gap_s: float = .75
    maximum_position_gap_m: float = 5.


def box_iou(a, b):
    a, b = np.asarray(a,dtype=float), np.asarray(b,dtype=float)
    intersect=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
    union=np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-intersect
    return float(intersect/union) if union>0 else 0.


class ImageContinuityLayer(DenseProtectedPositionLayer):
    def __init__(self):
        super().__init__()
        self.image_settings=ImageContinuitySettings()
        self.allowed_image_links=set()
        self.fallback_used=[]
        self._score_phase='none'

    def _new_state(self, observation, timestamp, node_id):
        state=super()._new_state(observation,timestamp,node_id)
        if 'image_box' in observation:
            state.update(image_box=list(observation['image_box']),image_timestamp_s=timestamp)
        return state

    def _update_state(self, state, observation, timestamp):
        if (id(observation),state['node_id']) in self.allowed_image_links:
            for record in self.fallback_used:
                if record['node_id']==state['node_id'] and np.max(np.abs(np.asarray(observation['center'])-record['observation_center']))<1e-8:
                    record['selected_update']=True
        super()._update_state(state,observation,timestamp)
        if 'image_box' in observation:
            state.update(image_box=list(observation['image_box']),image_timestamp_s=timestamp)

    def _score(self, observation, state):
        base=super()._score(observation,state)
        if np.isfinite(base):
            return base
        if (id(observation),state['node_id']) not in self.allowed_image_links:
            return -np.inf
        distance=float(np.linalg.norm((np.asarray(observation['center'])-state['center'])[:2]))
        return -distance+.25*cosine(observation['feature'],state['feature'])

    def update(self, sparse, dense_predictions, dense_measurements, timestamp):
        self.allowed_image_links=set()
        self.fallback_used=[]
        settings=self.image_settings
        # Only sparse->sparse continuity; dense states and dense handoff have no image override.
        tracks=[t for t in self.tracks if 'image_box' in t
            and 0<timestamp-t.get('image_timestamp_s',-np.inf)<=settings.maximum_image_gap_s]
        iou=np.zeros((len(sparse),len(tracks)),dtype=float)
        for i,observation in enumerate(sparse):
            if 'image_box' not in observation:
                continue
            for j,state in enumerate(tracks):
                if observation['category']!=state['category']:
                    continue
                iou[i,j]=box_iou(observation['image_box'],state['image_box'])
        for i,observation in enumerate(sparse):
            if not len(tracks):
                continue
            j=int(np.argmax(iou[i]))
            state=tracks[j]
            overlap=float(iou[i,j])
            row_other=max((float(v) for k,v in enumerate(iou[i]) if k!=j),default=0.)
            col_other=max((float(v) for k,v in enumerate(iou[:,j]) if k!=i),default=0.)
            if overlap<settings.minimum_box_iou or overlap-row_other<settings.minimum_iou_margin or overlap-col_other<settings.minimum_iou_margin:
                continue
            if int(np.argmax(iou[:,j]))!=i:
                continue
            distance=float(np.linalg.norm((np.asarray(observation['center'])-state['center'])[:2]))
            similarity=cosine(observation['feature'],state['feature'])
            if not (self.settings.association_distance_m<distance<=settings.maximum_position_gap_m
                    and similarity>=settings.appearance_minimum):
                continue
            self.allowed_image_links.add((id(observation),state['node_id']))
            self.fallback_used.append(dict(node_id=state['node_id'],category=observation['category'],
                image_iou=overlap,row_margin=overlap-row_other,column_margin=overlap-col_other,
                position_distance_m=distance,appearance_cosine=similarity,
                image_gap_s=timestamp-state['image_timestamp_s'],
                observation_center=np.asarray(observation['center']).tolist(),state_center=state['center'].tolist(),
                selected_update=False))
        output=super().update(sparse,dense_predictions,dense_measurements,timestamp)
        return output


class ImageContinuityR1(RefinedEvidenceMapper):
    def __init__(self,cfg,classes):
        super().__init__(cfg,classes,condition='R1')
        self.position_layer=ImageContinuityLayer()

    def sparse_callback(self,frame):
        original=super().sparse_callback(frame)

        def save(depth,mask,gobs,index,evidence):
            before=len(self.sparse_measurements)
            original(depth,mask,gobs,index,evidence)
            if len(self.sparse_measurements)>before:
                self.sparse_measurements[-1]['image_box']=np.asarray(gobs['xyxy'][index],dtype=float).tolist()
        return save

    def settings_record(self):
        return dict(image_continuity=asdict(self.position_layer.image_settings),
            sparse=self.position_layer.settings.to_dict(),dense=self.refinement_settings.to_dict())
