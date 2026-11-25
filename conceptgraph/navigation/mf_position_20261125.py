"""因果原测量账本与多帧语义位置候选；不授予车辆几何安全权限。"""
from __future__ import annotations

import copy
import hashlib
import math
import numpy as np

from conceptgraph.completion.association import gated_assignment
from conceptgraph.completion.evidence import target_evidence_decision
from conceptgraph.completion.motion import robust_xy_speed
from conceptgraph.completion.selection_map_transfer import EVIDENCE_POLICY, nodes_from_frozen_predictions
from conceptgraph.navigation.semantic import parse_navigation_query, rank_candidate_targets
from conceptgraph.scenegraph.relations import build_relations
from conceptgraph.slam.dense_refinement import DenseProtectedPositionLayer, RefinedEvidenceMapper
from conceptgraph.slam.sparse_observations import measured_points, measurement_variance

POLICY = dict(minimum_distinct_measurements=3, minimum_span_s=1., minimum_current_support=.4,
    maximum_raw_speed_mps=.5, maximum_heuristic_xy_variance_m2=4., history_length=8,
    require_current_update=True, require_unambiguous_lineage=True, allow_cross_id_history_merge=False,
    candidate_scope='semantic_position_only', vehicle_safety_authorized=False,
    variance_interpretation='uncalibrated_heuristic_not_probability', seed=17)


class LedgerPositionLayer(DenseProtectedPositionLayer):
    """原SparsePositionLayer.update顺序逐步镜像，仅新增旁路账本字段。"""
    def __init__(self):
        super().__init__(); self.events=[]; self._serial=0

    def _record(self,state,observation,timestamp,new=False):
        if new:
            state['_ledger_state_uid']=self._serial; self._serial+=1
        raw=observation.get('_ledger_raw')
        assignment=observation.get('_ledger_assignment',{})
        if raw is not None:
            record=dict(copy.deepcopy(raw),assignment=copy.deepcopy(assignment))
            state.setdefault('_ledger_trace',[]).append(record)
            state['_ledger_trace']=state['_ledger_trace'][-POLICY['history_length']:]
        self.events.append(dict(kind='state_created' if new else 'state_updated',
            state_uid=state['_ledger_state_uid'],node_id=state['node_id'],
            observation_key=None if raw is None else raw['key'],assignment=copy.deepcopy(assignment)))

    def _new_state(self,observation,timestamp,node_id):
        state=super()._new_state(observation,timestamp,node_id)
        self._record(state,observation,timestamp,new=True);return state

    def _update_state(self,state,observation,timestamp):
        super()._update_state(state,observation,timestamp)
        self._record(state,observation,timestamp)

    @staticmethod
    def _assignment(scores,i,j,stage):
        return dict(stage=stage,row_feasible=int(np.isfinite(scores[i]).sum()),
            column_feasible=int(np.isfinite(scores[:,j]).sum()),
            unambiguous=bool(np.isfinite(scores[i]).sum()==1 and np.isfinite(scores[:,j]).sum()==1))

    def update(self,sparse,dense_predictions,dense_measurements,timestamp):
        self.events=[]
        if not np.isfinite(timestamp) or (self.timestamp is not None and timestamp<=self.timestamp):
            raise ValueError('位置层时间必须严格递增')
        for observation in [*sparse,*dense_measurements.values()]:
            if (np.asarray(observation['center']).shape!=(3,) or not np.isfinite(observation['center']).all()
                or not np.isfinite(observation['variance']).all() or np.any(np.asarray(observation['variance'])<=0)):
                raise ValueError('测量或协方差非法')
        self.timestamp=timestamp
        for t in self.tracks:
            if timestamp-t['timestamp']>self.settings.ttl_s:
                self.events.append(dict(kind='sparse_expired',node_id=t['node_id'],state_uid=t['_ledger_state_uid']))
        self.tracks=[t for t in self.tracks if timestamp-t['timestamp']<=self.settings.ttl_s]
        live_ids={p['node_id'] for p in dense_predictions}
        for k,v in self.dense_states.items():
            if k not in live_ids and timestamp-v['timestamp']>self.settings.ttl_s:
                self.events.append(dict(kind='dense_expired',node_id=k,state_uid=v['_ledger_state_uid']))
        self.dense_states={k:v for k,v in self.dense_states.items()
            if k in live_ids or timestamp-v['timestamp']<=self.settings.ttl_s}
        for node_id,observation in dense_measurements.items():
            observation['_ledger_assignment']=dict(stage='original_dense_identity',unambiguous=False,
                reason='not_used_for_new_sparse_confirmation')
            if node_id in self.dense_states:self._update_state(self.dense_states[node_id],observation,timestamp)
            else:self.dense_states[node_id]=self._new_state(observation,timestamp,node_id)
        dense_keys=[p['node_id'] for p in dense_predictions if p['node_id'] in self.dense_states]
        scores=np.array([[self._score(o,self.dense_states[k]) for k in dense_keys]
            for o in sparse],dtype=float).reshape(len(sparse),len(dense_keys))
        assigned=gated_assignment(scores)
        for i,j in assigned.items():
            state=self.dense_states[dense_keys[j]]
            sparse[i]['_ledger_assignment']=self._assignment(scores,i,j,'sparse_to_dense')
            if state['timestamp']<timestamp:self._update_state(state,sparse[i],timestamp)
            else:self.events.append(dict(kind='sparse_to_dense_not_applied_already_current',node_id=dense_keys[j],
                observation_key=sparse[i]['_ledger_raw']['key'],assignment=sparse[i]['_ledger_assignment']))
        remaining=[o for i,o in enumerate(sparse) if i not in assigned]
        scores=np.array([[self._score(o,t) for t in self.tracks] for o in remaining],dtype=float).reshape(len(remaining),len(self.tracks))
        assigned=gated_assignment(scores)
        for i,observation in enumerate(remaining):
            if i in assigned:
                observation['_ledger_assignment']=self._assignment(scores,i,assigned[i],'sparse_to_sparse')
                self._update_state(self.tracks[assigned[i]],observation,timestamp)
            else:
                count=int(np.isfinite(scores[i]).sum())
                observation['_ledger_assignment']=dict(stage='new_sparse',row_feasible=count,unambiguous=count==0)
                self.tracks.append(self._new_state(observation,timestamp,f'sparse-{self.next_id:08d}'));self.next_id+=1
        output=copy.deepcopy(dense_predictions)
        for obj in output:
            state=self.dense_states.get(obj['node_id'])
            if state is not None and timestamp-state['timestamp']<=self.settings.maximum_prediction_age_s:
                obj.update(center=state['center'].tolist(),position_covariance_diag=state['variance'].tolist(),
                    position_source='quality_weighted_real_measurements',prediction_age_s=timestamp-state['timestamp'])
        used=set()
        for obj in output:
            observation=dense_measurements.get(obj['node_id'])
            if observation is None:continue
            options=[(self._score(observation,t),j) for j,t in enumerate(self.tracks) if j not in used]
            options=[(s,j) for s,j in options if np.isfinite(s)]
            if options:
                _,j=max(options,key=lambda v:(v[0],-v[1]));track=self.tracks[j]
                previous=self.aliases.get(obj['node_id'])
                if track['hits']>=self.settings.minimum_frames:self.aliases.setdefault(obj['node_id'],track['node_id'])
                self.events.append(dict(kind='original_sparse_dense_handoff',dense_node_id=obj['node_id'],
                    sparse_node_id=track['node_id'],sparse_state_uid=track['_ledger_state_uid'],
                    previous_alias=previous,resulting_alias=self.aliases.get(obj['node_id']),
                    alternative_feasible=len(options),history_merged=False,
                    retiring_raw_keys=[r['key'] for r in track.get('_ledger_trace',[])]))
                used.add(j)
        self.tracks=[t for j,t in enumerate(self.tracks) if j not in used]
        for obj in output:obj['node_id']=self.aliases.get(obj['node_id'],obj['node_id'])
        for track in self.tracks:
            frames_needed=self.settings.minimum_frames if track['confidence']>=self.settings.strong_confidence else self.settings.weak_minimum_frames
            if (track['hits']<frames_needed or track['total_points']<self.settings.minimum_total_points
                or timestamp-track['timestamp']>self.settings.maximum_prediction_age_s):continue
            if any(obj['category']==track['category'] and np.linalg.norm((np.asarray(obj['center'])-track['center'])[:2])<=self.settings.association_distance_m for obj in output):continue
            output.append(dict(node_id=track['node_id'],category=track['category'],center=track['center'].tolist(),
                num_detections=track['hits'],timestamp_s=track['timestamp'],prediction_age_s=timestamp-track['timestamp'],
                position_covariance_diag=track['variance'].tolist(),position_source='confirmed_sparse_lidar',
                measured_points_total=track['total_points'],shape_available=False))
        return output


def confirm_position(track,current_timestamp):
    trace=copy.deepcopy(track.get('_ledger_trace',[]));reasons=[]
    times=[r['timestamp_s'] for r in trace];frames=[r['frame_idx'] for r in trace]
    keys=[r['key'] for r in trace]
    if len(trace)<3 or len(set(times))!=len(trace) or len(set(frames))!=len(trace) or len(set(keys))!=len(trace):reasons.append('insufficient_distinct_raw_measurements')
    if not times or any(b<=a for a,b in zip(times,times[1:])) or any(t>current_timestamp for t in times):reasons.append('noncausal_or_unordered_raw_history')
    span=times[-1]-times[0] if times else 0.
    if span<POLICY['minimum_span_s']-1e-8:reasons.append('raw_span_below_one_second')
    latest=trace[-1] if trace else {};depth=latest.get('depth_evidence',{})
    if not times or abs(times[-1]-current_timestamp)>1e-6:reasons.append('no_current_real_update')
    if (depth.get('source')!='current_projected_lidar' or depth.get('units')!='m'
        or depth.get('retained_pixels',0)<1 or depth.get('support_fraction',0)<POLICY['minimum_current_support']):reasons.append('current_raw_support_invalid')
    if any(not r.get('assignment',{}).get('unambiguous',False) for r in trace):reasons.append('ambiguous_assignment_in_raw_history')
    speed=None
    if len(times)>=2 and not any(b<=a for a,b in zip(times,times[1:])):
        speed=robust_xy_speed([r['raw_measured_center'] for r in trace],times)
    if speed is None or speed>POLICY['maximum_raw_speed_mps']:reasons.append('raw_motion_not_confirmed_static')
    variance=np.asarray(track['variance'])
    if not np.isfinite(variance).all() or np.any(variance<=0) or max(variance[:2])>POLICY['maximum_heuristic_xy_variance_m2']:reasons.append('heuristic_position_variance_too_large')
    return dict(node_id=track['node_id'],state_uid=track['_ledger_state_uid'],raw_trace=trace,
        confirmed=not reasons,reasons=reasons,raw_speed_mps=speed,span_s=span,
        variance_diag=variance.tolist(),current_timestamp_s=current_timestamp,
        semantic_position_candidate=not reasons,vehicle_safety_authorized=False,
        original_five_current_point_gate_passed=bool(depth.get('retained_pixels',0)>=5),
        heuristic_uncertainty_not_probability=True,history_cross_id_merged=False)


class LedgerR1Mapper(RefinedEvidenceMapper):
    def __init__(self,cfg,classes,*,scene):
        super().__init__(cfg,classes,condition='R1');self.scene=scene
        self.position_layer=LedgerPositionLayer();self.raw_observations={}

    def observation_policy(self,frame):
        original=super().observation_policy(frame)
        def select(depth,mask,gobs,index):
            selected,evidence=original(depth,mask,gobs,index)
            points=measured_points(depth,selected,frame.cam_K,frame.camera_pose)
            key=f'{self.scene}:{frame.frame_idx}:{int(gobs.get("_audit_indices",np.arange(len(gobs["mask"])))[index])}:{frame.timestamp_s:.6f}'
            raw=dict(key=key,scene=self.scene,frame_idx=int(frame.frame_idx),timestamp_s=float(frame.timestamp_s),
                original_detection_index=int(gobs.get('_audit_indices',np.arange(len(gobs['mask'])))[index]),
                filtered_detection_index=int(index),category=gobs['classes'][gobs['class_id'][index]],
                raw_measured_center=np.median(points,axis=0).tolist() if len(points) else None,
                measured_points=len(points),depth_evidence=copy.deepcopy(evidence),
                selected_mask_sha256=hashlib.sha256(np.packbits(selected).tobytes()).hexdigest(),
                input_source='current_actual_depth_pixels',filtered_position_used_as_measurement=False)
            self.raw_observations[index]=raw
            return selected,evidence
        return select

    def extract_detections(self,frame):
        self.raw_observations={};return super().extract_detections(frame)

    def sparse_callback(self,frame):
        original=super().sparse_callback(frame)
        def save(depth,mask,gobs,index,evidence):
            count=len(self.sparse_measurements);original(depth,mask,gobs,index,evidence)
            if len(self.sparse_measurements)>count:
                observation=self.sparse_measurements[-1];raw=copy.deepcopy(self.raw_observations[index])
                if not np.array_equal(np.asarray(raw['raw_measured_center']),observation['center']):raise AssertionError('raw_measurement_mismatch')
                observation['_ledger_raw']=raw
        return save

    def position_predictions(self,predictions,frame):
        # 与原R1相同测量计算、顺序及视锥；只增加来源元数据。
        measurements={};ids={o['node_id'] for o in predictions}
        current={o['mask_idx'][-1]:o for o in self.last_observations};settings=self.refinement_settings
        for obj in self.objects:
            node_id=obj['stable_object_id']
            if node_id not in ids or obj['image_idx'][-1]!=frame.frame_idx:continue
            observed=current[obj['mask_idx'][-1]];center=np.asarray(obj['center_history'][-1],dtype=float)
            points=np.asarray(observed['pcd'].points);distance=float(np.linalg.norm(center-frame.camera_pose[:3,3]))
            variance=measurement_variance(points,range_m=distance)
            unique,counts=np.unique(np.asarray(obj['class_id'],dtype=int),return_counts=True)
            confidence,support=float(obj['conf'][-1]),int(obj['n_points'][-1])
            reliable=support>=settings.minimum_points and confidence>=settings.minimum_confidence and np.max(variance[:2])<=settings.maximum_xy_variance_m2
            measurements[node_id]=dict(center=center,category=self.classes[int(unique[np.argmax(counts)])],
                feature=observed['clip_ft'].detach().cpu().numpy(),points=support,confidence=confidence,variance=variance,
                reliable_current_dense=reliable,maximum_speed_m_s=settings.maximum_speed_m_s,
                _ledger_raw=copy.deepcopy(self.raw_observations[obj['mask_idx'][-1]]))
        output=self.position_layer.update(self.sparse_measurements,predictions,measurements,frame.timestamp_s)
        protected={self.position_layer.aliases.get(k,k) for k,v in measurements.items() if v['reliable_current_dense']}
        kept=[]
        for obj in output:
            if obj['node_id'] in protected:obj['position_source']='reliable_current_dense_lidar'
            camera=frame.camera_pose[:3,:3].T@(np.asarray(obj['center'])-frame.camera_pose[:3,3]);uv=frame.cam_K@camera
            if 1<camera[2]<65 and 0<=uv[0]/uv[2]<1600 and 0<=uv[1]/uv[2]<900:kept.append(obj)
        return kept

    def ledger(self,timestamp):
        return dict(raw_observations=list(self.raw_observations.values()),events=copy.deepcopy(self.position_layer.events),
            sparse_states=[confirm_position(t,timestamp) for t in self.position_layer.tracks],
            dense_states=[dict(node_id=k,state_uid=t['_ledger_state_uid'],raw_trace=copy.deepcopy(t.get('_ledger_trace',[])))
                for k,t in self.position_layer.dense_states.items()],aliases=copy.deepcopy(self.position_layer.aliases))


def select_position_candidates(query,predictions,pose,frame_index,confirmations):
    """保留全节点和原关系/分数，只把新增位置确认作为显式新候选证据。"""
    nodes=nodes_from_frozen_predictions(predictions);forward=pose[:2,2]
    edges=build_relations(nodes,dict(coordinate_frame='ego_heading',directional_mode='all_supported_axes'),
        frame_index=frame_index,ego_heading=forward).edges
    allowed={};new_ids=[]
    for i,node in nodes.items():
        obj=predictions[i]
        if obj.get('shape_available') is False:
            evidence=confirmations.get(obj['node_id']);ok=bool(evidence and evidence['confirmed'])
            allowed[i]=ok
            if ok:
                node.motion_state='static';node.is_short_term=False
                node.metadata['position_confirmation']=copy.deepcopy(evidence);new_ids.append(i)
        else:allowed[i]=target_evidence_decision(node,EVIDENCE_POLICY)[0]
    parsed=parse_navigation_query(query,8.);parsed.vehicle_heading_xy=forward
    ranked=rank_candidate_targets(parsed,nodes,edges,pose[:2,3],.55,.3,.25,
        relation_coordinate_frame='ego_heading',relation_coordinate_frame_fallback=None,evidence_policy=None)
    ranked=[c for c in ranked if allowed[c.node_id]]
    return ranked,dict(new_position_candidate_indices=new_ids,vehicle_safety_authorized=False,
        unknown_shape_stays_unknown=True,evidence_policy='explicit_multiframe_position_candidate')
