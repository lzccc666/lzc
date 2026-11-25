"""R1观测溯源旁路；新增记录不参与关联/几何，也不补造导航授权。"""
from __future__ import annotations
import copy
import numpy as np
from conceptgraph.completion.motion import robust_xy_speed
from conceptgraph.slam.dense_refinement import DenseProtectedPositionLayer,RefinedEvidenceMapper


class RecordingPositionLayer(DenseProtectedPositionLayer):
    def _record(self,state,observation,timestamp):
        evidence=observation.get('_navigation_raw_provenance')
        # 原层的dense测量没有完整mask来源时，不伪装成已可追溯观测。
        if evidence is not None:
            state.setdefault('_navigation_raw_trace',[]).append(copy.deepcopy(evidence))
            state['_navigation_raw_trace']=state['_navigation_raw_trace'][-8:]

    def _new_state(self,observation,timestamp,node_id):
        state=super()._new_state(observation,timestamp,node_id)
        self._record(state,observation,timestamp)
        return state

    def _update_state(self,state,observation,timestamp):
        super()._update_state(state,observation,timestamp)
        self._record(state,observation,timestamp)


class RecordingR1Mapper(RefinedEvidenceMapper):
    def __init__(self,cfg,classes):
        super().__init__(cfg,classes,condition='R1')
        self.position_layer=RecordingPositionLayer()

    def sparse_callback(self,frame):
        original=super().sparse_callback(frame)
        def save(depth,mask,gobs,index,evidence):
            count=len(self.sparse_measurements)
            original(depth,mask,gobs,index,evidence)
            if len(self.sparse_measurements)>count:
                observation=self.sparse_measurements[-1]
                observation['_navigation_raw_provenance']=dict(frame_idx=int(frame.frame_idx),
                    timestamp_s=float(frame.timestamp_s),raw_measured_center=np.asarray(observation['center']).tolist(),
                    measurement_points=int(observation['points']),filtered_detection_index=int(index),
                    category=observation['category'],depth_evidence=copy.deepcopy(evidence),
                    source='original_sparse_callback_actual_measured_points',future_used=False,
                    filtered_position_used_as_measurement=False)
        return save

    def navigation_trace(self,current_timestamp):
        result=[]
        for track in self.position_layer.tracks:
            trace=copy.deepcopy(track.get('_navigation_raw_trace',[]))
            times=[r['timestamp_s'] for r in trace]
            distinct=len(set(times))
            causal=bool(trace) and all(t<=current_timestamp for t in times) and all(b>a for a,b in zip(times,times[1:]))
            velocity=(robust_xy_speed([r['raw_measured_center'] for r in trace],times)
                if len(times)>=2 and causal else None)
            motion=('static' if velocity<=.5 else 'dynamic') if velocity is not None and distinct>=3 else 'unknown'
            latest=trace[-1]['depth_evidence'] if trace else {}
            current_measured=(bool(trace) and trace[-1]['timestamp_s']==current_timestamp)
            current_depth_passed=bool(current_measured and latest.get('source')=='current_projected_lidar'
                and latest.get('units')=='m' and latest.get('retained_pixels',0)>=5 and latest.get('support_fraction',0)>=.4)
            result.append(dict(node_id=track['node_id'],category=track['category'],independent_observations=track['hits'],
                raw_trace=trace,causal=causal,reconstructed_raw_speed_mps=velocity,
                diagnostic_motion_from_raw_observations=motion,current_measured=current_measured,
                current_depth_passed_original_gate=current_depth_passed,
                complete_original_navigation_evidence=bool(causal and motion=='static' and current_depth_passed),
                original_predictions_untouched=True))
        return result
