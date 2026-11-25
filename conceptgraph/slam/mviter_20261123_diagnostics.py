"""只附加因果观测来源，不改变上轮三视角关联和位置计算。"""
from __future__ import annotations

import numpy as np

from conceptgraph.slam.dense_refinement import DenseProtectedPositionLayer
from conceptgraph.slam.multiview_20261123_fusion import GroupedR1Mapper
from conceptgraph.slam.sparse_observations import measured_points


class SourceAuditedPositionLayer(DenseProtectedPositionLayer):
    def __init__(self, mapper):
        super().__init__()
        self.mapper = mapper
        self.events = []

    def _record(self, state, observation, action):
        sources = state.setdefault('_source_cameras', [])
        camera = observation.get('_camera', observation.get('camera'))
        if camera:
            sources.append(camera)
        state['_last_world_points'] = np.asarray(observation.get('_world_points', [])).reshape(-1, 3)
        state['_last_camera'] = camera
        if observation.get('_observation_key'):
            state.setdefault('_observation_keys', []).append(observation['_observation_key'])
        self.events.append(dict(action=action, node_id=state['node_id'], camera=camera,
            category=state['category'], hits=state['hits'], total_points=state['total_points'],
            observation_center=np.asarray(observation['center']).tolist(), corrected_center=state['center'].tolist(),
            observation_points=observation['points'], timestamp_s=state['timestamp']))

    def _new_state(self, observation, timestamp, node_id):
        state = super()._new_state(observation, timestamp, node_id)
        self._record(state, observation, 'new')
        return state

    def _update_state(self, state, observation, timestamp):
        super()._update_state(state, observation, timestamp)
        self._record(state, observation, 'update')

    def update(self, sparse, dense_predictions, dense_measurements, timestamp):
        self.events = []
        objects = {o['stable_object_id']: o for o in self.mapper.objects}
        for node_id, observation in dense_measurements.items():
            source = self.mapper.measurement_sources[objects[node_id]['mask_idx'][-1]]
            observation['_camera'] = source['camera']
            observation['_world_points'] = source['world_points']
            observation['_observation_key'] = source['_observation_key']
        return super().update(sparse, dense_predictions, dense_measurements, timestamp)


class SourceAuditedMapper(GroupedR1Mapper):
    def __init__(self, cfg, classes, *, deduplication):
        super().__init__(cfg, classes, deduplication=deduplication)
        self.position_layer = SourceAuditedPositionLayer(self)

    def sparse_callback(self, frame):
        original = super().sparse_callback(frame)
        camera = next(c for c, f in self.group_frames if f is frame)

        def save(depth, mask, gobs, index, evidence):
            before = len(self.sparse_measurements)
            original(depth, mask, gobs, index, evidence)
            if len(self.sparse_measurements) > before:
                self.sparse_measurements[-1].update(_world_points=measured_points(depth, mask, frame.cam_K, frame.camera_pose),
                    _observation_key=[camera, self.capture_times[camera], int(gobs['_audit_indices'][index])])
        return save

    def extract_detections(self, frame):
        fg, bg = super().extract_detections(frame)
        accepted = [a for a in self.frame_audit if a['stage'] == 'geometry' and a['reason'] == 'accepted']
        for i, source in self.measurement_sources.items():
            observed = accepted[i]
            assert observed['camera'] == source['camera']
            source['_observation_key'] = [source['camera'], source['camera_timestamp_s'], observed['detection_index']]
        return fg, bg

    def evidence_for_prediction(self, prediction):
        node_id = prediction['node_id']
        state = next((s for s in [*self.position_layer.tracks, *self.position_layer.dense_states.values()]
                      if self.position_layer.aliases.get(s['node_id'], s['node_id']) == node_id), None)
        objects = [o for o in self.objects if self.position_layer.aliases.get(o['stable_object_id'], o['stable_object_id']) == node_id]
        points = np.asarray(objects[0]['pcd'].points) if objects else (
            np.asarray(state.get('_last_world_points', [])).reshape(-1, 3) if state else np.empty((0, 3)))
        feature = state['feature'] if state else (objects[0]['clip_ft'].detach().cpu().numpy() if objects else np.empty(0))
        return dict(source_cameras=list(state.get('_source_cameras', [])) if state else [],
            last_camera=state.get('_last_camera') if state else None,
            state_timestamp_s=state['timestamp'] if state else None,
            world_points=points.tolist(), feature=np.asarray(feature).tolist(),
            observation_keys=list(state.get('_observation_keys', [])) if state else [],
            source_object_ids=[o['stable_object_id'] for o in objects])

    def internal_snapshot(self):
        layer = self.position_layer
        return dict(position_events=layer.events, aliases=dict(layer.aliases),
            objects=[dict(node_id=o['stable_object_id'], center=np.asarray(o['bbox'].get_center()).tolist(),
                category=o['class_name'][-1], image_idx=o['image_idx'], mask_idx=o['mask_idx'],
                n_points=o['n_points'], confidence=[float(c) for c in o['conf']], motion_state=o.get('motion_state'),
                is_short_term=o.get('is_short_term'), timestamp_s=o.get('timestamp_s'),
                last_seen_frame=o.get('last_seen_frame')) for o in self.objects],
            states=[dict(node_id=s['node_id'], category=s['category'], center=s['center'].tolist(),
                timestamp_s=s['timestamp'], hits=s['hits'], total_points=s['total_points'],
                cameras=s.get('_source_cameras', [])) for s in [*layer.tracks, *layer.dense_states.values()]])
