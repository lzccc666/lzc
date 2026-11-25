"""单一观测轨迹状态：统一稠密/稀疏ID，组内实测几何归并与前视保护。"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree

from conceptgraph.completion.association import gated_assignment
from conceptgraph.slam.dense_refinement import DenseProtectedPositionLayer
from conceptgraph.slam.multiview_20261123_fusion import in_view
from conceptgraph.slam.mviter_20261123_diagnostics import SourceAuditedMapper
from conceptgraph.slam.sparse_observations import cosine


@dataclass(frozen=True)
class UnifiedSettings:
    association_distance_m: float = 2.5
    appearance_minimum: float = .65
    shared_xy_m: float = 1.5
    shared_z_m: float = 1.5
    shared_radius_m: float = .2
    shared_fraction: float = .2
    shared_minimum_points: int = 2
    capture_gap_s: float = .075
    same_camera_box_iou: float = .5
    strong_confidence: float = .2
    sparse_minimum_frames: int = 2
    weak_minimum_frames: int = 3
    minimum_total_sparse_points: int = 4
    sparse_ttl_s: float = 1.5
    dense_ttl_s: float = 2.
    sparse_output_age_s: float = .75
    front_protection_s: float = .75
    motion_horizon_s: float = .75
    maximum_speed_m_s: float = 20.


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    area = np.maximum(np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]), 0).prod()
    union = np.maximum(a[2:]-a[:2], 0).prod()+np.maximum(b[2:]-b[:2], 0).prod()-area
    return float(area/union) if union else 0.


def shared_witness(a, b, settings):
    delta = np.asarray(a['center'])-np.asarray(b['center'])
    similarity = cosine(a['feature'], b['feature'])
    result = dict(key_a=a['source_key'], key_b=b['source_key'], cameras_a=a['cameras'], cameras_b=b['cameras'],
        xy_m=float(np.linalg.norm(delta[:2])), z_m=float(abs(delta[2])), cosine=similarity,
        capture_gap_s=abs(a['capture_timestamp_s']-b['capture_timestamp_s']), shared_points=0, shared_fraction=0., accepted=False)
    if a['category'] != b['category']:
        reason = 'class'
    elif result['xy_m'] > settings.shared_xy_m or result['z_m'] > settings.shared_z_m:
        reason = 'geometry_gate'
    elif result['capture_gap_s'] > settings.capture_gap_s:
        reason = 'exposure_gap'
    elif similarity < settings.appearance_minimum:
        reason = 'association_appearance_floor'
    elif a['cameras'] == b['cameras'] and box_iou(a['box'], b['box']) < settings.same_camera_box_iou:
        reason = 'same_camera_instance_separation'
    elif min(len(a['real_points']), len(b['real_points'])) < settings.shared_minimum_points:
        reason = 'insufficient_actual_points'
    else:
        # Full LiDAR identity key is always (token, point index), never bare index.
        ai, bi = set(map(tuple, a['lidar_point_keys'])), set(map(tuple, b['lidar_point_keys']))
        if ai and bi and ai & bi:
            count = len(ai & bi)
            fraction = count/min(len(ai), len(bi))
            source = 'exact_lidar_token_point_index'
        else:
            aa, bb = a['real_points'], b['real_points']
            small, large = (aa, bb) if len(aa) <= len(bb) else (bb, aa)
            count = int(np.count_nonzero(cKDTree(large).query(small)[0] <= settings.shared_radius_m))
            fraction = count/len(small)
            source = 'fresh_world_point_proximity_proxy'
        result.update(shared_points=count, shared_fraction=fraction, witness_source=source)
        result['accepted'] = count >= settings.shared_minimum_points and fraction >= settings.shared_fraction
        reason = 'shared_actual_geometry' if result['accepted'] else 'no_shared_actual_geometry'
    result['reason'] = reason
    return result


def group_observations(observations, settings):
    order = sorted(range(len(observations)), key=lambda i: (
        -int('CAM_FRONT' in observations[i]['cameras'] and observations[i]['reliable_current_dense']),
        -int(observations[i]['reliable_current_dense']), -int(observations[i]['dense']),
        -observations[i]['points'], -observations[i]['confidence'], i))
    pairs, clusters = {}, []
    for position, i in enumerate(order):
        for j in order[:position]:
            pairs[min(i, j), max(i, j)] = shared_witness(observations[j], observations[i], settings)
        options = [c for c in clusters if all(pairs[min(i, j), max(i, j)]['accepted'] for j in c)]
        if options:
            options[0].append(i)
        else:
            clusters.append([i])
    grouped = []
    for cluster in clusters:
        obj = dict(observations[cluster[0]])
        obj['representative_camera'] = observations[cluster[0]]['cameras'][0]
        obj['cameras'] = sorted({c for i in cluster for c in observations[i]['cameras']})
        obj['source_keys'] = [observations[i]['source_key'] for i in cluster]
        obj['cluster_members'] = cluster
        # Preserve all per-view witnesses, but count only the representative evidence.
        grouped.append(obj)
    return grouped, dict(pairs=list(pairs.values()), clusters=clusters,
                         input=len(observations), grouped=len(grouped), removed=len(observations)-len(grouped))


class UnifiedObservationExtractor(SourceAuditedMapper):
    def __init__(self, cfg, classes):
        super().__init__(cfg, classes, deduplication=False)
        self.projections = {}

    def sparse_callback(self, frame):
        original = super().sparse_callback(frame)

        def save(depth, mask, gobs, index, evidence):
            before = len(self.sparse_measurements)
            original(depth, mask, gobs, index, evidence)
            if len(self.sparse_measurements) > before:
                self.sparse_measurements[-1]['_image_box'] = np.asarray(gobs['xyxy'][index]).copy()
                self.sparse_measurements[-1]['_actual_mask'] = mask
        return save

    def extract_group(self, frames, capture_times, states, projections):
        self.group_frames, self.capture_times, self.projections = frames, capture_times, projections
        # The history-dependent depth selector queries this one causal state table.
        self.position_layer.tracks, self.position_layer.dense_states = states, {}
        dense, _ = self.extract_detections(frames[0][1])
        observations = []
        for obj in dense:
            source = self.measurement_sources[obj['mask_idx'][-1]]
            observations.append(dict(center=np.asarray(source['center']).copy(), category=source['category'],
                feature=np.asarray(source['feature']).copy(), points=source['points'], confidence=source['confidence'],
                variance=np.asarray(source['variance']).copy(), real_points=np.asarray(source['world_points']).copy(),
                reliable_current_dense=source['reliable'], dense=True, maximum_speed_m_s=20.,
                source_key=source['_observation_key'], cameras=[source['camera']],
                capture_timestamp_s=source['camera_timestamp_s'], box=np.asarray(obj['xyxy'][-1]).copy(),
                depth_evidence=obj['depth_evidence'][-1], mask=obj['mask'][-1]))
        for obj in self.sparse_measurements:
            observations.append(dict(center=np.asarray(obj['center']).copy(), category=obj['category'],
                feature=np.asarray(obj['feature']).copy(), points=obj['points'], confidence=obj['confidence'],
                variance=np.asarray(obj['variance']).copy(), real_points=np.asarray(obj['_world_points']).copy(),
                reliable_current_dense=False, dense=False, maximum_speed_m_s=20.,
                source_key=obj['_observation_key'], cameras=[obj['camera']],
                capture_timestamp_s=capture_times[obj['camera']], box=obj['_image_box'], mask=obj['_actual_mask']))
        for obs in observations:
            projection = projections[obs['cameras'][0]]
            keep = obs.pop('mask')[projection['pixel_y'], projection['pixel_x']]
            token = projection['lidar_token']
            obs['lidar_point_keys'] = [[token, int(i)] for i in projection['point_index'][keep]] if token else []
        return observations, self.frame_audit


class UnifiedMapper:
    def __init__(self, cfg, classes, settings=None):
        self.settings = settings or UnifiedSettings()
        self.extractor = UnifiedObservationExtractor(cfg, classes)
        self.filter = DenseProtectedPositionLayer()
        self.states, self.next_id, self.timestamp = [], 0, None
        self.audit = {}

    def settings_record(self):
        return dict(unified=asdict(self.settings), state_space='one persistent dense+sparse table',
            shared_geometry='physical evidence permits original .65 appearance floor; not global threshold relaxation',
            output_clock='position actual capture timestamp; observation clock kept separately',
            depth_selection='original R1 cluster selector querying the same unified causal table')

    def _prediction_center(self, state, timestamp):
        return state['center']+state.get('velocity', np.zeros(3))*min(max(timestamp-state['timestamp'], 0), self.settings.motion_horizon_s)

    def _scores(self, observations, states, timestamp):
        matrix = np.full((len(observations), len(states)), -np.inf)
        reasons = []
        for i, obs in enumerate(observations):
            for j, state in enumerate(states):
                distance = float(np.linalg.norm((obs['center']-self._prediction_center(state, timestamp))[:2]))
                appearance = cosine(obs['feature'], state['feature'])
                accepted = obs['category'] == state['category'] and distance <= self.settings.association_distance_m and appearance >= self.settings.appearance_minimum
                if accepted:
                    matrix[i, j] = -distance+.25*appearance
                reasons.append(dict(observation=i, node_id=state['node_id'], predicted_distance_m=distance,
                    cosine=appearance, accepted=accepted, dt_s=timestamp-state['timestamp']))
        return matrix, reasons

    def _update(self, state, obs, timestamp, group):
        protected = (obs['representative_camera'] != 'CAM_FRONT' and not obs['dense']
                     and state.get('front_position_timestamp_s') is not None
                     and timestamp-state['front_position_timestamp_s'] <= self.settings.front_protection_s)
        before = state['center'].copy()
        if not protected:
            self.filter._update_state(state, obs, timestamp)
            state['position_capture_timestamp_s'] = obs['capture_timestamp_s']
            if obs['representative_camera'] == 'CAM_FRONT':
                state['front_position_timestamp_s'] = obs['capture_timestamp_s']
        else:
            state['total_points'] += obs['points']
            state['hits'] += 1
            state['confidence'] = max(state['confidence'], obs['confidence'])
        self._metadata(state, obs, timestamp, group)
        return dict(node_id=state['node_id'], action='update', cameras=obs['cameras'],
            observation_keys=obs['source_keys'], before=before.tolist(), after=state['center'].tolist(),
            protected_recent_front_from_sparse_side=protected)

    def _metadata(self, state, obs, timestamp, group):
        state['last_observed_timestamp_s'] = timestamp
        state['last_capture_timestamp_s'] = obs['capture_timestamp_s']
        state['last_group'] = group
        state['groups'].add(group)
        state['hits'] = len(state['groups'])
        state['dense_seen'] |= obs['dense']
        state['dense_strong_seen'] |= obs['dense'] and obs['confidence'] >= self.settings.strong_confidence
        state['last_observation'] = obs
        state['source_keys'].extend(obs['source_keys'])
        state['last_cameras'] = obs['cameras']
        state['last_representative_camera'] = obs['representative_camera']

    def update_group(self, frames, capture_times, projections):
        timestamp, group = frames[0][1].timestamp_s, frames[0][1].frame_idx
        if self.timestamp is not None and timestamp <= self.timestamp:
            raise ValueError('组时钟必须严格递增')
        self.timestamp = timestamp
        retired = []
        live = []
        for state in self.states:
            ttl = self.settings.dense_ttl_s if state['dense_seen'] else self.settings.sparse_ttl_s
            if timestamp-state['last_observed_timestamp_s'] <= ttl:
                live.append(state)
            else:
                retired.append(dict(node_id=state['node_id'], reason='original_numeric_ttl_in_unified_state',
                                    age_s=timestamp-state['last_observed_timestamp_s'], ttl_s=ttl))
        self.states = live
        obs, geometry = self.extractor.extract_group(frames, capture_times, self.states, projections)
        grouped, grouping = group_observations(obs, self.settings)
        front = [i for i, o in enumerate(grouped) if o['representative_camera'] == 'CAM_FRONT']
        rest = [i for i in range(len(grouped)) if i not in front]
        assigned_states, consumed, events, score_audit = set(), set(), [], []
        for indices in [front, rest]:
            available = [j for j in range(len(self.states)) if j not in assigned_states]
            matrix, reasons = self._scores([grouped[i] for i in indices], [self.states[j] for j in available], timestamp)
            score_audit.append(dict(observation_indices=indices, state_indices=available, pairs=reasons))
            assignment = gated_assignment(matrix)
            for local, track in assignment.items():
                i, j = indices[local], available[track]
                events.append(self._update(self.states[j], grouped[i], timestamp, group))
                assigned_states.add(j)
                consumed.add(i)
        for i, observation in enumerate(grouped):
            if i in consumed:
                continue
            node_id = f'unified-{self.next_id:08d}'
            self.next_id += 1
            state = self.filter._new_state(observation, timestamp, node_id)
            state.update(groups=set(), dense_seen=False, dense_strong_seen=False, source_keys=[],
                position_capture_timestamp_s=observation['capture_timestamp_s'],
                front_position_timestamp_s=observation['capture_timestamp_s'] if observation['representative_camera'] == 'CAM_FRONT' else None)
            self._metadata(state, observation, timestamp, group)
            self.states.append(state)
            events.append(dict(node_id=node_id, action='new', cameras=observation['cameras'], observation_keys=observation['source_keys']))
        output, rejected = [], []
        views = [dict(camera_pose=f.camera_pose, intrinsic=f.cam_K, width=f.image_rgb.shape[1], height=f.image_rgb.shape[0]) for _, f in frames]
        for state in self.states:
            age = timestamp-state['position_capture_timestamp_s']
            if state['dense_seen']:
                confirmed = state['dense_strong_seen'] or len(state['groups']) >= 2
            else:
                needed = self.settings.sparse_minimum_frames if state['confidence'] >= self.settings.strong_confidence else self.settings.weak_minimum_frames
                confirmed = len(state['groups']) >= needed and state['total_points'] >= self.settings.minimum_total_sparse_points
            reason = None
            if not confirmed:
                reason = 'not_confirmed'
            elif not state['dense_seen'] and age > self.settings.sparse_output_age_s:
                reason = 'sparse_position_age'
            elif not any(in_view(state['center'], view) for view in views):
                reason = 'original_camera_union_fov'
            if reason:
                rejected.append(dict(node_id=state['node_id'], reason=reason, position_age_s=age))
                continue
            output.append(dict(node_id=state['node_id'], category=state['category'], center=state['center'].tolist(),
                num_detections=len(state['groups']), last_seen_frame=state['last_group'], timestamp_s=state['position_capture_timestamp_s'],
                actual_measurement_timestamp_s=state['last_capture_timestamp_s'], prediction_age_s=age,
                position_covariance_diag=state['variance'].tolist(), position_source='unified_measured_position',
                shape_available=state['dense_seen'], measured_points_total=state['total_points'],
                source_cameras=state['last_cameras'], representative_camera=state['last_representative_camera']))
        self.audit = dict(group=group, grouping=grouping, geometry=geometry, association=score_audit,
            events=events, retired=retired, rejected=rejected,
            states=[dict(node_id=s['node_id'], category=s['category'], center=s['center'].tolist(),
                position_capture_timestamp_s=s['position_capture_timestamp_s'], actual_capture_timestamp_s=s['last_capture_timestamp_s'],
                last_observed_timestamp_s=s['last_observed_timestamp_s'], variance=s['variance'].tolist(),
                hits=len(s['groups']), total_points=s['total_points'], dense_seen=s['dense_seen'], cameras=s['last_cameras']) for s in self.states])
        return copy.deepcopy(output)
