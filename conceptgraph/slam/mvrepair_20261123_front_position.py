"""L2：L1持久canonical内保存轻量前视位置证据，不另造图或输出身份。"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np

from conceptgraph.completion.association import gated_assignment
from conceptgraph.completion.stage9 import predict
from conceptgraph.slam.dense_refinement import DenseProtectedPositionLayer
from conceptgraph.slam.mviter_20261123_diagnostics import SourceAuditedPositionLayer
from conceptgraph.slam.mvrepair_lifecycle_20261123_mapper import LifecycleRetiringMapper
from conceptgraph.slam.sparse_observations import cosine


EXTRA_SOURCES = ['conceptgraph/slam/mvrepair_lifecycle_20261123_mapper.py',
                 'conceptgraph/slam/mvrepair_lifecycle_20261123_registry.py',
                 'conceptgraph/slam/mviter_20261123_diagnostics.py',
                 'tests/mvrepair_20261123_test_front_position.py']


@dataclass(frozen=True)
class UnifiedSettings:
    maximum_front_age_s: float = .75
    association_distance_m: float = 2.5
    appearance_minimum: float = .65
    minimum_sparse_groups: int = 2
    minimum_sparse_points: int = 4
    strong_confidence: float = .2


def position_decision(front, current, timestamp, settings):
    if front is None:
        return False, 'no_front_position'
    age = timestamp-front['_capture_timestamp_s']
    confirmed = (front['_dense_seen'] and front['confidence'] >= settings.strong_confidence) or (
        len(front['_groups']) >= settings.minimum_sparse_groups and front['total_points'] >= settings.minimum_sparse_points)
    if not confirmed:
        return False, 'front_not_confirmed'
    if age > settings.maximum_front_age_s:
        return False, 'front_position_too_old'
    if front['_reliable_dense'] and age <= .075:
        return True, 'reliable_current_front'
    if any(s.get('_last_reliable_dense', False) for s in current):
        return False, 'reliable_current_dense_preserved'
    side_weak = bool(current) and all(s.get('_last_camera') != 'CAM_FRONT' and s.get('_last_observation_points', 0) < 5 for s in current)
    if side_weak or not current:
        return True, 'confirmed_recent_front_over_weak_or_missing_side'
    return False, 'current_measurement_preserved'


class FrontEvidenceLayer(SourceAuditedPositionLayer):
    def __init__(self, mapper):
        super().__init__(mapper)
        self.front_states = {}
        self.front_events = []
        self.front_filter = DenseProtectedPositionLayer()

    def _record(self, state, observation, action):
        super()._record(state, observation, action)
        state['_last_world_points'] = np.asarray(state['_last_world_points']).copy()
        state['_last_reliable_dense'] = observation.get('reliable_current_dense', False)
        state['_last_observation_points'] = observation['points']

    def update(self, sparse, dense_predictions, dense_measurements, timestamp):
        output = super().update(sparse, dense_predictions, dense_measurements, timestamp)
        settings = self.mapper.front_settings
        observations = [o for o in [*dense_measurements.values(), *sparse]
                        if o.get('_camera', o.get('camera')) == 'CAM_FRONT']
        targets = [*self.tracks, *self.dense_states.values()]
        scores = np.full((len(observations), len(targets)), -np.inf)
        details = []
        for i, obs in enumerate(observations):
            for j, state in enumerate(targets):
                distance = float(np.linalg.norm((np.asarray(obs['center'])-state['center'])[:2]))
                appearance = cosine(obs['feature'], state['feature'])
                match = obs['category'] == state['category'] and distance <= settings.association_distance_m and appearance >= settings.appearance_minimum
                if match:
                    # Identical actual source is strongest; otherwise fixed original class/position/appearance gate.
                    exact = obs.get('_observation_key') in state.get('_observation_keys', [])
                    scores[i, j] = (10. if exact else 0.)-distance+.25*appearance
                details.append(dict(observation=i, target=state['node_id'], distance_m=distance,
                    feature_cosine=appearance, accepted=match, source_key=obs['_observation_key']))
        self.front_events = []
        for i, j in gated_assignment(scores).items():
            obs, target = observations[i], targets[j]
            logical = self.aliases.get(target['node_id'], target['node_id'])
            root = self.mapper.lifecycle.resolve(logical)
            front = self.front_states.get(root)
            group = self.mapper.group_frames[0][1].frame_idx
            if front is not None and group in front['_groups']:
                continue
            if front is None:
                front = self.front_filter._new_state(obs, timestamp, root)
                front.update(_groups=set(), _points_by_group={}, _dense_seen=False)
                self.front_states[root] = front
            else:
                self.front_filter._update_state(front, obs, timestamp)
            front['_groups'].add(group)
            front['_points_by_group'][group] = obs['points']
            front['hits'] = len(front['_groups'])
            front['total_points'] = sum(front['_points_by_group'].values())
            front['_capture_timestamp_s'] = obs['_observation_key'][1]
            front['_source_key'] = obs['_observation_key']
            front['_reliable_dense'] = obs.get('reliable_current_dense', False)
            front['_dense_seen'] |= obs['points'] >= 5
            self.front_events.append(dict(canonical_before_current_union=root, target_raw_state=target['node_id'],
                source_key=obs['_observation_key'], points=obs['points'], groups=sorted(front['_groups']),
                total_points=front['total_points'], center=front['center'].tolist(),
                capture_timestamp_s=front['_capture_timestamp_s'], reliable_dense=front['_reliable_dense']))
        self.front_pair_audit = details
        return output


class CanonicalFrontMapper(LifecycleRetiringMapper):
    def __init__(self, cfg, classes, settings=None):
        super().__init__(cfg, classes)
        self.front_settings = settings or UnifiedSettings()
        self.position_layer = FrontEvidenceLayer(self)
        self.audit = {}

    def settings_record(self):
        return dict(front_position=asdict(self.front_settings), lifecycle=asdict(self.lifecycle.settings),
            original_confirm_TTL='unchanged; covered-state retirement from L1 only',
            additional_state='one position filter per canonical; no independent object/output ID')

    def position_predictions(self, predictions, frame):
        output = super().position_predictions(predictions, frame)
        layer = self.position_layer
        # A newly witnessed union can combine two earlier front filters; counts use unique groups.
        merged = {}
        for node, state in layer.front_states.items():
            root = self.lifecycle.resolve(node)
            if root not in merged:
                merged[root] = state
            else:
                old = merged[root]
                keep, other = (state, old) if state['_capture_timestamp_s'] > old['_capture_timestamp_s'] else (old, state)
                keep['_groups'] |= other['_groups']
                for group, points in other['_points_by_group'].items():
                    keep['_points_by_group'][group] = max(keep['_points_by_group'].get(group, 0), points)
                keep['hits'], keep['total_points'] = len(keep['_groups']), sum(keep['_points_by_group'].values())
                merged[root] = keep
        layer.front_states = merged
        decisions = []
        states = [*layer.tracks, *layer.dense_states.values()]
        for obj in output:
            root = obj['node_id']
            front = merged.get(root)
            current = [s for s in states if self.lifecycle.resolve(layer.aliases.get(s['node_id'], s['node_id'])) == root and s['timestamp'] == frame.timestamp_s]
            use, reason = position_decision(front, current, frame.timestamp_s, self.front_settings)
            if front is not None:
                age = frame.timestamp_s-front['_capture_timestamp_s']
                if use:
                    before = obj['center']
                    obj.update(center=front['center'].tolist(), position_covariance_diag=front['variance'].tolist(),
                        position_source='canonical_confirmed_front_position', timestamp_s=front['_capture_timestamp_s'],
                        prediction_age_s=age, front_position_source_key=front['_source_key'])
                    decisions.append(dict(canonical=root, used=True, reason=reason, before=before,
                        after=obj['center'], source_key=front['_source_key'], groups=sorted(front['_groups']),
                        actual_capture_timestamp_s=front['_capture_timestamp_s'], age_s=age))
                    continue
            decisions.append(dict(canonical=root, used=False, reason=reason))
        self.front_audit = dict(observations=layer.front_events, association=layer.front_pair_audit,
                               position_decisions=decisions)
        return output

    def update_group(self, frames, capture_times, projections=None):
        self.capture_times = capture_times
        super().update_group(frames)
        output = self.position_predictions(predict(self.confirmed_objects(), self.classes), frames[0][1])
        self.audit = dict(lifecycle=self.lifecycle_audit, front=self.front_audit,
                          mapping=self.frame_audit, crossview=self.crossview_audit)
        return copy.deepcopy(output)
