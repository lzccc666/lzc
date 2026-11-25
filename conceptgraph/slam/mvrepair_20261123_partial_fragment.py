"""T1：跨两组互投与连续可靠track见证后，收束单组部分表面碎片。"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from conceptgraph.slam.mvrepair_20261123_combined import CanonicalPhysicalFrontMapper
from conceptgraph.slam.mvrepair_20261123_conflict_position import (
    EXTRA_SOURCES as POSITION_SOURCES,
    ConflictOnlyFrontMapper,
    UnifiedSettings as PositionSettings,
)
from conceptgraph.slam.mvrepair_lifecycle_20261123_registry import CanonicalLifecycle, shared_support_witness
from conceptgraph.slam.sparse_observations import cosine


EXTRA_SOURCES = [*POSITION_SOURCES,
    'conceptgraph/slam/mvrepair_20261123_conflict_position.py',
    'tests/mvrepair_20261123_test_partial_fragment.py']


@dataclass(frozen=True)
class UnifiedSettings(PositionSettings):
    conflict_only_position: bool = True


def projected_support(points, view, mask):
    points = np.asarray(points).reshape(-1, 3)
    if not len(points):
        return dict(points=0, valid=0, inside=0, fraction=0.)
    camera = np.linalg.solve(view['pose'], np.c_[points, np.ones(len(points))].T).T[:, :3]
    uvz = camera @ view['intrinsic'].T
    valid = np.isfinite(camera).all(1) & (camera[:, 2] > 1) & (camera[:, 2] < 65)
    pixel = np.zeros((len(points), 2), dtype=int)
    pixel[valid] = np.rint(uvz[valid, :2]/uvz[valid, 2:]).astype(int)
    height, width = mask.shape
    valid &= (pixel[:, 0] >= 0) & (pixel[:, 0] < width) & (pixel[:, 1] >= 0) & (pixel[:, 1] < height)
    inside = np.zeros(len(points), dtype=bool)
    inside[valid] = mask[pixel[valid, 1], pixel[valid, 0]]
    return dict(points=len(points), valid=int(valid.sum()), inside=int(inside.sum()), fraction=float(inside.mean()))


def temporal_witness(weak, previous, current, past_view, current_view, now):
    """固定已有数值门；5<=点<8来自dense入口与R1可靠dense边界。"""
    evidence = dict(accepted=False, reason='eligibility_gate')
    wp, pp, cp = [np.asarray(s['_last_world_points']).reshape(-1, 3) for s in [weak, previous, current]]
    wk, pk, ck = [s['_observation_keys'][-1] for s in [weak, previous, current]]
    if (weak['hits'] != 1 or not 5 <= len(wp) < 8 or previous['hits'] < 2
            or not previous.get('_last_reliable_dense') or not current.get('_last_reliable_dense')
            or current['timestamp'] != now or weak['timestamp'] != previous['timestamp']
            or not 0 < now-weak['timestamp'] <= .75 or wk[:2] != pk[:2] or wk[0] != ck[0]
            or past_view.get('capture_timestamp_s', wk[1]) != wk[1]
            or current_view.get('capture_timestamp_s', ck[1]) != ck[1]
            or len(cp) < 8 or len(pp) < 8):
        return evidence
    delta = np.asarray(weak['last_measured_center'])-current['last_measured_center']
    appearance = min(cosine(weak['feature'], previous['feature']), cosine(weak['feature'], current['feature']))
    evidence.update(weak_source=wk, previous_source=pk, current_source=ck,
        weak_points=len(wp), previous_points=len(pp), current_points=len(cp),
        capture_gap_s=ck[1]-wk[1], age_s=now-weak['timestamp'], xy_distance_m=float(np.linalg.norm(delta[:2])),
        z_distance_m=float(abs(delta[2])), minimum_feature_cosine=appearance)
    if np.linalg.norm(delta[:2]) > 2.5 or abs(delta[2]) > 1.5 or appearance < .8:
        evidence['reason'] = 'original_position_appearance_gate'
        return evidence
    wm, pm, cm = past_view['masks'][wk[2]], past_view['masks'][pk[2]], current_view['masks'][ck[2]]
    union = int((wm | pm).sum())
    overlap = float((wm & pm).sum()/union) if union else 0.
    evidence['previous_mask_IoU'] = overlap
    if overlap < .8:
        evidence['reason'] = 'previous_independent_pixel_instances'
        return evidence
    forward_weak = projected_support(wp, current_view, cm)
    forward_previous = projected_support(pp, current_view, cm)
    reverse_weak = projected_support(cp, past_view, wm)
    reverse_previous = projected_support(cp, past_view, pm)
    evidence.update(forward_weak=forward_weak, forward_previous=forward_previous,
        reverse_weak=reverse_weak, reverse_previous=reverse_previous)
    evidence['accepted'] = min(forward_weak['fraction'], forward_previous['fraction']) >= .9 and min(reverse_weak['fraction'], reverse_previous['fraction']) >= .2
    evidence['reason'] = 'bidirectional_partial_surface_continuity' if evidence['accepted'] else 'bidirectional_support_gate'
    return evidence


class TemporalLifecycle(CanonicalLifecycle):
    def __init__(self, mapper):
        super().__init__()
        self.mapper = mapper

    def observe_group(self, group_id, timestamp_s, observations):
        mapper, layer = self.mapper, self.mapper.position_layer
        states = [*layer.tracks, *layer.dense_states.values()]
        previous = mapper.previous_states
        records, accepted = [], []
        for weak in states:
            weak_root = self.resolve(layer.aliases.get(weak['node_id'], weak['node_id']))
            # 未测旧碎片的canonical若已有任何当前独立观测，不越过component约束吞并。
            if weak['timestamp'] == timestamp_s or any(self.resolve(o['track_id']) == weak_root for o in observations):
                continue
            if weak['node_id'] not in previous:
                continue
            for current in states:
                current_root = self.resolve(layer.aliases.get(current['node_id'], current['node_id']))
                if current_root == weak_root or current['timestamp'] != timestamp_s or current['node_id'] not in previous or weak['category'] != current['category']:
                    continue
                old = previous[current['node_id']]
                if weak['category'] not in ['car', 'truck', 'bus']:
                    continue
                wk, ck = weak.get('_observation_keys', [None])[-1], current.get('_observation_keys', [None])[-1]
                if wk is None or ck is None or wk[0] not in mapper.previous_views or ck[0] not in mapper.current_views:
                    continue
                witness = temporal_witness(weak, old, current, mapper.previous_views[wk[0]], mapper.current_views[ck[0]], timestamp_s)
                witness.update(weak_raw=weak['node_id'], strong_raw=current['node_id'], weak_canonical=weak_root, strong_canonical=current_root)
                if witness['accepted']:
                    component = [o for o in observations if self.resolve(o['track_id']) == current_root]
                    if any(tuple(a['source_key']) != tuple(b['source_key']) and not shared_support_witness(a, b, self.settings)['accepted'] for a, b in combinations(component, 2)):
                        witness.update(accepted=False, reason='current_component_complete_link_conflict')
                    # 另一独立当前同类实体同样覆盖旧点时，证据不唯一，保持两者。
                    for other in observations:
                        if self.resolve(other['track_id']) in [current_root, weak_root] or other['category'] != weak['category'] or other['source_key'][0] != ck[0]:
                            continue
                        mask = mapper.current_views[ck[0]]['masks'][other['source_key'][2]]
                        if cosine(weak['feature'], other['feature']) >= .8 and projected_support(weak['_last_world_points'], mapper.current_views[ck[0]], mask)['fraction'] >= .9:
                            witness.update(accepted=False, reason='independent_current_instance_also_explains_fragment')
                            break
                records.append(witness)
                if witness['accepted']:
                    accepted.append(witness)
        # 相互唯一候选才union，避免一个大mask把多个独立弱实体传递桥接。
        for witness in accepted:
            same_weak = sum(w['weak_canonical'] == witness['weak_canonical'] for w in accepted)
            same_strong = sum(w['strong_canonical'] == witness['strong_canonical'] for w in accepted)
            if same_weak != 1 or same_strong != 1:
                witness.update(accepted=False, reason='nonunique_fragment_component_link')
                continue
            self._union(witness['weak_canonical'], witness['strong_canonical'])
        mapper.temporal_audit = records
        result = super().observe_group(group_id, timestamp_s, observations)
        result['temporal_partial_surface_witnesses'] = records
        return result


class PartialFragmentMapper(CanonicalPhysicalFrontMapper):
    def __init__(self, cfg, classes, settings=None):
        super().__init__(cfg, classes, settings or UnifiedSettings())
        self.lifecycle = TemporalLifecycle(self)
        self.previous_states, self.previous_views, self.current_views = {}, {}, {}
        self.temporal_audit = []

    def settings_record(self):
        return dict(**super().settings_record(), partial_surface=dict(weak_points='original dense5<=N<reliable8',
            age_s=.75, association_distance_m=2.5, appearance=.8, previous_mask_iou=.8,
            forward_support=.9, reverse_support=.2, unique_component=True,
            no_TTL_confirmation_or_same_group_hit_change=True))

    def position_predictions(self, predictions, frame):
        method = ConflictOnlyFrontMapper.position_predictions if self.front_settings.conflict_only_position else CanonicalPhysicalFrontMapper.position_predictions
        return method(self, predictions, frame)

    def update_group(self, frames, capture_times, projections=None):
        self.current_views = {camera:dict(pose=frame.camera_pose.copy(), intrinsic=frame.cam_K.copy(), capture_timestamp_s=capture_times[camera],
            masks={i:np.asarray(mask, bool).copy() for i, mask in enumerate(frame.detections['mask'])}) for camera, frame in frames}
        output = super().update_group(frames, capture_times, projections)
        layer = self.position_layer
        self.previous_states = {s['node_id']:copy.deepcopy(s) for s in [*layer.tracks, *layer.dense_states.values()]}
        self.previous_views = self.current_views
        self.audit['temporal_partial_surface'] = self.temporal_audit
        return output
