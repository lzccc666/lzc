"""唯一D候选：独立前视锚定，仅以近期实测点支持规范化重复输出。"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.spatial import cKDTree

from conceptgraph.completion.stage9 import predict
from conceptgraph.slam.mviter_20261123_diagnostics import SourceAuditedMapper
from conceptgraph.slam.sparse_observations import cosine


@dataclass(frozen=True)
class AnchorSettings:
    xy_gate_m: float = 2.5
    z_gate_m: float = 1.5
    minimum_cosine: float = .8
    maximum_support_age_s: float = .75
    maximum_capture_gap_s: float = .075
    shared_point_radius_m: float = .2
    minimum_shared_fraction: float = .2
    minimum_shared_points: int = 2


def current_evidence(mapper, prediction):
    layer = mapper.position_layer
    state = next((s for s in [*layer.tracks, *layer.dense_states.values()]
                  if layer.aliases.get(s['node_id'], s['node_id']) == prediction['node_id']), None)
    points = np.asarray(state.get('_last_world_points', [])).reshape(-1, 3) if state else np.empty((0, 3))
    keys = state.get('_observation_keys', []) if state else []
    return dict(points=points, feature=np.asarray(state['feature']) if state else np.empty(0),
        capture_timestamp_s=keys[-1][1] if keys else None,
        state_timestamp_s=state['timestamp'] if state else None,
        source_camera=state.get('_last_camera') if state else None,
        confidence=float(state['confidence']) if state else 0.)


def support_pair(a, b, timestamp, settings):
    pa, pb = a['prediction'], b['prediction']
    ea, eb = a['evidence'], b['evidence']
    delta = np.asarray(pa['center'])-np.asarray(pb['center'])
    result = dict(source_a=a['source'], node_a=pa['node_id'], source_b=b['source'], node_b=pb['node_id'],
        xy_distance_m=float(np.linalg.norm(delta[:2])), z_distance_m=float(abs(delta[2])),
        capture_timestamp_a=ea['capture_timestamp_s'], capture_timestamp_b=eb['capture_timestamp_s'],
        camera_a=ea.get('source_camera'), camera_b=eb.get('source_camera'),
        measured_points_a=len(ea['points']), measured_points_b=len(eb['points']),
        state_timestamp_a=ea['state_timestamp_s'], state_timestamp_b=eb['state_timestamp_s'],
        feature_cosine=None, shared_fraction=None, shared_points=0, eligible=False)
    if a['source'] == b['source'] == 'front':
        result['reason'] = 'two_front_anchors_preserved'
    elif pa['category'] != pb['category']:
        result['reason'] = 'different_class'
    elif result['xy_distance_m'] > settings.xy_gate_m or result['z_distance_m'] > settings.z_gate_m:
        result['reason'] = 'position_gate'
    elif any(e['state_timestamp_s'] is None or not 0 <= timestamp-e['state_timestamp_s'] <= settings.maximum_support_age_s
             for e in [ea, eb]):
        result['reason'] = 'missing_or_old_actual_measurement'
    elif any(e['capture_timestamp_s'] is None for e in [ea, eb]) or abs(ea['capture_timestamp_s']-eb['capture_timestamp_s']) > settings.maximum_capture_gap_s:
        result['reason'] = 'capture_time_gap'
    elif min(len(ea['points']), len(eb['points'])) < settings.minimum_shared_points:
        result['reason'] = 'insufficient_current_points'
    else:
        similarity = cosine(ea['feature'], eb['feature'])
        result['feature_cosine'] = similarity
        if similarity < settings.minimum_cosine:
            result['reason'] = 'appearance_gate'
        else:
            small, large = (ea['points'], eb['points']) if len(ea['points']) <= len(eb['points']) else (eb['points'], ea['points'])
            count = int(np.count_nonzero(cKDTree(large).query(small)[0] <= settings.shared_point_radius_m))
            result['shared_points'], result['shared_fraction'] = count, count/len(small)
            result['eligible'] = count >= settings.minimum_shared_points and result['shared_fraction'] >= settings.minimum_shared_fraction
            result['reason'] = 'current_measured_support' if result['eligible'] else 'no_shared_current_support'
    return result


def canonicalize(entries, timestamp, settings=None):
    settings = settings or AnchorSettings()
    order = sorted(range(len(entries)), key=lambda i: (
        0 if entries[i]['source'] == 'front' else 1,
        0 if entries[i]['prediction'].get('position_source') == 'reliable_current_dense_lidar' else 1,
        -len(entries[i]['evidence']['points']), -entries[i]['evidence']['confidence'], entries[i]['prediction']['node_id']))
    pairs, clusters = {}, []
    for position, i in enumerate(order):
        for j in order[:position]:
            pairs[min(i, j), max(i, j)] = support_pair(entries[j], entries[i], timestamp, settings)
        compatible = [c for c in clusters if all(pairs[min(i, j), max(i, j)]['eligible'] for j in c)]
        if compatible:
            compatible[0].append(i)
        else:
            clusters.append([i])
    output = []
    for cluster in clusters:
        representative = entries[cluster[0]]
        obj = copy.deepcopy(representative['prediction'])
        obj['node_id'] = representative['source']+'/'+obj['node_id']
        output.append(obj)
    return output, dict(input_outputs=len(entries), kept_outputs=len(clusters), removed=len(entries)-len(clusters),
        pairs=list(pairs.values()), clusters=[dict(representative=c[0], members=c,
            sources=[entries[i]['source'] for i in c], node_ids=[entries[i]['prediction']['node_id'] for i in c]) for c in clusters])


class FrontAnchoredMapper:
    def __init__(self, cfg, classes):
        self.classes = classes
        self.front = SourceAuditedMapper(copy.deepcopy(cfg), classes, deduplication=False)
        self.multi = SourceAuditedMapper(copy.deepcopy(cfg), classes, deduplication=True)
        self.settings = AnchorSettings()
        self.audit = {}
        self.raw_predictions = {}

    def settings_record(self):
        return dict(anchor=asdict(self.settings), original_front=self.front.settings_record(),
                    original_multi=self.multi.settings_record(), maps=2, no_GT_or_FOV_suppression=True)

    def update_group(self, frames, capture_times):
        start = time.perf_counter()
        entries = []
        for source, mapper, selected in [('front', self.front, [(c, f) for c, f in frames if c == 'CAM_FRONT']),
                                         ('multi', self.multi, frames)]:
            np.random.seed(17+frames[0][1].frame_idx)
            mapper.capture_times = {c: capture_times[c] for c, _ in selected}
            cloned = [(c, copy.deepcopy(f)) for c, f in selected]
            mapper.update_group(cloned)
            predictions = mapper.position_predictions(predict(mapper.confirmed_objects(), self.classes), cloned[0][1])
            self.raw_predictions[source] = predictions
            entries.extend(dict(source=source, prediction=p, evidence=current_evidence(mapper, p)) for p in predictions)
        output, audit = canonicalize(entries, frames[0][1].timestamp_s, self.settings)
        self.audit = dict(canonicalization=audit, front_mapping=self.front.frame_audit,
            multi_mapping=self.multi.frame_audit, front_crossview=self.front.crossview_audit,
            multi_crossview=self.multi.crossview_audit, wall_s=time.perf_counter()-start)
        return output
