"""模块A：实测深度簇选择和独立稀疏位置层，不制造点云或三维框。"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np

from conceptgraph.completion.association import gated_assignment


@dataclass(frozen=True)
class SparseQualitySettings:
    minimum_frames: int = 2
    weak_minimum_frames: int = 3
    minimum_total_points: int = 4
    ttl_s: float = 1.5
    maximum_prediction_age_s: float = .75
    association_distance_m: float = 2.5
    appearance_minimum: float = .65
    strong_confidence: float = .2
    process_variance_per_s: float = .5
    support_weight: float = .5
    core_weight: float = 1.
    dispersion_weight: float = .15
    history_weight: float = .5

    def to_dict(self):
        return asdict(self)


def cosine(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-10 else 0.


def measured_points(depth, mask, intrinsic, pose):
    """只反投影实际有效像素；不加入扰动、复制或插值点。"""
    v, u = np.nonzero(mask & np.isfinite(depth) & (depth > 1) & (depth < 65))
    z = depth[v, u]
    k = np.asarray(intrinsic)
    camera = np.column_stack(((u - k[0, 2]) * z / k[0, 0],
                              (v - k[1, 2]) * z / k[1, 1], z))
    return camera @ pose[:3, :3].T + pose[:3, 3]


def select_depth_cluster(depth, mask, *, history_depths=(), settings=None):
    settings = settings or SparseQualitySettings()
    valid = mask & np.isfinite(depth) & (depth > 1) & (depth < 65)
    v, u = np.nonzero(valid)
    values = depth[v, u]
    evidence = dict(mask_pixels=int(mask.sum()), valid_pixels=len(values), retained_pixels=0,
                    source='current_projected_lidar', units='m', robust_filter=True,
                    selection='support_core_dispersion_past_depth', cluster_scores=[])
    if not len(values):
        return valid, evidence
    order = np.argsort(values, kind='stable')
    sorted_z = values[order]
    splits = np.flatnonzero(np.diff(sorted_z) > np.maximum(1., .03 * sorted_z[:-1])) + 1
    groups = np.split(order, splits)
    mv, mu = np.nonzero(mask)
    cx, cy = (float(mu.min() + mu.max()) / 2, float(mv.min() + mv.max()) / 2)
    sx, sy = max(float(mu.max() - mu.min()) / 2, 1.), max(float(mv.max() - mv.min()) / 2, 1.)
    core = 1 - np.clip(((u - cx) / sx)**2 / 2 + ((v - cy) / sy)**2 / 2, 0, 1)
    history = np.asarray(history_depths, dtype=float)
    history = history[np.isfinite(history) & (history > 1) & (history < 65)]
    ranked = []
    for index, group in enumerate(groups):
        median = float(np.median(values[group]))
        spread = float(np.ptp(np.quantile(values[group], [.25, .75])))
        history_cost = min(float(np.min(np.abs(history - median) / np.maximum(1., .05 * history))), 4.) if len(history) else 0.
        score = (settings.support_weight * np.log1p(len(group))
                 + settings.core_weight * float(core[group].mean())
                 - settings.dispersion_weight * spread / max(1., .03 * median)
                 - settings.history_weight * history_cost)
        ranked.append((float(score), len(group), -median, -index))
        evidence['cluster_scores'].append(dict(points=len(group), median_m=median,
            spread_m=spread, core=float(core[group].mean()), past_depth_cost=history_cost, score=float(score)))
    chosen_index = max(range(len(groups)), key=lambda j: ranked[j])
    selected = groups[chosen_index]
    result = np.zeros_like(valid)
    result[v[selected], u[selected]] = True
    evidence.update(retained_pixels=len(selected), cluster_count=len(groups),
        selected_cluster=chosen_index, depth_quantiles_m=np.quantile(values[selected], [0, .1, .5, .9, 1]).tolist(),
        support_fraction=len(selected) / len(values), sufficient_support=len(selected) >= 5)
    return result, evidence


def measurement_variance(points, *, range_m):
    points = np.asarray(points, dtype=float)
    base = (.25 + .02 * range_m)**2
    dispersion = np.var(points, axis=0) / max(len(points), 1) if len(points) else np.zeros(3)
    return np.maximum(base + dispersion, .04)


class SparsePositionLayer:
    """固定距离的一对一必要接口，模块B的两轮分配不在此实现。"""
    def __init__(self, settings=None):
        self.settings = settings or SparseQualitySettings()
        self.tracks = []
        self.dense_states = {}
        self.aliases = {}
        self.next_id = 0
        self.timestamp = None

    def _score(self, observation, state):
        if observation['category'] != state['category']:
            return -np.inf
        distance = float(np.linalg.norm((np.asarray(observation['center']) - state['center'])[:2]))
        similarity = cosine(observation['feature'], state['feature'])
        if distance > self.settings.association_distance_m or similarity < self.settings.appearance_minimum:
            return -np.inf
        return -distance + .25 * similarity

    def _update_state(self, state, observation, timestamp):
        center = np.asarray(observation['center'], dtype=float)
        variance = np.asarray(observation['variance'], dtype=float)
        dt = timestamp - state['timestamp']
        predicted = state['center'] + state.get('velocity', np.zeros(3)) * min(dt, .75)
        prior = state['variance'] + self.settings.process_variance_per_s * max(dt, 0.)
        gain = prior / (prior + variance)
        corrected = predicted + gain * (center - predicted)
        if dt >= .25:
            velocity = (center - state['last_measured_center']) / dt
            if np.linalg.norm(velocity[:2]) <= 20:
                state['velocity'] = .5 * state.get('velocity', np.zeros(3)) + .5 * velocity
        state.update(center=corrected, variance=(1 - gain) * prior,
                     timestamp=timestamp, feature=np.asarray(observation['feature']),
                     last_measured_center=center, total_points=state['total_points'] + observation['points'],
                     confidence=max(state['confidence'], observation['confidence']), hits=state['hits'] + 1)

    def _new_state(self, observation, timestamp, node_id):
        center = np.asarray(observation['center'], dtype=float)
        return dict(node_id=node_id, center=center.copy(), last_measured_center=center.copy(),
                    variance=np.asarray(observation['variance'], dtype=float),
                    velocity=np.zeros(3), category=observation['category'], timestamp=timestamp,
                    feature=np.asarray(observation['feature']), hits=1,
                    total_points=observation['points'], confidence=observation['confidence'])

    def update(self, sparse, dense_predictions, dense_measurements, timestamp):
        if not np.isfinite(timestamp) or (self.timestamp is not None and timestamp <= self.timestamp):
            raise ValueError('位置层时间必须严格递增')
        for observation in [*sparse, *dense_measurements.values()]:
            if (np.asarray(observation['center']).shape != (3,)
                    or not np.isfinite(observation['center']).all()
                    or not np.isfinite(observation['variance']).all()
                    or np.any(np.asarray(observation['variance']) <= 0)):
                raise ValueError('测量或协方差非法')
        self.timestamp = timestamp
        self.tracks = [t for t in self.tracks if timestamp - t['timestamp'] <= self.settings.ttl_s]
        live_ids = {p['node_id'] for p in dense_predictions}
        self.dense_states = {k: v for k, v in self.dense_states.items()
                             if k in live_ids or timestamp - v['timestamp'] <= self.settings.ttl_s}
        for node_id, observation in dense_measurements.items():
            if node_id in self.dense_states:
                self._update_state(self.dense_states[node_id], observation, timestamp)
            else:
                self.dense_states[node_id] = self._new_state(observation, timestamp, node_id)
        # 稀疏观测可补充已有点云节点的位置，不制造新点云。
        dense_keys = [p['node_id'] for p in dense_predictions if p['node_id'] in self.dense_states]
        scores = np.array([[self._score(o, self.dense_states[k]) for k in dense_keys]
                           for o in sparse], dtype=float).reshape(len(sparse), len(dense_keys))
        assigned = gated_assignment(scores)
        for i, j in assigned.items():
            state = self.dense_states[dense_keys[j]]
            if state['timestamp'] < timestamp:
                self._update_state(state, sparse[i], timestamp)
        remaining = [o for i, o in enumerate(sparse) if i not in assigned]
        scores = np.array([[self._score(o, t) for t in self.tracks] for o in remaining], dtype=float)
        scores = scores.reshape(len(remaining), len(self.tracks))
        assigned = gated_assignment(scores)
        for i, observation in enumerate(remaining):
            if i in assigned:
                self._update_state(self.tracks[assigned[i]], observation, timestamp)
            else:
                self.tracks.append(self._new_state(observation, timestamp, f'sparse-{self.next_id:08d}'))
                self.next_id += 1
        output = copy.deepcopy(dense_predictions)
        for obj in output:
            state = self.dense_states.get(obj['node_id'])
            if state is not None and timestamp - state['timestamp'] <= self.settings.maximum_prediction_age_s:
                obj.update(center=state['center'].tolist(), position_covariance_diag=state['variance'].tolist(),
                           position_source='quality_weighted_real_measurements',
                           prediction_age_s=timestamp - state['timestamp'])
        # 已有点云节点附近的位置假设不重复输出；重新获得充分点云时延续稀疏身份。
        used = set()
        for obj in output:
            observation = dense_measurements.get(obj['node_id'])
            if observation is None:
                continue
            options = [(self._score(observation, t), j) for j, t in enumerate(self.tracks) if j not in used]
            options = [(s, j) for s, j in options if np.isfinite(s)]
            if options:
                _, j = max(options, key=lambda v: (v[0], -v[1]))
                if self.tracks[j]['hits'] >= self.settings.minimum_frames:
                    self.aliases.setdefault(obj['node_id'], self.tracks[j]['node_id'])
                used.add(j)
        self.tracks = [t for j, t in enumerate(self.tracks) if j not in used]
        for obj in output:
            obj['node_id'] = self.aliases.get(obj['node_id'], obj['node_id'])
        for track in self.tracks:
            frames_needed = (self.settings.minimum_frames if track['confidence'] >= self.settings.strong_confidence
                             else self.settings.weak_minimum_frames)
            if (track['hits'] < frames_needed or track['total_points'] < self.settings.minimum_total_points
                    or timestamp - track['timestamp'] > self.settings.maximum_prediction_age_s):
                continue
            if any(obj['category'] == track['category'] and np.linalg.norm(
                   (np.asarray(obj['center']) - track['center'])[:2]) <= self.settings.association_distance_m
                   for obj in output):
                continue
            output.append(dict(node_id=track['node_id'], category=track['category'],
                center=track['center'].tolist(), num_detections=track['hits'],
                timestamp_s=track['timestamp'], prediction_age_s=timestamp - track['timestamp'],
                position_covariance_diag=track['variance'].tolist(), position_source='confirmed_sparse_lidar',
                measured_points_total=track['total_points'], shape_available=False))
        return output
