"""有实测见证的持久身份与被覆盖state退休；不修改全局TTL或确认门。

独立接口：CanonicalLifecycle.observe_group(group_id, timestamp_s, observations)。
每条观测带原track_id/source_key/category/real_points/feature/capture_timestamp_s。
输出aliases、仅已证实覆盖的retired_state_ids、独立group命中数和见证。
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree

from conceptgraph.slam.sparse_observations import cosine


@dataclass(frozen=True)
class LifecycleSettings:
    shared_point_radius_m: float = .2
    minimum_shared_points: int = 2
    minimum_shared_fraction: float = .2
    maximum_xy_distance_m: float = 1.5
    maximum_z_distance_m: float = 1.5
    maximum_capture_gap_s: float = .075
    minimum_feature_cosine: float = .65


def shared_support_witness(a, b, settings=None):
    """相机不同、同类、同组真实测量交叠；没有点/外观证据时拒绝。"""
    s = settings or LifecycleSettings()
    pa = np.asarray(a['real_points'], dtype=float).reshape(-1, 3)
    pb = np.asarray(b['real_points'], dtype=float).reshape(-1, 3)
    result = dict(track_a=a['track_id'], track_b=b['track_id'], accepted=False,
                  source_key_a=a['source_key'], source_key_b=b['source_key'],
                  reason='invalid_or_insufficient_real_points')
    if min(len(pa), len(pb)) < s.minimum_shared_points or not np.isfinite(pa).all() or not np.isfinite(pb).all():
        return result
    ca, cb = np.asarray(a['center'], dtype=float), np.asarray(b['center'], dtype=float)
    ta, tb = float(a['capture_timestamp_s']), float(b['capture_timestamp_s'])
    feature_cosine = cosine(a['feature'], b['feature'])
    if not np.isfinite([*ca, *cb, ta, tb, feature_cosine]).all():
        return result
    delta = ca - cb
    result.update(xy_distance_m=float(np.linalg.norm(delta[:2])), z_distance_m=float(abs(delta[2])),
                  capture_gap_s=abs(ta-tb), feature_cosine=feature_cosine)
    if a['category'] != b['category']:
        result['reason'] = 'different_category'
    elif a['source_key'][0] == b['source_key'][0]:
        result['reason'] = 'same_camera_no_crossview_witness'
    elif abs(ta-tb) > s.maximum_capture_gap_s:
        result['reason'] = 'capture_time_gate'
    elif result['xy_distance_m'] > s.maximum_xy_distance_m or result['z_distance_m'] > s.maximum_z_distance_m:
        result['reason'] = 'geometry_gate'
    elif feature_cosine < s.minimum_feature_cosine:
        result['reason'] = 'appearance_gate'
    else:
        small, large = (pa, pb) if len(pa) <= len(pb) else (pb, pa)
        count = int(np.sum(cKDTree(large).query(small)[0] <= s.shared_point_radius_m))
        result.update(shared_points=count, smaller_points=len(small), shared_fraction=count/len(small))
        result['accepted'] = count >= s.minimum_shared_points and count/len(small) >= s.minimum_shared_fraction
        result['reason'] = 'current_shared_real_support' if result['accepted'] else 'no_shared_real_support'
    return result


class CanonicalLifecycle:
    """canonical首次出现顺序固定；计数按group集合，绝不累加同组相机命中。"""
    def __init__(self, settings=None):
        self.settings = settings or LifecycleSettings()
        self.parent, self.order, self.groups, self.source_owner = {}, {}, {}, {}
        self.timestamp = None
        self.ambiguous_current_roots = set()

    def resolve(self, track_id):
        if track_id not in self.parent:
            self.order[track_id] = len(self.order)
            self.parent[track_id] = track_id
            self.groups[track_id] = set()
        if self.parent[track_id] != track_id:
            self.parent[track_id] = self.resolve(self.parent[track_id])
        return self.parent[track_id]

    def _union(self, a, b):
        a, b = self.resolve(a), self.resolve(b)
        if a == b:
            return a
        keep, retire = sorted([a, b], key=self.order.__getitem__)
        self.parent[retire] = keep
        self.groups[keep].update(self.groups[retire])
        return keep

    def observe_group(self, group_id, timestamp_s, observations):
        if not np.isfinite(timestamp_s) or (self.timestamp is not None and timestamp_s <= self.timestamp):
            raise ValueError('group时刻必须严格递增')
        self.timestamp = timestamp_s
        witnesses = []
        # 全部本组观测先登记；归并必须对两个component的每一对观测检查，防传递桥。
        for o in observations:
            self.resolve(o['track_id'])

        def component_merge(a_id, b_id, witness):
            a_root, b_root = self.resolve(a_id), self.resolve(b_id)
            if a_root == b_root:
                return
            left = [o for o in observations if self.resolve(o['track_id']) == a_root]
            right = [o for o in observations if self.resolve(o['track_id']) == b_root]
            for a in left:
                for b in right:
                    if tuple(a['source_key']) == tuple(b['source_key']) and a['category'] == b['category']:
                        continue
                    check = shared_support_witness(a, b, self.settings)
                    if not check['accepted']:
                        witness.update(accepted=False, reason='component_complete_link_conflict',
                                       conflicting_pair=check)
                        return
            self._union(a_id, b_id)

        for o in observations:
            key = tuple(o['source_key'])
            if len(key) != 3 or not isinstance(key[0], str) or isinstance(key[2], bool) or not isinstance(key[2], int):
                raise ValueError('source_key必须为camera/exposure_timestamp/detection_index完整三元组')
            capture = float(o['capture_timestamp_s'])
            if not np.isfinite(capture) or not 0 <= timestamp_s-capture <= self.settings.maximum_capture_gap_s:
                raise ValueError('仅可注册本组已采集实测观测')
            if key in self.source_owner:
                other = self.source_owner[key]
                if self.resolve(other) != self.resolve(o['track_id']):
                    witness = dict(track_a=other, track_b=o['track_id'], accepted=True,
                                   reason='identical_real_observation_key', source_key=list(key))
                    component_merge(other, o['track_id'], witness)
                    witnesses.append(witness)
            self.source_owner[key] = o['track_id']
        for i, a in enumerate(observations):
            for b in observations[:i]:
                if self.resolve(a['track_id']) == self.resolve(b['track_id']):
                    continue
                witness = shared_support_witness(a, b, self.settings)
                if witness['accepted']:
                    component_merge(a['track_id'], b['track_id'], witness)
                witnesses.append(witness)
        for o in observations:
            self.groups[self.resolve(o['track_id'])].add(group_id)
        aliases = {k: self.resolve(k) for k in self.parent if self.resolve(k) != k}
        # 这里只退役已同当前实测归并的状态，未命中旧state不会按距离/年龄删除。
        current_roots = {self.resolve(o['track_id']) for o in observations}
        self.ambiguous_current_roots = set()
        for i, a in enumerate(observations):
            for b in observations[:i]:
                root = self.resolve(a['track_id'])
                if root != self.resolve(b['track_id']) or tuple(a['source_key']) == tuple(b['source_key']):
                    continue
                if not shared_support_witness(a, b, self.settings)['accepted']:
                    self.ambiguous_current_roots.add(root)
        retired = [k for k in aliases if self.resolve(k) in current_roots and self.resolve(k) not in self.ambiguous_current_roots]
        return dict(canonical_by_track={k: self.resolve(k) for k in self.parent}, aliases=aliases,
                    retired_state_ids=retired, unique_group_hits={k: len(self.groups[k]) for k in current_roots},
                    ambiguous_current_roots=sorted(self.ambiguous_current_roots),
                    witnesses=witnesses, settings=asdict(self.settings))

    def canonicalize_predictions(self, predictions, current_support):
        """一个canonical只输出一个原状态；选择当帧证据更足者，不合成位置/box。"""
        groups = {}
        for index, prediction in enumerate(predictions):
            node_id = prediction['node_id']
            root = self.resolve(node_id)
            if root in self.ambiguous_current_roots:
                root = node_id  # 当帧冲突不合并输出/不回收，保留原证据供关联层处理。
            score = (int(node_id in current_support), current_support.get(node_id, 0),
                     int(prediction.get('position_source') == 'reliable_current_dense_lidar'), -index)
            if root not in groups or score > groups[root][0]:
                groups[root] = (score, prediction)
        output, representative = [], {}
        for root, (_, pred) in groups.items():
            copied = copy.deepcopy(pred)
            copied['node_id'] = root
            output.append(copied)
            representative[root] = pred['node_id']
        return output, representative
