"""独立L1候选：现有R1坐标/关联不变，只在实测见证后收束身份与旧状态。"""
from __future__ import annotations

import copy

import numpy as np

from conceptgraph.slam.mviter_20261123_diagnostics import SourceAuditedMapper
from conceptgraph.slam.mvrepair_lifecycle_20261123_registry import CanonicalLifecycle
from conceptgraph.slam.slam_classes import MapObjectList


class LifecycleRetiringMapper(SourceAuditedMapper):
    def __init__(self, cfg, classes):
        super().__init__(cfg, classes, deduplication=True)
        self.lifecycle = CanonicalLifecycle()
        self.lifecycle_audit = {}

    def position_predictions(self, predictions, frame):
        raw = super().position_predictions(predictions, frame)
        layer = self.position_layer
        states = [*layer.tracks, *layer.dense_states.values()]
        records, current_support, raw_to_logical = [], {}, {}
        for state in states:
            raw_id = state['node_id']
            logical = layer.aliases.get(raw_id, raw_id)
            raw_to_logical[raw_id] = logical
            if state['timestamp'] != frame.timestamp_s:
                continue
            keys = state.get('_observation_keys', [])
            if not keys:
                continue
            source_key = keys[-1]
            points = np.asarray(state.get('_last_world_points', [])).reshape(-1, 3)
            records.append(dict(track_id=logical, source_key=source_key, category=state['category'],
                center=state['last_measured_center'], feature=state['feature'], real_points=points,
                capture_timestamp_s=float(source_key[1])))
            current_support[logical] = max(current_support.get(logical, 0), len(points))
        audit = self.lifecycle.observe_group(frame.frame_idx, frame.timestamp_s, records)
        output, representatives = self.lifecycle.canonicalize_predictions(raw, current_support)
        # 仅当前簇已经由实际共同观测见证归并时，回收簇内重复原state；canonical不随keeper变化。
        by_root = {}
        for state in states:
            raw_id = state['node_id']
            logical = raw_to_logical[raw_id]
            root = self.lifecycle.resolve(logical)
            by_root.setdefault(root, []).append(state)
        retired = []
        for root, members in by_root.items():
            if root in self.lifecycle.ambiguous_current_roots:
                continue
            current = [s for s in members if s['timestamp'] == frame.timestamp_s]
            if len(members) < 2 or not current:
                continue
            # 一个簇可能同时包含同canonical稠密与稀疏状态；不把两次计数相加。
            member_order = {id(s): i for i, s in enumerate(members)}
            keeper = max(current, key=lambda s: (len(s.get('_last_world_points', [])),
                int(s['node_id'] in layer.dense_states), -member_order[id(s)]))
            for state in members:
                if state is keeper:
                    continue
                retired.append(dict(raw_state_id=state['node_id'], logical_id=raw_to_logical[state['node_id']],
                    canonical_id=root, covered_by_raw_state_id=keeper['node_id'],
                    last_measured_timestamp_s=state['timestamp'],
                    reason='persistent_canonical_has_current_real_observation'))
        retired_ids = {r['raw_state_id'] for r in retired}
        self.objects = MapObjectList([o for o in self.objects if o['stable_object_id'] not in retired_ids])
        layer.dense_states = {k: v for k, v in layer.dense_states.items() if k not in retired_ids}
        layer.tracks = [s for s in layer.tracks if s['node_id'] not in retired_ids]
        # 明列位置测量时钟与旧dense对象时钟；未实测state绝不刷新到now。
        for prediction in output:
            root = prediction['node_id']
            members = by_root.get(root, [])
            source = max(members, key=lambda s: s['timestamp']) if members else None
            if source is not None:
                prediction['dense_object_timestamp_s'] = prediction.get('timestamp_s')
                prediction['state_last_measurement_timestamp_s'] = source['timestamp']
                prediction['render_timestamp_s'] = frame.timestamp_s
                prediction['state_measurement_age_s'] = frame.timestamp_s-source['timestamp']
        self.lifecycle_audit = dict(registry=audit, retired_states=retired, output_representatives=representatives,
            raw_outputs=copy.deepcopy(raw), raw_output_count=len(raw), canonical_output_count=len(output),
            current_observation_count=len(records), GT_in_inference=False)
        return output
