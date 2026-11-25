"""P1：只在本组实测弱侧向冲突时使用front位置辅助，保留冻结L4。"""
from __future__ import annotations

from conceptgraph.slam.mvrepair_20261123_combined import (
    EXTRA_SOURCES as COMBINED_SOURCES,
    CanonicalPhysicalFrontMapper,
    UnifiedSettings,
)
from conceptgraph.slam.mvrepair_20261123_front_position import position_decision
from conceptgraph.slam.mvrepair_lifecycle_20261123_mapper import LifecycleRetiringMapper


EXTRA_SOURCES = [*COMBINED_SOURCES,
    'conceptgraph/slam/mvrepair_20261123_combined.py',
    'tests/mvrepair_20261123_test_conflict_position.py']


def conflict_position_decision(front, current, timestamp, settings):
    if not current:
        return False, 'no_current_side_measurement_preserve_original'
    if any(s.get('_last_camera') == 'CAM_FRONT' for s in current):
        return False, 'current_front_native_position_preserved'
    if any(s.get('_last_reliable_dense', False) for s in current):
        return False, 'current_reliable_dense_preserved'
    if any(s.get('_last_observation_points', 0) not in range(1, 5) for s in current):
        return False, 'not_exclusively_current_sparse_side'
    if front is not None and timestamp < front['_capture_timestamp_s']:
        return False, 'future_front_forbidden'
    accepted, reason = position_decision(front, current, timestamp, settings)
    return accepted, 'confirmed_front_for_current_sparse_side_only' if accepted else reason


class ConflictOnlyFrontMapper(CanonicalPhysicalFrontMapper):
    def settings_record(self):
        return dict(**super().settings_record(), position_intervention=
            'P1: only actual current1-4point side measurement; preserve absent/currentfront/strongdense native positions; original numeric gates unchanged')

    def position_predictions(self, predictions, frame):
        # 从原L1 canonical输出开始，避免先改位置再按新条件恢复时漏掉时钟/方差字段。
        output = LifecycleRetiringMapper.position_predictions(self, predictions, frame)
        layer = self.position_layer
        merged = {}
        for node, state in layer.front_states.items():
            root = self.lifecycle.resolve(node)
            if root not in merged:
                merged[root] = state
                continue
            old = merged[root]
            keep, other = (state, old) if state['_capture_timestamp_s'] > old['_capture_timestamp_s'] else (old, state)
            keep['_groups'] |= other['_groups']
            for group, points in other['_points_by_group'].items():
                keep['_points_by_group'][group] = max(keep['_points_by_group'].get(group, 0), points)
            keep['hits'], keep['total_points'] = len(keep['_groups']), sum(keep['_points_by_group'].values())
            merged[root] = keep
        layer.front_states = merged
        states = [*layer.tracks, *layer.dense_states.values()]
        decisions = []
        for obj in output:
            root, before = obj['node_id'], obj['center']
            front = merged.get(root)
            current = [s for s in states if self.lifecycle.resolve(layer.aliases.get(s['node_id'], s['node_id'])) == root and s['timestamp'] == frame.timestamp_s]
            use, reason = conflict_position_decision(front, current, frame.timestamp_s, self.front_settings)
            evidence = dict(canonical=root, used=use, reason=reason, before=before,
                current_measurements=[dict(raw_state_id=s['node_id'], camera=s.get('_last_camera'),
                    points=s.get('_last_observation_points'), reliable=s.get('_last_reliable_dense'),
                    source_key=s['_observation_keys'][-1], variance=s['variance'].tolist()) for s in current])
            if use:
                age = frame.timestamp_s-front['_capture_timestamp_s']
                obj.update(center=front['center'].tolist(), position_covariance_diag=front['variance'].tolist(),
                    position_source='canonical_front_over_current_sparse_side', timestamp_s=front['_capture_timestamp_s'],
                    prediction_age_s=age, front_position_source_key=front['_source_key'])
                evidence.update(after=obj['center'], source_key=front['_source_key'],
                    actual_capture_timestamp_s=front['_capture_timestamp_s'], age_s=age,
                    groups=sorted(front['_groups']), front_real_points_total=front['total_points'],
                    front_variance=front['variance'].tolist())
            decisions.append(evidence)
        self.front_audit = dict(observations=layer.front_events, association=layer.front_pair_audit,
            position_decisions=decisions)
        return output


__all__ = ['ConflictOnlyFrontMapper', 'UnifiedSettings']
