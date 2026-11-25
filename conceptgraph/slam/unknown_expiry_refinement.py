"""显式验证未知运动状态的过期策略；原R1/R3及其默认寿命不变。"""
from __future__ import annotations

import numpy as np

from conceptgraph.slam.dense_refinement import RefinedEvidenceMapper
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.slam.uncertainty_refinement import CombinedRefinementMapper


class UnknownPotentialDynamicExpiryMixin:
    """unknown不是dynamic：仅给未证实静止的潜在动态节点原有TTL。"""
    def update(self, frame):
        expired = []
        if frame.timestamp_s is not None:
            timestamp = float(frame.timestamp_s)
            if (not np.isfinite(timestamp) or (self._last_timestamp_s is not None
                    and timestamp <= self._last_timestamp_s)):
                raise ValueError('mapping requires strictly increasing finite timestamps')
            ttl = float(getattr(self.cfg, 'dynamic_ttl_s', 2.))
            kept = []
            for obj in self.objects:
                observed = obj.get('timestamp_s')
                if (obj.get('is_potential_dynamic', False)
                        and obj.get('motion_state') == 'unknown' and observed is not None
                        and timestamp - float(observed) > ttl):
                    expired.append(obj['stable_object_id'])
                else:
                    kept.append(obj)
            self.objects = MapObjectList(kept, device=self.cfg.device)
        result = super().update(frame)
        self.frame_audit.append(dict(stage='unknown_expiry', reason='same_original_ttl',
            ttl_s=float(getattr(self.cfg, 'dynamic_ttl_s', 2.)), expired_node_ids=expired,
            expired_unknown_potential_dynamic=len(expired)))
        return result


class UnknownExpiryR1Mapper(UnknownPotentialDynamicExpiryMixin, RefinedEvidenceMapper):
    pass


class UnknownExpiryR3Mapper(UnknownPotentialDynamicExpiryMixin, CombinedRefinementMapper):
    pass


def create_unknown_expiry_mapper(cfg, classes, *, condition):
    if condition == 'R1_unknown_ttl':
        return UnknownExpiryR1Mapper(cfg, classes)
    if condition == 'R3_unknown_ttl':
        return UnknownExpiryR3Mapper(cfg, classes)
    raise ValueError('未知的显式过期实验配置')
