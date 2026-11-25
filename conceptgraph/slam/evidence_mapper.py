"""双置信度观测：保留弱检测以续接轨迹，多帧支持后再输出弱目标。"""
from __future__ import annotations

import numpy as np

from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.slam.streamlined_mapping import StreamlinedMapper


def confirmed_observation(obj, *, strong_confidence=0.2, minimum_observations=2):
    confidence = np.asarray(obj.get("conf", []), dtype=float)
    if not confidence.size or not np.isfinite(confidence).all():
        return False
    if float(confidence.max()) >= strong_confidence:
        return True
    # 仅有重复帧编号不算多次独立观测。
    observations = len(set(obj.get("image_idx", [])))
    supports = obj.get("depth_evidence", [])
    return observations >= minimum_observations and any(
        int(e.get("retained_pixels", 0)) >= 5 for e in supports)


class EvidenceAwareMapper(StreamlinedMapper):
    """地图内部保留待确认对象，正式对象接口执行独立的观测证据门槛。"""

    def confirmed_objects(self):
        return MapObjectList([
            obj for obj in self.objects if confirmed_observation(
                obj, strong_confidence=float(getattr(self.cfg, "strong_confidence", 0.2)),
                minimum_observations=int(getattr(self.cfg, "weak_confirmation_observations", 2)))
        ])
