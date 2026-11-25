"""根据过去位置和真实时间差约束关联；不访问官方标签或未来观测。"""
from __future__ import annotations

import numpy as np
import torch

from conceptgraph.slam.evidence_mapper import EvidenceAwareMapper
from conceptgraph.slam.utils import get_object_center


def extrapolated_center(obj, timestamp_s, *, horizon_s=2.0, maximum_speed_mps=20.0):
    """仅最近有效历史可外推；不可靠历史退回当前几何，保留明确诊断。"""
    fallback = np.asarray(get_object_center(obj), dtype=float).copy()
    centers = np.asarray(obj.get("center_history", []), dtype=float)
    times = np.asarray(obj.get("timestamp_history_s", []), dtype=float)
    if (timestamp_s is None or not np.isfinite(timestamp_s) or len(times) < 2
            or centers.shape != (len(times), 3) or not np.isfinite(centers).all()
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        return fallback, 0.0, "insufficient_monotonic_history"
    age = float(timestamp_s - times[-1])
    if age <= 0 or age > horizon_s:
        return fallback, max(0.0, age), "outside_prediction_horizon"
    vectors = []
    for i in range(max(0, len(times) - 4), len(times) - 1):
        dt = float(times[-1] - times[i])
        if dt >= 0.25:
            vectors.append((centers[-1, :2] - centers[i, :2]) / dt)
    if not vectors:
        return fallback, age, "insufficient_time_span"
    velocity = np.median(vectors, axis=0)
    speed = float(np.linalg.norm(velocity))
    if speed > maximum_speed_mps:
        return fallback, age, "implausible_velocity"
    # 静态/近静态历史不外推深度噪声。
    velocity = velocity if speed > 0.5 else np.zeros(2)
    center = centers[-1].copy()
    center[:2] += velocity * age
    return center, age, "past_observations_only"


def motion_scores(detections, objects, spatial, visual, timestamp_s, *, base_gate_m=2.5):
    """同类、有限运动范围内综合重合、外观与预测位置；一对一分配沿用原实现。"""
    result = np.full((len(detections), len(objects)), -np.inf)
    for j, obj in enumerate(objects):
        center, age, _reason = extrapolated_center(obj, timestamp_s)
        gate = min(6.0, base_gate_m + 0.75 * age)
        classes = set(np.atleast_1d(obj["class_id"]).tolist())
        for i, det in enumerate(detections):
            if classes.isdisjoint(np.atleast_1d(det["class_id"]).tolist()):
                continue
            distance = float(np.linalg.norm((get_object_center(det) - center)[:2]))
            if not np.isfinite(distance) or distance > gate:
                continue
            proximity = max(0.0, 1.0 - distance / gate)
            result[i, j] = (0.5 * np.clip(spatial[i, j], 0, 1)
                            + 0.25 * np.clip(visual[i, j], 0, 1) + 0.25 * proximity)
    return result


class MotionAssociationMapper(EvidenceAwareMapper):
    """显式选择的实验方案，默认建图入口不启用。"""

    def refine_association_scores(self, frame, detections, spatial_sim, visual_sim, scores):
        refined = motion_scores(
            detections, self.objects, spatial_sim.detach().cpu().numpy(),
            visual_sim.detach().cpu().numpy(), frame.timestamp_s,
            base_gate_m=float(getattr(self.cfg, "motion_association_gate_m", 2.5)),
        )
        return torch.as_tensor(refined, dtype=scores.dtype, device=scores.device)
