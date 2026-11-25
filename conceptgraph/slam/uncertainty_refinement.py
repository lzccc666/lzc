"""按当帧证据分两轮关联，位置约束仅使用过去状态及真实测量。"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from conceptgraph.completion.association import gated_assignment
from conceptgraph.slam.dense_refinement import DenseRefinementSettings, RefinedEvidenceMapper
from conceptgraph.slam.motion_association_mapper import extrapolated_center
from conceptgraph.slam.quality_mapper import QualityEvidenceMapper
from conceptgraph.slam.sparse_observations import measurement_variance
from conceptgraph.slam.utils import get_object_center


@dataclass(frozen=True)
class UncertaintyAssociationSettings:
    appearance_minimum: float = .65
    maximum_mahalanobis_squared: float = 9.21
    physical_gate_m: float = 8.
    prediction_horizon_s: float = 2.
    process_variance_per_s: float = .5
    motion_uncertainty_m_s: float = .75
    overlap_weight: float = .25
    appearance_weight: float = .25
    proximity_weight: float = .5
    score_threshold: float = .35

    def to_dict(self):
        return asdict(self)


def two_stage_assignment(scores, strong):
    """先冻结高质量观测匹配，第二轮仅使用剩余观测和剩余节点。"""
    scores = np.asarray(scores, dtype=float)
    strong = np.asarray(strong, dtype=bool)
    if scores.ndim != 2 or strong.shape != (scores.shape[0],):
        raise ValueError('关联矩阵与观测质量标记形状不一致')
    assigned, stages = {}, {}
    free = list(range(scores.shape[1]))
    for stage, rows in [('reliable', np.flatnonzero(strong)), ('weak', np.flatnonzero(~strong))]:
        selected = gated_assignment(scores[np.ix_(rows, free)])
        for i, j in selected.items():
            assigned[int(rows[i])] = free[j]
            stages[int(rows[i])] = stage
        used = set(assigned.values())
        free = [j for j in free if j not in used]
    return assigned, stages


def uncertainty_scores(detections, objects, spatial, visual, frame, settings):
    scores = np.full((len(detections), len(objects)), -np.inf)
    reliable, evidence = [], {}
    quality = DenseRefinementSettings()
    centers = [get_object_center(det) for det in detections]
    variances = [measurement_variance(np.asarray(det['pcd'].points),
        range_m=float(np.linalg.norm(center - frame.camera_pose[:3, 3])))
        for det, center in zip(detections, centers)]
    for det, variance in zip(detections, variances):
        reliable.append(int(det['n_points'][-1]) >= quality.minimum_points
            and float(det['conf'][-1]) >= quality.minimum_confidence
            and np.max(variance[:2]) <= quality.maximum_xy_variance_m2)
    for j, obj in enumerate(objects):
        predicted, _age, reason = extrapolated_center(obj, frame.timestamp_s,
            horizon_s=settings.prediction_horizon_s)
        age = max(0., float(frame.timestamp_s - obj.get('timestamp_s', frame.timestamp_s)))
        # 超过可靠外推时域不继续放大协方差，位置退回已观测几何。
        elapsed = min(age, settings.prediction_horizon_s)
        previous = measurement_variance(np.asarray(obj['pcd'].points),
            range_m=float(np.linalg.norm(get_object_center(obj) - frame.camera_pose[:3, 3])))
        classes = set(np.atleast_1d(obj['class_id']).tolist())
        for i, det in enumerate(detections):
            if classes.isdisjoint(np.atleast_1d(det['class_id']).tolist()):
                continue
            physical = float(np.linalg.norm(centers[i] - get_object_center(obj)))
            if (not np.isfinite(physical) or not np.isfinite(visual[i, j])
                    or physical > settings.physical_gate_m or visual[i, j] < settings.appearance_minimum):
                continue
            covariance = (variances[i][:2] + previous[:2] + settings.process_variance_per_s * elapsed
                          + (settings.motion_uncertainty_m_s * elapsed)**2)
            delta = (centers[i] - predicted)[:2]
            distance = float(np.sum(delta**2 / covariance))
            if not np.isfinite(distance) or distance > settings.maximum_mahalanobis_squared:
                continue
            score = (settings.overlap_weight * np.clip(spatial[i, j], 0., 1.)
                     + settings.appearance_weight * np.clip(visual[i, j], 0., 1.)
                     + settings.proximity_weight * np.exp(-.5 * distance))
            if not np.isfinite(score) or score < settings.score_threshold:
                continue
            scores[i, j] = score
            evidence[i, j] = dict(mahalanobis_squared=distance, covariance_xy=covariance.tolist(),
                physical_distance_m=physical, past_age_s=age, prediction_reason=reason, score=float(score))
    return scores, reliable, evidence


class UncertaintyAssociationMixin:
    assignment_mode = 'two_stage'

    def refine_association_scores(self, frame, detections, spatial_sim, visual_sim, scores):
        settings = UncertaintyAssociationSettings()
        values, reliable, evidence = uncertainty_scores(detections, self.objects,
            spatial_sim.detach().cpu().numpy(), visual_sim.detach().cpu().numpy(), frame, settings)
        if self.assignment_mode == 'two_stage':
            assigned, stages = two_stage_assignment(values, reliable)
        elif self.assignment_mode == 'single_stage':
            assigned = gated_assignment(values)
            stages = {i: 'joint' for i in assigned}
        else:
            raise ValueError('未知关联分配模式')
        frozen = torch.full_like(scores, -torch.inf)
        for i, j in assigned.items():
            # 下游仅执行已冻结的一对一匹配，沿用历史几何更新和运动状态实现。
            frozen[i, j] = 1.
        for i, det in enumerate(detections):
            j = assigned.get(i)
            self.frame_audit.append(dict(stage='uncertainty_association',
                reason='matched_' + stages[i] if j is not None else 'new_unmatched',
                filtered_mask_index=int(det['mask_idx'][-1]), reliable=bool(reliable[i]),
                previous_node_id=self.objects[j]['stable_object_id'] if j is not None else None,
                evidence=evidence.get((i, j))))
        return frozen


class AssociationRefinedMapper(UncertaintyAssociationMixin, QualityEvidenceMapper):
    def __init__(self, cfg, classes):
        super().__init__(cfg, classes, module_a=True)


class CombinedRefinementMapper(UncertaintyAssociationMixin, RefinedEvidenceMapper):
    def __init__(self, cfg, classes):
        super().__init__(cfg, classes, condition='R1')


class CombinedSingleStageRefinementMapper(CombinedRefinementMapper):
    """机制消融：评分、门控和位置保护相同，仅取消高质量匹配优先级。"""
    assignment_mode = 'single_stage'


def create_refinement_mapper(cfg, classes, *, condition):
    if condition == 'R1':
        return RefinedEvidenceMapper(cfg, classes, condition='R1')
    if condition == 'R2':
        return AssociationRefinedMapper(cfg, classes)
    if condition == 'R3':
        return CombinedRefinementMapper(cfg, classes)
    if condition == 'R3_single_stage':
        return CombinedSingleStageRefinementMapper(cfg, classes)
    raise ValueError('未注册的继续优化配置')
