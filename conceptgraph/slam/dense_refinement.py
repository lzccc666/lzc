"""下一轮显式配置：保护可靠的当帧稠密测量，保留原B1实现。"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from conceptgraph.slam.quality_mapper import QualityEvidenceMapper
from conceptgraph.slam.sparse_observations import SparsePositionLayer, measurement_variance


@dataclass(frozen=True)
class DenseRefinementSettings:
    minimum_points: int = 8
    minimum_confidence: float = .2
    maximum_xy_variance_m2: float = 4.
    maximum_speed_m_s: float = 20.

    def to_dict(self):
        return asdict(self)


class DenseProtectedPositionLayer(SparsePositionLayer):
    """可靠当帧测量直接校正，弱测量继续采用B1滤波。"""
    def _update_state(self, state, observation, timestamp):
        if not observation.get('reliable_current_dense', False):
            return super()._update_state(state, observation, timestamp)
        center = np.asarray(observation['center'], dtype=float)
        dt = timestamp - state['timestamp']
        velocity = np.zeros(3)
        if dt >= .25:
            velocity = (center - state['last_measured_center']) / dt
            if np.linalg.norm(velocity[:2]) > observation['maximum_speed_m_s']:
                velocity = np.zeros(3)
        state.update(center=center.copy(), variance=np.asarray(observation['variance']).copy(),
            velocity=velocity, timestamp=timestamp, feature=np.asarray(observation['feature']),
            last_measured_center=center.copy(), total_points=state['total_points'] + observation['points'],
            confidence=max(state['confidence'], observation['confidence']), hits=state['hits'] + 1)


class RefinedEvidenceMapper(QualityEvidenceMapper):
    def __init__(self, cfg, classes, *, condition='R1'):
        if condition != 'R1':
            raise ValueError('当前只实现R1；未实现配置不得执行')
        super().__init__(cfg, classes, module_a=True)
        self.refinement_settings = DenseRefinementSettings()
        self.position_layer = DenseProtectedPositionLayer()

    def position_predictions(self, predictions, frame):
        measurements = {}
        ids = {obj['node_id'] for obj in predictions}
        # mask_idx来自过滤后的同一帧检测，不能用融合后的历史点云估计当帧误差。
        current = {obj['mask_idx'][-1]: obj for obj in self.last_observations}
        settings = self.refinement_settings
        for obj in self.objects:
            node_id = obj['stable_object_id']
            if node_id not in ids or obj['image_idx'][-1] != frame.frame_idx:
                continue
            observed = current[obj['mask_idx'][-1]]
            center = np.asarray(obj['center_history'][-1], dtype=float)
            points = np.asarray(observed['pcd'].points)
            distance = float(np.linalg.norm(center - frame.camera_pose[:3, 3]))
            variance = measurement_variance(points, range_m=distance)
            unique, counts = np.unique(np.asarray(obj['class_id'], dtype=int), return_counts=True)
            confidence, support = float(obj['conf'][-1]), int(obj['n_points'][-1])
            reliable = (support >= settings.minimum_points and confidence >= settings.minimum_confidence
                        and np.max(variance[:2]) <= settings.maximum_xy_variance_m2)
            measurements[node_id] = dict(center=center, category=self.classes[int(unique[np.argmax(counts)])],
                feature=observed['clip_ft'].detach().cpu().numpy(), points=support, confidence=confidence,
                variance=variance, reliable_current_dense=reliable,
                maximum_speed_m_s=settings.maximum_speed_m_s)
        output = self.position_layer.update(self.sparse_measurements, predictions, measurements, frame.timestamp_s)
        protected_ids = {self.position_layer.aliases.get(k, k) for k, v in measurements.items()
                         if v['reliable_current_dense']}
        kept = []
        for obj in output:
            if obj['node_id'] in protected_ids:
                obj['position_source'] = 'reliable_current_dense_lidar'
            camera = frame.camera_pose[:3, :3].T @ (np.asarray(obj['center']) - frame.camera_pose[:3, 3])
            uv = frame.cam_K @ camera
            if 1 < camera[2] < 65 and 0 <= uv[0] / uv[2] < 1600 and 0 <= uv[1] / uv[2] < 900:
                kept.append(obj)
        self.frame_audit.append(dict(stage='position_layer', reason='snapshot',
            sparse_measurements=len(self.sparse_measurements), sparse_hypotheses=len(self.position_layer.tracks),
            sparse_outputs=sum(o.get('position_source') == 'confirmed_sparse_lidar' for o in kept),
            reliable_dense_measurements=len(protected_ids),
            reliable_dense_outputs=sum(o['node_id'] in protected_ids for o in kept)))
        return kept
