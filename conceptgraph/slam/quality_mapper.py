"""配置对照的显式实验入口；默认地图保持原算法。"""
from __future__ import annotations

import numpy as np

from conceptgraph.slam.audited_mapper import AuditedEvidenceMapper
from conceptgraph.slam.sparse_observations import (
    SparsePositionLayer, SparseQualitySettings, cosine, measured_points,
    measurement_variance, select_depth_cluster,
)


SETTINGS = {'B1': dict(module_a=True, module_b=False),
            'B2': dict(module_a=False, module_b=True),
            'B3': dict(module_a=True, module_b=True)}


class QualityEvidenceMapper(AuditedEvidenceMapper):
    def __init__(self, cfg, classes, *, module_a=False, module_b=False):
        super().__init__(cfg, classes)
        self.module_a, self.module_b = module_a, module_b
        self.position_layer = SparsePositionLayer()
        self.quality_settings = SparseQualitySettings()
        self.sparse_measurements = []

    def extract_detections(self, frame):
        self.sparse_measurements = []
        return super().extract_detections(frame)

    def observation_policy(self, frame):
        if not self.module_a:
            return None
        def select(depth, mask, gobs, index):
            feature = gobs['image_feats'][index]
            category = gobs['classes'][gobs['class_id'][index]]
            histories = []
            candidates = list(self.position_layer.tracks) + list(self.position_layer.dense_states.values())
            box = gobs['xyxy'][index]
            for track in candidates:
                if (track['category'] != category or frame.timestamp_s - track['timestamp'] > 1.5
                        or cosine(feature, track['feature']) < .65):
                    continue
                camera = frame.camera_pose[:3, :3].T @ (track['center'] - frame.camera_pose[:3, 3])
                uv = frame.cam_K @ camera
                if camera[2] > 1 and box[0] <= uv[0] / uv[2] <= box[2] and box[1] <= uv[1] / uv[2] <= box[3]:
                    histories.append(float(camera[2]))
            return select_depth_cluster(depth, mask, history_depths=histories, settings=self.quality_settings)
        return select

    def sparse_callback(self, frame):
        if not self.module_a:
            return None
        def save(depth, mask, gobs, index, evidence):
            points = measured_points(depth, mask, frame.cam_K, frame.camera_pose)
            if not len(points):
                return
            center = np.median(points, axis=0)
            distance = float(np.linalg.norm(center - frame.camera_pose[:3, 3]))
            self.sparse_measurements.append(dict(center=center,
                category=gobs['classes'][gobs['class_id'][index]],
                feature=gobs['image_feats'][index], points=len(points),
                confidence=float(gobs['confidence'][index]),
                variance=measurement_variance(points, range_m=distance)))
        return save

    def position_predictions(self, predictions, frame):
        if not self.module_a:
            return predictions
        measurements = {}
        ids = {obj['node_id'] for obj in predictions}
        for obj in self.objects:
            node_id = obj['stable_object_id']
            if node_id not in ids or obj['image_idx'][-1] != frame.frame_idx:
                continue
            center = np.asarray(obj['center_history'][-1], dtype=float)
            points = np.asarray(obj['pcd'].points)
            distance = float(np.linalg.norm(center - frame.camera_pose[:3, 3]))
            class_ids = np.asarray(obj['class_id'], dtype=int)
            unique, counts = np.unique(class_ids, return_counts=True)
            measurements[node_id] = dict(center=center, category=self.classes[int(unique[np.argmax(counts)])],
                feature=obj['clip_ft'].detach().cpu().numpy(), points=int(obj['n_points'][-1]),
                confidence=float(obj['conf'][-1]), variance=measurement_variance(points, range_m=distance))
        output = self.position_layer.update(self.sparse_measurements, predictions, measurements, frame.timestamp_s)
        kept = []
        for obj in output:
            camera = frame.camera_pose[:3, :3].T @ (np.asarray(obj['center']) - frame.camera_pose[:3, 3])
            uv = frame.cam_K @ camera
            if 1 < camera[2] < 65 and 0 <= uv[0] / uv[2] < 1600 and 0 <= uv[1] / uv[2] < 900:
                kept.append(obj)
        self.frame_audit.append(dict(stage='position_layer', reason='snapshot',
            sparse_measurements=len(self.sparse_measurements), sparse_hypotheses=len(self.position_layer.tracks),
            sparse_outputs=sum(o.get('position_source') == 'confirmed_sparse_lidar' for o in kept)))
        return kept

    def merge_associated_detections(self, detections, scores):
        if not self.module_b:
            return super().merge_associated_detections(detections, scores)
        from conceptgraph.slam.uncertainty_association import merge_two_stage
        return merge_two_stage(self, detections, scores)
