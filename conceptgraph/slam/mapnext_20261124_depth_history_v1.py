"""Candidate H1: depth history needs a unique past physical mask witness.

No GT, new detections, thresholds for scoring, TTL or confirmation changes.
"""
import numpy as np

from conceptgraph.slam.mapnext_20261124_mapper import AuditedNativeMapper
from conceptgraph.slam.mvrepair_20261123_partial_fragment import projected_support
from conceptgraph.slam.sparse_observations import cosine, select_depth_cluster

EXTRA_SOURCES = ['conceptgraph/slam/mapnext_20261124_mapper.py']


class UniqueHistoryMapper(AuditedNativeMapper):
    def settings_record(self):
        record = super().settings_record()
        record['mapnext_depth_history'] = dict(version='H1', previous_reliable_dense_only=True,
            original_age_s=1.5, original_appearance=.65, minimum_projected_support=.5,
            unique_canonical=True, ambiguous_history='no_history_prior',
            point_source='previous real measured points, ego projection only')
        return record

    def observation_policy(self, frame):
        if not self.module_a:
            return None
        def select(depth, mask, gobs, index):
            feature = gobs['image_feats'][index]
            category = gobs['classes'][gobs['class_id'][index]]
            box = gobs['xyxy'][index]
            candidates = {}
            for track in [*self.position_layer.tracks, *self.position_layer.dense_states.values()]:
                if (track['category'] != category or not track.get('_last_reliable_dense', False)
                        or not 0 < frame.timestamp_s-track['timestamp'] <= 1.5
                        or cosine(feature, track['feature']) < .65):
                    continue
                camera = frame.camera_pose[:3, :3].T @ (track['center']-frame.camera_pose[:3, 3])
                uv = frame.cam_K @ camera
                if not (camera[2] > 1 and box[0] <= uv[0]/uv[2] <= box[2]
                        and box[1] <= uv[1]/uv[2] <= box[3]):
                    continue
                points = track.get('_last_world_points', [])
                witness = projected_support(points, dict(pose=frame.camera_pose, intrinsic=frame.cam_K), mask)
                if witness['fraction'] < .5:
                    continue
                raw = self.position_layer.aliases.get(track['node_id'], track['node_id'])
                root = self.lifecycle.resolve(raw)
                candidates.setdefault(root, []).append(float(camera[2]))
            histories = next(iter(candidates.values())) if len(candidates) == 1 else []
            selected, evidence = select_depth_cluster(depth, mask, history_depths=histories,
                                                     settings=self.quality_settings)
            evidence['mapnext_history'] = dict(unique_physical_candidates=len(candidates),
                histories_m=histories, policy='H1_reliable_measured_mask_witness')
            return selected, evidence
        return select
