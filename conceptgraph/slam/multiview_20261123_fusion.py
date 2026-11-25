"""三视角小试新入口：逐相机测量、同组一次更新，保留R1确认/TTL。"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree

from conceptgraph.slam.dense_refinement import RefinedEvidenceMapper
from conceptgraph.slam.slam_classes import DetectionList
from conceptgraph.slam.sparse_observations import cosine, measurement_variance


@dataclass(frozen=True)
class FusionSettings:
    maximum_xy_distance_m: float = 1.5
    maximum_z_distance_m: float = 1.5
    minimum_feature_cosine: float = .8
    minimum_points: int = 5
    minimum_confidence: float = .2
    maximum_xy_variance_m2: float = 4.
    maximum_capture_gap_s: float = .075
    shared_point_radius_m: float = .2
    minimum_shared_support: float = .2


def in_view(center, view):
    camera = np.asarray(view['camera_pose'])[:3, :3].T @ (
        np.asarray(center)-np.asarray(view['camera_pose'])[:3, 3])
    if not 1 < camera[2] < 65:
        return False
    uv = np.asarray(view['intrinsic']) @ camera
    return bool(0 <= uv[0]/uv[2] < view['width'] and 0 <= uv[1]/uv[2] < view['height'])


def pair_evidence(a, b, settings):
    """只有不同相机、可靠实测共享支持可合并，不读取框或GT。"""
    delta = np.asarray(a['center'])-np.asarray(b['center'])
    gap = abs(a['camera_timestamp_s']-b['camera_timestamp_s'])
    result = dict(camera_a=a['camera'], camera_b=b['camera'], category_a=a['category'],
                  category_b=b['category'], xy_distance_m=float(np.linalg.norm(delta[:2])),
                  z_distance_m=float(abs(delta[2])), capture_gap_s=gap,
                  feature_cosine=cosine(a['feature'], b['feature']), shared_support=None)
    reason = None
    if a['camera'] == b['camera']:
        reason = 'same_camera'
    elif a['category'] != b['category']:
        reason = 'different_class'
    elif min(a['points'], b['points']) < settings.minimum_points:
        reason = 'insufficient_measured_points'
    elif min(a['confidence'], b['confidence']) < settings.minimum_confidence:
        reason = 'low_confidence'
    elif max(np.max(a['variance'][:2]), np.max(b['variance'][:2])) > settings.maximum_xy_variance_m2:
        reason = 'uncertain_position'
    elif gap > settings.maximum_capture_gap_s:
        reason = 'capture_gap'
    elif result['xy_distance_m'] > settings.maximum_xy_distance_m or result['z_distance_m'] > settings.maximum_z_distance_m:
        reason = 'position_gate'
    elif result['feature_cosine'] < settings.minimum_feature_cosine:
        reason = 'appearance_gate'
    else:
        aa, bb = np.asarray(a['world_points']), np.asarray(b['world_points'])
        small, large = (aa, bb) if len(aa) <= len(bb) else (bb, aa)
        result['shared_support'] = float(np.mean(cKDTree(large).query(small)[0] <= settings.shared_point_radius_m))
        if result['shared_support'] < settings.minimum_shared_support:
            reason = 'no_shared_measured_support'
    result.update(eligible=reason is None, reason=reason or 'shared_measured_support')
    return result


def deduplicate(observations, settings=None):
    """完整链接簇、每相机至多一个观测；只保留一个代表，不叠点或证据次数。"""
    settings = settings or FusionSettings()
    pairs = {}
    for i, a in enumerate(observations):
        for j in range(i):
            pairs[j, i] = dict(index_a=j, index_b=i, **pair_evidence(observations[j], a, settings))
    order = sorted(range(len(observations)), key=lambda i: (
        -int(observations[i]['reliable']), -observations[i]['points'], -observations[i]['confidence'], i))
    clusters = []
    for i in order:
        compatible = [c for c in clusters if all(pairs[min(i, j), max(i, j)]['eligible'] for j in c)]
        if compatible:
            compatible[0].append(i)
        else:
            clusters.append([i])
    return [c[0] for c in clusters], dict(pairs=list(pairs.values()),
        clusters=[dict(representative=c[0], members=c, cameras=[observations[i]['camera'] for i in c]) for c in clusters],
        input_observations=len(observations), output_observations=len(clusters), removed=len(observations)-len(clusters))


class GroupedR1Mapper(RefinedEvidenceMapper):
    """R1派生适配，明确不声称逐字段等于历史单前R1。"""
    def __init__(self, cfg, classes, *, deduplication):
        super().__init__(cfg, classes)
        self.deduplication = deduplication
        self.fusion_settings = FusionSettings()
        self.group_views = []
        self.crossview_audit = {}
        self.measurement_sources = {}

    def settings_record(self):
        return dict(crossview=asdict(self.fusion_settings), sparse=self.position_layer.settings.to_dict(),
                    dense=self.refinement_settings.to_dict(),
                    group_update='once per reference timestamp; same image_idx for all cameras',
                    depth_merge='select representative; no point concatenation or extra confirmation',
                    motion='original R1; camera-time offsets audited, no object-motion compensation')

    def update_group(self, frames):
        if not frames or len({f.frame_idx for _, f in frames}) != 1:
            raise ValueError('每组必须同一frame_idx')
        reference = frames[0][1]
        if len({f.timestamp_s for _, f in frames}) != 1:
            raise ValueError('地图时钟必须统一到组参考时刻')
        self.group_frames = frames
        return super().update(reference)

    def extract_detections(self, unused):
        dense, sparse, audit = [], [], []
        self.measurement_sources = {}
        descriptors = []
        self.group_views = []
        for camera, frame in self.group_frames:
            view = dict(camera_pose=frame.camera_pose, intrinsic=frame.cam_K,
                        width=frame.image_rgb.shape[1], height=frame.image_rgb.shape[0])
            self.group_views.append(view)
            fg, bg = super().extract_detections(frame)
            if bg:
                raise ValueError('本小试只支持skip_bg=True')
            camera_time = self.capture_times[camera]
            for item in self.frame_audit:
                audit.append(dict(camera=camera, camera_timestamp_s=camera_time, **item))
            for obj in fg:
                points = np.asarray(obj['pcd'].points)
                center = np.asarray(obj['bbox'].get_center())
                variance = measurement_variance(points, range_m=float(np.linalg.norm(center-frame.camera_pose[:3, 3])))
                descriptor = dict(camera=camera, camera_timestamp_s=camera_time,
                    category=obj['class_name'][-1], center=center, feature=obj['clip_ft'].detach().cpu().numpy(),
                    points=len(points), confidence=float(obj['conf'][-1]), variance=variance,
                    world_points=points, reliable=len(points) >= 8 and float(obj['conf'][-1]) >= .2 and np.max(variance[:2]) <= 4.)
                obj['mask_idx'] = [len(dense)]
                self.measurement_sources[len(dense)] = descriptor
                dense.append(obj)
                descriptors.append(descriptor)
            for observation in self.sparse_measurements:
                observation['camera'] = camera
                sparse.append(observation)
        self.frame_audit = audit
        if self.deduplication:
            kept, self.crossview_audit = deduplicate(descriptors, self.fusion_settings)
        else:
            kept = list(range(len(dense)))
            self.crossview_audit = dict(input_observations=len(dense), output_observations=len(dense), removed=0,
                pairs=[], clusters=[dict(representative=i, members=[i], cameras=[descriptors[i]['camera']]) for i in kept])
        self.crossview_audit['sparse_observations_not_deduplicated'] = len(sparse)
        self.crossview_audit['camera_capture_times_s'] = self.capture_times
        self.crossview_audit['reference_timestamp_s'] = unused.timestamp_s
        self.sparse_measurements = sparse
        return DetectionList([dense[i] for i in kept]), DetectionList()

    def position_predictions(self, predictions, frame):
        current = {obj['mask_idx'][-1]: obj for obj in self.last_observations}
        measurements, ids = {}, {p['node_id'] for p in predictions}
        for obj in self.objects:
            node_id = obj['stable_object_id']
            if node_id not in ids or obj['image_idx'][-1] != frame.frame_idx:
                continue
            observed = current[obj['mask_idx'][-1]]
            source = self.measurement_sources[obj['mask_idx'][-1]]
            measurements[node_id] = dict(center=np.asarray(obj['center_history'][-1]), category=source['category'],
                feature=observed['clip_ft'].detach().cpu().numpy(), points=source['points'],
                confidence=source['confidence'], variance=source['variance'],
                reliable_current_dense=source['reliable'], maximum_speed_m_s=self.refinement_settings.maximum_speed_m_s)
        output = self.position_layer.update(self.sparse_measurements, predictions, measurements, frame.timestamp_s)
        protected_ids = {self.position_layer.aliases.get(k, k) for k, v in measurements.items() if v['reliable_current_dense']}
        kept = []
        for obj in output:
            if obj['node_id'] in protected_ids:
                obj['position_source'] = 'reliable_current_dense_lidar'
            if any(in_view(obj['center'], view) for view in self.group_views):
                kept.append(obj)
        self.frame_audit.append(dict(stage='position_layer', reason='group_snapshot',
            sparse_measurements=len(self.sparse_measurements), sparse_hypotheses=len(self.position_layer.tracks),
            outputs=len(kept), reliable_dense_measurements=len(protected_ids)))
        return copy.deepcopy(kept)
