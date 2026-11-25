"""跨视角错合并、坐标和同组证据计数的必要风险测试。"""
import copy
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from conceptgraph.slam.multiview_20261123_fusion import (
    GroupedR1Mapper, deduplicate, in_view, pair_evidence, FusionSettings,
)
from conceptgraph.slam.sparse_observations import measured_points
from conceptgraph.slam.streamlined_mapping import MappingFrame


def observation(camera, *, shift=0., confidence=.5, timestamp=1.):
    points = np.array([[shift+x, .1*y, 10.+.1*z] for x in [0., .1] for y in [0, 1] for z in [0, 1]])
    return dict(camera=camera, camera_timestamp_s=timestamp, category='car', center=points.mean(axis=0),
                feature=np.array([1., 0.]), points=len(points), confidence=confidence,
                variance=np.ones(3), world_points=points, reliable=True)


def test_same_world_support_selects_one_representative_without_camera_double_count():
    observations = [observation(c) for c in ['front', 'left', 'right']]
    kept, audit = deduplicate(observations)
    assert kept == [0] and audit['removed'] == 2
    assert observations[0]['points'] == 8
    same_camera = [observation('front'), observation('front')]
    assert len(deduplicate(same_camera)[0]) == 2


def test_nearby_distinct_support_and_time_class_mismatch_are_not_merged():
    a = observation('front')
    b = observation('left', shift=.7)
    assert not pair_evidence(a, b, FusionSettings())['eligible']
    b = observation('left', timestamp=1.2)
    assert pair_evidence(a, b, FusionSettings())['reason'] == 'capture_gap'
    b = observation('left')
    b['category'] = 'pedestrian'
    assert pair_evidence(a, b, FusionSettings())['reason'] == 'different_class'


def test_camera_pose_is_applied_before_world_fusion_and_fov_is_camera_specific():
    k = np.array([[10., 0., 10.], [0., 10., 10.], [0., 0., 1.]])
    depth, mask = np.zeros((20, 20)), np.zeros((20, 20), dtype=bool)
    depth[10, 10] = 10.
    mask[10, 10] = True
    pose = np.eye(4)
    pose[:3, 3] = [20., 3., 0.]
    point = measured_points(depth, mask, k, pose)[0]
    np.testing.assert_allclose(point, [20., 3., 10.])
    view = dict(intrinsic=k, camera_pose=pose, width=20, height=20)
    assert in_view(point, view)
    assert not in_view(point, dict(view, camera_pose=np.eye(4)))


def test_three_sparse_cameras_do_not_confirm_in_one_timestamp():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'outputs/results/nuscenes/validation/iterations/recall_refinement_R1_full_20261120_v2/release.json').read_text())['config']
    mapper = GroupedR1Mapper(OmegaConf.create(config), ['car'], deduplication=True)
    k = np.array([[10., 0., 10.], [0., 10., 10.], [0., 0., 1.]])
    depth = np.zeros((20, 20))
    depth[10, 10], depth[10, 11] = 10., 10.
    mask = np.zeros((1, 20, 20), dtype=bool)
    mask[:, 5:15, 5:15] = True
    gobs = dict(xyxy=np.array([[5., 5., 15., 15.]]), mask=mask,
                class_id=np.array([0]), classes=['car'], confidence=np.array([.3]),
                image_feats=np.array([[1., 0.]]), text_feats=np.array([[1., 0.]]))
    outputs = []
    for idx in range(2):
        timestamp = 1.+idx*.5
        mapper.capture_times = dict.fromkeys(['front', 'left', 'right'], timestamp)
        frames = [(c, MappingFrame(idx, np.zeros((20, 20, 3), dtype=np.uint8), depth, k,
            np.eye(4), copy.deepcopy(gobs), 'synthetic-risk-test', timestamp_s=timestamp)) for c in mapper.capture_times]
        mapper.update_group(frames)
        outputs.append(mapper.position_predictions([], frames[0][1]))
    assert outputs[0] == []
    assert len(outputs[1]) == 1  # original position layer suppresses nearby confirmed sparse outputs
    assert len(mapper.position_layer.tracks) == 3
    assert all(t['hits'] == 2 and t['total_points'] == 4 for t in mapper.position_layer.tracks)
    assert all(p['num_detections'] == 2 and p['measured_points_total'] == 4 for p in outputs[1])
