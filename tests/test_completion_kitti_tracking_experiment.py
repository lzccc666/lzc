"""不同分辨率、有效GT分母、全拒绝和评价隔离的独立边界用例。"""
import copy
import json

import numpy as np
import pytest

from conceptgraph.completion.kitti_tracking_experiment import (
    METHODS, eligible_truth, in_view, load_agent, object_statistics,
)


def test_kitti_real_image_dimensions_and_camera_pose():
    pose = np.eye(4)
    pose[0, 3] = 100
    row = dict(pose=pose.tolist(), intrinsic=np.eye(3).tolist(), image_wh=[10, 4])
    p = [dict(node_id=i, category='car', center=c) for i, c in enumerate([
        [100, 0, 2], [119, 0, 2], [120, 0, 2], [100, 8, 2], [100, 0, .5]])]
    assert [x['node_id'] for x in in_view(p, row)] == [0, 1]


def test_gt_keeps_occluded_tiny_objects_and_depth_failures():
    valid = dict(category='Car', track_id=5, dimensions_hwl=[1, 1, 1],
        center_camera2=[100, 0, 30], bbox=[-5, 0, 1, 1], occlusion=3,
        depth_points=0, instance_token='0000:5')
    eligible, excluded = eligible_truth([valid], [1242, 375])
    assert len(eligible) == 1 and eligible[0]['category'] == 'car' and not excluded
    # GT中心不在图像内但有效标注框相交，仍保留在端到端分母。
    bad = copy.deepcopy(valid)
    bad['center_camera2'][2] = 70
    assert eligible_truth([bad], [1242, 375])[1][0]['exclusion_reason'] == 'outside_fixed_depth_scope'
    invalid = valid | {'track_id': -1}
    assert not eligible_truth([invalid], [1242, 375])[0]


def test_empty_prediction_statistics_preserve_denominator():
    metric = dict(predictions=0, truth=2, true_positive=0, unmatched_predictions=0,
        false_negative=2, duplicate_nodes=0, matches=[])
    report = object_statistics({'0000': [dict(metrics={m: metric for m in METHODS})]})
    for row in report['methods'].values():
        assert row['counts']['truth'] == 2
        assert row['metrics']['recall'] == 0
        assert row['metrics']['center_error_m'] is None
        assert row['metrics']['matched_prediction_fraction'] is None


def test_agent_loader_rejects_gt_field_or_path(tmp_path):
    path = tmp_path/'agent_manifest.json'
    raw = dict(frames=[], classes=['car'], selection_sha256='abc', evaluation_only=[])
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='未声明'):
        load_agent(path)
    raw.pop('evaluation_only')
    raw['frames'] = [dict(rgb='evaluation_only/a.png', depth='d', calib='c', lidar='l')]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='评价路径'):
        load_agent(path)
