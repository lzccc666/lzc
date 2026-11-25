"""相机平移与IMU/标签坐标回归，不加载模型或真实数据。"""
import math

import numpy as np
import pytest

from conceptgraph.completion.kitti_tracking_geometry import calibration, oxts_poses, project_lidar, tracking_labels


def calib_text(shift=(.5, -.2, .1)):
    k = np.array([[10., 0, 4], [0, 10., 4], [0, 0, 1.]])
    p = np.column_stack([k, k@shift])
    return '\n'.join(name+': '+' '.join(map(str, array.ravel())) for name, array in
        [('P2', p), ('R_rect', np.eye(3)), ('Tr_velo_cam', np.eye(4)[:3]), ('Tr_imu_velo', np.eye(4)[:3])])


def test_camera2_shift_is_absorbed_in_depth_and_pose():
    c = calibration(calib_text())
    point = np.array([1., 1., 10.])
    actual = c['camera2_from_velo']@np.r_[point, 1.]
    full_projection = c['projection']@np.r_[point, 1.]
    shifted_projection = c['intrinsic']@actual[:3]
    assert np.allclose(full_projection/full_projection[2], shifted_projection/shifted_projection[2])
    assert actual[2] == pytest.approx(10.1)
    assert np.allclose(c['imu_from_camera2']@actual, np.r_[point, 1.])


def test_official_tracking_mixes_colon_and_whitespace_calibration_rows():
    mixed = calib_text().replace('R_rect:', 'R_rect').replace('Tr_velo_cam:', 'Tr_velo_cam').replace('Tr_imu_velo:', 'Tr_imu_velo')
    original, parsed = calibration(calib_text()), calibration(mixed)
    assert all(np.array_equal(original[name], value) for name, value in parsed.items())


def test_projection_keeps_nearest_valid_point_and_does_not_clip_far_to_valid_depth():
    c = calibration(calib_text((0, 0, .1)))
    points = np.array([[0, 0, 10, 1], [0, 0, 20, 1], [0, 0, -1, 1], [0, 0, 80, 1]])
    depth, report = project_lidar(points, c, (8, 8))
    assert depth[4, 4] == 10100 and np.count_nonzero(depth) == 1
    assert report['valid_projected_points'] == 2
    assert report['P2_translation_absorbed_in_camera2']


def test_oxts_world_uses_first_imu_origin_and_meter_scale():
    text = '49 8 100 0 0 0\n49 8.00001 100 0 0 0.1\n'
    poses = oxts_poses(text)
    expected_east = math.cos(math.radians(49))*6378137*math.radians(.00001)
    assert np.allclose(poses[0], np.eye(4))
    assert poses[1, 0, 3] == pytest.approx(expected_east, abs=1e-8)
    assert poses[1, 1, 3] == pytest.approx(0, abs=1e-8)
    assert poses[1, 1, 0] == pytest.approx(math.sin(.1))


def test_official_bottom_center_and_track_id_remain_evaluation_only():
    text = '0 7 Car 0 1 0 1 2 6 7 2 2 4 1 3 12 0'
    c = calibration(calib_text())
    result = tracking_labels(text, '0005', np.eye(4)[None], c)[0]
    assert result['instance_token'] == '0005:7' and result['evaluation_only']
    assert result['center'] == [1., 2., 12.]
    assert np.allclose(result['center_camera2'], [1.5, 1.8, 12.1])


@pytest.mark.parametrize('text', ['nan 8 100 0 0 0', '90 8 100 0 0 0', '49 8 0'])
def test_invalid_oxts_is_not_silently_repaired(text):
    with pytest.raises(ValueError):
        oxts_poses(text)


def test_missing_imu_transform_is_not_assumed_identity():
    with pytest.raises(ValueError, match='缺少KITTI Tracking标定'):
        calibration('\n'.join(calib_text().splitlines()[:-1]))
