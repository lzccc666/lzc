import numpy as np

from conceptgraph.completion.kitti_geometry import assignment, backproject, labels, project_lidar
from conceptgraph.completion.kitti_evaluate import common_supported_errors, evaluate_frame


def test_projection_fourth_column_round_trip():
    p = np.array([[700, 0, 600, 45], [0, 700, 180, -.35], [0, 0, 1, .005]])
    xyz = np.array([[2., 1, 8], [-4, .2, 23], [0, 0, 60]])
    q = xyz @ p[:, :3].T + p[:, 3]
    uv = q[:, :2]/q[:, 2, None]
    np.testing.assert_allclose(backproject(uv, xyz[:, 2], p), xyz, atol=1e-12)
    assert np.max(np.abs(backproject(uv, xyz[:, 2], np.c_[p[:, :3], np.zeros(3)])-xyz)) > .01


def test_nearest_measured_point_and_negative_pixel():
    calib = dict(P2=np.c_[np.eye(3), np.zeros(3)], R0_rect=np.eye(3),
                 Tr_velo_to_cam=np.c_[np.eye(3), np.zeros(3)])
    depth, xyz, info = project_lidar(np.array([[1, 1, 5], [2, 2, 10], [-.01, 1, 5]]), calib, (4, 4))
    assert info['occupied_pixels'] == 1
    assert depth[0, 0] == 5
    np.testing.assert_array_equal(xyz[0, 0], [1, 1, 5])


def test_official_bottom_center_conversion():
    row = labels('Car 0 0 0 1 2 30 40 2 1.8 4 5 3 20 0')[0]
    assert row['center_camera0'] == [5, 2, 20]
    assert row['bottom_center_camera0'] == [5, 3, 20]


def test_assignment_gate_does_not_steal_valid_pair():
    pred = [dict(evaluation_category='car', bbox=[0, 0, 10, 10]),
            dict(evaluation_category='pedestrian', bbox=[0, 0, 10, 10])]
    gt = [dict(evaluation_category='car', bbox=[0, 0, 10, 10])]
    assert assignment(pred, gt) == [(0, 0)]


def test_missing_depth_preserves_positive_and_precision_denominators():
    truth = labels('Car 0 0 0 1 2 30 40 2 1.8 4 5 3 20 0')
    pred = [dict(category='car', bbox=[1, 2, 30, 40], prediction_index=0,
                 methods={'m': dict(center_camera0=None, depth_evidence={})})]
    row = evaluate_frame(pred, truth, 'car', 'm')
    assert (row['truth'], row['predictions'], row['matches_2d'], row['matches_3d']) == (1, 1, 1, 0)
    assert row['conditional_localized'] == 0


def test_dontcare_does_not_hide_duplicate_on_real_object():
    truth = labels('Car 0 0 0 1 2 30 40 2 1.8 4 5 3 20 0\n'
                   'DontCare -1 -1 -10 0 0 100 100 -1 -1 -1 -1000 -1000 -1000 -10')
    pred = [dict(category='car', bbox=[1, 2, 30, 40], prediction_index=i,
                 methods={'m': dict(center_camera0=None, depth_evidence={})}) for i in range(2)]
    row = evaluate_frame(pred, truth, 'car', 'm')
    assert row['matches_2d'] == 1
    assert row['predictions'] == 2
    assert row['ignored_predictions'] == 0


def test_conditional_error_does_not_compare_different_supported_objects():
    base = dict(frame_id='f', drive='d', category='car')
    first = dict(prediction_index=0, label_index=0, error_m=10)
    second = dict(prediction_index=1, label_index=1, error_m=2)
    rows = [base | dict(method='mask_points_mean', conditional_center_errors=[first, second]),
            base | dict(method='mask_points_robust_mean', conditional_center_errors=[second | {'error_m': 3}])]
    paired = common_supported_errors(rows)
    assert len(paired) == 1
    assert paired[0]['prediction_index'] == 1
    assert paired[0]['errors_m'] == {'mask_points_mean': 2, 'mask_points_robust_mean': 3}
