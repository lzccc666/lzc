"""归因的实际风险：官方 wlh 轴顺序/旋转，以及稀疏支持不能被当稠密拒绝。"""
import numpy as np

from conceptgraph.slam.r1_failure_attribution import inside_official_box, state_evidence


def test_official_box_rotation_and_wlh_order():
    obj = dict(center=[10., 20., 0.], size_wlh=[2., 10., 2.],
        rotation_wxyz=[np.sqrt(.5), 0., 0., np.sqrt(.5)])
    points = [[10., 24.9, 0.], [12., 20., 0.], [10., 25.25, 0.]]
    assert inside_official_box(points, obj).tolist() == [True, False, False]
    assert inside_official_box(points, obj, .5).tolist() == [True, False, True]


def test_one_point_accurate_sparse_measurement_needs_real_confirmation():
    row = dict(category='car', center=[0., 0., 0.], matched=False, candidates=[dict(raw_index=2)],
        final_within_3m=[], iou025_detection_count=1, input_retained_count=1,
        max_valid_mask_depth_points=1, max_selected_mask_depth_points=1, depth_bin='1_to_4')
    track = dict(node_id='sparse-1', category='car', center=[1., 0., 0.], last_measured_center=[1., 0., 0.],
        hits=1, total_points=1, confidence=.8)
    states = dict(observations=[], sparse_measurements=[dict(category='car', raw_index=2, center=[1.,0.,0.],points=1)],
        dense_internal=[], dense_confirmed=[], position_outputs_before_fov=[], sparse_tracks=[track],
        dense_states={}, associations=[], confirmations=[], aliases={})
    truth = dict(center=[0.,0.,0.], size_wlh=[2.,4.,2.],rotation_wxyz=[1.,0.,0.,0.])
    result = state_evidence(row, states, truth)
    assert result['current_measurement_accurate']
    assert result['reason'] == 'accurate_sparse_observation_insufficient_confirmation'
    assert result['accurate_sparse_links'][0]['total_points'] == 1
