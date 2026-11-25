"""可靠当帧观测保护与弱证据回退的行为检查。"""
from dataclasses import replace

import numpy as np

from conceptgraph.completion.fault_robustness import mapping_config
from conceptgraph.completion.stage9 import predict
from conceptgraph.slam.dense_refinement import DenseProtectedPositionLayer, RefinedEvidenceMapper
from conceptgraph.slam.sparse_observations import SparsePositionLayer
from tests.test_sparse_quality_mapping import mapping_input, observation


def test_dense_measurement_avoids_filter_lag_but_keeps_identity():
    layer = DenseProtectedPositionLayer()
    dense = [dict(node_id='track-1', category='pedestrian', center=[0., 0., 10.])]
    layer.update([], dense, {'track-1': observation(points=12)}, 0.)
    new = observation(center=(4., 0., 10.), points=12)
    new.update(reliable_current_dense=True, maximum_speed_m_s=20.)
    out = layer.update([], dense, {'track-1': new}, .5)
    np.testing.assert_array_equal(out[0]['center'], new['center'])
    assert out[0]['node_id'] == 'track-1'
    assert layer.dense_states['track-1']['hits'] == 2


def test_weak_measurement_matches_original_filter_exactly():
    a, b = DenseProtectedPositionLayer(), SparsePositionLayer()
    dense = [dict(node_id='track-1', category='pedestrian', center=[0., 0., 10.])]
    for time, x in [(0., 0.), (.5, 3.), (1., 3.2)]:
        obs = observation(center=(x, 0., 10.), points=5)
        assert a.update([], dense, {'track-1': obs}, time) == b.update([], dense, {'track-1': obs}, time)


def test_impossible_velocity_resets_instead_of_predicting_jump():
    layer = DenseProtectedPositionLayer()
    state = layer._new_state(observation(), 0., 'track-1')
    new = observation(center=(50., 0., 10.), points=20)
    new.update(reliable_current_dense=True, maximum_speed_m_s=20.)
    layer._update_state(state, new, .5)
    np.testing.assert_array_equal(state['velocity'], np.zeros(3))


def test_real_entry_uses_current_detection_and_keeps_sparse_shape_absent():
    cfg = mapping_config()
    cfg.merge_with(dict(mask_area_threshold=0, max_bbox_area_ratio=1., mask_conf_threshold=.1))
    mapper = RefinedEvidenceMapper(cfg, ['pedestrian'])
    first = mapping_input(point_count=2)
    mapper.update(first)
    assert mapper.position_predictions([], first) == []
    second = replace(first, frame_idx=1, timestamp_s=.5)
    mapper.update(second)
    output = mapper.position_predictions([], second)
    assert output[0]['shape_available'] is False and output[0]['measured_points_total'] == 4
    assert not mapper.objects


def test_real_dense_entry_protects_center_and_reports_source():
    cfg = mapping_config()
    cfg.merge_with(dict(mask_area_threshold=0, max_bbox_area_ratio=1., mask_conf_threshold=.1))
    mapper = RefinedEvidenceMapper(cfg, ['pedestrian'])
    for index in range(2):
        frame = replace(mapping_input(point_count=12), frame_idx=index, timestamp_s=index * .5)
        np.random.seed(17 + index)
        mapper.update(frame)
        out = mapper.position_predictions(predict(mapper.confirmed_objects(), mapper.classes), frame)
        assert out[0]['position_source'] == 'reliable_current_dense_lidar'
        np.testing.assert_array_equal(out[0]['center'], mapper.objects[0]['center_history'][-1])
        assert mapper.frame_audit[-1]['reliable_dense_measurements'] == 1
