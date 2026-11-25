"""实测稀疏证据、身份交接和诊断兼容，而非仅检查配置字面值。"""
import copy
from dataclasses import replace

import numpy as np
import pytest

from conceptgraph.completion.config_comparison import paired_intervals
from conceptgraph.completion.fault_robustness import mapping_config
from conceptgraph.slam.audited_mapper import AuditedEvidenceMapper
from conceptgraph.slam.evidence_mapper import EvidenceAwareMapper
from conceptgraph.slam.sparse_observations import (
    SparsePositionLayer, measured_points, select_depth_cluster,
)
from conceptgraph.slam.streamlined_mapping import MappingFrame


def observation(center=(0., 0., 10.), points=2, confidence=.8):
    return dict(center=np.array(center), category='pedestrian', feature=np.array([1., 0.]),
                points=points, confidence=confidence, variance=np.ones(3))


def test_sparse_needs_distinct_frames_and_real_support_then_expires():
    layer = SparsePositionLayer()
    assert layer.update([observation()], [], {}, 0.) == []
    output = layer.update([observation()], [], {}, .5)
    assert len(output) == 1 and output[0]['measured_points_total'] == 4
    assert output[0]['shape_available'] is False
    with pytest.raises(ValueError):
        layer.update([observation()], [], {}, .5)
    assert layer.update([], [], {}, 2.1) == []


def test_one_point_does_not_manufacture_more_points_or_strong_confirmation():
    layer = SparsePositionLayer()
    for timestamp in [0., .5, 1.]:
        assert layer.update([observation(points=1)], [], {}, timestamp) == []
    output = layer.update([observation(points=1)], [], {}, 1.5)
    assert output[0]['measured_points_total'] == 4
    weak = SparsePositionLayer()
    assert weak.update([observation(confidence=.15)], [], {}, 0.) == []
    assert weak.update([observation(confidence=.15)], [], {}, .5) == []
    assert len(weak.update([observation(confidence=.15)], [], {}, 1.)) == 1


def test_sparse_to_dense_handoff_preserves_identity_without_duplicate():
    layer = SparsePositionLayer()
    layer.update([observation()], [], {}, 0.)
    identity = layer.update([observation()], [], {}, .5)[0]['node_id']
    dense = [dict(node_id='track-7', category='pedestrian', center=[0., 0., 10.])]
    output = layer.update([], dense, {'track-7': observation(points=6)}, 1.)
    assert len(output) == 1 and output[0]['node_id'] == identity
    assert not layer.tracks


def test_sparse_association_is_one_to_one_and_class_gated():
    layer = SparsePositionLayer()
    layer.update([observation(), observation(center=(1., 0., 10.))], [], {}, 0.)
    result = layer.update([observation(), observation(center=(1., 0., 10.))], [], {}, .5)
    # 邻近假设输出去重，但两个内部轨迹仍各自仅获得一次观测。
    assert len({t['node_id'] for t in layer.tracks}) == 2
    assert [t['hits'] for t in layer.tracks] == [2, 2]
    assert len(result) == 1


def test_depth_selector_uses_past_evidence_to_reject_dominant_background():
    depth = np.zeros((20, 20))
    mask = np.zeros_like(depth, dtype=bool)
    mask[2:18, 2:18] = True
    depth[8:10, 8:11] = 10.
    depth[2, 2:18] = 30.
    selected, audit = select_depth_cluster(depth, mask, history_depths=[10.])
    assert selected.sum() == 6
    assert np.all(depth[selected] == 10.)
    assert audit['valid_pixels'] == 22 and audit['retained_pixels'] == 6


def test_zero_depth_and_unprojection_have_no_invented_support():
    depth = np.zeros((20, 20))
    mask = np.ones_like(depth, dtype=bool)
    selected, audit = select_depth_cluster(depth, mask)
    assert not selected.any() and audit['retained_pixels'] == 0
    depth[10, 10] = 5.
    points = measured_points(depth, mask, np.array([[10, 0, 10], [0, 10, 10], [0, 0, 1]]), np.eye(4))
    np.testing.assert_array_equal(points, [[0., 0., 5.]])


def test_invalid_variance_does_not_mutate_sparse_state():
    layer = SparsePositionLayer()
    bad = observation()
    bad['variance'][0] = -1
    with pytest.raises(ValueError):
        layer.update([bad], [], {}, 0.)
    assert layer.timestamp is None and not layer.tracks


def test_log_bootstrap_is_paired_and_reproducible():
    base, new = [], []
    for i in range(11):
        counts = dict(true_positive=40, truth=100, predictions=80, IDTP=35, IDFP=45, IDFN=65)
        row = dict(scene=f's{i}', log=f'l{i}', frames=10, result=dict(counts=counts))
        base.append(row)
        row = copy.deepcopy(row)
        row['result']['counts']['true_positive'] += 10
        new.append(row)
    a = paired_intervals(base, new)
    assert a == paired_intervals(base, new)
    np.testing.assert_allclose(a['ci_bonferroni']['recall'], [10, 10])
    new[0]['frames'] = 9
    with pytest.raises(ValueError):
        paired_intervals(base, new)


def mapping_input(point_count=6):
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    depth = np.zeros((40, 40))
    for i in range(point_count):
        depth[10 + i // 3, 10 + i % 3] = 10 + .1 * (i % 2)
    mask = np.zeros((40, 40), dtype=bool)
    mask[8:18, 8:18] = True
    gobs = dict(xyxy=np.array([[8, 8, 18, 18]]), mask=np.array([mask]),
        classes=['pedestrian'], class_id=np.array([0]), confidence=np.array([.8]),
        image_feats=np.array([[1., 0.]]), text_feats=np.array([[1., 0.]]))
    k = np.array([[30, 0, 20], [0, 30, 20], [0, 0, 1]])
    return MappingFrame(0, image, depth, k, np.eye(4), gobs, 'fixture', timestamp_s=0.)


def test_audit_does_not_change_default_geometry_identity_or_confirmation():
    cfg = mapping_config()
    cfg.merge_with(dict(mask_area_threshold=0, max_bbox_area_ratio=1., mask_conf_threshold=.1))
    original = EvidenceAwareMapper(copy.deepcopy(cfg), ['pedestrian'])
    audited = AuditedEvidenceMapper(copy.deepcopy(cfg), ['pedestrian'])
    np.random.seed(17)
    original.update(mapping_input())
    np.random.seed(17)
    audited.update(mapping_input())
    assert len(original.confirmed_objects()) == len(audited.confirmed_objects()) == 1
    np.testing.assert_array_equal(original.objects[0]['bbox'].get_center(), audited.objects[0]['bbox'].get_center())
    assert original.objects[0]['stable_object_id'] == audited.objects[0]['stable_object_id']
    assert any(r['stage'] == 'geometry' and r['reason'] == 'accepted' for r in audited.frame_audit)


def test_diagnostic_uses_actual_five_point_gate():
    cfg = mapping_config()
    cfg.merge_with(dict(mask_area_threshold=0, max_bbox_area_ratio=1., mask_conf_threshold=.1))
    mapper = AuditedEvidenceMapper(cfg, ['pedestrian'])
    mapper.update(mapping_input(point_count=4))
    assert not mapper.objects
    rejected = [r for r in mapper.frame_audit if r['stage'] == 'geometry']
    assert rejected[0]['points'] == 4 and rejected[0]['required_points'] == 5


def test_module_a_actual_entry_retains_sparse_evidence_without_fabricated_shape():
    from conceptgraph.slam.quality_mapper import QualityEvidenceMapper
    cfg = mapping_config()
    cfg.merge_with(dict(mask_area_threshold=0, max_bbox_area_ratio=1., mask_conf_threshold=.1))
    mapper = QualityEvidenceMapper(cfg, ['pedestrian'], module_a=True)
    first = mapping_input(point_count=2)
    mapper.update(first)
    assert not mapper.objects and len(mapper.sparse_measurements) == 1
    assert mapper.position_predictions([], first) == []
    second = replace(mapping_input(point_count=2), frame_idx=1, timestamp_s=.5)
    mapper.update(second)
    output = mapper.position_predictions([], second)
    assert len(output) == 1 and output[0]['measured_points_total'] == 4
    assert not mapper.objects
    assert output[0]['shape_available'] is False
    assert 'bbox' not in output[0] and 'pcd' not in output[0]


def test_single_log_gain_does_not_pass_corrected_interval():
    base, candidate = [], []
    for i in range(11):
        counts = dict(true_positive=40, truth=100, predictions=80, IDTP=35, IDFP=45, IDFN=65)
        row = dict(scene=f's{i}', log=f'l{i}', frames=10, result=dict(counts=counts))
        base.append(row)
        row = copy.deepcopy(row)
        if i == 0:
            row['result']['counts']['true_positive'] += 20
        candidate.append(row)
    assert paired_intervals(base, candidate)['ci_bonferroni']['recall'][0] == 0
