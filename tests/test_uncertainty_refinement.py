"""两轮匹配的竞争、硬门限及实际地图更新验证。"""
from dataclasses import replace
import copy

import numpy as np
import pytest

from conceptgraph.completion.fault_robustness import mapping_config
from conceptgraph.slam.uncertainty_refinement import (
    UncertaintyAssociationSettings, create_refinement_mapper, two_stage_assignment, uncertainty_scores,
)
from tests.test_sparse_quality_mapping import mapping_input


def test_strong_observation_has_priority_and_weak_uses_only_remaining_track():
    assigned, stages = two_stage_assignment(np.array([[.7, -np.inf], [.9, .6]]), [True, False])
    assert assigned == {0: 0, 1: 1}
    assert stages == {0: 'reliable', 1: 'weak'}


def test_empty_forbidden_and_quality_shape_are_handled():
    assert two_stage_assignment(np.empty((0, 2)), [])[0] == {}
    assert two_stage_assignment(np.empty((2, 0)), [True, False])[0] == {}
    assert two_stage_assignment(np.full((2, 2), -np.inf), [True, False])[0] == {}
    with pytest.raises(ValueError):
        two_stage_assignment(np.zeros((2, 2)), [True])


@pytest.mark.parametrize('condition', ['R2', 'R3', 'R3_single_stage'])
def test_real_entry_freezes_unique_association_and_keeps_identity(condition):
    cfg = mapping_config()
    cfg.merge_with(dict(mask_area_threshold=0, max_bbox_area_ratio=1., mask_conf_threshold=.1))
    mapper = create_refinement_mapper(cfg, ['pedestrian'], condition=condition)
    for index in range(2):
        frame = replace(mapping_input(point_count=12), frame_idx=index, timestamp_s=index * .5)
        np.random.seed(17 + index)
        mapper.update(frame)
    assert len(mapper.objects) == 1 and mapper.objects[0]['num_detections'] == 2
    events = [e for e in mapper.frame_audit if e['stage'] == 'uncertainty_association']
    expected = 'matched_joint' if condition == 'R3_single_stage' else 'matched_reliable'
    assert len(events) == 1 and events[0]['reason'] == expected
    assert events[0]['previous_node_id'] == mapper.objects[0]['stable_object_id']


def test_category_appearance_physical_and_uncertainty_gates():
    cfg = mapping_config()
    cfg.merge_with(dict(mask_area_threshold=0, max_bbox_area_ratio=1., mask_conf_threshold=.1))
    mapper = create_refinement_mapper(cfg, ['pedestrian'], condition='R2')
    frame = mapping_input(point_count=12)
    mapper.update(frame)
    detection = mapper.last_observations
    later = replace(frame, frame_idx=1, timestamp_s=.5)
    settings = UncertaintyAssociationSettings()
    for appearance, wrong_class, move in [(0., False, 0.), (1., True, 0.), (1., False, 20.), (1., False, 5.)]:
        det = copy.deepcopy(detection[0])
        det['class_id'] = [1] if wrong_class else [0]
        det['bbox'].translate(np.array([move, 0., 0.]))
        det['pcd'].translate(np.array([move, 0., 0.]))
        det['center_history'] = [np.asarray(det['center_history'][-1]) + [move, 0., 0.]]
        values, _, _ = uncertainty_scores([det], mapper.objects, np.ones((1, 1)),
                                          np.full((1, 1), appearance), later, settings)
        assert not np.isfinite(values).any()


def test_invalid_similarity_is_not_allowed_into_assignment():
    cfg = mapping_config()
    cfg.merge_with(dict(mask_area_threshold=0, max_bbox_area_ratio=1., mask_conf_threshold=.1))
    mapper = create_refinement_mapper(cfg, ['pedestrian'], condition='R3')
    frame = mapping_input(point_count=12)
    mapper.update(frame)
    later = replace(frame, frame_idx=1, timestamp_s=.5)
    for spatial, appearance in [(1., np.nan), (np.nan, 1.)]:
        scores, _, _ = uncertainty_scores(mapper.last_observations, mapper.objects,
            np.array([[spatial]]), np.array([[appearance]]), later, UncertaintyAssociationSettings())
        assert not np.isfinite(scores).any()
