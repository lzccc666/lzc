"""未知/动态/静态寿命边界与原R1兼容，使用真实地图入口。"""
import copy
from dataclasses import replace

import numpy as np
import pytest

from conceptgraph.completion.fault_robustness import mapping_config
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.slam.uncertainty_refinement import create_refinement_mapper
from conceptgraph.slam.unknown_expiry_refinement import create_unknown_expiry_mapper
from tests.test_sparse_quality_mapping import mapping_input


def setup_mapper(condition):
    cfg = mapping_config()
    cfg.merge_with(dict(mask_area_threshold=0, max_bbox_area_ratio=1., mask_conf_threshold=.1))
    return create_unknown_expiry_mapper(cfg, ['pedestrian'], condition=condition)


@pytest.mark.parametrize('condition', ['R1_unknown_ttl', 'R3_unknown_ttl'])
def test_unknown_uses_original_ttl_while_static_and_recent_unknown_survive(condition):
    mapper = setup_mapper(condition)
    mapper.update(mapping_input(point_count=12))
    template = mapper.objects[0]
    objects = []
    for node, state, short_term, observed in [('old_unknown', 'unknown', False, 0.),
            ('old_static', 'static', False, 0.), ('old_dynamic', 'dynamic', True, 0.),
            ('recent_unknown', 'unknown', False, 1.5)]:
        obj = copy.deepcopy(template)
        obj.update(stable_object_id=node, motion_state=state, is_short_term=short_term,
                   timestamp_s=observed, is_potential_dynamic=True)
        objects.append(obj)
    mapper.objects = MapObjectList(objects)
    mapper.update(replace(mapping_input(point_count=0), frame_idx=1, timestamp_s=2.))
    assert {o['stable_object_id'] for o in mapper.objects} == {
        'old_unknown', 'old_static', 'old_dynamic', 'recent_unknown'}
    mapper.update(replace(mapping_input(point_count=0), frame_idx=2, timestamp_s=2.01))
    assert {o['stable_object_id'] for o in mapper.objects} == {'old_static', 'recent_unknown'}
    audit = next(e for e in mapper.frame_audit if e['stage'] == 'unknown_expiry')
    assert audit['expired_node_ids'] == ['old_unknown'] and audit['ttl_s'] == 2.


def test_confirmed_static_after_three_observations_is_not_expired():
    mapper = setup_mapper('R1_unknown_ttl')
    for index in range(3):
        frame = replace(mapping_input(point_count=12), frame_idx=index, timestamp_s=index*.5)
        np.random.seed(17+index)
        mapper.update(frame)
    assert len(mapper.objects) == 1 and mapper.objects[0]['motion_state'] == 'static'
    mapper.update(replace(mapping_input(point_count=0), frame_idx=3, timestamp_s=4.))
    assert len(mapper.objects) == 1 and mapper.objects[0]['motion_state'] == 'static'


def test_original_r1_keeps_its_old_behavior_and_invalid_time_does_not_mutate():
    cfg = mapping_config()
    cfg.merge_with(dict(mask_area_threshold=0, max_bbox_area_ratio=1., mask_conf_threshold=.1))
    original = create_refinement_mapper(cfg, ['pedestrian'], condition='R1')
    original.update(mapping_input(point_count=12))
    original.update(replace(mapping_input(point_count=0), frame_idx=1, timestamp_s=2.01))
    assert len(original.objects) == 1
    changed = setup_mapper('R1_unknown_ttl')
    changed.update(mapping_input(point_count=12))
    node = changed.objects[0]['stable_object_id']
    with pytest.raises(ValueError):
        changed.update(replace(mapping_input(point_count=0), frame_idx=1, timestamp_s=0.))
    assert changed.objects[0]['stable_object_id'] == node
