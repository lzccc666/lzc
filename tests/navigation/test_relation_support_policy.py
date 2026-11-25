"""避免不受图构建范围支持的几何回退影响目标选择；旧默认保持兼容。"""
import numpy as np
import pytest

from conceptgraph.navigation.semantic import parse_navigation_query, rank_candidate_targets
from conceptgraph.navigation.types import SceneGraphNode
from conceptgraph.scenegraph.relations import SceneGraphEdge


POLICY = dict(enabled=True, maximum_directional_distance_m=12., minimum_directional_component_m=1.)


def node(identity, tag, xy, **kwargs):
    return SceneGraphNode(identity, tag, [tag], tag, np.array([*xy, 0.]), np.ones(3), **kwargs)


def rank(nodes, *, support=POLICY, edges=(), evidence=None):
    query = parse_navigation_query('去卡车左侧的小汽车旁边', 8.)
    query.vehicle_heading_xy = np.array([1., 0.])
    return rank_candidate_targets(query, nodes, list(edges), np.zeros(2), .55, .3, .25,
        relation_coordinate_frame='ego_heading', relation_coordinate_frame_fallback=None,
        relation_support_policy=support, evidence_policy=evidence)


def test_far_geometry_fallback_does_not_override_local_relation_scope():
    nodes = {1: node(1, 'car', [10., 15.]), 2: node(2, 'truck', [10., 0.])}
    assert [c.node_id for c in rank(nodes, support=None)] == [1]
    assert rank(nodes) == []


def test_direction_dead_zone_applies_to_geometry_fallback():
    nodes = {1: node(1, 'car', [10., .5]), 2: node(2, 'truck', [10., 0.])}
    assert [c.node_id for c in rank(nodes, support=None)] == [1]
    assert rank(nodes) == []


def test_current_geometry_rejects_stale_or_wrong_coordinate_graph_edge():
    nodes = {1: node(1, 'car', [10., -3.]), 2: node(2, 'truck', [10., 0.])}
    edge = SceneGraphEdge(1, 2, 'left_of', confidence=.9, coordinate_frame='ego_heading')
    assert [c.node_id for c in rank(nodes, support=None, edges=[edge])] == [1]
    assert rank(nodes, edges=[edge]) == []


def test_supported_local_pair_keeps_graph_and_fallback_choices():
    nodes = {1: node(1, 'car', [10., 3.]), 2: node(2, 'truck', [10., 0.])}
    edge = SceneGraphEdge(1, 2, 'left_of', confidence=.9, coordinate_frame='ego_heading')
    assert [c.node_id for c in rank(nodes)] == [1]
    assert [c.node_id for c in rank(nodes, edges=[edge])] == [1]


def test_relation_policy_never_bypasses_authorization_evidence_gate():
    nodes = {1: node(1, 'car', [10., 3.], motion_state='dynamic'),
             2: node(2, 'truck', [10., 0.])}
    assert rank(nodes, evidence=dict(enabled=True)) == []


def test_invalid_policy_rejected_explicitly():
    nodes = {1: node(1, 'car', [10., 3.]), 2: node(2, 'truck', [10., 0.])}
    with pytest.raises(ValueError):
        rank(nodes, support={**POLICY, 'maximum_directional_distance_m': float('nan')})
