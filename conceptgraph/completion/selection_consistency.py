"""显式实验策略：图边和几何回退使用相同方向关系支持范围。"""
from __future__ import annotations

from conceptgraph.completion.navigation_diagnostics import to_nodes
from conceptgraph.navigation.semantic import parse_navigation_query, rank_candidate_targets
from conceptgraph.scenegraph.relations import RelationConfig, build_relations


def select_consistent_relations(query, predictions, pose, frame_index):
    """输入只有预测、位姿与文字；不接收GT或可接受目标集合。"""
    nodes = to_nodes(predictions)
    forward = pose[:2, 2]
    config = RelationConfig.from_mapping(dict(coordinate_frame='ego_heading',
                                            directional_mode='all_supported_axes'))
    edges = build_relations(nodes, config, frame_index=frame_index, ego_heading=forward).edges
    parsed = parse_navigation_query(query, 8.)
    parsed.vehicle_heading_xy = forward
    policy = dict(enabled=True, minimum_observations=3, minimum_lidar_pixels=5,
                  minimum_depth_support_fraction=.4)
    support = dict(enabled=True,
                   maximum_directional_distance_m=config.relation_distance_thresholds_m['directional'],
                   minimum_directional_component_m=config.minimum_direction_component_m)
    return rank_candidate_targets(parsed, nodes, edges, pose[:2, 3], .55, .3, .25,
        relation_coordinate_frame='ego_heading', relation_coordinate_frame_fallback=None,
        evidence_policy=policy, relation_support_policy=support)
