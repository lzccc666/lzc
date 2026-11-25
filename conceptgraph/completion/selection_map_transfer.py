"""冻结预测地图的规则选择接口；缺失稀疏形状/运动/深度不补造。"""
from __future__ import annotations

import numpy as np

from conceptgraph.navigation.semantic import parse_navigation_query, rank_candidate_targets
from conceptgraph.navigation.types import SceneGraphNode
from conceptgraph.scenegraph.relations import build_relations

EVIDENCE_POLICY = dict(enabled=True, minimum_observations=3, minimum_lidar_pixels=5,
                       minimum_depth_support_fraction=.4)


def nodes_from_frozen_predictions(predictions):
    nodes = {}
    for local_id, prediction in enumerate(predictions):
        center = np.asarray(prediction['center'],dtype=float)
        if center.shape!=(3,) or not np.isfinite(center).all():
            raise ValueError('冻结预测中心无效，不能静默丢弃')
        extent = prediction.get('extent')
        if extent is None and prediction.get('shape_available') is not False:
            raise ValueError('只有显式无形状的预测可以缺extent')
        # None是真正缺失，不用0或猜测的汽车尺寸构造虚假边界框。
        extent = None if extent is None else np.asarray(extent,dtype=float)
        motion = prediction.get('motion_state','unknown')
        nodes[local_id] = SceneGraphNode(local_id,prediction['category'],[prediction['category']],
            prediction['category'],center,extent,motion_state=motion,is_short_term=motion=='dynamic',
            metadata=dict(num_detections=prediction['num_detections'],
                depth_evidence=prediction.get('depth_evidence',[]),
                frozen_object_id=prediction['node_id'],shape_available=extent is not None))
    return nodes


def select_frozen_map(query,predictions,pose,frame_index):
    """相同原规则，输入不含GT，局部整数ID只由预测列表顺序产生。"""
    nodes = nodes_from_frozen_predictions(predictions)
    forward = pose[:2,2]
    edges = build_relations(nodes,dict(coordinate_frame='ego_heading',
        directional_mode='all_supported_axes'),frame_index=frame_index,ego_heading=forward).edges
    parsed = parse_navigation_query(query,8.)
    parsed.vehicle_heading_xy = forward
    return rank_candidate_targets(parsed,nodes,edges,pose[:2,3],.55,.3,.25,
        relation_coordinate_frame='ego_heading',relation_coordinate_frame_fallback=None,
        evidence_policy=EVIDENCE_POLICY)
