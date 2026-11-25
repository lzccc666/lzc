"""Consume the same near6/frame-explicit graph for static object approach goals.

The output region is an engineering road subgoal, not a parking-space label,
trajectory, collision certificate or vehicle execution authorization.
"""
from __future__ import annotations

import math
import numpy as np
from shapely.geometry import Point, mapping

from conceptgraph.completion.evidence import target_evidence_decision
from conceptgraph.completion.selection_map_transfer import EVIDENCE_POLICY
from conceptgraph.navigation.semantic import parse_navigation_query,rank_candidate_targets,infer_query_categories
from conceptgraph.navigation.types import SceneGraphNode
from conceptgraph.scenegraph.relation_nav_unified_20261125 import reference_query_status,FRAME_SOURCE


POLICY=dict(near_distance_m=6.,maximum_directional_distance_m=12.,minimum_directional_component_m=1.,
    minimum_approach_center_distance_m=2.,maximum_approach_center_distance_m=8.,
    task_kind='approach_confirmed_static_object',ambiguous_anchor='ask_clarification',
    maximum_road_sample_distance_m=100.,configuration_count=1,seed=17)


def adapt_nodes(records,registry):
    nodes={};eligible={};indices={}
    for index,record in enumerate(records):
        stable=record['node_id'];registry.setdefault(stable,len(registry));local=registry[stable]
        unified='center_world_m' in record
        center=record['center_world_m'] if unified else record['center']
        extent=record.get('extent_world_aabb_m') if unified else record.get('extent')
        motion=record.get('motion_state','unknown')
        if motion not in {'static','dynamic','unknown'}:raise ValueError('invalid_motion_tristate')
        legacy=record.get('legacy_prediction',{}) if unified else record
        metadata=dict(num_detections=legacy.get('num_detections',0),depth_evidence=legacy.get('depth_evidence',[]),
            shape_available=extent is not None,frozen_object_id=stable,motion_state=motion,
            body_heading_world_rad=record.get('body_heading_world_rad'),body_heading_source=record.get('body_heading_source','unknown'))
        node=SceneGraphNode(local,record['category'],[record['category']],record['category'],
            np.asarray(center,dtype=float),None if extent is None else np.asarray(extent,dtype=float),
            motion_state=motion,is_short_term=motion=='dynamic',metadata=metadata)
        nodes[local]=node;indices[local]=record.get('original_index',index)
        eligible[local]=(bool(record['semantic_position_candidate']) if unified else target_evidence_decision(node,EVIDENCE_POLICY)[0]) and motion=='static'
    return nodes,eligible,indices


def select_task(query,nodes,eligible,edges,*,heading,origin,reference_frame='ego_heading'):
    status=reference_query_status(query,reference_frame)
    result=dict(status=status['status'],reference_frame=reference_frame,reference_frame_source=FRAME_SOURCE,
        selected_id=None,candidate_ids=[],GT_in_method=False,Qwen_calls=0,
        task_kind=POLICY['task_kind'],vehicle_safety_authorized=False,trajectory_generated=False)
    if not status['supported']:return result
    parsed=parse_navigation_query(query,6.);parsed.vehicle_heading_xy=np.asarray(heading);parsed.near_distance=6.
    anchor_categories=infer_query_categories(parsed.anchor_text or '')
    if anchor_categories:
        anchors=[n.node_id for n in nodes.values() if n.object_tag in anchor_categories]
        result['reference_candidate_ids']=anchors
        if not anchors:result['status']='NO_REFERENCE_CANDIDATE';return result
        if len(anchors)>1:result['status']='NEED_REFERENCE_CLARIFICATION';return result
    ranked=rank_candidate_targets(parsed,nodes,edges,np.asarray(origin),.55,.3,.25,
        relation_coordinate_frame='ego_heading',relation_coordinate_frame_fallback=None,evidence_policy=None,
        relation_support_policy=dict(enabled=True,maximum_directional_distance_m=12.,minimum_directional_component_m=1.))
    ranked=[r for r in ranked if eligible[r.node_id]]
    result.update(status='TARGET_SELECTED' if ranked else 'NO_CONFIRMED_STATIC_TARGET',
        selected_id=ranked[0].node_id if ranked else None,candidate_ids=[r.node_id for r in ranked],
        candidate_scores=[dict(node_id=r.node_id,score=r.score,relation=r.relation_score,distance=r.distance_prior) for r in ranked])
    return result


def engineering_approach_region(target,prior,*,origin,heading):
    """Road/near geometry only; cannot infer a parking slot or unseen occupancy."""
    center=np.asarray(target.bbox_center[:2]);point=Point(center)
    zone=prior['drivable'].intersection(point.buffer(8.).difference(point.buffer(2.)))
    result=dict(status='NO_ENGINEERING_ROAD_SUBGOAL',source='shared_HD_drivable_and_predicted_target_center',
        region_kind='engineering_object_approach_road_region',position_frame='world_xy_m',
        target_node_id=target.node_id,relation_near_distance_m=6.,minimum_approach_center_distance_m=2.,maximum_approach_center_distance_m=8.,
        polygon=None,subgoal=None,GT_region_available=False,parking_slot_claim=False,curb_parking_claim=False,
        vehicle_safety_authorized=False,vehicle_sweep_checked=False,route_found=False,actual_arrival_credit=False)
    if zone.is_empty:return result
    options=[]
    for lane in prior['lanes']:
        samples=np.asarray(lane['centerline'])
        for sample in samples:
            if len(sample)<3:continue
            distance=float(np.linalg.norm(sample[:2]-center))
            if not 2.<=distance<=8. or np.linalg.norm(sample[:2]-origin)>100.:continue
            direction=np.array([math.cos(sample[2]),math.sin(sample[2])])
            if direction@heading<.5:continue
            if not zone.covers(Point(sample[:2])):continue
            options.append((float(np.linalg.norm(sample[:2]-origin)),lane['token'],sample.tolist()))
    if not options:return result
    _,lane,sample=min(options,key=lambda x:(x[0],x[1],x[2]))
    result.update(status='ENGINEERING_ROAD_SUBGOAL',polygon=mapping(zone),
        subgoal=dict(x=sample[0],y=sample[1],yaw_rad=sample[2],lane_token=lane),
        constraints='point lies in HD drivable approach annulus2..8m and lane heading agrees; relation near6 is separate; connectivity/obstacles/vehicle body not certified')
    return result
