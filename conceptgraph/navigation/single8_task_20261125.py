"""Read one pair edge without materializing cardinal or reverse edges."""
from __future__ import annotations

import copy
import re
import numpy as np

from conceptgraph.navigation.semantic import infer_query_categories

PHRASES = (
    ('左前方','front_left'),('右前方','front_right'),('左后方','back_left'),('右后方','back_right'),
    ('左前侧','front_left'),('右前侧','front_right'),('左后侧','back_left'),('右后侧','back_right'),
    ('左前','front_left'),('右前','front_right'),('左后','back_left'),('右后','back_right'),
    ('前方','front'),('前面','front'),('后方','back'),('后面','back'),
    ('左侧','left'),('左边','left'),('右侧','right'),('右边','right'))
INCLUDES = dict(front={'front','front_left','front_right'},back={'back','back_left','back_right'},
                left={'left','front_left','back_left'},right={'right','front_right','back_right'},
                front_left={'front_left'},front_right={'front_right'},back_left={'back_left'},back_right={'back_right'})


def parse_query(text):
    """Compound phrases precede cardinal phrases; no hidden GT/task flags."""
    result=dict(query_text=text,target_categories=[],anchor_categories=[],relation=None,
                near_m=6.,requires_static=any(w in text for w in ('停着','停放','静止')),
                nearest='最近' in text,supported=True)
    for phrase,relation in PHRASES:
        if phrase in text:
            before,after=text.split(phrase,1)
            result.update(relation=relation,anchor_categories=infer_query_categories(before),target_categories=infer_query_categories(after))
            if not result['anchor_categories'] or not result['target_categories']:
                result.update(supported=False,reason='directional_query_requires_explicit_reference_and_target_category')
            return result
    # Nearby in a goal phrase is not an anchor relation. A nearby relation needs
    # an object category on each side, e.g. 卡车附近的小汽车.
    for phrase in ('附近','旁边','近旁'):
        if phrase in text:
            before,after=text.split(phrase,1)
            anchors,targets=infer_query_categories(before),infer_query_categories(after)
            if anchors and targets:
                result.update(relation='near',anchor_categories=anchors,target_categories=targets)
                break
    if not result['target_categories']:
        result['target_categories']=infer_query_categories(text)
    number=re.search(r'(\d+(?:\.\d+)?)\s*(?:米|m\b)',text)
    if number:
        result['near_m']=float(number.group(1))
    if not result['target_categories']:
        result.update(supported=False,reason='unsupported_target_category')
    return result


def label_matches(stored,requested):
    return stored in INCLUDES[requested]


def pair_matches(edges,subject,reference,requested,*,near_m=6.,positions=None):
    from conceptgraph.scenegraph.relation_single8_20261125 import query_pair
    if subject==reference:
        return False,None
    description=query_pair(edges,subject,reference)
    if requested=='near':
        if description is not None:
            distance=float(description['distance_m'])
        elif positions is not None:
            # Distance queries can exceed the12m edge adjacency limit without
            # inserting an extra edge or changing any directional sector.
            distance=float(np.linalg.norm(np.asarray(positions[subject])[:2]-np.asarray(positions[reference])[:2]))
        else:
            return False,None
        return bool(distance<=near_m),dict(subject=subject,reference=reference,distance_m=distance,
                                          evidence='stored_pair_distance' if description else 'existing_node_positions_distance_only')
    return bool(description is not None and label_matches(description['relation'],requested)),description


def select(protocol,query,records,predictions,pose,idx,stamp,*,confirmed=False):
    from conceptgraph.scenegraph.relation_single8_20261125 import build_edges_from_centers
    parsed=parse_query(query);forward=np.asarray(pose)[:2,2];forward=forward/np.linalg.norm(forward)
    positions={r['node_id']:r['center_world_m'] for r in records}
    if len(positions)!=len(records):raise AssertionError('duplicate_input_node_identity')
    edges=build_edges_from_centers([dict(id=r['node_id'],center=r['center_world_m']) for r in records],forward,max_distance_m=12.)
    anchors=[r for r in records if r['category'] in parsed['anchor_categories']]
    allowed={r['original_index']:True for r in records}
    if confirmed:
        from conceptgraph.completion.relation_nav_runtime_20261125 import adapt_nodes
        safe=copy.deepcopy(records)
        for node in safe:
            original=predictions[node['original_index']]
            assert node['node_id']==original['node_id'];node['legacy_prediction']=original
        _,eligibility,indices=adapt_nodes(safe,{})
        allowed={indices[i]:v for i,v in eligibility.items()}
    result=dict(status='NO_VISIBLE_CATEGORY_TARGET',selected_id=None,selected_original_index=None,selected_source_node_id=None,
        candidate_ids=[],eligible_original_indices=[],visible_original_indices=[r['original_index'] for r in records],
        static_condition_requested=parsed['requires_static'],static_condition_verified=False,
        selection_kind='confirmed_static_selection' if confirmed else 'object_retrieval_diagnostic',
        motion_filter_applied=confirmed,GT_in_method=False,Qwen_calls=0,vehicle_safety_authorized=False,trajectory_generated=False,
        relation_contract='single8_45deg_source_relative_target',reference_frame='CAM_FRONT_optical_forward_XY_proxy',
        reference_original_indices=[r['original_index'] for r in anchors],graph_pair_edges=len(edges))
    if not parsed['supported']:
        result.update(status='UNSUPPORTED_QUERY',reason=parsed['reason']);return result
    if parsed['relation']:
        if not anchors:result['status']='NO_REFERENCE_CANDIDATE';return result
        if protocol=='unified1200' and len(anchors)>1:
            result['status']='NEED_REFERENCE_CLARIFICATION';return result
    candidates=[]
    for node in records:
        if node['category'] not in parsed['target_categories'] or not allowed[node['original_index']]:continue
        witnesses=[]
        if parsed['relation']:
            for anchor in anchors:
                matched,evidence=pair_matches(edges,node['node_id'],anchor['node_id'],parsed['relation'],near_m=parsed['near_m'],positions=positions)
                if matched:witnesses.append(dict(reference_original_index=anchor['original_index'],pair=evidence))
            if not witnesses:continue
        distance=float(np.linalg.norm(np.asarray(node['center_world_m'])[:2]-np.asarray(pose)[:2,3]))
        candidates.append((distance,int(node['original_index']),node,witnesses))
    candidates.sort(key=lambda v:(v[0],v[1]))
    result.update(candidate_ids=[v[1] for v in candidates],eligible_original_indices=[v[1] for v in candidates],
                  candidate_scores=[dict(original_index=v[1],distance_m=v[0],relation_witnesses=v[3]) for v in candidates])
    if candidates:
        _,index,node,_=candidates[0]
        result.update(status='CONFIRMED_STATIC_TARGET' if confirmed else 'OBJECT_RETRIEVAL_PROPOSAL',
            selected_id=index,selected_original_index=index,selected_source_node_id=node['node_id'],
            static_condition_verified=confirmed,
            selected_motion_evidence=dict(estimated=node.get('estimated_motion_state',node.get('legacy_motion_state','unknown')),
                                         verified=node.get('verified_motion_state',node.get('motion_state','unknown'))))
    return result
