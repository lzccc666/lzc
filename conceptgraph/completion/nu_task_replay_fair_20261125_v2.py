"""Two separate, preregistered outputs: geometry retrieval and confirmation.

Retrieval deliberately leaves the word 'parked' unverified.  It ranks the same
visible category/position/relation candidates for all six maps; unknown motion
does not become static.  Confirmation uses the frozen stricter v1 interface.
Neither output implies a drivable route, parking authorization or arrival.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from conceptgraph.completion import nu_task_replay_20261125 as base

AMENDMENT = base.DATA/'fair_comparison_amendment_v2.json'
CHANNELS = ('object_retrieval_diagnostic', 'confirmed_static_selection')
ORIGINAL_SELECT = base.select
POLICY = dict(configuration_count=1, seed=17, no_new_prediction_scores_seen_before_freeze=True,
    reason='single_frame and original_fusion lack audited raw motion confirmation; cannot use their unsupported confirmation as a zero retrieval baseline',
    object_retrieval_diagnostic=dict(
        category='parsed target category; identical rule for all methods',
        motion='no motion filter or motion penalty; static/dynamic/unknown all participate; static condition remains unverified',
        relation='source-target delta dot current front-camera forward/left; directional distance<=12m, absolute component>1m; near8 old or near6 new',
        reference='legacy1115 exists-any visible category reference; unified1200 multiple visible reference nodes requests clarification',
        ranking='nearest world XY to current front camera ground origin, deterministic original_index tie break',
        interpretation='object retrieval diagnostic under original target-ID labels; not completion of the static semantic task'),
    confirmed_static_selection=dict(method='frozen v1 verified-motion and raw-evidence gate; dense retains old depth gate',
        unsupported_baselines=['single_frame','original_fusion'],
        interpretation='separate confirmation contract; unsupported baselines are not a claimed retrieval improvement'),
    labels='all original 2315 source tasks and 60 pending unchanged',GT_in_method=False,GPU_calls=0,Qwen_calls=0)


def register():
    base.verify(method_only=True)
    paths=[Path(__file__),base.ROOT/'scripts/nu_task_replay_fair_20261125_v2.py']
    values=dict(created_utc=datetime.now(timezone.utc).isoformat(),policy=POLICY,
        original_manifest_sha256=base.sha(base.DATA/'manifest.json'),
        runtime_source_sha256={str(p):base.sha(p) for p in paths},
        original_v1_retained=True, results_seen=False)
    if AMENDMENT.exists():
        old=json.loads(AMENDMENT.read_text())
        for field in ('policy','original_manifest_sha256','runtime_source_sha256'):
            if values[field]!=old[field]:raise AssertionError('fair_runtime_amendment_changed')
    else:base.write(AMENDMENT,values)


def relation_holds(relation, delta, forward, near):
    distance=float(np.linalg.norm(delta))
    if relation=='near':return distance<=near
    if distance>12.:return False
    left=np.array([-forward[1],forward[0]])
    axis=float(delta@(left if relation in ('left_of','right_of') else forward))
    if relation in ('left_of','in_front_of'):return axis>1.
    if relation in ('right_of','behind'):return axis < -1.
    raise ValueError(f'unsupported_relation:{relation}')


def retrieve(protocol, query, records, predictions, pose, idx, stamp):
    from conceptgraph.navigation.semantic import parse_navigation_query,infer_query_categories
    near=base.POLICY['near_distance_m'][protocol]
    parsed=parse_navigation_query(query,near)
    categories=set(parsed.target_categories)
    anchors_categories=set(infer_query_categories(parsed.anchor_text or ''))
    anchors=[r for r in records if r['category'] in anchors_categories]
    result=dict(status='NO_VISIBLE_CATEGORY_TARGET',selected_id=None,selected_original_index=None,
        selected_source_node_id=None,candidate_ids=[],visible_original_indices=[r['original_index'] for r in records],
        eligible_original_indices=[],GT_in_method=False,Qwen_calls=0,
        static_condition_requested=bool(parsed.requires_static),static_condition_verified=False,
        motion_filter_applied=False,motion_state_not_overridden=True,
        selection_kind='object_retrieval_diagnostic',vehicle_safety_authorized=False,trajectory_generated=False,
        reference_frame='ego_heading',reference_frame_source='CAM_FRONT optical forward XY proxy; not object body heading',
        reference_original_indices=[r['original_index'] for r in anchors])
    if not categories:
        result['status']='UNSUPPORTED_TARGET_CATEGORY';return result
    if parsed.relation and anchors_categories:
        if not anchors:result['status']='NO_REFERENCE_CANDIDATE';return result
        if protocol=='unified1200' and len(anchors)>1:
            result['status']='NEED_REFERENCE_CLARIFICATION';return result
    forward=pose[:2,2];forward=forward/np.linalg.norm(forward);origin=pose[:2,3]
    candidates=[]
    for record in records:
        if record['category'] not in categories:continue
        witnesses=[]
        if parsed.relation:
            for anchor in anchors:
                if anchor['original_index']==record['original_index']:continue
                delta=np.asarray(record['center_world_m'][:2])-np.asarray(anchor['center_world_m'][:2])
                if relation_holds(parsed.relation,delta,forward,near):witnesses.append(anchor['original_index'])
            if not witnesses:continue
        distance=float(np.linalg.norm(np.asarray(record['center_world_m'][:2])-origin))
        candidates.append((distance,int(record['original_index']),record,witnesses))
    candidates.sort(key=lambda x:(x[0],x[1]))
    result.update(candidate_ids=[x[1] for x in candidates],eligible_original_indices=[x[1] for x in candidates],
        candidate_scores=[dict(original_index=x[1],distance_m=x[0],score=1/(1+x[0]),
            relation_witness_original_indices=x[3]) for x in candidates])
    if candidates:
        record=candidates[0][2];i=record['original_index']
        result.update(status='OBJECT_RETRIEVAL_PROPOSAL',selected_id=i,selected_original_index=i,
            selected_source_node_id=record['node_id'],
            selected_motion_evidence=dict(estimated=record.get('estimated_motion_state',record.get('legacy_motion_state','unknown')),
                verified=record.get('verified_motion_state',record.get('motion_state','unknown')),
                measurement_source_closed=record.get('measurement',{}).get('source_closed',False)))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','infer','evaluate'))
    parser.add_argument('--scene');parser.add_argument('--frame-file');args=parser.parse_args()
    register()
    if args.action=='prepare':print(json.dumps(dict(channels=CHANNELS,amendment=str(AMENDMENT))));return
    for channel in CHANNELS:
        base.RUN_ID='nu_task_replay_20261125_v2_'+('retrieval' if channel==CHANNELS[0] else 'confirmation')
        base.select=retrieve if channel==CHANNELS[0] else ORIGINAL_SELECT
        if args.action=='infer':base.infer(args.scene,args.frame_file)
        else:base.evaluate(args.scene)


if __name__=='__main__':main()
