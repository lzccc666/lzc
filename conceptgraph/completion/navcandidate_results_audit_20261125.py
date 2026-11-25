"""Independently recompute saved position evidence and all validation choices."""
from __future__ import annotations

from collections import Counter,defaultdict
import argparse
from datetime import datetime,timezone
import gzip
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

from conceptgraph.completion.navcandidate_independent_20261125 import ROOT,REPORT,read,sha,write,protection
from conceptgraph.completion.selection_map_transfer import nodes_from_frozen_predictions,select_frozen_map,EVIDENCE_POLICY
from conceptgraph.completion.evidence import target_evidence_decision
from conceptgraph.navigation.semantic import parse_navigation_query,rank_candidate_targets
from conceptgraph.scenegraph.relations import build_relations

DATA=ROOT/'dataset/processed/mf_position_20261125_v2'
RUN=ROOT/'outputs/results/nuscenes/validation/iterations/mf_position_20261125_v2'
R1=ROOT/'outputs/results/nuscenes/validation/iterations/recall_refinement_R1_full_20261120_v2'
TRANSFER=ROOT/'outputs/results/qwen/validation/iterations/selection_r1_map_transfer_20261123_v1'


def rows(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def independent_confirm(state,timestamp):
    trace=state['raw_trace'];times=[r['timestamp_s'] for r in trace];frames=[r['frame_idx'] for r in trace];keys=[r['key'] for r in trace];reasons=[]
    if len(trace)<3 or any(len(set(values))!=len(trace) for values in [times,frames,keys]):reasons.append('insufficient_distinct_raw_measurements')
    if not times or any(b<=a for a,b in zip(times,times[1:])) or any(t>timestamp for t in times):reasons.append('noncausal_or_unordered_raw_history')
    span=times[-1]-times[0] if times else 0.
    if span<1.-1e-8:reasons.append('raw_span_below_one_second')
    if not times or abs(times[-1]-timestamp)>1e-6:reasons.append('no_current_real_update')
    last=trace[-1] if trace else {};depth=last.get('depth_evidence',{})
    if depth.get('source')!='current_projected_lidar' or depth.get('units')!='m' or depth.get('retained_pixels',0)<1 or depth.get('support_fraction',0)<.4:reasons.append('current_raw_support_invalid')
    if any(not r.get('assignment',{}).get('unambiguous',False) for r in trace):reasons.append('ambiguous_assignment_in_raw_history')
    speed=None
    if len(times)>=2 and all(b>a for a,b in zip(times,times[1:])):
        vectors=[(np.asarray(b['raw_measured_center'])[:2]-np.asarray(a['raw_measured_center'])[:2])/(b['timestamp_s']-a['timestamp_s']) for a,b in itertools.combinations(trace,2) if b['timestamp_s']-a['timestamp_s']>=.75]
        vector=np.median(vectors,axis=0) if vectors else (np.asarray(trace[-1]['raw_measured_center'])[:2]-np.asarray(trace[-2]['raw_measured_center'])[:2])/(times[-1]-times[-2])
        speed=float(np.linalg.norm(vector))
    if speed is None or speed>.5:reasons.append('raw_motion_not_confirmed_static')
    variance=state['variance_diag']
    if not all(math.isfinite(v) and v>0 for v in variance) or max(variance[:2])>4.:reasons.append('heuristic_position_variance_too_large')
    return dict(confirmed=not reasons,reasons=reasons,speed=speed,span_s=span)


def independent_rank(query,predictions,pose,frame_idx,qualified):
    nodes=nodes_from_frozen_predictions(predictions);forward=pose[:2,2]
    edges=build_relations(nodes,dict(coordinate_frame='ego_heading',directional_mode='all_supported_axes'),frame_index=frame_idx,ego_heading=forward).edges
    retained=set()
    for i,node in nodes.items():
        if predictions[i].get('shape_available') is False:
            if predictions[i]['node_id'] in qualified:
                node.motion_state='static';node.is_short_term=False;retained.add(i)
        elif target_evidence_decision(node,EVIDENCE_POLICY)[0]:retained.add(i)
    parsed=parse_navigation_query(query,8.);parsed.vehicle_heading_xy=forward
    ranked=rank_candidate_targets(parsed,nodes,edges,pose[:2,3],.55,.3,.25,relation_coordinate_frame='ego_heading',relation_coordinate_frame_fallback=None,evidence_policy=None)
    return [r for r in ranked if r.node_id in retained]


def main():
    global DATA,RUN
    parser=argparse.ArgumentParser();parser.add_argument('--version',default='v2');args=parser.parse_args()
    DATA=ROOT/f'dataset/processed/mf_position_20261125_{args.version}'
    RUN=ROOT/f'outputs/results/nuscenes/validation/iterations/mf_position_20261125_{args.version}'
    begun=time.monotonic();manifest=read(DATA/'manifest.json');errors=[];checks=[];snapshots=[]
    for group in ['source_sha256','input_sha256','evaluation_only_sha256']:
        for path,expected in manifest[group].items():checks.append(dict(path=path,passed=sha(path)==expected))
    for path,expected in manifest['source_sha256'].items():
        p=RUN/'source_snapshot'/Path(path).relative_to(ROOT);snapshots.append(dict(path=str(p),passed=sha(p)==expected))
    frames=list(rows(DATA/'method_frames.jsonl'));frame_inputs={(r['scene'],r['frame_idx']):r for r in frames};queries=list(rows(DATA/'method_queries.jsonl'))
    registry=ROOT/'outputs/project_v2/thesis_completion_v1/main_experiments/causal_retrieval_full_001/queries.jsonl'
    labels={(r['scene'],r['frame_idx'],r['query_index']):r for r in rows(registry) if r['method']=='existing_object_joint_rule'}
    expected_queries=[{k:r[k] for k in ['scene','frame_idx','query_index','query_text']} for r in labels.values()]
    if queries!=expected_queries or len(queries)!=1115 or sum(r['positive'] for r in labels.values())!=733:errors.append('method_query_denominator_or_label_leak')
    frame_whitelist={'scene','frame_idx','timestamp_s','rgb','depth','detections','intrinsic','camera_pose'}
    if any(set(r)!=frame_whitelist for r in frames):errors.append('method_frame_field_leak')
    scenes={r['scene'] for r in frames};logs={r['scene']:r['log_token'] for r in manifest['scenes']}
    history=list(rows(ROOT/'dataset/processed/navstage2_20261124_v2/method_inputs.jsonl'));official_logs={r['name']:r['log_token'] for r in read(ROOT/'dataset/nuscenes_trainval_raw/v1.0-trainval/scene.json')}
    overlap_scene=scenes&{r['scene'] for r in history};overlap_log={logs[s] for s in scenes}&{official_logs[r['scene']] for r in history}
    if len(scenes)!=38 or len({logs[s] for s in scenes})!=11 or overlap_scene or overlap_log:errors.append('split_or_historical_overlap')
    old={};truthframes={}
    for scene in scenes:
        for r in rows(R1/scene/'frames.jsonl.gz'):
            old[scene,r['frame_idx']]=r['predictions']['R1']
            if any(q['scene']==scene and q['frame_idx']==r['frame_idx'] for q in queries):truthframes[scene,r['frame_idx']]=r
    saved_queries=list(rows(RUN/'queries.jsonl'));saved={(r['method'],r['scene'],r['frame_idx'],r['query_index']):r for r in saved_queries}
    original={(r['scene'],r['frame_idx'],r['query_index']):r for r in rows(TRANSFER/'queries.jsonl') if r['method']=='R1_map_original_rule'}
    qgroups=defaultdict(list)
    for q in queries:qgroups[q['scene'],q['frame_idx']].append(q)
    totals=defaultdict(Counter);bylog=defaultdict(lambda:defaultdict(Counter));state_totals=Counter();event_totals=Counter();frame_totals=Counter();raw_by_key={};dense_outcomes={};qualified_records=[];raw_checks=[];frame_details=[];seen=[];last_times={};state_histories={};retired=set();pred_compare_failures=[]
    for output in rows(RUN/'frames.jsonl.gz'):
        key=output['scene'],output['frame_idx'];seen.append(key);source=frame_inputs[key];timestamp=output['timestamp_s'];ledger=output['ledger'];predictions=output['predictions']
        exact=predictions==old[key];frame_totals['frames']+=1;frame_totals['predictions_exact']+=int(exact)
        if not exact:pred_compare_failures.append(key)
        if output['GT_in_method'] or output['vehicle_safety_authorized']:errors.append('frame_gt_or_safety_claim')
        if timestamp!=source['timestamp_s'] or timestamp<=last_times.get(key[0],-math.inf):errors.append('frame_time')
        last_times[key[0]]=timestamp
        for raw in ledger['raw_observations']:
            raw_by_key[raw['key']]=raw
            if raw['frame_idx']!=key[1] or raw['timestamp_s']!=timestamp or raw['scene']!=key[0] or raw['filtered_position_used_as_measurement'] or raw['input_source']!='current_actual_depth_pixels':errors.append('raw_source_or_time')
            if raw['measured_points']!=raw['depth_evidence']['retained_pixels']:errors.append('raw_point_vs_evidence')
        for event in ledger['events']:
            event_totals[event['kind']]+=1
            assignment=event.get('assignment',{})
            if event['kind']=='sparse_dense_assignment_outcome':dense_outcomes[event['observation_key']]=event
            if assignment.get('stage') in ['sparse_to_sparse','sparse_to_dense'] and assignment.get('unambiguous')!=(assignment.get('row_feasible')==1 and assignment.get('column_feasible')==1 and not assignment.get('unresolved_prior_dense_competition',False)):errors.append('ambiguous_assignment_mislabeled')
            if event['kind']=='original_sparse_dense_handoff':
                retired.add((key[0],event['sparse_state_uid']))
                if event['history_merged']:errors.append('handoff_history_merged')
        confirmed={}
        for state in ledger['sparse_states']:
            independent=independent_confirm(state,timestamp);state_totals['state_frames']+=1;state_totals['confirmed_state_frames']+=int(independent['confirmed'])
            for reason in independent['reasons']:state_totals['reject:'+reason]+=1
            if independent['confirmed']!=state['confirmed'] or independent['reasons']!=state['reasons']:errors.append('independent_confirmation_difference')
            if state['semantic_position_candidate']!=independent['confirmed'] or state['vehicle_safety_authorized'] or state['history_cross_id_merged']:errors.append('position_safety_or_merge')
            if (key[0],state['state_uid']) in retired:errors.append('retired_sparse_still_active')
            statekey=(key[0],state['state_uid']);trace=state['raw_trace']
            for raw in trace:
                original_raw=raw_by_key.get(raw['key']);copy={k:v for k,v in raw.items() if k!='assignment'}
                if original_raw!=copy:errors.append('raw_trace_not_original_measurement')
                assignment=raw.get('assignment',{})
                if assignment.get('stage') in ['new_sparse','sparse_to_sparse']:
                    prior=dense_outcomes.get(raw['key'])
                    if prior is None:errors.append('missing_prior_dense_assignment_outcome')
                    else:
                        unresolved=prior['dense_row_feasible']>0 and not prior['dense_assigned']
                        if assignment.get('unresolved_prior_dense_competition')!=unresolved or assignment.get('prior_dense_row_feasible')!=prior['dense_row_feasible'] or assignment.get('prior_dense_assigned')!=prior['dense_assigned']:errors.append('prior_dense_competition_not_propagated')
                        if unresolved and assignment.get('unambiguous'):errors.append('unresolved_prior_competition_qualified')
                    if assignment['stage']=='new_sparse' and assignment.get('unambiguous')!=(assignment.get('row_feasible')==0 and not assignment.get('unresolved_prior_dense_competition',False)):errors.append('new_sparse_ambiguity_mislabeled')
            previous=state_histories.get(statekey,[]);current=[r['key'] for r in trace]
            if previous and not all(v in current or v in previous[:-8] for v in previous[-7:]) and len(current)<8:errors.append('state_lineage_reset')
            state_histories[statekey]=current
            if independent['confirmed']:
                confirmed[state['node_id']]=state
                qualified_records.append(dict(scene=key[0],frame_idx=key[1],node_id=state['node_id'],state_uid=state['state_uid'],raw_keys=current,speed=independent['speed'],span_s=independent['span_s'],current_pixels=trace[-1]['depth_evidence']['retained_pixels']))
        if key not in qgroups:continue
        pose=np.asarray(source['camera_pose']);gtframe=truthframes[key];truth=gtframe['evaluation_only_official'];binding={m['prediction_index']:truth[m['truth_index']]['instance_token'] for m in gtframe['metrics']['R1']['matches']}
        for query in qgroups[key]:
            qid=(*key,query['query_index']);label=labels[qid];accepted=set(label['acceptable_instances'])
            baseline=select_frozen_map(query['query_text'],predictions,pose,key[1]);candidate=independent_rank(query['query_text'],predictions,pose,key[1],confirmed)
            for method,ranked in [('R1_original_rule',baseline),('R1_multiframe_position_rule',candidate)]:
                selected=ranked[0].node_id if ranked else None;identity=binding.get(selected);stored=saved[(method,*qid)];pos=label['positive'];correct=identity in accepted if pos else None;covered=any(binding.get(c.node_id) in accepted for c in ranked) if pos else None
                if stored['candidate_ids']!=[r.node_id for r in ranked] or stored['selected_index']!=selected or stored['selected_identity']!=identity or stored['selected_correct']!=correct or stored['candidate_covered']!=covered:errors.append('independent_choice_or_score_difference')
                if stored['vehicle_safety_authorized'] or stored['actual_arrival_credit'] or stored['GT_in_method']:errors.append('task_scope_claim')
                if method=='R1_original_rule' and (original[qid]['candidate_ids']!=[r.node_id for r in ranked] or original[qid]['selected_node_id']!=selected):errors.append('baseline_not_original')
                if selected is not None and predictions[selected].get('shape_available') is False and predictions[selected]['node_id'] not in confirmed:errors.append('unconfirmed_sparse_selected')
                increments=dict(tasks=1,positive=int(pos),negative=int(not pos),positive_selected_correct=int(bool(correct)),positive_raw_covered=int(any(v in accepted for v in binding.values()) if pos else 0),positive_candidate_covered=int(bool(covered)),negative_false_selection=int(not pos and selected is not None),rejected=int(selected is None),position_only_selected=int(selected is not None and predictions[selected].get('shape_available') is False),vehicle_safety_authorized=0)
                totals[method].update(increments);bylog[logs[key[0]]][method].update(increments)
    if seen!=[(r['scene'],r['frame_idx']) for r in frames] or len(seen)!=1508 or pred_compare_failures:errors.append('all1508_prediction_or_order')
    if len(saved_queries)!=2230 or len(saved)!=2230:errors.append('full1115_pair')
    summary=read(RUN/'selection_summary.json')
    if {k:dict(v) for k,v in totals.items()}!=summary['methods']:errors.append('summary_totals')
    if {k:{m:dict(v) for m,v in methods.items()} for k,methods in bylog.items()}!=summary['by_log']:errors.append('per_log_summary')
    preserved=protection()
    if not preserved['unchanged']:errors.append('protected_changed')
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=not errors and all(r['passed'] for r in checks+snapshots),failures=sorted(set(errors)),hash_checks=checks,snapshot_checks=snapshots,
        split=dict(scenes=38,logs=11,frames=1508,queries_per_method=1115,positive=733,negative=382,historical_scene_overlap=sorted(overlap_scene),historical_log_overlap=sorted(overlap_log)),frame_counts=dict(frame_totals),state_counts=dict(state_totals),event_counts=dict(event_totals),
        independent_methods={k:dict(v) for k,v in totals.items()},independent_by_log={k:{m:dict(v) for m,v in methods.items()} for k,methods in bylog.items()},qualified_state_frames=qualified_records,protection=preserved,elapsed_s=time.monotonic()-begun,
        limitations=['Qualification is a new position-only semantic policy, not the original per-frame five-point gate or vehicle-safety authorization.', 'Heuristic covariance is not a calibrated probability; no shape is inferred.', 'Same38 already seen validation scenes; no unseen generalization or route/arrival claim.', 'Full candidate/time/source/choice accounting recomputed; raw pixel reprojection, if performed, is recorded in a separate audit.'],model_calls=0,training_calls=0,planner_calls=0,GPU_calls=0)
    write(REPORT/f'independent_results_{args.version}.json',result)
    print(json.dumps({k:result[k] for k in ['passed','failures','split','frame_counts','state_counts','independent_methods','elapsed_s']},ensure_ascii=False))


if __name__=='__main__':main()
