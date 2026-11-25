"""Audit frozen validation counts and the separate 28-task historical explanation."""
from collections import Counter,defaultdict
from dataclasses import asdict
from datetime import datetime,timezone
import gzip
import json
import math
from pathlib import Path

from conceptgraph.completion.navfollowup_independent_20261124 import ROOT,REPORT,read,sha,write,protection
from conceptgraph.completion.navstage2_independent_audit_20261124 import association
from conceptgraph.navigation.semantic import parse_navigation_query


def rows(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else Path(path).open()) as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def main():
    folder=ROOT/'dataset/processed/navstage2_repair_20261124_v1';run=ROOT/'outputs/results/nuscenes/validation/iterations/navstage2_repair_20261124_diagnosis_v1'
    manifest=read(folder/'manifest.json');summary=read(run/'summary.json');errors=[];hashes=[]
    for group in ['source_sha256','input_sha256']:
        for path,expected in manifest[group].items():hashes.append(dict(path=path,passed=sha(path)==expected))
    snapshots=[]
    for path,expected in manifest['source_sha256'].items():
        p=run/'source_snapshot'/Path(path).relative_to(ROOT);snapshots.append(dict(path=str(p),passed=sha(p)==expected))
    query_rows=list(rows(folder/'validation_queries.jsonl'));queries={(r['scene'],r['frame_idx'],r['query_index']):r for r in query_rows}
    original=[r for r in rows(ROOT/'outputs/project_v2/thesis_completion_v1/main_experiments/causal_retrieval_full_001/queries.jsonl') if r['method']=='existing_object_joint_rule']
    if query_rows!=original:errors.append('changed_validation_queries')
    if len(queries)!=1115 or sum(r['positive'] for r in query_rows)!=733:errors.append('validation_denominator')
    official_logs={r['name']:r['log_token'] for r in read(ROOT/'dataset/nuscenes_trainval_raw/v1.0-trainval/scene.json')}
    scenes={r['scene'] for r in query_rows};historic_scenes={r['scene'] for r in rows(ROOT/'dataset/processed/navstage2_20261124_v2/method_inputs.jsonl')}
    split=dict(validation_scenes=len(scenes),validation_logs=len({official_logs[s] for s in scenes}),scene_overlap=sorted(scenes&historic_scenes),log_overlap=sorted({official_logs[s] for s in scenes}&{official_logs[s] for s in historic_scenes}))
    if split['scene_overlap'] or split['log_overlap']:errors.append('validation_historical_overlap')
    if {s:official_logs[s] for s in scenes}!=manifest['scene_logs']:errors.append('scene_log_manifest')
    methods=manifest['methods'];counts=defaultdict(Counter);subgroups=defaultdict(Counter);node_counts=defaultdict(Counter);accepted_counts=defaultdict(Counter)
    task_rows=list(rows(run/'tasks.jsonl'));tasks={(r['method'],r['scene'],r['frame_idx'],r['query_index']):r for r in task_rows}
    previous={(r['method'],r['scene'],r['frame_idx'],r['query_index']):r for r in rows(ROOT/'outputs/results/qwen/validation/iterations/selection_r1_map_transfer_20261123_v1/queries.jsonl')}
    node_candidates=defaultdict(list);accepted_node_indices=defaultdict(set)
    for node in rows(run/'nodes.jsonl.gz'):
        method=node['method'];key=(method,node['scene'],node['frame_idx'],node['query_index']);fail=node['all_failed_gates']
        if node['first_failed_gate']!=(fail[0] if fail else None) or node['candidate']!=bool(not fail):errors.append('node_gate_consistency')
        if node['candidate']:node_candidates[key].append(node['prediction_index'])
        group=node_counts[method];group['nodes']+=1;group['first:'+(node['first_failed_gate'] or 'passed')]+=1
        for gate in fail:group['all:'+gate]+=1
        if node['evaluation_accepted_node']:
            accepted_node_indices[key].add(node['prediction_index']);group=accepted_counts[method]
            group['nodes']+=1;group['first:'+(node['first_failed_gate'] or 'passed')]+=1;group['motion:'+node['motion_state']]+=1
            for gate in fail:group['all:'+gate]+=1
    for key,task in tasks.items():
        method,scene,frame,query_index=key;query=queries[scene,frame,query_index];old=previous[key]
        if set(node_candidates[key])!=set(old['candidate_ids']):errors.append('candidate_set_vs_frozen_original')
        if task['selected_index']!=old['selected_node_id'] or task['selected_identity']!=old['evaluation_identity']:errors.append('selected_vs_frozen_original')
        traces=task['correct_node_traces']
        if {r['prediction_index'] for r in traces}!=accepted_node_indices[key]:errors.append('accepted_node_task_trace')
        correct=task['selected_identity'] in query['acceptable_instances'];has_candidate=any(r['candidate'] for r in traces)
        if not query['positive']:stage='negative_false_selection' if task['selected_index'] is not None else 'negative_correct_rejection'
        elif not traces:stage='no_accepted_object_in_graph'
        elif not has_candidate:stage='accepted_object_in_graph_but_no_candidate'
        elif not correct:stage='accepted_candidate_but_not_selected_correct'
        else:stage='selected_correct'
        if stage!=task['stage']:errors.append('mutually_exclusive_task_stage')
        counts[method][stage]+=1;counts[method]['tasks']+=1
        first_set=sorted({t['first_failed_gate'] for t in traces if t['first_failed_gate']})
        if stage=='accepted_object_in_graph_but_no_candidate':subgroups[method][' | '.join(first_set)]+=1
    for method in methods:
        if dict(counts[method])!=summary['methods'][method]['tasks']:errors.append('summary_tasks')
        if dict(subgroups[method])!=summary['methods'][method]['blocked_task_mutually_exclusive_subgroups']:errors.append('summary_subgroups')
        if dict(node_counts[method])!=summary['all_nodes'][method] or dict(accepted_counts[method])!=summary['accepted_nodes'][method]:errors.append('summary_node_counts')
    if len(tasks)!=2230 or len(task_rows)!=2230:errors.append('two_method_complete_1115')
    historic=list(rows(ROOT/'dataset/processed/navstage2_20261124_v2/method_inputs.jsonl'))
    evaluation={r['task_id']:r for r in rows(ROOT/'outputs/results/nuscenes/test/iterations/navstage2_20261124_v2/evaluation_only/queries.jsonl')}
    labels={r['task_id']:r for r in rows(ROOT/'dataset/processed/navbench_stage1_20261124_v1/evaluation_only/conditional_reference_labels.jsonl')}
    truth={(r['scene'],r['frame_idx']):r for r in rows(ROOT/'outputs/project_v2/thesis_completion_v1/annotation/reserved_official_annotation_001/evaluation_only_frames.jsonl')}
    blocked=[];historical_counts=Counter();historical_node_records=Counter()
    for task in historic:
        frame=task['prediction_frame'];accepted=labels[task['task_id']]['source_query']['accepted_instance_tokens']
        identities=association(frame['objects'],truth[task['scene'],task['issue_frame_idx']]['objects'])
        objects=[o for o in frame['objects'] if identities.get(o['node_id']) in accepted]
        if not objects or evaluation[task['task_id']]['candidate_target_covered']:continue
        parsed=parse_navigation_query(task['instruction'],8.);notes=[]
        for obj in objects:
            distance=math.dist(obj['center'][:2],[frame['camera_pose'][0][3],frame['camera_pose'][1][3]])
            first=('query_requires_confirmed_static' if parsed.requires_static and obj['motion_state']!='static' else 'minimum_target_distance' if parsed.min_target_distance_m is not None and distance<parsed.min_target_distance_m else 'other')
            historical_node_records[first]+=1
            notes.append(dict(node_id=obj['node_id'],motion_state=obj['motion_state'],num_detections=obj['num_detections'],distance_m=distance,first_failed_gate=first,depth_evidence=obj['depth_evidence']))
        first_set=sorted({r['first_failed_gate'] for r in notes});historical_counts[' | '.join(first_set)]+=1
        blocked.append(dict(task_id=task['task_id'],query=task['instruction'],parsed=asdict(parsed),correct_task_node_records=notes))
    if len(blocked)!=28 or historical_counts!=Counter(query_requires_confirmed_static=27,minimum_target_distance=1):errors.append('historical_28_partition')
    preserved=protection()
    if not preserved['unchanged']:errors.append('protected_changed')
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=not errors and all(r['passed'] for r in hashes+snapshots),failures=sorted(set(errors)),
        frozen_checks=hashes,snapshot_checks=snapshots,split=split,validation=dict(task_counts={k:dict(v) for k,v in counts.items()},blocked_task_subgroups={k:dict(v) for k,v in subgroups.items()},all_node_records={k:dict(v) for k,v in node_counts.items()},accepted_node_records={k:dict(v) for k,v in accepted_counts.items()},original_candidate_sets_and_final_choices_match=True),
        historical_28_explanation_only=dict(tasks=28,task_first_gate_partition=dict(historical_counts),task_node_record_first_gates=dict(historical_node_records),records=blocked,used_to_choose_fixes=False),
        new_code_fix_verified=False,no_model_or_planner_calls=True,protection=preserved)
    write(REPORT/'validation_and_historical_counts_v1.json',result)
    print(json.dumps(dict(passed=result['passed'],failures=result['failures'],split=split,task_counts=result['validation']['task_counts'],blocked_task_subgroups=result['validation']['blocked_task_subgroups'],historical_28=dict(historical_counts)),ensure_ascii=False))


if __name__=='__main__':main()
