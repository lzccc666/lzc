"""只读已冻结预测的失败证据，不重跑规划、不修改条件；官方信息仅事后诊断。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import Counter
import json
import math
import numpy as np
from shapely.geometry import Point, shape
from conceptgraph.completion.evidence import target_evidence_decision
from conceptgraph.completion.navigation_diagnostics import to_nodes
from conceptgraph.navigation.semantic import parse_navigation_query
from conceptgraph.navigation.navstage2_runner_20261124 import ROOT,BASE,INPUT,RUN_ID,PACKAGE,sha,read,rows,dump
from conceptgraph.nuscenes.benchmark_ground_truth import associate_nodes
from conceptgraph.project_paths import experiment_directory

dest=experiment_directory('nuscenes','test',RUN_ID)
inputs={r['task_id']:r for r in rows(INPUT/'method_inputs.jsonl')}
labels={r['task_id']:r for r in rows(PACKAGE/'evaluation_only/conditional_reference_labels.jsonl')}
truths={(r['scene'],r['frame_idx']):r for r in rows(BASE/'annotation/reserved_official_annotation_001/evaluation_only_frames.jsonl')}
predictions=rows(dest/'method_predictions.jsonl')
evaluations={r['task_id']:r for r in rows(dest/'evaluation_only/queries.jsonl')}
policy=dict(enabled=True,minimum_observations=3,minimum_lidar_pixels=5,minimum_depth_support_fraction=.4)
counts=Counter();diagnostics=[]
for pred in predictions:
    task=inputs[pred['task_id']];label=labels[pred['task_id']];evaluation=evaluations[pred['task_id']]
    objects=task['prediction_frame']['objects'];truth=truths[pred['scene'],pred['frame_idx']]['objects']
    binding=associate_nodes(objects,truth,maximum_distance_m=3.,ambiguity_margin_m=.5)
    accepted=set(label['source_query']['accepted_instance_tokens'])
    correct={r['node_id'] for r in binding['matches'] if r['instance_token'] in accepted}
    nodes=to_nodes(objects);parsed=parse_navigation_query(task['instruction'],8.)
    evidence=[]
    for i,obj in enumerate(objects):
        if obj['node_id'] in correct:
            allowed,reason=target_evidence_decision(nodes[i],policy)
            evidence.append(dict(node_id=obj['node_id'],motion_state=obj['motion_state'],
                num_detections=obj['num_detections'],evidence_accepted=allowed,evidence_reason=reason,
                parsed_requires_static=parsed.requires_static,
                included_in_semantic_candidates=any(c['node_id']==obj['node_id'] for c in pred['semantic_candidates'])))
    if not correct:
        stage='no_accepted_object_associated_in_predicted_graph'
    elif not evaluation['candidate_target_covered']:
        stage='accepted_object_present_but_not_semantic_candidate'
    elif not evaluation['selected_correct']:
        stage='correct_candidate_present_but_no_verified_selection'
    elif not evaluation['broad_semantic_geometry_success']:
        stage='correct_selection_but_broad_terminal_geometry_failed'
    else:
        stage='broad_semantic_geometry_passed'
    counts[stage]+=1
    diagnostic=dict(task_id=pred['task_id'],status=pred['status'],stage=stage,
        accepted_object_predicted_graph_coverage=bool(correct),accepted_node_evidence=evidence,
        attempts=pred['planning_attempts'])
    if pred['predicted_rows']:
        end=pred['predicted_rows'][-1]['state'];context=read(BASE/'annotation/reference_regions_reserved_full_003/contexts'/f"{pred['scene']}_{pred['frame_idx']:04d}.json")
        diagnostic['terminal']=dict(final_state=end,
            distances_to_original_accepted_gt_footprints=[dict(instance=o['instance_token'],
                rear_axle_distance_m=shape(o['polygon_geojson']).distance(Point(end['x'],end['y'])))
                for o in context['objects'] if o['instance_token'] in accepted])
    diagnostics.append(diagnostic)
report=ROOT/'outputs/reports/navstage2_20261124'
(report/'failure_diagnostics_v2.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in diagnostics))
dump(report/'failure_diagnostics_v2_summary.json',dict(denominator=81,counts=dict(counts),
    accepted_object_predicted_graph_coverage=sum(d['accepted_object_predicted_graph_coverage'] for d in diagnostics),
    evidence_reason_counts=dict(Counter(e['evidence_reason'] for d in diagnostics for e in d['accepted_node_evidence'])),
    source_sha256=sha(__file__),prediction_sha256=sha(dest/'method_predictions.jsonl'),
    policy_change=False,new_planning_runs=0,GT_use='evaluation only; not fed back to method'))
print(json.dumps(dict(counts),ensure_ascii=False))
