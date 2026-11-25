"""Independently partition all saved tasks; no inference or planning."""
from collections import Counter
from datetime import datetime, timezone

from conceptgraph.completion.navstage2_independent_audit_20261124 import (
    DATA, PACKAGE, ROOT, RUN, BASE, REPORT, association, digest, dump, read, rows,
)


def main():
    tasks=rows(DATA/'method_inputs.jsonl');saved=rows(RUN/'method_predictions.jsonl')
    labels={r['task_id']:r for r in rows(PACKAGE/'evaluation_only/conditional_reference_labels.jsonl')}
    truth={(r['scene'],r['frame_idx']):r for r in rows(BASE/'annotation/reserved_official_annotation_001/evaluation_only_frames.jsonl')}
    results={r['task_id']:r for r in read(REPORT/'stage2_independent_v1.json')['independent_evaluation']}
    registered={r['task_id']:r for r in rows(PACKAGE/'task_registry.jsonl')}
    production={r['task_id']:r for r in rows(ROOT/'outputs/reports/navstage2_20261124/failure_diagnostics_v2.jsonl')}
    details=[];counts=Counter();errors=[];route_logs=set();route_scenes=set()
    for task,pred in zip(tasks,saved):
        task_id=task['task_id'];accepted=labels[task_id]['source_query']['accepted_instance_tokens']
        identities=association(task['prediction_frame']['objects'],truth[task['scene'],task['issue_frame_idx']]['objects'])
        covered=any(v in accepted for v in identities.values());evaluated=results[task_id]
        if not covered:stage='no_accepted_object_associated_in_predicted_graph'
        elif not evaluated['candidate_target_covered']:stage='accepted_object_present_but_not_semantic_candidate'
        elif not evaluated['selected_correct']:stage='correct_candidate_present_but_no_verified_selection'
        elif evaluated['broad_semantic_geometry_success']:stage='broad_semantic_geometry_passed'
        else:stage='correct_selection_but_broad_terminal_geometry_failed'
        if stage!=production[task_id]['stage']:errors.append(task_id)
        if evaluated['predicted_route']:
            route_logs.add(registered[task_id]['log_token']);route_scenes.add(task['scene'])
        counts[stage]+=1
        details.append(dict(task_id=task_id,stage=stage,graph_accepted_target_covered=covered))
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=not errors and len(details)==81,
        denominator=81,counts=dict(counts),production_difference=errors,details=details,
        predicted_route_scene_count=len(route_scenes),predicted_route_log_count=len(route_logs),
        route_scenes=sorted(route_scenes),route_logs=sorted(route_logs),
        interpretation='49/28 are task-level association/candidate stages, not whole-map recall or proof of a causal remedy. Motion evidence is a source-field observation; no gate relaxation or new score was tested.',
        predictions_sha256=digest(RUN/'method_predictions.jsonl'),planner_calls=0,model_calls=0,GPU_calls=0)
    dump(REPORT/'stage2_failure_partition_v1.json',result)
    print({k:result[k] for k in ['passed','denominator','counts','predicted_route_scene_count','predicted_route_log_count']})


if __name__=='__main__':main()
