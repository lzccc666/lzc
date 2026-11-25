"""Independently check the frozen task registry, certificates and denominators."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from conceptgraph.completion.navbench_stage1_independent_audit import ROOT,REPORT,digest,dump,now,verify_protection

DATA=ROOT/'dataset/processed/navbench_stage1_20261124_v1'


def rows(path):return [json.loads(line) for line in Path(path).open() if line.strip()]
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def content_sha(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report-name',default='package_independent_v1.json');parser.add_argument('--schema',default='task_record.schema.json');args=parser.parse_args()
    manifest=json.loads((DATA/'manifest.json').read_text());registry=rows(DATA/'task_registry.jsonl')
    certificates=rows(DATA/'evaluation_only/certificates.jsonl');inputs=rows(DATA/'method_inputs/candidate_tasks.jsonl')
    languages=rows(DATA/'method_inputs/unbound_language.jsonl');freeze=json.loads((DATA/'denominator_freeze.json').read_text())
    schema=json.loads((DATA/args.schema).read_text());validator=Draft202012Validator(schema)
    problems=[];hashes=[]
    for name,record in manifest['files'].items():hashes.append(dict(path=str(DATA/name),passed=digest(DATA/name)==record['sha256']))
    for group in ['source_sha256','input_sha256']:
        for path,expected in manifest[group].items():hashes.append(dict(path=path,passed=digest(path)==expected))
    for record in registry:
        errors=list(validator.iter_errors(record))
        if errors:problems.append(dict(task_id=record['task_id'],failure='schema',errors=[str(e.message) for e in errors]))
        if content_sha(record['contract'])!=record['contract_sha256']:problems.append(dict(task_id=record['task_id'],failure='contract_hash'))
    by_id={record['task_id']:record for record in registry};by_cert={record['certificate_id']:record for record in certificates}
    if len(by_id)!=len(registry) or len(by_cert)!=len(certificates):problems.append(dict(failure='duplicate_task_or_certificate_id'))
    expected_counts={'original_episode':6,'approved_language':16,'historical_retrieval':1115,'historical_conditional_reference':878}
    family=Counter(r['source_family'] for r in registry)
    if dict(family)!=expected_counts:problems.append(dict(failure='family_counts'))
    per_family={};per_log=defaultdict(Counter)
    for record in registry:
        per_log[record['log_token'] or 'unbound'][record['feasibility_status']]+=1
        if record['main_table_eligible'] and (record['feasibility_status']!='feasible' or record['action'] not in ['pull_over','park_nearest','follow_vehicle']):problems.append(dict(task_id=record['task_id'],failure='invalid_main_eligibility'))
        if record['feasibility_status']=='infeasible_proven' and not record['certificate_ids']:problems.append(dict(task_id=record['task_id'],failure='infeasible_without_proof'))
        if record['feasibility_status']=='feasible':
            if record['source_family']!='historical_conditional_reference' or not record['certificate_ids']:problems.append(dict(task_id=record['task_id'],failure='unsupported_feasible'))
        for cert_id in record['certificate_ids']:
            cert=by_cert.get(cert_id)
            if cert is None:problems.append(dict(task_id=record['task_id'],failure='unknown_certificate'));continue
            if cert['scene']!=record['scene'] or cert['frame_idx']!=record['issue_frame_idx'] or cert['task_contracts'].get(record['task_id'])!=record['contract_sha256']:problems.append(dict(task_id=record['task_id'],failure='certificate_task_contract_binding'))
            contract=record['contract']
            for field,cert_field in [('vehicle_model','vehicle'),('initial_speed_mps','initial_speed_mps'),('initial_steering_rad','initial_steering_rad'),('allowed_motion','allowed_motion'),('prediction_horizon_s','prediction_horizon_s'),('terminal_hold_s','terminal_hold_s'),('terminal_speed_max_mps','terminal_speed_max_mps'),('obstacle_temporal_scope','temporal_scope')]:
                if contract[field]!=cert[cert_field]:problems.append(dict(task_id=record['task_id'],failure='contract_certificate_'+field))
    for name in family:
        subset=[record for record in registry if record['source_family']==name]
        per_family[name]=dict(total=len(subset),feasibility=dict(Counter(record['feasibility_status'] for record in subset)),main=sum(record['main_table_eligible'] for record in subset),conditional=sum(record['conditional_reference_table_eligible'] for record in subset))
    old_checks=json.loads((ROOT/'outputs/project_v2/thesis_completion_v1/annotation/reference_reachability_reserved_full_002/audit_001/witness_checks.json').read_text())
    by_witness={record['witness_sha256']:record for record in old_checks}
    for cert in certificates:
        for artifact in cert['artifacts'].values():
            if digest(artifact['path'])!=artifact['sha256']:problems.append(dict(certificate_id=cert['certificate_id'],failure='artifact_hash'))
        witness=json.loads(Path(cert['artifacts']['witness']['path']).read_text());old=by_witness.get(cert['artifacts']['witness']['sha256'])
        if not old or not old['passed'] or old['failures']:problems.append(dict(certificate_id=cert['certificate_id'],failure='independent_witness_audit'))
        if witness.get('evaluated_method_called') is not False or not cert['independently_audited'] or cert['evaluated_method_called'] is not False:problems.append(dict(certificate_id=cert['certificate_id'],failure='independence_flag'))
        initial=json.loads(Path(cert['artifacts']['initial_pose']['path']).read_text())
        if initial['rear_axle_reference_world_pose']!=cert['initial_pose'] or initial['scene']!=cert['scene'] or initial['frame_idx']!=cert['frame_idx']:problems.append(dict(certificate_id=cert['certificate_id'],failure='initial_pose'))
        for name in ['heading_rad','max_heading_error_rad','polygon_xy']:
            if cert['region'][name]!=witness['region'][name]:problems.append(dict(certificate_id=cert['certificate_id'],failure='region_'+name))
    main_ids=[r['task_id'] for r in registry if r['main_table_eligible']]
    conditional_ids=[r['task_id'] for r in registry if r['conditional_reference_table_eligible']]
    infeasible_ids=[r['task_id'] for r in registry if r['feasibility_status']=='infeasible_proven']
    pending=sum(r['feasibility_status']=='pending' for r in registry)
    if freeze['proposed_three_task_main_ids']!=main_ids or freeze['conditional_historical_reference_ids']!=conditional_ids or freeze['independent_infeasible_ids']!=infeasible_ids or freeze['pending_count']!=pending:problems.append(dict(failure='denominator_ID_or_count_mismatch'))
    if freeze['registry_sha256']!=digest(DATA/'task_registry.jsonl') or freeze['policy_sha256']!=digest(DATA/'protocol_v2.json'):problems.append(dict(failure='denominator_hash'))
    banned={'acceptable_instances','accepted_instance_tokens','certified_instance_tokens','safe_regions','certificate_ids','contract_sha256','evaluation_identity','reference_trajectory','ground_truth','evaluation_only','feasibility_status','main_table_eligible'}
    def inspect(value,path):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in banned:problems.append(dict(path=path+'.'+key,failure='evaluation_key_in_method_input'))
                inspect(item,path+'.'+key)
        elif isinstance(value,list):
            for index,item in enumerate(value):inspect(item,f'{path}[{index}]')
        elif isinstance(value,str) and 'evaluation_only' in value:problems.append(dict(path=path,failure='evaluation_path_in_method_input'))
    for record in inputs+languages:inspect(record,record['task_id'])
    if {r['task_id'] for r in inputs+languages}!=set(by_id):problems.append(dict(failure='candidate_task_ID_accounting'))
    hist=rows(DATA/'evaluation_only/historical_retrieval_labels.jsonl')
    originals=rows(ROOT/'outputs/project_v2/thesis_completion_v1/main_experiments/causal_retrieval_full_001/queries.jsonl')
    original_by={f"retrieval:{r['scene']}:{r['frame_idx']}:{r['query_index']}":r for r in originals if r['method']=='existing_object_joint_rule'}
    for label in hist:
        original=original_by[label['task_id']]
        if label['source_positive']!=original['positive'] or label['acceptable_instances']!=original['acceptable_instances'] or by_id[label['task_id']]['instruction']!=original['query_text']:problems.append(dict(task_id=label['task_id'],failure='historical_retrieval_modified'))
    ref_labels=rows(DATA/'evaluation_only/conditional_reference_labels.jsonl')
    audited=rows(ROOT/'outputs/project_v2/thesis_completion_v1/annotation/reference_reachability_reserved_full_002/audit_001/queries.jsonl')
    old_ref={row['query_id']:row for row in audited}
    for label in ref_labels:
        original=old_ref[label['task_id'].removeprefix('reference:')];record=by_id[label['task_id']]
        if label['reachability_audit']!=original:problems.append(dict(task_id=label['task_id'],failure='reference_source_modified'))
        expected=original['reachability']=='conditionally_reachable_static_from_rest'
        if expected!=(record['feasibility_status']=='feasible'):problems.append(dict(task_id=label['task_id'],failure='reference_status_modified'))
        tokens={by_cert[c]['instance_token'] for c in record['certificate_ids']}
        if tokens!=set(original['certified_instance_tokens']) or not tokens.issubset(original['accepted_instance_tokens']):problems.append(dict(task_id=label['task_id'],failure='accepted_target_certificate_set'))
    example=next(r for r in registry if r['source_family']=='original_episode')
    probe_main=copy.deepcopy(example);probe_main['main_table_eligible']=True
    probe_infeasible=copy.deepcopy(example);probe_infeasible['feasibility_status']='infeasible_proven';probe_infeasible['certificate_ids']=[]
    schema_probes=dict(main_true_pending_rejected=not validator.is_valid(probe_main),unproven_infeasible_rejected=not validator.is_valid(probe_infeasible))
    appendices={}
    if args.schema=='task_record.schema_v2.json':
        amendment_manifest=json.loads((DATA/'schema_guard_manifest_v2.json').read_text())
        for name,expected in amendment_manifest['files'].items():
            if digest(DATA/name)!=expected:problems.append(dict(failure='amendment_file_hash',path=name))
        for path,expected in amendment_manifest['source_sha256'].items():
            if digest(path)!=expected:problems.append(dict(failure='amendment_source_hash',path=path))
        overlay=rows(DATA/'source_binding_overlay_v1.jsonl');hash_cache={};overlay_ids=[]
        reference_inputs={r['task_id']:r for r in inputs if by_id[r['task_id']]['source_family']=='historical_conditional_reference'}
        for binding in overlay:
            task=by_id[binding['task_id']];overlay_ids.append(binding['task_id'])
            for name in ['initial_manifest','predicted_graph_stream','compatibility_evidence']:
                artifact=binding[name]
                if artifact['path'] not in hash_cache:hash_cache[artifact['path']]=digest(artifact['path'])
                if hash_cache[artifact['path']]!=artifact['sha256']:problems.append(dict(task_id=binding['task_id'],failure='binding_artifact_hash_'+name))
            initial=json.loads(Path(binding['initial_manifest']['path']).read_text())
            for key in ['sample_token','camera_token','rear_axle_reference_world_pose','rear_axle_yaw_rad']:
                if binding[key]!=initial[key]:problems.append(dict(task_id=binding['task_id'],failure='binding_'+key))
            if binding['effective_issue_time_s']!=initial['camera_timestamp_s'] or binding['scene']!=task['scene'] or binding['issue_frame_idx']!=task['issue_frame_idx']:problems.append(dict(task_id=binding['task_id'],failure='binding_issue_identity'))
            if Path(reference_inputs[binding['task_id']]['historical_prediction_path']).resolve()!=Path(binding['predicted_graph_stream']['path']).resolve():problems.append(dict(task_id=binding['task_id'],failure='binding_prediction_path'))
            for cert_id in task['certificate_ids']:
                if by_cert[cert_id]['initial_pose']!=initial['rear_axle_reference_world_pose']:problems.append(dict(task_id=binding['task_id'],failure='certificate_binding_pose'))
        if len(overlay_ids)!=878 or len(set(overlay_ids))!=878 or set(overlay_ids)!=set(reference_inputs):problems.append(dict(failure='binding_population'))
        appendices=dict(amended_files_checked=len(amendment_manifest['files']),amended_sources_checked=len(amendment_manifest['source_sha256']),
          source_bindings_checked=len(overlay),unique_binding_artifacts_hashed=len(hash_cache),v1_registry_and_denominator_unchanged=True)
    protected=verify_protection()
    report=dict(created_utc=now(),data_instances_passed=not problems and all(r['passed'] for r in hashes) and protected['passed'],
      schema_future_guardrails_passed=all(schema_probes.values()),schema_path=str(DATA/args.schema),schema_sha256=digest(DATA/args.schema),
      registry_count=len(registry),certificate_count=len(certificates),counts=per_family,per_log={k:dict(v) for k,v in per_log.items()},
      main_denominator=len(main_ids),conditional_reference_denominator=len(conditional_ids),infeasible_proven=len(infeasible_ids),pending=pending,
      schema_guardrail_probes=schema_probes,hashes_checked=len(hashes),hash_failures=[r for r in hashes if not r['passed']],
      appendices=appendices,
      historical_retrieval_positive=sum(r['source_positive'] for r in hist),historical_retrieval_negative=sum(not r['source_positive'] for r in hist),
      method_input_records=len(inputs),unbound_language_records=len(languages),all_candidate_IDs_accounted=True,
      protected_files_unchanged=protected['checked'] if protected['passed'] else False,
      failures=problems,claims='Protocol/data preparation only; no method success or navigation score. All 81 are historical object approach, not the three new semantic tasks.',
      independent_geometry_this_audit=False,model_calls=0,GPU_calls=0,planner_calls=0,audit_source_sha256=digest(Path(__file__)))
    dump(REPORT/args.report_name,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['per_log','failures']},ensure_ascii=False))


if __name__=='__main__':main()
