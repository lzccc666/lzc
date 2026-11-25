"""Append minimal pre-evaluation schema guards and immutable source bindings."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

from jsonschema import Draft202012Validator

from conceptgraph.completion.navbench_stage1_20261124 import (
    DATA, REPORT, ROOT, canonical, content_sha, dump, jsonl, now, read, rows,
    sha, source_ref,
)


PROOF_TYPES = ['analytical_constraint_contradiction', 'exhaustive_complete_declared_domain']


def schema_v2():
    result = read(DATA / 'task_record.schema.json')
    result['$id'] = 'navbench-stage1-20261124-task-v2'
    artifact = {'type': 'object', 'additionalProperties': False,
        'required': ['path', 'sha256'], 'properties': {
            'path': {'type': 'string', 'minLength': 1},
            'sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}}}
    result['properties']['infeasibility_proof'] = {
        'type': 'object', 'additionalProperties': False,
        'required': ['proof_type', 'independent', 'complete_declared_domain',
                     'finite_search_failure_is_not_proof', 'task_id', 'contract_sha256',
                     'proof_artifact', 'independent_verification_artifact'],
        'properties': {
            'proof_type': {'enum': PROOF_TYPES}, 'independent': {'const': True},
            'complete_declared_domain': {'const': True},
            'finite_search_failure_is_not_proof': {'const': True},
            'task_id': {'type': 'string'},
            'contract_sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
            'proof_artifact': artifact, 'independent_verification_artifact': artifact}}
    result['allOf'].extend([
        {'if': {'properties': {'main_table_eligible': {'const': True}}},
         'then': {'properties': {
             'feasibility_status': {'const': 'feasible'},
             'source_family': {'const': 'original_episode'},
             'record_kind': {'const': 'scene_task'},
             'action': {'enum': ['pull_over', 'park_nearest', 'follow_vehicle']},
             'scene': {'type': 'string', 'minLength': 1},
             'log_token': {'type': 'string', 'minLength': 1},
             'issue_frame_idx': {'type': 'integer', 'minimum': 0},
             'issue_time_s': {'type': 'number'},
             'certificate_ids': {'minItems': 1}}}},
        {'if': {'properties': {'feasibility_status': {'const': 'infeasible_proven'}}},
         'then': {'required': ['infeasibility_proof'], 'properties': {
             'main_table_eligible': {'const': False},
             'conditional_reference_table_eligible': {'const': False}}}},
        {'if': {'properties': {'feasibility_status': {'const': 'pending'}}},
         'then': {'properties': {'main_table_eligible': {'const': False},
             'conditional_reference_table_eligible': {'const': False},
             'certificate_ids': {'maxItems': 0}}}},
    ])
    return result


def cross_file_guard(task, certificates):
    """Check references/binding; this does not invent a geometric proof verifier."""
    assert task['contract_sha256'] == content_sha(task['contract'])
    if task['main_table_eligible']:
        assert task['feasibility_status'] == 'feasible'
        for cert_id in task['certificate_ids']:
            cert = certificates[cert_id]
            assert cert['task_action'] == task['action']
            assert cert['scene'] == task['scene']
            assert cert['frame_idx'] == task['issue_frame_idx']
            assert cert['task_contracts'][task['task_id']] == task['contract_sha256']
            assert cert['independent_verification']['passed'] is True
            for artifact in cert['artifacts'].values():
                assert sha(artifact['path']) == artifact['sha256']
    if task['feasibility_status'] == 'infeasible_proven':
        proof = task['infeasibility_proof']
        assert proof['task_id'] == task['task_id']
        assert proof['contract_sha256'] == task['contract_sha256']
        for key in ['proof_artifact', 'independent_verification_artifact']:
            artifact = proof[key]
            assert sha(artifact['path']) == artifact['sha256']
            body = read(artifact['path'])
            assert body['task_id'] == task['task_id']
            assert body['contract_sha256'] == task['contract_sha256']
            assert body['proof_type'] in PROOF_TYPES
            assert body['result'] == 'infeasible_proven'
            assert body['independent'] is True
            assert body['complete_declared_domain'] is True
            assert body['finite_search_failure_is_not_proof'] is True
        # Actual proof validity must be audited by a separate geometry/domain checker.
        verification = read(proof['independent_verification_artifact']['path'])
        assert verification['proof_sha256'] == proof['proof_artifact']['sha256']
        assert verification['verification_passed'] is True
        assert sha(verification['verifier_source']['path']) == verification['verifier_source']['sha256']


def source_bindings(registry):
    tasks = {r['task_id']: r for r in registry}
    result, hashes = [], {}
    for entry in rows(DATA / 'method_inputs/candidate_tasks.jsonl'):
        task = tasks[entry['task_id']]
        if task['source_family'] != 'historical_conditional_reference':
            continue
        path = Path(entry['initial_input_manifest'])
        initial = read(path)
        assert initial['scene'] == task['scene']
        assert initial['frame_idx'] == task['issue_frame_idx']
        predicted = Path(entry['historical_prediction_path'])
        for source in [path, predicted]:
            if str(source) not in hashes:
                hashes[str(source)] = sha(source)
        result.append(dict(task_id=task['task_id'], scene=task['scene'],
            log_token=task['log_token'], issue_frame_idx=task['issue_frame_idx'],
            sample_token=initial['sample_token'], camera_token=initial['camera_token'],
            effective_issue_time_s=initial['camera_timestamp_s'],
            original_registry_issue_time_s=task['issue_time_s'],
            rear_axle_reference_world_pose=initial['rear_axle_reference_world_pose'],
            rear_axle_yaw_rad=initial['rear_axle_yaw_rad'],
            rear_axle_reference_source=initial['rear_axle_reference_source'],
            initial_manifest=dict(path=str(path), sha256=hashes[str(path)]),
            predicted_graph_stream=dict(path=str(predicted), sha256=hashes[str(predicted)]),
            compatibility_evidence=source_ref(REPORT / 'independent/existing30_predicted_graph_compatibility_v1.json'),
            does_not_change_contract_feasibility_or_denominator=True,
            is_method_graph_export=False))
    assert len(result) == 878 and all(r['effective_issue_time_s'] is not None for r in result)
    return result


def main():
    started = time.monotonic()
    registry = rows(DATA / 'task_registry.jsonl')
    frozen = {name: sha(DATA / name) for name in [
        'task_registry.jsonl', 'denominator_freeze.json', 'manifest.json',
        'task_record.schema.json', 'preregistration.json']}
    schema = schema_v2()
    validator = Draft202012Validator(schema)
    Draft202012Validator.check_schema(schema)
    certificates = {r['certificate_id']: r for r in rows(DATA / 'evaluation_only/certificates.jsonl')}
    for task in registry:
        validator.validate(task)
        cross_file_guard(task, certificates)
    pending = next(t for t in registry if t['source_family'] == 'original_episode')
    bad_cases = {
        'pending_marked_main': dict(pending, main_table_eligible=True),
        'unproven_infeasible': dict(pending, feasibility_status='infeasible_proven'),
        'pending_with_witness': dict(pending, certificate_ids=['unbound-certificate']),
        'finite_search_failure_claimed_proof': dict(pending, feasibility_status='infeasible_proven',
            infeasibility_proof={'proof_type': 'finite_search_exhausted'}),
    }
    for example in bad_cases.values():
        assert not validator.is_valid(example)
    # Hash and task binding reject even structurally well-formed but invented evidence.
    fake = deepcopy(pending)
    fake.update(feasibility_status='feasible', main_table_eligible=True,
        certificate_ids=[next(iter(certificates))])
    try:
        cross_file_guard(fake, certificates)
    except (AssertionError, KeyError):
        bad_cases['wrong_action_existing_certificate'] = fake
    else:
        raise AssertionError('Historical object-approach certificate leaked to a new task')
    schema_path = DATA / 'task_record.schema_v2.json'
    dump(schema_path, schema)
    binding_path = DATA / 'source_binding_overlay_v1.jsonl'
    jsonl(binding_path, source_bindings(registry))
    amendment = dict(version='stage1-effective-amendment-v3', created_utc=now(), seed=17,
        previous_protocol=source_ref(DATA / 'protocol_v2.json'),
        schema_v1=source_ref(DATA / 'task_record.schema.json'), schema_v2=source_ref(schema_path),
        immutable_registry_sha256=frozen['task_registry.jsonl'],
        immutable_denominator_sha256=frozen['denominator_freeze.json'],
        reason='Pre-method independent audit found two schema-only future admission loopholes; no real records were wrongly included.',
        main_table_rule='Only feasible, scene/log/time-bound original scene tasks with independent same-contract certificates; pending is never eligible.',
        infeasible_rule='Requires independently checked, hashed proof covering the complete declared domain; finite search failure is pending.',
        proof_validation_scope='Schema, hashes and declared independent verifier result are checked here; this stage produces no new impossibility proof and does not claim to verify arbitrary mathematics.',
        source_binding_overlay=source_ref(binding_path),
        witness_role='Independent reference proves existence only. It is not the unique correct route and must never become a pointwise imitation scoring target.',
        trajectory_acceptance='Accept every path/trajectory satisfying the same frozen task target, output mode, motion, horizon where applicable, road, whole-body and sweep constraints; no need to match reference waypoints.',
        finite_goal_scope='Historical static certified small terminal regions are a finite declared conditional benchmark. They do not cover all physically valid nearby locations. A witness-specific terminal region is not the complete target domain.',
        historical_goal_evaluation='For the finite conditional benchmark accept any trajectory reaching any allowed region in the frozen declared union for the acceptable target set; report finite-region membership separately from broader task semantics. Do not transfer the chosen witness instance or region into method inputs.',
        goal_scope_change_policy='A broader semantic target domain requires independent geometry/label definition and a new version before method evaluation; retain old conditional scores and all failures.',
        readiness='Existing 30-scene predictions independently checked: 1196 frames, 878 query bindings; all 81 conditional feasible queries retained, including 23 with empty current prediction graphs.',
        model_calls=0, planner_calls=0, GPU_calls=0)
    amendment_path = DATA / 'effective_amendment_v3.json'
    dump(amendment_path, amendment)
    for name, digest in frozen.items():
        assert sha(DATA / name) == digest
    checks = dict(passed=True, created_utc=now(), registry_records_validated=len(registry),
        negative_cases_rejected=list(bad_cases), conditional_reference_denominator=81,
        main_three_task_denominator=0, source_binding_records=878,
        frozen_files_unchanged=frozen, elapsed_s=round(time.monotonic() - started, 3),
        source_sha256={str(Path(__file__)): sha(__file__),
            str(ROOT / 'scripts/navbench_stage1_20261124_schema_guard.py'):
                sha(ROOT / 'scripts/navbench_stage1_20261124_schema_guard.py')},
        model_calls=0, planner_calls=0, GPU_calls=0)
    dump(DATA / 'schema_guard_verification_v2.json', checks)
    files = [schema_path, binding_path, amendment_path, DATA / 'schema_guard_verification_v2.json']
    dump(DATA / 'schema_guard_manifest_v2.json', dict(created_utc=now(),
        files={str(p.relative_to(DATA)): sha(p) for p in files},
        source_sha256=checks['source_sha256'], append_only=True))
    print(canonical(checks))


if __name__ == '__main__':
    main()
