"""Prepare an evidence-separated navigation benchmark inventory; no method runs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

from conceptgraph.project_paths import PROJECT_ROOT as ROOT, experiment_directory

DATA = ROOT / 'dataset/processed/navbench_stage1_20261124_v1'
REPORT = ROOT / 'outputs/reports/navbench_stage1_20261124'
HISTORY = ROOT / 'outputs/project_v2/thesis_completion_v1'
REFERENCE = HISTORY / 'annotation/reference_reachability_reserved_full_002'
REGIONS = HISTORY / 'annotation/reference_regions_reserved_full_003'
RETRIEVAL = HISTORY / 'main_experiments/causal_retrieval_full_001/queries.jsonl'
EPISODES = ROOT / 'dataset/processed/multiview_task_iteration_20261123/navigation_v1/episodes.json'
LANGUAGE = ROOT / 'dataset/qwen/command_review_import_20261124_v2/approved_language_intents.jsonl'
RUN_ID = 'navbench_stage1_20261124_v1'
SEED = 17
SOURCE_FILES = [Path(__file__), ROOT / 'scripts/navbench_stage1_20261124_prepare.py',
                ROOT / 'conceptgraph/project_paths.py']


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def content_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        for value in values:
            handle.write(canonical(value) + '\n')


def source_ref(path, line=None):
    result = dict(path=str(Path(path).resolve()), sha256=sha(path))
    if line is not None:
        result['line_1based'] = line
    return result


def schema():
    fields = {
        'task_id': {'type': 'string'}, 'source_family': {'enum': ['original_episode', 'approved_language',
            'historical_retrieval', 'historical_conditional_reference']},
        'record_kind': {'enum': ['scene_task', 'unbound_language', 'retrieval_query', 'conditional_approach_query']},
        'action': {'enum': ['pull_over', 'park_nearest', 'follow_vehicle', 'object_retrieval', 'object_approach']},
        'instruction': {'type': ['string', 'null']}, 'scene': {'type': ['string', 'null']},
        'log_token': {'type': ['string', 'null']}, 'issue_frame_idx': {'type': ['integer', 'null']},
        'issue_time_s': {'type': ['number', 'null']}, 'split_role': {'type': 'string'},
        'split_group_id': {'type': 'string'}, 'source_ref': {'type': 'object'},
        'feasibility_status': {'enum': ['feasible', 'infeasible_proven', 'pending']},
        'reason': {'type': 'string'}, 'main_table_eligible': {'type': 'boolean'},
        'conditional_reference_table_eligible': {'type': 'boolean'},
        'execution_readiness': {'enum': ['inputs_partial', 'language_unbound', 'historical_retrieval_only',
                                        'historical_input_compatibility_pending']},
        'certificate_ids': {'type': 'array', 'items': {'type': 'string'}, 'uniqueItems': True},
        'missing_fields': {'type': 'array', 'items': {'type': 'string'}},
        'contract': {'type': 'object'}, 'contract_sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
        'independent_sample_unit': {'type': 'string'},
    }
    return {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'type': 'object',
        'additionalProperties': False, 'required': list(fields), 'properties': fields,
        'allOf': [{'if': {'properties': {'feasibility_status': {'const': 'feasible'}}},
                   'then': {'properties': {'certificate_ids': {'minItems': 1}}}},
                  {'if': {'properties': {'conditional_reference_table_eligible': {'const': True}}},
                   'then': {'properties': {'source_family': {'const': 'historical_conditional_reference'},
                                           'feasibility_status': {'const': 'feasible'}}}}]}


def effective_protocol():
    initial = read(DATA / 'preregistration.json')
    amended = dict(initial)
    amended.update(version='navbench-stage1-20261124-v2', amended_utc=now(),
        supersedes_sha256=sha(DATA / 'preregistration.json'),
        amendment_reason='本阶段任务定义澄清：开环空间路径与带时间轨迹分层；工程模型可冻结，避免将闭环遥测误作路径前置')
    amended['execution_output_modes'] = {
        'path': {'requires': ['空间终点/姿态定义', '工程车辆模型与参考点变换', '车道/道路/障碍几何',
                               '整车与连续扫掠检查'], 'not_applicable': ['实际执行遥测', '速度正确性', '停稳保持', '动态跟随时距']},
        'timed_trajectory': {'requires': ['path全部几何要求', '明确时间参数/预测时域',
                                          '任务确实要求的速度/停车/动态跟随真值'], 'not_applicable': ['实际开过去才可结束']}}
    amended['engineering_vehicle_policy'] = ('可固定VehicleSpec及后轴参考变换假设，用于同条件规划/评价；'
        '明确这是工程模型，不伪称nuScenes实测车辆。不同原点不可静默当同一点。')
    amended['parking_region_policy'] = ('只依据冻结任务中的独立指定区域及声明限制；若研究不声称法规判断，'
        '不额外强制现实法规证明。drivable不得自动冒充停车区，具体车位/空闲仍不能编造。')
    amended['task_requirements'] = {
        'pull_over': ['指令/时刻/停靠侧', '独立指定停靠区域及其声明限制', '工程车辆/后轴变换',
                      '空间终点/航向与整车扫掠；仅timed_trajectory且任务要求时评价速度/保持'],
        'park_nearest': ['指令/时刻', '具体可接受车位集合/几何/声明占用限制', '最近定义及集合完整性',
                         '工程车辆/后轴变换', '空间终点/航向与整车扫掠；时间指标按实际输出定义'],
        'follow_vehicle': ['指令/时刻/真实最近同车道前车', '工程车辆/后轴变换',
                           '动态跟随必须有前车连续状态/预测时域/车距定义', '独立轨迹与道路/车身/运动学检查']}
    return amended


def base_record(task_id, family, kind, action, instruction, scene, log, frame, source, contract, **extra):
    record = dict(task_id=task_id, source_family=family, record_kind=kind, action=action,
        instruction=instruction, scene=scene, log_token=log, issue_frame_idx=frame, issue_time_s=None,
        split_role='validation_or_seen_regression', split_group_id='log:' + log if log else task_id,
        source_ref=source, feasibility_status='pending', reason='independent_task_feasibility_not_certified',
        main_table_eligible=False, conditional_reference_table_eligible=False,
        execution_readiness='inputs_partial', certificate_ids=[], missing_fields=[], contract=contract,
        contract_sha256=content_sha(contract), independent_sample_unit='scene/log; frames and paraphrases are dependent')
    record.update(extra)
    return record


def validate_package(records, certificates, inputs):
    from jsonschema import Draft202012Validator
    validator = Draft202012Validator(schema())
    assert len(records) == 2015 and len({r['task_id'] for r in records}) == len(records)
    assert Counter(r['source_family'] for r in records) == dict(original_episode=6,
        approved_language=16, historical_retrieval=1115, historical_conditional_reference=878)
    by_cert = {r['certificate_id']: r for r in certificates}
    for record in records:
        validator.validate(record)
        assert record['contract_sha256'] == content_sha(record['contract'])
        assert not record['main_table_eligible']
        if record['feasibility_status'] == 'feasible':
            assert record['source_family'] == 'historical_conditional_reference'
            assert record['certificate_ids']
        for cert_id in record['certificate_ids']:
            assert cert_id in by_cert
            assert by_cert[cert_id]['scene'] == record['scene']
            assert by_cert[cert_id]['frame_idx'] == record['issue_frame_idx']
            assert by_cert[cert_id]['task_contracts'][record['task_id']] == record['contract_sha256']
    for cert in certificates:
        for ref in cert['artifacts'].values():
            assert sha(ref['path']) == ref['sha256']
        assert cert['independently_audited'] and not cert['evaluated_method_called']
    banned = {'acceptable_instances', 'accepted_instance_tokens', 'certified_instance_tokens', 'safe_regions',
              'certificate_ids', 'contract_sha256', 'evaluation_identity', 'reference_trajectory', 'ground_truth'}
    for row in inputs:
        assert not banned.intersection(row)
        assert 'evaluation_only' not in canonical(row)
    return dict(schema_records=len(records), unique_ids=True, certificate_artifact_hashes_checked=len(certificates),
        same_task_contract_binding=True, method_inputs_no_answer_fields=True,
        main_three_task_denominator=0, conditional_reference_denominator=81,
        historical_1115_733_382_32_0_preserved=True, finite_search_failure_is_pending=True)


def prepare():
    started = time.monotonic()
    if shutil.disk_usage(ROOT).free < 20 * 1024 ** 3:
        raise RuntimeError('free disk below 20GiB')
    if (DATA / 'manifest.json').exists():
        raise FileExistsError('dataset manifest exists; use a new version, never overwrite')
    run = experiment_directory('nuscenes', 'validation', RUN_ID, create=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    prot = effective_protocol()
    core_inputs = [DATA / 'preregistration.json', DATA / '纳入协议_初版.md', EPISODES, LANGUAGE, RETRIEVAL,
        REFERENCE / 'audit_001/queries.jsonl', REFERENCE / 'audit_001/report.json',
        REFERENCE / 'audit_001/witness_checks.json', REFERENCE / 'targets.jsonl',
        REFERENCE / 'release.json', REFERENCE / 'summary.json', REFERENCE / 'acceptance_001.json',
        REGIONS / 'queries.jsonl', REGIONS / 'targets.jsonl', REGIONS / 'release.json',
        REGIONS / 'acceptance_001.json', HISTORY / 'splits/reserved_test_scenes_v2.json',
        HISTORY / 'mapping_evaluation/causal_mapping_full_001/release.json',
        ROOT / 'outputs/results/nuscenes/validation/iterations/recall_refinement_R1_full_20261120_v2/release.json',
        ROOT / 'outputs/reports/paper_campaign_20261123/navigation/target_selection_complete.json',
        ROOT / 'outputs/reports/paper_campaign_20261123/navigation/same_task_navigation_chain.json']
    protected = [ROOT / name for name in ['AGENTS.md', 'MEMORY.md', 'PAPER_MEMORY.md',
        'conceptgraph/navigation/semantic_tasks_20261123.py', 'conceptgraph/navigation/task_replay_20261123.py',
        'conceptgraph/navigation/vehicle_geometry.py', 'conceptgraph/navigation/pipeline.py',
        'conceptgraph/hydra_configs/base_mapping.yaml', 'paper/当前版本.json',
        'paper/当前版_v4/sources/论文正文.md']] + core_inputs
    protected_hashes = {str(p): sha(p) for p in protected if p.exists()}
    dump(run / 'registration.json', dict(started_utc=now(), seed=SEED,
        budgets=prot['budgets'], protected_sha256=protected_hashes,
        initial_protocol_sha256=sha(DATA / 'preregistration.json'), method_calls=0, model_calls=0, GPU_calls=0))
    dump(DATA / 'scope_amendment_v1.json', dict(created_utc=now(),
        initial_protocol_sha256=sha(DATA / 'preregistration.json'), effective_protocol='protocol_v2.json',
        reason=prot['amendment_reason'], based_on='user-authorized scope clarified by primary coordinator before method execution',
        benchmark_results_seen_for_change=False, no_threshold_or_historical_label_change=True))
    dump(DATA / 'protocol_v2.json', prot)
    dump(DATA / 'task_record.schema.json', schema())
    logs = {s['scene']: s['log_token'] for s in read(core_inputs[17])['scenes']}
    logs.update({s['scene']: s['log_token'] for s in read(HISTORY / 'splits/reserved_test_scenes_v2.json')['scenes']})
    records, labels, inputs, language_rows = [], [], [], []
    for index, ep in enumerate(read(EPISODES)['rows']):
        contract = dict(action=ep['action'], output_mode='timed_trajectory' if ep['action'] == 'follow_vehicle' else 'path',
            engineering_vehicle_model='VehicleSpec default; explicit rear-axle transform required at binding',
            task_semantics='unbound independent target/region; do not infer from replay trajectory',
            speed_or_hold='not_applicable_for_path', prediction_horizon_s=None, allowed_motion='not_yet_frozen')
        records.append(base_record('episode:' + ep['episode_id'], 'original_episode', 'scene_task', ep['action'],
            ep['instruction'], ep['scene'], ep['log_token'], ep['issue_frame_idx'], source_ref(EPISODES), contract,
            issue_time_s=ep['issue_time_s'], missing_fields=['independent_target_or_region',
                'task_specific_reference_path_or_trajectory', 'initial_reference_transform_and_contract']))
        inputs.append(dict(task_id=records[-1]['task_id'], instruction=ep['instruction'], scene=ep['scene'],
            issue_frame_idx=ep['issue_frame_idx'], issue_time_s=ep['issue_time_s'],
            observed_input_manifest=str(ROOT / 'dataset/processed/multiview_task_iteration_20261123/navigation_v1/task_frames.jsonl.gz')))
    for index, item in enumerate(rows(LANGUAGE), 1):
        contract = dict(action=item['intent']['action'], output_mode='unbound',
                        language_review_only=True, task_semantics=item['intent'])
        records.append(base_record('language:' + item['sample_id'], 'approved_language', 'unbound_language',
            item['intent']['action'], item['instruction'], None, None, None, source_ref(LANGUAGE, index), contract,
            split_role='unassigned_language_seen_regression', split_group_id=item['split_group_id'],
            execution_readiness='language_unbound', missing_fields=['scene', 'episode', 'issue_time',
                'independent_target_or_region', 'reference_path_or_trajectory']))
        language_rows.append(dict(task_id=records[-1]['task_id'], instruction=item['instruction'],
            scene=None, issue_frame_idx=None, split_group_id=item['split_group_id']))
    historical_labels = []
    for index, q in enumerate(rows(RETRIEVAL), 1):
        if q['method'] != 'existing_object_joint_rule':
            continue
        task_id = f"retrieval:{q['scene']}:{q['frame_idx']}:{q['query_index']}"
        contract = dict(action='object_retrieval', task_semantics=q['query_type'], output_mode='selected_object',
            route_evaluated=False, temporal_scope='current front annotated view', match_rule='same-class XY<=3m one-to-one')
        records.append(base_record(task_id, 'historical_retrieval', 'retrieval_query', 'object_retrieval',
            q['query_text'], q['scene'], logs[q['scene']], q['frame_idx'], source_ref(RETRIEVAL, index), contract,
            execution_readiness='historical_retrieval_only', missing_fields=['independent_navigation_reference'],
            reason='retrieval truth does not certify navigation feasibility'))
        historical_labels.append(dict(task_id=task_id, source_positive=q['positive'],
            acceptable_instances=q['acceptable_instances'], source_query_type=q['query_type'],
            navigation_feasibility='pending', historical_record_unchanged=True))
        inputs.append(dict(task_id=task_id, instruction=q['query_text'], scene=q['scene'], issue_frame_idx=q['frame_idx'],
            map_versions=['legacy_front1', 'R1_front1'], selection_only=True))
    assert Counter(q['source_positive'] for q in historical_labels) == {True: 733, False: 382}
    ref_release = read(REFERENCE / 'release.json')
    ref_audit = read(REFERENCE / 'audit_001/report.json')
    assert ref_audit['passed'] and ref_audit['accepted_static_from_rest_witnesses'] == 105
    audited = {(a['scene'], a['frame_idx'], a['instance_token']): a for a in rows_json(
        REFERENCE / 'audit_001/witness_checks.json') if a['passed']}
    witnesses, certificates = {}, []
    for t in rows(REFERENCE / 'targets.jsonl'):
        if t['reference'] is None:
            continue
        key = t['scene'], t['frame_idx'], t['instance_token']
        a = audited[key]
        path = REFERENCE / t['reference']['path']
        assert sha(path) == t['reference']['sha256'] == a['witness_sha256']
        w = read(path)
        initial = read(w['known_initial_pose']['path'])
        certificate_id = f"conditional-reference:{key[0]}:{key[1]}:{key[2]}"
        cert = dict(certificate_id=certificate_id, scene=key[0], frame_idx=key[1], instance_token=key[2],
            independently_audited=True, evaluated_method_called=False,
            kind='historical_static_from_rest_object_approach', task_contracts={},
            artifacts=dict(witness=source_ref(path), initial_pose=w['known_initial_pose'],
                static_context=source_ref(REGIONS / w['context']['path']),
                independent_audit=source_ref(REFERENCE / 'audit_001/report.json')),
            region=dict(polygon_xy=w['region']['polygon_xy'], heading_rad=w['region']['heading_rad'],
                        max_heading_error_rad=w['region']['max_heading_error_rad']),
            vehicle=ref_release['vehicle'], initial_pose=initial['rear_axle_reference_world_pose'],
            initial_speed_mps=0.0, initial_steering_rad=0.0,
            allowed_motion='forward_only', prediction_horizon_s=90.0, terminal_hold_s=2.0,
            terminal_speed_max_mps=0.02, temporal_scope=ref_release['policy']['static_obstacle_scope'],
            scope='conditional historical replay; not real parking permission, parking-space selection or following')
        assert sha(cert['artifacts']['static_context']['path']) == w['context']['sha256']
        witnesses[key] = cert
        certificates.append(cert)
    original_queries = {}
    annotation_source = HISTORY / 'annotation/reserved_official_annotation_001'
    for name in ['main_queries.jsonl', 'challenge_queries.jsonl']:
        path = annotation_source / name
        core_inputs.append(path)
        original_queries.update({q['query_id']: q for q in rows(path)})
    for index, q in enumerate(rows(REFERENCE / 'audit_001/queries.jsonl'), 1):
        original = original_queries[q['query_id']]
        task_id = 'reference:' + q['query_id']
        contract = dict(action='object_approach', instruction_semantics=original['query_text'], output_mode='timed_trajectory',
            scene=q['scene'], issue_frame_idx=q['frame_idx'], initial_pose_reference='known_rear_axle_world_pose',
            initial_speed_mps=0.0, initial_steering_rad=0.0, vehicle_model=ref_release['vehicle'], allowed_motion='forward_only',
            goal_geometry='certified_region_for_any_original_acceptable_instance', goal_reference_point='rear_axle',
            terminal_heading_tolerance_rad=0.08726646259971647, terminal_speed_max_mps=0.02,
            prediction_horizon_s=90.0, terminal_hold_s=2.0,
            obstacle_temporal_scope=ref_release['policy']['static_obstacle_scope'],
            map_geometry_scope='raw official drivable and directed lane polygons',
            nearest_definition='original_query_acceptable_set_unchanged; no nearest parking-space claim',
            acceptable_target_set_sha256=content_sha(q['accepted_instance_tokens']))
        certs = []
        for token in q['certified_instance_tokens']:
            cert = witnesses[(q['scene'], q['frame_idx'], token)]
            cert['task_contracts'][task_id] = content_sha(contract)
            certs.append(cert['certificate_id'])
        feasible = q['reachability'] == 'conditionally_reachable_static_from_rest'
        assert bool(certs) == feasible
        records.append(base_record(task_id, 'historical_conditional_reference', 'conditional_approach_query',
            'object_approach', original['query_text'], q['scene'], logs[q['scene']], q['frame_idx'],
            source_ref(REFERENCE / 'audit_001/queries.jsonl', index), contract,
            split_role='historical_used_test_regression_not_unseen', feasibility_status='feasible' if feasible else 'pending',
            conditional_reference_table_eligible=feasible, certificate_ids=certs,
            execution_readiness='historical_input_compatibility_pending',
            reason='independent historical conditional witness exists' if feasible else 'finite reference did not certify; not proven infeasible',
            missing_fields=['causal_prediction_compatibility_audit', 'method_execution'] if feasible else ['independent_conditional_witness', 'causal_prediction_compatibility_audit']))
        labels.append(dict(task_id=task_id, source_query=original, reachability_audit=q,
                           certificate_ids=certs, evaluation_only=True))
        inputs.append(dict(task_id=task_id, instruction=original['query_text'], scene=q['scene'],
            issue_frame_idx=q['frame_idx'], initial_input_manifest=str(
                HISTORY / 'data_manifest/reserved_causal_inputs_full_001/converted' / q['scene'] / 'inputs' / f"{q['frame_idx']:06d}.json"),
            predictions_available=True, prediction_compatibility_status='pending_independent_audit',
            historical_prediction_path=str(ROOT / 'outputs/results/nuscenes/test/mapping/one_to_one_depth_quality' / q['scene'] / 'frames.jsonl.gz')))
    checks = validate_package(records, certificates, inputs + language_rows)
    assert sum(r['feasibility_status'] == 'feasible' for r in records) == 81
    jsonl(DATA / 'task_registry.jsonl', records)
    jsonl(DATA / 'evaluation_only/certificates.jsonl', certificates)
    jsonl(DATA / 'evaluation_only/historical_retrieval_labels.jsonl', historical_labels)
    jsonl(DATA / 'evaluation_only/conditional_reference_labels.jsonl', labels)
    jsonl(DATA / 'method_inputs/candidate_tasks.jsonl', inputs)
    jsonl(DATA / 'method_inputs/unbound_language.jsonl', language_rows)
    dump(DATA / 'source_index.json', {str(p): dict(sha256=sha(p), bytes=p.stat().st_size,
         role='historical frozen source, unchanged') for p in core_inputs})
    versions = [dict(version_id='legacy_front1', cameras=['CAM_FRONT'],
        path=str(HISTORY / 'mapping_evaluation/causal_mapping_full_001'), condition='one_to_one_depth_quality',
        note='historical retrieval map; not B0 and not three-camera'),
        dict(version_id='R1_front1', cameras=['CAM_FRONT'],
        path=str(ROOT / 'outputs/results/nuscenes/validation/iterations/recall_refinement_R1_full_20261120_v2'),
        note='updated frontend, sparse confirmations and current reliable dense position protection'),
        dict(version_id='T3v2_three_front', cameras=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'],
        path=str(ROOT / 'outputs/results/nuscenes/validation/iterations/mapnext_20261124_baseline_v1'),
        note='seen six-scene mapping regression; not automatically bound to 1115 or 878 tasks'),
        dict(version_id='S2_three_front', cameras=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'],
        path=str(ROOT / 'outputs/results/nuscenes/validation/iterations/mapnext_20261124_S2_full_v1'),
        note='only one object-frame recovery; no automatic default or navigation gain claim')]
    dump(DATA / 'map_versions.json', versions)
    denominator = dict(created_utc=now(), stage='pre_method_inventory_freeze', seed=SEED,
        policy_sha256=sha(DATA / 'protocol_v2.json'), registry_sha256=sha(DATA / 'task_registry.jsonl'),
        proposed_three_task_main_ids=[], proposed_three_task_main_denominator=0,
        conditional_historical_reference_ids=[r['task_id'] for r in records if r['conditional_reference_table_eligible']],
        conditional_historical_reference_denominator=81, independent_infeasible_ids=[],
        pending_count=sum(r['feasibility_status'] == 'pending' for r in records),
        no_method_outcomes_used=True, no_case_drop_after_freeze=True, new_test=False,
        state='source/protocol inventory frozen; new three-task main benchmark remains unbound, do not report 0% success',
        historical_retrieval_denominator=dict(positive=733, negative=382, tasks=1115),
        historical_chain_unchanged=dict(correct_point_road_routes=32, correct_static_body_routes=0, positive=733),
        no_method_selection_or_planning_executed=True)
    dump(DATA / 'denominator_freeze.json', denominator)
    for path in SOURCE_FILES:
        target = run / 'source_snapshot' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    files = {str(p.relative_to(DATA)): dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in DATA.rglob('*') if p.is_file()}
    manifest = dict(created_utc=now(), seed=SEED, package_version=RUN_ID, files=files,
        source_sha256={str(p): sha(p) for p in SOURCE_FILES}, input_sha256={str(p): sha(p) for p in core_inputs},
        source_counts=dict(Counter(r['source_family'] for r in records)), registry_records=len(records),
        feasibility_counts=dict(Counter(r['feasibility_status'] for r in records)),
        main_three_task_denominator=0, conditional_historical_reference_denominator=81,
        conditional_reference_witnesses=len(certificates), inference_evaluation_separated=True,
        no_new_test=True, model_calls=0, GPU_calls=0, planner_calls=0, actual_execution_calls=0)
    dump(DATA / 'manifest.json', manifest)
    checks.update(protected_unchanged=all(sha(p) == h for p, h in protected_hashes.items()),
        protected_count=len(protected_hashes), elapsed_s=time.monotonic() - started,
        data_bytes=sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file()),
        free_bytes=shutil.disk_usage(ROOT).free, model_calls=0, GPU_calls=0, planner_calls=0)
    assert checks['protected_unchanged']
    assert checks['elapsed_s'] <= 2700 and checks['data_bytes'] <= 128 * 1024 ** 2
    dump(run / 'preparation_verification.json', checks)
    dump(run / 'manifest.json', manifest)
    dump(REPORT / 'preparation_verification.json', checks)
    report_text = ('# 导航阶段1：任务来源与独立有解证据\n\n'
        '本阶段实际生成协议、schema和数据包，未运行感知、Qwen、选择器或规划器。\n\n'
        '|来源|数量|本阶段状态|\n|---|---:|---|\n'
        '|原三类任务episode|6|目标/区域未绑定，pending|\n'
        '|已审核文字|16|仅文字通过；15个语言组，未绑定场景，pending|\n'
        '|历史检索查询|1115|733正/382负原样；不推出导航有解|\n'
        '|历史条件参考查询|878|81条件可行、797未知；105独立参考见证|\n\n'
        '三新任务主分母目前为0，不能报0%成功，更不能把未绑定任务称无解。'
        '81条仅可作历史对象接近的条件回归补充，来自已用测试；不是新的停车/车位/跟车任务。'
        '已找到30场景历史方法预测图，时间/因果/参考点兼容性正在独立核验；参考可行与被测输入准备分开记录。\n\n'
        '空间路径采用明确冻结的工程车身/后轴变换，检查道路、障碍和整车扫掠；'
        '不要求实车遥测，不把仅空间路径声称有速度/停稳效果。带时间轨迹按任务声明检查时间、速度和跟随关系。'
        '指定停靠区只按独立区域与声明限制评价，不新增假想法规前置；具体车位/空闲仍不可编造。\n\n'
        '见证只有与任务含义、起点、车辆、前向/倒车、终点、时域及几何口径完全相同才可继承。'
        '待评算法找不到路不是无解证明，有限参考搜索失败统一pending。所有方法固定分母，拒选、超时和失败保留。'
        '历史1115/733/382及32道路点/0静态车身交集完整保留。\n\n'
        f'数据：`{DATA}`；运行：`{run}`。\n'
        '下一步先用协议绑定独立指定区域/真实车位或连续前车与参考证据，随后冻结新主表；'
        '已有81条可行参考仅在补齐同源因果预测输入后作独立标记的历史条件回归。\n')
    (REPORT / '阶段1协议与数据盘点.md').write_text(report_text)
    print(canonical(dict(data=str(DATA), run=str(run), report=str(REPORT), checks=checks)))


def rows_json(path):
    return read(path)


def verify():
    manifest = read(DATA / 'manifest.json')
    for name, expected in manifest['files'].items():
        assert sha(DATA / name) == expected['sha256'], name
    for name, expected in manifest['input_sha256'].items():
        assert sha(name) == expected, name
    for name, expected in manifest['source_sha256'].items():
        assert sha(name) == expected, name
    checks = validate_package(rows(DATA / 'task_registry.jsonl'), rows(DATA / 'evaluation_only/certificates.jsonl'),
        rows(DATA / 'method_inputs/candidate_tasks.jsonl') + rows(DATA / 'method_inputs/unbound_language.jsonl'))
    print(canonical(dict(passed=True, **checks)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'verify'])
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else verify()


if __name__ == '__main__':
    main()
