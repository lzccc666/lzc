"""Aggregate complete jobs only, with explicit pending/failed denominators."""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from conceptgraph.nuscenes.benchmark_ground_truth import associate_nodes, retrieval_metrics
from conceptgraph.nuscenes.expansion_benchmark import paired_scene_bootstrap
from conceptgraph.scripts.lane_experiment_metrics import distribution
from conceptgraph.scripts.lane_expansion_production import resolve_online_directory
from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT
from conceptgraph.scripts.run_incremental_semantic_navigation import write_json


RETRIEVAL_METRIC_DEFINITIONS = {
    'top1': 'Positive queries only: first retrieved node is associated with any predeclared acceptable official instance. Unassociated or absent targets remain failures.',
    'mrr': 'Positive queries only: reciprocal rank of the first associated acceptable official instance; zero if none is retrieved.',
    'candidate_recall': 'Per positive query: number of distinct acceptable official instances recovered in candidates / number of all acceptable official instances. Then average query fractions. This is set recall, not any-target hit rate; with multiple acceptable targets it may be below Top-1.',
    'target_confirmation_correct': 'Positive queries only: the geometrically selected target is associated with an acceptable official instance. Online locked-target confirmation is separately evaluated at the recorded lock frame.',
    'correct_rejection': 'No-match queries only: retrieved candidate list is empty. Online no-match lock rejection is a separate metric.',
    'aggregation': 'Query-level fractions and indicators, scene macro means, and paired scene-cluster bootstrap are reported separately; non-applicable queries are excluded only from that metric denominator.',
}


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def average(values):
    kept = [float(v) for v in values if v is not None]
    return {'value': float(np.mean(kept)) if kept else None, 'denominator': len(kept), 'unknown_or_not_applicable': len(values)-len(kept), 'empty_reason': None if kept else 'no applicable observations'}


def grouped(rows, metric_names, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[k]) for k in keys)].append(row)
    result = []
    for key, members in sorted(groups.items()):
        by_scene = defaultdict(list)
        for row in members:
            by_scene[row['scene']].append(row)
        result.append({**dict(zip(keys, key)), 'query_condition_count': len(members), 'scene_count': len(by_scene), 'overall_query_means': {metric: average([r.get(metric) for r in members]) for metric in metric_names}, 'scene_macro_means': {metric: average([average([r.get(metric) for r in group])['value'] for group in by_scene.values()]) for metric in metric_names}})
    return result


def online_row(root, directory, job, queries, official):
    done = read(directory / 'complete.json')
    if done is None:
        return None
    first = done['first']
    metrics, summary = first['metrics'], first['summary']
    frames_path = directory / 'first/end_to_end_replay' / job['scene'] / 'frames.jsonl'
    frames = [json.loads(line) for line in frames_path.open()]
    query = queries[job['query_id']]
    initial = frames[0]
    association = associate_nodes(initial.get('benchmark_nodes', []), official[(job['scene'], 0)]['objects'])
    retrieval = retrieval_metrics(initial['candidate_ids'], query['accepted_instance_tokens'], association)
    locked = next((r for r in frames if 'LOCKED' in r.get('state_trace', [])), None)
    confirmation_mapping, confirmation = None, None
    start_projections = []
    for frame in frames:
        attempt = frame['experiment_audit'].get('planning_result')
        if not attempt or not attempt.get('selected_candidate'):
            continue
        candidate = attempt['selected_candidate']
        for goal in candidate.get('goal_candidates', []):
            if goal.get('path_found') and goal.get('goal_rc') == candidate.get('target_grid'):
                value = goal.get('debug', {}).get('start_projection_distance_m')
                if value is not None:
                    start_projections.append(value)
    if query['accepted_instance_tokens']:
        confirmation = False
        if locked is not None:
            confirmation_mapping = associate_nodes(locked.get('benchmark_nodes', []), official[(job['scene'], locked['frame_idx'])]['objects'])
            matched = {m['node_id']: m['instance_token'] for m in confirmation_mapping['matches']}
            confirmation = matched.get(locked['selected_node_id']) in query['accepted_instance_tokens']
    else:
        confirmation = locked is None
    method = f"{job['variant']}_{'lane' if job['lane'] else 'drivable'}"
    return {'scene': job['scene'], 'query_id': job['query_id'], 'role': job['role'], 'method': method, 'variant': job['variant'], 'lane': job['lane'], 'artifact': str(directory), 'execution_passed': done['execution_passed'], 'query_success': metrics['successful_plans'] > 0, 'target_confirmation_correct': confirmation if query['accepted_instance_tokens'] else None, 'target_decision_correct': confirmation, 'no_match_correct_rejection': confirmation if not query['accepted_instance_tokens'] else None, 'target_lock_achieved': locked is not None, 'confirmation_association': confirmation_mapping, 'initial_retrieval': retrieval, 'top1': retrieval['top1'], 'mrr': retrieval['mrr'], 'candidate_recall': retrieval['candidate_recall'], 'correct_rejection': retrieval['correct_rejection'], 'planning_attempts': metrics['planning_attempts'], 'successful_plans': metrics['successful_plans'], 'planning_success_rate': metrics['path_success_rate'], 'observed_planner_reachability_rate': metrics['reachable_rate'], 'lane_adherence_rate': metrics['lane_adherence_rate'], 'collision_rate': metrics['collision_rate'], 'direction_violation_sample_count': metrics['direction_violation_sample_count'], 'illegal_transition_count': metrics['illegal_transition_count'], 'reverse_event_count': metrics['reverse_event_count'], 'out_of_lane_point_ratio': metrics['out_of_lane_point_ratio'], 'out_of_lane_length_m': metrics['out_of_lane_length_m'], 'start_projection_distance_m': distribution(start_projections), 'minimum_safety_distance_m': metrics['minimum_safety_distance_m'], 'path_length_m': metrics['path_length_m'], 'total_absolute_turn_rad': metrics['total_absolute_turn_rad'], 'planning_time_s': metrics['planning_time_s'], 'total_frame_time_s': metrics['total_frame_time_s'], 'target_projection_distance_m': metrics['target_projection_distance_m'], 'failure_codes': metrics['failure_codes'], 'frame_failure_codes': dict(Counter(r['failure']['code'] for r in frames if r.get('failure'))), 'blocked_path_invalidated_frames': [r['frame_idx'] for r in frames if r['occupancy_audit'].get('unsafe_path_invalidated')], 'initialization_and_scene_runtime_s': summary['runtime_s'], 'qwen_first': first['qwen_audit'], 'qwen_cache_repeat': done['cache_repeat']['qwen_audit'], 'qwen_applications': summary.get('qwen', {}), 'gpu_peak_memory_allocated_bytes_process': summary.get('gpu_peak_memory_allocated_bytes'), 'determinism_passed': done['determinism_passed'], 'cache_zero_new_backend_calls': done['cache_zero_new_backend_calls'], 'source_failures': summary['source_failures']}


def main(root=None):
    root = Path(root or DEFAULT_ROOT)
    scenes = read(root / 'scene_selection_manifest.json')['scenes']
    run_ids = read(root / 'experiment_run_manifest.json', {}).get('active_runs', {})
    scene_index = {s['scene']: s for s in scenes}
    queries = {r['query_id']: r for line in (root / 'benchmark_queries.jsonl').open() if (r := json.loads(line))}
    retrieval_queries = [q for q in queries.values() if q['role'] == 'sealed_test' or q['scene'] in {'scene-0003', 'scene-0916'}]
    expected_retrieval = len(retrieval_queries) * 6
    expected_static = len(queries) * 4
    official = {(r['scene'], r['frame_idx']): r for line in (root / 'ground_truth_frames.jsonl').open() if (r := json.loads(line))}
    online, missing_online, required_directories = [], [], []
    for query in queries.values():
        if not query['online_replay_selected']:
            continue
        variants = ['off']
        if query['role'] == 'sealed_test' or query['scene'] in {'scene-0003', 'scene-0916'}:
            variants += ['adapter', 'base']
        for variant in variants:
            for lane in (False, True):
                directory = resolve_online_directory(root, query, variant, lane)
                required_directories.append(str(directory))
                job = {'scene': query['scene'], 'query_id': query['query_id'], 'role': query['role'], 'variant': variant, 'lane': lane}
                row = online_row(root, directory, job, queries, official)
                if row is None:
                    missing_online.append({**job, 'expected': str(directory)})
                else:
                    row['log_token'] = scene_index[row['scene']]['log_token']
                    online.append(row)
    static, missing_static = [], []
    for scene in scenes:
        run_id = run_ids.get('static_test', 'static_test_v1') if scene['role'] == 'sealed_test' else run_ids.get('static_development', 'static_development_all_v3')
        path = root / 'experiments' / run_id / scene['scene'] / 'complete.json'
        complete = read(path)
        if complete is None:
            missing_static.append(scene['scene'])
            continue
        for row in complete['results']:
            result = read(Path(row['artifact']) / 'target_selection_result.json')
            association = read(path.parent / 'association.json')
            mapping = {m['node_id']: m['instance_token'] for m in association['matches']}
            accepted = queries[row['query_id']]['accepted_instance_tokens']
            correct = mapping.get(row['selected_node_id']) in accepted if accepted else row['selected_node_id'] is None
            static.append({**row, 'failure_code': result['failure_code'], 'query_success': row['success'], 'top1': row['retrieval']['top1'], 'mrr': row['retrieval']['mrr'], 'candidate_recall': row['retrieval']['candidate_recall'], 'correct_rejection': row['retrieval']['correct_rejection'], 'target_confirmation_correct': correct if accepted else None, 'target_decision_correct': correct})
    retrieval, missing_retrieval = [], []
    for query in queries.values():
        if query['role'] != 'sealed_test' and query['scene'] not in {'scene-0003', 'scene-0916'}:
            continue
        for variant in ('off', 'base', 'adapter'):
            for lane in (False, True):
                method = f"{variant}_{'lane' if lane else 'drivable'}"
                run_id = run_ids.get(f'retrieval_{variant}', f'retrieval_{variant}_v2' if variant == 'off' else f'retrieval_{variant}_v1')
                path = root / 'experiments' / run_id / query['query_id'] / method / 'complete.json'
                done = read(path)
                if done is None:
                    missing_retrieval.append({'query_id': query['query_id'], 'method': method, 'expected': str(path)})
                    continue
                first = done['phases']['first']
                row = {'scene': query['scene'], 'query_id': query['query_id'], 'role': query['role'], 'method': method, 'variant': variant, 'lane': lane, 'artifact': str(path.parent), 'top1': first['retrieval']['top1'], 'mrr': first['retrieval']['mrr'], 'candidate_recall': first['retrieval']['candidate_recall'], 'correct_rejection': first['retrieval']['correct_rejection'], 'target_confirmation_correct': first['target_confirmation_correct'] if query['accepted_instance_tokens'] else None, 'target_decision_correct': first['target_confirmation_correct'], 'query_parse_field_correct': first['query_parse_field_correct'], 'planning_success': first['planning_success'], 'failure_code': first['failure_code'], 'determinism_passed': done['determinism_passed'], 'cache_repeat_zero_new_backend_calls': done['cache_repeat_zero_new_backend_calls'], 'qwen_first': first['qwen_audit'], 'qwen_cache_repeat': done['phases']['cache_repeat']['qwen_audit'], 'static_rerank_selection_applied': False}
                row['log_token'] = scene_index[row['scene']]['log_token']
                retrieval.append(row)
    metric_names = ['top1', 'mrr', 'candidate_recall', 'correct_rejection', 'target_confirmation_correct']
    navigation = {'schema_version': 'expansion-navigation-v1', 'rows': online, 'missing_jobs': missing_online, 'expected_job_pairs': len(required_directories), 'completed_job_pairs': len(online), 'all_complete': not missing_online and all(r['execution_passed'] for r in online), 'summaries': grouped(online, ['query_success', 'planning_success_rate', 'target_confirmation_correct', 'target_decision_correct', 'no_match_correct_rejection', 'target_lock_achieved', *metric_names[:4]], ['role', 'method']), 'units': {'query_success': 'any accepted path in one frozen scene/query replay', 'planning_success_rate': 'successful attempts / all attempts, distinct from query count', 'target_confirmation_correct': 'positive queries only: first LOCKED target mapped to official accepted instance; unmapped positives are failures; negative rejection is a separate metric', 'top1': 'first-frame online candidate retrieval; all-query static retrieval is separate'}, 'unknown_association_handling': 'Unmatched accepted targets remain retrieval/confirmation failures, never removed.'}
    write_json(root / 'navigation_report.json', navigation)
    write_json(root / 'retrieval_report.json', {'rows': retrieval, 'missing_jobs': missing_retrieval, 'expected_query_condition_pairs': expected_retrieval, 'completed_query_condition_pairs': len(retrieval), 'all_complete': not missing_retrieval and all(r['determinism_passed'] and r['cache_repeat_zero_new_backend_calls'] for r in retrieval), 'summaries': grouped(retrieval, metric_names, ['role', 'method']), 'metric_definitions': RETRIEVAL_METRIC_DEFINITIONS, 'negative_query_metric_rule': 'Top1/MRR/candidate recall undefined for no-match queries; report correct rejection separately.', 'static_rerank_semantics': 'Qwen recommendation audited within verified candidates; static selector remains authoritative. Actual online rerank application is reported separately.'})
    write_json(root / 'ablation_report.json', {'rows': static, 'missing_scenes': missing_static, 'expected_cases': expected_static, 'completed_cases': len(static), 'all_complete': not missing_static, 'summaries': grouped(static, ['query_success', *metric_names], ['role', 'method']), 'same_frozen_frame0_inputs': True})
    write_json(root / 'scene_reports.json', {'scenes': [{'scene': s['scene'], 'role': s['role'], 'strata': s['strata'], 'online_summaries': grouped([r for r in online if r['scene'] == s['scene']], ['query_success', *metric_names], ['method']), 'static_case_count': sum(r['scene'] == s['scene'] for r in static), 'official_frame0_objects': len(official[(s['scene'], 0)]['objects'])} for s in scenes]})
    write_json(root / 'lane_constraint_report.json', {'online_summaries': grouped(online, ['lane_adherence_rate', 'collision_rate', 'direction_violation_sample_count', 'illegal_transition_count', 'reverse_event_count', 'out_of_lane_point_ratio', 'out_of_lane_length_m'], ['role', 'method']), 'path_audit_reports': [str(p) for p in sorted((root / 'path_audits').glob('*.json'))], 'zero_denominator': 'null; zero accepted paths never means 100 percent safety', 'normal_mode_lane_violations': 'diagnostic, not strict-mode enforcement failures', 'reachability_unit': 'observed successful planning attempts / all actual attempts; not independent reachable-object truth'})
    qwen_rows = [{k: r[k] for k in ('scene', 'query_id', 'role', 'method', 'qwen_first', 'qwen_cache_repeat', 'qwen_applications', 'gpu_peak_memory_allocated_bytes_process', 'cache_zero_new_backend_calls', 'determinism_passed')} for r in online if r['variant'] != 'off']
    totals = {}
    for role in ('legacy_regression', 'sealed_test'):
        for method in ('base_drivable', 'base_lane', 'adapter_drivable', 'adapter_lane'):
            members = [r for r in qwen_rows if r['role'] == role and r['method'] == method]
            applications = Counter()
            for r in members:
                applications.update(r['qwen_applications'])
            totals[f'{role}/{method}'] = {'completed_query_conditions': len(members), 'backend_call_count': sum(r['qwen_first']['backend_call_count'] for r in members), 'successful_backend_response_count': sum(r['qwen_first']['successful_backend_response_count'] for r in members), 'cache_repeat_backend_call_count': sum(r['qwen_cache_repeat']['backend_call_count'] for r in members), 'cache_repeat_hits': sum(r['qwen_cache_repeat']['cache_hit_count'] for r in members), 'fallback_count': sum(r['qwen_first']['fallback_count'] for r in members), 'application_counts': dict(applications), 'peak_process_gpu_allocated_bytes': max([r['gpu_peak_memory_allocated_bytes_process'] or 0 for r in members], default=None)}
    write_json(root / 'qwen_online_report.json', {'rows': qwen_rows, 'totals': totals, 'runtime_scope': 'actual CUDA, frozen weights, recorded frame barrier; shared GPU workload', 'call_response_application_are_distinct': True, 'synthetic_fault_calls_excluded': True})
    determinism = [{'scene': r['scene'], 'query_id': r['query_id'], 'method': r['method'], 'passed': r['determinism_passed'], 'cache_zero_new_backend_calls': r['cache_zero_new_backend_calls']} for r in online]
    retrieval_determinism = [{'scene': r['scene'], 'query_id': r['query_id'], 'method': r['method'], 'passed': r['determinism_passed'], 'cache_zero_new_backend_calls': r['cache_repeat_zero_new_backend_calls']} for r in retrieval]
    write_json(root / 'determinism_report.json', {'online': determinism, 'retrieval': retrieval_determinism, 'expected_online_pairs': len(required_directories), 'expected_retrieval_pairs': expected_retrieval, 'all_complete': len(online) == len(required_directories) and len(retrieval) == expected_retrieval, 'observed_passed': all(r['passed'] and r['cache_zero_new_backend_calls'] for r in determinism+retrieval_determinism), 'structured_hash_excludes_timing': True})
    write_json(root / 'performance_report.json', {'scope': 'recorded replay, concurrent CPU/GPU workload; no vehicle control accuracy claim', 'rows': [{k: r[k] for k in ('scene', 'query_id', 'role', 'method', 'initialization_and_scene_runtime_s', 'planning_time_s', 'total_frame_time_s', 'path_length_m', 'minimum_safety_distance_m', 'total_absolute_turn_rad', 'start_projection_distance_m', 'target_projection_distance_m')} for r in online], 'initialization_note': 'scene runtime includes mapper/navigation setup; persistent model cold initialization is logged separately per run; GPU peak is allocator peak since process initialization'})
    stats = []
    for role in ('legacy_regression', 'development', 'sealed_test'):
        members = [r for r in online if r['role'] == role]
        if members:
            stats.append({'scope': 'online', 'role': role, **paired_scene_bootstrap(members, 'off_drivable', 'off_lane', 'query_success')})
    for role in ('legacy_regression', 'sealed_test'):
        online_members = [r for r in online if r['role'] == role]
        for lane in ('drivable', 'lane'):
            for a, b in (('off', 'base'), ('off', 'adapter'), ('base', 'adapter')):
                for metric in ('query_success', 'target_confirmation_correct', 'observed_planner_reachability_rate'):
                    if online_members:
                        stats.append({'scope': 'online', 'role': role, **paired_scene_bootstrap(online_members, f'{a}_{lane}', f'{b}_{lane}', metric)})
        members = [r for r in retrieval if r['role'] == role]
        if not members:
            continue
        for lane in ('drivable', 'lane'):
            for a, b in (('off', 'base'), ('off', 'adapter'), ('base', 'adapter')):
                for metric in ('top1', 'mrr', 'candidate_recall', 'target_confirmation_correct'):
                    stats.append({'scope': 'all_query_retrieval', 'role': role, **paired_scene_bootstrap(members, f'{a}_{lane}', f'{b}_{lane}', metric)})
    for role in ('legacy_regression', 'development', 'sealed_test'):
        members = [r for r in static if r['role'] == role]
        for method in ('drivable_area', 'lane_area', 'lane_constrained'):
            stats.append({'scope': 'four_map_static', 'role': role, **paired_scene_bootstrap(members, 'free_space', method, 'query_success')})
    semantic = read(root/'semantic_report.json', {})
    stats += [{'scope': 'query_parse_semantics', **comparison} for comparison in semantic.get('comparisons', [])]
    stratified = []
    for scope, rows in (('online', online), ('all_query_retrieval', retrieval), ('four_map_static', static)):
        for dimension in ('location', 'road_context', 'intersection', 'turn', 'light', 'weather'):
            enriched = [{**r, 'stratum': str(scene_index[r['scene']]['strata'][dimension])} for r in rows]
            stratified.append({'scope': scope, 'dimension': dimension, 'groups': grouped(enriched, ['query_success' if scope != 'all_query_retrieval' else 'target_confirmation_correct', *metric_names], ['role', 'stratum', 'method'])})
    coverage = read(root/'coverage_audit.json', {})
    coverage_index = {r['scene']: r for r in coverage.get('scenes', [])}
    for scope, rows in (('online', online), ('all_query_retrieval', retrieval), ('four_map_static', static)):
        for dimension in ('visible_target_density_frame0', 'category_complexity', 'query_type', 'target_presence'):
            enriched = []
            for row in rows:
                scene_coverage = coverage_index[row['scene']]
                query = queries[row['query_id']]
                count = scene_coverage['visible_official_target_count_frame0']
                categories = scene_coverage['distinct_target_categories']
                value = {
                    'visible_target_density_frame0': 'zero' if count == 0 else '1_to_4' if count < 5 else '5_or_more',
                    'category_complexity': '0_to_1_categories' if categories < 2 else '2_to_3_categories' if categories < 4 else '4_or_more_categories',
                    'query_type': query['query_type'],
                    'target_presence': 'positive' if query['accepted_instance_tokens'] else 'no_match',
                }[dimension]
                enriched.append({**row, 'stratum': value})
            stratified.append({'scope': scope, 'dimension': dimension, 'post_selection_descriptive_analysis': True, 'groups': grouped(enriched, ['query_success' if scope != 'all_query_retrieval' else 'target_confirmation_correct', *metric_names], ['role', 'stratum', 'method'])})
    physical = read(root/'physical_coverage_audit.json', {})
    physical_index = {row['scene']: row for row in physical.get('scenes', [])}
    if physical_index:
        for scope, rows in (('online', online), ('all_query_retrieval', retrieval), ('four_map_static', static)):
            for dimension in ('physical_parking_context', 'physical_intersection_context', 'physical_turn_context'):
                enriched = [{**row, 'stratum': physical_index[row['scene']][dimension]} for row in rows]
                stratified.append({'scope': scope, 'dimension': dimension, 'post_selection_descriptive_analysis': True, 'source': 'physical_coverage_audit.json', 'groups': grouped(enriched, ['query_success' if scope != 'all_query_retrieval' else 'target_confirmation_correct', *metric_names], ['role', 'stratum', 'method'])})
    if all('prospective_strata' in scene for scene in scenes):
        for scope, rows in (('online', online), ('all_query_retrieval', retrieval), ('four_map_static', static)):
            for dimension in sorted(scenes[0]['prospective_strata']):
                enriched = [{**row, 'stratum': str(scene_index[row['scene']]['prospective_strata'][dimension])} for row in rows]
                stratified.append({'scope': scope, 'dimension': dimension, 'prospective_sampling_dimension': True, 'source': 'scene_selection_manifest.json + sampling_plan.json', 'groups': grouped(enriched, ['query_success' if scope != 'all_query_retrieval' else 'target_confirmation_correct', *metric_names], ['role', 'stratum', 'method'])})
    write_json(root/'stratified_report.json', {'stratified_results': stratified, 'density_bin_edges': [0, 1, 5], 'category_count_bin_edges': [0, 2, 4], 'additional_analysis_scope': 'Post-selection descriptive bins from frozen official truth and official-map/pose physical coverage; no selection, queries, labels, methods or primary comparison changed. Category complexity counts distinct visible categories across saved frames; target density counts only supported visible official objects in frame 0, not all physical obstacles.', 'interpretation': 'Descriptive only; description keyword parking means presence of park/parked text, not proof of ego presence in a carpark. Small and unbalanced strata, unspecified light/weather and post-selection density auditing do not establish population representativeness.'})
    log_sensitivity = []
    members = [{**r, 'scene': r['log_token']} for r in online if r['role'] == 'sealed_test']
    if members:
        log_sensitivity.append(paired_scene_bootstrap(members, 'off_drivable', 'off_lane', 'query_success'))
    test_scenes = [s for s in scenes if s['role'] == 'sealed_test']
    test_logs = {s['log_token'] for s in test_scenes}
    write_json(root / 'statistical_report.json', {'comparisons': stats, 'log_cluster_sensitivity': log_sensitivity, 'primary_resampling_unit': 'scene; paired query before scene means', 'iterations': 2000, 'seed': 17, 'test_scene_count': len(test_scenes), 'test_acquisition_log_count': len(test_logs), 'limitation': f'Only {len(test_scenes)} scenes and {len(test_logs)} acquisition logs; same-log scenes remain correlated. Log-level sensitivity uses equal log weights and is separate from scene macro means.', 'incomplete_comparisons_are_provisional': bool(missing_online or missing_retrieval)})
    counts = {'static_cases': len(static), 'online_pairs': len(online), 'retrieval_pairs': len(retrieval), 'expected_static_cases': expected_static, 'expected_online_pairs': len(required_directories), 'expected_retrieval_pairs': expected_retrieval}
    annotation = read(root/'annotation_audit.json', {})
    coverage_by_role = []
    for role in ('legacy_regression', 'development', 'sealed_test'):
        members = [q for q in queries.values() if q['role'] == role]
        coverage_by_role.append({'role': role, 'all_query_types': dict(Counter(q['query_type'] for q in members)), 'positive_query_types': dict(Counter(q['query_type'] for q in members if q['accepted_instance_tokens'])), 'no_match_query_types': dict(Counter(q['query_type'] for q in members if not q['accepted_instance_tokens']))})
    positive_relations = sum(q['role'] == 'sealed_test' and q['query_type'] == 'relation' and bool(q['accepted_instance_tokens']) for q in queries.values())
    distinct_instructions = len({q['query_text'] for q in queries.values()})
    write_json(root/'query_coverage_audit.json', {'roles': coverage_by_role, 'positive_queries': sum(bool(q['accepted_instance_tokens']) for q in queries.values()), 'no_match_queries': sum(not q['accepted_instance_tokens'] for q in queries.values()), 'distinct_raw_instructions': distinct_instructions, 'positive_relation_test_queries': positive_relations, 'limitation': f'{len(queries)} scene-query records reuse {distinct_instructions} raw instruction templates; scene diversity does not establish language diversity. Positive relational retrieval has {positive_relations} test queries; zero denotes an undefined metric, never a successful reasoning result.'})
    annotation.update(retrieval_metrics_computed=bool(retrieval), retrieval_matrix_complete=not missing_retrieval, retrieval_query_ids_evaluated=len({r['query_id'] for r in retrieval}), test_evaluation_started=True, confirmation_denominator='positive target queries only; negative rejection separately reported', authoring_metadata_preserved='annotation_audit_authoring_v1.json', reporting_revision='reporting_protocol_v2.json')
    write_json(root/'annotation_audit.json', annotation)
    truth_manifest = read(root/'ground_truth_manifest.json')
    truth_manifest.update(test_evaluation_started=True, authoring_metadata_preserved='ground_truth_manifest_authoring_v1.json', evaluation_status_reference='validation_report.json')
    write_json(root/'ground_truth_manifest.json', truth_manifest)
    write_json(root / 'progress_report.json', {'goal_complete': False, 'updated_unix': time.time(), 'counts': counts, 'missing_online_jobs': missing_online, 'missing_retrieval_jobs': missing_retrieval, 'missing_static_scenes': missing_static, 'pending_validation': ['final external path audit', 'full GPU/retrieval matrix', 'final checks and Chinese report']})
    print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
