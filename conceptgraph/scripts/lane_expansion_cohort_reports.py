"""Keep the accessed diagnostic cohort and the prospective test cohort separate."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from conceptgraph.nuscenes.benchmark_fingerprints import file_sha256
from conceptgraph.nuscenes.expansion_benchmark import content_sha256, paired_scene_bootstrap
from conceptgraph.scripts.lane_expansion_report_tables import rate
from conceptgraph.scripts.summarize_lane_expansion import RETRIEVAL_METRIC_DEFINITIONS, grouped
from conceptgraph.scripts.run_incremental_semantic_navigation import write_json


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def refresh_annotation_status(root, queries):
    """Refresh derived coverage without rewriting the frozen labels or audits."""
    geometry = read(root / 'annotation_geometry_audit.json', {})
    counts = geometry.get('counts', {})
    crops = read(root / 'crop_leakage_audit.json', {})
    extended = read(root / 'leakage_extended_audit.json', {})
    cross = read(root / 'cross_cohort_leakage_audit.json', {})
    release = read(root / 'test_release.json', {})
    hashes = release.get('label_query_files_sha256', {})
    hashes_match = bool(hashes) and all((root / name).is_file() and file_sha256(root / name) == digest for name, digest in hashes.items())
    positive = sum(bool(q['accepted_instance_tokens']) for q in queries)
    official, nodes, matched = (counts.get(name, 0) for name in ('official_visible_instances', 'system_frame0_nodes', 'matched_nodes'))
    coverage = {
        'query_count': len(queries), 'official_derived_query_count': len(queries),
        'positive_queries': positive, 'no_match_queries': len(queries) - positive,
        'human_review_count': 0, 'queries_excluded_for_unknown_retrieval_truth': 0,
        'no_match_queries_excluded_from_top1_mrr_only': len(queries) - positive,
        'official_first_visible_instance_records': official,
        'system_frame0_node_count': nodes, 'matched_node_count': matched,
        'matched_node_fraction': matched / nodes if nodes else None,
        'official_instance_association_fraction': matched / official if official else None,
        'unmatched_system_node_count': counts.get('unmatched_nodes'),
        'ambiguous_system_node_count': counts.get('ambiguous_nodes'),
        'ambiguous_nodes_are_also_counted_as_unmatched': True,
        'unknown_color_instance_records': official,
        'color_truth_unknown_fraction': 1.0 if official else None,
        'unreviewed_fine_visual_attribute_instance_records': official,
        'unknown_ordinary_reachability_positive_queries': positive,
        'unknown_strict_reachability_positive_queries': positive - sum(bool(q.get('strict_lane_unreachable_proven')) for q in geometry.get('queries', [])),
        'frozen_label_query_hashes_match_release': hashes_match,
        'association_failures_remain_in_positive_query_denominators': True,
        'units': 'query counts use unique scene-query records; object coverage uses first-visible per-scene official records, not independently sampled frames',
    }
    write_json(root / 'annotation_coverage_report.json', coverage)
    status = read(root / 'annotation_audit.json', {})
    status.update(
        node_mapping_review='algorithmic geometric audit completed, not human reviewed',
        official_node_association_counts=counts, frozen_label_fields_unchanged=hashes_match,
        near_route_and_crop_audit='completed; see leakage_status_report.json and preserved raw audits',
        unreachable_and_projection_difficulty_coverage=f"{len(queries)-positive} independently derived no-match queries; {geometry.get('projection_difficulty_query_count', 0)} boundary/projection-difficulty queries; {coverage['unknown_strict_reachability_positive_queries']} positive queries lack independently proven strict reachability status.",
        quantitative_unknown_and_exclusion_reference='annotation_coverage_report.json',
    )
    write_json(root / 'annotation_audit.json', status)
    leakage = {
        'test_released': release.get('approved_for_frozen_evaluation', False),
        'frozen_label_query_hashes_match_release': hashes_match,
        'crop_exact_duplicate_gate_passed': crops.get('exact_duplicate_gate_passed', False),
        'within_cohort_near_route_pairs': len(extended.get('cross_split_near_routes', [])),
        'within_cohort_and_legacy_crop_audit_near_pairs': crops.get('near_crop_pairs_count'),
        'prior_50_cohort_cross_audit_performed': bool(cross),
        'prior_cohort_near_route_pairs': len(cross.get('near_routes_30m', [])),
        'prior_cohort_near_image_pairs': len(cross.get('images', {}).get('near_threshold_6', [])),
        'prior_cohort_near_crop_pairs': len(cross.get('crops', {}).get('near_threshold_6', [])),
        'prior_cohort_minimum_same_location_temporal_gap_s': cross.get('minimum_same_location_temporal_gap_s'),
        'interpretation': 'leakage_extended_audit.json preserves preparation-time pending fields. This derived status uses the later completed audits and frozen release; it does not retroactively rewrite the original audit. All near pairs retained and disclosed; no geographic or appearance independence claim.',
    }
    write_json(root / 'leakage_status_report.json', leakage)
    return coverage


def supplement_readiness(root, selected, queries, faults):
    """Keep each acceptance component pending until its own evidence exists."""
    navigation = read(root / 'navigation_report.json', {})
    release = read(root / 'test_release.json', {})
    sampling_frozen = bool(
        release.get('all_requested_sampling_dimensions_prospectively_allocated')
        and release.get('method_results_used_to_select_test_scenes_or_queries') is False
        and release.get('sampling_plan_sha256') == file_sha256(root / 'sampling_plan.json')
    )
    retrieval = read(root / 'retrieval_report.json', {})
    ablation = read(root / 'ablation_report.json', {})
    geometry = read(root / 'annotation_geometry_audit.json', {})
    labels = read(root / 'ground_truth_manifest.json', {})
    counts = Counter(q['scene'] for q in queries)
    off = [r for r in navigation.get('rows', []) if r['variant'] == 'off']
    gpu = [r for r in navigation.get('rows', []) if r['variant'] != 'off']
    determinism = read(root / 'determinism_report.json', {})
    repeat_rows = determinism.get('online', []) + determinism.get('retrieval', [])
    repeat_content = read(root / 'semantic_repeat_audit.json', {})
    active_paths = read(root / 'path_audits/production_active_online.json', {})
    index = read(root / 'production_job_index.json', {})
    run_names = set(read(root / 'experiment_run_manifest.json', {}).get('active_runs', {}).values())
    path_audits = {name: read(root / 'path_audits' / f'{name}.json', {}) for name in run_names}
    paths_passed = bool(
        active_paths.get('passed') and active_paths.get('completed_job_pairs') == 234
        and active_paths.get('source_index_sha256') == index.get('sha256')
        and len(path_audits) == 9 and all(p.get('passed') for p in path_audits.values())
    )
    statistics = read(root / 'statistical_report.json', {})
    semantics = read(root / 'semantic_report.json', {})
    visual = read(root / 'visualizations/visualization_manifest.json', {})
    browser = read(root / 'visualizations/browser_qa_report.json', {})
    visual_source = root / 'visualizations/source_manifest.json'
    visual_metadata = read(visual_source, {})
    renderer_current = (
        visual_metadata.get('renderer_template_sha256') == file_sha256(Path(__file__).with_name('lane_expansion_browser.html'))
        and visual_metadata.get('exporter_sha256') == file_sha256(Path(__file__).with_name('visualize_lane_expansion.py'))
    )
    cache_gpu = gpu + [r for r in retrieval.get('rows', []) if r['variant'] != 'off']
    return {
        'data_scale': {
            'complete': len(selected) == 19 and sampling_frozen and all(read(root / name, {}).get('passed') for name in ('preprocessing_integrity_report.json', 'dataset_integrity_report.json', 'missing_data_manifest.json', 'sampling_feature_consistency_audit.json', 'complete_input_resolution_audit_v2.json')),
            'fresh_scene_count': len(selected), 'fresh_frame_count': labels.get('frame_count'),
            'prospective_sampling_plan_matches_pre_execution_release': sampling_frozen,
        },
        'ground_truth': {
            'complete': len(geometry.get('scenes', [])) == len(selected) == len(counts) == 19 and len(queries) == 190 and min(counts.values(), default=0) >= 10 and labels.get('derivation_uses_qwen_or_rule_ranking_or_astar') is False and read(root / 'annotation_coverage_report.json', {}).get('frozen_label_query_hashes_match_release', False),
            'fresh_query_count': len(queries), 'automatic_association_counts': geometry.get('counts'), 'human_review_count': geometry.get('human_review_count'),
        },
        'split': {
            'complete': bool(selected) and not read(root / 'leakage_audit.json', {}).get('same_log_leakage', True) and not read(root / 'leakage_audit.json', {}).get('legacy_test_overlap', True) and all(read(root / name, {}).get('passed') for name in ('official_split_audit.json', 'temporal_adjacency_audit.json', 'cross_cohort_leakage_audit.json')) and read(root / 'crop_leakage_audit.json', {}).get('exact_duplicate_gate_passed', False),
            'test_scene_count': sum(r['role'] == 'sealed_test' for r in selected),
            'test_log_count': len({r['log_token'] for r in selected if r['role'] == 'sealed_test'}),
        },
        'cpu': {
            'complete': len(off) == 114 and all(r['execution_passed'] for r in off) and ablation.get('all_complete', False) and ablation.get('completed_cases') == 760,
            'completed_online_pairs': len(off), 'expected_online_pairs': 114, 'completed_static_cases': ablation.get('completed_cases', 0), 'expected_static_cases': 760,
        },
        'real_gpu': {
            'complete': len(gpu) == 120 and all(r['execution_passed'] and r['qwen_first']['backend_call_count'] > 0 for r in gpu) and retrieval.get('all_complete', False) and len(retrieval.get('rows', [])) == 600,
            'completed_online_pairs': len(gpu), 'expected_online_pairs': 120, 'completed_retrieval_pairs': len(retrieval.get('rows', [])), 'expected_retrieval_pairs': 600,
        },
        'cache': {
            'complete': determinism.get('all_complete', False) and len(repeat_rows) == 834 and all(r['passed'] and r['cache_zero_new_backend_calls'] for r in repeat_rows) and len(cache_gpu) == 520 and all(r['qwen_cache_repeat']['cache_hit_count'] > 0 and r['qwen_cache_repeat']['backend_call_count'] == 0 for r in cache_gpu) and repeat_content.get('passed', False) and repeat_content.get('all_complete', False),
            'completed_repeat_pairs': len(repeat_rows), 'expected_repeat_pairs': 834, 'intermediate_delivery_state_different_pairs': repeat_content.get('strict_delivery_state_different_pairs'),
        },
        'fault_safety': {
            'complete': faults['passed'] and paths_passed and repeat_content.get('cross_condition_shared_geometry', {}).get('passed', False),
            'fault_summary': faults, 'active_online_pairs_audited': active_paths.get('completed_job_pairs', 0), 'expected_online_pairs': 234, 'accepted_or_reused_path_checks': active_paths.get('accepted_or_reused_paths_checked', 0), 'required_other_path_audits': len(run_names), 'other_path_audits_passed': sum(p.get('passed', False) for p in path_audits.values()),
        },
        'statistics': {
            'complete': navigation.get('all_complete', False) and retrieval.get('all_complete', False) and statistics.get('iterations', 0) >= 2000 and not statistics.get('incomplete_comparisons_are_provisional', True) and semantics.get('all_complete', False) and (root / 'stratified_report.json').exists(),
            'bootstrap_iterations': statistics.get('iterations'), 'distinct_instruction_templates': semantics.get('unique_instruction_count'),
        },
        'documentation': {
            'complete': visual.get('all_required_views_present', False) and visual.get('all_experiments_complete', False) and browser.get('passed', False) and visual_source.exists() and browser.get('source_manifest_sha256') == file_sha256(visual_source) and renderer_current,
            'browser_qa_source': str(root / 'visualizations/browser_qa_report.json'),
        },
    }


def build_cohort_reports(root):
    supplement = root / 'prospective_supplement_v2'
    if not (supplement / 'sampling_plan.json').exists():
        return None
    old_selection = read(root / 'scene_selection_manifest.json')['scenes']
    new_selection = read(supplement / 'scene_selection_manifest.json', {}).get('scenes', [])
    legacy = {r['scene'] for r in old_selection if r['role'] == 'legacy_regression'}
    old_queries = [json.loads(line) for line in (root / 'benchmark_queries.jsonl').open()]
    new_queries = [json.loads(line) for line in (supplement / 'benchmark_queries.jsonl').open()] if (supplement / 'benchmark_queries.jsonl').exists() else []
    annotation_coverage = {
        'original_50': refresh_annotation_status(root, old_queries),
        'fresh_19': refresh_annotation_status(supplement, new_queries),
    }
    assert not {r['scene'] for r in old_selection} & {r['scene'] for r in new_selection}
    queries = [q for q in old_queries if q['scene'] in legacy] + new_queries
    primary_scenes = [r for r in old_selection if r['scene'] in legacy] + new_selection
    all_scenes = old_selection + new_selection
    expected = {'online_pairs': 324, 'retrieval_pairs': 720, 'static_cases': 1200}
    report_names = {'online_pairs': 'navigation_report', 'retrieval_pairs': 'retrieval_report', 'static_cases': 'ablation_report'}
    primary_rows, global_counts = {}, {}
    completed = True
    for key, filename in report_names.items():
        original = read(root / f'{filename}.json', {})
        fresh = read(supplement / f'{filename}.json', {})
        rows = original.get('rows', [])
        new_rows = fresh.get('rows', [])
        primary_rows[key] = [r for r in rows if r['scene'] in legacy] + new_rows
        assert len({(r['query_id'], r['method']) for r in primary_rows[key]}) == len(primary_rows[key])
        global_counts[key] = len(rows) + len(new_rows)
        completed = completed and original.get('all_complete', False) and fresh.get('all_complete', False) and len(primary_rows[key]) == expected[key]
    complete_marker = read(supplement / 'matrix_complete.json', {})
    cohort_manifest = read(supplement / 'prospective_evaluation_cohort_manifest.json', {})
    labels_match = bool(queries) and content_sha256(queries) == cohort_manifest.get('all_query_records_sha256')
    completed = bool(completed and complete_marker.get('passed') and labels_match and len(primary_scenes) == 30 and len(queries) == 300)
    retrieval_metrics = ['top1', 'mrr', 'candidate_recall', 'target_confirmation_correct', 'correct_rejection']
    online_metrics = ['query_success', 'planning_success_rate', 'target_confirmation_correct', 'no_match_correct_rejection', *retrieval_metrics[:3]]
    navigation = {'rows': primary_rows['online_pairs'], 'summaries': grouped(primary_rows['online_pairs'], online_metrics, ['role', 'method'])}
    retrieval = {'rows': primary_rows['retrieval_pairs'], 'summaries': grouped(primary_rows['retrieval_pairs'], retrieval_metrics, ['role', 'method']), 'metric_definitions': RETRIEVAL_METRIC_DEFINITIONS}
    ablation = {'rows': primary_rows['static_cases'], 'summaries': grouped(primary_rows['static_cases'], ['query_success', *retrieval_metrics], ['role', 'method'])}
    comparisons = []
    for role in ('legacy_regression', 'development', 'sealed_test'):
        for scope, rows, metrics in (('online', primary_rows['online_pairs'], ['query_success', 'target_confirmation_correct']), ('all_query_retrieval', primary_rows['retrieval_pairs'], retrieval_metrics), ('four_map_static', primary_rows['static_cases'], ['query_success'])):
            members = [r for r in rows if r['role'] == role]
            if not members:
                continue
            if scope == 'four_map_static':
                methods = [('free_space', method) for method in ('drivable_area', 'lane_area', 'lane_constrained')]
            else:
                methods = [('off_drivable', 'off_lane')]
                methods += [(f'{a}_{mode}', f'{b}_{mode}') for mode in ('drivable', 'lane') for a, b in (('off', 'base'), ('off', 'adapter'), ('base', 'adapter'))] if role != 'development' else []
            for a, b in methods:
                for metric in metrics:
                    comparisons.append({'scope': scope, 'role': role, **paired_scene_bootstrap(members, a, b, metric)})
    features = read(supplement / 'candidate_metadata_features.json', {}).get('scenes', [])
    feature_index = {r['scene']: r['strata'] for r in features}
    stratified = []
    for scope, rows, metrics in (('online', primary_rows['online_pairs'], online_metrics), ('all_query_retrieval', primary_rows['retrieval_pairs'], retrieval_metrics), ('four_map_static', primary_rows['static_cases'], ['query_success', *retrieval_metrics])):
        for dimension in read(supplement / 'sampling_plan.json')['dimensions']:
            enriched = [{**r, 'stratum': str(feature_index[r['scene']][dimension])} for r in rows]
            stratified.append({'scope': scope, 'dimension': dimension, 'groups': grouped(enriched, metrics, ['role', 'stratum', 'method']), 'fixed_legacy_not_resampled': True})
    test_scenes = [r for r in new_selection if r['role'] == 'sealed_test']
    query_counts = Counter(q['scene'] for q in queries)
    qwen_rows = [r for r in primary_rows['online_pairs'] if r['variant'] != 'off']
    tasks = defaultdict(Counter)
    for row in qwen_rows:
        key = row['role'], row['method']
        tasks[key].update({'backend_call_count': row['qwen_first']['backend_call_count'], 'schema_valid_fresh_response_count': row['qwen_first']['successful_backend_response_count'], 'cache_repeat_backend_call_count': row['qwen_cache_repeat']['backend_call_count'], 'cache_repeat_hits': row['qwen_cache_repeat']['cache_hit_count'], 'fallback_count': row['qwen_first']['fallback_count']})
        tasks[key].update(row['qwen_applications'])
    qwen_totals = [{'role': k[0], 'method': k[1], **dict(value)} for k, value in sorted(tasks.items())]
    fault_run = read(supplement / 'experiment_run_manifest.json', {}).get('active_runs', {}).get('faults', 'faults_matrix_v2')
    fault_reports = [read(path) for path in sorted((supplement / 'experiments' / fault_run).glob('*/complete.json'))]
    fault_cases = [case for report in fault_reports for case in report['cases']]
    fault_summary = {'fresh_test_scenes': len(fault_reports), 'case_count': len(fault_cases), 'expected_case_count': 360, 'passed': len(fault_reports) == 10 and len(fault_cases) == 360 and all(r['passed'] for r in fault_reports), 'not_applicable_count': sum(r.get('passed') is None for r in fault_cases), 'failed_cases': [r for r in fault_cases if r.get('passed') is False], 'baseline_path_presence_counts': dict(Counter(str(r.get('baseline_legal_path_exists')) for r in fault_cases)), 'all_backends_synthetic': True, 'real_qwen_backend_calls': 0}
    log_rows = [{**r, 'scene': r['log_token']} for r in primary_rows['online_pairs'] if r['role'] == 'sealed_test']
    log_sensitivity = [paired_scene_bootstrap(log_rows, 'off_drivable', 'off_lane', 'query_success')] if log_rows else []
    primary = {'all_complete': completed, 'scope': 'Original mandatory eleven fixed development/regression scenes plus nineteen prospectively stratified fresh scenes. New independent test is only the ten fresh scenes, never the previously accessed original test cohort.', 'manifest': str(supplement / 'prospective_evaluation_cohort_manifest.json'), 'scene_count': len(primary_scenes), 'query_count': len(queries), 'queries_per_scene': dict(query_counts), 'query_records_match_pre_execution_frozen_manifest': labels_match, 'test_scene_count': len(test_scenes), 'test_acquisition_log_count': len({r['log_token'] for r in test_scenes}), 'completed_counts': {key: len(rows) for key, rows in primary_rows.items()}, 'expected_counts': expected, 'navigation': navigation, 'retrieval': retrieval, 'four_map_static': ablation, 'qwen_online_totals': qwen_totals, 'statistical_comparisons': comparisons, 'stratified_results': stratified, 'log_cluster_sensitivity': log_sensitivity, 'bootstrap_iterations': 2000, 'seed': 17, 'new_fault_summary': fault_summary, 'query_coverage': {'positive': sum(bool(q['accepted_instance_tokens']) for q in queries), 'no_match': sum(not q['accepted_instance_tokens'] for q in queries), 'positive_test': sum(q['role'] == 'sealed_test' and bool(q['accepted_instance_tokens']) for q in queries), 'no_match_test': sum(q['role'] == 'sealed_test' and not q['accepted_instance_tokens'] for q in queries), 'positive_test_relation': sum(q['role'] == 'sealed_test' and q['query_type'] == 'relation' and bool(q['accepted_instance_tokens']) for q in queries), 'distinct_raw_instruction_templates': len({q['query_text'] for q in queries})}, 'limitations': ['Original fixed eleven scenes were previously used for Qwen training/development; their results are regression only.', 'New test selection is independent of method outcomes, conditional on official metadata and the local authorized source pool; not population representative.', 'Small and uneven strata, seven fresh test acquisition logs, unknown light/weather and nearby routes/crops limit generalization.', 'Ten repeated raw instruction templates do not establish language diversity.', 'Association is conservative automatic official-instance geometry, not human review; omitted positive targets remain failures.', 'Navigation is controlled recorded replay; no vehicle control or camera lane-line recognition.']}
    write_json(root / 'prospective_evaluation_report.json', primary)
    combined = {'all_complete': completed, 'unique_scene_count': len(all_scenes), 'benchmark_query_count': len(old_queries)+len(new_queries), 'counts': global_counts, 'expected_counts': {'online_pairs': 678, 'retrieval_pairs': 1320, 'static_cases': 2760}, 'no_scene_query_or_run_repetition_increases_scene_count': True, 'original_accessed_cohort': {'scene_count': 50, 'test_scene_count': 10, 'report': 'navigation_report.json + retrieval_report.json + ablation_report.json'}, 'prospective_primary_cohort': {'scene_count': 30, 'mandatory_legacy_count': 11, 'fresh_expansion_count': 19, 'fresh_test_count': 10, 'report': 'prospective_evaluation_report.json'}, 'overlap_policy': 'The eleven required fixed legacy scenes appear in both cohort reports and are counted once in global unique counts; never pool accessed and fresh test averages.', 'fresh_execution_evidence': str(supplement / 'matrix_complete.json'), 'source_manifests_sha256': {str(path.relative_to(root)): file_sha256(path) for path in (root / 'scene_selection_manifest.json', supplement / 'scene_selection_manifest.json', supplement / 'sampling_plan.json', supplement / 'test_release.json') if path.exists()}}
    combined['annotation_coverage_by_disjoint_cohort'] = annotation_coverage
    combined['data_inventory_scopes'] = {
        'original_selected_50': 'data_inventory.json + missing_data_manifest.json',
        'fresh_selected_19_and_latest_850_scene_availability_snapshot': 'prospective_supplement_v2/data_inventory.json + missing_data_manifest.json',
        'latest_global_availability_summary': read(supplement / 'data_inventory.json', {}).get('summary'),
        'note': 'Original inventory refresh updates only its original fifty selected rows; the supplementary inventory records later archive-02 eligibility and the sixty-nine actual prepared scenes. Initial inventories remain preserved separately.',
    }
    write_json(root / 'combined_evidence_report.json', combined)
    lines = ['# 补正后的事前分层评测', '', '验收状态：'+('完整矩阵与专项核验已完成，最终全局工程及旧产物验收另见validation_report.json。' if completed else '仍在执行，未将待完成条件记作成功。'), '', '原50场景的全部结果与原划分保留。该组未在选场前量化全部分层维度，且曾在安全修订前访问测试结果，继续作为有明确限制的诊断组。补正组由必须保留的原11个开发/回归scene，以及另19个按九维规则事前抽样的新scene组成，合计30场景、300条查询；其中新开发9个、新测试10个。新测试排除了此前50场景及新开发的所有采集log，方法、查询、标签和配置在新测试执行前冻结，没有按方法结果选场或调参。', '', '整体共69个不同scene、690条scene-query记录。原11场景在两个报告中有意复用，整体场景计数只计一次；原测试与新测试分别汇总。抽样计划、全部候选官方地图/标注特征、未覆盖类别及实际数据的一致性核验见prospective_supplement_v2/。', '', '| 补正组范围 | 已完成 | 要求 |', '|---|---:|---:|']
    lines += [f'| {key} | {len(primary_rows[key])} | {value} |' for key, value in expected.items()]
    lines += ['', '## 全查询检索与独立目标真值', '', '| 集合 | 条件 | 查询条件数 | Top-1 | MRR | 候选召回 | 正目标确认 | 无匹配正确拒绝 |', '|---|---|---:|---|---|---|---|---|']
    for group in retrieval['summaries']:
        values = group['overall_query_means']
        lines.append(f"| {group['role']} | {group['method']} | {group['query_condition_count']} | "+' | '.join(rate(values[k]) for k in retrieval_metrics)+' |')
    lines += ['', f"补正组正查询{primary['query_coverage']['positive']}条，无匹配{primary['query_coverage']['no_match']}条；新测试正查询{primary['query_coverage']['positive_test']}条、无匹配{primary['query_coverage']['no_match_test']}条，其中关系正查询{primary['query_coverage']['positive_test_relation']}条。正查询的关联遗漏保留为失败，无匹配查询不计Top-1/MRR。颜色、细粒度属性及未证明的可达性仍为未知，没有声称人工审核。", '', '## 逐帧导航与规划尝试', '', '| 集合 | 条件 | 查询数 | 查询路径成功率 | scene宏平均 | 规划成功/尝试 | 正目标确认 | 无匹配锁定拒绝 |', '|---|---|---:|---|---|---|---|---|']
    for group in navigation['summaries']:
        members = [r for r in navigation['rows'] if r['role'] == group['role'] and r['method'] == group['method']]
        values = group['overall_query_means']
        lines.append(f"| {group['role']} | {group['method']} | {len(members)} | {rate(values['query_success'])} | {rate(group['scene_macro_means']['query_success'])} | {sum(r['successful_plans'] for r in members)}/{sum(r['planning_attempts'] for r in members)} | {rate(values['target_confirmation_correct'])} | {rate(values['no_match_correct_rejection'])} |")
    lines += ['', '路径成功表示查询回放中至少接受一条路径，不表示选对或已到达目标。全量四地图、逐scene、九个分层维度、scene宏平均、实际规划尝试、失败事件、投影/路径/clearance/耗时分布分别在本报告JSON及补充组对应报告中保留。不能把栅格距离解释成车辆控制精度。', '', '## 新测试配对差值', '', '| 范围 | 指标 | 差值方向 | 配对查询 | scene | 差值 | 95% CI |', '|---|---|---|---:|---:|---:|---|']
    for comparison in comparisons:
        if comparison['role'] != 'sealed_test' or comparison['scope'] not in {'online', 'all_query_retrieval'} or comparison['metric'] not in {'query_success', 'top1', 'mrr'}:
            continue
        lines.append(f"| {comparison['scope']} | {comparison['metric']} | {comparison['difference']} | {comparison['paired_queries']} | {comparison['scene_count']} | {comparison['mean_difference']} | {comparison['ci95']} |")
    lines += ['', f"同scene、同query先配对，再按scene聚类bootstrap，seed=17，2000次；新测试来自{primary['test_acquisition_log_count']}个log，另报log聚类敏感性。小样本、地板效应、同log关联和少量正关系查询限制结论，零差值或退化区间不能证明模型等效。", '', '## Qwen实际调用与在线应用', '', '| 集合/条件 | 首次调用 | 首次合法响应 | 解析/属性/重排应用 | 缓存新增后端 | 缓存命中 | fallback |', '|---|---:|---:|---|---:|---:|---:|']
    for total in qwen_totals:
        apps = '/'.join(str(total.get(name+'_applied', 0)) for name in ('query_parse', 'object_attributes', 'candidate_rerank'))
        lines.append(f"| {total['role']}/{total['method']} | {total['backend_call_count']} | {total['schema_valid_fresh_response_count']} | {apps} | {total['cache_repeat_backend_call_count']} | {total['cache_repeat_hits']} | {total['fallback_count']} |")
    lines += ['', '这里是补正30场景范围中的144项真实GPU在线条件，全部查询六条件另列，不混入在线应用计数。base主提示的解析schema失败继续保留；base解析成功应用证据来自独立原训练场景的既有stage4完整提示验证，不替换主矩阵协议或并入新测试平均。三类任务实际调用、合法schema、应用、缓存与回退分别记录，未新增训练或下载Qwen。', '', '## 安全、缓存和证据范围', '', '全部接受路径及每帧复用剩余段均按保存的实际occupancy独立核验连续碰撞，严格车道条件另外核验线段polygon覆盖、方向和合法有向转换。故障后端为合成，与真实模型计数分开；原两场景及两组各10个测试场景都保留故障矩阵，实际没有可用对象的过期版本注入记不适用。', '', '首次/缓存导航结果、实际栅格数组和原始模型响应分别核验。帧末active_rerank建议可能因缓存更快而提前可见；严格包含中间投递状态的差异单独保留，不宣称该状态逐帧完全相同，实际导航与缓存输出的内容哈希仍须通过。', '', '新测试的车道及地图来自真实HD Map。Qwen只承担语义解析、真实crop属性和安全候选内重排；A*与几何安全由确定性规划层承担。这是受控录制回放，没有实现真实车辆闭环控制或摄像头车道线检测。', '', '新测试浏览器：[实际地图、GT、路径、重排、缓存与故障](prospective_supplement_v2/visualizations/index.html)。原50场景浏览器：[原诊断组](visualizations/index.html)。所有失败、不适用、未知和近似路段/crop均保留；代表案例使用冻结规则选择。', '']
    leakage_status = read(supplement / 'leakage_status_report.json', {})
    lines += ['', f"额外与此前50个诊断场景交叉比较：{leakage_status.get('prior_cohort_near_route_pairs')}对相近路段、{leakage_status.get('prior_cohort_near_image_pairs')}对近似整图和{leakage_status.get('prior_cohort_near_crop_pairs')}对近似crop，均保留；同地区最近时间间隔为{leakage_status.get('prior_cohort_minimum_same_location_temporal_gap_s')}秒。没有跨组精确重复，仍不能宣称地域或外观独立。新增组内部另有2对近似路段，内部/原11场景crop比较另有11对近似crop，各审计比较范围分开记录。", '', '准备时的pending字段保留在原始leakage_extended_audit.json中；当前完成状态另见leakage_status_report.json，不倒改原记录。定量未知字段、关联覆盖与排除数见两组annotation_coverage_report.json；实际应用与候选顺序/逐帧选择变化另见qwen_effect_report.json。', '']
    lines += ['候选召回率先对每条正查询计算“候选中找回的不同可接受官方实例数／该查询全部可接受实例数”，再平均各查询比例，属于集合召回而非任一目标命中率。多目标查询可以Top-1正确但仅找回部分可接受目标，所以该召回均值可能低于Top-1。Top-1/MRR的正查询分母和无匹配正确拒绝分母分别列出。', '']
    (root / 'prospective_evaluation_report.md').write_text('\n'.join(lines))
    readiness = supplement_readiness(supplement, new_selection, new_queries, fault_summary)
    write_json(root / 'supplementary_component_readiness.json', readiness)
    return {'complete': completed, 'combined': combined, 'primary': primary, 'supplement_path': str(supplement), 'readiness': readiness}
