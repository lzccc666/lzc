"""Describe model applications separately from observed ranking/choice changes."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from conceptgraph.nuscenes.benchmark_fingerprints import file_sha256
from conceptgraph.scripts.run_incremental_semantic_navigation import write_json


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def build_qwen_effect_report(root):
    summaries, choices, sources = [], [], {}
    complete = True
    for name, directory in (
        ('original_accessed_diagnostic_50', root),
        ('prospective_fresh_19', root / 'prospective_supplement_v2'),
    ):
        if not directory.exists():
            continue
        selection = read(directory / 'scene_selection_manifest.json', {}).get('scenes', [])
        roles = {row['scene']: row['role'] for row in selection}
        visual = read(directory / 'visualizations/visualization_manifest.json', {})
        event_path = directory / 'visualizations/rerank_before_after.json'
        navigation_path = directory / 'navigation_report.json'
        events = read(event_path, {}).get('events', [])
        navigation = read(navigation_path, {})
        complete = complete and visual.get('all_experiments_complete', False) and navigation.get('all_complete', False)
        counts = defaultdict(Counter)
        for event in events:
            group = counts[roles[event['scene']], event['method'], event['repeat']]
            before, after = event['before_verified_ids'], event['response_ranking']
            group['delivered_rerank_events'] += 1
            group['applied_events'] += bool(event['applied'])
            group['multi_candidate_events'] += len(before) > 1
            group['events_with_ranking_response'] += after is not None
            if before and after:
                group['observed_order_changes'] += before != after
                group['observed_top_candidate_changes'] += before[0] != after[0]
                if event['applied'] and len(before) > 1:
                    group['applied_multi_candidate_events_with_response'] += 1
                    group['applied_multi_candidate_order_changes'] += before != after
                    group['applied_multi_candidate_top_changes'] += before[0] != after[0]
        for (role, method, repeat), group in sorted(counts.items()):
            denominator = group['applied_multi_candidate_events_with_response']
            summaries.append({
                'cohort': name, 'role': role, 'method': method, 'repeat': repeat,
                **dict(group),
                'applied_multi_candidate_order_change_rate': group['applied_multi_candidate_order_changes'] / denominator if denominator else None,
                'order_change_rate_denominator': denominator,
                'null_reason': None if denominator else 'no applied multi-candidate event with an available ranking response',
            })
        rows = {(r['query_id'], r['method']): r for r in navigation.get('rows', [])}
        frame_cache = {}

        def selected_frames(row):
            path = Path(row['artifact']) / 'first/end_to_end_replay' / row['scene'] / 'frames.jsonl'
            if path not in frame_cache:
                frame_cache[path] = [(frame['frame_idx'], frame['selected_node_id']) for frame in (json.loads(line) for line in path.open())]
                sources[str(path.relative_to(root))] = file_sha256(path)
            return frame_cache[path]

        for (query_id, method), a in sorted(rows.items()):
            if a['variant'] != 'off':
                continue
            mode = 'lane' if a['lane'] else 'drivable'
            for variant in ('base', 'adapter'):
                b = rows.get((query_id, f'{variant}_{mode}'))
                if b is None:
                    continue
                first, second = selected_frames(a), selected_frames(b)
                assert [r[0] for r in first] == [r[0] for r in second], 'paired frame sequences differ'
                choices.append({
                    'cohort': name, 'role': a['role'], 'scene': a['scene'],
                    'query_id': query_id, 'method_a': method, 'method_b': b['method'],
                    'frame_count': len(first),
                    'frames_with_different_selected_node': sum(x[1] != y[1] for x, y in zip(first, second)),
                    'query_success_changed': a['query_success'] != b['query_success'],
                    'positive_target_confirmation_changed': a['target_confirmation_correct'] != b['target_confirmation_correct'],
                })
        for path in (event_path, navigation_path):
            if path.exists():
                sources[str(path.relative_to(root))] = file_sha256(path)
    paired_summary = []
    for cohort, role, method in sorted({(r['cohort'], r['role'], r['method_b']) for r in choices}):
        members = [r for r in choices if (r['cohort'], r['role'], r['method_b']) == (cohort, role, method)]
        paired_summary.append({
            'cohort': cohort, 'role': role, 'method_b': method, 'paired_query_count': len(members),
            'queries_with_changed_selected_node_sequence': sum(r['frames_with_different_selected_node'] > 0 for r in members),
            'queries_with_changed_path_success': sum(r['query_success_changed'] for r in members),
            'queries_with_changed_positive_target_confirmation': sum(r['positive_target_confirmation_changed'] for r in members),
        })
    report = {
        'all_complete': bool(complete), 'rerank_summaries': summaries,
        'paired_choice_summaries': paired_summary, 'paired_choice_rows': choices,
        'source_file_sha256': sources,
        'denominator_units': 'rerank events and paired scene-query replay conditions; repeated events/frames are not independent samples',
        'interpretation': [
            'Successful application can retain the original candidate order; application count is not an improvement count.',
            'Order changes compare the actual validated pre-model candidate sequence with the actual normalized ranking response.',
            'Choice-sequence differences compare complete controlled replay records against Qwen-off for the same scene/query/mode; parse and attribute effects are included, so differences cannot be attributed solely to reranking.',
            'No change, floor effects and narrow repeated language templates do not establish model equivalence or language generalization.',
            'Original accessed and fresh prospective test summaries remain separate; first and cache repeats remain separate.',
        ],
    }
    write_json(root / 'qwen_effect_report.json', report)
    return report
