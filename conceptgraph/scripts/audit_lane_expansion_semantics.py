"""Evaluate usable query semantics against the frozen instruction templates."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from conceptgraph.nuscenes.expansion_benchmark import paired_scene_bootstrap
from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT
from conceptgraph.scripts.run_incremental_semantic_navigation import write_json


def mean(values):
    defined = [float(value) for value in values if value is not None]
    return {'value': sum(defined) / len(defined) if defined else None, 'denominator': len(defined), 'not_applicable': len(values) - len(defined), 'empty_reason': None if defined else 'no applicable model parsing responses'}


def main():
    root = DEFAULT_ROOT
    retrieval = json.loads((root / 'retrieval_report.json').read_text())
    if not retrieval['all_complete']:
        raise RuntimeError('semantic audit requires all frozen query conditions')
    protocol = json.loads((root / 'benchmark_protocol.json').read_text())
    queries = {row['query_id']: row for line in (root / 'benchmark_queries.jsonl').open() if (row := json.loads(line))}
    rows = []
    for original in retrieval['rows']:
        row = {key: original[key] for key in ('scene', 'query_id', 'role', 'method', 'variant', 'lane')}
        row['query_text'] = queries[row['query_id']]['query_text']
        row.update(schema_valid=None, fresh_model_response=None, usable_semantic_parse_correct=None, field_correct=None, conditional_valid_schema_correct=None)
        if row['variant'] != 'off':
            requests = sorted((Path(original['artifact']) / 'first/qwen_audit').glob('*_query_parse.json'))
            if len(requests) != 1:
                raise ValueError(f'expected exactly one query parse response: {original["artifact"]}')
            path = requests[0]
            record = json.loads(path.read_text())
            response = record.get('result', record)
            output = response.get('normalized_output') or {}
            expected = protocol['templates'][int(row['query_id'].rsplit('q', 1)[1])]
            expected_fields = {'target_categories': [expected['category']], 'anchor_categories': [expected['anchor_category']] if expected.get('anchor_category') else [], 'prefer_nearest': expected['nearest'], 'relation': expected.get('relation') or 'near', 'target_colors': [], 'anchor_colors': []}
            valid = bool(response['output_schema_valid'])
            fresh = bool(response.get('used_qwen')) and not response.get('cache', {}).get('hit', False)
            fields = {key: valid and fresh and output.get(key) == value for key, value in expected_fields.items()}
            correct = all(fields.values())
            row.update(schema_valid=valid, fresh_model_response=fresh, usable_semantic_parse_correct=correct, field_correct=fields, conditional_valid_schema_correct=correct if valid and fresh else None, expected_fields=expected_fields, normalized_output=output, request_artifact=str(path), request_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['role'], row['method'])].append(row)
    summaries = []
    for (role, method), members in sorted(grouped.items()):
        scenes = defaultdict(list)
        for row in members:
            scenes[row['scene']].append(row['usable_semantic_parse_correct'])
        summaries.append({'role': role, 'method': method, 'scene_count': len(scenes), 'query_condition_count': len(members), 'schema_valid_rate': mean([row['schema_valid'] for row in members]), 'usable_semantic_parse_accuracy': mean([row['usable_semantic_parse_correct'] for row in members]), 'conditional_valid_schema_accuracy': mean([row['conditional_valid_schema_correct'] for row in members]), 'scene_macro_accuracy': mean([mean(values)['value'] for values in scenes.values()]), 'field_accuracy': {field: mean([row['field_correct'][field] if row['field_correct'] else None for row in members]) for field in ('target_categories', 'anchor_categories', 'prefer_nearest', 'relation', 'target_colors', 'anchor_colors')}})
    comparisons = []
    for role in ('legacy_regression', 'sealed_test'):
        for lane in ('drivable', 'lane'):
            comparisons.append({'role': role, **paired_scene_bootstrap([row for row in rows if row['role'] == role], f'base_{lane}', f'adapter_{lane}', 'usable_semantic_parse_correct')})
    report = {'all_complete': len(rows) == retrieval['expected_query_condition_pairs'], 'rows': rows, 'summaries': summaries, 'comparisons': comparisons, 'resampling_unit': 'scene', 'iterations': 2000, 'seed': 17, 'truth_source': 'frozen query instruction templates, not planner/Qwen/ranking outcomes', 'expected_fields': 'category, anchor category, nearest flag, relation, and absence of unrequested color constraints', 'unknown_or_not_scored': ['paraphrasing of target/anchor text', 'prefer_front interpretation of current camera image', 'object color/attribute truth', 'confidence calibration'], 'model_usable_accuracy_requires_fresh_schema_valid_response': True, 'invalid_schema_or_fallback_credited_as_correct_model_parse': False, 'off_condition': 'No Qwen query parse task; model-parse accuracy is not applicable, while retrieval/navigation remain evaluated.', 'primary_protocol_or_labels_changed': False, 'post_release_analysis': True, 'test_previously_accessed': True, 'interpretation': 'Field truth is derived from the original instruction, not from inferred object attributes. An unrequested color constraint is incorrect query parsing even though actual object colors remain unknown.'}
    patterns = defaultdict(list)
    for row in rows:
        if row['variant'] != 'off':
            patterns[(row['variant'], row['query_text'])].append(row)
    report['unique_instruction_count'] = len({query['query_text'] for query in queries.values()})
    report['instruction_patterns'] = [{'variant': variant, 'query_text': instruction, 'unique_scene_count': len({row['scene'] for row in members}), 'query_condition_count': len(members), 'usable_parse_accuracy': mean([row['usable_semantic_parse_correct'] for row in members]), 'failed_fields': sorted({key for row in members for key, value in row['field_correct'].items() if not value})} for (variant, instruction), members in sorted(patterns.items())]
    report['language_diversity_limitation'] = 'The scene-query labels reuse ten frozen raw instruction templates. Scene clustering evaluates scene variation, not independent language prompts. Degenerate parse intervals from repeated deterministic templates do not establish language generalization.'
    write_json(root / 'semantic_report.json', report)
    print(json.dumps({'all_complete': report['all_complete'], 'summaries': summaries}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
