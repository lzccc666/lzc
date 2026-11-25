import json

from conceptgraph.completion.semantic_evaluation import permutation_metrics


def test_permutation_consistency_counts_invalid_and_same_position_bias_as_failures():
    rows, records = [], []
    for case, selections in [('stable', [7, 7]), ('position_bias', [7, 9]), ('invalid', [7, 'broken'])]:
        for permutation, selected in enumerate(selections):
            ids = [7, 9] if permutation == 0 else [9, 7]
            sample = f'{case}-{permutation}'
            rows.append(dict(task='candidate_rerank', sample_id=sample, query_text=case,
                candidate_nodes=[dict(node_id=i) for i in ids], candidate_ids=ids,
                relation_metadata=[], acceptable_node_ids=[7]))
            raw = 'invalid' if selected == 'broken' else json.dumps(dict(schema_version='1.0.0',
                selected_node_id=selected, no_match=False, ranking=[selected, *[i for i in ids if i != selected]],
                explanation='test', confidence=.9))
            records.append(dict(sample_id=sample, envelope=dict(raw_output=raw, used_qwen=True, fallback_used=False)))
    result = permutation_metrics(records, rows)
    assert result['permutation_groups'] == 3
    assert result['identity_consistency'] == 1 / 3
    assert result['consistently_correct'] == 1 / 3
