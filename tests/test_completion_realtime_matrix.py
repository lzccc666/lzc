import json

import pytest

from conceptgraph.completion.realtime_matrix import trial_plan, verified_records
from conceptgraph.completion.stage9 import digest


def test_all_tasks_seeds_and_model_switches_have_unique_complete_denominators():
    tasks = [dict(task_id=f'c{i}-t{j}', config_id=f'c{i}', kind='positive')
             for i in range(10) for j in range(5)]
    seeds = [17, 29, 43]
    main = trial_plan(tasks, seeds, 'main', 'frozen')
    minus = trial_plan(tasks, seeds, 'semantic-minus', 'frozen_ablation')
    assert len(main) == len(minus) == len({r['run_id'] for r in main}) == 450
    assert main == trial_plan(tasks, seeds, 'main', 'frozen')
    assert {r['semantic_tasks'] for r in minus} == {
        'object_attributes,candidate_rerank', 'query_parse,candidate_rerank', 'query_parse,object_attributes'}
    for variant in {r['variant'] for r in main}:
        assert len({(r['task'], r['seed']) for r in main if r['variant'] == variant}) == 150
    with pytest.raises(ValueError, match='denominator'):
        trial_plan(tasks+[tasks[0]], seeds, 'main', 'duplicate')


def test_resume_retains_failures_and_detects_changed_results(tmp_path):
    result = tmp_path/'result.json'
    result.write_text('{"actual_full_perception_arrival": false}')
    planned = dict(run_id='trial', variant='new_full', seed=17)
    row = dict(**planned, raw_case=str(tmp_path), result_sha256=digest(result))
    records = tmp_path/'trial_records.jsonl'
    records.write_text(json.dumps(row)+'\n')
    assert verified_records(tmp_path, dict(trials=[planned])) == [row]
    result.write_text('{"actual_full_perception_arrival": true}')
    with pytest.raises(ValueError, match='changed'):
        verified_records(tmp_path, dict(trials=[planned]))
    row['result_sha256'] = None
    records.write_text(json.dumps(row)+'\n')
    assert verified_records(tmp_path, dict(trials=[planned])) == [row]
