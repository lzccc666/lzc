"""逐条核对新旧语义原始记录与现成统计，不重跑模型、不创造标注。"""
from collections import Counter
import csv
import datetime
import gzip
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]
D = R / 'nuscenes_ablation/full_semantics_near6_002'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with gzip.open(path, 'rt') as stream:
        return list(map(json.loads, stream))


def main():
    oldfile = R/'nuscenes_ablation/full_semantics_001/decisions.jsonl.gz'
    newfile = D/'decisions.jsonl.gz'
    old, new = load(oldfile), load(newfile)
    assert len(old) == len(new) == 9048
    frozen_fields = ['condition', 'sample_id', 'scene', 'log', 'role', 'kind', 'positive',
                     'ordinary_no_match', 'fault', 'covered', 'Qwen_called']
    changed_prompts, changed_raw, changed_decisions = Counter(), Counter(), Counter()
    same_prompt_different_output = []
    seen = set()
    for before, after in zip(old, new):
        key = (after['condition'], after['sample_id']); assert key not in seen; seen.add(key)
        assert all(before[k] == after[k] for k in frozen_fields), key
        assert not after['actual_lock_credit'] and not after['actual_arrival_credit']
        c = after['condition']
        if before['prompt_sha256'] != after['prompt_sha256']: changed_prompts[c] += 1
        if before['raw_output'] != after['raw_output']: changed_raw[c] += 1
        if before['answer'] != after['answer']: changed_decisions[c] += 1
        if (before['Qwen_called'] and before['prompt_sha256'] == after['prompt_sha256']
                and before['raw_output'] != after['raw_output']):
            same_prompt_different_output.append(key)
    metrics = list(csv.DictReader((D/'metrics.csv').open()))
    for row in metrics:
        subset = [x for x in new if x['condition'] == row['condition']
                  and (row['role'] == 'all' or x['role'] == row['role'])]
        if row['scope'] == 'ordinary_positive':
            subset = [x for x in subset if x['positive'] and not x['fault']]
        elif row['scope'] == 'ordinary_no_match':
            subset = [x for x in subset if x['ordinary_no_match']]
        elif row['scope'] != 'all':
            # Other scopes are task-kind or fault groups, governed by the existing summary.
            continue
        metric = row['metric']; assert metric in subset[0]
        numerator = sum(bool(x[metric]) for x in subset)
        assert numerator == int(row['numerator']) and len(subset) == int(row['denominator']), row
        assert abs(numerator/len(subset)-float(row['value'])) < 1e-12
    run = json.loads((D/'run_report.json').read_text())
    assert sum(x['Qwen_called'] for x in new) == run['actual_new_GPU_requests'] == 1846
    assert sha(newfile) == run['raw_sha256']
    summary = Counter(x['condition'] for x in new); assert set(summary.values()) == {1508}
    for condition in summary:
        subset = [x for x in new if x['condition']==condition]
        assert len({x['scene'] for x in subset})==38 and len({x['log'] for x in subset})==11
    out = dict(
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), records=9048,
        actual_GPU_calls=1846, frozen_task_fields_equal_to_old_run=True,
        candidate_coverage_and_call_routes_unchanged=True, changes_only_in_relation_prompt_or_model_decision=True,
        changed_prompts_by_condition=dict(changed_prompts), changed_raw_output_by_condition=dict(changed_raw),
        changed_final_proposal_by_condition=dict(changed_decisions),
        same_prompt_different_actual_model_output=same_prompt_different_output,
        common_metric_numerators_and_denominators_recomputed=True,
        source_sha256={str(x):sha(x) for x in [oldfile,newfile,D/'metrics.csv',Path(__file__)]},
        limitations=['只核验保存的任务字段和结果，不是另建官方GT或人工复核',
                     '开发图输入与旧版仅near关系边变化；不能把结果当最终封存或物理到达'],
        goal_complete=False,
    )
    with (D/'raw_readback_audit_001.json').open('x') as stream:
        json.dump(out, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in out.items() if k not in ['source_sha256','limitations']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
