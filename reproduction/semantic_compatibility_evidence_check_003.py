"""完整主矩阵的独立分母、原始证据、CSV及配置聚类统计复核。"""
from collections import Counter
import csv
from datetime import datetime, timezone
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np


PROJECT = Path('/root/autodl-tmp/lizichong')
OUT = PROJECT/'outputs/project_v2/thesis_completion_v1'
FIELDS = ('route_generated', 'selected_object_evaluated_correct',
          'stable_correct_lock', 'actual_full_perception_arrival')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(1048576):
            result.update(block)
    return result.hexdigest()


def reject(value):
    raise ValueError('非有限JSON常量: '+value)


def pairs(values):
    result = {}
    for key, value in values:
        assert key not in result, '重复JSON字段: '+key
        result[key] = value
    return result


def parse(value):
    return json.loads(value, parse_constant=reject, object_pairs_hook=pairs)


def read(path):
    return parse(Path(path).read_text())


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def interval(rows, numerator):
    """逐配置重采样、逐样本求和；不调用生产统计函数。"""
    grouped = {}
    for row in rows:
        group = grouped.setdefault(row['config_id'], [0., 0])
        group[0] += numerator(row)
        group[1] += 1
    if not grouped:
        return dict(estimate=None, ci95=None, clusters=0, denominator=0)
    groups = list(grouped.values())
    total = sum(value[1] for value in groups)
    sample_indices = np.random.default_rng(17).integers(0, len(groups), (4000, len(groups)))
    draws = []
    for indices in sample_indices:
        n = sum(groups[int(index)][0] for index in indices)
        d = sum(groups[int(index)][1] for index in indices)
        draws.append(n/d)
    return dict(estimate=sum(value[0] for value in groups)/total,
                ci95=np.percentile(draws, [2.5, 97.5]).tolist(),
                clusters=len(groups), denominator=total,
                numerator=sum(value[0] for value in groups))


def same_interval(actual, expected):
    assert set(actual) == set(expected)
    for field in actual:
        if actual[field] is None:
            assert expected[field] is None
        else:
            assert np.allclose(actual[field], expected[field], rtol=0, atol=1e-12), (field, actual, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-run', default='semantic_ablation_combined_003')
    parser.add_argument('--run-id', default='semantic_compatibility_evidence_check_003')
    parser.add_argument('--expected', type=int, default=750)
    parser.add_argument('--methods', nargs='+', default=['rule', 'new_full', 'new_minus_parse', 'new_minus_attributes', 'new_minus_rerank'])
    args = parser.parse_args()
    started = time.monotonic()
    source = OUT/'interactive_simulation'/args.input_run
    stats = OUT/'statistics'/args.input_run
    # 必须先有完整主矩阵及完整统计，不将等待或部分记录称为审核完成。
    summary, report = read(source/'summary.json'), read(stats/'report.json')
    release = read(source/'release.json')
    assert summary['attempted'] == summary['expected'] == report['trials'] == args.expected
    assert sha(source/'trial_records.jsonl') == summary['trial_records_sha256'] == report['source_records_sha256']
    assert sha(source/'release.json') == report['source_release_sha256']
    assert sha(source/'summary.json') == report['source_summary_sha256']
    assert sha(stats/'audited_trials.jsonl') == report['audited_trials_sha256']
    taskpath = OUT/'interactive_simulation/synthetic_tasks_v1.json'
    task_release = read(taskpath)
    assert sha(taskpath) == release['task_release_sha256']
    tasks = {task['task_id']: task for task in task_release['tasks']}
    expected = {(method, task, seed) for method in args.methods
                for task in tasks for seed in task_release['seeds']}
    assert len(expected) == args.expected and len(tasks) == 50
    def key(row):
        return row['variant'], row['task'], row['seed']
    plans = {key(row): row for row in release['trials']}
    raw = [parse(line) for line in (source/'trial_records.jsonl').open()]
    assert len(raw) == len({key(row) for row in raw}) == len(plans) == args.expected
    assert set(plans) == {key(row) for row in raw} == expected
    audited = [parse(line) for line in (stats/'audited_trials.jsonl').open()]
    assert len(audited) == args.expected
    files, counts, rows = {}, Counter(), []
    for path, value in release['source_sha256'].items():
        assert sha(PROJECT/path) == value
        files[str(PROJECT/path)] = value
    destination = OUT/'statistics'/args.run_id
    destination.mkdir(exist_ok=False)
    for record, derived in zip(raw, audited):
        assert time.monotonic()-started < 900
        assert all(record[field] == value for field, value in plans[key(record)].items())
        task = tasks[record['task']]
        assert (record['kind'], record['config_id']) == (task['kind'], task['config_id'])
        root = Path(record['raw_case'])
        result = record['result']
        assert derived['method'] == record['variant'] and derived['task_id'] == record['task']
        assert derived['config_id'] == record['config_id'] and derived['seed'] == record['seed']
        assert derived['status'] == record['status'] and derived['result_missing'] == (result is None)
        assert record['actual_arrival'] == bool(result and result[FIELDS[-1]])
        row = {field: record[field] for field in ['config_id', 'seed', 'kind', 'status']}
        row.update(method=record['variant'], task_id=record['task'], any_lock=False,
                   actual_path_length_m=0., actual_state_records=0,
                   independent_execution_audit=dict(passed=False))
        row.update({field: None if result is None else result[field] for field in FIELDS})
        if result is None:
            assert record['result_sha256'] is None and not (root/'result.json').exists()
            counts['missing_complete_results'] += 1
        else:
            assert read(root/'result.json') == result and sha(root/'result.json') == record['result_sha256']
            assert all(isinstance(result[field], bool) for field in FIELDS)
            assert result['task_id'] == task['task_id'] and result['seed'] == record['seed']
            assert not result['gt_in_agent'] and result['real_clock_async']
            assert result[FIELDS[-1]] == (result['status'] == 'ARRIVED')
            assert read(root/'independent_execution_audit.json') == result['independent_execution_audit']
            row['independent_execution_audit'] = result['independent_execution_audit']
            if result[FIELDS[-1]]:
                assert row['independent_execution_audit']['passed'] and task['kind'] == 'reachable_positive'
            files[str(root/'result.json')] = record['result_sha256']
        for field in FIELDS:
            assert row[field] == derived[field]
        for filename in ['execution.jsonl', 'perception.jsonl', 'semantic_events.json', 'independent_execution_audit.json']:
            path = root/filename
            if not path.exists():
                assert result is None
                continue
            value = sha(path)
            assert derived['raw_sha256'][str(path)] == value
            files[str(path)] = value
            if path.suffix != '.jsonl':
                continue
            entries = []
            lines = path.read_text().splitlines()
            for index, line in enumerate(lines):
                try:
                    entries.append(parse(line))
                except json.JSONDecodeError:
                    assert result is None and index == len(lines)-1
                    counts['incomplete_failed_process_tail_records'] += 1
            if filename == 'execution.jsonl':
                states = [entry['state'] for entry in entries]
                row['actual_path_length_m'] = sum(math.dist([a['x'], a['y']], [b['x'], b['y']])
                                                  for a, b in zip(states, states[1:]))
                row['actual_state_records'] = len(states)
                counts['saved_control_records'] += len(states)
            else:
                row['any_lock'] = any(entry.get('selected_node_id') is not None for entry in entries)
                assert all(not entry['gt_sent_to_agent'] for entry in entries)
                counts['saved_perception_records'] += len(entries)
        assert row['actual_state_records'] == derived['actual_state_records']
        assert abs(row['actual_path_length_m']-derived['actual_path_length_m']) < 1e-9
        assert row['any_lock'] == derived['any_lock']
        assert derived['process_returncode'] == record['returncode']
        assert derived['process_timeout'] == record['process_timeout']
        rows.append(row)
    with (stats/'trials.csv').open(newline='') as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == args.expected
    for row, record in zip(table, audited):
        for field, value in row.items():
            expected_value = record.get(field)
            assert value == ('' if expected_value is None else str(expected_value)), (field, value, expected_value)
    statistics_checked = 0
    for method in sorted({row['method'] for row in rows}):
        subset = [row for row in rows if row['method'] == method]
        actual = report['methods'][method]
        arrived = [row for row in subset if row[FIELDS[-1]]]
        assert actual['functional_evidence'] == dict(arrivals=len(arrived),
            distinct_positive_tasks=len({row['task_id'] for row in arrived}),
            scene_configs=len({row['config_id'] for row in arrived}),
            minimum_10_tasks_5_configs=len({row['task_id'] for row in arrived}) >= 10
            and len({row['config_id'] for row in arrived}) >= 5)
        for kind in sorted({row['kind'] for row in subset}):
            group = [row for row in subset if row['kind'] == kind]
            summary = actual['by_kind'][kind]
            assert summary['trials'] == len(group)
            assert summary['unique_tasks'] == len({row['task_id'] for row in group})
            assert summary['configs'] == len({row['config_id'] for row in group})
            assert summary['statuses'] == dict(Counter(row['status'] for row in group))
            for field in FIELDS:
                same_interval(summary['four_outcomes'][field], interval(group, lambda row: bool(row[field])))
                statistics_checked += 1
            same_interval(summary['any_lock'], interval(group, lambda row: row['any_lock']))
            moving = [row for row in group if row['actual_path_length_m'] > .01]
            same_interval(summary['safety_among_moving'], interval(moving, lambda row: row['independent_execution_audit']['passed']))
            assert summary['actually_moving_trials'] == len(moving)
            assert summary['stationary_audited_trials'] == len(group)-len(moving)
            assert summary['all_audits_passed'] == sum(row['independent_execution_audit']['passed'] for row in group)
            for seed in task_release['seeds']:
                members = [row for row in group if row['seed'] == seed]
                assert summary['seeds'][str(seed)] == dict(trials=len(members),
                    four_outcomes={field: sum(bool(row[field]) for row in members) for field in FIELDS})
            statistics_checked += 2
    lookup = {(row['method'], row['task_id'], row['seed']): row for row in rows}
    differences = {(row['a'], row['b'], row['metric']): row for row in report['paired_positive_differences']}
    assert len(differences) == len(args.methods)*(len(args.methods)-1)//2*len(FIELDS)
    for a, b in itertools.combinations(sorted({row['method'] for row in rows}), 2):
        for field in FIELDS:
            paired = [dict(config_id=row['config_id'], difference=int(bool(row[field]))
                           -int(bool(lookup[b, row['task_id'], row['seed']][field])))
                      for row in rows if row['method'] == a and row['kind'] == 'reachable_positive']
            actual = differences[a, b, field]
            same_interval({key: value for key, value in actual.items() if key not in ['a', 'b', 'metric']},
                          interval(paired, lambda row: row['difference']))
            statistics_checked += 1
    for path in [source/'release.json', source/'summary.json', source/'trial_records.jsonl',
                 taskpath, stats/'report.json', stats/'audited_trials.jsonl', stats/'trials.csv']:
        files[str(path)] = sha(path)
    save(destination/'files.json', files)
    save(destination/'report.json', dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
         trials=args.expected, method_count=len(args.methods), tasks=50, seeds=task_release['seeds'], configs=10,
         counts=dict(counts), independently_recomputed_cluster_statistics=statistics_checked,
         SHA_files=len(files), files_manifest_sha256=sha(destination/'files.json'), source_sha256=sha(__file__),
         methods={method: report['methods'][method]['functional_evidence'] for method in report['methods']},
         wall_CPU_shared_s=time.monotonic()-started, actual_model_calls=0, new_executions=0, new_arrivals=0,
         limits=['输入、分母和统计复核，不是第三套运动学/车身或身份真值审核',
                 '完整失败保留，未知字段不给成功信用；配置和种子不增加独立真实场景数',
                 '开发合成任务，不是新自然语言封存测试，未增加正确到达'], goal_complete=False))
    print(json.dumps(read(destination/'report.json'), ensure_ascii=False))


if __name__ == '__main__':
    main()
