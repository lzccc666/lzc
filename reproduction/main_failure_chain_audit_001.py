"""从原450日志重新核对阶段事件、响应/训练分母及配置区间；不调用诊断生成函数。"""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT = Path('/root/autodl-tmp/lizichong')
OUT = PROJECT/'outputs/project_v2/thesis_completion_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).open()]


def main():
    started = time.monotonic()
    source = OUT/'statistics/main_failure_chain_full_002'
    report = read(source/'report.json')
    cover_path = OUT/'statistics/main_failure_chain_coverage_001/report.json'
    coverage = read(cover_path)
    trials = lines(source/'trials.jsonl')
    raw_rows = lines(OUT/'interactive_simulation/realtime_three_model_full_001/trial_records.jsonl')
    assert len(trials) == len(raw_rows) == len({row['run'] for row in trials}) == report['trials'] == 450
    assert coverage['source_report_sha256'] == sha(source/'report.json')
    files = read(source/'files.json')
    assert all(sha(name) == expected for name, expected in files.items())
    for name in ['trials', 'frames', 'responses']:
        assert sha(source/(name+'.jsonl')) == report[name+'_sha256']
        small = OUT/'statistics/main_failure_chain_smoke_002'/(name+'.jsonl')
        prefix = small.read_bytes()
        with (source/(name+'.jsonl')).open('rb') as handle:
            assert handle.read(len(prefix)) == prefix
    task_release = read(OUT/'interactive_simulation/synthetic_tasks_v1.json')
    tasks = {row['task_id']: row for row in task_release['tasks']}
    configs = {row['config_id']: row for row in task_release['configs']}
    frames_checked, saved_mapping_omissions, valid_selected_mapping_frames = 0, 0, 0
    with (source/'frames.jsonl').open() as handle:
        for trial, raw in zip(trials, raw_rows):
            assert time.monotonic()-started < 900
            assert trial['run'] == raw['run_id']
            root = Path(raw['raw_case'])
            targets = configs[raw['config_id']]['targets']
            accepted = set(tasks[raw['task']]['accepted_instance_ids'])
            observed = Counter()
            streak = 0
            omitted = 0
            for line in (root/'perception.jsonl').open():
                frame = json.loads(line)
                derived = json.loads(next(handle))
                assert derived['run'] == raw['run_id'] and derived['frame'] == frame['frame_idx']
                nodes = sorted(frame['nodes'], key=lambda node: node['node_id'])
                pairs = {}
                if nodes:
                    distances = np.linalg.norm(np.asarray([node['center'][:2] for node in nodes])[:, None, :]
                                              -np.asarray([target['center_xy'] for target in targets])[None, :, :], axis=2)
                    for i, node in enumerate(nodes):
                        for j, target in enumerate(targets):
                            if node['category'] != target['category']:
                                distances[i, j] = 1e6
                    a, b = linear_sum_assignment(distances)
                    pairs = {str(nodes[int(i)]['node_id']): targets[int(j)]['instance_id']
                             for i, j in zip(a, b) if distances[i, j] <= 2.5}
                if frame['selected_node_id'] is None:
                    assert frame['evaluation_only_matches'] == {} and frame['evaluation_only_identity'] is None
                    omitted += bool(pairs)
                else:
                    assert pairs == frame['evaluation_only_matches']
                    valid_selected_mapping_frames += 1
                correct_nodes = {int(node) for node, instance in pairs.items() if instance in accepted}
                audit = frame['semantic_audit']
                final = [candidate['node_id'] for candidate in frame['candidates']]
                selected_good = frame['evaluation_only_identity'] in accepted
                streak = streak+1 if selected_good else 0
                expected = dict(mapped_accepted=bool(correct_nodes),
                    pre_safety_accepted=bool(set(audit['rule_order']) & correct_nodes),
                    safety_accepted=bool(set(audit['safety_verified_order']) & correct_nodes),
                    final_top_accepted=bool(final and final[0] in correct_nodes),
                    selected_accepted=selected_good, three_selected_accepted=streak >= 3,
                    correct_selected_target_route=bool(selected_good and frame['planning'] and frame['path_world']
                                                       and frame['path_purpose'] == 'target_approach'))
                for key, value in expected.items():
                    assert derived[key] is value
                    observed[key] += value
                assert derived['changed_order'] == audit['changed_order']
                assert derived['changed_final_first'] == audit['changed_final_first']
                assert not frame['gt_sent_to_agent']
                frames_checked += 1
            assert trial['frames'] == sum(1 for _ in (root/'perception.jsonl').open())
            assert all(trial[key] == bool(value) for key, value in observed.items())
            assert trial['saved_mapping_omitted_without_selected'] == omitted
            saved_mapping_omissions += omitted
        assert not handle.read()
    assert frames_checked == report['frames'] == 57460
    assert saved_mapping_omissions == report['saved_mapping_omitted_without_selected_frames'] == 30538
    with (source/'trials.csv').open() as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 450
    for csv_row, trial in zip(table, trials):
        for key, value in csv_row.items():
            assert value == ('' if trial[key] is None else str(trial[key]))
    responses = lines(source/'responses.jsonl')
    response_count = 0
    for row in responses:
        raw = read(row['source'])
        assert sha(row['source']) == row['source_sha256']
        assert row['request_id'] == raw['request']['request_id'] and row['task'] == raw['request']['task']
        assert row['accepted'] == raw['accepted_at_receipt']
        assert row['cache_hit'] == raw['envelope']['cache']['hit']
        if row['task'] == 'query_parse':
            record = next(trial for trial in trials if trial['run'] == row['run'])
            task = tasks[record['task_id']]
            target = configs[record['config_id']]['targets'][task['target_index']] if task['target_index'] is not None else None
            out = raw['envelope']['normalized_output']
            colors = [target['color']] if target else []
            category = 'car' if target else 'bus'
            flags = dict(category_exact=sorted(out['target_categories']) == [category],
                colors_exact=sorted(out['target_colors']) == sorted(colors),
                no_extra_anchor_relation=not out['relation'] and not out['anchor_text'] and not out['anchor_categories'],
                self_anchor_near=bool(out['anchor_text'] and out['target_text'] == out['anchor_text'] and out['relation'] == 'near'),
                expected_color_missing=bool(colors and not set(colors).issubset(out['target_colors'])))
            flags['category_color_relation_exact'] = flags['category_exact'] and flags['colors_exact'] and flags['no_extra_anchor_relation']
            assert flags == row['flags']
        response_count += 1
    statistics_checked = 0
    lookup = {(row['method'], row['task_id'], row['seed']): row for row in trials}
    for contrast in report['paired_config_differences']:
        key = contrast['metric'].removeprefix('ever_')
        selected = [row for row in trials if row['method'] == contrast['a'] and row['kind'] == 'reachable_positive']
        by_config = {}
        for row in selected:
            pair = lookup[contrast['b'], row['task_id'], row['seed']]
            by_config.setdefault(row['config_id'], []).append(int(bool(row[key]))-int(bool(pair[key])))
        totals = np.asarray([[sum(values), len(values)] for values in by_config.values()], dtype=float)
        ids = np.random.Generator(np.random.PCG64(17)).integers(len(totals), size=(4000, len(totals)))
        sums = np.sum(totals[ids], axis=1)
        interval = np.quantile(sums[:, 0]/sums[:, 1], [.025, .975])
        assert np.allclose(interval, contrast['ci95'], rtol=0, atol=1e-12)
        assert abs(totals[:, 0].sum()/totals[:, 1].sum()-contrast['value']) < 1e-12
        assert contrast['denominator'] == 90 and contrast['configs'] == 10
        statistics_checked += 1
    assert statistics_checked == 27
    for name, group in coverage['groups'].items():
        method, kind = name.split('/')
        subset = [row for row in trials if row['method'] == method and row['kind'] == kind]
        assert len(subset) == group['trials']
        assert sum(row['applied_counts'] is None for row in subset) == group['trials_without_complete_semantic_event_log']
        for task, counts in group['tasks'].items():
            selected = [row for row in responses if row['task'] == task and row['run'] in {trial['run'] for trial in subset}]
            assert counts['response_counts'].get('delivered_saved_responses', 0) == len(selected)
            assert counts['known_complete_event_applications'] == sum((row['applied_counts'] or {}).get(task, 0) for row in subset)
    for split, values in report['training_parse_coverage'].items():
        data = lines(OUT/'semantic_dataset/structured_supervision_002'/(split+'.jsonl'))
        selected = [row for row in data if row['task'] == 'query_parse']
        assert len(data) == values['total_rows'] and len(selected) == values['parse_rows']
        assert dict(Counter(str(row['response'].get('relation')) for row in selected)) == values['relations']
        assert sum(not row['response'].get('anchor_text') and not row['response'].get('relation')
                   and bool(row['response'].get('target_colors')) for row in selected) == values['no_anchor_no_relation_with_colors']
    dest = OUT/'statistics/main_failure_chain_audit_001'
    dest.mkdir(exist_ok=False)
    result = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(), trials=450,
        frames=frames_checked, reconstructed_empty_saved_mapping_frames=saved_mapping_omissions,
        selected_mapping_frames_matched_original=valid_selected_mapping_frames,
        original_source_files_SHA_checked=len(files), response_records_checked=response_count,
        smoke_full_prefixes_equal=['trials.jsonl', 'frames.jsonl', 'responses.jsonl'],
        CSV_rows_checked=450, paired_config_intervals_checked=statistics_checked,
        coverage_groups_checked=len(coverage['groups']), train_parse_rows_checked=1400, validation_parse_rows_checked=180,
        source_report_sha256=sha(source/'report.json'), coverage_report_sha256=sha(cover_path),
        audit_source_sha256=sha(__file__), actual_model_calls=0, new_executions=0, new_arrivals=0,
        CPU_shared_wall_s=time.monotonic()-started,
        limits=['全部节点匹配仍使用相同类别/2.5m阈值和SciPy后端，不是第三种实例真值或人工审核',
                '逐帧阶段和响应分母独立重算，不新增模块因果贡献或正确到达信用',
                '不改原保存的空映射、失败响应与原主450结果'], goal_complete=False)
    with (dest/'report.json').open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
