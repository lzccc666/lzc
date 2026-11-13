"""主450后验失败环节和已投递语义响应诊断；不改变或运行实际方法。"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT = Path('/root/autodl-tmp/lizichong')
OUT = PROJECT/'outputs/project_v2/thesis_completion_v1'
TASKS = ['query_parse', 'object_attributes', 'candidate_rerank']
FIELDS = ['route_generated', 'selected_object_evaluated_correct', 'stable_correct_lock', 'actual_full_perception_arrival']
STAGES = ['mapped_accepted', 'pre_safety_accepted', 'safety_accepted', 'final_top_accepted',
          'selected_accepted', 'three_selected_accepted', 'correct_selected_target_route']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_lines(path, failed):
    lines = Path(path).read_text().splitlines()
    values = []
    for index, line in enumerate(lines):
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            assert failed and index == len(lines)-1
    return values


def parse_flags(value, expected_category, expected_colors):
    category = set(value.get('target_categories', [])) == {expected_category}
    color = set(value.get('target_colors', [])) == set(expected_colors)
    no_anchor_relation = not value.get('relation') and not value.get('anchor_text') and not value.get('anchor_categories')
    # 保存帧的ParsedQuery没有anchor_categories，类别与颜色以及显式锚点仍可判定。
    return dict(category_exact=category, colors_exact=color, no_extra_anchor_relation=no_anchor_relation,
                category_color_relation_exact=category and color and no_anchor_relation,
                self_anchor_near=bool(value.get('anchor_text') and value.get('anchor_text') == value.get('target_text')
                                      and value.get('relation') == 'near'),
                expected_color_missing=bool(expected_colors and not set(expected_colors) <= set(value.get('target_colors', []))))


def dataset_coverage(path, task_texts):
    result = {}
    for split in ['train', 'validation']:
        rows = [json.loads(line) for line in (path/f'{split}.jsonl').open()]
        selected = [row for row in rows if row['task'] == 'query_parse']
        result[split] = dict(total_rows=len(rows), parse_rows=len(selected),
            relations=dict(Counter(str(row['response'].get('relation')) for row in selected)),
            expression_families=dict(Counter(row['expression_family'] for row in selected)),
            supervision_sources=dict(Counter(row['supervision_source'] for row in selected)),
            no_anchor_no_relation=sum(not row['response'].get('anchor_text') and not row['response'].get('relation') for row in selected),
            no_anchor_no_relation_with_colors=sum(not row['response'].get('anchor_text') and not row['response'].get('relation')
                                                  and bool(row['response'].get('target_colors')) for row in selected),
            self_anchor_near=sum(bool(row['response'].get('anchor_text'))
                and row['response']['anchor_text'] == row['response'].get('target_text')
                and row['response'].get('relation') == 'near' for row in selected),
            exact_query_text_overlap=sum(row['query_text'] in task_texts for row in selected),
            excluded_categories_or_colors=sum(bool(row['response'].get('excluded_categories') or row['response'].get('excluded_colors')) for row in selected))
    return result


def interval(rows, key):
    groups = {}
    for row in rows:
        group = groups.setdefault(row['config_id'], [0., 0])
        group[0] += row[key]
        group[1] += 1
    values = np.asarray(list(groups.values()), dtype=float)
    indices = np.random.default_rng(17).integers(0, len(values), (4000, len(values)))
    sampled = values[indices].sum(axis=1)
    return dict(value=float(values[:, 0].sum()/values[:, 1].sum()),
        ci95=np.percentile(sampled[:, 0]/sampled[:, 1], [2.5, 97.5]).tolist(),
        denominator=int(values[:, 1].sum()), configs=len(values))


def trial(record, task, config, frame_handle, response_handle, files, deadline):
    root, result = Path(record['raw_case']), record['result']
    accepted = set(task['accepted_instance_ids'])
    expected_color = [config['targets'][task['target_index']]['color']] if task['target_index'] is not None else []
    category = 'bus' if task['kind'] == 'no_match' else 'car'
    output = dict(run=record['run_id'], method=record['variant'], task_id=record['task'],
        config_id=record['config_id'], kind=record['kind'], seed=record['seed'], status=record['status'],
        result_missing=result is None, frames=0, source_route=False, after_parse_frames=0,
        after_parse_color_missing_frames=0, after_parse_self_anchor_frames=0,
        color_removed_transitions=0, changed_order_frames=0, changed_first_frames=0,
        saved_mapping_omitted_without_selected=0)
    output.update({key: False for key in STAGES})
    events_path = root/'semantic_events.json'
    events = read(events_path) if events_path.exists() else None
    if events:
        files[str(events_path)] = sha(events_path)
    if result is not None:
        assert read(root/'result.json') == result and sha(root/'result.json') == record['result_sha256']
        files[str(root/'result.json')] = record['result_sha256']
    frames = load_lines(root/'perception.jsonl', result is None)
    files[str(root/'perception.jsonl')] = sha(root/'perception.jsonl')
    streak, app_frames, first_stage = 0, {}, {}
    previous_colors = None
    all_node_matches_by_frame = {}
    for frame in frames:
        assert time.monotonic() < deadline
        assert frame['gt_sent_to_agent'] is False
        audit = frame['semantic_audit']
        assert audit['gt_in_semantic_input'] is False
        nodes = sorted(frame['nodes'], key=lambda node: node['node_id'])
        matches = {}
        if nodes:
            costs = np.asarray([[np.linalg.norm(np.asarray(node['center'][:2])-target['center_xy'])
                if node['category'] == target['category'] else 1e6 for target in config['targets']] for node in nodes])
            row_indices, column_indices = linear_sum_assignment(costs)
            matches = {str(nodes[i]['node_id']): config['targets'][j]['instance_id']
                       for i, j in zip(row_indices, column_indices) if costs[i, j] <= 2.5}
        if frame['selected_node_id'] is not None:
            assert matches == frame['evaluation_only_matches']
            assert matches.get(str(frame['selected_node_id'])) == frame['evaluation_only_identity']
        else:
            assert frame['evaluation_only_matches'] == {} and frame['evaluation_only_identity'] is None
            output['saved_mapping_omitted_without_selected'] += bool(matches)
        all_node_matches_by_frame[frame['frame_idx']] = matches
        def good(node):
            return matches.get(str(node)) in accepted
        pre, safe, final = audit['rule_order'], audit['safety_verified_order'], audit['final_order']
        assert set(final) <= set(safe) <= set(pre)
        assert final == [row['node_id'] for row in frame['candidates']]
        selected = frame['evaluation_only_identity'] in accepted
        streak = streak+1 if selected else 0
        route = bool(frame['planning'] and frame['path_world'] and frame['path_purpose'] == 'target_approach')
        flags = dict(mapped_accepted=bool(set(matches.values()) & accepted),
            pre_safety_accepted=any(good(node) for node in pre), safety_accepted=any(good(node) for node in safe),
            final_top_accepted=bool(final and good(final[0])), selected_accepted=selected,
            three_selected_accepted=streak >= 3, correct_selected_target_route=selected and route)
        for key, value in flags.items():
            output[key] |= value
            if value and key not in first_stage:
                first_stage[key] = dict(frame=frame['frame_idx'], sim_time_s=frame['sim_time_s'],
                                        completed_wall_s=frame['completed_wall_elapsed_s'])
        output['source_route'] |= route
        parsed = parse_flags(frame['parsed_query'], category, expected_color)
        after = audit['applied_counts']['query_parse'] > 0
        output['after_parse_frames'] += after
        output['after_parse_color_missing_frames'] += after and parsed['expected_color_missing']
        output['after_parse_self_anchor_frames'] += after and parsed['self_anchor_near']
        colors = set(frame['parsed_query']['target_colors'])
        if previous_colors is not None and expected_color and set(expected_color) <= previous_colors and not set(expected_color) <= colors:
            output['color_removed_transitions'] += 1
        previous_colors = colors
        output['changed_order_frames'] += audit['changed_order']
        output['changed_first_frames'] += audit['changed_final_first']
        for event in audit['events']:
            if event['kind'] == 'response_received':
                assert event['request_id'] not in app_frames
                app_frames[event['request_id']] = frame
        frame_handle.write(json.dumps(dict(run=record['run_id'], frame=frame['frame_idx'],
            **flags, after_query_parse=after, parse_flags=parsed, changed_order=audit['changed_order'],
            changed_final_first=audit['changed_final_first']), ensure_ascii=False, allow_nan=False)+'\n')
    output['frames'] = len(frames)
    output['first_stage_evidence'] = first_stage
    output['applied_counts'] = events['applied_counts'] if events else None
    if result is not None:
        assert output['source_route'] == result[FIELDS[0]]
        assert output['final_top_accepted'] == result[FIELDS[1]]
        assert output['three_selected_accepted'] == result[FIELDS[2]]
        assert output['applied_counts'] == result['semantic_applied_by_task']
    output['actual_arrival'] = result[FIELDS[-1]] if result is not None else None
    output['four_outcomes_complete_only'] = {field: result[field] if result else None for field in FIELDS}
    if result is None:
        failure = 'unknown_missing_complete_result'
    elif result[FIELDS[-1]]:
        failure = 'arrived'
    else:
        failure = next((key for key in STAGES if not output[key]), 'all_saved_stage_events_but_no_arrival')
    output['ordered_stage_gap'] = failure
    task_counts = {task_name: Counter() for task_name in TASKS}
    requests = {}
    if events:
        for event in events['events']:
            identity, task_name, kind = event['request_id'], event['task'], event['kind']
            old = requests.setdefault(identity, dict(task=task_name, payload_sha256=event['payload_sha256']))
            assert old['task'] == task_name and old['payload_sha256'] == event['payload_sha256']
            task_counts[task_name][kind] += 1
            if kind == 'backend_completed':
                task_counts[task_name]['completed_'+event['response_status']] += 1
            if kind == 'discarded':
                task_counts[task_name]['discarded_'+event['reason']] += 1
        for event in events['application_events']:
            if event['kind'] == 'ranking_context_checked':
                task_counts['candidate_rerank']['context_applied' if event['applied'] else 'context_rejected'] += 1
    response_counts = {task_name: Counter() for task_name in TASKS}
    for path in sorted((root/'semantic_responses').glob('*.json')):
        value = read(path)
        request, envelope = value['request'], value['envelope']
        identity, task_name = request['request_id'], request['task']
        assert request['payload_sha256'] == canonical(request['payload'])
        if events:
            assert requests[identity]['payload_sha256'] == request['payload_sha256']
        files[str(path)] = sha(path)
        counts = response_counts[task_name]
        counts['saved_delivered_response'] += 1
        for key in ['output_schema_valid', 'used_qwen', 'fallback_used']:
            counts[key] += bool(envelope[key])
        cache_hit = envelope.get('cache', {}).get('hit')
        counts['known_cache_hit'] += cache_hit is True
        counts['known_noncache_response'] += cache_hit is False
        counts['schema_and_confidence_accepted'] += value['accepted_at_receipt']
        saved = dict(run=record['run_id'], method=record['variant'], task=task_name, request_id=identity,
            source=str(path), source_sha256=files[str(path)], accepted=value['accepted_at_receipt'],
            cache_hit=cache_hit, flags=None, first_receipt_frame=app_frames.get(identity, {}).get('frame_idx'))
        normalized = envelope.get('normalized_output')
        if task_name == 'query_parse' and normalized is not None:
            flags = parse_flags(normalized, category, expected_color)
            counts.update({'parse_'+key: int(flag) for key, flag in flags.items()})
            saved['flags'] = flags
            saved['normalized_output'] = normalized
        elif task_name == 'object_attributes' and normalized is not None:
            colors = set(normalized.get('colors', []))-{'unknown'}
            counts['attribute_all_colors_unknown'] += not colors
            frame = app_frames.get(identity)
            node = request['payload']['node_id']
            instance = all_node_matches_by_frame[frame['frame_idx']].get(str(node)) if frame else None
            truth = next((target['color'] for target in config['targets'] if target['instance_id'] == instance), None)
            counts['attribute_known_color_and_receipt_associated'] += bool(colors and truth)
            counts['attribute_exact_color_at_receipt'] += bool(truth and colors == {truth})
            saved['attribute_colors'] = sorted(colors)
            saved['receipt_evaluation_instance'] = instance
            saved['receipt_synthetic_color'] = truth
        response_handle.write(json.dumps(saved, ensure_ascii=False, allow_nan=False)+'\n')
    if events:
        for task_name in TASKS:
            assert response_counts[task_name]['saved_delivered_response'] == task_counts[task_name]['delivered']
            assert response_counts[task_name]['schema_and_confidence_accepted'] == sum(
                event['kind'] == 'response_received' and event['task'] == task_name and event['schema_and_confidence_passed']
                for event in events['application_events'])
    output['queue_event_counts'] = {key: dict(value) for key, value in task_counts.items()}
    output['response_counts'] = {key: dict(value) for key, value in response_counts.items()}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started+900
    assert shutil.disk_usage(OUT).free >= 6442450944
    dest = OUT/'statistics'/args.run_id
    dest.mkdir(exist_ok=False)
    shutil.copy2(__file__, dest/'generator_source.py')
    source = OUT/'interactive_simulation/realtime_three_model_full_001'
    release, summary = read(source/'release.json'), read(source/'summary.json')
    assert summary['attempted'] == summary['expected'] == 450
    assert sha(source/'trial_records.jsonl') == summary['trial_records_sha256']
    assert all(sha(PROJECT/name) == value for name, value in release['source_sha256'].items())
    tasks_path = OUT/'interactive_simulation/synthetic_tasks_v1.json'
    assert sha(tasks_path) == release['task_release_sha256']
    task_release = read(tasks_path)
    tasks = {task['task_id']: task for task in task_release['tasks']}
    configs = {config['config_id']: config for config in task_release['configs']}
    rows = [json.loads(line) for line in (source/'trial_records.jsonl').open()]
    assert len(rows) == 450
    if args.limit:
        assert args.limit == 6
        rows = rows[:args.limit]
    dataset = OUT/'semantic_dataset/structured_supervision_002'
    dataset_manifest = read(dataset/'manifest.json')
    assert all(sha(dataset/name) == value for name, value in dataset_manifest['files'].items())
    files = {str(path): sha(path) for path in [source/'release.json', source/'summary.json', source/'trial_records.jsonl',
        tasks_path, dataset/'manifest.json', dataset/'train.jsonl', dataset/'validation.jsonl']}
    save(dest/'release.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
        input_files=files, protocol_sha256=sha(OUT/'task_protocol/main_failure_chain_002.md'),
        selected_source_run=source.name, selected_limit=args.limit, selection='原冻结顺序前6或全部450，不按结果挑选',
        wall_budget_s=900, metadata_budget_bytes=134217728, minimum_free_bytes=6442450944,
        posthoc_development_diagnostic=True, actual_model_calls=0, new_executions=0, goal_complete=False))
    try:
        diagnostics = []
        with (dest/'frames.jsonl').open('x') as frame_handle, (dest/'responses.jsonl').open('x') as response_handle:
            for index, record in enumerate(rows):
                output = trial(record, tasks[record['task']], configs[record['config_id']], frame_handle, response_handle, files, deadline)
                diagnostics.append(output)
                if index % 30 == 0:
                    print(json.dumps(dict(processed=index+1, expected=len(rows))), flush=True)
        with (dest/'trials.jsonl').open('x') as handle:
            for value in diagnostics:
                handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
        fields = ['run', 'method', 'task_id', 'config_id', 'kind', 'seed', 'status', 'result_missing', 'frames',
                  *STAGES, 'actual_arrival', 'ordered_stage_gap', 'after_parse_frames', 'after_parse_color_missing_frames',
                  'after_parse_self_anchor_frames', 'color_removed_transitions', 'changed_order_frames', 'changed_first_frames', 'saved_mapping_omitted_without_selected']
        with (dest/'trials.csv').open('x') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in diagnostics)
        methods = {}
        for method in sorted({row['method'] for row in diagnostics}):
            subset = [row for row in diagnostics if row['method'] == method]
            counts = {}
            for kind in sorted({row['kind'] for row in subset}):
                group = [row for row in subset if row['kind'] == kind]
                counts[kind] = dict(trials=len(group), missing_complete_results=sum(row['result_missing'] for row in group),
                    saved_prefix_stage_events={key: sum(row[key] for row in group) for key in STAGES},
                    complete_only_four_outcomes={key: sum(bool(row['four_outcomes_complete_only'][key]) for row in group) for key in FIELDS},
                    ordered_stage_gaps=dict(Counter(row['ordered_stage_gap'] for row in group)),
                    parse_color_removed_trials=sum(row['color_removed_transitions'] > 0 for row in group),
                    parse_self_anchor_after_application_trials=sum(row['after_parse_self_anchor_frames'] > 0 for row in group),
                    semantic_changed_first_trials=sum(row['changed_first_frames'] > 0 for row in group))
            merged = {}
            for task_name in TASKS:
                queue, response = Counter(), Counter()
                for row in subset:
                    queue.update(row['queue_event_counts'][task_name])
                    response.update(row['response_counts'][task_name])
                merged[task_name] = dict(queue_events=dict(queue), delivered_response_metrics=dict(response),
                                        actual_applications=sum((row['applied_counts'] or {}).get(task_name, 0) for row in subset))
            methods[method] = dict(by_kind=counts, semantic_tasks=merged)
        differences = []
        if not args.limit:
            lookup = {(row['method'], row['task_id'], row['seed']): row for row in diagnostics}
            for a, b in itertools.combinations(sorted(methods), 2):
                for field in STAGES+['color_removed_transitions', 'after_parse_self_anchor_frames']:
                    paired = [dict(config_id=row['config_id'], difference=int(bool(row[field]))-int(bool(lookup[b, row['task_id'], row['seed']][field])))
                              for row in diagnostics if row['method'] == a and row['kind'] == 'reachable_positive']
                    differences.append(dict(a=a, b=b, metric='ever_'+field, **interval(paired, 'difference')))
        save(dest/'files.json', files)
        report = dict(created_utc=datetime.now(timezone.utc).isoformat(), trials=len(diagnostics),
            is_smoke=bool(args.limit), methods=methods, paired_config_differences=differences,
            training_parse_coverage=dataset_coverage(dataset, {task['query_text'] for task in tasks.values()}),
            frames=sum(row['frames'] for row in diagnostics),
            saved_mapping_omitted_without_selected_frames=sum(row['saved_mapping_omitted_without_selected'] for row in diagnostics),
            all_node_mapping_recomputed=True, reconstructed_mapping_is_not_new_saved_observation=True,
            all_complete_four_outcomes_match_original=True, source_files=len(files), files_manifest_sha256=sha(dest/'files.json'),
            trials_sha256=sha(dest/'trials.jsonl'), frames_sha256=sha(dest/'frames.jsonl'), responses_sha256=sha(dest/'responses.jsonl'),
            bootstrap=dict(configs=10, repetitions=4000, seed=17), actual_model_calls=0, new_executions=0, new_arrivals=0,
            wall_CPU_shared_s=time.monotonic()-started,
            limits=['后验合成开发诊断，阶段缺口是描述性顺序分类，不证明因果；不同时间的存在事件不是严格单调漏斗',
                    '安全前候选已经受到解析和属性影响，不能称纯规则候选；全部节点关联由保存节点重建，沿用类别/2.5m和原Hungarian后端，不称新增人工真值',
                    '旧无selected时保存的空mapping保留；重建不是新增观测，只用于本版本诊断',
                    '缺完整结果前缀不计完整业务成功，仍留在冻结分母',
                    '语义准确性只针对已投递保存响应；后端入口可能缓存，未投递原文未知',
                    '颜色/锚点诊断只针对冻结合成模板，非自然语言审核；待完整缺一消融解释模块贡献',
                    '训练/验证表达分布只是相关证据，不据此断言单一失败根因；共享时钟非独占GPU性能'],
            manuscript_text_written=False, independent_test=False, new_test_opened=False, goal_complete=False)
        save(dest/'report.json', report)
        assert sum(path.stat().st_size for path in dest.rglob('*') if path.is_file()) < 134217728
        print(json.dumps(dict(trials=len(diagnostics), frames=report['frames'], source_files=len(files),
            same_complete_four_outcomes=True, goal_complete=False)), flush=True)
    except BaseException as exc:
        save(dest/'interrupted.json', dict(error=repr(exc), actual_model_calls=0, goal_complete=False))
        raise


if __name__ == '__main__':
    main()
