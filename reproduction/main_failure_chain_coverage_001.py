"""为002逐条证据补充分母、任务类型和不完整事件覆盖；不覆盖原报告。"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PROJECT = Path('/root/autodl-tmp/lizichong')
OUT = PROJECT/'outputs/project_v2/thesis_completion_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    source = OUT/'statistics/main_failure_chain_full_002'
    report = read(source/'report.json')
    trials = [json.loads(line) for line in (source/'trials.jsonl').open()]
    responses = [json.loads(line) for line in (source/'responses.jsonl').open()]
    lookup = {row['run']: row for row in trials}
    assert len(lookup) == report['trials'] == 450
    for name in ['trials', 'responses', 'frames']:
        assert sha(source/(name+'.jsonl')) == report[name+'_sha256']
    dest = OUT/'statistics/main_failure_chain_coverage_001'
    dest.mkdir(exist_ok=False)
    groups = {}
    for method in ['rule', 'base_full', 'new_full']:
        for kind in ['reachable_positive', 'no_match', 'inaccessible']:
            subset = [row for row in trials if row['method'] == method and row['kind'] == kind]
            group = dict(trials=len(subset), complete_results=sum(not row['result_missing'] for row in subset),
                missing_results=sum(row['result_missing'] for row in subset),
                trials_with_complete_semantic_event_log=sum(row['applied_counts'] is not None for row in subset),
                trials_without_complete_semantic_event_log=sum(row['applied_counts'] is None for row in subset),
                saved_prefix_is_not_complete_success=True, tasks={})
            for task_name in ['query_parse', 'object_attributes', 'candidate_rerank']:
                selected = [row for row in responses if row['task'] == task_name and row['run'] in {x['run'] for x in subset}]
                counter, queue = Counter(), Counter()
                for row in subset:
                    queue.update(row['queue_event_counts'][task_name])
                for response in selected:
                    counter['delivered_saved_responses'] += 1
                    counter['schema_and_confidence_accepted'] += response['accepted']
                    counter['cache_hit'] += response['cache_hit'] is True
                    counter['known_noncache'] += response['cache_hit'] is False
                    if response['flags'] is not None:
                        counter.update({key: int(value) for key, value in response['flags'].items()})
                    if task_name == 'object_attributes':
                        counter['all_colors_unknown'] += not response.get('attribute_colors')
                        if response.get('receipt_synthetic_color'):
                            counter['receipt_associated'] += 1
                            counter['receipt_associated_known_color'] += bool(response['attribute_colors'])
                            counter['receipt_associated_exact_color'] += response['attribute_colors'] == [response['receipt_synthetic_color']]
                group['tasks'][task_name] = dict(response_counts=dict(counter),
                    trials_with_saved_response=len({row['run'] for row in selected}),
                    known_complete_event_applications=sum((row['applied_counts'] or {}).get(task_name, 0) for row in subset),
                    unknown_complete_event_trials=sum(row['applied_counts'] is None for row in subset),
                    complete_event_log_counts=dict(queue),
                    response_and_queue_denominators_may_differ=any(row['applied_counts'] is None for row in subset))
            groups[method+'/'+kind] = group
    overlap = {}
    dataset = OUT/'semantic_dataset/structured_supervision_002'
    for split in ['train', 'validation']:
        rows = [json.loads(line) for line in (dataset/(split+'.jsonl')).open()]
        selected = [row for row in rows if row['task'] == 'query_parse']
        homonymous = [row for row in selected if row['response'].get('anchor_text')
            and row['response']['anchor_text'] == row['response'].get('target_text') and row['response'].get('relation') == 'near']
        overlap[split] = dict(samples=len(homonymous),
            records=[dict(sample_id=row['sample_id'], query_text=row['query_text'], response=row['response']) for row in homonymous],
            interpretation='文字同名不一定是同一实例，本处不将自然表达同名自动判成错误标签；仅披露语法监督分布。')
    value = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_report_sha256=sha(source/'report.json'), source_trials_sha256=sha(source/'trials.jsonl'),
        source_responses_sha256=sha(source/'responses.jsonl'), source_sha256=sha(__file__),
        groups=groups, training_homonymous_anchor_examples=overlap,
        correction_scope='002 semantic_tasks.actual_applications只汇总有完整semantic_events.json试验的应用；此补充明确对应覆盖分母和未知项，不更改原文件。',
        limits=['已投递响应是条件分母；未投递原文不可得，不计全体后端语义准确率',
                'base缺完整结果条件仍有保存响应，但缺完整事件日志，因此响应数可多于完整事件delivered计数',
                '训练同名锚点不自动证明标签错误；仿真简单靠近任务没有独立锚点才构成明确的额外关系',
                '后验开发解释，不代替缺一消融；没有新增模型、执行、到达或人工标签'],
        actual_model_calls=0, new_executions=0, new_arrivals=0,
        manuscript_text_written=False, new_test_opened=False, goal_complete=False)
    with (dest/'report.json').open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(groups=len(groups), response_records=len(responses),
        source_trials=len(trials), explicit_unknown_complete_event_trials=1, goal_complete=False)))


if __name__ == '__main__':
    main()
