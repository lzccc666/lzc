"""完整公平验证的配对聚类统计与类别覆盖；区分真实crop/构造解析/符号排序。"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import time

import numpy as np

from conceptgraph.completion.semantic_evaluation import metrics_for
from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-run',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--wait-budget-s',type=float,default=0.)
    args=parser.parse_args()
    source=OUTPUT/'main_experiments'/args.input_run
    started=time.monotonic()
    while not (source/'report.json').exists():
        if time.monotonic()-started>args.wait_budget_s:
            raise TimeoutError('all five variants must finish; no partial main statistics')
        time.sleep(5)
    completed=json.loads((source/'report.json').read_text())
    release=json.loads((source/'release.json').read_text())
    dataset=OUTPUT/'semantic_dataset/structured_supervision_002'
    manifest=json.loads((dataset/'manifest.json').read_text())
    if digest(dataset/'validation.jsonl')!=manifest['files']['validation.jsonl'] or \
            digest(dataset/'manifest.json')!=release['dataset_manifest_sha256']:
        raise ValueError('frozen validation data changed')
    lookup={r['sample_id']:r for r in map(json.loads,(dataset/'validation.jsonl').open())}
    rows=[lookup[i] for i in release['sample_ids']]
    variants=[v['name'] for v in release['variants']]
    if set(variants)!=set(completed['variants']) or len(rows)!=603:
        raise ValueError('incomplete or changed comparison')
    outputs={}
    hashes={}
    source_rows=[]
    grouped_details={}
    for name in variants:
        path=source/name/'raw_records.jsonl'
        hashes[name]=digest(path)
        records=[json.loads(line) for line in path.open()]
        if [r['sample_id'] for r in records]!=[r['sample_id'] for r in rows]:
            raise ValueError('not the same ordered requests')
        recalculated=metrics_for(records,rows)
        report=completed['variants'][name]
        if recalculated!=report['metrics'] or report['cache_replay_added_calls']!=0:
            raise ValueError('saved scores or cache replay do not match raw records')
        outputs[name]={r['sample_id']:r for r in records}
        for record,row in zip(records,rows):
            source_rows.append(dict(model=name,task=row['task'],sample_id=row['sample_id'],
                scene_id=row['scene_id'],log_token=row['log_token'],
                semantic_correct=record['semantic_correct'],schema_valid=record['envelope']['output_schema_valid'],
                fallback=record['envelope']['fallback_used'],input_cache_hit=record['envelope']['cache']['hit']))
        categories=sorted({r['response']['object_tag'] for r in rows if r['task']=='object_attributes'})
        by_category={}
        for category in categories:
            subset=[r for r in rows if r['task']=='object_attributes' and r['response']['object_tag']==category]
            successes=sum(outputs[name][r['sample_id']]['semantic_correct'] for r in subset)
            by_category[category]=dict(correct=successes,requests=len(subset),accuracy=successes/len(subset),
                unique_instances=len({r['instance_token'] for r in subset}),logs=len({r['log_token'] for r in subset}))
        query_types=sorted({r['query_kind'] for r in rows if r['task']=='candidate_rerank'})
        by_rank_type={}
        for kind in query_types:
            subset=[r for r in rows if r['task']=='candidate_rerank' and r['query_kind']==kind]
            successes=sum(outputs[name][r['sample_id']]['semantic_correct'] for r in subset)
            by_rank_type[kind]=dict(correct=successes,requests=len(subset),accuracy=successes/len(subset))
        grouped_details[name]=dict(crop_categories=by_category,
            category_macro_over_present_labels=float(np.mean([v['accuracy'] for v in by_category.values()])),
            controlled_ranking_query_types=by_rank_type,
            parse_field_counts={k:v for k,v in recalculated['query_parse'].items() if k.endswith('_correct')},
            candidate_permutation=report['candidate_permutation'])
    task_results={}
    paired=[]
    rng=np.random.default_rng(17)
    baseline=variants.index('base_fullschema')
    for task in sorted({r['task'] for r in rows}):
        subset=[r for r in rows if r['task']==task]
        groups=sorted({r['log_token'] for r in subset})
        numerator=np.zeros((len(groups),len(variants)))
        denominator=np.zeros(len(groups))
        for row in subset:
            i=groups.index(row['log_token'])
            denominator[i]+=1
            for j,name in enumerate(variants):
                numerator[i,j]+=outputs[name][row['sample_id']]['semantic_correct']
        draws=rng.integers(0,len(groups),(4000,len(groups)))
        samples=numerator[draws].sum(axis=1)/denominator[draws].sum(axis=1)[:,None]
        point=numerator.sum(axis=0)/denominator.sum()
        unit='独立受控符号bank' if task=='candidate_rerank' else '采集log'
        task_results[task]=dict(requests=len(subset),cluster_unit=unit,clusters=len(groups),
            group_counts=dict(Counter(r['log_token'] for r in subset)),
            models={name:dict(correct=int(numerator[:,j].sum()),requests=len(subset),accuracy=float(point[j]),
                ci95=np.quantile(samples[:,j],[.025,.975]).tolist()) for j,name in enumerate(variants)})
        for j,name in enumerate(variants):
            if j!=baseline:
                paired.append(dict(task=task,model=name,baseline='base_fullschema',
                    difference=float(point[j]-point[baseline]),
                    ci95=np.quantile(samples[:,j]-samples[:,baseline],[.025,.975]).tolist(),
                    cluster_unit=unit,clusters=len(groups)))
    dest=OUTPUT/'statistics'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    dump(dest/'report.json',dict(input_run=args.input_run,
        completed_report_sha256=digest(source/'report.json'),release_sha256=digest(source/'release.json'),
        raw_record_hashes=hashes,dataset_manifest_sha256=digest(dataset/'manifest.json'),
        statistics_source_sha256=digest(__file__),tasks=task_results,paired_differences=paired,
        detailed_groups=grouped_details,bootstrap=dict(repetitions=4000,seed=17,paired=True),
        limitations=['解析和crop仅2个验证log，聚类区间离散且极不稳定，不能作充分泛化或显著性证据',
            '解析指令为构造语法；排序来自6个受控验证bank/44种去置换证据，180行不等于180个独立道路场景',
            'crop类别以官方框投影及检测唯一匹配监督，未经逐crop人工审核；颜色/细属性/单图运动不参与该类别指标',
            '候选置换的身份一致性不等于选择正确；三个新种子逐个显示，不按本完整验证挑选种子',
            '全体603相同请求包含自然缓存复用，不把请求数全称为不同CUDA调用；回退无准确率信用',
            '不评价真实道路检索或车辆实际到达；独立封存测试未开启'],
        performance_credit=False,independent_test=False,stage10_complete=False,goal_complete=False))
    with (dest/'paired_request_scores.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    print(json.dumps(dict(tasks=task_results,paired_differences=paired),ensure_ascii=False))


if __name__=='__main__':
    main()
