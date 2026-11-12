"""检索开发基线的配对log聚类统计；保留未知关联及无目标误选。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json

import numpy as np

from conceptgraph.completion.stage9 import OUTPUT, available_development, digest, dump


def aggregate(rows, scene_logs):
    methods=sorted({r['method'] for r in rows})
    grouped=defaultdict(dict)
    counts=np.zeros((len(set(scene_logs.values())),len(methods),4),dtype=int)
    logs=sorted(set(scene_logs.values()))
    for row in rows:
        key=(row['scene'],row['frame_idx'],row['query_index'])
        if row['method'] in grouped[key]:
            raise ValueError('duplicate method for same frozen query')
        grouped[key][row['method']]=row
        i,j=logs.index(scene_logs[row['scene']]),methods.index(row['method'])
        positive=bool(row['positive'])
        counts[i,j]+=np.array([int(positive and row['selected_correct']),int(positive),
            int(not positive and row['false_selection_on_no_match']),int(not positive)])
    for group in grouped.values():
        if set(group)!=set(methods):
            raise ValueError('incomplete paired query; do not drop missing predictions')
        signatures={(r['query_text'],r['positive'],tuple(r['acceptable_instances'])) for r in group.values()}
        if len(signatures)!=1:
            raise ValueError('method-specific truth or query detected')
    return methods,logs,counts


def bootstrap(counts, repetitions=4000, seed=17):
    rng=np.random.default_rng(seed)
    totals=counts.sum(axis=0)
    estimate=np.divide(totals[:,[0,2]],totals[:,[1,3]],
        out=np.full((counts.shape[1],2),np.nan),where=totals[:,[1,3]]>0)
    indices=rng.integers(0,len(counts),size=(repetitions,len(counts)))
    samples=counts[indices].sum(axis=1)
    rates=np.divide(samples[:,:,[0,2]],samples[:,:,[1,3]],
        out=np.full((repetitions,counts.shape[1],2),np.nan),where=samples[:,:,[1,3]]>0)
    return totals,estimate,rates


def interval(values):
    valid=values[np.isfinite(values)]
    return dict(ci95=np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,
        valid_resamples=len(valid))


def nullable_number(value):
    return float(value) if np.isfinite(value) else None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-run',required=True)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    source=OUTPUT/'main_experiments'/args.input_run
    raw=source/'queries.jsonl'
    raw_sha=digest(raw)
    release=json.loads((source/'release.json').read_text())
    report=json.loads((source/'report.json').read_text())
    for path,sha in release['source_frames_sha256'].items():
        if digest(path)!=sha:
            raise ValueError('mapping evidence changed')
    rows=[json.loads(line) for line in raw.open()]
    scenes={r['scene'] for r in rows}
    scene_logs={r['scene']:r['log_token'] for r in available_development() if r['scene'] in scenes}
    methods,logs,counts=aggregate(rows,scene_logs)
    totals,estimates,rates=bootstrap(counts)
    metric_names=['positive_selection','no_match_false_selection']
    results={}
    differences=[]
    baseline=methods.index('existing_object_joint_rule')
    for j,method in enumerate(methods):
        expected=report['methods'][method]
        if totals[j].tolist()!=[expected[k] for k in
                ['positive_selected_correct','positive_queries','no_match_false_selection','negative_queries']]:
            raise ValueError('raw records disagree with completed report')
        results[method]={}
        for m,metric in enumerate(metric_names):
            results[method][metric]=dict(estimate=nullable_number(estimates[j,m]),
                numerator=int(totals[j,m*2]),denominator=int(totals[j,m*2+1]),
                **interval(rates[:,j,m]))
            if j!=baseline:
                differences.append(dict(method=method,baseline=methods[baseline],metric=metric,
                    difference=nullable_number(estimates[j,m]-estimates[baseline,m]),
                    **interval(rates[:,j,m]-rates[:,baseline,m])))
    if digest(raw)!=raw_sha:
        raise ValueError('records changed while computing statistics')
    dest=OUTPUT/'statistics'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    dump(dest/'report.json',dict(source_run=args.input_run,source_sha256=raw_sha,
        release_sha256=digest(source/'release.json'),completed_report_sha256=digest(source/'report.json'),
        statistics_source_sha256=digest(__file__),scenes=len(scenes),logs=len(logs),
        paired_queries=len(rows)//len(methods),methods=results,paired_differences=differences,
        bootstrap=dict(unit='采集log',repetitions=4000,seed=17,paired=True),
        denominators=f'正查询{int(totals[0,1])}及无匹配{int(totals[0,3])}；未知关联/目标未入候选保留为正任务失败；不以帧或调用数为独立样本',
        association_limits='当前检测/网格为类别及3m中心关联(次近差<0.5m为unknown)，对象图为原独立一对一关联；口径不同，不将点估计差全归因于检索模块',
        adaptation='VLMaps式稀疏crop CLIP二维网格；不等同完整VLMaps复现',
        independent_test=False,arrival_evaluated=False,stage12_complete=False,goal_complete=False))
    with (dest/'source_data_by_log.csv').open('x') as f:
        writer=csv.writer(f)
        writer.writerow(['log_token','method','positive_correct','positive_queries','negative_false_selection','negative_queries'])
        for i,log in enumerate(logs):
            for j,method in enumerate(methods):
                writer.writerow([log,method,*counts[i,j].tolist()])
    print(json.dumps(dict(methods=results,paired_queries=len(rows)//len(methods),logs=len(logs)),ensure_ascii=False))


if __name__=='__main__':
    main()
