"""真实录制扰动实验的配对log聚类评价；种子不增加独立log数。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time

import numpy as np

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


METRICS={'recall':(0,1),'precision':(0,2),'center_error_m_mean':(3,0),
    'identity_switch_rate':(4,5),'duplicates_per_gt_observation':(6,1)}


def ratios(counts):
    result=[]
    for numerator,denominator in METRICS.values():
        result.append(np.divide(counts[...,numerator],counts[...,denominator],
            out=np.full(counts.shape[:-1],np.nan),where=counts[...,denominator]>0))
    return np.stack(result,axis=-1)


def finite(value):
    return float(value) if math.isfinite(float(value)) else None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-run',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--wait-budget-s',type=float,default=0)
    args=parser.parse_args()
    source=OUTPUT/'faults'/args.input_run
    started=time.monotonic()
    while not (source/'summary.json').exists():
        if time.monotonic()-started>args.wait_budget_s:
            raise TimeoutError('full perturbation run incomplete; no final statistics produced')
        time.sleep(5)
    release=json.loads((source/'release.json').read_text())
    summary=json.loads((source/'summary.json').read_text())
    for path,sha in release['source_frames'].items():
        if digest(path)!=sha:
            raise ValueError('frozen original frames changed')
    expected={(scene,seed,condition) for scene in release['scene_logs']
        for seed in release['seeds'] for condition in release['conditions']}
    conditions=sorted(release['conditions'])
    logs=sorted(set(release['scene_logs'].values()))
    counts=np.zeros((len(logs),len(conditions),7))
    by_seed={seed:np.zeros((len(conditions),7)) for seed in release['seeds']}
    observed=set()
    records=[]
    sources={}
    for path in sorted(source.glob('scene-*/seed*/*/report.json')):
        report=json.loads(path.read_text())
        key=(report['scene'],report['seed'],report['condition'])
        if key in observed:
            raise ValueError('duplicate trial')
        observed.add(key)
        if digest(path.parent/'frames.jsonl')!=report['records_sha256']:
            raise ValueError('raw mapping records changed')
        metric=report['metrics']
        tp=metric['true_positive']
        vector=np.array([tp,metric['truth'],metric['predictions'],
            tp*(metric['center_error_m_mean'] or 0),metric['identity_switches'],
            metric['adjacent_matched_transitions'],metric['duplicate_nodes']])
        i=logs.index(release['scene_logs'][report['scene']])
        j=conditions.index(report['condition'])
        counts[i,j]+=vector
        by_seed[report['seed']][j]+=vector
        sources[str(path.relative_to(source))]=digest(path)
        records.append(report)
    if observed!=expected or summary['cases']!=len(expected):
        raise ValueError('all frozen cases required; failures cannot be dropped')
    rng=np.random.default_rng(17)
    draws=rng.integers(0,len(logs),(4000,len(logs)))
    point=ratios(counts.sum(axis=0))
    bootstrap=ratios(counts[draws].sum(axis=1))
    baseline=conditions.index('unperturbed')
    estimates={}
    differences=[]
    for j,condition in enumerate(conditions):
        estimates[condition]={}
        for k,metric in enumerate(METRICS):
            samples=bootstrap[:,j,k]
            valid=samples[np.isfinite(samples)]
            estimates[condition][metric]=dict(estimate=finite(point[j,k]),
                ci95=np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,
                valid_resamples=len(valid))
            if j!=baseline:
                diff=samples-bootstrap[:,baseline,k]
                valid=diff[np.isfinite(diff)]
                differences.append(dict(condition=condition,baseline='unperturbed',metric=metric,
                    difference=finite(point[j,k]-point[baseline,k]),
                    ci95=np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,
                    valid_resamples=len(valid)))
    dest=OUTPUT/'statistics'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    dump(dest/'report.json',dict(source_run=args.input_run,source_release_sha256=digest(source/'release.json'),
        source_summary_sha256=digest(source/'summary.json'),case_report_hashes=sources,
        statistics_source_sha256=digest(__file__),scenes=len(release['scene_logs']),logs=len(logs),
        frames=summary['input_frames'],conditions=estimates,paired_differences=differences,
        seed_estimates={str(seed):{condition:{metric:finite(v) for metric,v in zip(METRICS,row)}
            for condition,row in zip(conditions,ratios(matrix))} for seed,matrix in by_seed.items()},
        counts_fields=['true_positive','truth','predictions','sum_center_error_m','identity_switches',
            'adjacent_matched_transitions','duplicate_nodes'],
        bootstrap=dict(unit='采集log',clusters=len(logs),repetitions=4000,seed=17,paired=True),
        interpretation=['3种子重复不增加独立场景/log；GT观测计数会重复3次',
            '扰动保持同一GT和真实相机可见范围；不以规划失败自证不可达',
            '中心误差为匹配对象条件均值；未匹配预测含关联不足，不全解释为误检',
            '合成随机缺失及白噪声不是标定后的真实传感器误差分布',
            '只评建图压力，不给实际到达、语义准确或动态道路安全信用'],
        navigation_arrival_evaluated=False,independent_test=False,stage12_complete=False,goal_complete=False))
    with (dest/'counts_by_log.csv').open('x') as f:
        writer=csv.writer(f)
        writer.writerow(['log_token','condition','true_positive','truth','predictions','sum_center_error_m',
            'identity_switches','adjacent_matched_transitions','duplicate_nodes'])
        for i,log in enumerate(logs):
            for j,condition in enumerate(conditions):
                writer.writerow([log,condition,*counts[i,j].tolist()])
    dump(dest/'scene_seed_cases.json',records)
    print(json.dumps(dict(scenes=len(release['scene_logs']),logs=len(logs),cases=len(records),conditions=estimates),ensure_ascii=False))


if __name__=='__main__':
    main()
