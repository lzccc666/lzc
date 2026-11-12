"""关联门限实际重建结果的配对log统计；严格核验全部冻结场景和原始记录。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from conceptgraph.completion.fault_statistics import METRICS, finite, ratios
from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-run',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--wait-budget-s',type=float,default=0.)
    args=parser.parse_args()
    source=OUTPUT/'performance'/args.input_run
    begun=time.monotonic()
    while not (source/'summary.json').exists():
        if time.monotonic()-begun>args.wait_budget_s:
            raise TimeoutError('full association parameter run has not completed')
        time.sleep(5)
    release=json.loads((source/'release.json').read_text())
    summary=json.loads((source/'summary.json').read_text())
    rows=[json.loads(line) for line in (source/'cases.jsonl').open()]
    expected={(s,t) for s in release['scene_logs'] for t in release['thresholds']}
    if len(rows)!=len(expected) or {(r['scene'],r['threshold']) for r in rows}!=expected:
        raise ValueError('missing or duplicated frozen case')
    if digest(source/'cases.jsonl')!=summary['records_sha256']:
        raise ValueError('raw case list changed')
    logs=sorted(set(release['scene_logs'].values()))
    thresholds=release['thresholds']
    counts=np.zeros((len(logs),len(thresholds),7))
    for row in rows:
        report_path=Path(row['report_path'])
        report=json.loads(report_path.read_text())
        if digest(report_path)!=row['report_sha256'] or digest(report_path.parent/'frames.jsonl')!=report['records_sha256']:
            raise ValueError('case report or raw frame data changed')
        cfg=dict(row['inference_config'])
        cfg.pop('sim_threshold')
        fixed=dict(release['fixed_config'])
        fixed.pop('sim_threshold')
        if cfg!=fixed or row['inference_config']['sim_threshold']!=row['threshold']:
            raise ValueError('more than one inference parameter changed')
        metric=report['metrics']
        tp=metric['true_positive']
        vector=[tp,metric['truth'],metric['predictions'],tp*(metric['center_error_m_mean'] or 0),
            metric['identity_switches'],metric['adjacent_matched_transitions'],metric['duplicate_nodes']]
        counts[logs.index(row['log_token']),thresholds.index(row['threshold'])]+=vector
    rng=np.random.default_rng(17)
    points=ratios(counts.sum(axis=0))
    samples=ratios(counts[rng.integers(0,len(logs),(4000,len(logs)))].sum(axis=1))
    baseline=thresholds.index(.35)
    estimates,paired={},[]
    for j,threshold in enumerate(thresholds):
        estimates[str(threshold)]={}
        for k,name in enumerate(METRICS):
            valid=samples[:,j,k]
            valid=valid[np.isfinite(valid)]
            estimates[str(threshold)][name]=dict(estimate=finite(points[j,k]),
                ci95=np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,valid_resamples=len(valid))
            if j!=baseline:
                difference=samples[:,j,k]-samples[:,baseline,k]
                valid=difference[np.isfinite(difference)]
                paired.append(dict(threshold=threshold,baseline=.35,metric=name,
                    difference=finite(points[j,k]-points[baseline,k]),
                    ci95=np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,valid_resamples=len(valid)))
    dest=OUTPUT/'statistics'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    dump(dest/'report.json',dict(source_run=args.input_run,release_sha256=digest(source/'release.json'),
        summary_sha256=digest(source/'summary.json'),case_records_sha256=digest(source/'cases.jsonl'),
        statistics_source_sha256=digest(__file__),scenes=len(release['scene_logs']),logs=len(logs),
        input_frames=summary['unique_frames'],cases=len(rows),conditions=estimates,paired_differences=paired,
        bootstrap=dict(unit='采集log',paired=True,repetitions=4000,seed=17),fixed_seed=17,
        limitations=['只改变推理sim_threshold，评估3米距离门限保持固定；并非改评价阈值制造收益',
            '基线0.35复用相同观测/种子/配置的无扰动完整记录，未增加独立样本',
            'CPU与真实时钟导航共享主机；不作为独占性能成绩，参数未应用于在运行的导航',
            '开发范围敏感性分析，不是新封存主表或下游到达实验'],
        parameter_applied_to_running_navigation=False,independent_test=False,goal_complete=False))
    with (dest/'source_data_by_log.csv').open('x') as handle:
        writer=csv.writer(handle)
        writer.writerow(['log','threshold','true_positive','truth','predictions','center_error_sum_m',
                         'identity_switches','matched_transitions','duplicate_nodes'])
        for i,log in enumerate(logs):
            for j,threshold in enumerate(thresholds):
                writer.writerow([log,threshold,*counts[i,j].tolist()])
    print(json.dumps(estimates,ensure_ascii=False))


if __name__=='__main__':
    main()
