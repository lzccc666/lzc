"""对象图开发实验按log聚类的配对统计，帧不作为独立样本。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import numpy as np

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def aggregate(rows):
    keys = ['true_positive', 'truth', 'predictions', 'duplicate_nodes',
            'identity_switches', 'adjacent_matched_transitions',
            'nodes_associated_with_multiple_instances']
    total = {k: sum(r[k] for r in rows) for k in keys}
    total['recall'] = total['true_positive']/total['truth'] if total['truth'] else None
    total['precision'] = total['true_positive']/total['predictions'] if total['predictions'] else None
    total['identity_switch_rate'] = (total['identity_switches']/total['adjacent_matched_transitions']
                                     if total['adjacent_matched_transitions'] else None)
    total['center_error_m_mean'] = (sum(r['center_error_m_mean']*r['true_positive'] for r in rows
                                       if r['true_positive'])/total['true_positive']
                                      if total['true_positive'] else None)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping-run', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    root = OUTPUT/'mapping_evaluation'/args.mapping_run
    summary = json.loads((root/'summary.json').read_text())
    reports = [json.loads(open(p).read()) for p in summary['reports']]
    methods = list(reports[0]['methods'])
    by_log = defaultdict(list)
    for r in reports:
        by_log[r['log_token']].append(r)
    logs = sorted(by_log)
    rng = np.random.default_rng(17)
    comparisons = defaultdict(list)
    for _ in range(4000):
        sample = [r for log in rng.choice(logs, len(logs), replace=True) for r in by_log[log]]
        scores = {m: aggregate([r['methods'][m] for r in sample]) for m in methods}
        for m in methods:
            if m == 'original_fusion':
                continue
            for metric in ['recall', 'precision', 'identity_switch_rate', 'center_error_m_mean']:
                if scores[m][metric] is not None and scores['original_fusion'][metric] is not None:
                    comparisons[(m, metric)].append(scores[m][metric]-scores['original_fusion'][metric])
    observed = {m: aggregate([r['methods'][m] for r in reports]) for m in methods}
    intervals = []
    for (method, metric), values in comparisons.items():
        intervals.append(dict(method=method, baseline='original_fusion', metric=metric,
            difference=observed[method][metric]-observed['original_fusion'][metric],
            ci95=np.quantile(values, [.025,.975]).tolist(), valid_resamples=len(values)))
    dest = OUTPUT/'statistics'/args.run_id
    dest.mkdir(parents=True, exist_ok=False)
    dump(dest/'report.json', dict(source=str(root), source_summary_sha256=digest(root/'summary.json'),
        scene_count=len(reports), log_count=len(logs), frames=sum(r['frame_count'] for r in reports),
        metrics=observed, paired_log_cluster_bootstrap=intervals, seed=17, iterations=4000,
        denominator='GT是限定可见范围内的跨帧对象观测；独立聚类单位为采集log',
        statistical_limits='11个开发log，区间不代表新测试泛化；未知关联不全计作误检；多实例节点是自动诊断',
        independent_test=False))
    lines = ['# 开发对象图实验统计', '', f'{len(reports)}场景，{len(logs)}个log；非封存测试。', '',
        '|方法|GT观测|预测|匹配|召回|匹配精确率|中心误差(m)|重复|身份变化/连续匹配|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m, r in observed.items():
        lines.append(f"|{m}|{r['truth']}|{r['predictions']}|{r['true_positive']}|{r['recall']:.4f}|"
            f"{r['precision']:.4f}|{r['center_error_m_mean']:.3f}|{r['duplicate_nodes']}|"
            f"{r['identity_switches']}/{r['adjacent_matched_transitions']}|")
    lines += ['', '区间按log重采样并保持方法配对，详见report.json；不能把单帧或模型调用作为独立样本。']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(observed, ensure_ascii=False))


if __name__ == '__main__':
    main()
