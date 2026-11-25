"""完整日志配对重采样的附指标；不改变原召回主检验或推理。"""
from __future__ import annotations

import numpy as np

from conceptgraph.completion.config_comparison import paired_intervals


def paired_metric_uncertainty(baseline, candidate, *, seed=20261120, replicates=10000):
    """同一日志抽样联合统计各指标，区间针对微平均差值而非场景均值。"""
    primary = paired_intervals(baseline, candidate, seed=seed, replicates=replicates)
    if len({r['scene'] for r in baseline}) != len(baseline) or len({r['scene'] for r in candidate}) != len(candidate):
        raise ValueError('场景重复会改变重采样权重')
    by_scene = {r['scene']: r for r in candidate}
    if any(r['result']['counts']['truth'] != by_scene[r['scene']]['result']['counts']['truth'] for r in baseline):
        raise ValueError('完整真值分母不一致')
    logs = primary['logs']
    keys = ('true_positive', 'truth', 'predictions', 'IDTP', 'IDFP', 'IDFN',
            'duplicate_nodes', 'center_error_sum_m')
    draws = np.random.default_rng(seed).integers(0, len(logs), (replicates, len(logs)))

    def sampled(rows):
        grouped = np.array([[sum(r['result']['counts'][k] for r in rows if r['log'] == log)
                             for k in keys] for log in logs], dtype=float)
        c = grouped[draws].sum(axis=1)
        return dict(recall=100 * c[:, 0] / np.maximum(c[:, 1], 1),
            precision=100 * c[:, 0] / np.maximum(c[:, 2], 1),
            F1=200 * c[:, 0] / np.maximum(c[:, 1] + c[:, 2], 1),
            IDF1=200 * c[:, 3] / np.maximum(2 * c[:, 3] + c[:, 4] + c[:, 5], 1),
            duplicate_nodes=c[:, 6],
            center_error_m=np.divide(c[:, 7], c[:, 0], out=np.full(len(c), np.nan), where=c[:, 0] > 0))

    a, b = sampled(baseline), sampled(candidate)
    alpha = .05 / 3
    intervals = {}
    for metric in a:
        delta = b[metric] - a[metric]
        undefined = int(np.count_nonzero(~np.isfinite(delta)))
        intervals[metric] = dict(units=('m' if metric == 'center_error_m' else 'node_observations'
            if metric == 'duplicate_nodes' else 'percentage_points'),
            undefined_replicates=undefined,
            ci95=np.quantile(delta, [.025, .975]).tolist() if not undefined else None,
            ci_bonferroni=np.quantile(delta, [alpha / 2, 1 - alpha / 2]).tolist() if not undefined else None)
    for key in ('recall', 'F1', 'IDF1'):
        np.testing.assert_allclose(intervals[key]['ci95'], primary['ci95'][key], atol=1e-12)
        np.testing.assert_allclose(intervals[key]['ci_bonferroni'], primary['ci_bonferroni'][key], atol=1e-12)
    return dict(logs=logs, seed=seed, replicates=replicates, comparisons_registered=3,
        intervals=intervals, primary_exactly_preserved=True,
        interpretation='三配置校正沿用原主检验；附指标为同一抽样的描述性区间，不是六指标联合家族错误率控制。中心误差仅针对各方法已匹配对象，样本组成不同。')
