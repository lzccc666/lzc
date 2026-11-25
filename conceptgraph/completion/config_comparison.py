"""固定全范围计数与日志级配对统计；不参与推理。"""
from __future__ import annotations

import numpy as np

from conceptgraph.completion.temporal_identity_evaluation import identity_counts, prepare_sequence


def ratios(c):
    tp = c['true_positive']
    return dict(recall=tp / max(c['truth'], 1), precision=tp / max(c['predictions'], 1),
                F1=2 * tp / max(c['truth'] + c['predictions'], 1),
                IDF1=2 * c['IDTP'] / max(2 * c['IDTP'] + c['IDFP'] + c['IDFN'], 1),
                center_error_m=c['center_error_sum_m'] / tp if tp else None)


def summarize(records, method):
    c = {k: sum(r['metrics'][method][k] for r in records)
         for k in ('true_positive', 'truth', 'predictions', 'duplicate_nodes')}
    c['center_error_sum_m'] = sum(m['distance_m'] for r in records for m in r['metrics'][method]['matches'])
    c.update(identity_counts(prepare_sequence(records, method), 'all_gated')['counts'])
    return dict(counts=c, metrics=ratios(c))


def paired_intervals(baseline, candidate, *, seed=20261120, replicates=10000):
    logs = sorted({r['log'] for r in baseline})
    if {r['scene'] for r in baseline} != {r['scene'] for r in candidate}:
        raise ValueError('必须比较相同完整场景')
    by_name = {r['scene']: r for r in candidate}
    if any(by_name[r['scene']]['log'] != r['log'] or by_name[r['scene']]['frames'] != r['frames']
           for r in baseline):
        raise ValueError('帧数或日志不对应')
    keys = ('true_positive', 'truth', 'predictions', 'IDTP', 'IDFP', 'IDFN')
    def grouped(rows):
        return np.array([[sum(r['result']['counts'][k] for r in rows if r['log'] == log)
                          for k in keys] for log in logs])
    draws = np.random.default_rng(seed).integers(0, len(logs), (replicates, len(logs)))
    def sampled(rows):
        c = grouped(rows)[draws].sum(axis=1)
        return dict(recall=c[:, 0] / np.maximum(c[:, 1], 1),
                    F1=2 * c[:, 0] / np.maximum(c[:, 1] + c[:, 2], 1),
                    IDF1=2 * c[:, 3] / np.maximum(2 * c[:, 3] + c[:, 4] + c[:, 5], 1))
    a, b = sampled(baseline), sampled(candidate)
    delta = {k: 100 * (b[k] - a[k]) for k in a}
    alpha = .05 / 3
    return dict(logs=logs, seed=seed, replicates=replicates, comparisons_registered=3,
                units='percentage_points',
                ci95={k: np.quantile(v, [.025, .975]).tolist() for k, v in delta.items()},
                ci_bonferroni={k: np.quantile(v, [alpha / 2, 1 - alpha / 2]).tolist()
                               for k, v in delta.items()})
