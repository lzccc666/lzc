"""带门限的一对一匹配，允许不匹配；不访问官方实例真值。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def gated_assignment(scores, *, policy='one_to_one'):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        raise ValueError('匹配分数必须为二维有限数或负无穷')
    m, n = scores.shape
    if not m or not n:
        return {}
    if policy == 'greedy_many_to_one':
        return {i: int(np.argmax(scores[i])) for i in range(m)
                if np.any(np.isfinite(scores[i]))}
    if policy != 'one_to_one':
        raise ValueError('未知关联策略')
    finite = np.isfinite(scores)
    # 每行独立dummy，优先最大合法匹配数，再最大总分；禁止边在求解前置为无效。
    normalized = np.zeros_like(scores)
    if finite.any():
        low, high = scores[finite].min(), scores[finite].max()
        normalized[finite] = (scores[finite]-low)/max(high-low, 1.0)
    cost = np.full((m, n+m), 1e9)
    cost[:, :n] = np.where(finite, -normalized-(m+1), 1e9)
    cost[:, n:] = 0.0
    rows, cols = linear_sum_assignment(cost)
    return {int(i): int(j) for i, j in zip(rows, cols) if j < n and finite[i, j]}


def gated_distance_match(predictions, truth, threshold_m=3.0):
    if threshold_m <= 0:
        raise ValueError('距离阈值必须为正')
    score = np.full((len(predictions), len(truth)), -np.inf)
    for i, pred in enumerate(predictions):
        for j, obj in enumerate(truth):
            if pred['category'] != obj['category']:
                continue
            d = float(np.linalg.norm(np.asarray(pred['center'][:2])-obj['center'][:2]))
            if np.isfinite(d) and d <= threshold_m:
                score[i, j] = -d
    matches = gated_assignment(score)
    return [dict(prediction_index=i, truth_index=j, distance_m=float(-score[i, j]))
            for i, j in sorted(matches.items())]
