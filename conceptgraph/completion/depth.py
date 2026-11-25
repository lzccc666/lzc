"""只使用当前掩膜内的实测稀疏深度，保留支持量和背景污染诊断。"""
from __future__ import annotations

import numpy as np


def depth_support(depth, mask, *, robust=False, minimum=1.0, maximum=65.0):
    depth = np.asarray(depth, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    valid = mask & np.isfinite(depth) & (depth > minimum) & (depth < maximum)
    values = depth[valid]
    evidence = dict(mask_pixels=int(mask.sum()), valid_pixels=int(valid.sum()),
        retained_pixels=int(valid.sum()), units='m', source='current_projected_lidar',
        robust_filter=bool(robust), depth_quantiles_m=np.quantile(values, [0,.1,.5,.9,1]).tolist() if len(values) else None)
    if robust and len(values) >= 5:
        sorted_values = np.sort(values)
        gaps = np.diff(sorted_values)
        # 当前开发协议的固定空间门限；不使用GT选择前景。
        splits = np.flatnonzero(gaps > np.maximum(1.0, .03*sorted_values[:-1]))+1
        clusters = np.split(sorted_values, splits)
        # 同支持量选更近的簇，保留unknown和少量点，不虚构补全深度。
        chosen = max(clusters, key=lambda c: (len(c), -float(np.median(c))))
        valid &= (depth >= chosen[0]) & (depth <= chosen[-1])
        evidence.update(retained_pixels=int(valid.sum()), cluster_count=len(clusters),
            support_fraction=float(len(chosen)/len(values)))
    evidence['sufficient_support'] = evidence['retained_pixels'] >= 5
    return valid, evidence
