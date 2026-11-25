"""论文构图P0的固定评价；不改预测或原评价分母。"""
from __future__ import annotations

from collections import Counter

from conceptgraph.scenegraph.relations import inverse_relation


RELATION_CONFIG = dict(
    coordinate_frame='ego_heading', coordinate_frame_fallback=None,
    relation_distance_thresholds_m=dict(near=6., directional=12., left_of=12., right_of=12., in_front_of=12., behind=12.),
    directional_mode='all_supported_axes', minimum_direction_component_m=1.,
    position_update_threshold_m=.1)


def expand(edges):
    facts = set()
    for edge in edges:
        facts.add((edge.source, edge.relation, edge.target))
        facts.add((edge.target, inverse_relation(edge.relation), edge.source))
    return facts


def scores(counts):
    def ratio(a, b):
        return a / b if b else None
    tp, pred, truth, conditional = (counts[x] for x in ['TP', 'all_pred', 'all_truth', 'conditional_truth'])
    return dict(conditional_precision=ratio(tp, counts['resolved_pred']),
        conditional_recall=ratio(tp, conditional),
        conditional_F1=ratio(2 * tp, counts['resolved_pred'] + conditional),
        full_precision=ratio(tp, pred), full_recall=ratio(tp, truth),
        full_F1=ratio(2 * tp, pred + truth),
        endpoint_pair_coverage=ratio(conditional, truth))


def edge_counts(predicted, truth, mapping, relation=None):
    predicted = {e for e in predicted if relation is None or e[1] == relation}
    truth = {e for e in truth if relation is None or e[1] == relation}
    matched = set(mapping.values())
    conditional = {e for e in truth if e[0] in matched and e[2] in matched}
    resolved = {(mapping[s], r, mapping[t]) for s, r, t in predicted if s in mapping and t in mapping}
    unresolved = {e for e in predicted if e[0] not in mapping or e[2] not in mapping}
    tp = len(resolved & truth)
    count = Counter(TP=tp, all_pred=len(predicted), resolved_pred=len(resolved),
        unresolved_pred=len(unresolved), all_truth=len(truth), conditional_truth=len(conditional),
        full_FP=len(predicted) - tp, full_FN=len(truth) - tp,
        missing_endpoint_FN=len(truth) - len(conditional),
        covered_endpoint_FN=len(conditional - resolved))
    if count['full_FN'] != count['missing_endpoint_FN'] + count['covered_endpoint_FN']:
        raise AssertionError('关系FN分解不闭合')
    if len(predicted) != len(resolved) + len(unresolved):
        raise AssertionError('一对一端点映射丢边')
    return count


def semantic_content(edges):
    return {(e.source, e.target, e.relation, e.directed, e.coordinate_frame) for e in edges}


def temporal_content(edges):
    # 比较当前查询所用来源观测帧；不要求增量历史跨度与全量重建天然相同。
    return {(e.source, e.target, e.relation, e.coordinate_frame,
        e.metadata.get('source_observed_frame'), e.metadata.get('target_observed_frame'),
        e.metadata.get('source_timestamp_s'), e.metadata.get('target_timestamp_s')) for e in edges}
