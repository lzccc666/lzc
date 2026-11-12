"""局部对象图诊断指标；非nuScenes官方榜单协议。"""
from __future__ import annotations

from collections import defaultdict
import numpy as np
from conceptgraph.completion.association import gated_distance_match


def object_frame_metrics(predictions, truth, *, threshold_m=3.0):
    matches = gated_distance_match(predictions, truth, threshold_m)
    matched_pred = {m['prediction_index'] for m in matches}
    duplicates = 0
    for i, p in enumerate(predictions):
        if i in matched_pred:
            continue
        duplicates += any(p['category'] == t['category'] and np.linalg.norm(
            np.asarray(p['center'][:2])-t['center'][:2]) <= threshold_m for t in truth)
    distances = [m['distance_m'] for m in matches]
    return dict(predictions=len(predictions), truth=len(truth), true_positive=len(matches),
                unmatched_predictions=len(predictions)-len(matches),
                false_negative=len(truth)-len(matches), duplicate_nodes=int(duplicates),
                recall=len(matches)/len(truth) if truth else None,
                precision=len(matches)/len(predictions) if predictions else None,
                center_error_m_mean=float(np.mean(distances)) if distances else None,
                matches=matches)


def identity_metrics(frame_pairs):
    """只跨连续有证据帧计身份变化；多实例合并是自动关联诊断，不能当人工真值。"""
    last = {}
    node_instances = defaultdict(set)
    switches = transitions = 0
    for frame_idx, pairs in frame_pairs:
        for node_id, instance in pairs:
            if instance in last:
                prev_frame, prev_node = last[instance]
                if prev_frame == frame_idx-1:
                    transitions += 1
                    switches += prev_node != node_id
            last[instance] = (frame_idx, node_id)
            node_instances[node_id].add(instance)
    return dict(identity_switches=int(switches), adjacent_matched_transitions=transitions,
                nodes_associated_with_multiple_instances=sum(len(s)>1 for s in node_instances.values()),
                association_status='自动中心距离关联诊断，未声称人工验证错合并')


def relations_from_centers(objects, *, forward_xy, max_distance_m=15.0,
                           component_margin_m=1.0, near_m=8.0):
    forward = np.asarray(forward_xy, dtype=float)
    if forward.shape != (2,) or not np.isfinite(forward).all() or np.linalg.norm(forward) < 1e-8:
        raise ValueError('方向参考必须为有效二维单位方向')
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0]])
    edges = set()
    for source in objects:
        for target in objects:
            if source['id'] == target['id']:
                continue
            delta = np.asarray(source['center'][:2])-target['center'][:2]
            distance = np.linalg.norm(delta)
            if distance > max_distance_m:
                continue
            if distance <= near_m:
                edges.add((source['id'], 'near', target['id']))
            for relation, component in [('left_of', delta@left), ('right_of', -delta@left),
                                         ('in_front_of', delta@forward), ('behind', -delta@forward)]:
                if component > component_margin_m:
                    edges.add((source['id'], relation, target['id']))
    return edges


def relation_metrics(predicted, truth):
    p, t = set(predicted), set(truth)
    tp = len(p & t)
    precision = tp/len(p) if p else None
    recall = tp/len(t) if t else None
    return dict(true_positive=tp, predicted=len(p), truth=len(t), precision=precision,
                recall=recall, f1=2*tp/(len(p)+len(t)) if p or t else None)
