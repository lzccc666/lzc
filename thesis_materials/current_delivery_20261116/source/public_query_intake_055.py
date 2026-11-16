"""公开数据指令的受限句法入口；只读文字，不读标准答案或官方对象。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re

from real_graph_contract_003 import reliable

CATEGORIES = dict(car='小汽车', truck='卡车', bus='公交车', pedestrian='行人',
                  bicycle='自行车', motorcycle='摩托车')
DIRECTIONS = dict(左侧='left_of', 右侧='right_of', 前方='in_front_of', 后方='behind')
CONDITIONS = {'full', 'without_rule_query_resolution', 'without_attribute_check',
              'without_candidate_reranking', 'without_evidence_check'}


@dataclass(frozen=True)
class Query:
    targets: tuple[str, ...]
    anchors: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    relation: str | None = None
    requires_static: bool = False
    explicit_authorization: bool = False
    prefer_nearest: bool = False
    ordinal: int = 1
    supported: bool = True
    reason: str = 'supported_restricted_Chinese_query'


def mentioned(text):
    return tuple(k for k, word in CATEGORIES.items() if word in text)


def parse(text):
    """兼容已声明的公开句式和九类开发句式，不使用expression_family字段。"""
    target_text = text
    anchor_text = ''
    direction = None
    # 先隔离否定类别，避免把“而不是公交车”作为另一类候选。
    positive, _, excluded_text = text.partition('而不是')
    target_text = positive
    if positive.startswith('目标是在'):
        match = re.fullmatch(r'目标是在(.+?)(左侧|右侧|前方|后方)的(.+?)，请接近它', positive)
        if match is None:
            return Query((), supported=False, reason='unrecognized_relation_syntax')
        anchor_text, direction, target_text = match.groups()
    elif positive.startswith('以'):
        match = re.fullmatch(r'以(.+?)为参照，选择其(左侧|右侧|前方|后方)(.+)', positive)
        if match is None:
            return Query((), supported=False, reason='unrecognized_relation_syntax')
        anchor_text, direction, target_text = match.groups()
    elif any(word in positive for word in DIRECTIONS):
        return Query((), supported=False, reason='unrecognized_relation_syntax')
    targets = mentioned(target_text)
    anchors = mentioned(anchor_text)
    supported = bool(targets) and (not direction or len(anchors) == 1)
    # 未经核验的颜色/细车型不在此受限类别任务的承诺范围内。
    if any(word in text for word in ['红色', '蓝色', '白色', '黑色', '橙色', '面包车', '轿车']):
        return Query(targets, supported=False, reason='unsupported_fine_attribute')
    return Query(targets=targets, anchors=anchors, excluded=mentioned(excluded_text),
                 relation=DIRECTIONS.get(direction), requires_static='静止' in text,
                 explicit_authorization='授权' in text or '证据可靠' in text,
                 prefer_nearest='最近' in text, ordinal=2 if '第二近' in text else 1,
                 supported=supported,
                 reason='supported_restricted_Chinese_query' if supported else 'no_unambiguous_target_category')


def pool_for(query, nodes, *, attribute_check=True, evidence_check=True):
    """所有条件只使用预测节点；不修改调用者的节点或几何。"""
    if not query.supported:
        return []
    allowed = list(nodes)
    if attribute_check:
        allowed = [n for n in allowed if n['category'] in query.targets
                   and n['category'] not in query.excluded
                   and (not query.requires_static or n['motion_state'] == 'static')]
    if query.explicit_authorization and evidence_check:
        allowed = [n for n in allowed if reliable(n)]
    return allowed


def reference_available(query, nodes):
    if not query.anchors:
        return True
    anchors = [n for n in nodes if n['category'] in query.anchors]
    targets = pool_for(query, nodes, evidence_check=False)
    return any(a['node_id'] != t['node_id'] for a in anchors for t in targets)


def finish(query, nodes, proposal, *, source, evidence_check, backend_used=False, reason='resolved'):
    chosen = next((n for n in nodes if n['node_id'] == proposal), None)
    passed = chosen is not None and reliable(chosen)
    return dict(proposed_node_id=proposal,
                authorized_node_id=proposal if chosen is not None and (passed or not evidence_check) else None,
                valid=query.supported, parsed_query=asdict(query), source=source, reason=reason,
                backend_output_used=backend_used,
                explicit_authorization_query=query.explicit_authorization,
                evidence_check_enabled=evidence_check, current_evidence_passed=passed,
                ordinary_proposal_does_not_authorize_actuation=True)


def choose(text, nodes, prediction=None, condition='full'):
    if condition not in CONDITIONS:
        raise ValueError('unknown ablation condition')
    query = parse(text)
    evidence = condition != 'without_evidence_check'
    allowed = pool_for(query, nodes, attribute_check=condition != 'without_attribute_check',
                       evidence_check=evidence)
    if not allowed:
        return finish(query, nodes, None, source='no_candidate', evidence_check=evidence,
                      reason=query.reason if not query.supported else 'no_compatible_observed_candidate')
    clear = not query.relation and query.ordinal == 1
    if clear and condition != 'without_rule_query_resolution':
        proposal = min(allowed, key=lambda n: (n['distance_m'], n['node_id']))['node_id']
        return finish(query, nodes, proposal, source='clear_rule', evidence_check=evidence)
    if condition == 'without_candidate_reranking':
        return finish(query, nodes, allowed[0]['node_id'], source='fixed_input_order', evidence_check=evidence)
    if prediction is None:
        return finish(query, nodes, None, source='backend_required', evidence_check=evidence,
                      backend_used=True, reason='requires_actual_model_response')
    proposal = None
    if prediction.get('schema_valid') and prediction.get('normalized_output'):
        candidate = prediction['normalized_output']['selected_node_id']
        if candidate in {n['node_id'] for n in allowed}:
            proposal = candidate
    return finish(query, nodes, proposal, source='actual_model_output', evidence_check=evidence,
                  backend_used=True, reason='validated_model_output_or_abstention')


def strong_rule(text, nodes, *, relation_radius_m):
    """有完整方向几何的强规则对照，不能称作外部发表方法。"""
    if not math.isfinite(relation_radius_m) or relation_radius_m <= 0:
        raise ValueError('relation radius must be positive and finite')
    query = parse(text)
    allowed = pool_for(query, nodes)
    if query.relation:
        anchors = [n for n in nodes if n['category'] in query.anchors]
        axis = 1 if query.relation in {'left_of', 'right_of'} else 0
        sign = 1 if query.relation in {'left_of', 'in_front_of'} else -1
        allowed = [t for t in allowed if any(
            a['node_id'] != t['node_id']
            and sign * (t['center_xy'][axis] - a['center_xy'][axis]) > 1
            and math.dist(t['center_xy'], a['center_xy']) <= relation_radius_m
            for a in anchors)]
    allowed.sort(key=lambda n: (n['distance_m'], n['node_id']))
    selected = allowed[query.ordinal - 1]['node_id'] if len(allowed) >= query.ordinal else None
    return finish(query, nodes, selected, source='restricted_complete_geometry_rule', evidence_check=True)
