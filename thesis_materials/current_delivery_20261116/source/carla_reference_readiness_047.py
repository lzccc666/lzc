"""已观测参照物不足时等待/安全搜索；新开发版本，不改旧冻结成绩。"""
from carla_lane_seed_bridge_006 import LaneSeedBridge
from carla_real_graph_bridge_003 import fingerprint, observed_row
from real_graph_contract_003 import reliable
from real_graph_cooperation_002 import C, categories


def reference_readiness(query, row):
    """仅检查实际送给模型的候选图；未观测不等于道路范围内不存在。"""
    if not query.startswith('以') or '为参照' not in query:
        return dict(ready=True, reason='no_explicit_reference_requirement')
    anchor_text = query.split('为参照', 1)[0][1:]
    anchor_categories = {c for c, word in C.items() if word in anchor_text}
    target_categories = categories(query)
    nodes = row['candidate_nodes']
    observed = [n for n in nodes if n['category'] in anchor_categories]
    anchors = [n for n in observed if reliable(n)]
    targets = [n for n in nodes if n['category'] in target_categories and reliable(n)]
    reason = 'reference_and_target_observed'
    if not anchor_categories:
        reason = 'unsupported_reference_category'
    elif not observed:
        reason = 'reference_not_in_observed_input'
    elif not anchors:
        reason = 'reference_evidence_not_current_and_supported'
    elif not any(a['node_id'] != t['node_id'] for a in anchors for t in targets):
        reason = 'no_distinct_supported_target_reference_pair'
    return dict(ready=reason == 'reference_and_target_observed', reason=reason,
        required_reference_categories=sorted(anchor_categories),
        observed_reference_ids=[n['node_id'] for n in observed],
        supported_reference_ids=[n['node_id'] for n in anchors],
        supported_target_ids=[n['node_id'] for n in targets],
        absence_proven=False, GT_used=False)


class ReferenceReadyLaneBridge(LaneSeedBridge):
    """规则/base/Adapter共用门闸；不足时不提交新的模型请求或授权目标驾驶。"""

    def select(self, agent, candidates, edges, state, index):
        row, binding, _, detail = observed_row(self.query, agent.nodes, edges, state, index,
            self.maximum_candidates, getattr(agent, 'road', None))
        readiness = reference_readiness(self.query, row)
        if readiness['ready']:
            final, plans, audit = super().select(agent, candidates, edges, state, index)
            audit['reference_readiness'] = readiness
            return final, plans, audit
        self.intent = None
        self.previous_authorization = None
        self.active = None
        self.hold_for_pending_semantics = False
        self.stop_path(agent)
        audit = dict(mode=self.mode, query_text=self.query, frame_idx=index,
            input_row=row, input_sha256=fingerprint(row), current_binding=binding, pool=detail,
            original_evidence_candidate_order=[c.node_id for c in candidates],
            proposal_semantic_id=None, proposed_node_id=None, authorized_node_id=None,
            current_evidence_passed=False, evidence_check_enabled=self.evidence_check_enabled,
            decision=dict(proposed_node_id=None, authorized_node_id=None,
                source='reference_readiness_gate', reason=readiness['reason'],
                backend_output_used=False),
            plan_check=dict(passed=False, reason='waiting_for_reference_observation'),
            final_order=[], background_request_id=self.pending, active_request_id=None,
            background_result_used=False, events=[],
            actual_model_calls=getattr(self.backend, 'calls', 0),
            reference_readiness=readiness, intent_committed=False,
            hold_for_pending_semantics=False, GT_in_semantic_input=False,
            geometry_or_control_produced_by_Qwen=False,
            ordinary_proposal_is_not_control_authorization=True,
            observation_action='only_independently_checked_bounded_lane_search_allowed',
            actual_CARLA_validation_pending=True)
        return [], {}, audit
