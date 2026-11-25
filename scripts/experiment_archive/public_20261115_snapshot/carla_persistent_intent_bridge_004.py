"""语言结果须及时投递；已确认的目标身份由新观测持续核验，不因自车前进重选。"""
from __future__ import annotations

import numpy as np

from carla_real_graph_bridge_003 import (ActualQwenBackend, RealGraphNavigationBridge,
    fingerprint, observed_row)
from conceptgraph.navigation.types import CandidateTarget
from real_graph_contract_003 import reliable
from real_graph_cooperation_001 import categories


class PersistentIntentBridge(RealGraphNavigationBridge):
    """规则/base/Adapter共用同一身份锁定；全部驾驶授权仍基于本帧证据及路线。"""

    def __init__(self,*args,maximum_intent_shift_m=1.5,**kwargs):
        super().__init__(*args,**kwargs)
        self.maximum_intent_shift_m=float(maximum_intent_shift_m)
        if not np.isfinite(self.maximum_intent_shift_m) or self.maximum_intent_shift_m<=0:
            raise ValueError('positive finite identity drift limit required')
        self.intent=None
        self.previous_authorization=None
        self.hold_for_pending_semantics=False

    def current_evidence(self,agent,row,identity):
        original=agent.nodes.get(identity)
        node=next((n for n in row['candidate_nodes'] if n['node_id']==100+identity),None)
        if original is None or node is None:
            return False,'observed_identity_missing'
        requested=categories(self.query)
        if requested and node['category'] not in requested:
            return False,'current_category_mismatch'
        latest=(original.metadata.get('depth_evidence') or [{}])[-1]
        if self.evidence_check_enabled and (not reliable(node) or original.is_short_term or latest.get('units')!='m'):
            return False,'current_static_depth_or_age_check_failed'
        if not original.metadata.get('object_identity'):
            return False,'missing_stable_object_identity'
        return True,'current_static_depth_identity_supported'

    def select(self,agent,candidates,edges,state,index):
        row,binding,origins,detail=observed_row(self.query,agent.nodes,edges,state,index,
                                               self.maximum_candidates,getattr(agent,'road',None))
        events=[]
        # observe()在上一帧select返回后才形成LOCKED；只提升已经真实锁定的授权。
        last=self.previous_authorization
        machine=getattr(agent,'machine',None)
        if self.intent is None and last is not None and machine is not None:
            locked=getattr(machine,'locked_target_id',None)
            good,_=self.current_evidence(agent,row,last['node_id'])
            original=agent.nodes.get(last['node_id'])
            if (locked==last['node_id'] and good and last['query']==self.query
                    and original.metadata.get('object_identity')==last['object_identity']):
                self.intent=dict(last,committed_wall_s=self.clock())
                events.append(dict(kind='identity_intent_committed',node_id=locked,
                    source=last['source'],backend_request_id=last['backend_request_id']))
        if self.intent is not None:
            intent=self.intent
            identity=intent['node_id']
            original=agent.nodes.get(identity)
            reason=None
            if intent['query']!=self.query:reason='query_changed'
            elif original is None:reason='identity_missing'
            elif original.metadata.get('object_identity')!=intent['object_identity']:reason='identity_replaced'
            elif original.object_tag!=intent['category']:reason='category_changed'
            elif np.linalg.norm(np.asarray(original.bbox_center[:2])-intent['center_xy'])>self.maximum_intent_shift_m:
                reason='static_identity_position_drift'
            elif self.evidence_check_enabled and (original.metadata.get('timestamp_s') is None or state.time_s-float(original.metadata['timestamp_s'])>2.):
                reason='observation_expired'
            if reason is not None:
                events.append(dict(kind='identity_intent_revoked',node_id=identity,reason=reason))
                self.intent=None
                self.previous_authorization=None
                self.active=None
                self.stop_path(agent)
        if self.intent is None:
            final,plans,audit=super().select(agent,candidates,edges,state,index)
            self.hold_for_pending_semantics=bool(self.pending is not None and self.active is None
                and any(not categories(self.query) or n['category'] in categories(self.query)
                        for n in row['candidate_nodes']))
            if final:
                identity=final[0].node_id
                original=agent.nodes[identity]
                self.previous_authorization=dict(node_id=identity,
                    object_identity=original.metadata.get('object_identity'),
                    category=original.object_tag,center_xy=np.asarray(original.bbox_center[:2]).copy(),
                    query=self.query,source=audit['decision']['source'],
                    backend_request_id=audit['active_request_id'],
                    based_on_model=bool(audit['background_result_used']))
            audit.update(intent_committed=False,hold_for_pending_semantics=self.hold_for_pending_semantics,
                additional_identity_events=events,model_result_delivery_deadline_unchanged=True)
            self.events.extend(events)
            return final,plans,audit
        intent=self.intent
        identity=intent['node_id']
        passed,reason=self.current_evidence(agent,row,identity)
        plan,check=None,dict(passed=False,reason=reason)
        if passed:plan,check=self.current_plan(agent,identity,state)
        authorized=identity if passed and plan is not None else None
        if authorized is None:self.stop_path(agent)
        final=[CandidateTarget(node_id=identity,score=.95,text_score=.95,
            relation_score=0.,distance_prior=0.,reachable=True)] if authorized is not None else []
        self.hold_for_pending_semantics=False
        audit=dict(mode=self.mode,query_text=self.query,frame_idx=index,input_row=row,
            input_sha256=fingerprint(row),current_binding=binding,pool=detail,
            original_evidence_candidate_order=[c.node_id for c in candidates],
            proposal_semantic_id=100+identity,proposed_node_id=identity,
            current_evidence_passed=passed,evidence_check_enabled=self.evidence_check_enabled,authorized_node_id=authorized,
            decision=dict(proposed_node_id=100+identity,authorized_node_id=100+identity if passed else None,
                source='persistent_observed_identity',reason=reason,
                backend_output_used=False,original_intent_from_model=intent['based_on_model']),
            plan_check=check,final_order=[c.node_id for c in final],
            background_request_id=self.pending,active_request_id=None,background_result_used=False,
            intent_committed=True,intent_origin_request_id=intent['backend_request_id'],
            intent_age_wall_s=self.clock()-intent['committed_wall_s'],
            intent_identity=intent['object_identity'],hold_for_pending_semantics=False,
            intent_reobserved_and_replanned=True,events=events,
            actual_model_calls=getattr(self.backend,'calls',None),
            injected_backend=self.backend is not None and not isinstance(self.backend,ActualQwenBackend),
            active_proposal_reuse=False,backend_response_cache=False,
            GT_in_semantic_input=False,geometry_or_control_produced_by_Qwen=False,
            ordinary_proposal_is_not_control_authorization=True,
            model_result_delivery_deadline_unchanged=True)
        self.events.extend(events)
        return final,{identity:plan} if authorized is not None else {},audit


def hold_search_until_semantic_delivery(agent,result):
    """当前有可供判断的对象且模型仍在工作时停车观察；不阻塞后台模型。"""
    bridge=agent.semantic_bridge
    suppressed=bool(bridge.hold_for_pending_semantics
        and agent.path_purpose=='bounded_lane_search')
    if suppressed:
        agent.path=[]
        agent.path_purpose=None
        agent.plan_target=None
        agent.goal=None
        result['path_world']=[]
        result['path_purpose']=None
    result['pending_semantics_search_suppressed']=suppressed
    result['waiting_policy']='base/Adapter相同：当前对象可判断且模型待投递时停车；无观测对象时仍允许有界搜索'
    return result
