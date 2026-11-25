"""三任务语义与实测对象/车辆候选的桥接；不访问仿真目标或可达性GT。"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
import time

import numpy as np

from conceptgraph.completion.realtime_semantics import RealtimeSemanticQueue
from conceptgraph.completion.semantic_contract import validate
from conceptgraph.completion.stage9 import digest, dump
from conceptgraph.completion.kinematic_benchmark import SPEC
from conceptgraph.completion.planning_reuse import revalidate_route
from conceptgraph.navigation.semantic import apply_qwen_query_parse
from conceptgraph.navigation.vehicle_geometry import audit_executed_states
from conceptgraph.navigation.vehicle_planner import hybrid_astar


TASKS = {'query_parse','object_attributes','candidate_rerank'}


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()


def verified_output(task,envelope,ids=None):
    if not isinstance(envelope,dict) or not envelope.get('used_qwen') or envelope.get('fallback_used'):
        return None
    result = validate(task,envelope.get('raw_output'),allowed_candidate_ids=ids)
    output = result.normalized_output if result.valid else None
    return output if output and output.get('confidence',0)>=.6 else None


class NavigationSemanticBridge:
    """单后台线程；先确认身份/时效，再允许语义参与选择，最终仍由车辆守卫控制。"""

    def __init__(self,query,dest,delegate,*,tasks=TASKS,ttl_s=20.,post_delay_s=0.,
                 maximum_candidates=3,candidate_plan_budget_s=2.,use_plan_reuse=True):
        self.query,self.dest,self.delegate = query,dest,delegate
        self.tasks = set(tasks)
        if not self.tasks<=TASKS or maximum_candidates<1 or candidate_plan_budget_s<=0:
            raise ValueError('invalid semantic bridge budget')
        self.ttl_s,self.post_delay_s = ttl_s,post_delay_s
        self.maximum_candidates = maximum_candidates
        self.candidate_plan_budget_s = candidate_plan_budget_s
        self.use_plan_reuse = use_plan_reuse
        self.plan_cache = {}
        self.queue = RealtimeSemanticQueue(self._backend) if delegate is not None else None
        self.attributes = {}
        self.last_attribute_attempt = {}
        self.ranking_results = []
        self.active_ranking = None
        self.last_rank_signature = None
        self.last_rank_attempt = float('-inf')
        self.applied = dict.fromkeys(sorted(TASKS),0)
        self.events = []
        self.frame_events = []
        if self.queue is not None and 'query_parse' in self.tasks:
            self.queue.submit('query_parse',dict(query_text=query),dict(query_version=1),ttl_s=ttl_s)

    def _backend(self,task,payload):
        if task=='query_parse':
            output = self.delegate.parse_query(payload['query_text'])
        elif task=='object_attributes':
            if digest(payload['crop_paths'][0])!=payload['crop_sha256']:
                raise ValueError('immutable past crop changed before inference')
            output = self.delegate.describe_object(payload['crop_paths'],payload['metadata'],
                object_version=payload['crop_sha256'])
        else:
            output = self.delegate.rerank_candidates(payload['query_text'],payload['candidate_nodes'],
                payload['relation_metadata'],edge_version=payload['context_signature'])
        if self.post_delay_s:
            time.sleep(self.post_delay_s)
        return output

    def _event(self,kind,**extra):
        event = dict(kind=kind,wall_s=time.monotonic(),**extra)
        self.events.append(event)
        self.frame_events.append(event)

    @staticmethod
    def _identity(node):
        return [node.metadata.get('object_identity'),node.object_tag]

    @staticmethod
    def _stop_old_plan(agent):
        agent.machine.reset()
        agent.selected = None
        agent.path = []
        agent.plan_target = None

    def prepare(self,agent,index):
        self.frame_events = []
        binding = dict(query_version=1,**{f'object_{i}':self._identity(n) for i,n in agent.nodes.items()})
        if self.queue is not None:
            for request,envelope in self.queue.poll(binding):
                output = verified_output(request.task,envelope)
                accepted = output is not None
                if accepted and request.task=='query_parse':
                    before = asdict(agent.parsed)
                    before.pop('vehicle_heading_xy',None)
                    parsed = apply_qwen_query_parse(agent.parsed,output,8.)
                    after = asdict(parsed)
                    after.pop('vehicle_heading_xy',None)
                    if before!=after:
                        self._stop_old_plan(agent)
                        self.active_ranking = None
                    agent.parsed = parsed
                    self.applied[request.task] += 1
                elif accepted and request.task=='object_attributes':
                    node_id = request.payload['node_id']
                    self.attributes[node_id] = dict(output=copy.deepcopy(output),
                        identity=request.binding[f'object_{node_id}'],
                        deadline=request.deadline_wall_s,crop_sha256=request.payload['crop_sha256'],
                        crop_paths=request.payload['crop_paths'],submitted_frame=request.payload['frame_idx'])
                    self.applied[request.task] += 1
                elif accepted:
                    # 排序必须在本帧安全候选生成后再次核验；投递不等于应用。
                    self.ranking_results.append((request,envelope))
                self._event('response_received',task=request.task,request_id=request.request_id,
                    schema_and_confidence_passed=accepted,ranking_application_pending=request.task=='candidate_rerank')
                dump(self.dest/'semantic_responses'/f'{request.request_id:05d}.json',
                    dict(request=asdict(request),envelope=envelope,accepted_at_receipt=accepted))
        now = time.monotonic()
        for node_id,saved in list(self.attributes.items()):
            node = agent.nodes.get(node_id)
            if node is None or self._identity(node)!=saved['identity'] or now>saved['deadline']:
                del self.attributes[node_id]
                continue
            output = saved['output']
            # 类别、位置、运动状态仍由检测/多帧几何模块负责。
            visible_colors = [c for c in output.get('colors',[]) if c!='unknown']
            attributes = copy.deepcopy(node.metadata.get('attributes',{}))
            if visible_colors:
                attributes['colors'] = visible_colors
            if output.get('body_type') not in {None,'unknown'}:
                attributes['body_type'] = output['body_type']
            node.metadata['attributes'] = attributes
            node.metadata['qwen_attributes'] = copy.deepcopy(saved)
            node.caption = node.object_tag+' '+' '.join(attributes.get('colors',[]))
            node.metadata['qwen_category_advisory'] = output.get('object_tag')
            node.metadata['qwen_motion_advisory'] = output.get('motion_state')
        if self.queue is not None and 'object_attributes' in self.tasks:
            for node_id,node in sorted(agent.nodes.items()):
                if node.metadata.get('num_detections',0)<3 or now-self.last_attribute_attempt.get(node_id,float('-inf'))<8:
                    continue
                crops = node.metadata.get('crop_paths',[])
                if not crops:
                    continue
                payload = dict(node_id=node_id,crop_paths=crops[:1],crop_sha256=digest(crops[0]),frame_idx=index,
                    metadata=dict(detection_label=node.object_tag,observation_count=node.metadata['num_detections']))
                identity = self.queue.submit('object_attributes',payload,
                    {f'object_{node_id}':self._identity(node)},ttl_s=self.ttl_s)
                if identity is not None:
                    self.last_attribute_attempt[node_id] = now
                break

    def select(self,agent,candidates,edges,state,index):
        before = [c.node_id for c in candidates]
        verified,plans,checks = [],{},[]
        # 对照和消融沿用相同候选安全检查；是否使用Qwen排序是独立开关。
        for candidate in candidates[:self.maximum_candidates]:
            goal,yaw = agent.approach_goal(candidate.node_id)
            plan = None
            reuse_reason = 'no_matching_cached_route'
            saved = self.plan_cache.get(candidate.node_id) if self.use_plan_reuse else None
            identity = self._identity(agent.nodes[candidate.node_id])
            if (saved is not None and saved['identity']==identity
                    and np.linalg.norm(np.asarray(goal)-saved['goal'])<=.5
                    and abs((yaw-saved['yaw']+np.pi)%(2*np.pi)-np.pi)<=np.radians(5)):
                plan,reuse_reason = revalidate_route(state,saved['path'],goal,yaw,
                    agent.road,agent.obstacles,agent.line,SPEC)
            reused = plan is not None
            if plan is None:
                plan = hybrid_astar(state,goal,yaw,agent.road,agent.obstacles,agent.line,SPEC,
                    budget_s=self.candidate_plan_budget_s,max_expanded=2000)
            audit = audit_executed_states(plan.states,agent.road,agent.obstacles,SPEC) if plan.states else None
            passed = bool(plan.states) and plan.failure is None and bool(audit and audit['passed'])
            checks.append(dict(node_id=candidate.node_id,passed=passed,failure=plan.failure,
                duration_s=plan.duration_s,expanded=plan.expanded,independent_sweep_audit=audit,
                route_reused=reused,revalidation_reason=reuse_reason))
            if passed:
                verified.append(candidate)
                plans[candidate.node_id] = plan
                if self.use_plan_reuse and not reused:
                    self.plan_cache[candidate.node_id] = dict(identity=identity,goal=np.asarray(goal).copy(),
                        yaw=yaw,path=[[s.x,s.y] for s in plan.states])
            else:
                self.plan_cache.pop(candidate.node_id,None)
        self.plan_cache = {i:v for i,v in self.plan_cache.items() if i in {c.node_id for c in verified}}
        ids = [c.node_id for c in verified]
        geometry = [dict(node_id=i,identity=self._identity(agent.nodes[i]),
            center_half_m=np.round(np.asarray(agent.nodes[i].bbox_center)*2).astype(int).tolist(),
            motion=agent.nodes[i].motion_state,colors=agent.nodes[i].metadata.get('attributes',{}).get('colors',[]))
            for i in sorted(ids)]
        facts = [dict(source=e.source,target=e.target,relation=e.relation,coordinate_frame=e.coordinate_frame)
            for e in edges if e.source in agent.nodes and e.target in agent.nodes]
        facts.sort(key=lambda e:(e['source'],e['target'],e['relation'],str(e['coordinate_frame'])))
        parsed = asdict(agent.parsed)
        parsed.pop('vehicle_heading_xy',None)
        context = signature(dict(candidates=geometry,relations=facts,query=parsed))
        now = time.monotonic()
        for request,envelope in self.ranking_results:
            output = verified_output('candidate_rerank',envelope,ids)
            accepted = (output is not None and request.payload['context_signature']==context
                and now<=request.deadline_wall_s)
            self._event('ranking_context_checked',request_id=request.request_id,applied=accepted,
                reason='current_safe_evidence_matches' if accepted else 'candidate_or_geometry_or_query_changed')
            if accepted:
                self.active_ranking = dict(output=output,context=context,deadline=request.deadline_wall_s)
                self.applied['candidate_rerank'] += 1
        self.ranking_results.clear()
        if self.active_ranking is not None and (self.active_ranking['context']!=context or now>self.active_ranking['deadline']):
            self._event('ranking_expired',reason='context_changed_or_deadline')
            self.active_ranking = None
            # 旧语义选择授权失效：清除计划后用本帧证据重新确认。
            self._stop_old_plan(agent)
        if self.queue is not None and 'candidate_rerank' in self.tasks and ids and (
                context!=self.last_rank_signature or now-self.last_rank_attempt>=5.):
            payload = dict(query_text=self.query,context_signature=context,
                candidate_nodes=[dict(node_id=i,category=agent.nodes[i].object_tag,
                    colors=agent.nodes[i].metadata.get('attributes',{}).get('colors',[]),
                    center_xy=agent.nodes[i].bbox_center[:2].tolist(),motion_state=agent.nodes[i].motion_state,
                    distance_m=float(np.linalg.norm(agent.nodes[i].bbox_center[:2]-[state.x,state.y])),
                    path_validated=True,safety_evidence='current observed obstacles; forward bicycle plan; independent rectangular sweep audit') for i in ids],
                relation_metadata=facts)
            identity = self.queue.submit('candidate_rerank',payload,dict(query_version=1),ttl_s=self.ttl_s)
            if identity is not None:
                self.last_rank_signature,self.last_rank_attempt = context,now
        if self.active_ranking is not None:
            output = self.active_ranking['output']
            if output['no_match']:
                self._stop_old_plan(agent)
                verified = []
            else:
                lookup = {c.node_id:c for c in verified}
                maximum = max(c.score for c in verified)
                verified = [replace(lookup[i],score=maximum+.5 if i==output['selected_node_id'] else lookup[i].score)
                    for i in output['ranking']]
        return verified,plans,dict(frame_idx=index,enabled_tasks=sorted(self.tasks),
            rule_order=before,safety_verified_order=ids,final_order=[c.node_id for c in verified],
            changed_order=ids!=[c.node_id for c in verified],
            changed_final_first=(ids[0] if ids else None)!=(verified[0].node_id if verified else None),
            candidate_checks=checks,events=self.frame_events[:],applied_counts=dict(self.applied),
            qwen_motion_and_category_are_advisory=True,gt_in_semantic_input=False)

    def close(self,*,timeout_s=10.):
        worker = self.queue.close(timeout_s=timeout_s) if self.queue is not None else None
        return dict(events=self.queue.snapshot_events() if self.queue else [],
            application_events=self.events,worker=worker,applied_counts=self.applied,
            actual_cuda_backend=self.delegate is not None,injected_post_delay_s=self.post_delay_s)
