"""把已训练的真实图协议接入异步驾驶选择；语言提议与当前控制授权分开。"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
from shapely.geometry import Point

from conceptgraph.completion.realtime_semantics import RealtimeSemanticQueue
from conceptgraph.navigation.types import CandidateTarget, ParsedNavigationQuery
from real_graph_contract_003 import build_completion_prompt, reliable, validate
from real_graph_cooperation_002 import categories, choose, complete_geometry_rule


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def observed_row(query, nodes, edges, state, index, maximum_candidates=8, road=None):
    """只转换当前预测节点与边；没有环境对象、目标actor或答案入口。"""
    forward = np.array([math.cos(state.yaw_rad), math.sin(state.yaw_rad)])
    left = np.array([-forward[1], forward[0]])
    requested = categories(query)
    translated = []
    origins = {}
    for identity, node in sorted(nodes.items()):
        if '本车道' in query and road is not None and node.object_tag in requested and not road.covers(Point(node.bbox_center[:2])):
            continue
        metadata = node.metadata
        delta = np.asarray(node.bbox_center[:2]) - [state.x, state.y]
        latest = (metadata.get('depth_evidence') or [{}])[-1]
        # version对应产生该条观测证据的帧，不能拿当前帧覆盖旧观测的时间。
        version = int(metadata.get('last_seen_frame', -1)) + 1
        seen = metadata.get('timestamp_s')
        number = 100 + int(identity)
        origins[number] = int(identity)
        translated.append(dict(node_id=number, category=node.object_tag,
            center_xy=[round(float(delta @ forward), 3), round(float(delta @ left), 3)],
            distance_m=round(float(np.linalg.norm(delta)), 3), motion_state=node.motion_state,
            version=version, now_s=float(state.time_s),
            last_seen_s=float(seen) if seen is not None else -1.,
            evidence=dict(source=latest.get('source', 'unknown'), version=version,
                observations=int(metadata.get('num_detections', 0)),
                lidar_pixels=int(latest.get('retained_pixels', 0)),
                support_fraction=float(latest.get('support_fraction', 0)),
                conflict=not bool(latest.get('sufficient_support', False)))))
    pool = sorted(translated, key=lambda n: (n['category'] not in requested,
        n['distance_m'], n['node_id']))[:maximum_candidates]
    # 打乱展示顺序，但三种方法使用完全相同的输入；不使用正确性挑选对象。
    pool.sort(key=lambda n: fingerprint(['carla-online-pool-v1', n['node_id']]))
    kept = {n['node_id'] for n in pool}
    facts = []
    for edge in edges:
        edge = edge.to_dict() if hasattr(edge, 'to_dict') else asdict(edge)
        source, target = 100 + int(edge['source']), 100 + int(edge['target'])
        if source in kept and target in kept:
            facts.append(dict(source=source, target=target, relation=edge['relation']))
    facts.sort(key=lambda e: (e['source'], e['target'], e['relation']))
    facts = facts[:16]
    row = dict(task='candidate_rerank', sample_id=f'carla_observation_{index:06d}',
        query_text=query, candidate_nodes=pool, candidate_ids=[n['node_id'] for n in pool],
        relation_metadata=facts)
    geometry = [dict(node_id=n['node_id'],
        identity=nodes[origins[n['node_id']]].metadata.get('object_identity'),
        category=n['category'], motion=n['motion_state'],
        world_half_m=np.round(np.asarray(nodes[origins[n['node_id']]].bbox_center) * 2).astype(int).tolist())
        for n in sorted(pool, key=lambda n: n['node_id'])]
    ordering = sorted(pool, key=lambda n: (n['category'], n['distance_m'], n['node_id']))
    binding = dict(context=fingerprint(dict(query=query, geometry=geometry, relations=facts,
        distance_order=[n['node_id'] for n in ordering],
        ego_half_m=np.round(np.array([state.x, state.y])*2).astype(int).tolist(),
        ego_yaw_5deg=int(round(state.yaw_rad/math.radians(5))))))
    details = dict(total_observed_nodes=len(translated), retained_nodes=len(pool),
        candidate_cap=maximum_candidates, candidate_pool_truncated=len(pool)<len(translated),
        id_translation={str(n['node_id']): origins[n['node_id']] for n in pool},
        provenance='source/pixels/support/observation_count/timestamp来自当前mapper，未人工补足',
        position_frame='current ego heading; x forward, y left; metre')
    return row, binding, origins, details


class ActualQwenBackend:
    """单个真实CUDA模型；不截断、不回退、不缓存模型响应。"""

    def __init__(self, model_path, adapter_path, dest, *, seed=17):
        import ctypes
        ctypes.CDLL('/root/miniconda3/envs/conceptgraph/lib/libstdc++.so.6', mode=ctypes.RTLD_GLOBAL)
        import torch
        from transformers import AutoProcessor
        from scripts.train_qwen_lora_semantic_delegate_v2 import _load_model
        from conceptgraph.completion.stage9 import digest
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('actual bf16 CUDA required')
        torch.manual_seed(seed)
        self.torch = torch
        self.model = _load_model(Path(model_path), Path(adapter_path) if adapter_path else None).eval()
        self.model.config.use_cache = True
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True,
                                                       trust_remote_code=True)
        self.dest = Path(dest)
        self.dest.mkdir(exist_ok=False, parents=True)
        self.calls = 0
        self.log = (self.dest/'calls.jsonl').open('x')
        binding = dict(model=str(model_path), adapter=str(adapter_path) if adapter_path else None,
            adapter_sha256=digest(Path(adapter_path)/'adapter_model.safetensors') if adapter_path else None,
            max_input_tokens=2048, max_new_tokens=256, dtype='bfloat16', seed=seed,
            do_sample=False, response_cache=False, fallback=False, input_truncation=False,
            actual_CUDA=True, goal_complete=False)
        (self.dest/'binding.json').write_text(json.dumps(binding, ensure_ascii=False, indent=2))

    def __call__(self, task, row):
        if task != 'candidate_rerank':
            raise ValueError('rank-only Adapter cannot execute parsing/attribute tasks')
        prompt = build_completion_prompt(task, row)
        messages = [dict(role='user', content=[dict(type='text', text=prompt)])]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        x = self.processor(text=[text], images=None, padding=False, return_tensors='pt').to(self.model.device)
        if x['input_ids'].shape[-1] > 2048:
            raise ValueError('refuse >2048 tokens; no truncation')
        self.torch.cuda.synchronize()
        start = time.monotonic()
        with self.torch.inference_mode():
            y = self.model.generate(**x, max_new_tokens=256, do_sample=False, temperature=None,
                top_p=None, top_k=None, use_cache=True, pad_token_id=self.processor.tokenizer.eos_token_id)
        self.torch.cuda.synchronize()
        raw = self.processor.tokenizer.decode(y[0, x['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
        result = validate(task, raw, allowed_candidate_ids=row['candidate_ids'])
        self.calls += 1
        record = dict(call=self.calls, payload=row, payload_sha256=fingerprint(row),
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            input_tokens=int(x['input_ids'].shape[-1]), output_tokens=int(y.shape[-1]-x['input_ids'].shape[-1]),
            GPU_latency_s=time.monotonic()-start, raw_output=raw, schema_valid=result.valid,
            validation_errors=result.validation_errors, actual_CUDA=True, fallback_used=False)
        self.log.write(json.dumps(record, ensure_ascii=False)+'\n')
        self.log.flush()
        return raw

    def close(self):
        self.log.close()


class RealGraphNavigationBridge:
    """相同候选与控制门闸的规则/base/Adapter三种选择；模型在后台工作。"""

    def __init__(self, query, mode, *, backend=None, ttl_s=20., maximum_candidates=8,
                 clock=time.monotonic, planner=None, evidence_check_enabled=True):
        if mode not in {'rule', 'base', 'adapter'}:
            raise ValueError(mode)
        if (mode != 'rule') != (backend is not None):
            raise ValueError('base/adapter require an actual or explicitly injected backend')
        self.query, self.mode, self.clock = query, mode, clock
        self.ttl_s, self.maximum_candidates = ttl_s, maximum_candidates
        self.queue = RealtimeSemanticQueue(backend, maximum_queued=1, clock=clock) if backend else None
        self.backend = backend
        self.planner = planner
        self.evidence_check_enabled = bool(evidence_check_enabled)
        self.active = None
        self.pending = None
        self.plans = {}
        self.events = []
        self.applied = 0

    def prepare(self, agent, index):
        # 原rank函数只提供通用证据候选，不能先按复杂指令剔除干扰物。
        agent.parsed = ParsedNavigationQuery(raw_text='最近的静止目标', target_text='目标',
            anchor_text=None, relation=None, query_mode='nearest_static')

    @staticmethod
    def stop_path(agent):
        # 未获目标授权只撤销目标接近路线；有独立安全检查的搜索路线继续使用。
        if getattr(agent, 'path_purpose', None) == 'bounded_lane_search':
            return
        agent.path = []
        agent.plan_target = None

    def current_plan(self, agent, identity, state):
        if self.planner is not None:
            return self.planner(agent, identity, state)
        import carla_agent_isolated_002 as nav
        from conceptgraph.completion.planning_reuse import revalidate_route
        from conceptgraph.navigation.vehicle_geometry import audit_executed_states
        from conceptgraph.navigation.vehicle_planner import hybrid_astar
        goal, yaw = agent.approach_goal(identity)
        saved = self.plans.get(identity)
        plan, reason = None, 'no_matching_route'
        object_identity = agent.nodes[identity].metadata.get('object_identity')
        if (saved and saved['identity'] == object_identity
                and np.linalg.norm(np.asarray(goal)-saved['goal']) <= .5
                and abs((yaw-saved['yaw']+np.pi)%(2*np.pi)-np.pi) <= np.radians(5)):
            plan, reason = revalidate_route(state, saved['path'], goal, yaw, agent.road,
                agent.obstacles, agent.line, nav.SPEC, budget_s=1.)
        reused = plan is not None
        if plan is None:
            plan = hybrid_astar(state, goal, yaw, agent.road, agent.obstacles, agent.line,
                                nav.SPEC, budget_s=10., max_expanded=6000)
        audit = audit_executed_states(plan.states, agent.road, agent.obstacles, nav.SPEC) if plan.states else None
        passed = bool(plan.states) and plan.failure is None and bool(audit and audit['passed'])
        if passed:
            self.plans[identity] = dict(identity=object_identity, goal=np.asarray(goal).copy(), yaw=yaw,
                path=[[s.x, s.y] for s in plan.states])
        else:
            self.plans.pop(identity, None)
        return plan if passed else None, dict(passed=passed, route_reused=reused,
            revalidation_reason=reason, failure=plan.failure, independent_sweep_audit=audit)

    def select(self, agent, candidates, edges, state, index):
        row, binding, origins, detail = observed_row(self.query, agent.nodes, edges, state,
                                                    index, self.maximum_candidates, getattr(agent,'road',None))
        now = self.clock()
        events = []
        if self.queue is not None:
            for request, raw in self.queue.poll(binding):
                v = validate('candidate_rerank', raw, allowed_candidate_ids=row['candidate_ids'])
                events.append(dict(kind='delivered_schema_checked', request_id=request.request_id,
                    valid=v.valid, errors=v.validation_errors))
                if v.valid:
                    self.active = dict(output=v.normalized_output, binding=binding,
                                       deadline=request.deadline_wall_s, request_id=request.request_id)
                else:
                    self.active = None
            if self.pending is not None:
                terminal = {'backend_completed', 'backend_exception', 'expired_in_queue', 'cancelled_before_backend'}
                if any(e['request_id'] == self.pending and e['kind'] in terminal
                       for e in self.queue.snapshot_events()):
                    self.pending = None
        if self.active and (self.active['binding'] != binding or now > self.active['deadline']):
            events.append(dict(kind='active_result_revoked', reason='changed_context_or_expired'))
            self.active = None
            self.stop_path(agent)
        clear = not any(x in self.query for x in ['参照', '第二', '不要'])
        policy = 'full' if self.evidence_check_enabled else 'without_evidence_check'
        pre = choose(self.query, row['candidate_nodes'], row['relation_metadata'], None, condition=policy)
        if self.mode == 'rule':
            decision = complete_geometry_rule(self.query, row['candidate_nodes'], row['relation_metadata'], evidence_check_enabled=self.evidence_check_enabled)
        elif clear or not pre['backend_output_used']:
            decision = pre
        else:
            prediction = dict(schema_valid=True, normalized_output=self.active['output']) if self.active else None
            decision = choose(self.query, row['candidate_nodes'], row['relation_metadata'], prediction, condition=policy)
            if self.active is None and self.pending is None:
                self.pending = self.queue.submit('candidate_rerank', row, binding, ttl_s=self.ttl_s)
                events.append(dict(kind='submitted', request_id=self.pending,
                                   prompt_sha256=hashlib.sha256(build_completion_prompt('candidate_rerank', row).encode()).hexdigest()))
        proposed = decision['proposed_node_id']
        selected = origins.get(proposed)
        node = next((n for n in row['candidate_nodes'] if n['node_id'] == proposed), None)
        original = agent.nodes.get(selected)
        latest_depth = ((original.metadata.get('depth_evidence') or [{}])[-1]
                        if original is not None else {})
        strict_evidence_pass = bool(node is not None and reliable(node) and original is not None
            and not original.is_short_term and latest_depth.get('units') == 'm')
        evidence_pass = strict_evidence_pass if self.evidence_check_enabled else bool(node is not None and original is not None)
        plan, check = None, dict(passed=False, reason='no_current_supported_proposal')
        if selected is not None and evidence_pass:
            plan, check = self.current_plan(agent, selected, state)
        authorized = selected if plan is not None and evidence_pass else None
        if authorized is None or (agent.selected is not None and agent.selected != authorized):
            self.stop_path(agent)
        final = [CandidateTarget(node_id=authorized, score=.95, text_score=.95,
            relation_score=0., distance_prior=0., reachable=True)] if authorized is not None else []
        if self.active is not None and authorized is not None and decision['backend_output_used']:
            self.applied += 1
        audit = dict(mode=self.mode, query_text=self.query, frame_idx=index, input_row=row,
            input_sha256=fingerprint(row), current_binding=binding, pool=detail,
            original_evidence_candidate_order=[c.node_id for c in candidates],
            proposal_semantic_id=proposed, proposed_node_id=selected,
            current_evidence_passed=strict_evidence_pass, evidence_check_enabled=self.evidence_check_enabled, authorized_node_id=authorized,
            decision=decision, plan_check=check, final_order=[c.node_id for c in final],
            background_request_id=self.pending,
            active_request_id=self.active['request_id'] if self.active else None,
            background_result_used=bool(self.active and decision['backend_output_used']),
            events=events, actual_model_calls=getattr(self.backend, 'calls', None),
            injected_backend=self.backend is not None and not isinstance(self.backend, ActualQwenBackend),
            active_proposal_reuse=True, backend_response_cache=False,
            GT_in_semantic_input=False, geometry_or_control_produced_by_Qwen=False,
            ordinary_proposal_is_not_control_authorization=True)
        self.events.extend(events)
        return final, {authorized: plan} if authorized is not None else {}, audit

    def close(self, timeout_s=10.):
        worker = self.queue.close(timeout_s=timeout_s) if self.queue else None
        # 没结束的后台推理仍可能写日志，不能提前关闭文件。
        if hasattr(self.backend, 'close') and worker and worker['worker_stopped']:
            self.backend.close()
        return dict(worker=worker, events=self.queue.snapshot_events() if self.queue else [],
            application_events=self.events, authorized_application_frames=self.applied,
            actual_model_calls=getattr(self.backend, 'calls', None), goal_complete=False)
