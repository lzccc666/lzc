"""实际墙钟控制线程与真实感知主链；可独立切换三个后台语义任务。"""
from __future__ import annotations

import argparse
import copy
import gzip
from dataclasses import asdict
import json
import math
from pathlib import Path
import pickle
import tarfile
from types import SimpleNamespace
import threading
import time

import numpy as np
from omegaconf import OmegaConf
from shapely import from_wkb, to_wkb
from shapely.geometry import Point

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, dump, digest
from conceptgraph.completion.interactive_navigation import (
    PerceptionVehicleAgent, evaluation_identity, json_safe,
)
from conceptgraph.completion.kinematic_benchmark import SPEC, setup
from conceptgraph.completion.protocol import ArrivalEvaluator, GoalTruth
from conceptgraph.completion.realtime_semantics import RealtimeSemanticQueue
from conceptgraph.completion.navigation_semantics import NavigationSemanticBridge, TASKS
from conceptgraph.navigation.semantic import apply_qwen_query_parse
from conceptgraph.navigation.vehicle_geometry import VehicleState, step_vehicle, audit_executed_states


class WallClockVehicle:
    """控制独立于渲染/前端/语义；过期观测只能制动，不延长感知有效期。"""

    def __init__(self, state, path, *, sensor_ttl_s=1.5):
        if not math.isfinite(sensor_ttl_s) or sensor_ttl_s<=0:
            raise ValueError('sensor TTL must be positive')
        self.state = state
        self.path = path
        self.sensor_ttl_s = sensor_ttl_s
        self.started = time.monotonic()
        self.packet = None
        self._lock = threading.Lock()
        self._state_changed = threading.Condition(self._lock)
        self._stop = threading.Event()
        self.states = [state]
        self.records = []
        self.error = None
        self.guard_latched = False
        self._thread = threading.Thread(target=self._guard_run, name='wall-clock-vehicle', daemon=True)
        self._thread.start()

    def snapshot(self):
        with self._lock:
            return self.state

    def snapshot_after(self, previous_time_s, *, timeout_s=.5):
        """感知只采集真实心跳推进后的状态；不为重复观测伪造时间。"""
        if previous_time_s is not None and not math.isfinite(previous_time_s):
            raise ValueError('previous observation time must be finite')
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError('state wait timeout must be finite and nonnegative')
        with self._state_changed:
            ready = self._state_changed.wait_for(lambda:
                previous_time_s is None or self.state.time_s > previous_time_s or
                self._stop.is_set() or self.error is not None, timeout=timeout_s)
            if not ready or self.error is not None or self._stop.is_set():
                return None
            return self.state

    def publish(self, agent, sensed_at):
        if not math.isfinite(sensed_at) or sensed_at > time.monotonic()+.01:
            raise ValueError('传感器时间必须为当前或过去的真实单调时钟')
        # prep()会原地修改GEOS prepared状态。执行器拥有独立几何，不能与
        # 主线程的规划器共享同一个road/obstacle底层对象。
        # WKB重建保留坐标/孔洞但不携带prepared缓存；不改变几何安全条件。
        packet = SimpleNamespace(path=copy.deepcopy(agent.path),
            goal=np.array(agent.goal).copy() if agent.goal is not None else None,
            selected=agent.selected,path_purpose=agent.path_purpose,
            road=from_wkb(to_wkb(agent.road)),
            obstacles=tuple(from_wkb(to_wkb(o)) for o in agent.obstacles),
            guard_latched=agent.guard_latched)
        with self._lock:
            self.packet = (packet, sensed_at)

    def _guard_run(self):
        try:
            self._run()
        except Exception as exc:
            with self._state_changed:
                self.error = repr(exc)
                self._state_changed.notify_all()

    def _run(self):
        deadline = self.started + SPEC.dt_s
        with self.path.open('x') as handle:
            while True:
                delay = deadline - time.monotonic()
                if delay > 0:
                    # 停止请求也必须完成真实制动，不能通过set event跳过积分。
                    time.sleep(delay)
                now = time.monotonic()
                with self._lock:
                    state, bundle = self.state, self.packet
                age = None if bundle is None else now - bundle[1]
                stale = bundle is None or age > self.sensor_ttl_s
                stopping = self._stop.is_set()
                if stale or stopping or self.guard_latched:
                    acceleration, steering = -SPEC.max_braking_mps2, state.steering_rad
                    tracking = dict(reason='stop_requested' if stopping else
                        'safety_guard_latched' if self.guard_latched else 'sensor_packet_expired')
                else:
                    acceleration, steering, tracking = PerceptionVehicleAgent.control(bundle[0], state)
                    self.guard_latched |= bundle[0].guard_latched
                updated = step_vehicle(state, acceleration, steering, SPEC)
                row = dict(state=asdict(updated),wall_elapsed_s=now-self.started,
                    wake_lateness_s=max(0.,now-deadline),sensor_age_s=age,
                    stale_sensor_braking=stale,stop_requested=stopping,
                    control=dict(acceleration_mps2=acceleration,steering_command_rad=steering),tracking=tracking)
                with self._lock:
                    self.state = updated
                    self.states.append(updated)
                    self.records.append(row)
                    self._state_changed.notify_all()
                handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                handle.flush()
                if stopping and updated.speed_mps == 0:
                    break
                # 记录调度延迟，不用瞬间补算多个旧控制步冒充真实频率。
                deadline = max(deadline+SPEC.dt_s,time.monotonic())

    def stop(self):
        self._stop.set()
        with self._state_changed:
            self._state_changed.notify_all()
        self._thread.join(timeout=5.)
        if self._thread.is_alive():
            raise RuntimeError('车辆控制线程未在制动预算内停止')
        if self.error is not None:
            raise RuntimeError('车辆控制线程异常：'+self.error)


def run_task(config, task, seed, dest, models, frontend_cfg, classes, delegate,
             *, maximum_wall_s=55., semantic_ttl_s=20., injected_post_delay_s=0.,
             semantic_tasks=('query_parse',),maximum_candidates=3,candidate_plan_budget_s=2.,
             sensor_ttl_s=1.5,verified_candidate_chain=True,use_plan_reuse=True):
    from conceptgraph.scripts.streamlined_detections import detect_rgb_file
    from conceptgraph.completion.interactive_environment import InteractiveRoadEnvironment
    rng = np.random.default_rng(seed)
    state = VehicleState(*(np.asarray(config['start_xy'])+rng.normal(0,.03,2)),
        config['start_yaw_rad']+rng.normal(0,math.radians(.5)))
    env = InteractiveRoadEnvironment(config,task,seed,state)
    bridge = None
    # 保留首轮解析实验的旧入口；完整三任务及其消融使用同一确定性安全候选链。
    full_bridge = verified_candidate_chain
    if not full_bridge and set(semantic_tasks)!={'query_parse'}:
        raise ValueError('legacy chain only supports the original parse-only experiment')
    if full_bridge:
        bridge = NavigationSemanticBridge(task['query_text'],dest,delegate,tasks=semantic_tasks,
            ttl_s=semantic_ttl_s,post_delay_s=injected_post_delay_s,
            maximum_candidates=maximum_candidates,candidate_plan_budget_s=candidate_plan_budget_s,
            use_plan_reuse=use_plan_reuse)
    agent = PerceptionVehicleAgent(task['query_text'],{k:config[k] for k in
        ['centerline_xy','road_polygon_xy']},classes,dest,semantic_bridge=bridge)
    def backend(task_name, payload):
        if task_name != 'query_parse':
            raise ValueError('该首轮真实时钟实验只启用指令解析')
        result = delegate.parse_query(payload['query_text'])
        if injected_post_delay_s:
            time.sleep(injected_post_delay_s)
        return result
    queue = RealtimeSemanticQueue(backend) if delegate is not None and not full_bridge else None
    if queue is not None:
        queue.submit('query_parse',dict(query_text=task['query_text']),dict(query_version=1),ttl_s=semantic_ttl_s)
    _, road, obstacles, proof = setup(config,task)
    truth = GoalTruth(task['task_id'],tuple(task['accepted_instance_ids']),tuple(task['regions']),
        config['config_id'],(0,90),'冻结独立身份与区域；不进入控制器',
        reachability='reachable' if task['kind']=='reachable_positive' else
            'inaccessible' if task['kind']=='inaccessible' else 'unknown')
    evaluator = ArrivalEvaluator(truth)
    driver = WallClockVehicle(state,dest/'execution.jsonl',sensor_ttl_s=sensor_ttl_s)
    evaluated = 0
    current_identity = None
    identities = []
    correct_selection = False
    correct_lock = False
    lock_streak = 0
    routes = 0
    semantic_applied = 0
    final = 'WALL_CLOCK_TIMEOUT'
    started = time.monotonic()
    sensor_hashes = set()
    try:
        with (dest/'perception.jsonl').open('x') as handle:
            index = 0
            previous_observation_time_s = None
            while time.monotonic()-started < maximum_wall_s:
                state = driver.snapshot_after(previous_observation_time_s)
                if state is None:
                    if driver.error is not None:
                        final = 'EXECUTION_FAILED'
                        break
                    continue
                sensed_at = time.monotonic()
                previous_observation_time_s = state.time_s
                env.state = state
                rgb,depth,pose,color,sensor = env.observe(dest,index)
                sensor_hashes.add((sensor['rgb_sha256'],sensor['depth_sha256']))
                if queue is not None:
                    for request, envelope in queue.poll(dict(query_version=1)):
                        applied = bool(envelope.get('used_qwen') and not envelope.get('fallback_used')
                            and envelope.get('output_schema_valid') and
                            (envelope.get('normalized_output') or {}).get('confidence',0)>=.6)
                        if applied:
                            agent.parsed = apply_qwen_query_parse(agent.parsed,envelope['normalized_output'],8)
                            semantic_applied += 1
                        dump(dest/'semantic_responses'/f'{request.request_id:05d}.json',
                            dict(request=asdict(request),envelope=envelope,applied=applied))
                gobs,vis = detect_rgb_file(frontend_cfg,models,color,classes)
                import cv2
                detpath = dest/'detections'/f'{index:06d}.pkl.gz'
                detpath.parent.mkdir(exist_ok=True)
                with gzip.open(detpath,'xb') as output:
                    pickle.dump(gobs,output)
                (dest/'detection_vis').mkdir(exist_ok=True)
                cv2.imwrite(str(dest/'detection_vis'/f'{index:06d}.jpg'),vis)
                agent.safety_update = lambda current:driver.publish(current,sensed_at)
                record = agent.perceive(index,state,rgb,depth,pose,color,gobs)
                # 先审计从上次感知到本次感知之间的实际执行，再更新身份对应。
                with driver._lock:
                    states = driver.states[:]
                for before, after in zip(states[evaluated:],states[evaluated+1:]):
                    from conceptgraph.navigation.vehicle_geometry import swept_body_polygon
                    swept = swept_body_polygon(before,after,SPEC)
                    arrived = evaluator.update(time_s=after.time_s,xy=[after.x,after.y],yaw_rad=after.yaw_rad,
                        speed_mps=after.speed_mps,evaluated_instance_id=current_identity,
                        body_collision=any(swept.intersects(o) for o in obstacles),lane_violation=not road.covers(swept))
                    if arrived['actually_arrived']:
                        final = 'ARRIVED'
                    elif not arrived['safe_so_far']:
                        final = 'EXECUTION_FAILED'
                evaluated = len(states)-1
                current_identity,matches = evaluation_identity(agent.nodes,config['targets'],agent.selected)
                identities.append(dict(wall_elapsed_s=sensed_at-driver.started,
                    observed_state_time_s=state.time_s,independent_identity=current_identity))
                top = record['candidates'][0]['node_id'] if record['candidates'] else None
                top_identity,_ = evaluation_identity(agent.nodes,config['targets'],top)
                correct_selection |= top_identity in task['accepted_instance_ids']
                lock_streak = lock_streak+1 if current_identity in task['accepted_instance_ids'] else 0
                correct_lock |= lock_streak>=3
                routes += bool(record['planning'] and record['path_world'] and record['path_purpose']=='target_approach')
                record.update(sensor=sensor,sensed_wall_elapsed_s=sensed_at-driver.started,
                    completed_wall_elapsed_s=time.monotonic()-driver.started,
                    detector_sha256=digest(detpath),
                    evaluation_only_identity=current_identity,evaluation_only_matches=matches)
                handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False,default=json_safe)+'\n')
                handle.flush()
                driver.publish(agent,sensed_at)
                if final in {'ARRIVED','EXECUTION_FAILED'}:
                    break
                if (agent.guard_latched or driver.guard_latched) and driver.snapshot().speed_mps==0:
                    final = 'SAFETY_STOP'
                    break
                if agent.selected is None and agent.line.project(Point(state.x,state.y))>=agent.line.length-6:
                    final = 'SEARCH_EXHAUSTED_WITHOUT_VERIFIED_MATCH'
                    break
                index += 1
    finally:
        driver.stop()
        # 在销毁EGL/CUDA渲染资源前等待本任务后台调用结束；不让存活线程进入下个任务。
        queue_status = queue.close(timeout_s=45.) if queue is not None else None
        semantic_report = bridge.close(timeout_s=45.) if bridge is not None else dict(events=queue.snapshot_events() if queue else [],
            worker=queue_status,actual_cuda_backend=delegate is not None,injected_post_delay_s=injected_post_delay_s)
        worker=semantic_report.get('worker')
        if worker and not worker['worker_stopped']:
            dump(dest/'cleanup_failure.json',dict(reason='model_worker_alive_after_45s',
                model_worker=worker,vehicle_stopped=driver.snapshot().speed_mps==0,
                rendering_resource_not_deleted_while_worker_alive=True,goal_complete=False))
            raise RuntimeError('semantic worker did not stop; abort this process before another scene')
        env.close()
    audit = audit_executed_states(driver.states,road,obstacles,SPEC)
    if not audit['passed']:
        final = 'EXECUTION_FAILED'
    events = queue.snapshot_events() if queue is not None else []
    dump(dest/'semantic_events.json',semantic_report)
    if bridge is not None:
        semantic_applied = sum(bridge.applied.values())
        events = semantic_report['events']
    dump(dest/'independent_execution_audit.json',audit)
    result = dict(task_id=task['task_id'],config_id=config['config_id'],kind=task['kind'],seed=seed,
        status=final,route_generated=routes>0,selected_object_evaluated_correct=correct_selection,
        stable_correct_lock=correct_lock,actual_full_perception_arrival=final=='ARRIVED',
        final_state=asdict(driver.snapshot()),independent_execution_audit=audit,
        wall_duration_s=time.monotonic()-started,actual_control_ticks=len(driver.records),
        controller_wake_lateness_s_p95=float(np.percentile([r['wake_lateness_s'] for r in driver.records],95)),
        stale_sensor_braking_ticks=sum(r['stale_sensor_braking'] for r in driver.records),
        semantic_applied=semantic_applied,semantic_events=len(events),real_clock_async=True,
        semantic_tasks=sorted(semantic_tasks),semantic_applied_by_task=(dict(bridge.applied) if bridge else
            dict(query_parse=semantic_applied,object_attributes=0,candidate_rerank=0)),
        gt_in_agent=False,actions_changed_sensor=len(sensor_hashes)>1,disconnection_proof=proof,
        limitations=(['三任务闭环开发实验；尚非完整冻结主对比/消融验收'] if full_bridge else
            ['首轮只启用后台Qwen解析；属性和多候选重排闭环未验收'])+[
            '0.1s控制心跳独立运行；调度延迟单列，不宣称硬实时',
            '静态mesh和低速自行车模型；未证明动态交通安全'])
    dump(dest/'result.json',result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--model',choices=['rule','base','new'],default='rule')
    parser.add_argument('--adapter-training-run',default='three_task_v2_formal_seed17_001')
    parser.add_argument('--task',action='append',required=True)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--maximum-wall-s',type=float,default=55.)
    parser.add_argument('--semantic-ttl-s',type=float,default=20.)
    parser.add_argument('--injected-post-delay-s',type=float,default=0.)
    parser.add_argument('--semantic-tasks',default='query_parse,object_attributes,candidate_rerank')
    parser.add_argument('--legacy-parse-only-chain',action='store_true')
    parser.add_argument('--maximum-candidates',type=int,default=3)
    parser.add_argument('--candidate-plan-budget-s',type=float,default=2.)
    parser.add_argument('--sensor-ttl-s',type=float,default=1.5)
    parser.add_argument('--disable-plan-reuse',action='store_true')
    args = parser.parse_args()
    semantic_tasks = set(filter(None,args.semantic_tasks.split(',')))
    if not semantic_tasks<=TASKS:
        raise ValueError('unknown semantic task')
    if args.legacy_parse_only_chain and semantic_tasks!={'query_parse'}:
        raise ValueError('legacy chain requires exactly query_parse')
    dest = OUTPUT/'interactive_simulation'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    release_path = OUTPUT/'interactive_simulation/synthetic_tasks_v1.json'
    release = json.loads(release_path.read_text())
    tasks = [t for t in release['tasks'] if t['task_id'] in args.task]
    if len(tasks)!=len(set(args.task)):
        raise ValueError('unknown or repeated task')
    configs = {c['config_id']:c for c in release['configs']}
    dump(dest/'release.json',dict(arguments=vars(args),task_release_sha256=digest(release_path),
        source_sha256={str(p.relative_to(PROJECT)):digest(p) for p in (PROJECT/'conceptgraph').rglob('*.py')},
        development=True,independent_new_test=False,real_clock=True,
        only_parse_qwen=semantic_tasks=={'query_parse'},same_safety_filter_all_models=not args.legacy_parse_only_chain))
    with tarfile.open(dest/'source_snapshot.tar.gz','x:gz') as archive:
        for path in sorted((PROJECT/'conceptgraph').rglob('*.py')):
            archive.add(path,arcname=str(path.relative_to(PROJECT)))
    from conceptgraph.scripts.streamlined_detections import load_detection_models
    frontend_cfg = OmegaConf.create(dict(device='cuda:0'))
    models = load_detection_models(frontend_cfg)
    classes = [s.strip() for s in (PROJECT/'conceptgraph/vehicle_classes.txt').read_text().splitlines() if s.strip()]
    delegate = None
    if args.model != 'rule':
        from conceptgraph.vlm.qwen_semantic_delegate import QwenDelegateConfig,QwenSemanticDelegate
        from conceptgraph.completion.semantic_evaluation import base_revision
        adapter = None
        if args.model == 'new':
            report = json.loads((OUTPUT/'training'/args.adapter_training_run/'training_report.json').read_text())
            adapter = Path(report['best_checkpoint']['path'])
        delegate = QwenSemanticDelegate(QwenDelegateConfig(enabled=True,
            model_name_or_path=str(PROJECT/'checkpoint/Qwen2-VL-2B-Instruct'),adapter_path=adapter,
            model_revision=base_revision(),adapter_revision=digest(adapter/'adapter_model.safetensors') if adapter else None,
            device='cuda:0',prompt_profile='completion_v3',max_new_tokens=256,torch_dtype='bfloat16',
            use_qwen_query_parse='query_parse' in semantic_tasks,
            use_qwen_object_attributes='object_attributes' in semantic_tasks,
            use_qwen_rerank='candidate_rerank' in semantic_tasks,
            timeout_s=30,cache_dir=OUTPUT/'cache_audit'/args.run_id/'cache',log_dir=dest/'qwen_audit'))
    results = []
    try:
        for task in tasks:
            case = dest/task['task_id']/f'seed{args.seed}'
            case.mkdir(parents=True)
            result = run_task(configs[task['config_id']],task,args.seed,case,models,frontend_cfg,classes,delegate,
                maximum_wall_s=args.maximum_wall_s,semantic_ttl_s=args.semantic_ttl_s,
                injected_post_delay_s=args.injected_post_delay_s,semantic_tasks=semantic_tasks,
                maximum_candidates=args.maximum_candidates,candidate_plan_budget_s=args.candidate_plan_budget_s,
                sensor_ttl_s=args.sensor_ttl_s,verified_candidate_chain=not args.legacy_parse_only_chain,
                use_plan_reuse=not args.disable_plan_reuse)
            results.append(result)
            print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        if delegate is not None:
            delegate.close()
    dump(dest/'summary.json',dict(results=results,real_clock_async_evaluated=True,
        full_semantic_navigation_evaluation_complete=False,stage11_complete=False,goal_complete=False))


if __name__=='__main__':
    main()
