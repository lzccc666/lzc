"""同一真实前端/增量建图/语义/车辆执行链的可交互开发实验。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import json
import math
import pickle
import shutil
import tarfile
import time

import cv2
import numpy as np
from omegaconf import OmegaConf
from shapely.geometry import LineString, Point, Polygon, MultiPoint
from sklearn.cluster import DBSCAN

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, dump, digest
from conceptgraph.completion.kinematic_benchmark import SEEDS, SPEC, setup, pose_on
from conceptgraph.completion.protocol import GoalTruth, ArrivalEvaluator
from conceptgraph.navigation.semantic import parse_navigation_query, rank_candidate_targets
from conceptgraph.navigation.state_machine import NavigationStateMachine
from conceptgraph.navigation.vehicle_geometry import (
    VehicleState, step_vehicle, swept_body_polygon, body_is_legal, audit_executed_states,
)
from conceptgraph.navigation.vehicle_planner import hybrid_astar, tracking_control
from conceptgraph.scenegraph.incremental_graph import IncrementalSceneGraph


def mapping_config():
    cfg = OmegaConf.load(PROJECT/'conceptgraph/hydra_configs/base_mapping.yaml')
    cfg.merge_with(dict(device='cpu',min_points_threshold=4,obj_min_detections=1,
        dbscan_remove_noise=False,match_method='sim_sum',sim_threshold=.35,phys_bias=.5,
        spatial_sim_type='iou',frame_dt=.5,skip_bg=True,association_policy='one_to_one',
        dynamic_ttl_s=2.,initial_motion_unknown=True,static_confirmation_observations=3,
        depth_policy='largest_supported_cluster',motion_estimator='robust_xy'))
    return cfg


def lidar_obstacles(depth, intrinsic, pose):
    """当前LiDAR实测高于路面点的聚类，不访问环境对象/屏障真值。"""
    v,u = np.nonzero(depth>0)
    z = depth[v,u]
    camera = np.column_stack([(u-intrinsic[0,2])*z/intrinsic[0,0],
                              (v-intrinsic[1,2])*z/intrinsic[1,1],z])
    world = camera@pose[:3,:3].T+pose[:3,3]
    points = world[(world[:,2]>.25)&(world[:,2]<4),:2]
    if len(points)<2:
        return []
    labels = DBSCAN(eps=.8,min_samples=2).fit_predict(points)
    return [MultiPoint(points[labels==label]).convex_hull.buffer(.18)
            for label in sorted(set(labels)) if label>=0]


class PerceptionVehicleAgent:
    """仅持有声明的有向道路先验；没有对象身份、类别、颜色或终点GT。"""

    def __init__(self, query, prior, classes, dest, *, semantic_bridge=None):
        from conceptgraph.slam.streamlined_mapping import StreamlinedMapper
        self.mapper = StreamlinedMapper(mapping_config(),classes)
        self.classes = classes
        self.dest = dest
        self.line = LineString(prior['centerline_xy'])
        self.road = Polygon(prior['road_polygon_xy'])
        self.parsed = parse_navigation_query(query,8)
        self.graph = IncrementalSceneGraph(dict(coordinate_frame='ego_heading',directional_mode='all_supported_axes'))
        self.machine = NavigationStateMachine()
        self.attributes = {}
        self.nodes = {}
        self.obstacles = []
        self.path = []
        self.path_purpose = None
        self.goal = None
        self.selected = None
        self.plan_target = None
        self.guard_latched = False
        self.semantic_bridge = semantic_bridge
        self.safety_update = None
        self.evidence_policy = dict(enabled=True,minimum_observations=3,minimum_lidar_pixels=5,
                                    minimum_depth_support_fraction=.4)

    def perceive(self, index, state, rgb, depth, pose, color, gobs):
        from conceptgraph.completion.interactive_environment import InteractiveRoadEnvironment, observed_mask_color
        from conceptgraph.slam.streamlined_mapping import MappingFrame
        from conceptgraph.scripts.run_incremental_semantic_navigation import mapping_objects_to_nodes
        self.mapper.update(MappingFrame(index,rgb,depth,InteractiveRoadEnvironment.intrinsic,
            pose,gobs,str(color),timestamp_s=state.time_s))
        self.nodes = mapping_objects_to_nodes(self.mapper.objects,self.classes,index,
                                             semantic_crop_dir=self.dest/'actual_crops')
        observed_colors = [observed_mask_color(rgb,mask) for mask in gobs['mask']]
        for node_id,node in self.nodes.items():
            obj = next(o for o in self.mapper.objects if o.get('stable_object_id')==node.metadata['object_identity'])
            indices = np.flatnonzero(np.asarray(obj.get('image_idx',[]))==index)
            if len(indices) and len(gobs['xyxy']):
                box = np.asarray(obj['xyxy'][int(indices[-1])])
                distances = np.linalg.norm(np.asarray(gobs['xyxy'])-box,axis=1)
                best = int(np.argmin(distances))
                if distances[best]<1e-3:
                    color_name,source = observed_colors[best]
                    if color_name!='unknown':
                        self.attributes[node_id] = dict(colors=[color_name],category=node.object_tag,
                            source=source,frame_idx=index,rgb_sha256=digest(color))
            if node_id in self.attributes:
                node.metadata['attributes'] = self.attributes[node_id]
                node.caption = node.object_tag+' '+' '.join(self.attributes[node_id]['colors'])
        heading = np.array([math.cos(state.yaw_rad),math.sin(state.yaw_rad)])
        relations = self.graph.update_snapshot(self.nodes,frame_index=index,ego_heading=heading,
                                                ego_position=np.array([state.x,state.y]))
        if self.semantic_bridge is not None:
            self.semantic_bridge.prepare(self,index)
        self.parsed.vehicle_heading_xy = heading
        candidates = rank_candidate_targets(self.parsed,self.nodes,relations.edges,
            np.array([state.x,state.y]),.55,.3,.25,relation_coordinate_frame='ego_heading',
            relation_coordinate_frame_fallback=None,evidence_policy=self.evidence_policy)
        self.obstacles = lidar_obstacles(depth,InteractiveRoadEnvironment.intrinsic,pose)
        if self.selected is not None and self.selected not in {c.node_id for c in candidates}:
            self.path = []
            self.plan_target = None
        # 真实时钟控制器先接收最新障碍和失效信息，再等待多候选规划。
        if self.safety_update is not None:
            self.safety_update(self)
        prepared_plans = {}
        semantic_audit = None
        if self.semantic_bridge is not None:
            candidates,prepared_plans,semantic_audit = self.semantic_bridge.select(
                self,candidates,relations.edges,state,index)
        pairs = [(c.node_id,c.score) for c in candidates]
        observation = self.machine.observe(index,pairs,valid_target_ids=set(self.nodes))
        self.selected = self.machine.locked_target_id
        if observation.target_lost:
            self.path = []
            self.plan_target = None
        supported_ids = {c.node_id for c in candidates}
        if self.selected is not None and self.selected not in supported_ids:
            # Hysteresis may retain identity; it never authorizes using an unsupported path.
            self.path = []
            self.plan_target = None
        planning = None
        if self.selected is not None and self.selected in supported_ids and self.plan_target!=self.selected:
            goal,yaw = self.approach_goal(self.selected)
            self.goal = goal
            plan = prepared_plans.get(self.selected)
            if plan is None:
                plan = hybrid_astar(state,goal,yaw,self.road,self.obstacles,self.line,SPEC)
            self.path = [[s.x,s.y] for s in plan.states]
            self.path_purpose = 'target_approach'
            self.plan_target = self.selected
            planning = dict(purpose='target_approach',expanded=plan.expanded,duration_s=plan.duration_s,failure=plan.failure,
                            goal_xy=goal.tolist(),goal_yaw_rad=yaw,
                            goal_source='5% real LiDAR longitudinal surface + declared 4.6m parked-car prior - 2m approach offset')
        if self.selected is None and index>=3:
            progress = self.line.project(Point(state.x,state.y))
            search_finished = self.goal is not None and np.linalg.norm(self.goal-[state.x,state.y])<.65
            if (not self.path or search_finished) and progress<self.line.length-6:
                goal,yaw = pose_on(self.line,min(progress+8,self.line.length-5))
                plan = hybrid_astar(state,goal,yaw,self.road,self.obstacles,self.line,SPEC)
                self.goal = goal
                self.path = [[s.x,s.y] for s in plan.states]
                self.path_purpose = 'bounded_lane_search'
                planning = dict(purpose=self.path_purpose,expanded=plan.expanded,duration_s=plan.duration_s,
                    failure=plan.failure,goal_xy=goal.tolist(),goal_yaw_rad=yaw,
                    goal_source='declared directed lane ahead; no object GT, slow bounded search')
        nodes = [dict(node_id=i,category=n.object_tag,center=n.bbox_center.tolist(),
                      extent=n.bbox_extent.tolist(),motion=n.motion_state,
                      metadata=dict(n.metadata,depth_evidence=n.metadata.get('depth_evidence',[])[-1:]))
                 for i,n in self.nodes.items()]
        return dict(frame_idx=index,sim_time_s=state.time_s,nodes=nodes,
            relations=[e.to_dict() if hasattr(e,'to_dict') else asdict(e) for e in relations.edges],
            candidates=[dict(node_id=c.node_id,score=c.score) for c in candidates],
            lock=asdict(observation),selected_node_id=self.selected,planning=planning,
            path_world=self.path,path_purpose=self.path_purpose,parsed_query=asdict(self.parsed),
            known_obstacle_polygons=[list(p.exterior.coords) for p in self.obstacles],
            semantic_audit=semantic_audit,
            gt_sent_to_agent=False)

    def approach_goal(self,node_id):
        node = self.nodes[node_id]
        obj = next(o for o in self.mapper.objects if o.get('stable_object_id')==node.metadata['object_identity'])
        points = np.asarray(obj['pcd'].points)
        # 声明的4.6m静止车辆长度/车道对齐先验；位置只来自实际LiDAR表面。
        progress = np.array([self.line.project(Point(p[:2])) for p in points])
        goal_s = float(np.quantile(progress,.05)+4.6/2-2.0)
        return pose_on(self.line,goal_s)

    def control(self,state):
        if not self.path or self.goal is None or (self.selected is None and self.path_purpose!='bounded_lane_search'):
            return -SPEC.max_braking_mps2,state.steering_rad,dict(reason='waiting_for_evidence_or_plan')
        acceleration,steering,tracking = tracking_control(state,self.path,self.goal,SPEC)
        predicted = step_vehicle(state,acceleration,steering,SPEC)
        braking = predicted
        for _ in range(math.ceil(SPEC.max_speed_mps/SPEC.max_braking_mps2/SPEC.dt_s)+1):
            nxt = step_vehicle(braking,-SPEC.max_braking_mps2,braking.steering_rad,SPEC)
            sweep = swept_body_polygon(braking,nxt,SPEC)
            self.guard_latched |= not self.road.covers(sweep) or any(sweep.intersects(o) for o in self.obstacles)
            braking = nxt
            if braking.speed_mps==0:
                break
        self.guard_latched |= not body_is_legal(predicted,self.road,self.obstacles,SPEC)
        if self.guard_latched:
            return -SPEC.max_braking_mps2,state.steering_rad,dict(tracking,emergency_braking=True)
        return acceleration,steering,tracking


def evaluation_identity(nodes, targets, selected):
    """隔离评估：类别及2.5m中心门限的一对一关联，未知保留。"""
    from scipy.optimize import linear_sum_assignment
    ids = sorted(nodes)
    if selected is None or not ids:
        return None,{}
    costs = np.array([[np.linalg.norm(nodes[i].bbox_center[:2]-t['center_xy'])
                      if nodes[i].object_tag==t['category'] else 1e6 for t in targets] for i in ids])
    rows,cols = linear_sum_assignment(costs)
    matched = {ids[i]:targets[j]['instance_id'] for i,j in zip(rows,cols) if costs[i,j]<=2.5}
    return matched.get(selected),matched


def json_safe(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def run_task(config,task,seed,dest,models,frontend_cfg,classes,max_sim_s,recovery_wait_s=0.):
    from conceptgraph.completion.interactive_environment import InteractiveRoadEnvironment
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    xy = np.asarray(config['start_xy'])+rng.normal(0,.03,2)
    state = VehicleState(*xy,config['start_yaw_rad']+rng.normal(0,math.radians(.5)))
    env = InteractiveRoadEnvironment(config,task,seed,state)
    # This allowlist is the only environment metadata exposed to the agent.
    agent = PerceptionVehicleAgent(task['query_text'],{k:config[k] for k in
        ['centerline_xy','road_polygon_xy']},classes,dest)
    _,road,truth_obstacles,proof = setup(config,task)
    truth = GoalTruth(task['task_id'],tuple(task['accepted_instance_ids']),tuple(task['regions']),
        config['config_id'],(0,90),'冻结合成对象身份及独立区域；不向agent传递',
        reachability='reachable' if task['kind']=='reachable_positive' else
            'inaccessible' if task['kind']=='inaccessible' else 'unknown')
    evaluator = ArrivalEvaluator(truth)
    states = [state]
    correct_lock = False
    correct_selection = False
    lock_streak = 0
    max_lock_streak = 0
    any_lock = False
    routes = 0
    search_routes = 0
    actual_rgb_changes = set()
    actual_depth_changes = set()
    started = time.monotonic()
    final = 'TIMEOUT'
    pending_termination = None
    unavailable_since = None
    evaluated_id = None
    try:
        from conceptgraph.scripts.streamlined_detections import detect_rgb_file
        with (dest/'perception.jsonl').open('x') as perception, (dest/'execution.jsonl').open('x') as execution:
            braking_ticks = math.ceil(SPEC.max_speed_mps/SPEC.max_braking_mps2/SPEC.dt_s)+2
            for tick in range(int(max_sim_s/SPEC.dt_s)+braking_ticks):
                if tick%5==0:
                    index = tick//5
                    rgb,depth,pose,color,sensor = env.observe(dest,index)
                    actual_rgb_changes.add(sensor['rgb_sha256'])
                    actual_depth_changes.add(sensor['depth_sha256'])
                    t = time.monotonic()
                    gobs,vis = detect_rgb_file(frontend_cfg,models,color,classes)
                    detpath = dest/'detections'/f'{index:06d}.pkl.gz'
                    detpath.parent.mkdir(exist_ok=True)
                    with gzip.open(detpath,'xb') as f:
                        pickle.dump(gobs,f)
                    (dest/'detection_vis').mkdir(exist_ok=True)
                    cv2.imwrite(str(dest/'detection_vis'/f'{index:06d}.jpg'),vis)
                    record = agent.perceive(index,state,rgb,depth,pose,color,gobs)
                    evaluated_id,matches = evaluation_identity(agent.nodes,config['targets'],agent.selected)
                    candidate_ids = [c['node_id'] for c in record['candidates']]
                    top_id,_ = evaluation_identity(agent.nodes,config['targets'],
                                                  candidate_ids[0] if candidate_ids else None)
                    correct_selection |= top_id in task['accepted_instance_ids']
                    any_lock |= agent.selected is not None
                    lock_streak = lock_streak+1 if evaluated_id in task['accepted_instance_ids'] else 0
                    max_lock_streak = max(max_lock_streak,lock_streak)
                    correct_lock |= lock_streak>=3
                    if record['planning'] and record['path_world']:
                        routes += int(record['planning']['purpose']=='target_approach')
                        search_routes += int(record['planning']['purpose']=='bounded_lane_search')
                    record.update(frontend_mapping_duration_s=time.monotonic()-t,
                                  evaluation_only_identity=evaluated_id,evaluation_only_matches=matches,
                                  sensor=sensor,detector_sha256=digest(detpath))
                    perception.write(json.dumps(record,ensure_ascii=False,allow_nan=False,default=json_safe)+'\n')
                    perception.flush()
                acceleration,steering,tracking = agent.control(state)
                if state.time_s>=max_sim_s and pending_termination is None:
                    pending_termination = 'TIMEOUT'
                if pending_termination is not None:
                    acceleration,steering = -SPEC.max_braking_mps2,state.steering_rad
                    tracking = dict(tracking,terminal_braking_reason=pending_termination)
                old = state
                state = env.act(acceleration,steering)
                states.append(state)
                sweep = swept_body_polygon(old,state,SPEC)
                collision = any(sweep.intersects(o) for o in truth_obstacles)
                violation = not road.covers(sweep)
                arrival = evaluator.update(time_s=state.time_s,xy=[state.x,state.y],yaw_rad=state.yaw_rad,
                    speed_mps=state.speed_mps,evaluated_instance_id=evaluated_id,
                    body_collision=collision,lane_violation=violation)
                execution.write(json.dumps(dict(state=asdict(state),control=dict(acceleration_mps2=acceleration,
                    steering_command_rad=steering),tracking=tracking,independent_arrival=arrival,
                    body_collision=collision,lane_violation=violation),ensure_ascii=False,allow_nan=False)+'\n')
                if collision or violation:
                    final = 'EXECUTION_FAILED'
                    break
                if arrival['actually_arrived'] and pending_termination is None:
                    final = 'ARRIVED'
                    break
                if agent.guard_latched and state.speed_mps==0:
                    final = 'SAFETY_STOP'
                    break
                if pending_termination is None:
                    unavailable_since = state.time_s if not agent.path and unavailable_since is None else unavailable_since
                    if agent.path:
                        unavailable_since = None
                    exhausted_wait = unavailable_since is not None and state.time_s-unavailable_since>=recovery_wait_s
                    if state.time_s>=6 and not agent.path and agent.selected is not None and exhausted_wait:
                        pending_termination = 'PERCEPTION_OR_PLAN_LOST' if routes else 'PLANNING_FAILED'
                    elif state.time_s>=8 and agent.selected is None and not agent.path and exhausted_wait:
                        pending_termination = 'INSUFFICIENT_EVIDENCE_IN_OBSERVED_SCOPE'
                    elif agent.selected is None and agent.line.project(Point(state.x,state.y))>=agent.line.length-6 and state.speed_mps<.15:
                        pending_termination = 'SEARCH_EXHAUSTED_WITHOUT_VERIFIED_MATCH'
                if pending_termination is not None and state.speed_mps==0:
                    final = pending_termination
                    break
    finally:
        env.close()
    audit = audit_executed_states(states,road,truth_obstacles,SPEC)
    dump(dest/'independent_execution_audit.json',audit)
    if not audit['passed']:
        final = 'EXECUTION_FAILED'
    return dict(task_id=task['task_id'],config_id=config['config_id'],kind=task['kind'],seed=seed,
        method='actual_frontend_rule_semantics_hybrid_astar',status=final,route_generated=routes>0,
        route_count=routes,search_route_count=search_routes,selected_object_evaluated_correct=correct_selection,stable_correct_lock=correct_lock,
        max_consecutive_independently_correct_locked_observations=max_lock_streak,
        any_lock=any_lock,actual_full_perception_arrival=final=='ARRIVED',
        actual_kinematic_oracle=False,gt_in_agent=False,real_yolo_sam_clip=True,
        actions_changed_rgb=len(actual_rgb_changes)>1,actions_changed_lidar=len(actual_depth_changes)>1,
        independent_disconnection_proof=proof,independent_execution_audit=audit,
        final_state=asdict(state),wall_duration_s=time.monotonic()-started,
        terminal_vehicle_stopped=state.speed_mps<=.15,terminal_zero_speed=state.speed_mps==0,
        low_evidence_stopped_recovery_wait_s=recovery_wait_s,
        simulation_clock='fixed 0.1s bicycle integration; perception every 0.5s simulated time',
        real_clock_async_qwen_evaluated=False,scene_color_truth='actual mesh material; deterministic RGB baseline only',
        limitations=['简单合成静态环境；没有轮胎物理或动态道路泛化证据',
            'parked-car 4.6m长度/与车道对齐先验；身份以独立中心关联核验',
            '当前为规则闭环；真实时钟异步Qwen尚未接入'])


def main():
    from conceptgraph.completion.interactive_environment import ASSET_SHA
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--task',action='append')
    parser.add_argument('--seed',type=int,action='append')
    parser.add_argument('--max-sim-s',type=float,default=70)
    parser.add_argument('--max-wall-s',type=float,default=5400)
    parser.add_argument('--recovery-wait-s',type=float,default=0.)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    dest = OUTPUT/'interactive_simulation'/args.run_id
    dest.mkdir(parents=True,exist_ok=args.resume)
    releasepath = OUTPUT/'interactive_simulation/synthetic_tasks_v1.json'
    release = json.loads(releasepath.read_text())
    tasks = [t for t in release['tasks'] if not args.task or t['task_id'] in args.task]
    configs = {c['config_id']:c for c in release['configs']}
    sources = sorted((PROJECT/'conceptgraph').rglob('*.py'))
    frozen = dict(task_release_sha256=digest(releasepath),asset_sha256=ASSET_SHA,
        task_ids=[t['task_id'] for t in tasks],seeds=args.seed or SEEDS,max_sim_s=args.max_sim_s,
        max_wall_s=args.max_wall_s,minimum_free_storage_bytes=400*1024**2,
        recovery_wait_s=args.recovery_wait_s,
        source_sha256={str(p.relative_to(PROJECT)):digest(p) for p in sources},
        gt_assisted=False,real_clock_async_qwen_evaluated=False,development_run=True)
    if args.resume:
        if json.loads((dest/'release.json').read_text())!=frozen:
            raise ValueError('resume requires unchanged source, tasks and budgets')
    else:
        dump(dest/'release.json',frozen)
        with tarfile.open(dest/'source_snapshot.tar.gz','w:gz') as archive:
            for path in sources:
                archive.add(path,arcname=str(path.relative_to(PROJECT)))
    from conceptgraph.scripts.streamlined_detections import load_detection_models
    frontend_cfg = OmegaConf.create(dict(device='cuda:0'))
    models = load_detection_models(frontend_cfg)
    classes = [s.strip() for s in (PROJECT/'conceptgraph/vehicle_classes.txt').read_text().splitlines() if s.strip()]
    results = [json.loads(s) for s in (dest/'results.jsonl').read_text().splitlines()] if args.resume and (dest/'results.jsonl').exists() else []
    completed = {(r['task_id'],r['seed']) for r in results}
    if len(completed)!=len(results):
        raise ValueError('duplicate completed trials')
    started = time.monotonic()
    for task in tasks:
        for seed in args.seed or SEEDS:
            if (task['task_id'],seed) in completed:
                continue
            if time.monotonic()-started>args.max_wall_s or shutil.disk_usage(dest).free<400*1024**2:
                dump(dest/f'budget_stop_{time.time_ns()}.json',dict(completed_trials=len(results),
                    reason='wall clock or storage budget',goal_complete=False))
                return
            case = dest/task['task_id']/f'seed{seed}'
            if case.exists():
                case = case/f'resume_attempt_{time.time_ns()}'
            case.mkdir(parents=True)
            try:
                result = run_task(configs[task['config_id']],task,seed,case,models,frontend_cfg,classes,args.max_sim_s,args.recovery_wait_s)
            except Exception as exc:
                dump(case/'failure.json',dict(error=repr(exc),goal_complete=False))
                raise
            dump(case/'result.json',result)
            results.append(result)
            with (dest/'results.jsonl').open('a') as f:
                f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
            print(json.dumps(result,ensure_ascii=False),flush=True)
    dump(dest/'summary.json',dict(trials=len(results),tasks=len(tasks),
        configs=len({t['config_id'] for t in tasks}),results=results,
        actual_full_perception_arrivals=sum(r['actual_full_perception_arrival'] for r in results),
        stage11_complete=False,goal_complete=False))


if __name__=='__main__':
    main()
