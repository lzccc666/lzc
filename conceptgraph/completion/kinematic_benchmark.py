"""给定正确对象的独立运动学执行诊断；不能计为感知导航到达。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import heapq
import inspect
import json
import math
from pathlib import Path
import tarfile
import time

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from conceptgraph.completion.protocol import GoalTruth, ArrivalEvaluator
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, dump, digest
from conceptgraph.navigation.vehicle_geometry import (
    VehicleSpec, VehicleState, step_vehicle, body_is_legal, swept_body_polygon,
    footprint_polygon, audit_executed_states,
)
from conceptgraph.navigation.vehicle_planner import hybrid_astar, tracking_control

SPEC=VehicleSpec()
SEEDS=[17,29,43]


def road_points(index):
    x=np.linspace(-6,52,291)
    shape=index%3
    if shape==0:
        y=np.zeros_like(x)
    elif shape==1:
        y=.004*(x-5)**2
    else:
        y=1.4*np.sin((x-5)/12)
    angle=math.radians((index//3-1)*12)
    return np.column_stack([x,y])@np.array([[math.cos(angle),math.sin(angle)],[-math.sin(angle),math.cos(angle)]])


def pose_on(line,distance):
    p=line.interpolate(distance)
    a=line.interpolate(max(0,distance-.1))
    b=line.interpolate(min(line.length,distance+.1))
    return np.array([p.x,p.y]),math.atan2(b.y-a.y,b.x-a.x)


def rectangle(xy,yaw,length,width):
    c,s=math.cos(yaw),math.sin(yaw)
    corners=np.array([[-length/2,-width/2],[length/2,-width/2],[length/2,width/2],[-length/2,width/2]])
    return Polygon(corners@np.array([[c,s],[-s,c]])+xy)


def make_release():
    tasks=[]
    configs=[]
    for index in range(10):
        line=LineString(road_points(index))
        start_s=6.0
        origin,heading=pose_on(line,start_s)
        targets=[]
        for j,color in enumerate(['red','blue','white']):
            goal,goal_yaw=pose_on(line,start_s+18+7*j)
            tangent=np.array([math.cos(goal_yaw),math.sin(goal_yaw)])
            left=np.array([-tangent[1],tangent[0]])
            position=goal+2*tangent+3.2*left
            targets.append(dict(instance_id=f'config{index:02d}-car{j}',category='car',color=color,
                center_xy=position.tolist(),yaw_rad=goal_yaw,length_m=4.6,width_m=1.85,
                goal_xy=goal.tolist(),goal_heading_rad=goal_yaw))
        road=line.buffer(1.8,cap_style=2,join_style=1)
        road_reference_valid=True
        distances=np.linspace(5,line.length-5,160)
        headings=[]
        for distance in distances:
            xy,yaw=pose_on(line,float(distance))
            headings.append(yaw)
            road_reference_valid &= road.covers(footprint_polygon(VehicleState(*xy,yaw),SPEC))
        curvature=np.gradient(np.unwrap(headings),distances)
        steering=np.arctan(SPEC.wheelbase_m*curvature)
        steering_rate=np.gradient(steering,distances)*SPEC.max_speed_mps
        reference_valid=bool(road_reference_valid and max(abs(curvature))<=1/SPEC.minimum_turn_radius_m
            and max(abs(steering_rate))<=SPEC.max_steering_rate_radps)
        if not reference_valid:
            raise ValueError('independent reference vehicle feasibility failed before task freeze')
        config=dict(config_id=f'config{index:02d}',centerline_xy=np.asarray(line.coords).tolist(),
            road_polygon_xy=list(road.exterior.coords),start_xy=origin.tolist(),start_yaw_rad=heading,
            targets=targets,shared_generator_family=index%3,
            independent_reference_body_contained=bool(road_reference_valid),
            independent_reference_vehicle_feasible=reference_valid,
            reference_max_curvature_per_m=float(max(abs(curvature))),
            reference_max_steering_rate_at_max_speed_radps=float(max(abs(steering_rate))),
            topology='单向单车道，没有变道、倒车、逆行或未声明连接')
        configs.append(config)
        for j,target in enumerate(targets):
            region=rectangle(np.array(target['goal_xy']),target['goal_heading_rad'],2.4,.9)
            tasks.append(dict(task_id=f'config{index:02d}-positive{j}',config_id=config['config_id'],
                kind='reachable_positive',query_text=f"去{dict(red='红色',blue='蓝色',white='白色')[target['color']]}小汽车附近",
                accepted_instance_ids=[target['instance_id']],target_index=j,
                regions=[dict(instance_id=target['instance_id'],polygon_xy=list(region.exterior.coords),
                    heading_rad=target['goal_heading_rad'])],
                reachability_provenance='独立有向中心线参考通过整车包含、曲率与转角变化上界检查；不由待评测规划器成功决定',
                obstacle_mode='parked_targets'))
        tasks.append(dict(task_id=f'config{index:02d}-no_match',config_id=config['config_id'],kind='no_match',
            query_text='去公交车附近',accepted_instance_ids=[],regions=[],target_index=None,
            reachability_provenance='冻结场景仅三个car，无bus，范围为整个当前合成配置',obstacle_mode='parked_targets'))
        tasks.append(dict(task_id=f'config{index:02d}-inaccessible',config_id=config['config_id'],kind='inaccessible',
            query_text='去白色小汽车附近',accepted_instance_ids=[targets[2]['instance_id']],
            target_index=2,regions=[],reachability_provenance='独立几何连通分量证明全宽实体屏障切断道路，非本规划器失败',
            obstacle_mode='full_width_barrier'))
    return dict(version='kinematic-oracle-and-synthetic-simulation-v1',vehicle=asdict(SPEC),seeds=SEEDS,
        configs=configs,tasks=tasks,task_count=len(tasks),config_count=len(configs),
        task_counts=dict(reachable_positive=30,no_match=10,inaccessible=10),
        generator_sha256=__import__('hashlib').sha256(''.join(inspect.getsource(f) for f in
            [road_points,pose_on,rectangle,make_release]).encode()).hexdigest(),
        related_config_disclosure='10个不同参数配置，来自3种共享走廊生成家族；不是10座独立城市或真实道路',
        kinematic_condition='给定正确对象/类别集合的GT辅助诊断；不计真实感知导航成绩',
        full_perception_minimum_arrivals_credit=0)


def setup(config,task):
    line=LineString(config['centerline_xy'])
    road=Polygon(config['road_polygon_xy'])
    obstacles=[rectangle(np.array(t['center_xy']),t['yaw_rad'],t['length_m'],t['width_m']) for t in config['targets']]
    proof=None
    if task['kind']=='inaccessible':
        xy,yaw=pose_on(line,6+12)
        barrier=rectangle(xy,yaw,2,100)
        obstacles.append(barrier)
        pieces=road.difference(unary_union(obstacles))
        components=list(pieces.geoms) if hasattr(pieces,'geoms') else [pieces]
        start=Point(config['start_xy'])
        goal=Point(config['targets'][task['target_index']]['goal_xy'])
        proof=not any(p.covers(start) and p.covers(goal) for p in components)
        if not proof:
            raise AssertionError('independent disconnection proof failed')
    return line,road,obstacles,proof


def point_astar(start,goal,road,obstacles,reference,resolution=.4):
    started=time.perf_counter()
    bounds=road.bounds
    origin=np.array(bounds[:2])
    shape=np.ceil((np.array(bounds[2:])-origin)/resolution).astype(int)+1
    def cell(xy):
        return tuple(np.round((np.array(xy)-origin)/resolution).astype(int))
    def position(k):
        return origin+np.array(k)*resolution
    def valid(k):
        if any(k[i]<0 or k[i]>=shape[i] for i in range(2)):
            return False
        p=Point(position(k))
        return road.covers(p) and not any(o.covers(p) for o in obstacles)
    first,last=cell([start.x,start.y]),cell(goal)
    queue=[(0,first)]
    best={first:0.0}
    parent={}
    expanded=0
    while queue:
        _,current=heapq.heappop(queue)
        if current==last:
            path=[current]
            while path[-1]!=first:
                path.append(parent[path[-1]])
            points=[position(k).tolist() for k in reversed(path)]
            return dict(points=points,expanded=expanded,duration_s=time.perf_counter()-started,failure=None)
        expanded+=1
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]:
            neighbor=current[0]+dx,current[1]+dy
            if not valid(neighbor) or (dx and dy and (not valid((current[0]+dx,current[1])) or not valid((current[0],current[1]+dy)))):
                continue
            length=math.hypot(dx,dy)*resolution
            deviation=reference.distance(Point(position(neighbor)))
            cost=best[current]+length+.08*deviation*length
            if cost>=best.get(neighbor,math.inf):
                continue
            best[neighbor]=cost
            parent[neighbor]=current
            heuristic=float(np.linalg.norm(position(neighbor)-position(last)))
            heapq.heappush(queue,(cost+heuristic,neighbor))
    return dict(points=[],expanded=expanded,duration_s=time.perf_counter()-started,failure='NO_POINT_ROUTE')


def execute(config,task,seed,method,dest):
    line,road,obstacles,proof=setup(config,task)
    rng=np.random.default_rng(seed)
    xy=np.asarray(config['start_xy'])+rng.normal(0,.03,2)
    state=VehicleState(*xy,config['start_yaw_rad']+rng.normal(0,math.radians(.5)))
    summary=dict(task_id=task['task_id'],config_id=config['config_id'],kind=task['kind'],seed=seed,
        method=method,gt_assisted_correct_object=True,independent_disconnection_proof=proof,
        actual_perception_called=False,full_perception_arrival_credit=False,
        route_generated=False,object_selection_evaluated=False,stable_lock_evaluated=False,
        actual_kinematic_arrival=False)
    if task['kind']=='no_match':
        summary.update(status='NO_MATCH_IN_SCOPE',termination_reason='给定正确类别集合的上限条件，无bus')
        return summary
    target=config['targets'][task['target_index']]
    goal=np.asarray(target['goal_xy'])
    if method=='hybrid_astar':
        plan=hybrid_astar(state,goal,target['goal_heading_rad'],road,obstacles,line,SPEC)
        points=[[s.x,s.y] for s in plan.states]
        planning=dict(expanded=plan.expanded,duration_s=plan.duration_s,failure=plan.failure)
    else:
        planning=point_astar(state,goal,road,obstacles,line)
        points=planning.pop('points')
    summary['planning']=planning
    summary['route_generated']=bool(points)
    if not points:
        summary.update(status='PLANNING_FAILED',termination_reason=planning['failure'])
        return summary
    dump(dest/'plan.json',dict(points=points,**planning,goal_xy=goal.tolist(),target_instance_id=target['instance_id']))
    truth=GoalTruth(task['task_id'],tuple(task['accepted_instance_ids']),tuple(task['regions']),
        config['config_id'],(0,90),'冻结合成场景独立目标区域，已知正确对象上限',
        reachability='reachable' if task['kind']=='reachable_positive' else 'inaccessible')
    evaluator=ArrivalEvaluator(truth)
    states=[state]
    events=[]
    final='TIMEOUT'
    reason='execution_time_budget'
    guard_latched=False
    for _ in range(900):
        acceleration,steering,tracking=tracking_control(state,points,goal,SPEC)
        predicted=step_vehicle(state,acceleration,steering,SPEC)
        braking=predicted
        braking_safe=True
        for _ in range(math.ceil(SPEC.max_speed_mps/SPEC.max_braking_mps2/SPEC.dt_s)+1):
            braking_next=step_vehicle(braking,-SPEC.max_braking_mps2,braking.steering_rad,SPEC)
            sweep=swept_body_polygon(braking,braking_next,SPEC)
            if not road.covers(sweep) or any(sweep.intersects(o) for o in obstacles):
                braking_safe=False
                break
            braking=braking_next
            if braking.speed_mps==0:
                break
        guard_latched |= not braking_safe or not body_is_legal(predicted,road,obstacles,SPEC)
        if guard_latched:
            acceleration=-SPEC.max_braking_mps2
            steering=state.steering_rad
            predicted=step_vehicle(state,acceleration,steering,SPEC)
        swept=swept_body_polygon(state,predicted,SPEC)
        collision=any(swept.intersects(o) for o in obstacles)
        violation=not road.covers(swept)
        state=predicted
        states.append(state)
        arrival=evaluator.update(time_s=state.time_s,xy=[state.x,state.y],yaw_rad=state.yaw_rad,
            speed_mps=state.speed_mps,evaluated_instance_id=target['instance_id'],
            body_collision=collision,lane_violation=violation)
        events.append(dict(state=asdict(state),control=dict(acceleration_mps2=acceleration,
            steering_command_rad=steering),tracking=tracking,arrival_evaluation=arrival,
            emergency_braking=guard_latched,body_collision=collision,lane_violation=violation))
        if collision or violation:
            final='EXECUTION_FAILED'
            reason='actual_swept_collision_or_lane_violation'
            break
        if guard_latched and state.speed_mps==0:
            final='EXECUTION_FAILED'
            reason='safety_guard_stopped_vehicle'
            break
        if arrival['actually_arrived']:
            final='ARRIVED'
            reason='independent_region_identity_speed_heading_hold_passed'
            break
    audit=audit_executed_states(states,road,obstacles,SPEC)
    if not audit['passed']:
        final='EXECUTION_FAILED'
        reason='independent_execution_audit_failed'
    with (dest/'execution.jsonl').open('x') as f:
        for event in events:
            f.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
    dump(dest/'independent_execution_audit.json',audit)
    summary.update(status=final,termination_reason=reason,executed_states=len(states),
        execution_duration_s=state.time_s,actual_kinematic_arrival=final=='ARRIVED',
        independent_execution_audit_passed=audit['passed'],
        path_length_m=sum(math.hypot(b.x-a.x,b.y-a.y) for a,b in zip(states,states[1:])),
        max_cross_track_error_m=max((e['tracking']['cross_track_error_m'] for e in events),default=None),
        final_state=asdict(state),position_modified_only_by_bicycle_controls=True)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--task',action='append')
    parser.add_argument('--seed',type=int,action='append')
    parser.add_argument('--methods',nargs='+',choices=['hybrid_astar','point_astar'],default=['hybrid_astar','point_astar'])
    args=parser.parse_args()
    releasepath=OUTPUT/'interactive_simulation/synthetic_tasks_v1.json'
    release=json.loads(json.dumps(make_release(),allow_nan=False))
    if releasepath.exists():
        if json.loads(releasepath.read_text())!=release:
            raise ValueError('frozen synthetic task release changed; create a new version')
    else:
        dump(releasepath,release)
    dest=OUTPUT/'vehicle_execution'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    tasks=[t for t in release['tasks'] if not args.task or t['task_id'] in args.task]
    configs={c['config_id']:c for c in release['configs']}
    dump(dest/'release.json',dict(task_release_sha256=digest(releasepath),task_ids=[t['task_id'] for t in tasks],
        seeds=args.seed or SEEDS,methods=args.methods,gt_assisted=True,
        source_sha256={str(p.relative_to(PROJECT)):digest(p) for p in [Path(__file__),
            PROJECT/'conceptgraph/navigation/vehicle_geometry.py',PROJECT/'conceptgraph/navigation/vehicle_planner.py']},
        full_perception_navigation_evaluated=False))
    with tarfile.open(dest/'source_snapshot.tar.gz','w:gz') as archive:
        for path in [Path(__file__),PROJECT/'conceptgraph/navigation/vehicle_geometry.py',
                     PROJECT/'conceptgraph/navigation/vehicle_planner.py',PROJECT/'conceptgraph/completion/protocol.py']:
            archive.add(path,arcname=str(path.relative_to(PROJECT)))
    results=[]
    for task in tasks:
        for seed in args.seed or SEEDS:
            for method in args.methods:
                case=dest/task['task_id']/f'seed{seed}'/method
                case.mkdir(parents=True)
                result=execute(configs[task['config_id']],task,seed,method,case)
                dump(case/'result.json',result)
                results.append(result)
                with (dest/'results.jsonl').open('a') as f:
                    f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
                print(json.dumps({k:result.get(k) for k in ['task_id','seed','method','status','actual_kinematic_arrival']},ensure_ascii=False),flush=True)
    dump(dest/'summary.json',dict(task_count=len(tasks),config_count=len({t['config_id'] for t in tasks}),
        trials=len(results),results=results,actual_kinematic_arrivals=sum(r['actual_kinematic_arrival'] for r in results),
        actual_full_perception_navigation_arrivals=0,stage11_complete=False,goal_complete=False))


if __name__=='__main__':
    main()
