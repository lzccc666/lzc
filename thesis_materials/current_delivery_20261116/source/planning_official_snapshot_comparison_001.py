"""官方CARLA实测感知快照的同输入规划诊断；不增加任何闭环到达或外部实现信用。"""
import copy,csv,dataclasses,inspect,json,math,time,resource
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon,LineString
from conceptgraph.completion.stage9 import dump,digest,PROJECT
from conceptgraph.completion.kinematic_benchmark import point_astar
from conceptgraph.navigation.vehicle_geometry import VehicleSpec,VehicleState,audit_executed_states,footprint_polygon,swept_body_polygon,angle_difference
from conceptgraph.navigation.vehicle_planner import hybrid_astar
R=Path(__file__).resolve().parents[1];O=R/'method_comparison/planning_official_snapshots_001'
METHODS=['point_astar','body_hybrid_astar_without_steering_rate','full_vehicle_hybrid_astar']
def main():
 O.mkdir(parents=True,exist_ok=False);cases=[];sources={}
 # 只读取已完成的官方CARLA开发实验；固定按episode和时间各取前6个目标接近规划事件。
 for name in ['pipeline_episode_003','pipeline_episode_004','pipeline_episode_005','pipeline_episode_006','pipeline_episode_007']:
  d=R/'carla_official'/name;release=json.loads((d/'release.json').read_text());states={r['frame']:r['state'] for r in map(json.loads,(d/'execution.jsonl').open())};observations=list(map(json.loads,(d/'perception.jsonl').open()));selected=[q for q in observations if q.get('planning') and q['planning']['purpose']=='target_approach'][:6]
  for q in selected:
   cases.append(dict(case_id=f"{name}_frame{q['sensor_frame']}",episode=name,sensor_frame=q['sensor_frame'],state=states[q['sensor_frame']],goal_xy=q['planning']['goal_xy'],goal_yaw_rad=q['planning']['goal_yaw_rad'],road_polygon_xy=release['prior']['road_polygon_xy'],centerline_xy=release['prior']['centerline_xy'],observed_obstacles=q['known_obstacle_polygons'],spec=release['vehicle_spec'],goal_source='实际感知LiDAR表面推导，非目标真值',official_CARLA_input=True))
  for p in [d/'release.json',d/'execution.jsonl',d/'perception.jsonl']:sources[str(p)]=digest(p)
 if not cases:raise ValueError('no frozen official sensor cases')
 dump(O/'release.json',dict(cases=cases,methods=METHODS,seed=17,wall_budget_per_plan_s=4.,max_expanded=5000,point_resolution_m=.4,source_sha256=sources,source_code_sha256={str(p):digest(p) for p in [Path(__file__),PROJECT/'conceptgraph/completion/kinematic_benchmark.py',PROJECT/'conceptgraph/navigation/vehicle_planner.py',PROJECT/'conceptgraph/navigation/vehicle_geometry.py']},scope='同一官方传感器快照的内部规划结构比较；无新环境、无闭环执行；不宣称复现Dolgov作者实现',ground_truth_used=False,goal_complete=False))
 # 同一个原有点A*添加与其他方法相同的4秒上限；原实现不改。
 source=inspect.getsource(point_astar).replace('    while queue:\n','    while queue:\n        if time.perf_counter()-started>4.:return dict(points=[],expanded=expanded,duration_s=time.perf_counter()-started,failure="PLANNING_TIME_BUDGET")\n');namespace=copy.copy(point_astar.__globals__);exec(compile(source,'<bounded_point_astar>','exec'),namespace);bounded=namespace['point_astar'];results=[]
 for case in cases:
  start=VehicleState(**case['state']);spec=VehicleSpec(**case['spec']);road=Polygon(case['road_polygon_xy']);obs=[Polygon(p) for p in case['observed_obstacles']];ref=LineString(case['centerline_xy']);goal=np.array(case['goal_xy'])
  for method in METHODS:
   begun=time.monotonic();audit=None
   if method=='point_astar':
    answer=bounded(start,goal,road,obs,ref,.4);points=answer.pop('points');failure=answer['failure'];expanded=answer['expanded'];states=[]
    if len(points)>1:
     line=LineString(points);dist=np.arange(0,line.length+.0001,.1);xy=np.array([line.interpolate(v).coords[0] for v in dist]);angles=np.arctan2(np.diff(xy[:,1]),np.diff(xy[:,0]));angles=np.r_[angles,angles[-1]]
     states=[VehicleState(*v,float(a)) for v,a in zip(xy,angles)];body_bad=sum(not road.covers(footprint_polygon(s,spec)) or any(footprint_polygon(s,spec).intersects(o) for o in obs) for s in states);curvature=np.abs(np.diff(np.unwrap(angles)))/.1;audit=dict(body_violating_samples=body_bad,curvature_violating_samples=int((curvature>1/spec.minimum_turn_radius_m+1e-6).sum()),passed=not body_bad and not (curvature>1/spec.minimum_turn_radius_m+1e-6).any(),steering_rate_measured=False)
   else:
    internal=dataclasses.replace(spec,max_steering_rate_radps=1e6) if method=='body_hybrid_astar_without_steering_rate' else spec
    plan=hybrid_astar(start,goal,case['goal_yaw_rad'],road,obs,ref,internal,budget_s=4.,max_expanded=5000);states=plan.states;points=[[s.x,s.y] for s in states];failure=plan.failure;expanded=plan.expanded
    if states:audit=audit_executed_states(states,road,obs,spec)
   elapsed=time.monotonic()-begun;row=dict(case_id=case['case_id'],episode=case['episode'],method=method,route_generated=bool(points),plan_passes_full_vehicle_audit=bool(points and audit and audit['passed']),failure=failure,expanded=expanded,planner_and_diagnostic_wall_s=elapsed,planned_length_m=float(sum(np.linalg.norm(np.asarray(b)-a) for a,b in zip(points,points[1:]))),audit=audit,peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,actual_control_executed=False,actual_arrival_credit=False)
   dump(O/(case['case_id']+'_'+method+'.json'),dict(**row,points=points,planned_states=[dataclasses.asdict(s) for s in states]));results.append(row);print(json.dumps({k:row[k] for k in ['case_id','method','route_generated','plan_passes_full_vehicle_audit','failure']}),flush=True)
 summary=[]
 for method in METHODS:
  rr=[r for r in results if r['method']==method];summary.append(dict(method=method,cases=len(rr),source_episodes=len({r['episode'] for r in rr}),routes=sum(r['route_generated'] for r in rr),full_vehicle_audit_pass=sum(r['plan_passes_full_vehicle_audit'] for r in rr),mean_planner_and_diagnostic_s=float(np.mean([r['planner_and_diagnostic_wall_s'] for r in rr])),actual_arrival_evaluated=False))
 with (O/'comparison.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
 dump(O/'report.json',dict(summary=summary,results=results,unique_snapshots=len(cases),failures_retained=True,algorithm_references=dict(point_astar='https://doi.org/10.1109/TSSC.1968.300136',hybrid_astar='https://ai.stanford.edu/~ddolgov/papers/dolgov_gpp_stair08.pdf'),implementation_provenance='项目内部简化诊断；非作者官方代码或其论文数字',dependent_development_snapshots=True,goal_complete=False))
if __name__=='__main__':main()
