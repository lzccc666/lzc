"""官方CARLA调用边界与帧对齐复核；静态证据不冒充形式化信息流证明或真人审核。"""
import ast,json
from pathlib import Path
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];O=R/'carla_official/input_boundary_audit_001'

def main():
 O.mkdir(exist_ok=False);main=R/'source/carla_ablation_episode_003.py';tree=ast.parse(main.read_text());constructors=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Agent'];assert len(constructors)==1
 args=[ast.unparse(n) for n in constructors[0].args];assert args==["PARAMS['query']",'prior','classes','OUT','condition']
 prior=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prior' for t in n.targets));assert isinstance(prior.value,ast.Call) and {n.arg for n in prior.value.keywords}=={'centerline_xy','road_polygon_xy'}
 prior_values=[ast.unparse(n.value) for n in prior.value.keywords];assert prior_values==['points.tolist()','list(road.exterior.coords)']
 classes=[];sources={str(main):digest(main)}
 forbidden={'truth','truth_center','accepted','target','target_t','GoalTruth','ArrivalEvaluator','evaluator','truth_obstacles','evaluation_identity'}
 for name,kind in [('carla_ablation_agent_001.py','Agent'),('carla_agent_isolated_002.py','PerceptionVehicleAgent')]:
  p=R/'source'/name;sources[str(p)]=digest(p);t=ast.parse(p.read_text());cls=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==kind);names={n.id for n in ast.walk(cls) if isinstance(n,ast.Name)};reads=[ast.unparse(n.func) for n in ast.walk(cls) if isinstance(n,ast.Call) and (isinstance(n.func,ast.Name) and n.func.id=='open' or isinstance(n.func,ast.Attribute) and n.func.attr in {'read_text','read_bytes','open'})];assert not names&forbidden and not reads;classes.append(dict(path=str(p),class_name=kind,line=cls.lineno,forbidden_evaluation_names_present=[],direct_file_reads=[]))
 rows=[]
 for case in sorted((R/'carla_official/ablation_batch_003').glob('*/*')):
  if not (case/'release.json').exists():continue
  release=json.loads((case/'release.json').read_text());assert release['server_version']==release['client_version']=='0.9.16' and release['source_sha256'][str(main)]==digest(main)
  execution=list(map(json.loads,(case/'execution.jsonl').open()));assert execution
  assert all(x['frame']==x['snapshot_frame'] and set(x['sensor_frames'])=={'rgb','depth','lidar'} and all(v==x['frame'] for v in x['sensor_frames'].values()) for x in execution)
  assert all(b['frame']==a['frame']+1 for a,b in zip(execution,execution[1:]))
  perceptions=list(map(json.loads,(case/'perception.jsonl').open()));frames={x['frame'] for x in execution};assert all(p['sensor_frame'] in frames for p in perceptions)
  rows.append(dict(episode=release['episode']['episode'],condition=release['condition'],complete=(case/'summary.json').exists(),execution_ticks=len(execution),perception_updates=len(perceptions),official_server_api_matches=True,source_matches_audited_entry=True,all_sensor_snapshot_frames_align=True,continuous_official_tick_frame_ids=True))
 assert len(rows)==15 and sum(r['complete'] for r in rows)==14
 dump(O/'report.json',dict(source_sha256={**sources,str(Path(__file__)):digest(__file__)},agent_constructor_args=args,prior_keys=['centerline_xy','road_polygon_xy'],prior_sources=prior_values,classes_checked=classes,episodes=rows,episode_count=15,complete_episodes=14,evaluation_only_truth_not_in_direct_agent_call_arguments=True,actual_official_state_source='snapshot.find(ego.id).get_transform/get_velocity与实测前轮角；不是自建仿真积分状态',actual_control_execution='carla.command.ApplyVehicleControl -> apply_batch_sync -> world.tick -> next official snapshot/sensors',target_truth_use='仅生成evaluation_only终区、身份配对与ArrivalEvaluator；终止回合是外部评估，未将真值目标位置传给Agent',road_prior_scope='公开地图道路中心线/走廊。起点固定后沿路65个1m步长构造，与20m/30m目标参数分开；有地图先验，不是纯视觉无地图驾驶',reused_code_scope='复用原模块的感知/几何/规划类与数据结构，未调用其自建仿真runner来生成官方物理状态',limitations=['AST只检查直接边界、类内名称/直接文件读取和保存帧对齐，不是整个Python依赖图的形式化非干涉证明','允许自车官方位姿/速度/轮角和道路地图先验；不声称独立传感器定位已验证','没有真人颜色/细属性审核信用，没有新的GPU检测或物理到达信用'],new_GPU_requests=0,new_physical_episodes=0,human_review_credit=False,goal_complete=False));print(json.dumps(dict(episodes=15,complete=14,all_frame_alignment_passed=True)))

if __name__=='__main__':main()
