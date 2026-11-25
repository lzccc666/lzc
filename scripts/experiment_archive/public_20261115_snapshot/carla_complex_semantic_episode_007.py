"""官方CARLA开发episode：传感器→原有真实前端→对象/关系→规则选择→Hybrid A*→物理控制。"""
import argparse,copy,gzip,json,math,pickle,queue,time,traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import carla,cv2,numpy as np
from shapely.geometry import LineString,Polygon,Point
from conceptgraph.completion.stage9 import PROJECT,digest,dump
from carla_agent_isolated_002 import evaluation_identity,json_safe
from carla_ablation_agent_001 import Agent,CONDITIONS
from carla_compact_evidence_001 import save_compact
from carla_prius_braking_pilot_007 import AccelerationFeedback
from carla_real_graph_bridge_002 import ActualQwenBackend,RealGraphNavigationBridge
import carla_agent_isolated_002 as nav
from conceptgraph.completion.protocol import ArrivalEvaluator,GoalTruth
from conceptgraph.navigation.vehicle_geometry import VehicleSpec,VehicleState,footprint_polygon
from conceptgraph.scripts.streamlined_detections import load_detection_models,detect_rgb_file
ROOT=Path(__file__).resolve().parents[1];OUT=None
PARAMS=None
S=np.diag([1.,-1.,1.,1.]);B=np.array([[0,0,1,0],[1,0,0,0],[0,-1,0,0],[0,0,0,1.]])
K=np.array([[320.,0,320],[0,320.,180],[0,0,1.]])
def sensor_get(q,frame,kind):
 while True:
  try:d=q.get(timeout=30)
  except queue.Empty:raise TimeoutError(f'{kind} missing at {frame}')
  if d.frame==frame:return d
  if d.frame>frame:raise ValueError('future sensor frame')
def main():
 global OUT,PARAMS
 parser=argparse.ArgumentParser();parser.add_argument('--episode',required=True);parser.add_argument('--mode',choices=['rule','base','adapter'],required=True);args=parser.parse_args()
 bank=json.loads((ROOT/'carla_official/complex_semantic_registration_007.json').read_text());PARAMS=next(p for p in bank['episodes'] if p['episode']==args.episode);condition='full_rule_geometry';OUT=ROOT/'carla_official/complex_semantic_navigation_007'/args.episode/args.mode
 for p,h in bank['source_sha256'].items():
  if digest(p)!=h:raise ValueError('frozen source changed: '+p)
 np.random.seed(bank['seed'])
 import random,torch
 random.seed(bank['seed']);torch.manual_seed(bank['seed'])
 OUT.mkdir(parents=True,exist_ok=False);started=time.monotonic();c=carla.Client('127.0.0.1',2010);c.set_timeout(120);w=c.reload_world(False);c.set_timeout(30);old=w.get_settings()
 if c.get_server_version()!='0.9.16' or 'Town10HD_Opt' not in w.get_map().name:raise RuntimeError('official server version/map changed')
 actors=[];collisions=[];rows=[];perceptions=[];bridge=None
 try:
  settings=w.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.1;settings.no_rendering_mode=False;w.apply_settings(settings);w.set_weather(carla.WeatherParameters.ClearNoon)
  bp=w.get_blueprint_library();start=w.get_map().get_spawn_points()[0];wp=w.get_map().get_waypoint(start.location);waypoints=[wp]
  for _ in range(65):
   nxt=sorted(waypoints[-1].next(1.),key=lambda p:(p.road_id,p.section_id,p.lane_id,p.s))
   if not nxt:break
   waypoints.append(nxt[0])
  if len(waypoints)<40:raise ValueError('预选道路不足40m，保留失败不换路线')
  ego=w.spawn_actor(bp.find('vehicle.toyota.prius'),start);actors.append(ego)
  target_t=carla.Transform(waypoints[PARAMS['target_waypoint_index']].transform.location,waypoints[PARAMS['target_waypoint_index']].transform.rotation);target_t.location.z+=.3
  target=w.spawn_actor(bp.find('vehicle.tesla.model3'),target_t);actors.append(target);target.set_simulate_physics(False)
  side_index=PARAMS['distractor_waypoint_index'];side_t=carla.Transform(waypoints[side_index].transform.location,waypoints[side_index].transform.rotation);side_t.location.z+=.3;side_right=side_t.get_right_vector();side_t.location.x+=side_right.x*PARAMS['distractor_lateral_right_m'];side_t.location.y+=side_right.y*PARAMS['distractor_lateral_right_m']
  distractor=w.spawn_actor(bp.find('vehicle.tesla.model3'),side_t);actors.append(distractor);distractor.set_simulate_physics(False)
  original_physics=ego.get_physics_control();original_torque_curve=[[v.x,v.y] for v in original_physics.torque_curve];assert [[v.x,v.y] for v in ego.get_physics_control().torque_curve]==original_torque_curve
  for _ in range(10):w.tick(30)
  physics=ego.get_physics_control();wheel_max_deg=float(max(q.max_steer_angle for q in physics.wheels));wheel_world=np.array([[q.position.x,q.position.y,q.position.z,100.] for q in physics.wheels])/100.;wheel_local=(np.asarray(ego.get_transform().get_inverse_matrix())@wheel_world.T).T[:,:3];front_axle=np.mean(wheel_local[:2,:],axis=0);rear_axle=np.mean(wheel_local[2:,:],axis=0);rear_offset=-float(rear_axle[0]);wheelbase=float(front_axle[0]-rear_axle[0]);ebox=ego.bounding_box;spec=VehicleSpec(wheelbase_m=wheelbase,front_from_rear_axle_m=float(ebox.extent.x+rear_offset),rear_overhang_m=float(ebox.extent.x-rear_offset),width_m=float(2*ebox.extent.y),dt_s=.1);nav.SPEC=spec
  tf=ego.get_transform();yaw=-math.radians(tf.rotation.yaw);origin=np.array([tf.location.x,-tf.location.y]);first=origin-8*np.array([math.cos(yaw),math.sin(yaw)])
  points=np.array([first.tolist()]+[[p.transform.location.x,-p.transform.location.y] for p in waypoints]);line=LineString(points);lane_width=min(p.lane_width for p in waypoints);road=line.buffer(lane_width/2,cap_style=2)
  t=target.get_transform();truth_center=np.array([t.location.x,-t.location.y]);target_s=line.project(Point(truth_center));goal_s=target_s-8.;goal=line.interpolate(goal_s);delta=np.array(line.interpolate(goal_s+.1).coords[0])-np.array(line.interpolate(goal_s-.1).coords[0]);goal_yaw=math.atan2(delta[1],delta[0]);region=Point(goal.x,goal.y).buffer(1.3).intersection(road)
  accepted=str(target.id);target_polygon=Polygon([(v.x,-v.y) for v in target.bounding_box.get_world_vertices(t)]).convex_hull
  other_t=distractor.get_transform();other_center=np.array([other_t.location.x,-other_t.location.y]);other_polygon=Polygon([(v.x,-v.y) for v in distractor.bounding_box.get_world_vertices(other_t)]).convex_hull
  assert np.linalg.norm(other_center-origin)<np.linalg.norm(truth_center-origin), '预选干扰车应比指定目标近；失败不换场景'
  certificate=[]
  for progress in np.arange(line.project(Point(origin-rear_offset*np.array([math.cos(yaw),math.sin(yaw)]))),goal_s+.05,.2):
   xy=np.array(line.interpolate(progress).coords[0]);v1=np.array(line.interpolate(min(line.length,progress+.05)).coords[0])-np.array(line.interpolate(max(0,progress-.05)).coords[0]);a=math.atan2(v1[1],v1[0]);state=VehicleState(*xy,a);body=footprint_polygon(state,spec);certificate.append(bool(road.covers(body) and not body.intersects(target_polygon) and not body.intersects(other_polygon)))
  reachable=bool(certificate and all(certificate));truth=GoalTruth(PARAMS['episode'],(accepted,) if PARAMS['positive'] else (),({'instance_id':accepted,'polygon_xy':list(region.exterior.coords),'heading_rad':goal_yaw},) if PARAMS['positive'] else (),w.get_map().name,(0,PARAMS['timeout_s']),'官方CARLA目标actor及冻结区域；道路中心线逐0.2m整车检查，仅已知固定目标碰撞几何',reachability=('reachable' if reachable else 'unknown') if PARAMS['positive'] else 'unknown',timeout_s=PARAMS['timeout_s'])
  gt=dict(target_actor_id=accepted,center_xy=truth_center.tolist(),category='car',target_polygon_xy=list(target_polygon.exterior.coords),arrival_truth=asdict(truth),independent_centerline_certificate=dict(steps=len(certificate),all_legal=reachable,limitation='校验已声明道路走廊和目标车；世界碰撞另由官方碰撞传感器评价'))
  gt['other_controlled_actors']=[dict(instance_id=str(distractor.id),category='car',center_xy=other_center.tolist(),polygon_xy=list(other_polygon.exterior.coords))]
  gt['controlled_object_scope']='两辆预注册官方车辆；最近/第二近按当前自车距离定义。未把这些actor身份或位置发给智能体；不证明世界内所有未标注静态道具均已分类。'
  gt['initial_target_distance_m']=float(np.linalg.norm(truth_center-origin));gt['initial_distractor_distance_m']=float(np.linalg.norm(other_center-origin))
  dump(OUT/'evaluation_only/truth.json',gt)
  prior=dict(centerline_xy=points.tolist(),road_polygon_xy=list(road.exterior.coords));classes=(PROJECT/'conceptgraph/vehicle_classes.txt').read_text().splitlines()
  release=dict(server_version=c.get_server_version(),client_version=c.get_client_version(),map=w.get_map().name,weather='ClearNoon',seed=17,traffic_density=0,vehicle='vehicle.toyota.prius',rear_axle_offset_from_actor_center_m=rear_offset,measured_wheel_positions_local_m=wheel_local.tolist(),vehicle_spec=asdict(spec),query=PARAMS['query'],condition=condition,target_initial_location=str(target_t),target_speed_mps=0,initial_pose=str(start),fixed_dt_s=.1,maximum_sim_s=PARAMS['timeout_s'],maximum_wall_s=600,actuator_policy='原厂Prius；冻结pilot007同一反馈控制器，停车刹车触发0.15m/s，最终速度验收仍0.08m/s；desired speed<=.55；actual acceleration反馈；物理转向命令slew10deg/s',perception_every_ticks=2,RGB=dict(width=640,height=360,fov=90),LiDAR=dict(channels=32,range_m=60,points_per_second=64000,rotation_hz=10,sensor_tick=0),camera_transform=[1.5,0,1.8],lidar_transform=[0,0,2.2],sensor_set=['rgb','depth','lidar','pose','collision'],prior=prior,source_sha256={str(p):digest(p) for p in [Path(__file__),ROOT/'source/carla_agent_isolated_002.py',ROOT/'source/carla_sensor_utils_001.py',ROOT/'source/carla_ablation_agent_001.py',ROOT/'source/carla_compact_evidence_001.py',PROJECT/'conceptgraph/completion/interactive_navigation.py',PROJECT/'conceptgraph/slam/streamlined_mapping.py',PROJECT/'conceptgraph/slam/mapping.py',PROJECT/'conceptgraph/slam/utils.py',PROJECT/'conceptgraph/scripts/streamlined_detections.py']},truth_sha256=digest(OUT/'evaluation_only/truth.json'),task_kind='pre_registered_development_ablation_not_whole_project_sealed',performance_credit=False,goal_complete=False)
  release['ego_physics_control']=dict(changed=False,reason='原厂Prius动力学，未调用apply_physics_control；同一pilot007反馈器',original_torque_curve_Nm=original_torque_curve,applied_torque_curve_Nm=[[v.x,v.y] for v in ego.get_physics_control().torque_curve],calibration_sha256=digest(ROOT/'carla_official/prius_braking_pilot_007/report.json'))
  release['steering_calibration']=dict(physical_max_steer_deg=wheel_max_deg,observed='actual FL/FR wheel steer degrees; equivalent bicycle tangent mean; RH sign inverted',throttle_up_per_tick=.001,throttle_down_per_tick=.004,extra_reduction_if_acceleration_above_point3=.008,throttle_max=.24,command_steering_slew_degps=10,coasting_calibration_sha256=digest(ROOT/'carla_official/prius_braking_pilot_007/report.json'),vehicle_choice_rule='固定Prius及pilot007停车控制：两条官方标定通过；旧Lincoln完整导航0/2及先前Prius停车失败保留。未查看本次导航结果；不改到达区域、时间和限值。')
  release['registration_sha256']=digest(ROOT/'carla_official/complex_semantic_registration_007.json')
  release['episode']=PARAMS
  release['storage_schema']='原RGB PNG、原CARLA24bit深度PNG、原LiDAR bin、可逆mask packbits、crop按原RGB+20px padding恢复；逐帧数组和crop等价核对'
  release['controller_calibration_note']='原厂Prius及pilot007反馈控制。两项外生转向标定通过不代表本次真实感知导航通过；旧Lincoln及Tesla实验另列。'
  release['comparison_scope']='官方CARLA开发实验：相同两车场景/传感器/预算下比较强规则、基础Qwen、Adapter；复杂否定/第二近实际交给后台模型，未证明封存或大规模成功'
  release['graph_config']=bank['graph_config'];release['semantic_mode']=args.mode;release['semantic_ttl_wall_s']=20.;release['semantic_registration']=bank['semantic_protocol'];release['maximum_wall_s']=900
  release['model_sha256']={str(p):digest(p) for p in [PROJECT/'checkpoint/yolov8l-world.pt',PROJECT/'checkpoint/mobile_sam.pt',PROJECT/'checkpoint/hf_models/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_model.safetensors']}
  dump(OUT/'release.json',release)
  backend=None
  if args.mode!='rule':
   binding=json.loads(Path(bank['base_model_binding']).read_text());assert all(digest(p['path'])==p['sha256'] for p in binding['base_model_files'])
   backend=ActualQwenBackend(PROJECT/'checkpoint/Qwen2-VL-2B-Instruct',bank['adapter'] if args.mode=='adapter' else None,OUT/'semantic_backend',seed=bank['seed'])
  bridge=RealGraphNavigationBridge(PARAMS['query'],args.mode,backend=backend,ttl_s=20.)
  agent=Agent(PARAMS['query'],prior,classes,OUT,condition);agent.semantic_bridge=bridge;agent.graph=nav.IncrementalSceneGraph(bank['graph_config']);evaluator=ArrivalEvaluator(truth);cfg=SimpleNamespace(device='cuda:0');models=load_detection_models(cfg)
  for model in models[:2]:model.to('cuda:0');model.overrides['device']='cuda:0'
  queues={}
  for kind,identifier,tr in [('rgb','sensor.camera.rgb',carla.Transform(carla.Location(x=1.5,z=1.8))),('depth','sensor.camera.depth',carla.Transform(carla.Location(x=1.5,z=1.8))),('lidar','sensor.lidar.ray_cast',carla.Transform(carla.Location(z=2.2)))]:
   b=bp.find(identifier);options={'image_size_x':'640','image_size_y':'360','fov':'90','sensor_tick':'0.0'} if kind!='lidar' else {'channels':'32','range':'60','points_per_second':'64000','rotation_frequency':'10','sensor_tick':'0.0'}
   for k,v in options.items():b.set_attribute(k,v)
   a=w.spawn_actor(b,tr,attach_to=ego);actors.append(a);q=queue.Queue();a.listen(q.put);queues[kind]=q
  sensor=w.spawn_actor(bp.find('sensor.other.collision'),carla.Transform(),attach_to=ego);actors.append(sensor);sensor.listen(lambda e:collisions.append(dict(frame=e.frame,other=e.other_actor.type_id,impulse=[e.normal_impulse.x,e.normal_impulse.y,e.normal_impulse.z])))
  ctrl=carla.VehicleControl(brake=1,manual_gear_shift=True,gear=1);controller=AccelerationFeedback();previous_throttle=0.;terminal_target=None;selected_id=None;correct_selection=False;candidate_target_coverage=False;correct_lock=False;route_generated=False;reached=False;final='TIMEOUT';lane_violation=False;last_state=None;first_time=None
  with (OUT/'execution.jsonl').open('x') as execution,(OUT/'perception.jsonl').open('x') as perception:
   for tick in range(round(PARAMS['timeout_s']/.1)):
    if time.monotonic()-started>900:final='WALL_TIMEOUT';break
    import shutil
    if shutil.disk_usage(ROOT).free<20*1024**3:raise OSError('free space below20GiB; raw prefix retained')
    if tick%20==0:
     import os
     used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
     policy=json.loads((ROOT/'json/resource_policy_002.json').read_text())
     if used+64*1024**2>=policy['new_project_artifact_budget_bytes']:raise OSError('project output near25GiB; prefix retained')
    applied_control=dict(throttle=ctrl.throttle,steer=ctrl.steer,brake=ctrl.brake,manual_gear_shift=True,gear=1);reply=c.apply_batch_sync([carla.command.ApplyVehicleControl(ego.id,ctrl)],False)
    if reply[0].error:raise RuntimeError(reply[0].error)
    frame=w.tick(30);snapshot=w.get_snapshot()
    if snapshot.frame!=frame:raise RuntimeError('snapshot frame mismatch')
    measured=snapshot.find(ego.id);data={k:sensor_get(q,frame,k) for k,q in queues.items()};tf=measured.get_transform();vel=measured.get_velocity();angle=-math.radians(tf.rotation.yaw);center=np.array([tf.location.x,-tf.location.y]);rear=center-rear_offset*np.array([math.cos(angle),math.sin(angle)]);speed=math.sqrt(vel.x**2+vel.y**2);first_time=data['rgb'].timestamp if first_time is None else first_time;sim_t=data['rgb'].timestamp-first_time
    wheel_angles=[ego.get_wheel_steer_angle(loc) for loc in [carla.VehicleWheelLocation.FL_Wheel,carla.VehicleWheelLocation.FR_Wheel]]
    measured_steering=-math.atan(float(np.mean(np.tan(np.radians(wheel_angles)))))
    state=VehicleState(*rear,angle,speed,measured_steering,sim_t)
    body=Polygon([(v.x,-v.y) for v in ego.bounding_box.get_world_vertices(tf)]).convex_hull;lane_violation|=not road.buffer(.05).covers(body)
    if tick%2==0:
     folder=OUT/'frames'/f'{tick//2:06d}';folder.mkdir(parents=True);rgb=np.frombuffer(data['rgb'].raw_data,np.uint8).reshape(360,640,4)[:,:,:3][:,:,::-1].copy();cv2.imwrite(str(folder/'rgb.png'),rgb[:,:,::-1]);(folder/'lidar.bin').write_bytes(data['lidar'].raw_data)
     raw=np.frombuffer(data['depth'].raw_data,np.uint8).reshape(360,640,4)[:,:,:3].copy();cv2.imwrite(str(folder/'official_depth_raw_bgr.png'),raw,[cv2.IMWRITE_PNG_COMPRESSION,6]) # lossless official24bit depth, metres=(R+256G+65536B)/16777215*1000
     optical=S@np.asarray(data['rgb'].transform.get_matrix())@B;lidar_w=np.asarray(data['lidar'].transform.get_matrix());lidar=np.frombuffer(data['lidar'].raw_data,np.float32).reshape(-1,4);points_l=np.column_stack([lidar[:,:3],np.ones(len(lidar))]);pc=(np.linalg.inv(optical)@S@lidar_w@points_l.T).T[:,:3];pc=pc[pc[:,2]>.1];uv=pc@K.T;uv=uv[:,:2]/uv[:,2,None];u=np.floor(uv[:,0]).astype(int);v=np.floor(uv[:,1]).astype(int);valid=(u>=0)&(u<640)&(v>=0)&(v<360)&(pc[:,2]<65);u,v,z=u[valid],v[valid],pc[valid,2];depth=np.full((360,640),np.inf);np.minimum.at(depth,(v,u),z);depth[~np.isfinite(depth)]=0;cv2.imwrite(str(folder/'lidar_depth_mm.png'),(depth*1000).astype(np.uint16))
     t0=time.monotonic();gobs,vis=detect_rgb_file(cfg,models,folder/'rgb.png',classes);cv2.imwrite(str(folder/'detections.jpg'),vis)
     save_compact(folder,gobs,rgb)
     result=agent.perceive(tick//2,state,rgb,depth,optical,folder/'rgb.png',gobs);result['measured_wall_s']=time.monotonic()-t0;result['sensor_frame']=frame;result['files_sha256']={str(p):digest(p) for p in folder.iterdir()};result['camera_pose_cv']=optical.tolist();result['lidar_pose_carla']=lidar_w.tolist();result['input_depth_source']='current official LiDAR projected into current RGB; dense official depth retained only for diagnostic';perception.write(json.dumps(result,ensure_ascii=False,default=json_safe)+'\n');perception.flush();perceptions.append(result)
     selected_id,_=evaluation_identity(agent.nodes,[dict(instance_id=accepted,center_xy=truth_center.tolist(),category='car')],agent.selected);candidate_target_coverage|=PARAMS['positive'] and any(evaluation_identity(agent.nodes,[dict(instance_id=accepted,center_xy=truth_center.tolist(),category='car')],a['node_id'])[0]==accepted for a in result['candidates']);correct_selection|=bool(PARAMS['positive'] and result['candidates'] and evaluation_identity(agent.nodes,[dict(instance_id=accepted,center_xy=truth_center.tolist(),category='car')],result['candidates'][0]['node_id'])[0]==accepted);correct_lock|=bool(PARAMS['positive'] and result['lock']['state_after']=='LOCKED' and selected_id==accepted);route_generated|=bool(agent.path and agent.path_purpose=='target_approach')
     print(json.dumps(dict(tick=tick,nodes=len(agent.nodes),candidates=len(result['candidates']),selected=agent.selected,selected_correct=selected_id==accepted,path=len(agent.path),speed=speed)),flush=True)
    if speed>spec.max_speed_mps:acceleration,steering,tracking=-2.,0.,dict(reason='measured_speed_limit_braking')
    else:acceleration,steering,tracking=agent.control(state)
    if collisions or lane_violation:acceleration=-2.
    desired=min(.55,float(tracking.get('desired_speed_mps',0.)))
    if agent.selected!=terminal_target:terminal_target=None
    if agent.selected is not None and agent.path_purpose=='target_approach' and agent.goal is not None and np.linalg.norm(np.array([state.x,state.y])-agent.goal)<=max(.8,speed**2/(2*.2)+speed*previous_throttle/.004*.1/2+.3):terminal_target=agent.selected
    stop_required=terminal_target is not None or not agent.path or agent.guard_latched or collisions or lane_violation
    measured_acceleration=0. if last_state is None else (speed-last_state.speed_mps)/(state.time_s-last_state.time_s)
    requested_acceleration=float(np.clip(.4*(desired-speed),-.3,.15))
    throttle,brake=controller.step(requested_acceleration,measured_acceleration,speed,desired,bool(stop_required or desired<=.1))
    tracking=dict(tracking,actuator_requested_acceleration=requested_acceleration,actual_acceleration_feedback=measured_acceleration,stock_controller_version='pilot007')
    previous_throttle=throttle
    last_command_steering=-math.radians(applied_control['steer']*wheel_max_deg)
    steering=float(np.clip(steering,last_command_steering-math.radians(10)*spec.dt_s,last_command_steering+math.radians(10)*spec.dt_s))
    ctrl=carla.VehicleControl(manual_gear_shift=True,gear=1,throttle=throttle,steer=float(np.clip(-math.degrees(steering)/wheel_max_deg,-1,1)),brake=brake)
    evaluation=evaluator.update(time_s=sim_t,xy=rear.tolist(),yaw_rad=angle,speed_mps=speed,evaluated_instance_id=selected_id,body_collision=bool(collisions),lane_violation=lane_violation)
    row=dict(tick=tick,frame=frame,sensor_frames={k:x.frame for k,x in data.items()},state=asdict(state),ego_transform_carla=tf.get_matrix(),snapshot_frame=snapshot.frame,actual_front_wheel_deg=wheel_angles,actual_velocity_carla=[vel.x,vel.y,vel.z],applied_control=applied_control,next_control=dict(throttle=ctrl.throttle,steer=ctrl.steer,brake=ctrl.brake,manual_gear_shift=True,gear=1),tracking=tracking,evaluation=evaluation,collisions=copy.deepcopy(collisions));execution.write(json.dumps(row,ensure_ascii=False,default=json_safe)+'\n');execution.flush();rows.append(row);last_state=state
    if evaluation['actually_arrived']:reached=True;final='ARRIVED';break
    if collisions or lane_violation:final='COLLISION' if collisions else 'LANE_VIOLATION';break
  ego.apply_control(carla.VehicleControl(brake=1));w.tick(30)
  dump(OUT/'summary.json',dict(server_version=c.get_server_version(),status=final,frames=len(rows),perception_calls=len(perceptions),route_generated=route_generated,episode=PARAMS,condition=condition,correct_object_selected=correct_selection,candidate_target_coverage=candidate_target_coverage,summary_metric_schema=2,correctly_locked=correct_lock,actually_arrived=reached,collisions=collisions,lane_violation=lane_violation,independent_reachability_certificate=reachable,wall_s=time.monotonic()-started,sim_s=None if last_state is None else last_state.time_s,execution_sha256=digest(OUT/'execution.jsonl'),perception_sha256=digest(OUT/'perception.jsonl'),GT_in_agent=False,rule_only=args.mode=='rule',Qwen_evaluated=bool(backend and backend.calls),semantic_mode=args.mode,actual_Qwen_calls=backend.calls if backend else 0,no_match_task=not PARAMS['positive'],no_match_any_selection=bool(not PARAMS['positive'] and any(p['candidates'] for p in perceptions)),real_time_claim=False,goal_complete=False))
 except BaseException as e:dump(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),frames=len(rows),perception_calls=len(perceptions),goal_complete=False));raise
 finally:
  if bridge is not None:dump(OUT/'semantic_queue_final.json',bridge.close(timeout_s=30.))
  for a in reversed(actors):
   try:
    if hasattr(a,'stop'):a.stop()
    a.destroy()
   except Exception:pass
  w.apply_settings(old)
if __name__=='__main__':main()
