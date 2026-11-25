"""官方CARLA开发episode：传感器→原有真实前端→对象/关系→规则选择→Hybrid A*→物理控制。"""
import copy,gzip,json,math,pickle,queue,time,traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import carla,cv2,numpy as np
from shapely.geometry import LineString,Polygon,Point
from conceptgraph.completion.stage9 import PROJECT,digest,dump
from conceptgraph.completion.interactive_navigation import PerceptionVehicleAgent,evaluation_identity,json_safe
import conceptgraph.completion.interactive_navigation as nav
from conceptgraph.completion.protocol import ArrivalEvaluator,GoalTruth
from conceptgraph.navigation.vehicle_geometry import VehicleSpec,VehicleState,footprint_polygon
from conceptgraph.scripts.streamlined_detections import load_detection_models,detect_rgb_file
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'carla_official/pipeline_episode_001'
S=np.diag([1.,-1.,1.,1.]);B=np.array([[0,0,1,0],[1,0,0,0],[0,-1,0,0],[0,0,0,1.]])
K=np.array([[320.,0,320],[0,320.,180],[0,0,1.]])
class Agent(PerceptionVehicleAgent):
 def approach_goal(self,node_id):
  obj=next(o for o in self.mapper.objects if o.get('stable_object_id')==self.nodes[node_id].metadata['object_identity'])
  progress=np.array([self.line.project(Point(p[:2])) for p in np.asarray(obj['pcd'].points)])
  goal_s=max(0.,float(np.quantile(progress,.05))-nav.SPEC.front_from_rear_axle_m-2.)
  from conceptgraph.completion.kinematic_benchmark import pose_on
  return pose_on(self.line,goal_s)
def sensor_get(q,frame,kind):
 while True:
  try:d=q.get(timeout=30)
  except queue.Empty:raise TimeoutError(f'{kind} missing at {frame}')
  if d.frame==frame:return d
  if d.frame>frame:raise ValueError('future sensor frame')
def main():
 OUT.mkdir(parents=True,exist_ok=False);started=time.monotonic();c=carla.Client('127.0.0.1',2010);c.set_timeout(30);w=c.get_world();old=w.get_settings();actors=[];collisions=[];rows=[];perceptions=[]
 try:
  settings=w.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.1;settings.no_rendering_mode=False;w.apply_settings(settings);w.set_weather(carla.WeatherParameters.ClearNoon)
  bp=w.get_blueprint_library();start=w.get_map().get_spawn_points()[0];wp=w.get_map().get_waypoint(start.location);waypoints=[wp]
  for _ in range(65):
   nxt=sorted(waypoints[-1].next(1.),key=lambda p:(p.road_id,p.section_id,p.lane_id,p.s))
   if not nxt:break
   waypoints.append(nxt[0])
  if len(waypoints)<40:raise ValueError('预选道路不足40m，保留失败不换路线')
  ego=w.spawn_actor(bp.find('vehicle.tesla.model3'),start);actors.append(ego)
  target_t=carla.Transform(waypoints[30].transform.location,waypoints[30].transform.rotation);target_t.location.z+=.3
  target=w.spawn_actor(bp.find('vehicle.tesla.model3'),target_t);actors.append(target);target.set_simulate_physics(False)
  for _ in range(10):w.tick(30)
  ebox=ego.bounding_box;spec=VehicleSpec(wheelbase_m=2.875,front_from_rear_axle_m=float(ebox.extent.x+1.4),rear_overhang_m=float(ebox.extent.x-1.4),width_m=float(2*ebox.extent.y),dt_s=.1);nav.SPEC=spec
  tf=ego.get_transform();yaw=-math.radians(tf.rotation.yaw);origin=np.array([tf.location.x,-tf.location.y]);first=origin-8*np.array([math.cos(yaw),math.sin(yaw)])
  points=np.array([first.tolist()]+[[p.transform.location.x,-p.transform.location.y] for p in waypoints]);line=LineString(points);lane_width=min(p.lane_width for p in waypoints);road=line.buffer(lane_width/2,cap_style=2)
  t=target.get_transform();truth_center=np.array([t.location.x,-t.location.y]);target_s=line.project(Point(truth_center));goal_s=target_s-8.;goal=line.interpolate(goal_s);delta=np.array(line.interpolate(goal_s+.1).coords[0])-np.array(line.interpolate(goal_s-.1).coords[0]);goal_yaw=math.atan2(delta[1],delta[0]);region=Point(goal.x,goal.y).buffer(1.3).intersection(road)
  accepted=str(target.id);target_polygon=Polygon([(v.x,-v.y) for v in target.bounding_box.get_world_vertices(t)]).convex_hull
  certificate=[]
  for progress in np.arange(line.project(Point(origin-1.4*np.array([math.cos(yaw),math.sin(yaw)]))),goal_s+.05,.2):
   xy=np.array(line.interpolate(progress).coords[0]);v1=np.array(line.interpolate(min(line.length,progress+.05)).coords[0])-np.array(line.interpolate(max(0,progress-.05)).coords[0]);a=math.atan2(v1[1],v1[0]);state=VehicleState(*xy,a);body=footprint_polygon(state,spec);certificate.append(bool(road.covers(body) and not body.intersects(target_polygon)))
  reachable=bool(certificate and all(certificate));truth=GoalTruth('carla_dev_town10_spawn0_car_001',(accepted,),({'instance_id':accepted,'polygon_xy':list(region.exterior.coords),'heading_rad':goal_yaw},),w.get_map().name,(0,45),'官方CARLA目标actor及冻结区域；道路中心线逐0.2m整车检查，仅已知固定目标碰撞几何',reachability='reachable' if reachable else 'unknown',timeout_s=45)
  gt=dict(target_actor_id=accepted,center_xy=truth_center.tolist(),category='car',target_polygon_xy=list(target_polygon.exterior.coords),arrival_truth=asdict(truth),independent_centerline_certificate=dict(steps=len(certificate),all_legal=reachable,limitation='校验已声明道路走廊和目标车；世界碰撞另由官方碰撞传感器评价'))
  dump(OUT/'evaluation_only/truth.json',gt)
  prior=dict(centerline_xy=points.tolist(),road_polygon_xy=list(road.exterior.coords));classes=(PROJECT/'conceptgraph/vehicle_classes.txt').read_text().splitlines()
  release=dict(server_version=c.get_server_version(),client_version=c.get_client_version(),map=w.get_map().name,weather='ClearNoon',seed=17,traffic_density=0,vehicle='vehicle.tesla.model3',vehicle_spec=asdict(spec),query='靠近汽车',condition='rule_full_pipeline',target_initial_location=str(target_t),target_speed_mps=0,initial_pose=str(start),fixed_dt_s=.1,maximum_sim_s=45,maximum_wall_s=600,perception_every_ticks=2,RGB=dict(width=640,height=360,fov=90),LiDAR=dict(channels=32,range_m=60,points_per_second=64000,rotation_hz=10,sensor_tick=0),camera_transform=[1.5,0,1.8],lidar_transform=[0,0,2.2],sensor_set=['rgb','depth','lidar','pose','collision'],prior=prior,source_sha256={str(p):digest(p) for p in [Path(__file__),PROJECT/'conceptgraph/completion/interactive_navigation.py',PROJECT/'conceptgraph/slam/streamlined_mapping.py',PROJECT/'conceptgraph/slam/mapping.py',PROJECT/'conceptgraph/slam/utils.py',PROJECT/'conceptgraph/scripts/streamlined_detections.py']},truth_sha256=digest(OUT/'evaluation_only/truth.json'),task_kind='development_integration_not_sealed',performance_credit=False,goal_complete=False)
  dump(OUT/'release.json',release)
  agent=Agent('靠近汽车',prior,classes,OUT);evaluator=ArrivalEvaluator(truth);cfg=SimpleNamespace(device='cuda:0');models=load_detection_models(cfg)
  for model in models[:2]:model.to('cuda:0');model.overrides['device']='cuda:0'
  queues={}
  for kind,identifier,tr in [('rgb','sensor.camera.rgb',carla.Transform(carla.Location(x=1.5,z=1.8))),('depth','sensor.camera.depth',carla.Transform(carla.Location(x=1.5,z=1.8))),('lidar','sensor.lidar.ray_cast',carla.Transform(carla.Location(z=2.2)))]:
   b=bp.find(identifier);options={'image_size_x':'640','image_size_y':'360','fov':'90','sensor_tick':'0.0'} if kind!='lidar' else {'channels':'32','range':'60','points_per_second':'64000','rotation_frequency':'10','sensor_tick':'0.0'}
   for k,v in options.items():b.set_attribute(k,v)
   a=w.spawn_actor(b,tr,attach_to=ego);actors.append(a);q=queue.Queue();a.listen(q.put);queues[kind]=q
  sensor=w.spawn_actor(bp.find('sensor.other.collision'),carla.Transform(),attach_to=ego);actors.append(sensor);sensor.listen(lambda e:collisions.append(dict(frame=e.frame,other=e.other_actor.type_id,impulse=[e.normal_impulse.x,e.normal_impulse.y,e.normal_impulse.z])))
  ctrl=carla.VehicleControl(brake=1);selected_id=None;correct_selection=False;correct_lock=False;route_generated=False;reached=False;final='TIMEOUT';lane_violation=False;last_state=None;first_time=None
  with (OUT/'execution.jsonl').open('x') as execution,(OUT/'perception.jsonl').open('x') as perception:
   for tick in range(450):
    if time.monotonic()-started>600:final='WALL_TIMEOUT';break
    ego.apply_control(ctrl);frame=w.tick(30);data={k:sensor_get(q,frame,k) for k,q in queues.items()};tf=ego.get_transform();vel=ego.get_velocity();angle=-math.radians(tf.rotation.yaw);center=np.array([tf.location.x,-tf.location.y]);rear=center-1.4*np.array([math.cos(angle),math.sin(angle)]);speed=math.sqrt(vel.x**2+vel.y**2);first_time=data['rgb'].timestamp if first_time is None else first_time;sim_t=data['rgb'].timestamp-first_time
    state=VehicleState(*rear,angle,speed,-ctrl.steer*spec.max_steering_rad,sim_t)
    body=Polygon([(v.x,-v.y) for v in ego.bounding_box.get_world_vertices(tf)]).convex_hull;lane_violation|=not road.buffer(.05).covers(body)
    if tick%2==0:
     folder=OUT/'frames'/f'{tick//2:06d}';folder.mkdir(parents=True);rgb=np.frombuffer(data['rgb'].raw_data,np.uint8).reshape(360,640,4)[:,:,:3][:,:,::-1].copy();cv2.imwrite(str(folder/'rgb.png'),rgb[:,:,::-1]);(folder/'lidar.bin').write_bytes(data['lidar'].raw_data)
     raw=np.frombuffer(data['depth'].raw_data,np.uint8).reshape(360,640,4).astype(np.uint32);dense=(raw[:,:,2]+raw[:,:,1]*256+raw[:,:,0]*65536).astype(np.float32)/16777215*1000;np.save(folder/'official_depth_m.npy',dense)
     optical=S@np.asarray(data['rgb'].transform.get_matrix())@B;lidar_w=np.asarray(data['lidar'].transform.get_matrix());lidar=np.frombuffer(data['lidar'].raw_data,np.float32).reshape(-1,4);points_l=np.column_stack([lidar[:,:3],np.ones(len(lidar))]);pc=(np.linalg.inv(optical)@S@lidar_w@points_l.T).T[:,:3];pc=pc[pc[:,2]>.1];uv=pc@K.T;uv=uv[:,:2]/uv[:,2,None];u=np.floor(uv[:,0]).astype(int);v=np.floor(uv[:,1]).astype(int);valid=(u>=0)&(u<640)&(v>=0)&(v<360)&(pc[:,2]<65);u,v,z=u[valid],v[valid],pc[valid,2];depth=np.full((360,640),np.inf);np.minimum.at(depth,(v,u),z);depth[~np.isfinite(depth)]=0;cv2.imwrite(str(folder/'lidar_depth_mm.png'),(depth*1000).astype(np.uint16))
     t0=time.monotonic();gobs,vis=detect_rgb_file(cfg,models,folder/'rgb.png',classes);cv2.imwrite(str(folder/'detections.jpg'),vis)
     with gzip.open(folder/'detections.pkl.gz','xb') as f:pickle.dump(gobs,f)
     result=agent.perceive(tick//2,state,rgb,depth,optical,folder/'rgb.png',gobs);result['measured_wall_s']=time.monotonic()-t0;result['sensor_frame']=frame;result['files_sha256']={str(p):digest(p) for p in folder.iterdir()};result['camera_pose_cv']=optical.tolist();result['lidar_pose_carla']=lidar_w.tolist();result['input_depth_source']='current official LiDAR projected into current RGB; dense official depth retained only for diagnostic';perception.write(json.dumps(result,ensure_ascii=False,default=json_safe)+'\n');perception.flush();perceptions.append(result)
     selected_id,_=evaluation_identity(agent.nodes,[dict(instance_id=accepted,center_xy=truth_center.tolist(),category='car')],agent.selected);correct_selection|=any(evaluation_identity(agent.nodes,[dict(instance_id=accepted,center_xy=truth_center.tolist(),category='car')],a['node_id'])[0]==accepted for a in result['candidates']);correct_lock|=selected_id==accepted;route_generated|=bool(agent.path and agent.path_purpose=='target_approach')
     print(json.dumps(dict(tick=tick,nodes=len(agent.nodes),candidates=len(result['candidates']),selected=agent.selected,selected_correct=selected_id==accepted,path=len(agent.path),speed=speed)),flush=True)
    if speed>spec.max_speed_mps:acceleration,steering,tracking=-2.,0.,dict(reason='measured_speed_limit_braking')
    else:acceleration,steering,tracking=agent.control(state)
    if collisions or lane_violation:acceleration=-2.
    throttle=float(np.clip(.08+.18*(1.2-speed),0,.25)) if acceleration>0 else 0.;brake=float(np.clip(-acceleration/2,0,1)) if acceleration<0 else (min(1.,(speed-1.5)*.5) if speed>1.5 else 0.)
    ctrl=carla.VehicleControl(throttle=throttle,steer=float(np.clip(-steering/spec.max_steering_rad,-1,1)),brake=brake)
    evaluation=evaluator.update(time_s=sim_t,xy=rear.tolist(),yaw_rad=angle,speed_mps=speed,evaluated_instance_id=selected_id,body_collision=bool(collisions),lane_violation=lane_violation)
    row=dict(tick=tick,frame=frame,sensor_frames={k:x.frame for k,x in data.items()},state=asdict(state),ego_transform_carla=tf.get_matrix(),next_control=dict(throttle=ctrl.throttle,steer=ctrl.steer,brake=ctrl.brake),tracking=tracking,evaluation=evaluation,collisions=copy.deepcopy(collisions));execution.write(json.dumps(row,ensure_ascii=False,default=json_safe)+'\n');execution.flush();rows.append(row);last_state=state
    if evaluation['actually_arrived']:reached=True;final='ARRIVED';break
    if collisions or lane_violation:final='COLLISION' if collisions else 'LANE_VIOLATION';break
  ego.apply_control(carla.VehicleControl(brake=1));w.tick(30)
  dump(OUT/'summary.json',dict(server_version=c.get_server_version(),status=final,frames=len(rows),perception_calls=len(perceptions),route_generated=route_generated,correct_object_selected=correct_selection,correctly_locked=correct_lock,actually_arrived=reached,collisions=collisions,lane_violation=lane_violation,independent_reachability_certificate=reachable,wall_s=time.monotonic()-started,sim_s=None if last_state is None else last_state.time_s,execution_sha256=digest(OUT/'execution.jsonl'),perception_sha256=digest(OUT/'perception.jsonl'),GT_in_agent=False,rule_only=True,Qwen_evaluated=False,real_time_claim=False,goal_complete=False))
 except BaseException as e:dump(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),frames=len(rows),perception_calls=len(perceptions),goal_complete=False));raise
 finally:
  for a in reversed(actors):
   try:
    if hasattr(a,'stop'):a.stop()
    a.destroy()
   except Exception:pass
  w.apply_settings(old)
if __name__=='__main__':main()
