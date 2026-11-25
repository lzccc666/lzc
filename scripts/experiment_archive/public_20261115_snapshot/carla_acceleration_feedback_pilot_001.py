"""官方CARLA低容量纵向执行器开发标定：固定转向输入，真实物理，不给导航信用。"""
import argparse,gzip,hashlib,json,math,shutil,traceback
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];O=R/'carla_official/acceleration_feedback_pilot_001'
SOURCES={case:R/f'carla_official/ablation_batch_003/{case}/full_rule_geometry/execution.jsonl' for case in ['car20','car30_curve']}

def guard(reserve=0):
 size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
 if size+reserve>=10*2**30 or shutil.disk_usage(R).free<20*2**30:raise OSError('项目10GiB/物理空闲20GiB保护')

class AccelerationFeedback:
 def __init__(self):self.throttle=0.
 def step(self,requested_acceleration,measured_acceleration,speed,desired_speed,stop):
  if stop:self.throttle=max(0.,self.throttle-.02)
  else:
   correction=float(np.clip(.5*(requested_acceleration-measured_acceleration)*.1,-.03,.005))
   self.throttle=float(np.clip(self.throttle+correction,0.,.8))
   if measured_acceleration>max(.08,requested_acceleration*.5):self.throttle=max(0.,self.throttle-.025)
   if speed>desired_speed+.03:self.throttle=max(0.,self.throttle-.02)
  return self.throttle,1. if stop and speed<.08 else 0.

def prepare():
 guard(16*1024);O.mkdir(exist_ok=False)
 dump(O/'registration.json',dict(source_sha256={str(Path(__file__)):digest(__file__),**{str(p):digest(p) for p in SOURCES.values()}},official_version='0.9.16',town='Town10HD_Opt',seed=17,weather='ClearNoon',traffic_density=0,spawn_index=0,vehicle='vehicle.tesla.model3',torque_cap_Nm=25,fixed_dt_s=.1,policies=['legacy_speed_slew','acceleration_feedback'],cases=list(SOURCES),phases='10ticks刹车静置，400ticks目标速度0.55，100ticks滑行停车；每策略/转向条件均重载世界；4组共2040ticks',steering='前400个已有开发完整配置的实际applied_control.steer，后100ticks保持末值；作为相同外部转向输入，非重跑原导航',acceleration_request='clip(0.6*(desired_speed-measured_speed),-0.5,0.2)，旧执行器不用该量，新执行器显式闭环使用；参数在新物理测试前冻结',hard_acceptance=dict(max_acceleration_mps2=1.,max_braking_mps2=2.,minimum_peak_speed_mps=.25,minimum_path_distance_m=2.,final_speed_max_mps=.08,no_collision=True),output_columns=['tick','frame','snapshot_frame','official_timestamp','vx','vy','vz','x','y','z','yaw_deg','front_left_deg','front_right_deg','applied_throttle','applied_brake','applied_steer','requested_acceleration','measured_speed','measured_acceleration','reported_throttle','reported_brake','reported_gear'],scope='开发执行器标定；仅官方位姿/轮角/速度/碰撞，无RGB/深度/LiDAR前端；不得替代完整传感器闭环、目标选择、路线或实际到达',budget_reserve_bytes=300*1024,goal_complete=False))
 print('actuator pilot frozen; no physical run yet',flush=True)

def run():
 import carla
 registration=json.loads((O/'registration.json').read_text());assert all(digest(p)==h for p,h in registration['source_sha256'].items());guard(registration['budget_reserve_bytes']);client=carla.Client('127.0.0.1',2010);client.set_timeout(120);assert client.get_server_version()=='0.9.16';reports=[]
 for case,path in SOURCES.items():
  inputs=list(map(json.loads,path.open()))[:400];assert len(inputs)==400
  for policy in registration['policies']:
   world=client.reload_world(False);assert 'Town10HD_Opt' in world.get_map().name;old=world.get_settings();settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.1;settings.no_rendering_mode=False;world.apply_settings(settings);world.set_weather(carla.WeatherParameters.ClearNoon);actor=None;sensor=None;collisions=[];rows=[];controller=AccelerationFeedback();previous_speed=0.;previous_throttle=0.;previous_time=None;casepath=O/f'{case}_{policy}.jsonl.gz'
   try:
    actor=world.spawn_actor(world.get_blueprint_library().find('vehicle.tesla.model3'),world.get_map().get_spawn_points()[0]);physics=actor.get_physics_control();original_curve=[[q.x,q.y] for q in physics.torque_curve];physics.torque_curve=[carla.Vector2D(q.x,min(q.y,25.)) for q in physics.torque_curve];actor.apply_physics_control(physics);sensor=world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'),carla.Transform(),attach_to=actor);sensor.listen(lambda e:collisions.append(dict(frame=e.frame,other=e.other_actor.type_id,impulse=[e.normal_impulse.x,e.normal_impulse.y,e.normal_impulse.z])));control=carla.VehicleControl(brake=1.,manual_gear_shift=True,gear=1)
    dump(O/f'{case}_{policy}_release.json',dict(server=client.get_server_version(),client=client.get_client_version(),map=world.get_map().name,actual_torque_curve=[[q.x,q.y] for q in actor.get_physics_control().torque_curve],original_torque_curve=original_curve,mass=physics.mass,source_sha256=registration['source_sha256'],registration_sha256=digest(O/'registration.json'),actual_sensor_set=['official world snapshot pose/velocity','actual wheel angles','official collision'],scope=registration['scope'],goal_complete=False))
    with gzip.open(casepath,'xt',compresslevel=6) as output:
     for tick in range(510):
      if tick%100==0:guard(16*1024)
      reply=client.apply_batch_sync([carla.command.ApplyVehicleControl(actor.id,control)],False);assert not reply[0].error;frame=world.tick(30);snapshot=world.get_snapshot();assert snapshot.frame==frame;observed=snapshot.find(actor.id);assert observed is not None;velocity=observed.get_velocity();transform=observed.get_transform();speed=float(math.hypot(velocity.x,velocity.y));now=snapshot.timestamp.elapsed_seconds;measured_acceleration=(speed-previous_speed)/(now-previous_time) if previous_time is not None else 0.;driving=10<=tick<410;desired=.55 if driving else 0.;requested=float(np.clip(.6*(desired-speed),-.5,.2));steer=inputs[min(max(tick-9,0),399)]['applied_control']['steer'] if tick>=10 else 0.
      row=[tick,frame,snapshot.frame,now,velocity.x,velocity.y,velocity.z,transform.location.x,transform.location.y,transform.location.z,transform.rotation.yaw,actor.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel),actor.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel),control.throttle,control.brake,control.steer,requested,speed,measured_acceleration,actor.get_control().throttle,actor.get_control().brake,actor.get_control().gear];output.write(json.dumps(row,separators=(',',':'))+'\n');output.flush();rows.append(row)
      if policy=='legacy_speed_slew':
       target=float(np.clip(.75+.25*(desired-speed),0,.8)) if driving else 0.;throttle=float(np.clip(target,max(0.,previous_throttle-.02),previous_throttle+.005))
       if speed>=desired+.03:throttle=max(0.,previous_throttle-.02)
       brake=1. if not driving and speed<.08 else 0.
      else:throttle,brake=controller.step(requested,measured_acceleration,speed,desired,not driving)
      if tick<9:throttle,brake=0.,1.
      previous_speed=speed;previous_time=now;previous_throttle=throttle;control=carla.VehicleControl(manual_gear_shift=True,gear=1,throttle=throttle,brake=brake,steer=steer)
      if collisions:break
    x=np.asarray(rows);a=np.diff(x[:,17])/np.diff(x[:,3]);travel=float(np.linalg.norm(np.diff(x[:,7:9],axis=0),axis=1).sum());functional=bool(x[:,17].max()>=.25 and travel>=2 and x[-1,17]<=.08);safe=bool(a.max()<=1.+1e-5 and a.min()>=-2.-1e-5 and not collisions);report=dict(case=case,policy=policy,ticks=len(rows),complete_510=len(rows)==510,peak_speed_mps=float(x[:,17].max()),final_speed_mps=float(x[-1,17]),path_distance_m=travel,maximum_acceleration_mps2=float(a.max()),minimum_acceleration_mps2=float(a.min()),acceleration_violations=int((a>1.+1e-5).sum()),braking_violations=int((a< -2.-1e-5).sum()),collisions=collisions,functional_drive_and_stop=functional,actuation_constraints_satisfied=safe,combined_calibration_pass=functional and safe and len(rows)==510,raw_sha256=digest(casepath),actual_navigation_or_arrival_credit=False,goal_complete=False);dump(O/f'{case}_{policy}_report.json',report);reports.append(report);print(json.dumps(report,ensure_ascii=False),flush=True)
   except BaseException as exc:
    dump(O/f'{case}_{policy}_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),saved_ticks=len(rows),goal_complete=False));raise
   finally:
    if sensor is not None:sensor.stop();sensor.destroy()
    if actor is not None:actor.destroy()
    world.apply_settings(old)
 dump(O/'report.json',dict(actual_new_physical_calibrations=len(reports),results=reports,registration_sha256=digest(O/'registration.json'),limitations=['仅同一开发地图/起点的两组外生转向，非独立道路/种子泛化','速度、加减速与碰撞是官方物理实测；无目标对象/终区/真实到达信用','如果新策略更安全但过慢或不动，仍算标定未通过，不替代90s完整任务','未按新结果改参数或提高1m/s2门限；现有闭环失败保持不变'],full_navigation_rerun_passed=False,goal_complete=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args();globals()[a.action]()
