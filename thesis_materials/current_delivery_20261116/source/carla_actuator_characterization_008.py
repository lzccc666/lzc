import json,time,traceback
from pathlib import Path
import carla,numpy as np
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];O=R/'carla_official/actuator_characterization_008'
def main():
 O.mkdir(exist_ok=False);c=carla.Client('127.0.0.1',2010);c.set_timeout(30);w=c.get_world();old=w.get_settings();setting=w.get_settings();setting.synchronous_mode=True;setting.fixed_delta_seconds=.1;setting.no_rendering_mode=False;w.apply_settings(setting);actor=None;reports=[]
 dump(O/'release.json',dict(controls=[.8],ramp_per_tick=.01,release_controls=[(0.,0.)],physics_engine_torque_cap_Nm=55.,physical_basis='m*r*a/final_ratio with conservative margin; fixed before this calibration',phases='10 brake=1 settling;160 ramp throttle up to .8 manual first gear, torque cap55Nm;80 coast then brake below .08mps',dt=.1,spawn_index=0,server=c.get_server_version(),source_sha256=digest(__file__),not_navigation=True,goal_complete=False))
 try:
  with (O/'frames.jsonl').open('x') as f:
   for release_throttle,release_brake in [(0.,0.)]:
    throttle=.8;prev_speed=0.
    actor=w.spawn_actor(w.get_blueprint_library().find('vehicle.tesla.model3'),w.get_map().get_spawn_points()[0]);rows=[];physics=actor.get_physics_control();physics.torque_curve=[carla.Vector2D(v.x,min(v.y,55.)) for v in physics.torque_curve];actor.apply_physics_control(physics)
    dump(O/'applied_physics.json',dict(torque_curve=[[v.x,v.y] for v in actor.get_physics_control().torque_curve],mass=physics.mass,final_ratio=physics.final_ratio,wheels_brake_torque=[q.max_brake_torque for q in physics.wheels]))
    for tick in range(250):
     ctl=carla.VehicleControl(manual_gear_shift=True,gear=1,throttle=min(throttle,max(0.,(tick-9)*.01)) if 10<=tick<170 else 0.,brake=1 if tick<10 or (tick>=170 and prev_speed<.08) else 0.);answer=c.apply_batch_sync([carla.command.ApplyVehicleControl(actor.id,ctl)],False)
     if answer[0].error:raise RuntimeError(answer[0].error)
     frame=w.tick(30);snapshot=w.get_snapshot();a=snapshot.find(actor.id)
     if a is None or snapshot.frame!=frame:raise RuntimeError(f'snapshot mismatch {snapshot.frame}/{frame}')
     v=a.get_velocity();pos=a.get_transform().location;acc=a.get_acceleration();proxy=actor.get_velocity();r=dict(release_condition=[release_throttle,release_brake],throttle_condition=throttle,tick=tick,frame=frame,snapshot_frame=snapshot.frame,timestamp=snapshot.timestamp.elapsed_seconds,velocity=[v.x,v.y,v.z],speed=float(np.hypot(v.x,v.y)),position=[pos.x,pos.y,pos.z],reported_acceleration=[acc.x,acc.y,acc.z],proxy_velocity=[proxy.x,proxy.y,proxy.z],wheel_angle_deg=actor.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel),gear=actor.get_control().gear,reported_throttle=actor.get_control().throttle,reported_brake=actor.get_control().brake,control=dict(throttle=ctl.throttle,brake=ctl.brake));f.write(json.dumps(r)+'\n');f.flush();rows.append(r);prev_speed=r['speed']
    t=np.array([x['timestamp'] for x in rows]);v=np.array([x['speed'] for x in rows]);a=np.diff(v)/np.diff(t);reports.append(dict(release_control=[release_throttle,release_brake],throttle=throttle,max_speed_mps=float(v.max()),observed_accel_max_mps2=float(a[9:169].max()),observed_brake_min_mps2=float(a[169:].min()),snapshot_proxy_max_difference=float(max(np.linalg.norm(np.array(r['velocity'])-r['proxy_velocity']) for r in rows))));actor.destroy();actor=None
  dump(O/'summary.json',dict(reports=reports,source_records_sha256=digest(O/'frames.jsonl'),goal_complete=False));print(reports,flush=True)
 except BaseException as e:dump(O/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),goal_complete=False));raise
 finally:
  if actor is not None:actor.destroy()
  w.apply_settings(old)
if __name__=='__main__':main()
