"""官方 CARLA 同步传感器与物理控制探针；不计入目标导航成功。"""
import hashlib,json,queue,time,traceback
from pathlib import Path
import carla
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'carla_official/sensor_control_probe_001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def get(q,frame):
 while True:
  x=q.get(timeout=30)
  if x.frame==frame:return x
  if x.frame>frame:raise ValueError(f'future sensor frame {x.frame}>{frame}')
def main():
 OUT.mkdir(exist_ok=False)
 c=carla.Client('127.0.0.1',2010);c.set_timeout(30)
 w=c.get_world();old=w.get_settings();actors=[];collisions=[];started=time.monotonic()
 points=w.get_map().get_spawn_points();start=points[0]
 release=dict(version=c.get_server_version(),client_version=c.get_client_version(),map=w.get_map().name,seed=17,fixed_delta_seconds=.05,frames=60,rgb=dict(width=640,height=360,fov=90),depth='official camera.depth; RGB coded meters',lidar=dict(channels=32,range_m=60,points_per_second=64000,rotation_frequency=20),vehicle='vehicle.tesla.model3',start=str(start),control='first 10 ticks brake; then throttle=0.3 straight',task='sensor_and_control_probe_only',arrival_credit=0,traffic_vehicles=0,source_sha256=sha(__file__),goal_complete=False)
 dump(OUT/'release.json',release)
 try:
  settings=w.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05;settings.no_rendering_mode=False;w.apply_settings(settings)
  w.set_weather(carla.WeatherParameters.ClearNoon)
  bp=w.get_blueprint_library();v=w.spawn_actor(bp.find('vehicle.tesla.model3'),start);actors.append(v)
  queues={}
  for kind,identifier,transform in [('rgb','sensor.camera.rgb',carla.Transform(carla.Location(x=1.5,z=1.8))),('depth','sensor.camera.depth',carla.Transform(carla.Location(x=1.5,z=1.8))),('lidar','sensor.lidar.ray_cast',carla.Transform(carla.Location(z=2.2)))]:
   b=bp.find(identifier)
   opts={'image_size_x':'640','image_size_y':'360','fov':'90'} if kind!='lidar' else {'channels':'32','range':'60','points_per_second':'64000','rotation_frequency':'20','sensor_tick':'0.05'}
   for k,val in opts.items():b.set_attribute(k,val)
   s=w.spawn_actor(b,transform,attach_to=v);actors.append(s);q=queue.Queue();s.listen(q.put);queues[kind]=q
  collision=w.spawn_actor(bp.find('sensor.other.collision'),carla.Transform(),attach_to=v);actors.append(collision)
  collision.listen(lambda e:collisions.append(dict(frame=e.frame,other_actor=e.other_actor.type_id,impulse=[e.normal_impulse.x,e.normal_impulse.y,e.normal_impulse.z])))
  rows=[]
  with (OUT/'frames.jsonl').open('x') as f:
   for idx in range(60):
    control=carla.VehicleControl(throttle=.3 if idx>=10 else 0,steer=0,brake=0 if idx>=10 else 1)
    v.apply_control(control);frame=w.tick(30)
    data={k:get(q,frame) for k,q in queues.items()};folder=OUT/'frames'/f'{idx:06d}';folder.mkdir(parents=True)
    rgb=np.frombuffer(data['rgb'].raw_data,np.uint8).reshape(360,640,4)[:,:,:3][:,:,::-1].copy();Image.fromarray(rgb).save(folder/'rgb.png')
    raw=np.frombuffer(data['depth'].raw_data,np.uint8).reshape(360,640,4).astype(np.uint32)
    depth=(raw[:,:,2]+raw[:,:,1]*256+raw[:,:,0]*65536).astype(np.float32)/16777215*1000
    np.save(folder/'depth_m.npy',depth);Image.fromarray((np.minimum(depth,65.535)*1000).astype(np.uint16)).save(folder/'depth_mm.png')
    (folder/'lidar.bin').write_bytes(data['lidar'].raw_data)
    t=v.get_transform();vel=v.get_velocity();row=dict(index=idx,frame=frame,timestamp=data['rgb'].timestamp,sensor_frames={k:x.frame for k,x in data.items()},ego_transform=t.get_matrix(),camera_transform=data['rgb'].transform.get_matrix(),lidar_transform=data['lidar'].transform.get_matrix(),velocity=[vel.x,vel.y,vel.z],control=dict(throttle=control.throttle,steer=control.steer,brake=control.brake),lidar_points=len(data['lidar']),files={str(p):sha(p) for p in folder.iterdir()},collision_count=len(collisions))
    f.write(json.dumps(row)+'\n');f.flush();rows.append(row)
    if idx%10==0:print(json.dumps(dict(index=idx,frame=frame,lidar_points=len(data['lidar']))),flush=True)
  xyz=np.array([r['ego_transform'] for r in rows])[:,:3,3]
  dump(OUT/'summary.json',dict(frames=len(rows),all_sensors_aligned=all(len(set(r['sensor_frames'].values()))==1 for r in rows),travel_distance_m=float(np.linalg.norm(xyz[-1]-xyz[0])),rgb_hash_changes=sum(list(a['files'].values())[0]!=list(b['files'].values())[0] for a,b in zip(rows,rows[1:])),collisions=collisions,wall_s=time.monotonic()-started,frames_sha256=sha(OUT/'frames.jsonl'),arrival_credit=0,closed_loop_navigation=False,goal_complete=False))
 except BaseException as e:
  dump(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),goal_complete=False));raise
 finally:
  for a in reversed(actors):
   try:
    if hasattr(a,'stop'):a.stop()
    a.destroy()
   except Exception:pass
  w.apply_settings(old)
if __name__=='__main__':main()
