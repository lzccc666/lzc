"""独立重算终区、身份、航向、停车保持及执行限值，不信任自报ARRIVED。"""
import argparse,json,math
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon,Point
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--episode',required=True);a=p.parse_args();E=R/'carla_official'/a.episode;O=E/'independent_audit_001';O.mkdir(exist_ok=False);release=json.loads((E/'release.json').read_text());gt=json.loads((E/'evaluation_only/truth.json').read_text());states=list(map(json.loads,(E/'execution.jsonl').open()));perceptions=list(map(json.loads,(E/'perception.jsonl').open()));T=gt['arrival_truth'];target=gt['target_actor_id'];regions=T['regions'];road=Polygon(release['prior']['road_polygon_xy']);spec=release['vehicle_spec'];cursor=0;held=None;maxheld=0;ever_arrived=False;rows=[];errors=[];laneviolations=[];previous=None
 for r in states:
  while cursor+1<len(perceptions) and perceptions[cursor+1]['sensor_frame']<=r['frame']:cursor+=1
  obs=perceptions[cursor];sid=obs['selected_node_id'];candidates=[n for n in obs['nodes'] if n['category']=='car'];dist=[np.linalg.norm(np.asarray(n['center'][:2])-gt['center_xy']) for n in candidates];nearest=candidates[int(np.argmin(dist))]['node_id'] if dist and min(dist)<=2.5 else None;correct=sid is not None and sid==nearest
  matrix=np.asarray(r['ego_transform_carla']);yaw=-math.atan2(matrix[1,0],matrix[0,0]);xy=matrix[:2,3]*[1,-1]-1.4*np.array([math.cos(yaw),math.sin(yaw)]);s=r['state'];errors.append(float(np.linalg.norm(xy-[s['x'],s['y']])));inside=False;heading=False
  for region in regions:
   if region['instance_id']==target and Polygon(region['polygon_xy']).covers(Point(xy)):
    inside=True;heading=abs(math.atan2(math.sin(yaw-region['heading_rad']),math.cos(yaw-region['heading_rad'])))<=T['max_heading_error_rad']
  # 由官方实际车身尺寸和原始车体变换重建整车多边形。
  ex=(spec['front_from_rear_axle_m']+spec['rear_overhang_m'])/2;ey=spec['width_m']/2;local=np.array([[-ex,-ey],[ex,-ey],[ex,ey],[-ex,ey]]);c,ss=math.cos(yaw),math.sin(yaw);body=Polygon(local@np.array([[c,ss],[-ss,c]])+matrix[:2,3]*[1,-1]);legal=road.buffer(.05).covers(body)
  if not legal:laneviolations.append(r['tick'])
  collision=bool(r['collisions']);stopped=s['speed_mps']<=T['max_speed_mps'];valid=correct and inside and heading and stopped and not collision and legal and T['reachability']=='reachable';time_s=s['time_s']
  if valid and held is None:held=time_s
  if not valid:held=None
  duration=0 if held is None else time_s-held;maxheld=max(maxheld,duration);arrived=duration>=T['hold_s'] and time_s<=T['timeout_s'];ever_arrived|=arrived
  rows.append(dict(tick=r['tick'],time_s=time_s,correct_identity=correct,inside=inside,heading_ok=heading,stopped=stopped,body_legal=legal,held_s=duration,geometric_arrival=arrived))
 t=np.array([r['state']['time_s'] for r in states]);v=np.array([r['state']['speed_mps'] for r in states]);steering=np.array([r['state']['steering_rad'] for r in states]);acc=np.diff(v)/np.diff(t);rate=np.abs(np.diff(steering)/np.diff(t));limits=dict(speed_violations=int((v>spec['max_speed_mps']+1e-5).sum()),acceleration_violations=int((acc>spec['max_acceleration_mps2']+1e-5).sum()),braking_violations=int((acc < -spec['max_braking_mps2']-1e-5).sum()),steering_rate_violations=int((rate>spec['max_steering_rate_radps']+1e-5).sum()),steering_limit_violations=int((np.abs(steering)>spec['max_steering_rad']+1e-5).sum()))
 dump(O/'report.json',dict(episode=a.episode,geometric_identity_stop_arrival=bool(ever_arrived),maximum_contiguous_hold_s=maxheld,pose_reconstruction_max_error_m=max(errors),lane_violation_ticks=laneviolations,actuation_limits=limits,minimum_acceleration_mps2=float(acc.min()),maximum_acceleration_mps2=float(acc.max()),strict_safe_arrival=bool(ever_arrived and not any(limits.values()) and not laneviolations and not any(r['collisions'] for r in states)),inputs_sha256={str(p):digest(p) for p in [E/'release.json',E/'execution.jsonl',E/'perception.jsonl',E/'evaluation_only/truth.json']},evaluator_source_sha256=digest(__file__),limitation='目前state.steering_rad由命令换算，未单独保存实际车轮转角；仍需补全轮角观测，不能据此验收实际转角限值',goal_complete=False))
 with (O/'frames.jsonl').open('x') as f:
  for r in rows:f.write(json.dumps(r)+'\n')
 print((O/'report.json').read_text())
if __name__=='__main__':main()
