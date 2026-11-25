"""正式感知运行前只检查官方车辆生成与独立静态几何；不读取检测或模型成绩。"""
from dataclasses import asdict
import json
import math
from pathlib import Path

import carla
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from carla_expansion_truth_010 import finalize_truth
from conceptgraph.completion.protocol import GoalTruth
from conceptgraph.completion.stage9 import digest, dump
from conceptgraph.navigation.vehicle_geometry import VehicleSpec, VehicleState, footprint_polygon

R=Path(__file__).resolve().parents[1]


def main():
    bankfile=R/'carla_official/expansion_registration_010.json';bank=json.loads(bankfile.read_text())
    out=R/'carla_official/expansion_spawn_preflight_010';out.mkdir(exist_ok=False)
    client=carla.Client('127.0.0.1',2010);client.set_timeout(120);world=client.reload_world(False);client.set_timeout(30)
    assert client.get_server_version()=='0.9.16' and 'Town10HD_Opt' in world.get_map().name
    old=world.get_settings();settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.1;settings.no_rendering_mode=False;world.apply_settings(settings)
    spec_release=R/'carla_official/guarded_complex_navigation_008/complex_second_nearest20/rule/release.json'
    previous=json.loads(spec_release.read_text());spec=VehicleSpec(**previous['vehicle_spec']);rear_offset=previous['rear_axle_offset_from_actor_center_m']
    bp=world.get_blueprint_library();rows=[]
    try:
        for params in [p for p in bank['episodes'] if p['weather']=='ClearNoon']:
            for seed in bank['seeds']:
                actors=[];row=dict(episode=params['episode'],seed=seed,official_vehicle_spawn_passed=False,independent_truth_passed=False)
                try:
                    start=world.get_map().get_spawn_points()[params['spawn_index']];wp=world.get_map().get_waypoint(start.location);waypoints=[wp]
                    for _ in range(65):waypoints.append(sorted(waypoints[-1].next(1.),key=lambda p:(p.road_id,p.section_id,p.lane_id,p.s))[0])
                    ego=world.spawn_actor(bp.find('vehicle.toyota.prius'),start);actors.append(ego);ego.set_simulate_physics(False)
                    def spawn(index,blueprint,lateral,yaw_offset=0.):
                        wp=waypoints[index];t=carla.Transform(wp.transform.location,wp.transform.rotation);t.location.z+=.3;right=t.get_right_vector();t.location.x+=lateral*right.x;t.location.y+=lateral*right.y;t.rotation.yaw+=yaw_offset
                        actor=world.spawn_actor(bp.find(blueprint),t);actors.append(actor);actor.set_simulate_physics(False);return actor
                    target=spawn(params['target_waypoint_index'],params['target_blueprint'],bank['seed_perturbations'][str(seed)]['target_lateral_m'])
                    other=spawn(params['distractor_waypoint_index'],params['distractor_blueprint'],params['distractor_lateral_right_m'],params['distractor_yaw_offset_deg']) if params['spawn_distractor'] else None
                    def polygon(actor):
                        return Polygon([(v.x,-v.y) for v in actor.bounding_box.get_world_vertices(actor.get_transform())]).convex_hull
                    origin=np.array([start.location.x,-start.location.y]);yaw=-math.radians(start.rotation.yaw);first=origin-8*np.array([math.cos(yaw),math.sin(yaw)])
                    points=[first.tolist()]+[[p.transform.location.x,-p.transform.location.y] for p in waypoints];line=LineString(points);road=line.buffer(min(p.lane_width for p in waypoints)/2,cap_style=2)
                    tf=target.get_transform();center=np.array([tf.location.x,-tf.location.y]);goal_s=line.project(Point(center))-8.;goal=line.interpolate(goal_s);delta=np.array(line.interpolate(goal_s+.1).coords[0])-np.array(line.interpolate(goal_s-.1).coords[0]);heading=math.atan2(delta[1],delta[0]);region=Point(goal.x,goal.y).buffer(1.3).intersection(road)
                    target_poly=polygon(target);other_poly=polygon(other) if other is not None else Polygon();checks=[]
                    begin=line.project(Point(origin-rear_offset*np.array([math.cos(yaw),math.sin(yaw)])))
                    for progress in np.arange(begin,goal_s+.05,.2):
                        xy=line.interpolate(progress);d=np.array(line.interpolate(progress+.05).coords[0])-np.array(line.interpolate(max(0,progress-.05)).coords[0]);a=math.atan2(d[1],d[0]);body=footprint_polygon(VehicleState(xy.x,xy.y,a),spec)
                        checks.append(bool(road.covers(body) and not body.intersects(target_poly) and not body.intersects(other_poly)))
                    reachable=bool(checks and all(checks))
                    truth=GoalTruth(params['episode'],(str(target.id),) if params['positive'] else (),({'instance_id':str(target.id),'polygon_xy':list(region.exterior.coords),'heading_rad':heading},) if params['positive'] else (),world.get_map().name,(0.,params['timeout_s']),'官方预检对象与声明道路；不使用任何模型或规划结果',reachability='reachable' if reachable and params['positive'] else 'unknown',timeout_s=params['timeout_s'])
                    gt=dict(other_controlled_actors=[dict(instance_id=str(other.id),category=params['distractor_category'])] if other else [])
                    truth,gt=finalize_truth(world,gt,truth,params,road,region,origin,other_poly)
                    valid=(truth.reachability=='reachable' if params['task_kind']=='positive' else truth.reachability=='inaccessible' if params['task_kind']=='blocked_target' else gt['no_match_scope_certificate']['verified_no_bus_in_declared_scope'])
                    row.update(official_vehicle_spawn_passed=True,independent_truth_passed=valid,truth=asdict(truth),certificates=gt,
                        centerline_certificate_steps=len(checks),centerline_certificate_all_legal=reachable,
                        target_polygon_xy=list(target_poly.exterior.coords),other_polygon_xy=list(other_poly.exterior.coords),
                        declared_road_polygon_xy=list(road.exterior.coords),origin_xy=origin.tolist())
                except RuntimeError as exc:
                    row['official_spawn_error']=str(exc)
                finally:
                    for actor in reversed(actors):actor.destroy()
                    world.tick(30)
                rows.append(row);dump(out/f'case_{len(rows):03d}.json',row)
                print(json.dumps({k:row[k] for k in ['episode','seed','official_vehicle_spawn_passed','independent_truth_passed']},ensure_ascii=False),flush=True)
    finally:
        world.apply_settings(old)
    report=dict(cases=len(rows),spawn_passed=sum(r['official_vehicle_spawn_passed'] for r in rows),truth_passed=sum(r['independent_truth_passed'] for r in rows),all_passed=len(rows)==75 and all(r['official_vehicle_spawn_passed'] and r['independent_truth_passed'] for r in rows),actual_navigation_episodes=0,actual_perception_frames=0,actual_Qwen_calls=0,
        scope='固定Town10、5起点、5任务几何、3偏移种子；两种天气共享几何，只预检ClearNoon。静态预检不证明物理执行/感知成功。',
        source_sha256={str(p):digest(p) for p in [Path(__file__),bankfile,spec_release]},goal_complete=False)
    dump(out/'report.json',report)
    if not report['all_passed']:raise RuntimeError('预设生成或独立真值证书不全，不启动导航性能运行。')


if __name__=='__main__':main()
