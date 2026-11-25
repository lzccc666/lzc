"""只读官方地图，为10配置50任务扩展取得实际道路信息；不推进当前回合。"""
from dataclasses import asdict
import datetime
import json
import math
from pathlib import Path

import carla
import numpy as np
from shapely.geometry import LineString, Point

from conceptgraph.completion.stage9 import digest, dump
from conceptgraph.navigation.vehicle_geometry import VehicleSpec, VehicleState, footprint_polygon

R = Path(__file__).resolve().parents[1]


def main():
    out = R/'carla_official/task_geometry_inventory_001'
    out.mkdir(exist_ok=False)
    registration = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=digest(__file__), spawn_indices=[0, 1, 2, 3, 4],
        planned_weathers=['ClearNoon', 'CloudyNoon'], waypoint_spacing_m=1.,
        maximum_waypoints=66, branch_selection='road_id/section_id/lane_id/s从小到大第一项',
        read_only_API_calls=['get_server_version', 'get_world', 'get_map', 'get_blueprint_library',
            'get_spawn_points', 'get_waypoint', 'waypoint.next'],
        forbidden_calls=['tick', 'wait_for_tick', 'apply_settings', 'reload_world',
            'set_weather', 'spawn_actor', 'set_transform', 'apply_control'],
        actual_new_navigation_episodes=0, goal_complete=False)
    dump(out/'registration.json', registration)
    client = carla.Client('127.0.0.1', 2010)
    client.set_timeout(15.)
    assert client.get_server_version() == '0.9.16'
    world = client.get_world()
    road_map = world.get_map()
    assert 'Town10HD_Opt' in road_map.name
    library = world.get_blueprint_library()
    blueprints = sorted(b.id for b in library.filter('vehicle.*'))
    dump(out/'official_vehicle_blueprints.json', blueprints)
    prior = json.loads((R/'carla_official/prius_controller_navigation_005/car20/full_rule_geometry/release.json').read_text())
    spec = VehicleSpec(**prior['vehicle_spec'])
    offset = prior['rear_axle_offset_from_actor_center_m']
    cases = []
    for index in registration['spawn_indices']:
        spawn = road_map.get_spawn_points()[index]
        wp = road_map.get_waypoint(spawn.location)
        route = [wp]
        for _ in range(65):
            choices = sorted(route[-1].next(1.), key=lambda p:(p.road_id,p.section_id,p.lane_id,p.s))
            if not choices:
                break
            route.append(choices[0])
        origin = np.array([spawn.location.x, -spawn.location.y])
        yaw = -math.radians(spawn.rotation.yaw)
        beginning = origin-8.*np.array([math.cos(yaw), math.sin(yaw)])
        xy = [beginning.tolist()]+[[wp.transform.location.x, -wp.transform.location.y] for wp in route]
        line = LineString(xy)
        width = min(wp.lane_width for wp in route)
        road = line.buffer(width/2, cap_style=2)
        initial = origin-offset*np.array([math.cos(yaw), math.sin(yaw)])
        start_s = line.project(Point(initial))
        certificates = []
        for target_index in [18, 20, 22]:
            if len(route) <= target_index:
                certificates.append(dict(target_index=target_index, usable=False, reason='frozen_waypoint_not_available'))
                continue
            target = route[target_index].transform.location
            target_s = line.project(Point(target.x, -target.y))
            goal_s = target_s-8.
            probes = []
            for progress in np.arange(start_s, goal_s+.05, .2):
                point = line.interpolate(progress)
                a, b = line.interpolate(max(0., progress-.05)), line.interpolate(min(line.length, progress+.05))
                angle = math.atan2(b.y-a.y, b.x-a.x)
                probes.append(road.covers(footprint_polygon(VehicleState(point.x, point.y, angle), spec)))
            certificates.append(dict(target_index=target_index,
                whole_body_centerline_within_declared_road=bool(probes and all(probes)),
                tested_poses=len(probes), desired_goal_progress_m=goal_s,
                scope='仅道路整车静态几何；未测试目标车碰撞、曲率/转向率、控制、感知、可到达性或成功'))
        z = np.array([wp.transform.location.z for wp in route])
        case = dict(spawn_index=index, initial_pose_carla=dict(x=spawn.location.x,y=spawn.location.y,
            z=spawn.location.z,pitch=spawn.rotation.pitch,yaw=spawn.rotation.yaw,roll=spawn.rotation.roll),
            waypoints=[dict(x=wp.transform.location.x,y=wp.transform.location.y,z=wp.transform.location.z,
                yaw_deg=wp.transform.rotation.yaw,lane_width_m=wp.lane_width,road_id=wp.road_id,
                lane_id=wp.lane_id,section_id=wp.section_id,s=wp.s) for wp in route],
            minimum_lane_width_m=width, centerline_xy=xy, road_polygon_xy=list(road.exterior.coords),
            ground_height_min_m=float(z.min()),ground_height_max_m=float(z.max()),
            first_40m_ground_range_m=float(np.ptp(z[:41])),
            static_road_certificates=certificates, actual_navigation_executed=False)
        dump(out/f'spawn_{index:03d}.json', case)
        cases.append(case)
    configs = [dict(config_id=f'town10_spawn{c["spawn_index"]:03d}_{weather}',
        spawn_index=c['spawn_index'], weather=weather,
        prepared_tasks=[dict(kind='positive',target_index=i,query='靠近汽车') for i in [18,20,22]]
            +[dict(kind='no_match',query='靠近公交车',requires_controlled_scope_and_world_label_check=True),
              dict(kind='blocked_target',query='靠近卡车',requires_independent_barrier_proof=True)],
        navigation_performance_not_observed=True) for c in cases for weather in registration['planned_weathers']]
    dump(out/'prepared_configuration_candidates.json', configs)
    dump(out/'report.json', dict(official_server_version=client.get_server_version(), map=road_map.name,
        actual_read_only_spawn_inventories=len(cases), prepared_configuration_candidates=len(configs),
        prepared_task_candidates=len(configs)*5, intended_future_seeds=[17,29,43],
        intended_future_episode_candidates=len(configs)*5*3,
        vehicle_blueprint_count=len(blueprints), vehicle_models_available={k:k in blueprints
            for k in ['vehicle.toyota.prius','vehicle.tesla.model3','vehicle.carlamotors.carlacola']},
        original_vehicle_geometry=asdict(spec), no_simulation_clock_or_world_mutation_called=True,
        unknowns=['每个不可达任务还需真实放置障碍后的独立几何断连证明',
            '无匹配任务需明确受控范围与原生世界未标注道具限制',
            '全车曲率、相机感知、闭环控制和实际到达不能由此道路证书推断',
            '候选配置尚未冻结成正式性能测试注册，不计已完成配置/任务'],
        actual_new_GPU_inference_calls=0, actual_new_navigation_episodes=0, goal_complete=False))
    print('已只读取得5个实际官方起点与10配置/50任务候选；没有推进或重载当前世界', flush=True)


if __name__ == '__main__':
    main()
