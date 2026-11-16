"""从每个物理时刻原始整圈LiDAR回读保护输入与实际下一控制，不依赖summary宣称。"""
import argparse
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np
from shapely.geometry import Polygon
from conceptgraph.completion.stage9 import digest, dump
from conceptgraph.navigation.vehicle_geometry import VehicleSpec, VehicleState
from carla_lossless_jsonl_001 import read_rows, actual_log
from carla_motion_guard_001 import current_full_lidar_obstacles, nominal_coasting_check

R = Path(__file__).resolve().parents[1]


def audit(episode):
    dest = episode/'independent_motion_guard_audit_002'
    dest.mkdir(exist_ok=False)
    release = json.loads((episode/'release.json').read_text())
    observations = {r['frame_idx']*2:r for r in read_rows(episode,'perception.jsonl')}
    rows = read_rows(episode,'execution.jsonl')
    spec = VehicleSpec(**release['vehicle_spec'])
    road = Polygon(release['prior']['road_polygon_xy'])
    path, goal = [], None
    checked, guarded_stops, no_path_checks = [], 0, 0
    for row in rows:
        state = VehicleState(**row['state'])
        tracking = row['tracking']
        observation = tracking['current_motion_observation']
        saved = tracking['always_on_motion_guard']
        assert saved['always_executed'] and not saved['actual_physical_safety_proven']
        assert row['sensor_frames']['lidar'] == observation['sensor_frame'] == row['snapshot_frame']
        path_to_raw = (episode/observation['raw_file']).resolve()
        bulk=Path(release['bulk_sensor_root']).resolve()
        policy=json.loads((R/'json/sensor_bulk_authorization_003.json').read_text())
        assert digest(R/'json/sensor_bulk_authorization_003.json')==release['storage_authorization_sha256']
        assert bulk.is_relative_to(Path(policy['authorized_sensor_root']).resolve())
        assert path_to_raw.is_relative_to(bulk)
        raw = path_to_raw.read_bytes()
        if observation['raw_encoding'] == 'zlib_lossless_float32_xyzi':
            raw = zlib.decompress(raw)
        else:
            assert observation['raw_encoding'] == 'raw_float32_xyzi'
        assert hashlib.sha256(raw).hexdigest() == observation['raw_uncompressed_sha256']
        points = np.frombuffer(raw, np.float32).reshape(-1, 4)
        obstacles, source = current_full_lidar_obstacles(points,
            np.asarray(observation['lidar_pose_carla']), state, spec,
            sensor_frame=observation['sensor_frame'], snapshot_frame=row['snapshot_frame'])
        assert source['valid'] == observation['valid']
        predicted_polygons = [Polygon(p) for p in observation['obstacle_polygons_xy']]
        assert len(obstacles) == len(predicted_polygons)
        assert all(a.symmetric_difference(b).area < 1e-8
            for a, b in zip(obstacles, predicted_polygons))
        if row['tick'] in observations:
            obs = observations[row['tick']]
            path = obs['path_world']
            if obs['planning'] is not None:
                goal = obs['planning']['goal_xy']
            if obs.get('pending_semantics_search_suppressed'):
                goal = None
            reported = obs['known_obstacle_polygons']
            assert len(reported) == len(obstacles)
            assert all(Polygon(p).symmetric_difference(o).area < 1e-8
                for p, o in zip(reported, obstacles))
        follow = nominal_coasting_check(state, road, obstacles, spec,
            observation=observation, path=path, goal=goal)
        coast = nominal_coasting_check(state, road, obstacles, spec,
            observation=observation, path=None, goal=None)
        assert follow == saved['route_following_forecast']
        assert coast == saved['current_steering_coasting_forecast']
        must_stop = bool(follow['stop_required'] or coast['stop_required'])
        assert must_stop == saved['stop_required']
        no_path_checks += not bool(path)
        if must_stop:
            guarded_stops += 1
            assert tracking['current_motion_guard_stop'] and tracking['emergency_braking']
            assert tracking['desired_speed_mps'] == 0.
            assert row['next_control']['throttle'] == 0.
            expected_brake = 1. if state.speed_mps < .15 else 0.
            assert row['next_control']['brake'] == expected_brake
        if row['tick']+1 < len(rows):
            applied = rows[row['tick']+1]['applied_control']
            assert all(abs(applied[k]-row['next_control'][k]) < 1e-7
                for k in ['throttle', 'brake', 'steer'])
        checked.append(dict(tick=row['tick'], actual_sensor_frame=row['frame'],
            current_raw_sha256_verified=True, full360_obstacles_reconstructed=True,
            always_checked_without_path=not bool(path), stop_required=must_stop,
            command_respected_motion_guard=True))
    with (dest/'ticks.jsonl').open('x') as stream:
        for row in checked:
            stream.write(json.dumps(row)+'\n')
    report = dict(physical_ticks=len(rows), raw_LiDAR_frames_verified=len(checked),
        every_tick_guard_executed=True, every_tick_obstacles_reconstructed=True,
        nominal_forecasts_recomputed=True, actual_stop_commands_verified=guarded_stops,
        checks_without_route=no_path_checks, next_controls_match_next_actual_applied=True,
        physical_safety_requires_independent_arrival_report=True,
        actual_physical_safety_proven_by_this_audit=False,
        limitation='核验保护确实执行和输入/命令来源，不能把名义滑行模型当通用物理安全证明',
        source_sha256={str(p): digest(p) for p in [Path(__file__), episode/'release.json',
            actual_log(episode,'execution.jsonl'), actual_log(episode,'perception.jsonl'),
            R/'source/carla_motion_guard_001.py', R/'source/carla_guarded_control_001.py']},
        goal_complete=False)
    dump(dest/'report.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', required=True)
    args = parser.parse_args()
    audit(R/'carla_official'/args.episode)
