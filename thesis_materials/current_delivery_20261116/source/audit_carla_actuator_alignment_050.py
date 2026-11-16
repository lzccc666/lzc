"""对既有官方标定和真实导航轨迹核查制动/转向假设；不是新仿真或到达实验。"""
from collections import Counter
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

from carla_prius_cutoff_pilot_008 import AccelerationFeedback
from carla_stock_stopping_profile_049 import StockStoppingProfile, observed_approach_profile
from conceptgraph.navigation.vehicle_geometry import VehicleState
from diagnose_carla_ten_bottlenecks_043 import sha

R = Path(__file__).resolve().parents[1]


def rows(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def main():
    out = R / 'carla_official/actuator_alignment_050'
    out.mkdir(exist_ok=False)
    pilot = R / 'carla_official/prius_cutoff_pilot_008'
    profile = StockStoppingProfile()
    inputs = {str(p): sha(p) for p in [
        R / 'source/carla_stock_stopping_profile_049.py',
        R / 'source/carla_prius_cutoff_pilot_008.py',
        R / 'source/audit_carla_actuator_alignment_050.py',
        pilot / 'registration.json', pilot / 'report.json']}
    pilot_files = sorted(pilot.glob('*.jsonl.gz'))
    for p in pilot_files:
        inputs[str(p)] = sha(p)
    jobs = []
    for number in (1, 2, 5, 6, 10):
        jp = R / f'json/expansion_jobs_011_primary/job_{number:03d}.json'
        j = json.loads(jp.read_text())
        d = Path(j['episode_directory'])
        for p in [jp, d / 'release.json', d / 'execution.jsonl.gz', d / 'perception.jsonl.gz']:
            inputs[str(p)] = sha(p)
        jobs.append((number, d))
    protocol = dict(inputs_sha256=inputs, pilot_cases='all2_existing_510tick_cutoff008_cases',
        navigation_cases=[j[0] for j in jobs], frozen_profile_parameters=vars(profile),
        stop_threshold_mps=.08, actual_new_CARLA_budget=0, actual_new_GPU_budget=0,
        output_budget_bytes=32*1024**2,
        claims='实际停车距离/命令重放/本帧速度建议；不授予新导航、消融或安全保证',
        samples_not_independent_scenes=True, goal_complete=False)
    with (out / 'protocol.json').open('x') as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)
    calibration = []
    for p in pilot_files:
        data = list(rows(p))
        assert len(data) == 510
        controller = AccelerationFeedback()
        errors = []
        for i, r in enumerate(data[:-1]):
            tick = r[0]
            driving = 10 <= tick < 410
            throttle, brake = controller.step(r[16], r[18], r[17], .55 if driving else 0., not driving)
            if tick < 9:
                throttle, brake = 0., 1.
            errors += [abs(throttle-data[i+1][13]), abs(brake-data[i+1][14])]
        # 410时刻请求切油，在后续已实际施加零油门的快照上开始计算。
        start = next(i for i, r in enumerate(data) if r[0] >= 410 and r[13] == 0.)
        stopped = next(i for i in range(start, len(data)) if data[i][17] <= .08)
        braked = next(i for i in range(start, len(data)) if data[i][14] > 0.)
        distance = sum(math.dist(data[i-1][7:9], data[i][7:9]) for i in range(start+1, stopped+1))
        v = data[start][17]
        ideal = max(0., v*v-.08**2) / 4.
        calibration.append(dict(case=p.name, applied_cutoff_tick=data[start][0],
            speed_at_applied_cutoff_mps=v, first_applied_brake_tick=data[braked][0],
            measured_time_until_speed_008_s=data[stopped][3]-data[start][3],
            measured_travel_until_speed_008_m=distance,
            idealized_immediate_2mps2_distance_to_008_m=ideal,
            measured_to_idealized_distance_ratio=distance/ideal,
            nominal_profile_distance_including_reserve_m=profile.distance(v),
            nominal_profile_covers_this_measured_stop=profile.distance(v) >= distance,
            replayed_next_command_pairs=len(data)-1,
            maximum_controller_replay_error=max(errors)))
        assert max(errors) < 1.e-7
    navigation = []
    with (out / 'navigation_frame_advice.csv').open('x', newline='', encoding='utf-8-sig') as f:
        writer = None
        for number, d in jobs:
            release = json.loads((d / 'release.json').read_text())
            wheelbase = release['vehicle_spec']['wheelbase_m']
            wheelmax = math.radians(release['steering_calibration']['physical_max_steer_deg'])
            obs = iter(rows(d / 'perception.jsonl.gz'))
            current = next(obs)
            following = next(obs, None)
            goal = None
            previous = None
            yaw_errors = []
            steering_errors = []
            counts = Counter()
            for e in rows(d / 'execution.jsonl.gz'):
                while following is not None and following['sensor_frame'] <= e['frame']:
                    current = following
                    following = next(obs, None)
                if current['planning'] and current['planning'].get('goal_xy'):
                    goal = current['planning']['goal_xy']
                state = VehicleState(**e['state'])
                if previous is not None:
                    before = previous['state']
                    dt = state.time_s-before['time_s']
                    travel = (state.speed_mps+before['speed_mps'])/2*dt
                    observed_yaw = math.atan2(math.sin(state.yaw_rad-before['yaw_rad']), math.cos(state.yaw_rad-before['yaw_rad']))
                    predicted_yaw = travel*math.tan((state.steering_rad+before['steering_rad'])/2)/wheelbase
                    yaw_errors.append(abs(observed_yaw-predicted_yaw))
                expected_steering = -e['applied_control']['steer']*wheelmax
                steering_errors.append(abs(state.steering_rad-expected_steering))
                previous = e
                counts['physical_ticks'] += 1
                if current['path_purpose'] != 'target_approach' or not current['path_world'] or goal is None:
                    continue
                original = min(.55, float(e['tracking'].get('desired_speed_mps', 0.)))
                tracking = dict(e['tracking'], desired_speed_mps=original)
                advised = observed_approach_profile(state, current['path_world'], goal, tracking, profile)
                counts['observed_target_route_ticks'] += 1
                reduced = advised['desired_speed_mps'] < original-1.e-8
                counts['lower_speed_advice_ticks'] += reduced
                flags = advised['stock_stopping_profile']
                counts['measured_speed_above_nominal_envelope_ticks'] += flags['current_speed_exceeds_nominal_envelope']
                r = dict(job=number, tick=e['tick'], measured_speed_mps=state.speed_mps,
                    original_effective_desired_mps=original, replay_advised_desired_mps=advised['desired_speed_mps'],
                    observed_distance_to_goal_m=flags['remaining_observed_distance_m'],
                    nominal_stopping_distance_m=flags['nominal_stopping_distance_m'],
                    already_motion_guard_stopped=bool(e['tracking'].get('current_motion_guard_stop')),
                    actual_throttle=e['applied_control']['throttle'], actual_brake=e['applied_control']['brake'])
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(r))
                    writer.writeheader()
                writer.writerow(r)
            navigation.append(dict(job=number, episode=d.parts[-4], counts=dict(counts),
                bicycle_yaw_residual_mean_abs_rad=float(np.mean(yaw_errors)),
                bicycle_yaw_residual_p95_abs_rad=float(np.percentile(yaw_errors, 95)),
                measured_vs_normalized_command_steering_p95_abs_rad=float(np.percentile(steering_errors, 95)),
                interpretation='一步模型残差与轮角响应差异，仅描述真实轨迹；不证明是失败的唯一原因'))
    with (out / 'calibration.csv').open('x', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(calibration[0]))
        w.writeheader()
        w.writerows(calibration)
    report = dict(calibration=calibration, navigation=navigation,
        actual_replayed_controller_command_pairs=sum(r['replayed_next_command_pairs'] for r in calibration),
        controller_replay_matched_recordings=all(r['maximum_controller_replay_error'] < 1.e-7 for r in calibration),
        local_nominal_stop_check_passed=all(r['nominal_profile_covers_this_measured_stop'] for r in calibration),
        actual_new_CARLA_calls=0, actual_new_GPU_calls=0, actual_arrival_credit=0,
        actual_new_closed_loop_ablation_credit=0, frozen_profile_deployed=False,
        limitations=['2条同一开发配置标定不足以构成全地图停车保证',
            '原安全保护已有.03滑行检查；差异集中在目标接近速度/规划模型与执行器的衔接，不能说全部安全检查都按2m/s2停车',
            '新速度建议只在旧轨迹上计算；建议会改变新轨迹，不能据旧轨迹计算干预后的到达率',
            '观测轮角与标准单轨模型残差不能直接解释为某一部件故障'],
        protocol_sha256=sha(out / 'protocol.json'), inputs_sha256=inputs,
        source_sha256=sha(__file__), goal_complete=False)
    with (out / 'report.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    assert sum(p.stat().st_size for p in out.iterdir()) < protocol['output_budget_bytes']
    print(json.dumps(dict(calibration=calibration, navigation=navigation, goal_complete=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
