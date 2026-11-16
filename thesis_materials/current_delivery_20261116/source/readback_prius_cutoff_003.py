"""独立重算1020条官方标定记录、实际控制命令及名义滑行距离的适用边界。"""
import datetime
import gzip
import json
from pathlib import Path

import numpy as np
from shapely.geometry import box
from conceptgraph.completion.stage9 import digest, dump
from conceptgraph.navigation.vehicle_geometry import VehicleSpec, VehicleState
from carla_motion_guard_001 import nominal_coasting_check
from carla_prius_cutoff_pilot_008 import AccelerationFeedback

R = Path(__file__).resolve().parents[1]


def main():
    source = R/'carla_official/prius_cutoff_pilot_008'
    dest = R/'carla_official/prius_cutoff_readback_003'
    dest.mkdir(exist_ok=False)
    registration = json.loads((source/'registration.json').read_text())
    assert all(digest(p) == h for p, h in registration['source_sha256'].items())
    report = json.loads((source/'report.json').read_text())
    results = []
    for item in report['results']:
        stem = f"{item['case']}_{item['policy']}"
        raw = source/(stem+'.jsonl.gz')
        x = np.asarray(list(map(json.loads, gzip.open(raw, 'rt'))))
        assert len(x) == 510 and np.isfinite(x).all()
        assert np.array_equal(x[:, 0], np.arange(510))
        assert np.array_equal(x[:, 1], x[:, 2])
        assert np.allclose(np.diff(x[:, 3]), .1, atol=1e-5)
        speeds = np.linalg.norm(x[:, 4:6], axis=1)
        assert np.allclose(speeds, x[:, 17], atol=1e-9)
        acceleration = np.diff(speeds)/np.diff(x[:, 3])
        assert np.allclose(acceleration, x[1:, 18], atol=1e-9)
        assert np.allclose(x[:, 13:15], x[:, 19:21], atol=1e-7)
        controller = AccelerationFeedback()
        command = (0., 1.)
        for i, row in enumerate(x):
            assert np.allclose(command, row[13:15], atol=1e-7)
            drive = 10 <= i < 410
            desired = .55 if drive else 0.
            request = float(np.clip(.4*(desired-speeds[i]), -.3, .15))
            assert abs(request-row[16]) < 1e-9
            command = controller.step(request, row[18], speeds[i], desired, not drive)
            if i < 9:
                command = (0., 1.)
        physics = json.loads((source/(stem+'_release.json')).read_text())
        assert physics['actual_torque_curve'] == physics['original_torque_curve']
        assert digest(raw) == item['raw_sha256']
        assert abs(acceleration.max()-item['maximum_acceleration_mps2']) < 1e-9
        assert abs(acceleration.min()-item['minimum_acceleration_mps2']) < 1e-9
        assert item['combined_calibration_pass'] and not item['collisions']
        final_stop = next(i for i in range(411, 510) if speeds[i] <= .08)
        stop_distances = []
        for i in range(410, final_stop+1):
            measured_remaining = float(np.linalg.norm(np.diff(x[i:final_stop+1, 7:9], axis=0), axis=1).sum())
            nominal = nominal_coasting_check(VehicleState(0., 0., 0., speed_mps=float(speeds[i])),
                box(-100., -100., 100., 100.), [], VehicleSpec(), observation={'valid': True})
            predicted = nominal['predicted_travel_m']
            stop_distances.append(dict(tick=i, actual_speed_mps=float(speeds[i]),
                actual_remaining_travel_until_point08_m=measured_remaining,
                nominal_stopping_travel_m=predicted,
                nominal_minus_actual_margin_m=predicted-measured_remaining))
        with (dest/(stem+'_stopping_distances.jsonl')).open('x') as stream:
            for row in stop_distances:
                stream.write(json.dumps(row)+'\n')
        minimum = min(r['nominal_minus_actual_margin_m'] for r in stop_distances)
        results.append(dict(case=item['case'], physical_ticks=510,
            reconstructed_control_commands=510, stock_torque_curve_unchanged=True,
            actual_coasting_stop_distance_m=stop_distances[0]['actual_remaining_travel_until_point08_m'],
            initial_cutoff_speed_mps=float(speeds[410]),
            cutoff_transient_peak_speed_mps=float(speeds[410:414].max()),
            final_stop_tick=final_stop, full_stop_speed_limit_mps=.08,
            maximum_actual_acceleration_mps2=float(acceleration.max()),
            minimum_actual_acceleration_mps2=float(acceleration.min()),
            nominal_stopping_distance_compared_states=len(stop_distances),
            minimum_nominal_minus_actual_stopping_distance_m=minimum,
            nominal_bounds_observed_travel_with_2cm_tolerance=minimum >= -.02,
            calibration_constraints_pass=True, actual_navigation_or_arrival_credit=False,
            source_sha256={str(p): digest(p) for p in [raw, source/(stem+'_release.json'),
                source/(stem+'_report.json')]}))
    value = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_new_physical_calibrations=0, existing_actual_calibrations_read_back=2,
        official_physics_records_checked=1020, source=str(source), results=results,
        reconstructed_commands_match=True, actual_cutoff_calibration_passed=all(
            r['calibration_constraints_pass'] for r in results),
        nominal_stopping_distance_bound_covers_observed_calibration=all(
            r['nominal_bounds_observed_travel_with_2cm_tolerance'] for r in results),
        nominal_parameters=dict(passive_deceleration_mps2=.03, reaction_s=.3,
            low_speed_brake_gate_mps=.15, tolerance_m=.02),
        physical_safety_generalization_proven=False,
        limitation='仅两条外生转向下的停车标定；名义模型即使覆盖这些轨迹，也不保证其他坡度、速度、路面或闭环安全。完整导航必须重新实测。',
        source_sha256={str(p): digest(p) for p in [Path(__file__), source/'registration.json',
            source/'report.json', R/'source/carla_motion_guard_001.py',
            R/'source/carla_prius_cutoff_pilot_008.py']}, goal_complete=False)
    dump(dest/'report.json', value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
