"""独立复算冻结故障事件与控制时钟；不调用生产exposure_metrics。"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OUTPUT = Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def audit(row):
    case = Path(row['raw_case'])
    if row['result'] is None:
        return dict(run_id=row['run_id'], result_missing=True, actual_exposure=None)
    result = read(case / 'result.json')
    assert result == row['result'] and sha(case / 'result.json') == row['result_sha256']
    assert not result['gt_in_agent'] and result['real_clock_async']
    events = lines(case / 'fault_events.jsonl')
    controls = lines(case / 'execution.jsonl')
    clock = read(case / 'fault_driver_clock.json')
    exposure = row['fault_exposure']
    raw_exposure = read(case / 'fault_exposure.json')
    assert exposure['raw_event_sha256'] == sha(case / 'fault_events.jsonl')
    assert exposure['raw_execution_sha256'] == sha(case / 'execution.jsonl')
    assert all(exposure[k] == v for k, v in raw_exposure.items())
    sensors = [r for r in events if r['event'] == 'sensor']
    detectors = [r for r in events if r['event'] == 'detector']
    assert len({r['frame_idx'] for r in sensors}) == len(sensors)
    assert len({r['frame_idx'] for r in detectors}) == len(detectors)
    assert all(b['wall_s'] >= a['wall_s'] for a, b in zip(events, events[1:]))
    assert all(b['wall_elapsed_s'] >= a['wall_elapsed_s'] for a, b in zip(controls, controls[1:]))
    active = [r for r in sensors if r['active']]
    assert [r['frame_idx'] for r in active] == exposure['active_frames']
    assert bool(active) == exposure['actual_fault_exposed']
    if active:
        assert exposure['onset_distance_m'] >= .5 and exposure['onset_speed_mps'] >= .2
    removals = sum(r['input_detections'] - r['retained_detections'] for r in detectors)
    depth_removals = sum(max(0, r['input_depth_pixels'] - r['retained_depth_pixels']) for r in sensors)
    noisy_pose = sum(any(v != 0 for v in r['position_delta_m']) for r in sensors)
    computed = dict(actual_sensor_frames=len(sensors), actual_detector_calls=len(detectors),
                    recorded_control_ticks=len(controls), actual_detection_removals=removals,
                    actual_depth_pixels_removed=depth_removals, actual_noisy_pose_frames=noisy_pose)
    assert all(exposure[k] == v for k, v in computed.items())
    for event in detectors:
        kept = event['retained_indices']
        assert len(kept) == event['retained_detections'] == len(set(kept))
        assert all(0 <= i < event['input_detections'] for i in kept)
        if not event['active']:
            assert kept == list(range(event['input_detections']))
    begins = [r for r in events if r['event'] == 'blackout_begin']
    ends = [r for r in events if r['event'] == 'blackout_end']
    blackout = None
    if begins:
        assert row['fault'] == 'sensor_blackout_4s' and len(begins) == len(ends) == 1
        start, end = begins[0]['wall_s'], ends[0]['wall_s']
        assert end - start >= 4 and begins[0]['physical_speed_mps'] >= .2
        ticks = [r for r in controls if start <= clock['started_monotonic_s'] + r['wall_elapsed_s'] <= end]
        after = [r for r in controls if clock['started_monotonic_s'] + r['wall_elapsed_s'] > end]
        blackout = dict(duration_s=end-start, control_ticks=len(ticks),
                        stale_braking_ticks=sum(r['stale_sensor_braking'] for r in ticks),
                        stopped=any(r['stale_sensor_braking'] and r['state']['speed_mps'] == 0 for r in ticks),
                        fresh_motion_after=any(not r['stale_sensor_braking'] and r['state']['speed_mps'] > .2 for r in after))
        for ours, original in [('control_ticks', 'blackout_control_ticks'), ('stale_braking_ticks', 'blackout_stale_braking_ticks'),
                               ('stopped', 'blackout_stopped_during_gap'), ('fresh_motion_after', 'after_gap_fresh_motion_observed')]:
            assert blackout[ours] == exposure[original]
        assert abs(blackout['duration_s'] - exposure['blackout_duration_s']) < 1e-9
    else:
        assert not ends and exposure['blackout_control_ticks'] is None
    effective = {'baseline': None, 'detection_dropout_30pct': removals > 0,
                 'lidar_dropout_80pct': depth_removals > 0, 'pose_noise_025m_2deg': noisy_pose > 0,
                 'sensor_blackout_4s': bool(begins), 'depth_noise_02m': None}[row['fault']]
    return dict(run_id=row['run_id'], fault=row['fault'], result_missing=False,
                protocol_triggered=bool(active), measurable_effect_from_event_ledger=effective,
                depth_noise_amplitude_not_verified_here=True, blackout=blackout, counts=computed,
                reported_arrival=bool(result['actual_full_perception_arrival']),
                original_geometry_audit_passed=result['independent_execution_audit']['passed'],
                geometry_not_recomputed_here=True, new_arrival_credit=False,
                inputs={str(case / n): sha(case / n) for n in ['result.json', 'fault_events.jsonl',
                        'execution.jsonl', 'fault_driver_clock.json', 'fault_exposure.json']})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-run', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    source = OUTPUT / 'interactive_simulation' / args.input_run
    raw = (source / 'trial_records.jsonl').read_bytes()
    if raw and not raw.endswith(b'\n'):
        raw = raw[:raw.rfind(b'\n')+1]
    rows = [json.loads(s) for s in raw.splitlines()]
    release = read(source / 'release.json')
    assert rows and len(rows) <= len(release['trials'])
    for row, planned in zip(rows, release['trials']):
        assert all(row[k] == v for k, v in planned.items())
    dest = OUTPUT / 'faults' / args.run_id
    dest.mkdir(exist_ok=False)
    (dest / 'input_prefix.jsonl').write_bytes(raw)
    outputs = [audit(row) for row in rows]
    with (dest / 'cases.jsonl').open('x') as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    complete = len(rows) == len(release['trials']) and (source / 'summary.json').exists()
    if complete:
        assert read(source / 'summary.json')['trial_records_sha256'] == hashlib.sha256(raw).hexdigest()
    report = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
                  source_run=args.input_run, audited_cases=len(rows), expected=len(release['trials']),
                  complete_cohort=complete, conditions=dict(Counter(r['fault'] for r in rows)),
                  missing_results=sum(r['result_missing'] for r in outputs),
                  blackouts_observed=sum(r.get('blackout') is not None for r in outputs),
                  blackouts_stopped=sum(bool(r.get('blackout') and r['blackout']['stopped']) for r in outputs),
                  blackouts_fresh_motion_after=sum(bool(r.get('blackout') and r['blackout']['fresh_motion_after']) for r in outputs),
                  depth_noise_numeric_amplitude_verified=False, protocol_trigger_is_not_effect=True,
                  continuous_body_geometry_not_recomputed=True, actual_model_calls=0, new_arrival_credit=False,
                  input_prefix_sha256=sha(dest / 'input_prefix.jsonl'), release_sha256=sha(source / 'release.json'),
                  cases_sha256=sha(dest / 'cases.jsonl'), source_sha256=sha(Path(__file__)), goal_complete=False)
    (dest / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
