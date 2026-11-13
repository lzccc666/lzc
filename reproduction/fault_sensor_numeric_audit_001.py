"""逐帧比较未扰动/有效RGB、深度和位姿；不调用故障注入实现。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

OUTPUT = Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-run', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    source = OUTPUT / 'interactive_simulation' / args.input_run
    raw = (source / 'trial_records.jsonl').read_bytes()
    assert raw.endswith(b'\n')
    rows = [json.loads(s) for s in raw.splitlines()]
    release = read(source / 'release.json')
    summary = read(source / 'summary.json')
    assert len(rows) == len(release['trials']) == summary['attempted'] == summary['expected']
    assert hashlib.sha256(raw).hexdigest() == summary['trial_records_sha256']
    dest = OUTPUT / 'faults' / args.run_id
    dest.mkdir(exist_ok=False)
    outputs = []
    with (dest / 'frames.jsonl').open('x') as stream:
        for row, planned in zip(rows, release['trials']):
            assert all(row[k] == v for k, v in planned.items())
            case = Path(row['raw_case'])
            if row['result'] is None:
                outputs.append(dict(run_id=row['run_id'], missing=True))
                continue
            assert sha(case / 'result.json') == row['result_sha256']
            events = [json.loads(s) for s in (case / 'fault_events.jsonl').read_text().splitlines()]
            sensors = [e for e in events if e['event'] == 'sensor']
            n, changed, squared, total, removed, active_frames = 0, 0, 0., 0, 0, 0
            pose_translation_squared, yaw_squared = 0., 0.
            for event in sensors:
                i = event['frame_idx']
                path = case / 'sensor_evidence' / f'{i:06d}.json'
                effective = read(path)
                original_path = Path(effective['unmodified_sensor_evidence'])
                original = read(original_path)
                assert effective['fault_active'] == event['active']
                assert effective['fault_condition'] == row['fault']
                paths = [path, original_path]
                for evidence in [original, effective]:
                    for kind in ['rgb', 'depth']:
                        p = Path(evidence[kind])
                        assert sha(p) == evidence[kind + '_sha256']
                        paths.append(p)
                assert original['rgb_sha256'] == effective['rgb_sha256']
                with Image.open(original['depth']) as image:
                    before = np.asarray(image, dtype=np.int64)
                with Image.open(effective['depth']) as image:
                    after = np.asarray(image, dtype=np.int64)
                assert before.shape == after.shape and np.all(after[before == 0] == 0)
                assert np.count_nonzero(before) == event['input_depth_pixels']
                assert np.count_nonzero(after) == event['retained_depth_pixels']
                a, b = np.array(original['camera_pose_cv']), np.array(effective['camera_pose_cv'])
                delta = b[:3, 3] - a[:3, 3]
                assert np.allclose(delta, event['position_delta_m'], rtol=0, atol=1e-12)
                relative = b[:3, :3] @ a[:3, :3].T
                yaw = math.atan2(relative[1, 0], relative[0, 0])
                active = event['active']
                if not active or row['fault'] not in ['depth_noise_02m', 'lidar_dropout_80pct']:
                    assert np.array_equal(before, after)
                if not active or row['fault'] != 'pose_noise_025m_2deg':
                    assert np.array_equal(a, b)
                elif active:
                    assert abs(delta[2]) < 1e-12
                    c, s = math.cos(yaw), math.sin(yaw)
                    assert np.allclose(relative, [[c, -s, 0], [s, c, 0], [0, 0, 1]], atol=1e-12)
                valid = before > 0
                difference = (after[valid] - before[valid]) / 1000.
                frame_changed = int(np.count_nonzero(difference))
                frame_removed = int(np.count_nonzero(valid & (after == 0)))
                if active and row['fault'] == 'lidar_dropout_80pct':
                    assert np.all((after == 0) | (after == before))
                if active:
                    active_frames += 1
                    n += len(difference)
                    changed += frame_changed
                    squared += float(difference @ difference)
                    total += int(valid.sum())
                    removed += frame_removed
                    pose_translation_squared += float(delta[:2] @ delta[:2])
                    yaw_squared += math.degrees(yaw) ** 2
                record = dict(run_id=row['run_id'], frame=i, active=active,
                              valid_depth_pixels=int(valid.sum()), changed_depth_pixels=frame_changed,
                              removed_depth_pixels=frame_removed,
                              depth_RMSE_m=float(np.sqrt(np.mean(difference**2))) if len(difference) else None,
                              position_delta_m=delta.tolist(), yaw_delta_deg=math.degrees(yaw),
                              files_sha256={str(p): sha(p) for p in paths})
                stream.write(json.dumps(record) + '\n')
            outputs.append(dict(run_id=row['run_id'], fault=row['fault'], missing=False,
                                frames=len(sensors), active_frames=active_frames,
                                active_valid_depth_pixels=n, changed_depth_pixels=changed,
                                active_depth_RMSE_m=math.sqrt(squared/n) if n else None,
                                removed_depth_fraction=removed/total if total else None,
                                active_xy_translation_RMS_m=math.sqrt(pose_translation_squared/active_frames) if active_frames else None,
                                active_yaw_RMS_deg=math.sqrt(yaw_squared/active_frames) if active_frames else None,
                                numeric_exposure_verified=bool(active_frames and (
                                    changed if row['fault'] in ['depth_noise_02m', 'lidar_dropout_80pct']
                                    else pose_translation_squared if row['fault'] == 'pose_noise_025m_2deg' else False))))
    (dest / 'cases.json').write_text(json.dumps(outputs, ensure_ascii=False, indent=2))
    report = dict(passed=True, source_run=args.input_run, cases=len(rows), complete_cohort=True,
                  frame_pairs=sum(r.get('frames', 0) for r in outputs),
                  missing_results=sum(r['missing'] for r in outputs),
                  actual_model_calls=0, new_arrival_credit=False, distribution_normality_not_claimed=True,
                  empirical_RMS_not_configured_standard_deviation=True,
                  source_sha256=sha(Path(__file__)), input_records_sha256=hashlib.sha256(raw).hexdigest(),
                  frame_ledger_sha256=sha(dest / 'frames.jsonl'), cases_sha256=sha(dest / 'cases.json'),
                  goal_complete=False)
    (dest / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))
    for row in outputs:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
