"""复核新关键帧的真实过去LiDAR、缺失掩码、原始RGB和后轴/相机坐标。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import imageio.v2 as imageio
import numpy as np

from conceptgraph.completion.stage9 import OUTPUT, digest, dump
from conceptgraph.nuscenes.expansion_benchmark import iter_records


def independent_transform(record):
    w, x, y, z = np.asarray(record['rotation'])/np.linalg.norm(record['rotation'])
    xyz = np.array([x, y, z])
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    rotation = 2*np.outer(xyz, xyz)+(w*w-xyz@xyz)*np.eye(3)+2*w*skew
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, record['translation']
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--audit-id', default='audit_001')
    args = parser.parse_args()
    started = time.monotonic()
    source = OUTPUT/'data_manifest'/args.run_id
    complete = json.loads((source/'complete.json').read_text())
    release = json.loads((source/'release.json').read_text())
    dest = source/args.audit_id
    dest.mkdir(exist_ok=False)
    shutil.copy2(__file__, dest/'auditor_source.py')
    for group in ['input_sha256', 'source_sha256']:
        for filename, sha in release[group].items():
            if digest(filename) != sha:
                raise ValueError('conversion input/source fingerprint changed')
    for filename, sha in complete['files_sha256'].items():
        if digest(source/filename) != sha:
            raise ValueError('converted output fingerprint changed')
    rows = [json.loads(line) for line in (source/'sensor_provenance.jsonl').open()]
    provenance = {(r['camera_token'], r['channel']): r for r in rows}
    if len(provenance) != len(rows) or len(rows) != 2*complete['cameras']:
        raise ValueError('missing or duplicated sensor evidence')
    old_rgb = {r['member']: r for r in map(json.loads,
        (OUTPUT/'data_manifest'/release['old_input_run']/'sensor_provenance.jsonl').open()) if r['channel'] == 'CAM_FRONT'}
    plans = {p['camera_token']: p for p in json.loads((source/'alignment_manifest.json').read_text())['plans']}
    inputs = [json.loads(p.read_text()) for p in sorted(source.glob('converted/scene-*/inputs/*.json'))]
    if len(inputs) != complete['cameras'] or {r['camera_token'] for r in inputs} != set(release['camera_tokens']):
        raise ValueError('camera population changed')
    source_index = Path(next(p for p in release['input_sha256'] if p.endswith('/new_scene_sources_03_04.json')))
    cameras = {f['sensors']['CAM_FRONT']['token']: f['sensors']['CAM_FRONT']
               for scene in json.loads(source_index.read_text()).values() for f in scene['sequence']
               if f['sensors']['CAM_FRONT']['token'] in set(release['camera_tokens'])}
    pose_path = Path(next(p for p in release['input_sha256'] if p.endswith('/ego_pose.json')))
    calib_path = Path(next(p for p in release['input_sha256'] if p.endswith('/calibrated_sensor.json')))
    wanted_poses = {c['ego_pose_token'] for c in cameras.values()}
    raw_poses = {p['token']: p for p in iter_records(pose_path) if p['token'] in wanted_poses}
    raw_calib = {p['token']: p for p in iter_records(calib_path)}
    if len(cameras) != len(inputs) or set(raw_poses) != wanted_poses:
        raise ValueError('missing independent raw rear axle pose/calibration')
    checks, ages = [], []
    maximum_old_pose_error = 0.
    for row in inputs:
        plan = plans[row['camera_token']]
        rgb, depth = [provenance[(row['camera_token'], c)] for c in ['CAM_FRONT', 'LIDAR_TOP']]
        if rgb['output_sha256'] != old_rgb[rgb['member']]['member_sha256']:
            raise ValueError('RGB differs from original archive bytes')
        image = imageio.imread(row['depth'])
        if image.shape != (row['height'], row['width']) or image.dtype != np.uint16:
            raise ValueError('depth shape/unit encoding differs')
        valid_pixels = int(np.count_nonzero(image))
        if row['depth_available']:
            if plan['lidar'] is None or depth['status'] != 'real_past_measurement' or not depth['member_sha256']:
                raise ValueError('available depth lacks real source evidence')
            age = (plan['camera_timestamp']-plan['lidar']['timestamp'])/1e6
            if not 0 <= age <= .1 or row['lidar_timestamp_s'] > row['camera_timestamp_s']:
                raise ValueError('future or expired LiDAR entered causal inputs')
            if abs(age-row['lidar_age_s']) > 1e-12 or valid_pixels != depth['valid_lidar_pixels']:
                raise ValueError('LiDAR age/pixel evidence differs')
            ages.append(age)
        elif (plan['lidar'] is not None or row['lidar_timestamp_s'] is not None
              or row['lidar_age_s'] is not None or depth['member'] is not None
              or depth['member_sha256'] is not None or valid_pixels != 0
              or depth['status'] != 'MISSING_NO_PAST_MEASUREMENT'):
            raise ValueError('missing LiDAR gained invented measurement credit')
        old_poses = np.loadtxt(OUTPUT/'data_manifest'/release['old_input_run']/
                              'converted'/row['scene']/'poses.txt').reshape(-1, 4, 4)
        error = float(np.max(np.abs(old_poses[row['frame_idx']]-row['camera_pose'])))
        maximum_old_pose_error = max(error, maximum_old_pose_error)
        if error > 1e-8 or not np.allclose(np.asarray(row['rear_axle_reference_world_pose'])[3], [0, 0, 0, 1]):
            raise ValueError('camera/rear-axle pose record invalid')
        camera = cameras[row['camera_token']]
        expected_ego = independent_transform(raw_poses[camera['ego_pose_token']])
        expected_camera = expected_ego@independent_transform(raw_calib[camera['calibrated_sensor_token']])
        if (not np.allclose(row['rear_axle_reference_world_pose'], expected_ego, rtol=0, atol=1e-9)
                or not np.allclose(row['camera_pose'], expected_camera, rtol=0, atol=1e-9)):
            raise ValueError('rear axle or camera differs from original calibrated raw pose')
        if row['gt_in_agent'] or row['future_lidar_used'] or row['inference_executed']:
            raise ValueError('incorrect conversion/inference status')
        checks.append(dict(scene=row['scene'], frame_idx=row['frame_idx'], camera_token=row['camera_token'],
            real_depth_available=row['depth_available'], valid_lidar_pixels=valid_pixels,
            lidar_age_s=row['lidar_age_s'], camera_pose_maximum_difference=error,
            missing_depth_gets_measurement_credit=False))
    with (dest/'frame_checks.jsonl').open('x') as handle:
        for check in checks:
            handle.write(json.dumps(check, ensure_ascii=False, allow_nan=False)+'\n')
    report = dict(passed=True, cameras=len(inputs), original_RGB_exact_checks=len(inputs),
        real_past_lidar_pairs=len(ages), explicit_missing_masks=len(inputs)-len(ages),
        future_lidar_pairs=0, maximum_past_lidar_age_s=max(ages) if ages else None,
        maximum_original_camera_pose_difference=maximum_old_pose_error,
        checked_output_files=len(complete['files_sha256']), all_fingerprints_passed=True,
        rear_axle_raw_pose_and_calibration_verified=True,
        complete_sha256=digest(source/'complete.json'), auditor_sha256=digest(__file__),
        checks_sha256=digest(dest/'frame_checks.jsonl'), elapsed_s=time.monotonic()-started,
        missing_measurements_preserved=True, new_test_opened=False, inference_executed=False, goal_complete=False)
    dump(dest/'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
