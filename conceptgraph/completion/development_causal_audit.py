"""独立重算开发集过去LiDAR选择、坐标变换、z-buffer与前端复用证据。"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import gzip
import json
from pathlib import Path
import pickle
import shutil
import time

import imageio.v2 as imageio
import numpy as np

from conceptgraph.completion.causal_input_audit import independent_transform
from conceptgraph.completion.stage9 import OUTPUT, digest, dump
from conceptgraph.nuscenes.expansion_benchmark import iter_records


def independent_depth(points, transform, intrinsic, height, width):
    """按像素、距离排序取首项，与生产代码的minimum.at实现独立。"""
    homogeneous = np.column_stack((np.asarray(points, dtype=np.float64), np.ones(len(points))))
    camera = (np.asarray(transform)@homogeneous.T).T[:, :3]
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1) & (camera[:, 2] < 65)
    camera = camera[valid]
    result = np.zeros(height*width, dtype=np.uint16)
    if not len(camera):
        return result.reshape(height, width)
    projected = (np.asarray(intrinsic)@camera.T).T
    columns, rows = np.rint(projected[:, :2]/projected[:, 2:3]).astype(np.int64).T
    inside = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    keys = rows[inside]*width+columns[inside]
    distances = camera[inside, 2]
    order = np.lexsort((distances, keys))
    ordered_keys = keys[order]
    first = np.r_[True, ordered_keys[1:] != ordered_keys[:-1]] if len(order) else np.array([], bool)
    result[ordered_keys[first]] = np.rint(distances[order[first]]*1000).astype(np.uint16)
    return result.reshape(height, width)


def past_selection(candidates, timestamp):
    """独立的排序/二分选择，拒绝超时测量。"""
    candidates = sorted(candidates, key=lambda row: (row['timestamp'], row['token']))
    times = [row['timestamp'] for row in candidates]
    index = bisect_right(times, timestamp)-1
    if index < 0 or timestamp-times[index] > 100000:
        return None
    return candidates[index]


def audit(source, destination):
    started = time.monotonic()
    complete = json.loads((source/'complete.json').read_text())
    release = json.loads((source/'release.json').read_text())
    destination.mkdir(exist_ok=False)
    shutil.copy2(__file__, destination/'auditor_source.py')
    for group in ['input_sha256', 'source_sha256']:
        for filename, sha in release[group].items():
            if digest(filename) != sha:
                raise ValueError('转换输入或源码指纹变化')
    for filename, sha in complete['files_sha256'].items():
        if digest(source/filename) != sha:
            raise ValueError('转换产物指纹变化')
    plans_list = json.loads((source/'alignment_manifest.json').read_text())['plans']
    plans = {p['camera']['token']: p for p in plans_list}
    provenance_rows = list(map(json.loads, (source/'sensor_provenance.jsonl').open()))
    provenance = {r['camera_token']: r for r in provenance_rows}
    binding_rows = json.loads((source/'detection_reuse.json').read_text())['frames']
    bindings = {r['camera_token']: r for r in binding_rows}
    if (len(plans) != len(plans_list) or len(provenance) != len(provenance_rows)
            or len(bindings) != len(binding_rows) or set(plans) != set(provenance)
            or set(bindings) != set(plans) or len(plans) != complete['cameras']
            or set(plans) != set(release['camera_tokens'])):
        raise ValueError('相机、点云、前端证据不完整或重复')
    # 从原始表另选过去测量；不信任生成器的alignment_manifest选择。
    def metadata_path(name):
        return Path(next(p for p in release['input_sha256'] if p.endswith('/'+name)))
    scene_names = {p['camera']['scene_token']: p['camera']['scene'] for p in plans_list}
    sample_scenes = {r['token']: scene_names[r['scene_token']]
                     for r in iter_records(metadata_path('sample.json')) if r['scene_token'] in scene_names}
    lidar_by_scene, original_cameras = defaultdict(list), {}
    for row in iter_records(metadata_path('sample_data.json')):
        if row['token'] in plans:
            original_cameras[row['token']] = row
        scene = sample_scenes.get(row['sample_token'])
        if scene and '/LIDAR_TOP/' in row['filename']:
            lidar_by_scene[scene].append(row)
    required_poses = {p['camera']['ego_pose_token'] for p in plans_list}
    required_poses |= {p['lidar']['ego_pose_token'] for p in plans_list if p['lidar']}
    raw_poses = {r['token']: r for r in iter_records(metadata_path('ego_pose.json')) if r['token'] in required_poses}
    raw_calib = {r['token']: r for r in iter_records(metadata_path('calibrated_sensor.json'))}
    splits = {str(Path(p).parent): json.loads(Path(p).read_text())['scene_to_split']
              for p in release['input_sha256'] if p.endswith('/split_manifest.json')}
    if set(original_cameras) != set(plans) or set(raw_poses) != required_poses:
        raise ValueError('缺少原始相机或位姿记录')
    old_poses = {}
    records, ages, offsets = [], [], []
    maximum_pose_error = 0.
    for token in release['camera_tokens']:
        plan, evidence, binding = plans[token], provenance[token], bindings[token]
        camera = plan['camera']
        scene, index = camera['scene'], camera['frame_idx']
        if splits[camera['original_root']].get(scene) != 'development':
            raise ValueError('审计发现非开发场景')
        original = original_cameras[token]
        for key in ['timestamp', 'filename', 'ego_pose_token', 'calibrated_sensor_token', 'sample_token']:
            if original[key] != camera[key]:
                raise ValueError('原始相机元数据变化')
        selected = past_selection(lidar_by_scene[scene], original['timestamp'])
        if (selected is None) != (plan['lidar'] is None) or (selected and selected != plan['lidar']):
            raise ValueError('未选择真实最近过去测量')
        folder = source/'converted'/scene
        agent = json.loads((folder/'inputs'/f'{index:06d}.json').read_text())
        camera_world = independent_transform(raw_poses[original['ego_pose_token']]) @ independent_transform(raw_calib[original['calibrated_sensor_token']])
        pose_error = float(np.max(np.abs(camera_world-np.asarray(agent['camera_pose']))))
        maximum_pose_error = max(pose_error, maximum_pose_error)
        key = camera['original_root'], scene
        if key not in old_poses:
            old_poses[key] = np.loadtxt(Path(camera['original_root'])/'converted'/scene/'poses.txt').reshape(-1, 4, 4)
        if pose_error > 1e-9 or not np.allclose(old_poses[key][index], camera_world, atol=1e-8, rtol=0):
            raise ValueError('独立原始标定变换或旧相机位姿不同')
        rgb_sha = digest(agent['rgb'])
        if rgb_sha != digest(camera['old_rgb']) or rgb_sha != binding['rgb_sha256']:
            raise ValueError('原RGB未逐字节保留')
        if (digest(agent['detections']) != binding['detection_sha256']
                or digest(binding['source_detection']) != binding['detection_sha256']):
            raise ValueError('复用真实检测文件不同')
        with gzip.open(agent['detections'], 'rb') as handle:
            saved = pickle.load(handle)
        original_complete = Path(binding['source_detection']).parents[2]/'frontend_complete.json'
        real_complete = json.loads(original_complete.read_text())
        if (digest(original_complete) != binding['original_complete_sha256']
                or real_complete['output_sha256'].get(binding['source_detection']) != binding['detection_sha256']
                or saved['frame_provenance']['rgb_sha256'] != rgb_sha
                or saved['frame_provenance']['image_size_wh'] != [camera['width'], camera['height']]
                or not saved['frame_provenance']['sam_prompt_tensor_isolated']):
            raise ValueError('原真实检测与同RGB证据不成立')
        depth = imageio.imread(agent['depth'])
        if depth.dtype != np.uint16 or depth.shape != (camera['height'], camera['width']):
            raise ValueError('深度尺寸或毫米编码错误')
        intrinsic = raw_calib[original['calibrated_sensor_token']]['camera_intrinsic']
        if not np.array_equal(intrinsic, agent['intrinsic']):
            raise ValueError('相机内参变化')
        age, raw_sha = None, None
        if selected:
            age = (original['timestamp']-selected['timestamp'])/1e6
            if not 0 <= age <= .1 or age != agent['lidar_age_s'] or not agent['depth_available']:
                raise ValueError('未来、超时LiDAR或可用性错误')
            raw_sha = digest(evidence['raw_lidar'])
            archive = Path(evidence['archive'])
            if (raw_sha != evidence['raw_sha256'] or evidence['member'] != selected['filename']
                    or archive.stat().st_size != evidence['archive_size']
                    or archive.stat().st_mtime_ns != evidence['archive_mtime_ns']
                    or evidence['status'] != 'real_past_measurement'):
                raise ValueError('过去LiDAR缺少原归档成员证据')
            points = np.fromfile(evidence['raw_lidar'], np.float32).reshape(-1, 5)[:, :3]
            lidar_world = independent_transform(raw_poses[selected['ego_pose_token']]) @ independent_transform(raw_calib[selected['calibrated_sensor_token']])
            recomputed = independent_depth(points, np.linalg.inv(camera_world)@lidar_world,
                intrinsic, camera['height'], camera['width'])
            ages.append(age)
        else:
            recomputed = np.zeros_like(depth)
            if (agent['depth_available'] or agent['lidar_age_s'] is not None
                    or agent['lidar_timestamp_s'] is not None or evidence['raw_lidar'] is not None
                    or evidence['member'] is not None or evidence['raw_sha256'] is not None
                    or evidence['status'] != 'MISSING_NO_PAST_MEASUREMENT'):
                raise ValueError('缺测帧取得伪造测量信用')
        mismatch = int(np.count_nonzero(recomputed != depth))
        pixels = int(np.count_nonzero(depth))
        if mismatch or pixels != evidence['valid_lidar_pixels'] or digest(agent['depth']) != evidence['depth_sha256']:
            raise ValueError(f'独立深度重投影不一致：{scene}/{index} {mismatch}像素')
        if (agent['gt_in_agent'] or agent['future_lidar_used']
                or agent['actual_new_frontend_calls'] != 0 or binding['actual_new_frontend_calls'] != 0):
            raise ValueError('GT、未来或新前端调用状态错误')
        sample_offset = (camera['official_sample_timestamp']-original['timestamp'])/1e6
        offsets.append(sample_offset)
        records.append(dict(scene=scene, frame_idx=index, camera_token=token,
            past_lidar_token=selected['token'] if selected else None, lidar_age_s=age,
            independent_depth_mismatched_pixels=mismatch, valid_depth_pixels=pixels,
            original_RGB_sha256=rgb_sha, raw_lidar_sha256=raw_sha,
            sample_GT_time_offset_s=sample_offset, missing_gets_measurement_credit=False))
        if len(records) % 100 == 0:
            print(json.dumps(dict(audited=len(records), requested=len(plans), elapsed_s=time.monotonic()-started)), flush=True)
    with (destination/'frame_checks.jsonl').open('x') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
    report = dict(passed=True, cameras=len(records), original_RGB_exact_checks=len(records),
        detection_reuse_same_RGB_checks=len(records), actual_new_frontend_calls=0,
        independent_projection_frames=len(records), independent_projection_mismatched_pixels=0,
        real_past_lidar_pairs=len(ages), explicit_missing_masks=len(records)-len(ages), future_lidar_pairs=0,
        maximum_past_lidar_age_s=max(ages) if ages else None,
        maximum_original_camera_pose_difference=maximum_pose_error,
        sample_GT_time_offset_s=dict(minimum=min(offsets), maximum=max(offsets)),
        evaluation_limit='GT仍为原官方sample时刻；移动对象时间近似，不能称完全同步GT',
        checked_output_files=len(complete['files_sha256']), all_fingerprints_passed=True,
        complete_sha256=digest(source/'complete.json'), auditor_sha256=digest(__file__),
        transform_helper_sha256=digest(Path(__file__).with_name('causal_input_audit.py')),
        checks_sha256=digest(destination/'frame_checks.jsonl'), elapsed_s=time.monotonic()-started,
        no_GT_used_in_conversion=True, no_new_model_calls=True, methods_evaluated=False,
        new_test_opened=False, goal_complete=False)
    dump(destination/'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run', required=True)
    parser.add_argument('--audit-id', default='audit_001')
    args = parser.parse_args()
    source = OUTPUT/'data_manifest'/args.input_run
    audit(source, source/args.audit_id)


if __name__ == '__main__':
    main()
