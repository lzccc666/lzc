"""从只读归档准备自然指令原相机帧与不晚于相机的LiDAR，不执行封存方法。"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time

import imageio.v2 as imageio
import numpy as np

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump
from conceptgraph.nuscenes.expansion_benchmark import iter_records
from conceptgraph.scripts.convert_lane_expansion import nearest_depth_image
from conceptgraph.scripts.preprocess_nuscenes_to_minimal import build_transform, inverse_transform


def latest_past_lidar(candidates, camera_timestamp, maximum_age_us=100000):
    """只返回过去或同刻测量；无合格测量时保留缺失，不借用未来帧。"""
    candidates = sorted(candidates, key=lambda r: r['timestamp'])
    i = bisect.bisect_right([r['timestamp'] for r in candidates], camera_timestamp)-1
    if i < 0 or camera_timestamp-candidates[i]['timestamp'] > maximum_age_us:
        return None
    return candidates[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--max-cameras', type=int)
    parser.add_argument('--wall-budget-s', type=float, default=1800)
    args = parser.parse_args()
    started = time.monotonic()
    dest = OUTPUT/'data_manifest'/args.run_id
    dest.mkdir(exist_ok=False)
    storage_budget = 700000000 if args.max_cameras is None else 25000000
    free = shutil.disk_usage(dest).free
    if free < storage_budget+1000000000:
        raise RuntimeError('原项目卷空间不足，保留现有文件')
    audit = OUTPUT/'data_manifest/natural_distribution_audit_001'
    report = json.loads((audit/'report.json').read_text())
    records = audit/'evaluation_only_natural_commands.jsonl'
    assert digest(records) == report['records_sha256']
    rows = list(map(json.loads, records.open()))
    cameras = {r['camera_sample_data']['token']: dict(r['camera_sample_data'], scene=r['scene'])
               for r in rows if r['camera_sample_data'] is not None}
    ordered = sorted(cameras.values(), key=lambda r: (r['scene'], r['timestamp'], r['token']))
    if args.max_cameras is not None:
        if args.max_cameras <= 0:
            raise ValueError('max-cameras must be positive')
        ordered = ordered[:args.max_cameras]
    dump(dest/'release.json', dict(source_sha256=digest(Path(__file__)),
        source_audit_sha256=digest(audit/'report.json'), raw_records_sha256=digest(records),
        selected_camera_tokens=[r['token'] for r in ordered], total_raw_commands=len(rows),
        unique_cameras=len(cameras), camera_selection='按场景、原始时间戳排序；不看目标/方法结果',
        wall_budget_s=args.wall_budget_s, storage_budget_bytes=storage_budget, free_before_bytes=free,
        alignment='latest past LIDAR_TOP within 100ms, no future returns',
        dataset_license='CC BY-NC-SA 4.0; attribution/license in talk2car_source_001',
        inference_executed=False, new_test_opened=False, goal_complete=False))
    shutil.copy2(__file__, dest/'generator_source.py')
    meta = PROJECT/'dataset/nuscenes_trainval_raw/v1.0-trainval'
    scene_by_token = {s['scene_token']: s['scene'] for s in
        json.loads((OUTPUT/'splits/reserved_test_scenes_v2.json').read_text())['scenes']}
    samples = {r['token']: r for r in iter_records(meta/'sample.json')
               if r['scene_token'] in scene_by_token}
    lidar_by_scene = {}
    wanted_scenes = {r['scene'] for r in ordered}
    for r in iter_records(meta/'sample_data.json'):
        sample = samples.get(r['sample_token'])
        if sample is not None and '/LIDAR_TOP/' in r['filename']:
            scene = scene_by_token[sample['scene_token']]
            if scene in wanted_scenes:
                lidar_by_scene.setdefault(scene, []).append(r)
    plans, missing = [], []
    for camera in ordered:
        lidar = latest_past_lidar(lidar_by_scene.get(camera['scene'], []), camera['timestamp'])
        if lidar is None:
            missing.append(dict(camera_token=camera['token'], reason='no past LiDAR within 100ms'))
        else:
            plans.append(dict(camera=camera, lidar=lidar))
    needed_poses = {r[k]['ego_pose_token'] for r in plans for k in ['camera', 'lidar']}
    poses = {r['token']: r for r in iter_records(meta/'ego_pose.json') if r['token'] in needed_poses}
    calibration = {r['token']: r for r in iter_records(meta/'calibrated_sensor.json')}
    members = {}
    for plan in plans:
        camera, lidar = plan['camera'], plan['lidar']
        cp, lp = poses[camera['ego_pose_token']], poses[lidar['ego_pose_token']]
        cc, lc = calibration[camera['calibrated_sensor_token']], calibration[lidar['calibrated_sensor_token']]
        cw = build_transform(cp['translation'], cp['rotation']) @ build_transform(cc['translation'], cc['rotation'])
        lw = build_transform(lp['translation'], lp['rotation']) @ build_transform(lc['translation'], lc['rotation'])
        folder = dest/'converted'/camera['scene']/camera['token']
        folder.mkdir(parents=True)
        agent = dict(scene=camera['scene'], camera_token=camera['token'],
            camera_timestamp_s=camera['timestamp']/1e6, lidar_timestamp_s=lidar['timestamp']/1e6,
            lidar_age_s=(camera['timestamp']-lidar['timestamp'])/1e6,
            rgb=str(folder/'rgb.jpg'), depth=str(folder/'depth.png'), depth_scale=1000,
            camera_pose=cw.tolist(), intrinsic=cc['camera_intrinsic'],
            width=camera['width'], height=camera['height'],
            depth_source='原始过去LiDAR，位姿补偿后稀疏投影，最近深度zbuffer',
            limitations='仅补偿传感器运动，不用GT补偿动态物体；未填充无返回像素',
            gt_in_agent=False, inference_executed=False)
        dump(folder/'input.json', agent)
        members.setdefault(camera['filename'], []).append(dict(channel='CAM_FRONT', output=folder/'rgb.jpg',
            camera_token=camera['token']))
        members.setdefault(lidar['filename'], []).append(dict(channel='LIDAR_TOP', output=folder/'depth.png',
            camera_token=camera['token'], transform=inverse_transform(cw)@lw,
            intrinsic=cc['camera_intrinsic'], height=camera['height'], width=camera['width']))
    dump(dest/'alignment_manifest.json', dict(plans=[dict(camera=p['camera'], lidar=p['lidar']) for p in plans],
        missing=missing, gt_in_inference=False, new_test_opened=False))
    complete = set()
    written = 0
    with (dest/'sensor_provenance.jsonl').open('x') as provenance:
        for part in ['03', '04']:
            archive = Path('/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval')/f'v1.0-trainval{part}_blobs.tgz'
            archive_started = time.monotonic()
            with tarfile.open(archive, 'r|gz') as tar:
                for member in tar:
                    if time.monotonic()-started > args.wall_budget_s:
                        raise TimeoutError('自然指令传感器准备达到预算；已写证据保留')
                    name = member.name.removeprefix('./')
                    if name not in members or name in complete:
                        continue
                    if not member.isfile() or member.size > 20000000:
                        raise ValueError('unexpected sensor archive member')
                    if shutil.disk_usage(dest).free < 1000000000:
                        raise RuntimeError('保留1GB空间；不删除任何产物')
                    data = tar.extractfile(member).read()
                    source_sha = hashlib.sha256(data).hexdigest()
                    for row in members[name]:
                        output = row['output']
                        if output.exists():
                            raise FileExistsError(output)
                        if row['channel'] == 'CAM_FRONT':
                            with output.open('xb') as f:
                                f.write(data)
                        else:
                            raw = np.frombuffer(data, dtype=np.float32)
                            if len(raw) % 5:
                                raise ValueError('invalid nuScenes LiDAR columns')
                            xyz = raw.reshape(-1,5)[:,:3]
                            t = row['transform']
                            xyz = xyz@t[:3,:3].T+t[:3,3]
                            depth = nearest_depth_image(xyz,row['intrinsic'],row['height'],row['width'])
                            imageio.imwrite(output,np.rint(depth*1000).astype(np.uint16))
                        written += output.stat().st_size
                        if written > storage_budget:
                            raise RuntimeError('new data storage budget exceeded')
                        provenance.write(json.dumps(dict(camera_token=row['camera_token'],channel=row['channel'],
                            archive=str(archive), archive_size=archive.stat().st_size,
                            archive_mtime_ns=archive.stat().st_mtime_ns, member=name, member_sha256=source_sha,
                            output=str(output),output_sha256=digest(output)),ensure_ascii=False)+'\n')
                        provenance.flush()
                    complete.add(name)
                    if len(complete)%50 == 0:
                        print(json.dumps(dict(complete_members=len(complete),expected_members=len(members),
                            elapsed_s=time.monotonic()-started)),flush=True)
                    if len(complete)==len(members):
                        break
            dump(dest/f'archive{part}_read.json',dict(elapsed_s=time.monotonic()-archive_started,
                complete_members=len(complete),expected_members=len(members)))
            if len(complete)==len(members):
                break
    if set(members)-complete or missing:
        dump(dest/'missing.json',dict(members=sorted(set(members)-complete),alignments=missing,goal_complete=False))
        raise RuntimeError('原始自然指令传感器仍有缺失，不替代或伪造')
    dump(dest/'complete.json',dict(cameras=len(plans),unique_members=len(complete),written_bytes=written,
        provenance_sha256=digest(dest/'sensor_provenance.jsonl'),elapsed_s=time.monotonic()-started,
        raw_lidar_retained=False,inference_executed=False,new_test_opened=False,goal_complete=False))


if __name__ == '__main__':
    main()
