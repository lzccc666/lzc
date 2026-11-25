"""开发关键帧严格过去LiDAR转换；保留原RGB并复用相同RGB的真实前端。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import tarfile
import time

import imageio.v2 as imageio
import numpy as np

from conceptgraph.completion.natural_inputs import latest_past_lidar
from conceptgraph.completion.reserved_causal_inputs import write_unchanged_or_new
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, available_development, digest, dump
from conceptgraph.nuscenes.expansion_benchmark import iter_records
from conceptgraph.scripts.convert_lane_expansion import nearest_depth_image
from conceptgraph.scripts.preprocess_nuscenes_to_minimal import build_transform, inverse_transform


def exact_link(path, source):
    source = Path(source).resolve()
    if path.exists() or path.is_symlink():
        if not path.is_symlink() or path.resolve() != source:
            raise ValueError(f'已有入口不同，拒绝覆盖：{path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(source, target_is_directory=source.is_dir())


def development_cameras(scenes, *, limit=None):
    """只选原开发身份，索引与原相机/位姿逐帧一致，不读取GT。"""
    by_root = {}
    cameras, input_files = [], []
    for scene in scenes:
        base = Path(scene['root'])
        if base not in by_root:
            index_path = base/'source_file_index.json'
            split_path = base/'split_manifest.json'
            selection_path = base/'scene_selection_manifest.json'
            by_root[base] = (json.loads(index_path.read_text()), json.loads(split_path.read_text())['scene_to_split'],
                            {r['scene']: r for r in json.loads(selection_path.read_text())['scenes']})
            input_files.extend([index_path, split_path, selection_path])
        indices, splits, selections = by_root[base]
        name = scene['scene']
        if splits.get(name) != 'development':
            raise ValueError('禁止访问非开发场景')
        old = base/'converted'/name
        alignment_path, poses_path = old/'frame_alignment.json', old/'poses.txt'
        alignment = json.loads(alignment_path.read_text())
        poses = np.loadtxt(poses_path).reshape(-1, 4, 4)
        sequence = indices[name]['sequence']
        if len(alignment) != len(sequence) or len(poses) != len(sequence):
            raise ValueError('原始转换帧数不一致')
        input_files.extend([alignment_path, poses_path])
        for i, frame in enumerate(sequence):
            camera = frame['sensors']['CAM_FRONT']
            if (not alignment[i]['camera_source'].endswith(camera['filename'])
                    or alignment[i]['sample_token'] != frame['sample_token']):
                raise ValueError('原相机或帧顺序不同')
            cameras.append(dict(camera, scene=name, log_token=scene['log_token'],
                scene_token=selections[name]['scene_token'], frame_idx=i, original_root=str(base),
                old_rgb=str(old/'color'/f'{i:06d}.jpg'),
                old_camera_world=poses[i].tolist(), official_sample_timestamp=frame['timestamp'],
                old_lidar_timestamp=frame['sensors']['LIDAR_TOP']['timestamp']))
    if limit is not None:
        if limit < 1:
            raise ValueError('limit必须为正')
        cameras = cameras[:limit]
    if len({r['token'] for r in cameras}) != len(cameras):
        raise ValueError('相机重复')
    return cameras, sorted(set(input_files))


def bind_detection(camera, frontend, destination):
    """真实旧前端调用只复用，不增加新调用信用；SHA和逐帧provenance均检查。"""
    folder = frontend/'inputs'/camera['scene']
    completed = json.loads((folder/'frontend_complete.json').read_text())
    detection = folder/'exp_streamlined_yolo_outdoor_objects_v1/detections'/f'{camera["frame_idx"]:06d}.pkl.gz'
    sha = digest(detection)
    if completed['output_sha256'].get(str(detection)) != sha:
        raise ValueError('旧真实前端记录SHA不同')
    with gzip.open(detection, 'rb') as handle:
        saved = pickle.load(handle)
    rgb_sha = digest(camera['old_rgb'])
    provenance = saved['frame_provenance']
    if (provenance['rgb_sha256'] != rgb_sha
            or provenance['image_size_wh'] != [camera['width'], camera['height']]
            or not provenance['sam_prompt_tensor_isolated']):
        raise ValueError('前端非当前原图或SAM提示未隔离')
    path = destination/'exp_streamlined_yolo_outdoor_objects_v1/detections'/detection.name
    exact_link(path, detection)
    return dict(detection=str(path), source_detection=str(detection), detection_sha256=sha,
        rgb_sha256=rgb_sha, classes_sha256=hashlib.sha256(json.dumps(saved['classes']).encode()).hexdigest(),
        actual_new_frontend_calls=0, reused_real_frontend=True, original_complete_sha256=digest(folder/'frontend_complete.json'))


def sources():
    return [Path(__file__), PROJECT/'conceptgraph/completion/natural_inputs.py',
        PROJECT/'conceptgraph/completion/reserved_causal_inputs.py',
        PROJECT/'conceptgraph/scripts/convert_lane_expansion.py',
        PROJECT/'conceptgraph/scripts/preprocess_nuscenes_to_minimal.py']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--frontend-run', default='frontend_corrected_dev_001')
    parser.add_argument('--scene', action='append')
    parser.add_argument('--limit-cameras', type=int)
    parser.add_argument('--wall-budget-s', type=float, default=7200)
    parser.add_argument('--storage-budget-gb', type=float, default=3)
    parser.add_argument('--minimum-free-gb', type=float, default=6)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.run_id.replace('_', '').replace('-', '').isalnum():
        raise ValueError('非法run-id')
    begun = time.monotonic()
    scenes = available_development()
    if args.scene:
        scenes = [s for s in scenes if s['scene'] in args.scene]
        if len(scenes) != len(set(args.scene)):
            raise ValueError('只允许不同的开发场景')
    cameras, input_files = development_cameras(scenes, limit=args.limit_cameras)
    dest = OUTPUT/'data_manifest'/args.run_id
    if dest.exists() and not args.resume:
        raise FileExistsError('已有run需要核验续跑')
    dest.mkdir(exist_ok=args.resume)
    meta = PROJECT/'dataset/nuscenes_trainval_raw/v1.0-trainval'
    frontend = OUTPUT/'mapping_evaluation'/args.frontend_run
    input_files += [frontend/'release.json', frontend/'complete.json']
    input_files += [frontend/'inputs'/s/'frontend_complete.json' for s in sorted({c['scene'] for c in cameras})]
    input_files += [meta/n for n in ['sample.json', 'sample_data.json', 'ego_pose.json', 'calibrated_sensor.json']]
    release = dict(version='development-strict-past-lidar-v1', scenes=scenes,
        camera_tokens=[c['token'] for c in cameras], camera_count=len(cameras), frontend_run=args.frontend_run,
        source_sha256={str(p): digest(p) for p in sources()},
        input_sha256={str(p): digest(p) for p in input_files},
        archives=[str(Path('/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval')/f'v1.0-trainval{part}_blobs.tgz') for part in ['01', '02']],
        alignment='latest LIDAR_TOP at or before camera within 100ms',
        minimum_depth_m=1, maximum_depth_m=65, depth_units='uint16 millimeters; 0 missing',
        limits=dict(wall_s=args.wall_budget_s, physical_bytes=int(args.storage_budget_gb*1024**3),
                    minimum_free_bytes=int(args.minimum_free_gb*1024**3)),
        only_development=True, no_new_frontend_calls=True, raw_past_lidar_retained=True,
        gt_accessed=False, new_test_opened=False, performance_credit=False,
        missing_policy='explicit empty measurement mask; no depth support or successful detection credit',
        evaluation_time_limit='官方GT沿用原sample时刻，较相机晚约33–42ms；动态误差作为时间近似单列，不称完整严格同步GT',
        goal_complete=False)
    write_unchanged_or_new(dest/'release.json', release)
    if args.resume and (dest/'complete.json').exists():
        complete = json.loads((dest/'complete.json').read_text())
        if any(digest(dest/p) != h for p, h in complete['files_sha256'].items()):
            raise ValueError('已完成结果变化')
        print(json.dumps(dict(resume_verified=True, new_archive_reads=0, new_outputs=0,
            cameras=complete['cameras'], model_calls=0, new_test_opened=False)), flush=True)
        return
    if not (dest/'generator_source.py').exists():
        shutil.copy2(__file__, dest/'generator_source.py')
    def guard():
        if time.monotonic()-begun > release['limits']['wall_s']:
            raise TimeoutError('开发因果转换达到事前墙钟预算')
        if shutil.disk_usage(dest).free < release['limits']['minimum_free_bytes']:
            raise OSError('保留磁盘保护线，不删除文件')
    try:
        scene_tokens = {c['scene_token']: c['scene'] for c in cameras}
        sample_scenes = {s['token']: scene_tokens[s['scene_token']] for s in iter_records(meta/'sample.json') if s['scene_token'] in scene_tokens}
        lidar_by_scene, original_cameras = defaultdict(list), {}
        wanted_tokens = set(release['camera_tokens'])
        for row in iter_records(meta/'sample_data.json'):
            if row['token'] in wanted_tokens:
                original_cameras[row['token']] = row
            scene = sample_scenes.get(row['sample_token'])
            if scene and '/LIDAR_TOP/' in row['filename']:
                lidar_by_scene[scene].append(row)
        plans = []
        for camera in cameras:
            raw_camera = original_cameras[camera['token']]
            if any(raw_camera[k] != camera[k] for k in ['timestamp','filename','ego_pose_token','calibrated_sensor_token','sample_token']):
                raise ValueError('索引与原sample_data不一致')
            plans.append(dict(camera=camera, lidar=latest_past_lidar(lidar_by_scene[camera['scene']], camera['timestamp'])))
        pose_tokens = {p['camera']['ego_pose_token'] for p in plans} | {p['lidar']['ego_pose_token'] for p in plans if p['lidar']}
        poses = {p['token']: p for p in iter_records(meta/'ego_pose.json') if p['token'] in pose_tokens}
        calib = {p['token']: p for p in iter_records(meta/'calibrated_sensor.json')}
        write_unchanged_or_new(dest/'alignment_manifest.json', dict(plans=plans, only_development=True,
            no_future_lidar=True, gt_accessed=False))
        members, alignments, camera_poses = defaultdict(list), defaultdict(list), defaultdict(list)
        bindings, existing = [], {}
        provenance_path = dest/'sensor_provenance.jsonl'
        if provenance_path.exists():
            for row in map(json.loads, provenance_path.open()):
                if digest(row['depth']) != row['depth_sha256']:
                    raise ValueError('续跑深度SHA不同')
                if row['raw_lidar'] and digest(row['raw_lidar']) != row['raw_sha256']:
                    raise ValueError('续跑原始过去LiDAR SHA不同')
                existing[row['camera_token']] = row
        with provenance_path.open('a' if args.resume else 'x') as handle:
            for plan in plans:
                guard()
                camera, lidar = plan['camera'], plan['lidar']
                name, i = camera['scene'], camera['frame_idx']
                folder = dest/'converted'/name
                for part in ['color','depth','inputs']:
                    (folder/part).mkdir(parents=True, exist_ok=True)
                rgb, depth = folder/'color'/f'{i:06d}.jpg', folder/'depth'/f'{i:06d}.png'
                exact_link(rgb, camera['old_rgb'])
                binding = bind_detection(camera, frontend, folder)
                bindings.append(dict(scene=name, frame_idx=i, camera_token=camera['token'], **binding))
                camera_world = build_transform(poses[camera['ego_pose_token']]['translation'], poses[camera['ego_pose_token']]['rotation']) @ build_transform(calib[camera['calibrated_sensor_token']]['translation'], calib[camera['calibrated_sensor_token']]['rotation'])
                if not np.allclose(camera_world, camera['old_camera_world'], rtol=0, atol=1e-8):
                    raise ValueError('转换改变了原相机位姿')
                intrinsic = calib[camera['calibrated_sensor_token']]['camera_intrinsic']
                agent = dict(scene=name, frame_idx=i, camera_token=camera['token'],
                    camera_timestamp_s=camera['timestamp']/1e6,
                    lidar_timestamp_s=lidar['timestamp']/1e6 if lidar else None,
                    lidar_age_s=(camera['timestamp']-lidar['timestamp'])/1e6 if lidar else None,
                    depth_available=lidar is not None, missing_reason=None if lidar else 'no past scan within100ms',
                    rgb=str(rgb), depth=str(depth), depth_scale=1000, camera_pose=camera_world.tolist(), intrinsic=intrinsic,
                    width=camera['width'], height=camera['height'], detections=binding['detection'],
                    detections_reused=True, actual_new_frontend_calls=0, gt_in_agent=False, future_lidar_used=False)
                write_unchanged_or_new(folder/'inputs'/f'{i:06d}.json', agent)
                # 保留官方sample时间用于评价对应；LiDAR时间另列，mapper只用真实camera时间。
                alignments[name].append(dict(frame_idx=i, sample_token=camera['sample_token'],
                    timestamp=camera['official_sample_timestamp'], camera_timestamp=camera['timestamp'],
                    lidar_timestamp=lidar['timestamp'] if lidar else None, depth_available=lidar is not None,
                    timestamp_semantics='official sample timestamp; not depth capture time',
                    camera_source=camera['filename'], lidar_source=lidar['filename'] if lidar else None,
                    intrinsic=intrinsic, gt_accessed=False))
                camera_poses[name].append(camera_world.reshape(-1))
                if camera['token'] in existing:
                    continue
                if lidar is None:
                    if depth.exists():
                        raise FileExistsError('未记账深度不覆盖')
                    imageio.imwrite(depth, np.zeros((camera['height'], camera['width']), np.uint16))
                    row = dict(camera_token=camera['token'], status='MISSING_NO_PAST_MEASUREMENT',
                        raw_lidar=None, raw_sha256=None, member=None, archive=None,
                        depth=str(depth), depth_sha256=digest(depth), valid_lidar_pixels=0)
                    handle.write(json.dumps(row)+'\n')
                    handle.flush()
                else:
                    lidar_world = build_transform(poses[lidar['ego_pose_token']]['translation'], poses[lidar['ego_pose_token']]['rotation']) @ build_transform(calib[lidar['calibrated_sensor_token']]['translation'], calib[lidar['calibrated_sensor_token']]['rotation'])
                    members[lidar['filename']].append(dict(camera=camera, lidar=lidar, depth=depth,
                        intrinsic=intrinsic, transform=inverse_transform(camera_world)@lidar_world))
            found = set()
            for archive_path in release['archives']:
                if len(found) == len(members):
                    break
                archive = Path(archive_path)
                with tarfile.open(archive, 'r|gz') as tar:
                    for member in tar:
                        guard()
                        name = member.name.removeprefix('./')
                        if name not in members or name in found:
                            continue
                        if not member.isfile() or member.size > 20000000:
                            raise ValueError('不合法的归档点云成员')
                        data = tar.extractfile(member).read()
                        raw_sha = hashlib.sha256(data).hexdigest()
                        raw_path = dest/'raw_past_lidar'/f'{members[name][0]["lidar"]["token"]}.bin'
                        raw_path.parent.mkdir(exist_ok=True)
                        if raw_path.exists():
                            if digest(raw_path) != raw_sha:
                                raise ValueError('已保存原始扫描内容不同')
                        else:
                            raw_path.write_bytes(data)
                        points = np.frombuffer(data, dtype=np.float32).reshape(-1, 5)[:, :3]
                        for job in members[name]:
                            camera = job['camera']
                            transform = job['transform']
                            image = nearest_depth_image(points@transform[:3, :3].T+transform[:3, 3],
                                job['intrinsic'], camera['height'], camera['width'])
                            if job['depth'].exists():
                                raise FileExistsError('未记账深度不覆盖')
                            imageio.imwrite(job['depth'], np.rint(image*1000).astype(np.uint16))
                            row = dict(camera_token=camera['token'], status='real_past_measurement',
                                raw_lidar=str(raw_path), raw_sha256=raw_sha, member=name,
                                archive=archive_path, archive_size=archive.stat().st_size,
                                archive_mtime_ns=archive.stat().st_mtime_ns, depth=str(job['depth']),
                                depth_sha256=digest(job['depth']), valid_lidar_pixels=int(np.count_nonzero(image)))
                            handle.write(json.dumps(row)+'\n')
                            handle.flush()
                        found.add(name)
                        if len(found) % 50 == 0:
                            print(json.dumps(dict(found=len(found),requested=len(members),elapsed_s=time.monotonic()-begun)),flush=True)
                        if len(found) == len(members):
                            break
            if found != set(members):
                dump(dest/'missing_archive_members.json', dict(missing=sorted(set(members)-found),goal_complete=False))
                raise RuntimeError('归档中缺少真实过去测量，不替代')
        for name, rows in alignments.items():
            folder = dest/'converted'/name
            write_unchanged_or_new(folder/'frame_alignment.json', rows)
            if not (folder/'poses.txt').exists():
                np.savetxt(folder/'poses.txt', camera_poses[name], fmt='%.10f')
            elif not np.allclose(np.loadtxt(folder/'poses.txt'), camera_poses[name], rtol=0, atol=1e-8):
                raise ValueError('续跑相机位姿变化')
            if not (folder/'intrinsics.txt').exists():
                np.savetxt(folder/'intrinsics.txt', rows[0]['intrinsic'], fmt='%.10f')
            elif not np.allclose(np.loadtxt(folder/'intrinsics.txt'), rows[0]['intrinsic'], rtol=0, atol=1e-8):
                raise ValueError('续跑相机内参变化')
            old = frontend/'inputs'/name
            exact_link(folder/'gsa_classes_streamlined_yolo.json', old/'gsa_classes_streamlined_yolo.json')
        write_unchanged_or_new(dest/'detection_reuse.json', dict(frames=bindings,
            actual_new_frontend_calls=0, only_same_RGB_hash_reused=True, gt_accessed=False))
        files = {str(p.relative_to(dest)): digest(p) for p in dest.rglob('*') if p.is_file()}
        physical = sum(p.lstat().st_size for p in dest.rglob('*') if p.is_file())
        if physical > release['limits']['physical_bytes']:
            raise OSError('达到新增存储预算')
        if any(digest(p) != h for p,h in release['source_sha256'].items()):
            raise ValueError('运行中转换源码变化')
        ages = [(p['camera']['timestamp']-p['lidar']['timestamp'])/1e6 for p in plans if p['lidar']]
        complete = dict(cameras=len(plans), scenes=len(alignments),
            logs=len({p['camera']['log_token'] for p in plans}), real_past_lidar=len(ages),
            missing_past_lidar=len(plans)-len(ages), future_lidar_used=0,
            maximum_lidar_age_s=max(ages) if ages else None, minimum_lidar_age_s=min(ages) if ages else None,
            actual_new_frontend_calls=0, reused_real_frontend_frames=len(bindings),
            physical_storage_bytes=physical, files_sha256=files, elapsed_s=time.monotonic()-begun,
            strict_causal_inputs_prepared=True, methods_evaluated=False, new_test_opened=False, goal_complete=False)
        write_unchanged_or_new(dest/'complete.json', complete)
        print(json.dumps({k:v for k,v in complete.items() if k!='files_sha256'},ensure_ascii=False),flush=True)
    except BaseException as exc:
        dump(dest/f'interrupted_{time.time_ns()}.json',dict(error=repr(exc),goal_complete=False))
        raise


if __name__ == '__main__':
    main()
