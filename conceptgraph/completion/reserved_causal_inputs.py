"""预留关键帧改为严格过去LiDAR；首帧缺失明确保存，不开启测试方法。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tarfile
import time

import imageio.v2 as imageio
import numpy as np

from conceptgraph.completion.natural_inputs import latest_past_lidar
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump
from conceptgraph.nuscenes.expansion_benchmark import iter_records
from conceptgraph.scripts.convert_lane_expansion import nearest_depth_image
from conceptgraph.scripts.preprocess_nuscenes_to_minimal import build_transform, inverse_transform


def write_unchanged_or_new(path, value):
    """续跑只接受相同语义的元数据，不覆盖已有内容。"""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'resume metadata changed: {path}')
    else:
        dump(path, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--selection', default='reserved_test_scenes_v2.json')
    parser.add_argument('--old-inputs', default='reserved_inputs_002')
    parser.add_argument('--max-cameras', type=int)
    parser.add_argument('--wall-budget-s', type=float, default=3600.)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    dest = OUTPUT/'data_manifest'/args.run_id
    if dest.exists() and not args.resume:
        raise FileExistsError('existing run needs fingerprint-checked --resume')
    dest.mkdir(exist_ok=args.resume)
    meta = PROJECT/'dataset/nuscenes_trainval_raw/v1.0-trainval'
    selection_path = OUTPUT/'splits'/args.selection
    source_path = OUTPUT/'data_manifest/new_scene_sources_03_04.json'
    selected = json.loads(selection_path.read_text())['scenes']
    sources = json.loads(source_path.read_text())
    old = OUTPUT/'data_manifest'/args.old_inputs
    provenance = {r['member']: r for r in map(json.loads, (old/'sensor_provenance.jsonl').open())
                  if r['channel'] == 'CAM_FRONT'}
    cameras = []
    for scene in selected:
        alignment = json.loads((old/'converted'/scene['scene']/'frame_alignment.json').read_text())
        old_poses = np.loadtxt(old/'converted'/scene['scene']/'poses.txt').reshape(-1, 4, 4)
        for i, record in enumerate(sources[scene['scene']]['sequence']):
            camera, old_lidar = [record['sensors'][c] for c in ['CAM_FRONT', 'LIDAR_TOP']]
            if alignment[i]['camera_source'] != camera['filename']:
                raise ValueError('source RGB and preserved frame order differ')
            cameras.append(dict(camera, scene=scene['scene'], frame_idx=i,
                old_lidar_timestamp=old_lidar['timestamp'], old_camera_world=old_poses[i].tolist(),
                source_rgb=alignment[i]['camera_source'], source_sample_token=record['sample_token']))
    if args.max_cameras is not None:
        if args.max_cameras <= 0:
            raise ValueError('max-cameras must be positive')
        cameras = cameras[:args.max_cameras]
    dependencies = [Path(__file__), PROJECT/'conceptgraph/completion/natural_inputs.py',
        PROJECT/'conceptgraph/scripts/convert_lane_expansion.py',
        PROJECT/'conceptgraph/scripts/preprocess_nuscenes_to_minimal.py']
    release = dict(version='reserved-strict-past-lidar-v1',
        source_sha256={str(p): digest(p) for p in dependencies},
        input_sha256={str(p): digest(p) for p in [selection_path, source_path,
            old/'sensor_provenance.jsonl', meta/'sample.json', meta/'sample_data.json',
            meta/'ego_pose.json', meta/'calibrated_sensor.json']},
        camera_tokens=[c['token'] for c in cameras], old_input_run=args.old_inputs,
        cameras=len(cameras), max_cameras=args.max_cameras,
        wall_budget_s=args.wall_budget_s, storage_budget_bytes=1500000000, minimum_free_bytes=6000000000,
        alignment='latest LIDAR_TOP not later than camera, maximum age 100ms',
        missing_policy='no past scan: explicit depth_available=false, all-zero missing measurement mask, no source measurement credit',
        old_files_moved_or_deleted=False, inference_executed=False, new_test_opened=False, goal_complete=False)
    write_unchanged_or_new(dest/'release.json', release)
    if args.resume and (dest/'complete.json').exists():
        result = json.loads((dest/'complete.json').read_text())
        if any(digest(dest/name) != sha for name, sha in result['files_sha256'].items()):
            raise ValueError('completed run changed; no continuation or repair performed')
        print(json.dumps(dict(resume_verified=True, new_archive_reads=0, new_outputs=0,
                              cameras=result['cameras'], new_test_opened=False), ensure_ascii=False), flush=True)
        return
    if not (dest/'generator_source.py').exists():
        shutil.copy2(__file__, dest/'generator_source.py')
    selected_samples = {r['token']: r['scene_token'] for r in iter_records(meta/'sample.json')
                        if r['scene_token'] in {s['scene_token'] for s in selected}}
    names = {s['scene_token']: s['scene'] for s in selected}
    needed_scenes = {c['scene'] for c in cameras}
    lidar_by_scene, original_cameras = {}, {}
    tokens = {c['token'] for c in cameras}
    for record in iter_records(meta/'sample_data.json'):
        if record['token'] in tokens:
            original_cameras[record['token']] = record
        scene_token = selected_samples.get(record['sample_token'])
        if scene_token is not None and names[scene_token] in needed_scenes and '/LIDAR_TOP/' in record['filename']:
            lidar_by_scene.setdefault(names[scene_token], []).append(record)
    if set(original_cameras) != tokens:
        raise ValueError('missing original camera metadata')
    plans = []
    for camera in cameras:
        raw = original_cameras[camera['token']]
        if any(raw[k] != camera[k] for k in ['timestamp', 'filename', 'ego_pose_token', 'calibrated_sensor_token', 'sample_token']):
            raise ValueError('source camera differs from raw sample_data')
        lidar = latest_past_lidar(lidar_by_scene.get(camera['scene'], []), camera['timestamp'])
        plans.append(dict(camera=camera, lidar=lidar))
    needed_poses = {p['camera']['ego_pose_token'] for p in plans} | {
        p['lidar']['ego_pose_token'] for p in plans if p['lidar'] is not None}
    poses = {p['token']: p for p in iter_records(meta/'ego_pose.json') if p['token'] in needed_poses}
    calibration = {p['token']: p for p in iter_records(meta/'calibrated_sensor.json')}
    if set(poses) != needed_poses:
        raise ValueError('missing original ego pose')
    existing = {}
    if (dest/'sensor_provenance.jsonl').exists():
        for row in map(json.loads, (dest/'sensor_provenance.jsonl').open()):
            if digest(row['output']) != row['output_sha256']:
                raise ValueError('resume output SHA differs')
            existing[(row['camera_token'], row['channel'])] = row
    members, alignments, camera_poses, input_paths = {}, {}, {}, []
    maximum_pose_error = 0.
    with (dest/'sensor_provenance.jsonl').open('a' if args.resume else 'x') as handle:
        for plan in plans:
            camera, lidar = plan['camera'], plan['lidar']
            ego = poses[camera['ego_pose_token']]
            cc = calibration[camera['calibrated_sensor_token']]
            ew = build_transform(ego['translation'], ego['rotation'])
            cw = ew@build_transform(cc['translation'], cc['rotation'])
            pose_error = float(np.max(np.abs(cw-np.asarray(camera['old_camera_world']))))
            maximum_pose_error = max(maximum_pose_error, pose_error)
            if pose_error > 1e-8:
                raise ValueError('strict past conversion unexpectedly changes the original camera pose')
            folder = dest/'converted'/camera['scene']
            for sub in ['color', 'depth', 'inputs']:
                (folder/sub).mkdir(parents=True, exist_ok=True)
            rgb, depth = folder/'color'/f'{camera["frame_idx"]:06d}.jpg', folder/'depth'/f'{camera["frame_idx"]:06d}.png'
            source = provenance[camera['filename']]
            if digest(source['output']) != source['output_sha256']:
                raise ValueError('preserved RGB differs from original archive evidence')
            if not rgb.exists():
                rgb.symlink_to(Path(source['output']).resolve())
            if digest(rgb) != source['output_sha256']:
                raise ValueError('RGB reuse differs')
            if (camera['token'], 'CAM_FRONT') not in existing:
                handle.write(json.dumps(dict(camera_token=camera['token'], channel='CAM_FRONT',
                    status='exact_preserved_RGB', output=str(rgb), output_sha256=digest(rgb),
                    member=camera['filename'], member_sha256=source['member_sha256'], reused_from=source['output']))+'\n')
                handle.flush()
            agent = dict(scene=camera['scene'], frame_idx=camera['frame_idx'], sample_token=camera['sample_token'],
                camera_token=camera['token'], camera_timestamp_s=camera['timestamp']/1e6,
                lidar_timestamp_s=lidar['timestamp']/1e6 if lidar else None,
                lidar_age_s=(camera['timestamp']-lidar['timestamp'])/1e6 if lidar else None,
                depth_available=lidar is not None, missing_lidar_reason=None if lidar else 'no past LiDAR within 100ms',
                rgb=str(rgb), depth=str(depth), depth_scale=1000,
                depth_source='past real sparse LiDAR zbuffer' if lidar else 'explicit empty mask for missing measurement',
                camera_pose=cw.tolist(), intrinsic=cc['camera_intrinsic'],
                rear_axle_reference_world_pose=ew.tolist(), rear_axle_yaw_rad=math.atan2(ew[1, 0], ew[0, 0]),
                rear_axle_reference_source='raw calibrated ego_pose; rear axle projected onto ground',
                measured_speed_mps=None, width=camera['width'], height=camera['height'],
                gt_in_agent=False, future_lidar_used=False, inference_executed=False)
            input_path = folder/'inputs'/f'{camera["frame_idx"]:06d}.json'
            write_unchanged_or_new(input_path, agent)
            input_paths.append(input_path)
            alignments.setdefault(camera['scene'], []).append(dict(frame_idx=camera['frame_idx'],
                sample_token=camera['sample_token'], camera_timestamp=camera['timestamp'],
                timestamp=lidar['timestamp'] if lidar else None, camera_source=camera['filename'],
                lidar_source=lidar['filename'] if lidar else None, intrinsic=cc['camera_intrinsic'],
                depth_available=lidar is not None, missing_reason=agent['missing_lidar_reason'], gt_accessed=False))
            camera_poses.setdefault(camera['scene'], []).append(cw.reshape(-1))
            if (camera['token'], 'LIDAR_TOP') in existing:
                continue
            if lidar is None:
                if depth.exists():
                    raise FileExistsError('missing-depth output exists without checked provenance')
                imageio.imwrite(depth, np.zeros((camera['height'], camera['width']), dtype=np.uint16))
                handle.write(json.dumps(dict(camera_token=camera['token'], channel='LIDAR_TOP',
                    status='MISSING_NO_PAST_MEASUREMENT', member=None, member_sha256=None,
                    output=str(depth), output_sha256=digest(depth), valid_lidar_pixels=0))+'\n')
                handle.flush()
            else:
                lp, lc = poses[lidar['ego_pose_token']], calibration[lidar['calibrated_sensor_token']]
                lw = build_transform(lp['translation'], lp['rotation'])@build_transform(lc['translation'], lc['rotation'])
                members.setdefault(lidar['filename'], []).append(dict(camera_token=camera['token'], output=depth,
                    transform=inverse_transform(cw)@lw, intrinsic=cc['camera_intrinsic'],
                    height=camera['height'], width=camera['width']))
        write_unchanged_or_new(dest/'alignment_manifest.json', dict(plans=[dict(
            scene=p['camera']['scene'], frame_idx=p['camera']['frame_idx'], camera_token=p['camera']['token'],
            camera_timestamp=p['camera']['timestamp'], previous_lidar_timestamp=p['camera']['old_lidar_timestamp'],
            lidar=p['lidar']) for p in plans], future_lidar_used=False, new_test_opened=False))
        complete = set()
        for part in ['03', '04']:
            if len(complete) == len(members):
                break
            archive = Path('/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval')/f'v1.0-trainval{part}_blobs.tgz'
            with tarfile.open(archive, 'r|gz') as tar:
                for member in tar:
                    if time.monotonic()-started > args.wall_budget_s:
                        raise TimeoutError('causal input budget; resume only with same source/input fingerprints')
                    name = member.name.removeprefix('./')
                    if name not in members or name in complete:
                        continue
                    if not member.isfile() or member.size > 20000000:
                        raise ValueError('unexpected raw LiDAR member')
                    if shutil.disk_usage(dest).free < release['minimum_free_bytes']:
                        raise OSError('disk reserve; no deletion')
                    data = tar.extractfile(member).read()
                    raw = np.frombuffer(data, dtype=np.float32)
                    if len(raw) % 5:
                        raise ValueError('invalid LiDAR data')
                    for row in members[name]:
                        xyz = raw.reshape(-1, 5)[:, :3]@row['transform'][:3, :3].T+row['transform'][:3, 3]
                        image = nearest_depth_image(xyz, row['intrinsic'], row['height'], row['width'])
                        if row['output'].exists():
                            raise FileExistsError('depth without checked provenance')
                        imageio.imwrite(row['output'], np.rint(image*1000).astype(np.uint16))
                        handle.write(json.dumps(dict(camera_token=row['camera_token'], channel='LIDAR_TOP',
                            status='real_past_measurement', archive=str(archive), archive_size=archive.stat().st_size,
                            archive_mtime_ns=archive.stat().st_mtime_ns, member=name,
                            member_sha256=hashlib.sha256(data).hexdigest(), output=str(row['output']),
                            output_sha256=digest(row['output']), valid_lidar_pixels=int(np.count_nonzero(image))))+'\n')
                        handle.flush()
                    complete.add(name)
                    if len(complete) % 50 == 0:
                        print(json.dumps(dict(completed_lidar_members=len(complete), requested_members=len(members),
                            elapsed_s=time.monotonic()-started)), flush=True)
                    if len(complete) == len(members):
                        break
        if len(complete) != len(members):
            dump(dest/'missing_archive_members.json', dict(members=sorted(set(members)-complete), goal_complete=False))
            raise RuntimeError('past measurements missing in archive; no substitution')
    for scene, alignment in alignments.items():
        folder = dest/'converted'/scene
        write_unchanged_or_new(folder/'frame_alignment.json', alignment)
        if not (folder/'poses.txt').exists():
            np.savetxt(folder/'poses.txt', camera_poses[scene], fmt='%.10f')
        if not (folder/'intrinsics.txt').exists():
            np.savetxt(folder/'intrinsics.txt', alignment[0]['intrinsic'], fmt='%.10f')
    files = {str(p.relative_to(dest)): digest(p) for p in dest.rglob('*') if p.is_file()}
    physical_bytes = sum(p.lstat().st_size for p in dest.rglob('*') if p.is_file())
    if physical_bytes > release['storage_budget_bytes']:
        raise OSError('physical storage budget exceeded')
    if any(digest(name) != sha for name, sha in release['source_sha256'].items()):
        raise ValueError('causal conversion source changed during run')
    ages = [(p['camera']['timestamp']-p['lidar']['timestamp'])/1e6 for p in plans if p['lidar'] is not None]
    missing = [dict(scene=p['camera']['scene'], frame_idx=p['camera']['frame_idx'], camera_token=p['camera']['token'])
               for p in plans if p['lidar'] is None]
    result = dict(cameras=len(plans), scenes=len(alignments), real_past_lidar_pairs=len(ages),
        missing_past_lidar=len(missing), missing_frames=missing, maximum_past_lidar_age_s=max(ages) if ages else None,
        minimum_past_lidar_age_s=min(ages) if ages else None,
        previous_future_lidar_pairs=sum(p['camera']['old_lidar_timestamp'] > p['camera']['timestamp'] for p in plans),
        future_lidar_pairs=0, maximum_camera_pose_difference=maximum_pose_error,
        input_conversion_complete=True, sensor_coverage_complete=not missing,
        missing_measurement_not_counted_as_real_depth=True, physical_storage_bytes=physical_bytes,
        elapsed_s=time.monotonic()-started, files_sha256=files,
        inference_executed=False, new_test_opened=False, goal_complete=False)
    write_unchanged_or_new(dest/'complete.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['files_sha256', 'missing_frames']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
