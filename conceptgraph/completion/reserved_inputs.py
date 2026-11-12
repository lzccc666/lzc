"""只读归档直接转换预留传感器，不推理、不打开测试、避免保存重复LiDAR原始副本。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time

import imageio.v2 as imageio
import numpy as np

from conceptgraph.completion.stage9 import PROJECT, OUTPUT, dump, digest
from conceptgraph.nuscenes.expansion_benchmark import iter_records
from conceptgraph.scripts.convert_lane_expansion import nearest_depth_image
from conceptgraph.scripts.preprocess_nuscenes_to_minimal import build_transform, inverse_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--selection', default='reserved_test_scenes_v1.json')
    parser.add_argument('--reuse-inputs')
    args = parser.parse_args()
    selection = OUTPUT/'splits'/args.selection
    chosen = json.loads(selection.read_text())['scenes']
    sources = json.loads((OUTPUT/'data_manifest/new_scene_sources_03_04.json').read_text())
    out = OUTPUT/'data_manifest'/args.run_id
    out.mkdir(parents=True, exist_ok=False)
    meta = PROJECT/'dataset/nuscenes_trainval_raw/v1.0-trainval'
    calibration = {r['token']: r for r in iter_records(meta/'calibrated_sensor.json')}
    needed_poses = {s['ego_pose_token'] for r in chosen for f in sources[r['scene']]['sequence']
                    for s in f['sensors'].values()}
    poses = {r['token']: r for r in iter_records(meta/'ego_pose.json') if r['token'] in needed_poses}
    needed = {}
    alignments = {}
    transforms = {}
    for row in chosen:
        scene = row['scene']
        dest = out/'converted'/scene
        (dest/'color').mkdir(parents=True)
        (dest/'depth').mkdir()
        alignments[scene] = []
        transforms[scene] = []
        for idx, frame in enumerate(sources[scene]['sequence']):
            cam, lidar = [frame['sensors'][c] for c in ['CAM_FRONT', 'LIDAR_TOP']]
            cc, lc = [calibration[s['calibrated_sensor_token']] for s in [cam, lidar]]
            cp, lp = [poses[s['ego_pose_token']] for s in [cam, lidar]]
            camera_world = build_transform(cp['translation'], cp['rotation']) @ build_transform(cc['translation'], cc['rotation'])
            lidar_world = build_transform(lp['translation'], lp['rotation']) @ build_transform(lc['translation'], lc['rotation'])
            lidar_camera = inverse_transform(camera_world) @ lidar_world
            transforms[scene].append(camera_world.reshape(-1))
            for channel, sensor in [('CAM_FRONT', cam), ('LIDAR_TOP', lidar)]:
                needed[sensor['filename']] = dict(scene=scene, idx=idx, channel=channel,
                    lidar_camera=lidar_camera, intrinsic=cc['camera_intrinsic'], width=cam['width'], height=cam['height'])
            alignments[scene].append(dict(frame_idx=idx, sample_token=frame['sample_token'], timestamp=frame['timestamp'],
                camera_timestamp=cam['timestamp'], camera_source=cam['filename'], lidar_source=lidar['filename'],
                intrinsic=cc['camera_intrinsic'], depth_source='projected_official_lidar_sparse_zbuffer_nearest',
                gt_accessed=False))
    dump(out/'release.json', dict(selection_sha256=digest(selection), scenes=[r['scene'] for r in chosen],
        code_sha256=digest(Path(__file__)), inference_executed=False, new_test_opened=False,
        purpose='预留数据准备，方法冻结前不得评测', raw_lidar_retained=False,
        raw_lidar_reproduction='使用源归档成员名及内容SHA256重新投影，不保留重复原始LiDAR副本'))
    complete = set()
    reusable = {}
    if args.reuse_inputs:
        reuse_root = OUTPUT/'data_manifest'/args.reuse_inputs
        if not (reuse_root/'complete.json').exists():
            raise ValueError('复用来源未完成')
        for r in map(json.loads,(reuse_root/'sensor_provenance.jsonl').open()):
            if r['member'] in needed:
                reusable[r['member']] = r
    archives = [Path('/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval')/f'v1.0-trainval{part}_blobs.tgz'
                for part in ['03', '04']]
    with (out/'sensor_provenance.jsonl').open('x') as provenance:
        for name,r in reusable.items():
            source=Path(r['output'])
            if digest(source)!=r['output_sha256']:
                raise ValueError('复用传感器指纹不一致')
            row=needed[name]
            part,suffix=('color','jpg') if row['channel']=='CAM_FRONT' else ('depth','png')
            output=out/'converted'/row['scene']/part/f"{row['idx']:06d}.{suffix}"
            output.symlink_to(source.resolve())
            provenance.write(json.dumps(dict(r,output=str(output),reused_from=str(source)))+'\n')
            complete.add(name)
        provenance.flush()
        for archive in archives:
            started = time.monotonic()
            saved = 0
            with tarfile.open(archive, mode='r|gz') as tar:
                for member in tar:
                    name = member.name.removeprefix('./')
                    if name not in needed or name in complete:
                        continue
                    # 仅写预先构造的目的文件名，不使用归档路径直接解包。
                    if not member.isfile():
                        raise ValueError('传感器成员不是普通文件')
                    if shutil.disk_usage(out).free < 400_000_000:
                        raise RuntimeError('保留400MB空间；当前run和归档均保留')
                    data = tar.extractfile(member).read()
                    row = needed[name]
                    dest = out/'converted'/row['scene']
                    if row['channel'] == 'CAM_FRONT':
                        output = dest/'color'/f"{row['idx']:06d}.jpg"
                        with output.open('xb') as f:
                            f.write(data)
                    else:
                        raw = np.frombuffer(data, dtype=np.float32)
                        if len(raw) % 5:
                            raise ValueError('LiDAR成员不是5列float32')
                        points = raw.reshape(-1, 5)[:, :3]
                        t = row['lidar_camera']
                        points = points @ t[:3, :3].T + t[:3, 3]
                        depth = nearest_depth_image(points, row['intrinsic'], row['height'], row['width'])
                        output = dest/'depth'/f"{row['idx']:06d}.png"
                        if output.exists():
                            raise FileExistsError(output)
                        imageio.imwrite(output, np.rint(depth*1000).astype(np.uint16))
                    provenance.write(json.dumps(dict(scene=row['scene'], frame_idx=row['idx'], channel=row['channel'],
                        archive=str(archive), archive_size=archive.stat().st_size,
                        archive_mtime_ns=archive.stat().st_mtime_ns, member=name, member_size=len(data),
                        member_sha256=hashlib.sha256(data).hexdigest(), output=str(output), output_sha256=digest(output)))+'\n')
                    provenance.flush()
                    complete.add(name)
                    saved += 1
                    if saved % 100 == 0:
                        print(json.dumps(dict(archive=archive.name, saved=saved, total_complete=len(complete),
                            total_required=len(needed), elapsed_s=time.monotonic()-started)), flush=True)
            dump(out/f'{archive.stem}_complete.json', dict(saved=saved, elapsed_s=time.monotonic()-started))
    if set(needed)-complete:
        dump(out/'missing.json', dict(members=sorted(set(needed)-complete), goal_complete=False))
        raise RuntimeError('传感器缺失')
    for row in chosen:
        scene = row['scene']
        dest = out/'converted'/scene
        np.savetxt(dest/'poses.txt', transforms[scene], fmt='%.10f')
        np.savetxt(dest/'intrinsics.txt', alignments[scene][0]['intrinsic'], fmt='%.10f')
        dump(dest/'frame_alignment.json', alignments[scene])
        dump(dest/'conversion_complete.json', dict(scene=scene, frames=len(alignments[scene]), new_test_opened=False))
    dump(out/'complete.json', dict(scenes=len(chosen), frames=sum(len(v) for v in alignments.values()),
        sensor_members=len(complete), new_test_opened=False, inference_executed=False, goal_complete=False))


if __name__ == '__main__':
    main()
