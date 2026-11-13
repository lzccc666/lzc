"""只提取固定90帧的官方ZIP成员；不下载15/35GB完整大包。"""
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import time
import urllib.request
import zipfile
import zlib

import numpy as np
from PIL import Image

from conceptgraph.completion.kitti_tracking_geometry import calibration, oxts_poses, project_lidar, tracking_labels
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump

BASE = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/'
SEQUENCES = ['0000', '0005', '0013']
BEGUN = time.monotonic()
NETWORK_BYTES = 0
NETWORK_EVENTS = []


def guard():
    if time.monotonic()-BEGUN > 1800 or NETWORK_BYTES > 500_000_000 or shutil.disk_usage(OUTPUT).free < 6_000_000_000:
        raise RuntimeError('数据准备达到冻结时间、网络或余量预算')


class RemoteZip(io.RawIOBase):
    def __init__(self, name):
        self.url = BASE+name
        with urllib.request.urlopen(urllib.request.Request(self.url, method='HEAD'), timeout=30) as response:
            self.size, self.etag = int(response.headers['Content-Length']), response.headers['ETag']
        self.position = 0

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        if self.position < 0:
            raise ValueError('非法ZIP偏移')
        return self.position

    def range(self, start, size):
        global NETWORK_BYTES
        guard()
        size = min(size, self.size-start)
        if not 0 < size <= 32_000_000:
            raise ValueError('单次Range大小超限')
        request = urllib.request.Request(self.url, headers={'Range':f'bytes={start}-{start+size-1}',
            'If-Match':self.etag, 'Accept-Encoding':'identity'})
        with urllib.request.urlopen(request, timeout=45) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{start+size-1}/{self.size}':
                raise ValueError('服务端未严格提供请求Range，不读取完整大包')
            if response.headers.get('ETag') != self.etag:
                raise ValueError('官方ZIP在读取间改变')
            value = response.read(size+1)
        if len(value) != size:
            raise ValueError('Range长度错误')
        NETWORK_BYTES += len(value)
        NETWORK_EVENTS.append(dict(url=self.url, start=start, bytes=size, etag=self.etag,
            sha256=hashlib.sha256(value).hexdigest()))
        guard()
        return value

    def read(self, size=-1):
        if self.position >= self.size or size == 0:
            return b''
        value = self.range(self.position, self.size-self.position if size < 0 else size)
        self.position += len(value)
        return value


def selected_members(remote, infos):
    """连续成员合并Range，严格按ZIP中心目录的长度和CRC解码。"""
    result, groups = {}, []
    for info in sorted(infos, key=lambda i:i.header_offset):
        end = min(remote.size, info.header_offset+30+len(info.filename.encode())+65535+info.compress_size)
        if groups and end-groups[-1][0] <= 30_000_000 and info.header_offset-groups[-1][1] < 100_000:
            groups[-1][1] = max(groups[-1][1], end)
            groups[-1][2].append(info)
        else:
            groups.append([info.header_offset, end, [info]])
    for start, end, members in groups:
        block = remote.range(start, end-start)
        for info in members:
            offset = info.header_offset-start
            header = struct.unpack('<4s5H3I2H', block[offset:offset+30])
            if header[0] != b'PK\x03\x04' or header[2]&1 or header[3] != info.compress_type:
                raise ValueError('ZIP本地头、加密或压缩方式错误')
            begin = offset+30+header[-2]+header[-1]
            raw = block[begin:begin+info.compress_size]
            value = zlib.decompress(raw, -15) if info.compress_type == 8 else raw if info.compress_type == 0 else None
            if value is None or len(value) != info.file_size or zlib.crc32(value) != info.CRC:
                raise ValueError('ZIP成员长度或CRC不一致')
            result[info.filename] = value
    return result


def main():
    root = OUTPUT/'data_manifest/kitti_tracking_center90_002'
    root.mkdir(exist_ok=False)
    sources = {str(p.relative_to(PROJECT)):digest(p) for p in [Path(__file__).resolve(),
        PROJECT/'conceptgraph/completion/kitti_tracking_geometry.py', PROJECT/'conceptgraph/completion/kitti_geometry.py',
        OUTPUT/'task_protocol/kitti_tracking_001.md']}
    try:
        image_remote = RemoteZip('data_tracking_image_2.zip')
        image_zip = zipfile.ZipFile(image_remote)
        image_entries = {info.filename:info for info in image_zip.infolist()}
        selected = {}
        for seq in SEQUENCES:
            prefix = f'training/image_02/{seq}/'
            indices = sorted(int(Path(name).stem) for name in image_entries if name.startswith(prefix) and name.endswith('.png'))
            if len(indices) < 30 or indices != list(range(len(indices))):
                raise ValueError('所选序列帧不完整；不换样本')
            start = (len(indices)-30)//2
            selected[seq] = dict(total_frames=len(indices), frame_ids=indices[start:start+30])
        original = json.loads((OUTPUT/'data_manifest/kitti_tracking_center90_001/selection.json').read_text())
        assert selected == original['selected'], '格式修正不能重新选样'
        dump(root/'selection.json', dict(dataset='KITTI Tracking training', selected=selected,
            selected_before_label_read=True, original_selection='data_manifest/kitti_tracking_center90_001/selection.json',
            original_failed_attempt_preserved=True, source_sha256=sources, nominal_dt_s=.1,
            independent_test=False, Qwen_calls=0, GPU_credit=False, arrival_credit=0, goal_complete=False))
        small = {}
        for kind, name in [('calib','data_tracking_calib.zip'), ('oxts','data_tracking_oxts.zip'), ('labels','data_tracking_label_2.zip')]:
            remote = RemoteZip(name)
            value = remote.range(0, remote.size)
            directory = root/'official_small_archives'
            directory.mkdir(exist_ok=True)
            (directory/name).write_bytes(value)
            archive = zipfile.ZipFile(io.BytesIO(value))
            names = archive.namelist()
            entries = {}
            for seq in SEQUENCES:
                candidates = [n for n in names if n.endswith('/'+seq+'.txt') and n.startswith('training/')]
                if len(candidates) != 1:
                    raise ValueError('无法唯一定位官方'+kind+'成员 '+seq)
                entries[seq] = archive.read(candidates[0])
            small[kind] = entries
        lidar_remote = RemoteZip('data_tracking_velodyne.zip')
        lidar_zip = zipfile.ZipFile(lidar_remote)
        lidar_entries = {info.filename:info for info in lidar_zip.infolist()}
        agent, truth, provenance = [], [], []
        for seq in SEQUENCES:
            guard()
            ids = selected[seq]['frame_ids']
            image_names = [f'training/image_02/{seq}/{idx:06d}.png' for idx in ids]
            lidar_names = [f'training/velodyne/{seq}/{idx:06d}.bin' for idx in ids]
            images = selected_members(image_remote, [image_entries[name] for name in image_names])
            scans = selected_members(lidar_remote, [lidar_entries[name] for name in lidar_names])
            calib = calibration(small['calib'][seq].decode())
            poses = oxts_poses(small['oxts'][seq].decode())
            label_rows = tracking_labels(small['labels'][seq].decode(), seq, poses, calib)
            for index, (frame_id, image_name, lidar_name) in enumerate(zip(ids, image_names, lidar_names)):
                guard()
                dest = root/'agent_inputs'/seq/f'{frame_id:06d}'
                dest.mkdir(parents=True)
                (dest/'rgb.png').write_bytes(images[image_name])
                (dest/'velodyne.bin').write_bytes(scans[lidar_name])
                (dest/'calib.txt').write_bytes(small['calib'][seq])
                points = np.frombuffer(scans[lidar_name], dtype='<f4').reshape(-1, 4)
                with Image.open(io.BytesIO(images[image_name])) as im:
                    wh = im.size
                depth, evidence = project_lidar(points, calib, wh)
                Image.fromarray(depth).save(dest/'depth.png')
                pose = poses[frame_id]@calib['imu_from_camera2']
                row = dict(sequence=seq, frame_idx=index, original_frame_id=frame_id,
                    nominal_timestamp_s=frame_id*.1, sensor_timestamp_measured=False,
                    rgb=str(dest/'rgb.png'), depth=str(dest/'depth.png'), calib=str(dest/'calib.txt'),
                    lidar=str(dest/'velodyne.bin'), intrinsic=calib['intrinsic'].tolist(), pose=pose.tolist(),
                    image_wh=list(wh), projection=evidence,
                    input_sha256={str(p):digest(p) for p in dest.iterdir()})
                agent.append(row)
                truth.append(dict(sequence=seq, frame_idx=index, original_frame_id=frame_id,
                    objects=[v for v in label_rows if v['frame_idx'] == frame_id]))
                provenance.append(dict(sequence=seq, original_frame_id=frame_id, image_member=image_name,
                    image_sha256=hashlib.sha256(images[image_name]).hexdigest(), lidar_member=lidar_name,
                    lidar_sha256=hashlib.sha256(scans[lidar_name]).hexdigest()))
            print(json.dumps(dict(sequence=seq, prepared=30, network_bytes=NETWORK_BYTES)), flush=True)
        dump(root/'agent_manifest.json', dict(frames=agent, selection_sha256=digest(root/'selection.json'),
            classes=(PROJECT/'conceptgraph/vehicle_classes.txt').read_text().splitlines()))
        dump(root/'evaluation_only/truth.json', dict(frames=truth))
        dump(root/'evaluation_only/member_provenance.json', provenance)
        dump(root/'network_events.json', NETWORK_EVENTS)
        object_manifest = json.loads((OUTPUT/'data_manifest/kitti_object_uniform64_001/agent_manifest.json').read_text())
        old_rgb = {digest(row['rgb']) for row in object_manifest['frames']}
        duplicate = [dict(sequence=row['sequence'], frame_id=row['original_frame_id']) for row in agent if digest(row['rgb']) in old_rgb]
        dump(root/'complete.json', dict(frames=len(agent), sequences=len(SEQUENCES),
            selection_sha256=digest(root/'selection.json'), agent_manifest_sha256=digest(root/'agent_manifest.json'),
            truth_sha256=digest(root/'evaluation_only/truth.json'), source_sha256=sources,
            network_bytes=NETWORK_BYTES, network_requests=len(NETWORK_EVENTS),
            exact_RGB_overlap_with_old_Object64=duplicate, old_Object_overlap_checked=64,
            full_archives_downloaded=False, nominal_dt_s=.1, actual_sensor_timestamps=False,
            shared_CPU_wall_s=time.monotonic()-BEGUN, goal_complete=False))
    except BaseException as exc:
        dump(root/'network_events_on_failure.json', NETWORK_EVENTS)
        dump(root/'interrupted.json', dict(error=repr(exc), network_bytes=NETWORK_BYTES, goal_complete=False))
        raise


if __name__ == '__main__':
    main()
