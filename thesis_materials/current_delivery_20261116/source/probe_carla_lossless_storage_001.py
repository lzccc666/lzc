"""在受保护旧帧的只读副本上实测无损视频，逐像素核对后才讨论未来存储。"""
import datetime
import hashlib
import json
from pathlib import Path
import time
import traceback
import zlib

import cv2
import numpy as np

from conceptgraph.completion.stage9 import digest, dump

R = Path(__file__).resolve().parents[1]


def main():
    out = R/'storage_audit/lossless_sensor_probe_001'
    out.mkdir(exist_ok=False)
    source = R/'carla_official/prius_controller_navigation_005/car20/full_rule_geometry/frames'
    folders = sorted(source.iterdir())[:40]
    assert len(folders)==40
    records, reports = [], []
    begin = time.monotonic()
    try:
        for filename in ['rgb.png','official_depth_raw_bgr.png']:
            path = out/(Path(filename).stem+'.mkv')
            first = cv2.imread(str(folders[0]/filename), cv2.IMREAD_COLOR)
            height,width = first.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.CAP_FFMPEG,
                cv2.VideoWriter_fourcc(*'FFV1'), 5., (width,height), True)
            assert writer.isOpened(), '此OpenCV不支持FFV1封装，保留失败不自动换有损编码'
            size = 0
            hashes = []
            start = time.monotonic()
            try:
                for folder in folders:
                    p = folder/filename
                    pixels = cv2.imread(str(p),cv2.IMREAD_COLOR)
                    assert pixels.shape == first.shape
                    checksum = hashlib.sha256(pixels.tobytes()).hexdigest()
                    records.append(dict(source=str(p), file_sha256=digest(p), raw_BGR_sha256=checksum))
                    hashes.append(checksum)
                    size += p.stat().st_size
                    writer.write(pixels)
            finally:
                writer.release()
            encode_s = time.monotonic()-start
            capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
            assert capture.isOpened()
            exact = 0
            start = time.monotonic()
            try:
                for folder,checksum in zip(folders,hashes):
                    success, actual = capture.read()
                    assert success, '无损文件丢帧'
                    expected = cv2.imread(str(folder/filename),cv2.IMREAD_COLOR)
                    assert np.array_equal(actual,expected), '逐像素不等，不能声明无损'
                    assert hashlib.sha256(actual.tobytes()).hexdigest() == checksum
                    exact += 1
                assert not capture.read()[0], '存在额外未登记帧'
            finally:
                capture.release()
            reports.append(dict(kind=filename, frames=len(folders), pixel_exact_frames=exact,
                original_PNG_bytes=size, FFV1_MKV_bytes=path.stat().st_size,
                reduction_fraction=1-path.stat().st_size/size, encode_wall_s=encode_s,
                decode_and_PNG_comparison_wall_s=time.monotonic()-start,
                stored_channel_order='BGR8; official depth remains original packed24bit',
                video_sha256=digest(path)))
        lidar_size = 0
        combined = bytearray()
        for folder in folders:
            p=folder/'lidar.bin'
            raw=p.read_bytes()
            combined.extend(raw)
            lidar_size+=len(raw)
            records.append(dict(source=str(p),file_sha256=digest(p),bytes=len(raw)))
        compressed=zlib.compress(combined,6)
        assert zlib.decompress(compressed)==combined
        lidar=out/'lidar_concatenation.zlib'
        lidar.open('xb').write(compressed)
        reports.append(dict(kind='lidar.bin',frames=len(folders),raw_bytes=lidar_size,
            compressed_bytes=len(compressed),byte_exact_roundtrip=True,
            reduction_fraction=1-len(compressed)/lidar_size,sha256=digest(lidar)))
        dump(out/'input_manifest.json',records)
        assert all(digest(r['source'])==r['file_sha256'] for r in records)
        dump(out/'report.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            methods=reports,distinct_existing_sensor_frames=40,actual_new_sensor_frames=0,
            actual_GPU_calls=0,all_original_files_rehashed_unchanged=True,
            wall_s=time.monotonic()-begin,source_sha256=digest(__file__),
            scope='既有原厂Prius car20连续40帧的无损存储实测；不删除旧文件，不更改当前冻结实验格式',
            future_use='必须在未来新实验注册前冻结块大小/索引/解码哈希与崩溃前缀恢复规则；本试验不是导航或实时证据',
            goal_complete=False))
        print(json.dumps(reports,ensure_ascii=False),flush=True)
    except BaseException as exc:
        dump(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),
            all_prefixes_retained=True,goal_complete=False))
        raise


if __name__ == '__main__':
    main()
