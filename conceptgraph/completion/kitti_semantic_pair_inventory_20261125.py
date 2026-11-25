"""Prove local raw-RGB/odometry sensor intersections without running a model."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import tarfile
import time
import zipfile

import numpy as np

from conceptgraph.completion.kitti_local_all_20261125 import ROOT, REPORT, write


def main():
    begun = time.monotonic()
    mapping = json.loads((REPORT / 'official_odometry_raw_mapping_receipt_v1.json').read_text())
    rows = mapping['rows']; by_drive = {r['raw_drive'] + '_sync': r for r in rows}
    archive = Path('/root/autodl-pub/KITTI_Depth_Completion.tar')
    rgb = {r['sequence']: {} for r in rows}; unexpected_late_raw = False
    # Existing complete header audit established raw/ is a contiguous initial
    # subtree. Stop at first projected-depth subtree; no sensor extraction.
    with tarfile.open(archive, 'r:') as stream:
        for member in stream:
            if not member.isfile():
                continue
            if '/raw/' not in member.name:
                break
            if '/image_02/data/' not in member.name or not member.name.endswith('.png'):
                continue
            drive = next((part for part in member.name.split('/') if '_drive_' in part), None)
            if drive not in by_drive:
                continue
            row = by_drive[drive]; raw_frame = int(Path(member.name).stem)
            if row['raw_start'] <= raw_frame <= row['raw_end']:
                frame = raw_frame - row['raw_start']
                rgb[row['sequence']][frame] = dict(name=member.name, raw_frame=raw_frame,
                    bytes=member.size, offset_data=member.offset_data)
    semantic = Path('/root/autodl-pub/SemanticKITTI')
    velo = zipfile.ZipFile(semantic / 'data_odometry_velodyne.zip')
    calib = zipfile.ZipFile(semantic / 'data_odometry_calib.zip')
    labels = zipfile.ZipFile(semantic / 'data_odometry_labels.zip')
    matrix = []; pairs = []
    velos = {}; label_names = set(labels.namelist())
    for item in velo.infolist():
        if '/velodyne/' in item.filename and item.filename.endswith('.bin'):
            parts = item.filename.split('/'); velos[parts[2], int(Path(item.filename).stem)] = item
    for row in rows:
        sequence = row['sequence']; expected = row['raw_end'] - row['raw_start'] + 1
        pose_path = f'dataset/sequences/{sequence}/poses.txt'
        pose_bytes = labels.read(pose_path)
        pose_array = np.loadtxt(pose_bytes.decode().splitlines()).reshape(-1, 3, 4)
        calib_path = f'dataset/sequences/{sequence}/calib.txt'; calib_bytes = calib.read(calib_path)
        times_path = f'dataset/sequences/{sequence}/times.txt'; times_bytes = calib.read(times_path)
        times = np.atleast_1d(np.loadtxt(times_bytes.decode().splitlines()))
        if not np.isfinite(pose_array).all() or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
            raise AssertionError('invalid_supplied_pose_or_time')
        paired = []
        for frame, member in sorted(rgb[sequence].items()):
            if (sequence, frame) not in velos or frame >= len(pose_array) or frame >= len(times):
                continue
            scan = velos[sequence, frame]
            paired.append(frame)
            pairs.append(dict(sequence=sequence, frame_idx=frame, raw_drive=row['raw_drive'],
                RGB=member, LiDAR_member=scan.filename, LiDAR_CRC32=scan.CRC, LiDAR_bytes=scan.file_size,
                pose_row=frame, times_row=frame,
                point_instance_label_available=f'dataset/sequences/{sequence}/labels/{frame:06d}.label' in label_names))
        matrix.append(dict(sequence=sequence, raw_drive=row['raw_drive'], raw_start=row['raw_start'], raw_end=row['raw_end'],
            expected_odometry_frames=expected, local_RGB_frames=len(rgb[sequence]),
            common_RGB_LiDAR_SuMa_pose_time_frames=len(paired), common_frame_ids=paired,
            common_complete=len(paired) == expected, supplied_pose_rows=len(pose_array), supplied_time_rows=len(times),
            calib_sha256=hashlib.sha256(calib_bytes).hexdigest(), pose_sha256=hashlib.sha256(pose_bytes).hexdigest(),
            times_sha256=hashlib.sha256(times_bytes).hexdigest(),
            status='new_SemanticKITTI_protocol_required' if paired else 'no_matching_local_raw_RGB'))
    for item in [velo, calib, labels]:
        item.close()
    destination = REPORT / 'semantic_RGB_LiDAR_pose_pair_inventory_v1.json'
    write(destination, dict(complete=True, official_mapping_metadata_source=mapping,
        source='official KITTI devkit exact raw ranges, including sequence08 raw_start1100; no guessed frame offsets',
        matrix=matrix, total_common_frames=len(pairs), matched_sequences=sum(bool(r['common_RGB_LiDAR_SuMa_pose_time_frames']) for r in matrix),
        pairs=pairs,
        SuMa_pose_source='https://semantic-kitti.org/dataset.html: provider-supplied SLAM estimates used for annotation; not OXTS and no strict causal claim',
        existing_paper_Tracking_Object_results_unchanged=True, new_model_calls=0, new_GPU_calls=0,
        warning='point semantic/instance labels are not Object3D boxes; cannot insert point-surface means into old3m box-center results; new protocol and budget needed',
        no_sensor_download_or_extraction=True, header_scan_wall_s=time.monotonic() - begun,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(common_frames=len(pairs), matched_sequences=sum(bool(r['common_RGB_LiDAR_SuMa_pose_time_frames']) for r in matrix),
        by_sequence={r['sequence']: r['common_RGB_LiDAR_SuMa_pose_time_frames'] for r in matrix})))


if __name__ == '__main__':
    main()
