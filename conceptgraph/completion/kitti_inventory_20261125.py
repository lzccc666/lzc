"""Read local sensor directories and archive headers without extracting them."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import tarfile
import time
import zipfile

from conceptgraph.completion.kitti_local_all_20261125 import ROOT, DATA, REPORT, rows, sha, write


def tracking():
    roots = list(Path('/autodl-fs/data/thesis_completion_v1_bulk').glob('kitti*'))
    roots.extend(Path(p).resolve() for p in [ROOT / 'dataset/processed/preparation_history/kitti_tracking_center90_001',
        ROOT / 'dataset/processed/preparation_history/kitti_tracking_center90_002'])
    roots = sorted(set(roots)); physical = []; grouped = defaultdict(list)
    for root in roots:
        for rgb in root.rglob('rgb.png'):
            frame = rgb.parent.name; sequence = rgb.parent.parent.name
            if not frame.isdigit() or not sequence.isdigit():
                continue
            paired = all((rgb.parent / f).is_file() for f in ['velodyne.bin', 'calib.txt', 'depth.png'])
            entry = dict(root=str(root), sequence=sequence, original_frame_id=int(frame),
                rgb=str(rgb), LiDAR=str(rgb.parent / 'velodyne.bin'),
                has_LiDAR_calib_depth=paired,
                RGB_bytes=rgb.stat().st_size,
                LiDAR_bytes=(rgb.parent / 'velodyne.bin').stat().st_size if (rgb.parent / 'velodyne.bin').exists() else None)
            physical.append(entry)
            if paired:
                grouped[sequence, int(frame)].append(entry)
    frozen = {(r['sequence'], r['original_frame_id']) for r in rows(DATA / 'method_frames.jsonl')}
    actual = set(grouped)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), roots_scanned=[str(p) for p in roots],
        physical_RGB_locations=len(physical), actual_paired_unique_count=len(actual),
        frozen_method_count=len(frozen), exact_union_equal=actual == frozen,
        missing_from_frozen=[dict(sequence=k[0], original_frame_id=k[1]) for k in sorted(actual - frozen)],
        frozen_not_actual=[dict(sequence=k[0], original_frame_id=k[1]) for k in sorted(frozen - actual)],
        physical_locations=physical,
        no_unenumerated_Tracking_image_LiDAR_archives='known Tracking archives are only calibration/OXTS/labels; Object/Semantic/Depth archives inventoried as separate scopes',
        raw_Depth_Semantic_matching='separately pending legal frame mapping, not silently assumed available Tracking GT')
    write(REPORT / 'actual_sensor_inventory_v1.json', result)
    if actual != frozen:
        raise AssertionError('local_Tracking_sensor_union_does_not_match_frozen1030')
    print(json.dumps({k: result[k] for k in ['physical_RGB_locations', 'actual_paired_unique_count', 'frozen_method_count', 'exact_union_equal']}))


def depth_tar():
    begun = time.monotonic(); archive = Path('/root/autodl-pub/KITTI_Depth_Completion.tar')
    counts = Counter(); by_drive = defaultdict(Counter); metadata = []; paired_depth = Counter()
    # Tar header seek only. Sensor content is neither copied nor decompressed.
    with tarfile.open(archive, 'r:') as stream:
        for item in stream:
            if not item.isfile():
                continue
            name = item.name; parts = name.split('/'); counts['files'] += 1
            drive = next((p for p in parts if '_drive_' in p), None)
            if drive is None:
                if item.size < 50000:
                    metadata.append(dict(name=name, bytes=item.size, offset_data=item.offset_data))
                continue
            if '/image_02/data/' in name:
                counts['RGB2'] += 1; by_drive[drive]['RGB2'] += 1
            elif '/image_03/data/' in name:
                counts['RGB3'] += 1; by_drive[drive]['RGB3'] += 1
            elif '/velodyne_points/data/' in name or '/velodyne/' in name:
                counts['raw_LiDAR'] += 1; by_drive[drive]['raw_LiDAR'] += 1
            elif '/oxts/data/' in name:
                counts['OXTS'] += 1; by_drive[drive]['OXTS'] += 1
            elif 'proj_depth' in name:
                counts['projected_depth'] += 1; by_drive[drive]['projected_depth'] += 1
            if 'calib' in name or name.endswith('timestamps.txt') or name.endswith('poses.txt'):
                metadata.append(dict(name=name, bytes=item.size, offset_data=item.offset_data))
            if counts['files'] % 10000 == 0:
                print(json.dumps(dict(scanned_files=counts['files'], counts=dict(counts), wall_s=time.monotonic() - begun)), flush=True)
    write(REPORT / 'depth_tar_header_inventory_v1.json', dict(complete=True,
        archive=str(archive), archive_bytes=archive.stat().st_size, archive_mtime_ns=archive.stat().st_mtime_ns,
        counts=dict(counts), drives={key: dict(value) for key, value in by_drive.items()}, metadata=metadata,
        metadata_identity='name/offset/size header receipt, no full large tar SHA claimed',
        no_extraction=True, wall_s=time.monotonic() - begun,
        pairing_status='raw drive names must be matched by independent official mapping; no guessed odometry/raw IDs or GT-PCA heading'))
    print(json.dumps(dict(complete=True, counts=dict(counts), drives=len(by_drive), wall_s=time.monotonic() - begun)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['tracking', 'depth_tar'])
    {'tracking': tracking, 'depth_tar': depth_tar}[parser.parse_args().action]()
