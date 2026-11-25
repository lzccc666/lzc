"""Stream every local KITTI Object frame through one shared frozen frontend."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import time
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image

from conceptgraph.project_paths import experiment_directory
from conceptgraph.completion import kitti_transfer as old
from conceptgraph.completion import kitti_evaluate as evaluator
from conceptgraph.completion.kitti_geometry import calibration, project_lidar, labels
from conceptgraph.completion.kitti_local_all_20261125 import sha, read, rows, write, jsonl

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'dataset/processed/kitti_object_all_20261125_v1'
REPORT = ROOT / 'outputs/reports/kitti_object_all_20261125'
RUN_ID = 'kitti_object_all_20261125_v1'
ARCHIVES = Path('/root/autodl-pub/KITTI_Object/raw')
TEMP = ROOT / 'outputs/reports/kitti_object_all_20261125_tmp'
LOCK = ROOT / 'outputs/reports/shared_gpu_20261125.lock'
ARCHIVE_NAMES = dict(rgb='data_object_image_2.zip', lidar='data_object_velodyne.zip',
    calib='data_object_calib.zip', labels='data_object_label_2.zip', devkit='devkit_object.zip')


def guard(started, allowance=6000):
    if time.monotonic() - started > allowance:
        raise TimeoutError('Object_frozen_GPU_wall_budget_exhausted_preserve_progress')
    dest = experiment_directory('kitti', 'validation', RUN_ID)
    for p in [DATA, REPORT, dest, TEMP]:
        actual = p.resolve() if p.exists() else p.parent.resolve()
        if shutil.disk_usage(actual).free < 20 * 1024**3:
            raise OSError('actual_output_volume_below20GiB_preserve_progress')
    persistent = sum(p.stat().st_size for folder in [DATA, REPORT, dest] if folder.exists()
                     for p in folder.rglob('*') if p.is_file())
    if persistent > .75 * 1024**3:
        raise OSError('Object_persistent_above0.75GiB_report_before_expanding')
    if TEMP.exists() and sum(p.stat().st_size for p in TEMP.rglob('*') if p.is_file()) > 512 * 1024**2:
        raise OSError('Object_explicit_temporary_above512MiB')


def archive_open():
    return {key: zipfile.ZipFile(ARCHIVES / name) for key, name in ARCHIVE_NAMES.items()}


def load_frame(archives, row):
    members = row['members']
    raw = {key: archives[key].read(members[key]['name']) for key in ['rgb', 'lidar', 'calib']}
    for key, content in raw.items():
        if len(content) != members[key]['uncompressed_bytes']:
            raise AssertionError('zip_member_size_changed')
    with Image.open(io.BytesIO(raw['rgb'])) as image:
        wh = image.size
    points = np.frombuffer(raw['lidar'], dtype='<f4').reshape(-1, 4)
    calib = calibration(raw['calib'].decode())
    depth, xyz, projection = project_lidar(points, calib, wh)
    return raw, dict(depth=depth, xyz_camera0=xyz), projection


def verify(manifest):
    for path, expected in manifest['source_sha256'].items():
        if sha(path) != expected:
            raise AssertionError('frozen_Object_source_changed:' + path)
    for path, expected in manifest['model_sha256'].items():
        # Full weight hashes are evaluated at prepare and pre-GPU; no copies.
        if Path(path).stat().st_size != manifest['model_stat'][path]['size'] or Path(path).stat().st_mtime_ns != manifest['model_stat'][path]['mtime_ns']:
            raise AssertionError('model_file_stat_changed')
    for path, original in manifest['archive_stat'].items():
        p = Path(path)
        if p.stat().st_size != original['size'] or p.stat().st_mtime_ns != original['mtime_ns']:
            raise AssertionError('source_archive_changed')
    if sha(DATA / 'method_frames.jsonl') != manifest['input_index_sha256']:
        raise AssertionError('all7481_input_index_changed')


def prepare():
    DATA.mkdir(parents=True, exist_ok=False); REPORT.mkdir(parents=True, exist_ok=True)
    (DATA / 'evaluation_only').mkdir()
    archives = archive_open(); all_rows = []; label_index = []
    devkit_mapping = archives['devkit'].read('mapping/train_mapping.txt').decode().splitlines()
    permutation = [int(x) for x in archives['devkit'].read('mapping/train_rand.txt').decode().replace('\n', '').split(',') if x]
    assert len(devkit_mapping) == 7481 and sorted(permutation) == list(range(1, 7482))
    for frame in range(7481):
        name = f'{frame:06d}'; member_names = dict(rgb=f'training/image_2/{name}.png',
            lidar=f'training/velodyne/{name}.bin', calib=f'training/calib/{name}.txt')
        members = {}
        for key, member in member_names.items():
            item = archives[key].getinfo(member)
            members[key] = dict(name=member, CRC32=item.CRC, uncompressed_bytes=item.file_size,
                compressed_bytes=item.compress_size, header_offset=item.header_offset)
        date, drive, raw_frame = devkit_mapping[permutation[frame] - 1].split()
        all_rows.append(dict(frame_id=name, date=date, drive=drive, raw_frame=raw_frame, members=members))
        member = archives['labels'].getinfo(f'training/label_2/{name}.txt')
        label_index.append(dict(frame_id=name, member=member.filename, CRC32=member.CRC, uncompressed_bytes=member.file_size))
    for archive in archives.values():
        archive.close()
    jsonl(DATA / 'method_frames.jsonl', all_rows)
    jsonl(DATA / 'evaluation_only/label_members.jsonl', label_index)
    weights = [ROOT / 'checkpoint/yolov8l-world.pt', ROOT / 'checkpoint/mobile_sam.pt',
        ROOT / 'checkpoint/hf_models/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_model.safetensors']
    sources = [Path(__file__), ROOT / 'scripts/kitti_object_all_20261125.py',
        ROOT / 'conceptgraph/completion/kitti_local_all_20261125.py',
        ROOT / 'conceptgraph/completion/kitti_transfer.py', ROOT / 'conceptgraph/completion/kitti_geometry.py',
        ROOT / 'conceptgraph/completion/kitti_evaluate.py', ROOT / 'conceptgraph/completion/depth.py',
        ROOT / 'conceptgraph/scripts/streamlined_detections.py', ROOT / 'conceptgraph/utils/model_utils.py',
        ROOT / 'conceptgraph/vehicle_classes.txt', ROOT / 'conceptgraph/project_paths.py']
    protection = [ROOT / 'MEMORY.md', ROOT / 'PAPER_MEMORY.md',
        ROOT / 'outputs/project_v2/thesis_completion_v1/mapping_evaluation/kitti_object_cpu_full_001/summary.json',
        ROOT / 'outputs/project_v2/thesis_completion_v1/statistics/kitti_object_cpu_full_001/report.json']
    write(DATA / 'manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        frames=7481, frame_ids='all integers000000..007480, no success/GT subset selection',
        input_index_sha256=sha(DATA / 'method_frames.jsonl'), evaluation_index_sha256=sha(DATA / 'evaluation_only/label_members.jsonl'),
        archive_stat={str(ARCHIVES / name): dict(size=(ARCHIVES / name).stat().st_size,
            mtime_ns=(ARCHIVES / name).stat().st_mtime_ns) for name in ARCHIVE_NAMES.values()},
        archive_identity='frozen member names, CRC32, sizes, offsets; each actually read raw member additionally gets SHA256; no whole large archive SHA claimed',
        source_sha256={str(p): sha(p) for p in sources}, model_sha256={str(p): sha(p) for p in weights},
        model_stat={str(p):dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in weights},
        protection_sha256={str(p): sha(p) for p in protection},
        classes=read(ROOT / 'dataset/processed/preparation_history/kitti_object_uniform64_001/agent_manifest.json')['classes'],
        methods=old.METHODS, frontend='one shared actual YOLOv8l-world/MobileSAM/CLIP per frame, conf0.1 unchanged',
        seed=17, configuration_count=1, device='cuda:0', torch_CPU_threads=1,
        geometry='rectified camera0 3D, P2 full fourth column, each pixel nearest current projected LiDAR',
        depth=dict(minimum_m=1., maximum_m=65., minimum_retained_points=5, robust='old largest supported current depth cluster; unchanged'),
        metric='same class maximum-cardinality full3D center distance<=3m; not official AP/HOTA or Tracking XY',
        primary_categories=evaluator.PRIMARY, cyclist_bicycle='separate proxy diagnostic only',
        label_ignore_policy='unchanged legacy DontCare/Van/Person_sitting IoA and eligible-IoU guards; every ignored count saved',
        GT_in_method=False, actual_fusion_or_motion_or_navigation=False,
        split='full local Object labelled training split, used as external regression; no parameter updates or unseen-test claim',
        GPU_lock=str(LOCK.resolve()), budgets=dict(GPU_wall_s=6000, CPU_evaluation_wall_s=1800,
            persistent_bytes=int(.75 * 1024**3), temporary_bytes=512 * 1024**2, minimum_free_bytes=20 * 1024**3),
        persistent_output='numeric per-frame detections, 3D centers, depth evidence, raw/packed-mask checksums; no whole RGB/mask/CLIP cache'))
    print(json.dumps(dict(prepared=True, frames=7481)))


def regress64():
    manifest = read(DATA / 'manifest.json'); verify(manifest)
    old_run = ROOT / 'outputs/project_v2/thesis_completion_v1/mapping_evaluation/kitti_object_cpu_full_001'
    old_inputs = read(ROOT / 'dataset/processed/preparation_history/kitti_object_uniform64_001/agent_manifest.json')
    old_frames = {r['frame_id']: r for r in old_inputs['frames']}
    input_rows = {r['frame_id']: r for r in rows(DATA / 'method_frames.jsonl')}
    archives = archive_open(); mismatches = []; checked = []
    for saved in read(old_run / 'summary.json')['frames']:
        name = saved['frame_id']; observation = read(saved['observation'])
        cache = old_run / 'frames' / name / 'detections.pkl.gz'
        if sha(cache) != saved['detection_sha256']:
            raise AssertionError('protected_old64_detection_changed')
        with gzip.open(cache, 'rb') as stream:
            detection = pickle.load(stream)
        raw, data, projection = load_frame(archives, input_rows[name])
        assert hashlib.sha256(raw['rgb']).hexdigest() == detection['frame_provenance']['rgb_sha256']
        prediction = old.observations(detection, data)
        if prediction != observation['predictions']:
            mismatches.append(name)
        checked.append(name)
    for archive in archives.values():
        archive.close()
    write(REPORT / 'existing64_regression_v1.json', dict(passed=not mismatches, frames=checked,
        count=len(checked), mismatches=mismatches, actual_new_model_calls=0,
        source='same saved64 detections but freshly streamed original RGB/LiDAR/calibration and reprojection',
        no_parameter_search=True))
    if mismatches:
        raise AssertionError('old64_postprocessing_regression_failed')
    print(json.dumps(dict(passed=True, count=len(checked))))


def infer():
    import torch
    from conceptgraph.scripts.streamlined_detections import load_detection_models, detect_rgb_file
    manifest = read(DATA / 'manifest.json'); verify(manifest)
    if not read(REPORT / 'existing64_regression_v1.json')['passed']:
        raise AssertionError('must_pass_old64_before_full_run')
    for path, expected in manifest['model_sha256'].items():
        if sha(path) != expected:
            raise AssertionError('model_full_SHA_changed')
    dest = experiment_directory('kitti', 'validation', RUN_ID, create=True)
    for path in manifest['source_sha256']:
        source = Path(path); target = dest / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    TEMP.mkdir(parents=True, exist_ok=False)
    archives = archive_open(); began = None; done = 0; failures = []; raw_hashes = []
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a+') as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        began = time.monotonic()
        write(REPORT / 'gpu_started_v1.json', dict(utc=datetime.now(timezone.utc).isoformat(),
            PID=os.getpid(), lock=str(LOCK.resolve()), budget_s=6000))
        try:
            torch.set_num_threads(1); torch.manual_seed(17); np.random.seed(17)
            cfg = SimpleNamespace(device='cuda:0'); models = load_detection_models(cfg)
            for model in models[:2]:
                model.to('cuda:0'); model.overrides['device'] = 'cuda:0'
            cold = time.monotonic() - began
            with gzip.open(dest / 'inference_frames.jsonl.gz', 'wt') as stream:
                for row in rows(DATA / 'method_frames.jsonl'):
                    guard(began); tick = time.monotonic(); name = row['frame_id']
                    try:
                        raw, data, projection = load_frame(archives, row)
                        image = TEMP / 'current_rgb.png'; image.write_bytes(raw['rgb'])
                        np.random.seed(17 + int(name))
                        result, _ = detect_rgb_file(cfg, models, image, manifest['classes'])
                        rgb_sha = hashlib.sha256(raw['rgb']).hexdigest()
                        if result['frame_provenance']['rgb_sha256'] != rgb_sha:
                            raise AssertionError('actual_RGB_source_mismatch')
                        predictions = old.observations(result, data)
                        for prediction, mask in zip(predictions, result['mask']):
                            prediction['packed_mask_sha256'] = hashlib.sha256(np.packbits(mask.astype(bool)).tobytes()).hexdigest()
                            prediction['mask_shape'] = list(mask.shape)
                        record = dict(frame_id=name, date=row['date'], drive=row['drive'], raw_frame=row['raw_frame'],
                            status='INFERRED', predictions=predictions, projection=projection,
                            input_member_sha256={key: hashlib.sha256(content).hexdigest() for key, content in raw.items()},
                            frame_provenance=result['frame_provenance'], elapsed_s=time.monotonic() - tick,
                            GT_in_method=False, no_fusion_or_motion=True)
                        # This directory was created only for this run. Delete
                        # the sole transient image after durable numeric output.
                    except Exception as error:
                        record = dict(frame_id=name, date=row['date'], drive=row['drive'], raw_frame=row['raw_frame'],
                            status='FRAME_FAILED', error=repr(error), predictions=[], elapsed_s=time.monotonic() - tick,
                            GT_in_method=False, retained_in_full_denominator=True)
                        failures.append(dict(frame_id=name, error=repr(error)))
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'); stream.flush()
                    image = TEMP / 'current_rgb.png'
                    if image.exists():
                        image.unlink()
                    done += 1
                    if done % 100 == 0 or done == 1:
                        torch.cuda.synchronize()
                        print(json.dumps(dict(completed=done, total=7481, failed=len(failures),
                            elapsed_s=time.monotonic() - began, throughput_fps=done / (time.monotonic() - began),
                            GPU_allocated_MiB=torch.cuda.memory_allocated() / 1024**2)), flush=True)
            verify(manifest)
            write(dest / 'inference_summary.json', dict(completed_frames=done, failed_frames=failures,
                configuration_count=1, actual_new_frontend_frames=done - len(failures), GT_in_method=False,
                shared_single_frontend_for_both_methods=True, cold_start_s=cold,
                GPU_lock_held=True, GPU_wall_s=time.monotonic() - began,
                inference_frames_sha256=sha(dest / 'inference_frames.jsonl.gz'),
                source_manifest_sha256=sha(DATA / 'manifest.json')))
        except BaseException as error:
            write(dest / 'interrupted.json', dict(error=repr(error), completed_frames=done,
                GPU_wall_s=time.monotonic() - began, preserve_partial=True))
            raise
        finally:
            for archive in archives.values():
                archive.close()
            write(REPORT / 'gpu_released_v1.json', dict(utc=datetime.now(timezone.utc).isoformat(),
                completed_frames=done, lock=str(LOCK.resolve()), GPU_wall_s=time.monotonic() - began))
            fcntl.flock(gpu_lock, fcntl.LOCK_UN)
    print(json.dumps(dict(completed=done, failed=len(failures))))


def evaluate():
    begun = time.monotonic(); manifest = read(DATA / 'manifest.json'); verify(manifest)
    dest = experiment_directory('kitti', 'validation', RUN_ID)
    summary = read(dest / 'inference_summary.json')
    assert summary['completed_frames'] == 7481
    assert sha(dest / 'inference_frames.jsonl.gz') == summary['inference_frames_sha256']
    indexes = {r['frame_id']: r for r in rows(DATA / 'evaluation_only/label_members.jsonl')}
    assert sha(DATA / 'evaluation_only/label_members.jsonl') == manifest['evaluation_index_sha256']
    label_archive = zipfile.ZipFile(ARCHIVES / ARCHIVE_NAMES['labels'])
    totals = {method: [] for method in old.METHODS}; category_records = defaultdict(lambda: {method: [] for method in old.METHODS})
    drive_counts = Counter(); failures = 0; label_excluded = Counter()
    with gzip.open(dest / 'evaluation_frames.jsonl.gz', 'wt') as output:
        for frame in rows(dest / 'inference_frames.jsonl.gz'):
            guard(begun, allowance=1800)
            raw_labels = label_archive.read(indexes[frame['frame_id']]['member'])
            truth = labels(raw_labels.decode()); drive_counts[frame['drive']] += 1
            if frame['status'] != 'INFERRED':
                failures += 1
            frame_records = []
            for category in evaluator.PRIMARY + ['bicycle']:
                for method in old.METHODS:
                    record = evaluator.evaluate_frame(frame['predictions'], truth, category, method)
                    category_records[category][method].append(record)
                    if category in evaluator.PRIMARY:
                        totals[method].append(record)
                    frame_records.append(record)
            label_excluded.update(o['category'] for o in truth if o['category'] not in evaluator.CATEGORIES)
            output.write(json.dumps(dict(frame_id=frame['frame_id'], drive=frame['drive'],
                inference_status=frame['status'], label_member_sha256=hashlib.sha256(raw_labels).hexdigest(),
                metrics=frame_records, evaluation_only=True), ensure_ascii=False, allow_nan=False) + '\n')
    label_archive.close()
    scores = {method: evaluator.aggregate(records) for method, records in totals.items()}
    write(dest / 'evaluation_summary.json', dict(frames=7481, all_frames_retained=True, failed_inference_frames=failures,
        raw_drive_count=len(drive_counts), drive_frame_counts=dict(drive_counts), methods=scores,
        per_class={category: {method: evaluator.aggregate(records) for method, records in methods.items()}
                   for category, methods in category_records.items()},
        primary_categories=evaluator.PRIMARY, method_configuration_count=1,
        gate='same class full3D camera0 center<=3m; maximum cardinality',
        metric_AP=False, metric_HOTA=False, actual_fusion_or_motion_or_navigation=False,
        unsupported_label_categories=dict(label_excluded),
        GT_in_method=False, evaluation_only=True, CPU_wall_s=time.monotonic() - begun,
        limitations=['external already-seen regression; pretrained data overlap unknown',
            'no continuous pose, fusion, identity stability or dynamic classification credit',
            'DontCare/Van/Person_sitting ignore guards unchanged; ignored counts explicit',
            'primary full in-range GT denominator includes all detector misses/no-depth/frame failures',
            'Cyclist-bicycle separate proxy; do not merge into primary classes',
            'surface-point center differs from full-box geometric center; no official AP claim']))
    protected = {p: dict(expected=h, actual=sha(p)) for p, h in manifest['protection_sha256'].items()}
    if any(v['expected'] != v['actual'] for v in protected.values()):
        raise AssertionError('Object_old_protection_changed')
    write(REPORT / 'final_protection_v1.json', dict(passed=True, protected=protected,
        frozen_source_unchanged=True, completed_frames=7481))
    print(json.dumps(dict(frames=7481, methods=scores, failures=failures)))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'regress64', 'infer', 'evaluate'])
    args = parser.parse_args(); {'prepare': prepare, 'regress64': regress64, 'infer': infer, 'evaluate': evaluate}[args.action]()


if __name__ == '__main__':
    main()
