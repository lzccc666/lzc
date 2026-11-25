"""All locally available KITTI Tracking observations, with frozen comparisons."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

from conceptgraph.project_paths import experiment_directory
from conceptgraph.completion import kitti_tracking_experiment as old
from conceptgraph.completion.quality import object_frame_metrics

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'dataset/processed/kitti_local_all_20261125_v1'
REPORT = ROOT / 'outputs/reports/kitti_local_all_20261125'
RUN_ID = 'kitti_local_all_20261125_tracking_v1'
METHODS = old.METHODS
BULK = Path('/autodl-fs/data/thesis_completion_v1_bulk/kitti_tracking_expanded_006')
CENTER = ROOT / 'dataset/processed/preparation_history/kitti_tracking_center90_002'
FRONT = (ROOT / 'outputs/results/kitti/perception').resolve()
CENTER_FRONT = ROOT / 'outputs/project_v2/thesis_completion_v1/mapping_evaluation/kitti_tracking_frontend_full_001'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def jsonl(path, values):
    with Path(path).open('x') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def guard(started):
    if time.monotonic() - started > 1200:
        raise TimeoutError('CPU_stage_exceeds_20min')
    dest = experiment_directory('kitti', 'validation', RUN_ID)
    for folder in [DATA, REPORT, dest]:
        actual = folder.resolve() if folder.exists() else folder.parent.resolve()
        if shutil.disk_usage(actual).free < 20 * 1024**3:
            raise OSError('actual_output_volume_below_20GiB')
    if sum(p.stat().st_size for folder in [DATA, REPORT, dest] if folder.exists()
           for p in folder.rglob('*') if p.is_file()) > 256 * 1024**2:
        raise OSError('Tracking_persistent_budget_256MiB')


def verify_manifest(manifest):
    for group in ['source_sha256', 'frozen_metadata_sha256']:
        for path, expected in manifest[group].items():
            if sha(path) != expected:
                raise AssertionError('frozen_source_or_metadata_changed:' + path)


def prepare():
    import zipfile
    begun = time.monotonic(); DATA.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=True); (DATA / 'evaluation_only').mkdir()
    primary = read(BULK / 'agent_manifest.json')
    extra = read(CENTER / 'agent_manifest.json')
    cached = {(r['sequence'], r['original_frame_id']): r for r in read(FRONT / 'summary.json')['frames']}
    center_cached = {(r['sequence'], r['original_frame_id']): r for r in read(CENTER_FRONT / 'summary.json')['frames']}
    selected = {(r['sequence'], r['original_frame_id']): copy.deepcopy(r) for r in primary['frames']}
    for record in extra['frames']:
        selected.setdefault((record['sequence'], record['original_frame_id']), copy.deepcopy(record))
    methods = []
    for key in sorted(selected):
        row = selected[key]; saved = cached.get(key, center_cached.get(key))
        if saved is None:
            raise AssertionError('cached_frontend_missing:' + str(key))
        for field in ['rgb', 'depth', 'lidar', 'calib']:
            if not Path(row[field]).is_file():
                raise AssertionError('actual_sensor_missing:' + row[field])
        if Path(row['lidar']).stat().st_size % 16 or Path(row['lidar']).stat().st_size == 0:
            raise AssertionError('invalid_LiDAR_binary')
        pose = np.asarray(row['pose'])
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise AssertionError('invalid_OXTS_camera2_pose')
        row['cached_frontend'] = dict(path=saved['detections'], sha256=saved['output_sha256'][saved['detections']],
            original_frontend='expanded_GPU' if key in cached else 'center90_CPU',
            original_frontend_input_sha256=saved['input_sha256'])
        if not Path(saved['detections']).is_file():
            raise AssertionError('actual_frontend_cache_missing')
        methods.append(row)
    assert len(methods) == 1030 and len({r['sequence'] for r in methods}) == 11
    jsonl(DATA / 'method_frames.jsonl', methods)
    config_release = read(ROOT / 'outputs/results/kitti/mapping/release.json')
    center_release = read(ROOT / 'outputs/project_v2/thesis_completion_v1/mapping_evaluation/kitti_tracking_mapping_full_001/release.json')
    assert config_release['configs'] == center_release['configs']
    metadata = [BULK / 'agent_manifest.json', BULK / 'selection.json', FRONT / 'release.json', FRONT / 'summary.json',
        CENTER / 'agent_manifest.json', CENTER_FRONT / 'release.json', CENTER_FRONT / 'summary.json',
        ROOT / 'outputs/results/kitti/mapping/release.json',
        ROOT / 'outputs/project_v2/thesis_completion_v1/mapping_evaluation/kitti_tracking_mapping_full_001/release.json',
        DATA / 'method_frames.jsonl']
    eval_sources = [BULK / 'evaluation_only/truth.json', CENTER / 'evaluation_only/truth.json']
    write(DATA / 'evaluation_only/source_manifest.json', dict(
        truth_sha256={str(p): sha(p) for p in eval_sources},
        category_map=old.CATEGORIES, gate='same category maximum-cardinality world XY distance <=3m',
        official_AP_or_HOTA=False, DontCare='excluded from target truth; unmatched predictions remain counted; not complete false-positive rate',
        dynamic_truth='not inferred from label yaw; nominal time only, no measured-time dynamic accuracy claim'))
    seq = defaultdict(list)
    for row in methods:
        seq[row['sequence']].append(row['original_frame_id'])
    matrix = []
    with zipfile.ZipFile(BULK / 'official_small_archives/data_tracking_oxts.zip') as archive:
        for i in range(21):
            name = f'{i:04d}'; count = len(archive.read('training/oxts/' + name + '.txt').splitlines())
            frame_ids = seq.get(name, [])
            matrix.append(dict(sequence=name, official_pose_frames=count, local_RGB_LiDAR_pose_frames=len(frame_ids),
                local_frame_ids=frame_ids, raw_sequence_complete=len(frame_ids) == count,
                label_and_calibration_available=True,
                status='all_local_observations_included' if frame_ids else 'no_local_RGB_LiDAR',
                methods={method: dict(frames=len(frame_ids), operation='actual_CPU_replay_shared_fixed_frontend') for method in METHODS}))
    write(DATA / 'tracking_scene_matrix.json', matrix)
    source_paths = set(old.pipeline_sources())
    source_paths.update([Path(__file__), ROOT / 'scripts/kitti_local_all_20261125.py',
        ROOT / 'conceptgraph/completion/kitti_tracking_experiment.py',
        ROOT / 'conceptgraph/completion/kitti_tracking_geometry.py',
        ROOT / 'conceptgraph/completion/kitti_geometry.py', ROOT / 'conceptgraph/project_paths.py'])
    source_paths = sorted(p for p in source_paths if p.is_file())
    protection = [ROOT / 'MEMORY.md', ROOT / 'PAPER_MEMORY.md',
        ROOT / 'outputs/results/kitti/mapping/frames.jsonl', ROOT / 'outputs/results/kitti/evaluation/report.json',
        ROOT / 'outputs/project_v2/thesis_completion_v1/mapping_evaluation/kitti_tracking_mapping_full_001/frames.jsonl']
    write(DATA / 'manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        scope='all locally available actual KITTI Tracking RGB+LiDAR+OXTS observations after deduplication',
        frame_count=len(methods), sequence_count=len(seq), methods=METHODS,
        classes=primary['classes'], configs=config_release['configs'],
        source_sha256={str(p): sha(p) for p in source_paths},
        frozen_metadata_sha256={str(p): sha(p) for p in metadata},
        protection_sha256={str(p): sha(p) for p in protection},
        configs_identical_between_old_1000_and_old_center90=True,
        source_cache='fixed existing GPU observations for1000 + CPU observations for30 distinct0013 frames; same input for all methods',
        sensor_time=dict(mode='nominal_original_frame_id_times0.1s', measured=False, pose_source='official OXTS + calibration, not GT object boxes'),
        frame_reference='rectified camera2 pose in initial IMU world; forward is camera optical Z projected to world XY',
        object_relative_heading_supported=False, GT_in_method=False, Qwen_calls=0, GPU_calls=0,
        seed=17, configuration_count=1, selection_after_model_results=False,
        evaluation_threshold_m=3., primary_categories=['car', 'truck', 'pedestrian'],
        split='external previously-seen regression/validation, no model training',
        budgets=dict(CPU_wall_s=1200, persistent_bytes=256 * 1024**2, minimum_free_bytes=20 * 1024**3),
        adjacent_inventory=dict(Object_training=7481, Object_methods=['mask_points_mean', 'mask_points_robust_mean'],
            Object_pose_status='no matched continuous poses yet; no fusion/identity credit',
            SemanticKITTI_LiDAR_frames=43552, SemanticKITTI_RGB_status='absent within SemanticKITTI zips; Depth tar matching under audit'),
        model_fingerprints_from_cache_releases={str(p): read(p).get('model_sha256', {}) for p in [FRONT / 'release.json', CENTER_FRONT / 'release.json']},
        prepare_elapsed_s=time.monotonic() - begun))
    print(json.dumps(dict(prepared=True, frames=1030, sequences=11)))


def track():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU_replay_requires_explicitly_hidden_CUDA')
    from omegaconf import OmegaConf
    import torch
    from conceptgraph.slam.streamlined_mapping import MappingFrame, StreamlinedMapper
    begun = time.monotonic(); manifest = read(DATA / 'manifest.json'); verify_manifest(manifest)
    dest = experiment_directory('kitti', 'validation', RUN_ID, create=True)
    for path in manifest['source_sha256']:
        source = Path(path); target = dest / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    sequence = None; completed = []; counters = Counter()
    try:
        with gzip.open(dest / 'mapping_frames.jsonl.gz', 'wt') as stream:
            for row in rows(DATA / 'method_frames.jsonl'):
                guard(begun)
                for path, expected in row['input_sha256'].items():
                    if sha(path) != expected:
                        raise AssertionError('sensor_input_changed:' + path)
                cached = row['cached_frontend']
                if sha(cached['path']) != cached['sha256']:
                    raise AssertionError('frontend_cache_changed')
                if row['sequence'] != sequence:
                    sequence = row['sequence']
                    mappers = {method: StreamlinedMapper(OmegaConf.create(copy.deepcopy(config)), manifest['classes'])
                               for method, config in manifest['configs'].items()}
                with gzip.open(cached['path'], 'rb') as handle:
                    detections = pickle.load(handle)
                image = np.asarray(Image.open(row['rgb']).convert('RGB'))
                depth = np.asarray(Image.open(row['depth']), dtype=float) / 1000.
                predictions = {}; timings = {}
                for method, mapper in mappers.items():
                    np.random.seed(17 + row['frame_idx']); tick = time.monotonic()
                    mapper.update(MappingFrame(row['frame_idx'], image, depth, np.asarray(row['intrinsic']),
                        np.asarray(row['pose']), copy.deepcopy(detections), row['rgb'], timestamp_s=row['nominal_timestamp_s']))
                    predictions[method] = old.predict(mapper.objects, manifest['classes'])
                    timings[method] = time.monotonic() - tick
                predictions['single_frame'] = old.predict(mappers['original_fusion'].last_observations, manifest['classes'])
                visible = {method: old.in_view(predictions[method], row) for method in METHODS}
                record = dict(sequence=sequence, frame_idx=row['frame_idx'], original_frame_id=row['original_frame_id'],
                    nominal_timestamp_s=row['nominal_timestamp_s'], measured_sensor_timestamps=False,
                    camera_pose_world=row['pose'], camera_forward_world_xy=np.asarray(row['pose'])[:2, 2].tolist(),
                    coordinate_frame_source='official_OXTS_camera2_optical_axis_proxy_not_object_heading',
                    intrinsic=row['intrinsic'], image_wh=row['image_wh'], predictions=visible,
                    all_predictions=predictions, mapping_CPU_s=timings, GT_in_method=False,
                    source_input_index=dict(sequence=sequence, original_frame_id=row['original_frame_id']))
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush(); completed.append((sequence, row['original_frame_id'])); counters[sequence] += 1
                if len(completed) % 50 == 0:
                    print(json.dumps(dict(done=len(completed), sequence=sequence, wall_s=time.monotonic() - begun)), flush=True)
        if torch.cuda.is_initialized():
            raise AssertionError('GPU_initialized_in_CPU_replay')
        verify_manifest(manifest)
        write(dest / 'mapping_summary.json', dict(completed_frames=len(completed), sequence_counts=dict(counters),
            all_method_frames={method: len(completed) for method in METHODS}, GT_in_method=False, GPU_calls=0,
            mapping_frames_sha256=sha(dest / 'mapping_frames.jsonl.gz'), wall_s=time.monotonic() - begun,
            manifest_sha256=sha(DATA / 'manifest.json')))
    except BaseException as error:
        write(dest / 'interrupted.json', dict(error=repr(error), completed_frames=len(completed), elapsed_s=time.monotonic() - begun))
        raise
    print(json.dumps(dict(completed_frames=len(completed), wall_s=time.monotonic() - begun)))


def evaluate():
    begun = time.monotonic(); manifest = read(DATA / 'manifest.json'); verify_manifest(manifest)
    dest = experiment_directory('kitti', 'validation', RUN_ID)
    summary = read(dest / 'mapping_summary.json')
    assert summary['completed_frames'] == 1030
    assert sha(dest / 'mapping_frames.jsonl.gz') == summary['mapping_frames_sha256']
    sources = read(DATA / 'evaluation_only/source_manifest.json')
    truths = {}
    # GT source only opens after all inference has finished and closed its stream.
    for path, expected in sources['truth_sha256'].items():
        assert sha(path) == expected
        for row in read(path)['frames']:
            truths.setdefault((row['sequence'], row['original_frame_id']), row)
    totals = {method: Counter() for method in METHODS}; per_seq = defaultdict(lambda: {m: Counter() for m in METHODS})
    per_class = defaultdict(lambda: {m: Counter() for m in METHODS}); excluded = Counter(); by_sequence = defaultdict(list)
    with gzip.open(dest / 'evaluation_frames.jsonl.gz', 'wt') as stream:
        for frame in rows(dest / 'mapping_frames.jsonl.gz'):
            guard(begun); key = frame['sequence'], frame['original_frame_id']
            truth, omit = old.eligible_truth(truths[key]['objects'], frame['image_wh'])
            excluded.update(o['exclusion_reason'] for o in omit)
            record = dict(sequence=key[0], frame_idx=frame['frame_idx'], original_frame_id=key[1],
                evaluation_only=True, official_truth=truth, excluded=omit, metrics={})
            for method in METHODS:
                metric = object_frame_metrics(frame['predictions'][method], truth, threshold_m=3.)
                record['metrics'][method] = metric
                count = {k: metric[k] for k in ['predictions', 'truth', 'true_positive', 'unmatched_predictions', 'false_negative', 'duplicate_nodes']}
                count['center_error_sum_m'] = sum(m['distance_m'] for m in metric['matches'])
                totals[method].update(count); per_seq[key[0]][method].update(count)
                for category in old.CATEGORIES.values():
                    part = object_frame_metrics([o for o in frame['predictions'][method] if o['category'] == category],
                        [o for o in truth if o['category'] == category], threshold_m=3.)
                    per_class[category][method].update({k: part[k] for k in ['predictions', 'truth', 'true_positive', 'duplicate_nodes']})
            identity_frame = dict(frame_idx=frame['frame_idx'], predictions=frame['predictions'], evaluation_only_official=truth, metrics=record['metrics'])
            by_sequence[key[0]].append(identity_frame)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    identity_rows = []
    for sequence, frames in by_sequence.items():
        for method in METHODS:
            identity_input = old.prepare_sequence(frames, method); returned = old.return_diagnostics(identity_input)
            for support in old.SUPPORTS:
                identity = old.identity_counts(identity_input, support)
                identity_rows.append(dict(log_token=sequence, sequence=sequence, method=method, support=support,
                    counts=identity['counts'] | returned['counts']))
    identity, by_sequence_identity = old.aggregate_statistics(identity_rows)
    write(dest / 'identity_rows.json', identity_rows)
    write(dest / 'identity_by_sequence.json', by_sequence_identity)
    results = {method: dict(counts=dict(count), metrics=old.object_rates(count)) for method, count in totals.items()}
    write(dest / 'evaluation_summary.json', dict(dataset='KITTI Tracking all actual local observations',
        scope=dict(frames=1030, sequences=11, raw_complete_sequences=0), methods=results,
        per_sequence={key: {m: dict(v) for m, v in values.items()} for key, values in per_seq.items()},
        per_class={key: {m: dict(v) for m, v in values.items()} for key, values in per_class.items()},
        identity=identity, excluded_label_reasons=dict(excluded),
        gate='same class world XY<=3m maximum-cardinality; unchanged full local denominator',
        AP=False, HOTA=False, dynamic_GT_accuracy_evaluated=False, nominal_dt_s=.1,
        label_scope='only Car/Truck/Pedestrian with valid3D identity,1<camera2_z<65,bbox intersects image',
        limitations=['all local windows, not complete official sequences',
            'DontCare/unsupported/unlabelled predictions remain unmatched; matched fraction is not full false-positive precision',
            'cached frontend device differs by source; no new frontend latency claim',
            'no measured timestamps or raw multi-frame trace: legacy motion estimated, raw verified motion unknown',
            '3m object association is not official AP/HOTA or vehicle navigation success'],
        GT_in_method=False, wall_s=time.monotonic() - begun))
    protection = {path: dict(expected=expected, actual=sha(path)) for path, expected in manifest['protection_sha256'].items()}
    if any(r['expected'] != r['actual'] for r in protection.values()):
        raise AssertionError('legacy_protected_changed')
    write(REPORT / 'tracking_final_protection_v1.json', dict(passed=True, protection=protection,
        final_source_sha256={p: sha(p) for p in manifest['source_sha256']},
        all_source_frozen_unchanged=True, no_GPU=True))
    print(json.dumps(results))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'tracking', 'evaluate'])
    action = parser.parse_args().action
    {'prepare': prepare, 'tracking': track, 'evaluate': evaluate}[action]()


if __name__ == '__main__':
    main()
