"""Read-only independent checks for the local-data comparison campaign."""
from __future__ import annotations

from collections import Counter, defaultdict
from bisect import bisect_left
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import io
from itertools import zip_longest
import sys
import tarfile
import time
import zipfile


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def records(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def key(row):
    return row['sequence'], row['original_frame_id']


def cardinality_counts(predictions, truth, dimensions=2):
    """Independent augmenting paths, without using the experiment matcher."""
    adjacency = []
    for prediction in predictions:
        adjacency.append([i for i, target in enumerate(truth)
            if prediction['category'] == target['category']
            and math.dist(prediction['center'][:dimensions], target['center'][:dimensions]) <= 3.0])
    assigned = {}

    def augment(prediction, visited):
        for target in adjacency[prediction]:
            if target in visited:
                continue
            visited.add(target)
            if target not in assigned or augment(assigned[target], visited):
                assigned[target] = prediction
                return True
        return False

    matched = sum(augment(i, set()) for i in range(len(predictions)))
    return dict(predictions=len(predictions), truth=len(truth),
        true_positive=matched, unmatched_predictions=len(predictions) - matched,
        false_negative=len(truth) - matched,
        duplicate_nodes=sum(bool(neighbors) for neighbors in adjacency) - matched)


def tracking():
    started = time.monotonic()
    data = ROOT / 'dataset/processed/kitti_local_all_20261125_v1'
    run = ROOT / 'outputs/results/kitti/validation/iterations/kitti_local_all_20261125_tracking_v1'
    manifest = json.loads((data / 'manifest.json').read_text())
    verified = {}
    for group in ('source_sha256', 'frozen_metadata_sha256', 'protection_sha256'):
        for path, expected in manifest[group].items():
            assert digest(path) == expected, (group, path)
        verified[group] = len(manifest[group])
    method = list(records(data / 'method_frames.jsonl'))
    keys = [key(row) for row in method]
    assert len(keys) == len(set(keys)) == 1030
    assert dict(Counter(row['sequence'] for row in method)) == {
        **{name: 100 for name in ('0000', '0002', '0003', '0004', '0005',
            '0006', '0007', '0008', '0009', '0010')}, '0013': 30}
    assert all(not any(name in row for name in ('objects', 'truth', 'annotations', 'gt')) for row in method)
    roots = {next(parent for parent in Path(row['rgb']).parents if parent.name == 'agent_inputs')
        for row in method}
    actual = {(path.parent.parent.name, int(path.parent.name))
        for parent in roots for path in parent.rglob('rgb.png')
        if (path.parent / 'velodyne.bin').is_file()}
    assert actual == set(keys), 'available_RGB_LiDAR_pair_union_disagrees_with_denominator'
    census_path = ROOT / 'outputs/reports/kitti_local_all_20261125/actual_sensor_inventory_v1.json'
    census = json.loads(census_path.read_text())
    census_keys = set()
    for row in census['physical_locations']:
        assert Path(row['rgb']).is_file()
        if row['has_LiDAR_calib_depth']:
            assert Path(row['LiDAR']).is_file()
            census_keys.add(key(row))
    assert census_keys == set(keys)
    cache_sources = Counter(row['cached_frontend']['original_frontend'] for row in method)
    for row in method:
        assert abs(row['nominal_timestamp_s'] - row['original_frame_id'] * .1) < 1e-10
        assert not row['sensor_timestamp_measured']
        for name in ('rgb', 'lidar', 'depth', 'calib'):
            assert Path(row[name]).is_file()
    mapping = list(records(run / 'mapping_frames.jsonl.gz'))
    evaluation = list(records(run / 'evaluation_frames.jsonl.gz'))
    assert [key(row) for row in mapping] == [key(row) for row in evaluation] == keys
    assert all(row['GT_in_method'] is False for row in mapping)
    totals = {method: Counter() for method in manifest['methods']}
    sequence_totals = defaultdict(lambda: {name: Counter() for name in manifest['methods']})
    class_totals = defaultdict(lambda: {name: Counter() for name in manifest['methods']})
    for frame, score in zip(mapping, evaluation):
        for method_name in manifest['methods']:
            result = cardinality_counts(frame['predictions'][method_name], score['official_truth'])
            assert all(score['metrics'][method_name][name] == value for name, value in result.items()), key(frame)
            totals[method_name].update(result)
            sequence_totals[frame['sequence']][method_name].update(result)
            for category in manifest['primary_categories']:
                subset = cardinality_counts(
                    [row for row in frame['predictions'][method_name] if row['category'] == category],
                    [row for row in score['official_truth'] if row['category'] == category])
                class_totals[category][method_name].update(subset)
    amendment_path = data / 'evaluation_runtime_amendment_v2.json'
    amendment = json.loads(amendment_path.read_text())
    assert not amendment['configuration_or_matching_or_denominator_changes']
    assert not amendment['inference_rerun']
    assert digest(run / 'evaluation_frames.jsonl.gz') == amendment['original_completed_evaluation_file_sha256']
    for path, expected in amendment['source_sha256'].items():
        assert digest(path) == expected
    summary = json.loads((run / 'evaluation_summary_v2.json').read_text())
    for name, counts in totals.items():
        assert all(summary['methods'][name]['counts'][field] == value for field, value in counts.items())
    for sequence, methods in sequence_totals.items():
        for name, counts in methods.items():
            assert all(summary['per_sequence'][sequence][name][field] == value for field, value in counts.items())
    for category, methods in class_totals.items():
        for name, counts in methods.items():
            declared = summary['per_class'][category][name]
            assert all(declared[field] == value for field, value in counts.items() if field in declared)
    result = dict(passed=True, audit_utc=datetime.now(timezone.utc).isoformat(),
        mode='tracking_all_local_observations', frames=len(keys), sequences=len(set(k[0] for k in keys)),
        independent_matching='category+XY<=3m; unweighted augmenting-path maximum cardinality',
        actual_agent_input_pair_union=len(actual), sensor_directory_roots=sorted(map(str, roots)),
        broad_inventory_roots=len(census['roots_scanned']), physical_RGB_locations=len(census['physical_locations']),
        broad_inventory_pair_union=len(census_keys), census_sha256=digest(census_path),
        source_cache_counts=dict(cache_sources), verified_hashes=verified,
        evaluation_runtime_amendment_sha256=digest(amendment_path),
        original_evaluation_stream_preserved=True,
        counts={name: dict(value) for name, value in totals.items()},
        per_sequence={sequence: {name: dict(count) for name, count in methods.items()}
            for sequence, methods in sequence_totals.items()},
        per_class={category: {name: dict(count) for name, count in methods.items()}
            for category, methods in class_totals.items()},
        motion_comparison={'original_fusion': 'legacy center-difference estimate; frozen original configuration',
            'one_to_one_depth_robust_motion': 'one-to-one association + largest depth cluster + robust_xy + initially unknown + 3-observation static confirmation +2s dynamic TTL',
            'new_raw_verified_motion_protocol_evaluated': False,
            'actual_dynamic_attribute_accuracy_claimed': False},
        no_GPU=True, elapsed_s=time.monotonic() - started,
        audit_source_sha256=digest(__file__),
        manifest_sha256=digest(data / 'manifest.json'),
        evaluated_stream_sha256=digest(run / 'evaluation_frames.jsonl.gz'),
        limitations=['directory audit covers both known local Tracking input roots; external archive census is separately reported',
            'GT eligibility reuses frozen official evaluation rows; this audit independently reconstructs association counts, not official labels',
            'nominal 0.1s time is not measured sensor time',
            '1030 available local frames are not complete official sequences'])
    report = ROOT / 'outputs/reports/local_scope_audit_20261125/tracking_independent_v1.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


def nuscenes_prepare():
    """Check the complete fixed request universe before archive certification."""
    from conceptgraph.nuscenes.expansion_benchmark import iter_records
    started = time.monotonic()
    data = ROOT / 'dataset/processed/nuscenes_full_20261125_v1'
    manifest = json.loads((data / 'manifest.json').read_text())
    metadata = ROOT / 'dataset/raw/nuscenes_trainval/v1.0-trainval'
    scenes = json.loads((metadata / 'scene.json').read_text())
    samples = json.loads((metadata / 'sample.json').read_text())
    sample_by_token = {row['token']: row for row in samples}
    scene_by_token = {row['token']: row for row in scenes}
    expected = {}
    requested = []
    sample_keys = []
    counts = Counter()
    role_counts = Counter()
    for scene in manifest['scenes']:
        assert digest(scene['request']) == scene['request_sha256']
        request = next(records(scene['request']))
        assert scene_by_token[request['scene_token']]['name'] == request['scene'] == scene['scene']
        assert len(request['groups']) == scene['frames']
        role_counts[request['role']['access_role']] += 1
        for token, sensor in request['sensors'].items():
            value = tuple(sensor[name] for name in ('filename', 'timestamp_us', 'ego_pose_token', 'calibrated_sensor_token'))
            assert token not in expected or expected[token] == value
            expected[token] = value
        for index, group in enumerate(request['groups']):
            official = sample_by_token[group['sample_token']]
            assert group['frame_idx'] == index
            assert official['scene_token'] == request['scene_token']
            assert official['timestamp'] == group['official_sample_timestamp_us']
            sample_keys.append(group['sample_token'])
            times = []
            assert set(group['views']) == set(manifest['cameras'])
            for channel, view in group['views'].items():
                camera = request['sensors'][view['camera']]
                assert camera['filename'].split('/')[1] == channel
                times.append(camera['timestamp_us'])
                lidar = request['sensors'][view['lidar']] if view['lidar'] else None
                requested.append((request['scene_token'], camera['timestamp_us'], lidar['timestamp_us'] if lidar else None))
                counts['views'] += 1
                counts['causal_views'] += int(lidar is not None)
            assert group['reference_timestamp_us'] == max(times)
    assert len(sample_keys) == len(set(sample_keys)) == 34149
    assert set(sample_keys) == set(sample_by_token)
    assert len(manifest['scenes']) == len(scene_by_token) == 850
    checked = set()
    lidar_times = defaultdict(list)
    for row in iter_records(metadata / 'sample_data.json'):
        if row['filename'].split('/')[1] == 'LIDAR_TOP':
            lidar_times[sample_by_token[row['sample_token']]['scene_token']].append(row['timestamp'])
        if row['token'] in expected:
            value = tuple(row[name] for name in ('filename', 'timestamp', 'ego_pose_token', 'calibrated_sensor_token'))
            assert value == expected[row['token']], row['token']
            checked.add(row['token'])
    assert checked == set(expected)
    for values in lidar_times.values():
        values.sort()
    for scene_token, camera_time, selected_time in requested:
        values = lidar_times[scene_token]
        index = bisect_left(values, camera_time) - 1
        correct = values[index] if index >= 0 and 0 < camera_time - values[index] <= 100000 else None
        assert correct == selected_time
    hashes = {}
    for group in ('source_sha256', 'metadata_sha256', 'role_source_sha256'):
        for path, expected_hash in manifest[group].items():
            assert digest(path) == expected_hash, (group, path)
        hashes[group] = len(manifest[group])
    assert counts == dict(views=204894, causal_views=199957)
    result = dict(passed=True, scope='complete_metadata_and_request_contract_only',
        scenes=850, frames=34149, logs=len({row['log_token'] for row in scenes}),
        views=counts['views'], causal_depth_views=counts['causal_views'], empty_depth_views=4937,
        unique_requested_official_sensors=len(checked), role_counts=dict(role_counts),
        verified_hashes=hashes, every_camera_and_selected_LiDAR_metadata_exact=True,
        closest_strictly_past_LiDAR_within_100ms_independently_rebuilt=True,
        all_metadata_samples_retained=True, archive_members_certified=False,
        archive_source_certification='separate phaseA receipt required after ten original archives are indexed',
        caveats=['historical_holdout_access_to_audit is pending access provenance, not an unseen-test claim',
            'new scenes with previously used logs cannot be called independent unseen logs',
            'method scores, common-GT view comparison, and real model inference are outside this prepare audit'],
        no_GPU=True, elapsed_s=time.monotonic() - started,
        audit_utc=datetime.now(timezone.utc).isoformat(), audit_source_sha256=digest(__file__),
        manifest_sha256=digest(data / 'manifest.json'))
    report = ROOT / 'outputs/reports/local_scope_audit_20261125/nuscenes_prepare_independent_v1.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


def object_preflight():
    started = time.monotonic()
    data = ROOT / 'dataset/processed/kitti_object_all_20261125_v1'
    manifest = json.loads((data / 'manifest.json').read_text())
    method = list(records(data / 'method_frames.jsonl'))
    labels = list(records(data / 'evaluation_only/label_members.jsonl'))
    assert [row['frame_id'] for row in method] == [f'{i:06d}' for i in range(7481)]
    assert [row['frame_id'] for row in labels] == [row['frame_id'] for row in method]
    assert digest(data / 'method_frames.jsonl') == manifest['input_index_sha256']
    assert digest(data / 'evaluation_only/label_members.jsonl') == manifest['evaluation_index_sha256']
    archives = Path('/root/autodl-pub/KITTI_Object/raw')
    checked = Counter()
    for name, filename in (('rgb', 'data_object_image_2.zip'), ('lidar', 'data_object_velodyne.zip'),
            ('calib', 'data_object_calib.zip')):
        with zipfile.ZipFile(archives / filename) as archive:
            for row in method:
                member = row['members'][name]
                actual = archive.getinfo(member['name'])
                assert (actual.CRC, actual.file_size, actual.compress_size, actual.header_offset) == (
                    member['CRC32'], member['uncompressed_bytes'], member['compressed_bytes'], member['header_offset'])
                checked[name] += 1
    with zipfile.ZipFile(archives / 'data_object_label_2.zip') as archive:
        for row in labels:
            actual = archive.getinfo(row['member'])
            assert actual.CRC == row['CRC32'] and actual.file_size == row['uncompressed_bytes']
            checked['evaluation_only_labels'] += 1
    for group in ('source_sha256', 'protection_sha256'):
        for path, expected in manifest[group].items():
            assert digest(path) == expected, (group, path)
    for path, original in manifest['model_stat'].items():
        actual = Path(path).stat()
        assert actual.st_size == original['size'] and actual.st_mtime_ns == original['mtime_ns']
    regression_path = ROOT / 'outputs/reports/kitti_object_all_20261125/existing64_regression_v1.json'
    regression = json.loads(regression_path.read_text())
    assert regression['passed'] and regression['count'] == 64 and not regression['mismatches']
    assert regression['actual_new_model_calls'] == 0
    assert not manifest['GT_in_method'] and not manifest['actual_fusion_or_motion_or_navigation']
    assert manifest['budgets']['persistent_bytes'] == 805306368
    result = dict(passed=True, scope='7481 fixed membership and resource preflight, not full model result audit',
        frames=7481, members_verified=dict(checked), input_enumeration_exact=True,
        input_contains_official_boxes=False, preserved_legacy_reprojection_regression_frames=64,
        regression_report_sha256=digest(regression_path),
        legacy_regression_definition='same stored 64 detections; fresh original RGB/LiDAR/calibration reprojection; no new model inference comparison',
        SHA_checked_sources=len(manifest['source_sha256']), SHA_checked_protection=len(manifest['protection_sha256']),
        model_stats_checked=len(manifest['model_stat']), model_full_SHA_independently_recomputed=False,
        global_budget_GiB=dict(nuscenes=4.5, kitti_object=.75, kitti_tracking=.25, total_reserved=5.5, ceiling=6),
        no_fusion_or_motion_or_navigation_credit=True, no_GPU=True,
        elapsed_s=time.monotonic() - started, audit_utc=datetime.now(timezone.utc).isoformat(),
        audit_source_sha256=digest(__file__), manifest_sha256=digest(data / 'manifest.json'))
    report = ROOT / 'outputs/reports/local_scope_audit_20261125/object_preflight_independent_v1.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


def nuscenes_archive(number):
    """Verify a closed seek-index receipt and decode deterministic raw members."""
    from PIL import Image
    started = time.monotonic()
    assert number is not None and 1 <= number <= 10
    data = ROOT / 'dataset/processed/nuscenes_full_20261125_v1'
    complete = data / 'archives' / f'archive{number:02d}_complete.json'
    receipt = json.loads(complete.read_text())
    assert receipt['complete'] and receipt['gzip_CRC_validation']
    archive = Path(receipt['archive'])
    assert archive.stat().st_size == receipt['archive_bytes']
    assert archive.stat().st_mtime_ns == receipt['archive_mtime_ns']
    assert digest(receipt['members']) == receipt['members_sha256']
    assert digest(receipt['seek_index']) == receipt['seek_index_sha256']
    entries = list(records(receipt['members']))
    assert len(entries) == receipt['counts']['retained_members']
    assert all(offset > 0 and offset % 512 == 0 and size > 0 for _, offset, size in entries)
    channels = Counter(name.split('/')[1] for name, _, _ in entries)
    assert all(receipt['counts'][name] == count for name, count in channels.items())
    selected = {}
    for row in entries:
        selected.setdefault(row[0].split('/')[1], row)
    chosen = {row[0]: row for row in selected.values()}
    for row in random.Random(17).sample(entries, min(5, len(entries))):
        chosen[row[0]] = row
    sys.path.insert(0, '/tmp/nuscenes_full_index_20261125')
    import indexed_gzip
    assert indexed_gzip.__version__ == '1.10.3'
    evidence = []
    with indexed_gzip.IndexedGzipFile(filename=str(archive), skip_crc_check=True) as stream:
        stream.import_index(filename=receipt['seek_index'])
        for name, offset, size in sorted(chosen.values(), key=lambda row: row[1]):
            stream.seek(offset - 512)
            header = tarfile.TarInfo.frombuf(stream.read(512), encoding='utf-8', errors='strict')
            assert header.name.removeprefix('./') == name and header.size == size
            content = stream.read(size)
            assert len(content) == size
            record = dict(filename=name, offset=offset, bytes=size,
                payload_sha256=hashlib.sha256(content).hexdigest(), tar_header_name_size_checksum_valid=True)
            if name.endswith('.jpg'):
                with Image.open(io.BytesIO(content)) as image:
                    image.load()
                    record['decoded_image_wh'] = list(image.size)
            else:
                assert name.endswith('.bin') and size % 20 == 0
                record['LiDAR_five_float_record_count'] = size // 20
            evidence.append(record)
    result = dict(passed=True, archive=number, scope='single completed original archive',
        retained_members=len(entries), channels=dict(channels), random_seed=17,
        sampled_decoded_members=evidence, no_GPU=True,
        original_build_CRC_reported=True, independent_full_archive_CRC_or_SHA_repeated=False,
        imported_seek_index_CRC_disabled_by_library=True,
        member_and_seek_index_SHA_independently_verified=True,
        complete_receipt_sha256=digest(complete), all850_source_certified=False,
        elapsed_s=time.monotonic() - started, audit_utc=datetime.now(timezone.utc).isoformat(),
        audit_source_sha256=digest(__file__))
    report = ROOT / f'outputs/reports/local_scope_audit_20261125/nuscenes_archive{number:02d}_independent_v1.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'sampled_decoded_members'}, ensure_ascii=False))


def object_results():
    """Read official labels directly; independently check all completed results."""
    started = time.monotonic()
    data = ROOT / 'dataset/processed/kitti_object_all_20261125_v1'
    run = ROOT / 'outputs/results/kitti/validation/iterations/kitti_object_all_20261125_v3'
    manifest = json.loads((data / 'manifest.json').read_text())
    completion = json.loads((run / 'inference_summary.json').read_text())
    summary = json.loads((run / 'evaluation_summary.json').read_text())
    assert completion['completed_frames'] == summary['frames'] == 7481
    assert digest(run / 'inference_frames.jsonl.gz') == completion['inference_frames_sha256']
    assert digest(data / 'method_frames.jsonl') == manifest['input_index_sha256']
    methods = manifest['methods']
    mapping = {'Car': 'car', 'Truck': 'truck', 'Pedestrian': 'pedestrian', 'Cyclist': 'bicycle'}
    primary = ['car', 'truck', 'pedestrian']
    counts = {name: Counter() for name in methods}
    classes = defaultdict(lambda: {name: Counter() for name in methods})
    drives = Counter()
    failed = []
    distance_groups = defaultdict(lambda: {name: Counter() for name in methods})
    def distance_bin(z):
        return '1<z<10' if z < 10 else '10<=z<20' if z < 20 else '20<=z<40' if z < 40 else '40<=z<65'
    def matched_truth(predictions, truth):
        adjacency = [[i for i, target in enumerate(truth) if math.dist(row['center'], target['center']) <= 3.] for row in predictions]
        assigned = {}
        def augment(index, visited):
            for target in adjacency[index]:
                if target not in visited:
                    visited.add(target)
                    if target not in assigned or augment(assigned[target], visited):
                        assigned[target] = index
                        return True
            return False
        for index in range(len(predictions)):
            augment(index, set())
        return set(assigned)
    with zipfile.ZipFile('/root/autodl-pub/KITTI_Object/raw/devkit_object.zip') as devkit:
        raw_mapping = devkit.read('mapping/train_mapping.txt').decode().splitlines()
        permutation = [int(value) for value in devkit.read('mapping/train_rand.txt').decode().replace('\n', '').split(',') if value]
    assert len(raw_mapping) == 7481 and sorted(permutation) == list(range(1, 7482))

    def overlap(first, second, area_only=False):
        intersection = max(0., min(first[2], second[2]) - max(first[0], second[0])) * max(
            0., min(first[3], second[3]) - max(first[1], second[1]))
        one = max(0., first[2] - first[0]) * max(0., first[3] - first[1])
        two = max(0., second[2] - second[0]) * max(0., second[3] - second[1])
        denominator = one if area_only else one + two - intersection
        return intersection / denominator if denominator > 0 else 0.

    def count_2d(predictions, truth):
        adjacency = [[j for j, target in enumerate(truth) if overlap(row['bbox'], target['bbox']) >= .5]
            for row in predictions]
        assigned = {}
        def augment(index, visited):
            for target in adjacency[index]:
                if target not in visited:
                    visited.add(target)
                    if target not in assigned or augment(assigned[target], visited):
                        assigned[target] = index
                        return True
            return False
        return sum(augment(i, set()) for i in range(len(predictions)))

    with zipfile.ZipFile('/root/autodl-pub/KITTI_Object/raw/data_object_label_2.zip') as archive:
        for index, (frame, score) in enumerate(zip_longest(records(run / 'inference_frames.jsonl.gz'),
                records(run / 'evaluation_frames.jsonl.gz'))):
            assert frame is not None and score is not None
            assert frame['frame_id'] == score['frame_id'] == f'{index:06d}'
            assert frame['GT_in_method'] is False
            assert [frame[name] for name in ('date', 'drive', 'raw_frame')] == raw_mapping[permutation[index] - 1].split()
            raw_labels = archive.read(f'training/label_2/{index:06d}.txt')
            assert hashlib.sha256(raw_labels).hexdigest() == score['label_member_sha256']
            truth = []
            for line in raw_labels.decode().splitlines():
                if not line.strip():
                    continue
                fields = line.split()
                assert len(fields) == 15
                values = list(map(float, fields[1:]))
                truth.append(dict(category=fields[0], bbox=values[3:7],
                    center=[values[10], values[11] - values[7] / 2, values[12]]))
            drives[frame['drive']] += 1
            if frame['status'] != 'INFERRED':
                failed.append(frame['frame_id'])
            declared = {(row['category'], row['method']): row for row in score['metrics']}
            for category in primary + ['bicycle']:
                eligible = [row for row in truth if mapping.get(row['category']) == category and 1 < row['center'][2] < 65]
                ignored = [row for row in truth if row not in eligible and (row['category'] == 'DontCare'
                    or mapping.get(row['category']) == category or (category == 'car' and row['category'] == 'Van')
                    or (category == 'pedestrian' and row['category'] == 'Person_sitting'))]
                predictions = [row for row in frame['predictions'] if row['category'] == category]
                valid = [row for row in predictions if not (
                    any(overlap(row['bbox'], target['bbox'], True) >= .5 for target in ignored)
                    and not any(overlap(row['bbox'], target['bbox']) >= .5 for target in eligible))]
                matches_2d = count_2d(predictions, eligible)
                for method in methods:
                    localized = [dict(category=category, center=row['methods'][method]['center_camera0'])
                        for row in valid if row['methods'][method]['center_camera0'] is not None]
                    matched_indices = matched_truth(localized, eligible)
                    matches_3d = len(matched_indices)
                    for target_index, target in enumerate(eligible):
                        bucket = distance_groups[category, distance_bin(target['center'][2])][method]
                        bucket['truth'] += 1
                        bucket['matches_3d'] += int(target_index in matched_indices)
                    independent = dict(truth=len(eligible), predictions=len(valid),
                        ignored_predictions=len(predictions) - len(valid), matches_2d=matches_2d,
                        localized_predictions=len(localized), matches_3d=matches_3d)
                    assert all(declared[category, method][name] == value for name, value in independent.items()), (index, category, method)
                    classes[category][method].update(independent)
                    if category in primary:
                        counts[method].update(independent)
    assert sum(drives.values()) == 7481
    assert summary['failed_inference_frames'] == len(failed)
    assert summary['drive_frame_counts'] == dict(drives)
    for method, result in counts.items():
        assert all(summary['methods'][method][name] == value for name, value in result.items())
    for category, methods in classes.items():
        for method, result in methods.items():
            assert all(summary['per_class'][category][method][name] == value for name, value in result.items())
    for group in ('source_sha256', 'protection_sha256'):
        for path, expected in manifest[group].items():
            assert digest(path) == expected
    result = dict(passed=True, frames=7481, failed_frames=failed, raw_drive_count=len(drives),
        drive_frame_counts=dict(drives), methods={name: dict(value) for name, value in counts.items()},
        per_class={category: {name: dict(value) for name, value in methods.items()} for category, methods in classes.items()},
        independent_official_labels_parsed=True, gate='same-class full camera0 XYZ center<=3m',
        distance_groups=[dict(category=category, camera0_z_bin=bucket, methods={name:dict(value,recall=value['matches_3d']/value['truth'] if value['truth'] else None) for name,value in values.items()}) for (category,bucket),values in sorted(distance_groups.items())],
        distance_group_protocol_sha256=digest(ROOT/'outputs/reports/local_scope_audit_20261125/object_distance_group_protocol_v1.json'),
        independent_matching='augmenting paths; no experiment matcher/evaluator called',
        label_ignore_policy_independently_rebuilt=True, no_GPU=True,
        scope='all fixed frames/primary classes and separate bicycle proxy; integer counts, not error-sum tie resolution',
        elapsed_s=time.monotonic() - started, audit_utc=datetime.now(timezone.utc).isoformat(),
        audit_source_sha256=digest(__file__), manifest_sha256=digest(data / 'manifest.json'),
        inference_sha256=digest(run / 'inference_frames.jsonl.gz'), evaluation_sha256=digest(run / 'evaluation_frames.jsonl.gz'))
    report = ROOT / 'outputs/reports/local_scope_audit_20261125/object_results_independent_v1.json'
    with report.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('per_class', 'drive_frame_counts')}, ensure_ascii=False))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('scope', choices=['tracking', 'nuscenes-prepare', 'object-preflight', 'nuscenes-archive', 'object-results'])
    parser.add_argument('--archive', type=int)
    arguments = parser.parse_args()
    source = Path(__file__)
    snapshot = ROOT / 'outputs/reports/local_scope_audit_20261125/source_snapshot' / digest(source)
    for path in (source, ROOT / 'scripts/local_scope_audit_20261125.py'):
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == path.read_bytes()
        else:
            target.write_bytes(path.read_bytes())
    {'tracking': tracking, 'nuscenes-prepare': nuscenes_prepare,
        'object-preflight': object_preflight,
        'nuscenes-archive': lambda: nuscenes_archive(arguments.archive),
        'object-results': object_results}[arguments.scope]()


if __name__ == '__main__':
    main()
