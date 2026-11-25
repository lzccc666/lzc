"""增量只读核验全场景预测；不调用前端、不评分未封闭全量GT。"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'dataset/processed/nuscenes_full_20261125_v1'
RUN = ROOT / 'outputs/results/nuscenes/validation/iterations/nuscenes_full_20261125_main_v1'
REPORT = ROOT / 'outputs/reports/local_scope_audit_20261125'
METHODS = ('single_frame', 'original_fusion', 'R1', 'S2_1', 'S2_3', 'S2_6')
CHANNELS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def raw_speed(trace):
    if len(trace) < 2:
        return None
    times = [r['timestamp_s'] for r in trace]
    if any(b <= a for a, b in zip(times, times[1:])):
        return None
    points = [r['raw_center_world_m'] for r in trace]
    if any(p is None for p in points):
        return None
    pairs = [(np.asarray(points[j][:2]) - points[i][:2]) / (times[j] - times[i])
        for i in range(len(times)) for j in range(i + 1, len(times)) if times[j] - times[i] >= .75]
    vector = np.median(pairs, axis=0) if pairs else (
        np.asarray(points[-1][:2]) - points[-2][:2]) / (times[-1] - times[-2])
    return float(np.linalg.norm(vector))


def check_frame(frame, group, request, members, histories, counts):
    assert frame['sample_token'] == group['sample_token']
    assert frame['frame_idx'] == group['frame_idx']
    stamp = group['reference_timestamp_us'] / 1e6
    assert frame['render_timestamp_s'] == stamp
    assert frame['GT_in_method'] is False and frame['vehicle_safety_authorized'] is False
    assert set(frame['views']) == set(CHANNELS)
    assert set(frame['methods']) == set(METHODS)
    assert len(frame['frontends']) == 6
    assert {r['camera'] for r in frame['frontends']} == set(CHANNELS)
    for camera in CHANNELS:
        expected = group['views'][camera]
        view = frame['views'][camera]
        token = expected['camera']
        sensor = request['sensors'][token]
        assert view['camera_token'] == token and view['lidar_token'] == expected['lidar']
        assert view['camera_timestamp_s'] == sensor['timestamp_us'] / 1e6 <= stamp
        assert view['rgb_sha256'] == members[token]['sha256']
        assert view['width'] == sensor['width'] and view['height'] == sensor['height']
        if expected['lidar'] is None:
            assert view['lidar_timestamp_s'] is None and view['lidar_sha256'] is None
            counts['empty_depth_views'] += 1
        else:
            lidar = request['sensors'][expected['lidar']]
            assert view['lidar_timestamp_s'] == lidar['timestamp_us'] / 1e6
            assert 0 < sensor['timestamp_us'] - lidar['timestamp_us'] <= 100000
            assert view['lidar_sha256'] == members[expected['lidar']]['sha256']
    for name in METHODS:
        value = frame['methods'][name]
        assert len(value['nodes']) == len(value['predictions'])
        assert value['vehicle_safety_authorized'] is False and value['body_heading_available'] is False
        history = histories[name]
        for raw in value['raw_observations']:
            assert raw['key'] not in history['raw']
            camera, capture, detection_index = raw['source_key']
            view = frame['views'][camera]
            assert capture == raw['timestamp_s'] == view['camera_timestamp_s'] <= stamp
            assert raw['frame_idx'] == frame['frame_idx'] and raw['scene'] == frame['scene']
            assert raw['original_detection_index'] == detection_index
            assert raw['filtered_position_used_as_measurement'] is False
            assert raw['source'] == 'current_projected_lidar'
            for field in ('camera_token', 'lidar_token', 'rgb_sha256', 'lidar_sha256'):
                assert raw['input_sensor_tokens'][field] == view[field]
            pixels = np.asarray(raw['pixels_flat'], dtype=np.int64)
            depth = np.asarray(raw['depth_mm'], dtype=float)
            assert len(pixels) == len(depth) == raw['current_points']
            assert len(set(pixels.tolist())) == len(pixels)
            assert np.all((pixels >= 0) & (pixels < view['width'] * view['height']))
            assert np.all((depth > 1000) & (depth < 65000))
            if len(pixels):
                v, u = np.divmod(pixels, view['width'])
                z = depth / 1000.
                k = np.asarray(view['intrinsic'])
                points = np.column_stack(((u-k[0,2])*z/k[0,0], (v-k[1,2])*z/k[1,1], z))
                pose = np.asarray(view['camera_pose'])
                center = np.median(points @ pose[:3,:3].T + pose[:3,3], axis=0)
                assert np.allclose(center, raw['raw_center_world_m'], rtol=0, atol=1e-8)
            else:
                assert raw['raw_center_world_m'] is None
            history['raw'][raw['key']] = raw
            counts['raw_measurements'] += 1
            counts['stored_pixels_reprojected'] += len(pixels)
        for event in value['lineage_events']:
            if event['kind'] == 'state_raw_update':
                assert event['observation_key'] in history['raw']
                history['assignment'][(event['state_uid'], event['observation_key'])] = event
        states = {s['state_uid']: s for s in value['states']}
        assert len(states) == len(value['states'])
        for state in states.values():
            assert all(k in history['raw'] for k in state['raw_trace_keys'])
            assert all((state['state_uid'], k) in history['assignment'] for k in state['raw_trace_keys'])
        seen = set()
        for index, (prediction, node) in enumerate(zip(value['predictions'], value['nodes'])):
            assert node['node_id'] not in seen
            seen.add(node['node_id'])
            assert node['original_index'] == index
            assert node['node_id'] == prediction['node_id'] and node['category'] == prediction['category']
            assert node['center_world_m'] == prediction['center']
            assert node['estimated_motion_state'] == node['legacy_motion_state']
            assert node['verified_motion_state'] == node['motion_state']
            assert node['motion_state'] in ('unknown', 'static', 'dynamic')
            assert node['motion_history_cross_id_merged'] is False
            assert node['vehicle_safety_authorized'] is False and node['body_heading_available'] is False
            assert node['body_heading_world_rad'] is None
            if name in METHODS[:2]:
                assert node['motion_state'] == 'unknown' and not node['semantic_position_candidate']
            else:
                measurement = node['measurement']
                keys = measurement['observation_keys']
                uid = measurement.get('state_uid')
                if uid is not None:
                    assert uid in states
                    assert keys == states[uid]['raw_trace_keys']
                    assert measurement['raw_state_id'] == states[uid]['node_id']
                trace = [history['raw'][key] for key in keys]
                speed = raw_speed(trace)
                recorded = measurement['measured_raw_speed_mps']
                assert (speed is None and recorded is None) or (
                    speed is not None and recorded is not None and abs(speed-recorded) < 1e-8)
                if node['motion_state'] != 'unknown':
                    assert len(trace) >= 3
                    groups = {(r['scene'], r['frame_idx']) for r in trace}
                    assert len(groups) == len(trace)
                    times = [r['timestamp_s'] for r in trace]
                    assert all(b > a for a,b in zip(times,times[1:])) and max(times) <= stamp
                    assert times[-1] - times[0] >= 1 - 1e-8
                    assert abs(trace[-1]['render_timestamp_s'] - stamp) < 1e-6
                    assert all(history['assignment'][(uid,key)]['source_closed'] for key in keys)
                    assert trace[-1]['current_points'] >= 1
                    assert trace[-1]['depth_evidence']['support_fraction'] >= .4
                    variance = np.asarray(node['uncertainty']['diag_m2'])
                    assert np.isfinite(variance).all() and (variance > 0).all() and max(variance[:2]) <= 4
                    assert speed is not None
                    assert node['motion_state'] == ('static' if speed <= .5 else 'dynamic')
                    assert abs(node['motion_speed_xy_mps'] - speed) < 1e-8
                assert node['semantic_position_candidate'] == (node['motion_state'] == 'static')
            counts[name+'_nodes'] += 1
            counts[name+'_'+node['motion_state']] += 1
        counts[name+'_method_frames'] += 1
    counts['frames'] += 1


def audit_scene(scene, manifest, method_manifest):
    folder = RUN / scene['scene']
    complete_path = folder / 'complete.json'
    complete = json.loads(complete_path.read_text())
    fingerprint = sha(complete_path)
    destination = REPORT / 'nuscenes_scenes' / (scene['scene']+'__'+fingerprint[:16]+'.json')
    if destination.exists():
        prior = json.loads(destination.read_text())
        stream_stat = (folder/'frames.jsonl.gz').stat()
        assert prior['passed'] and prior['frames_sha256'] == complete['frames_sha256']
        assert prior['audited_stream_stat'] == dict(bytes=stream_stat.st_size, mtime_ns=stream_stat.st_mtime_ns)
        assert prior['audit_source_sha256'] == sha(__file__), 'audit source changed: use a new versioned receipt namespace'
        return prior, True
    started = time.monotonic()
    assert complete['complete'] and complete['scene'] == scene['scene']
    assert complete['CPU_worker_CUDA_initialized'] is False
    assert sha(folder/'frames.jsonl.gz') == complete['frames_sha256']
    assert sha(folder/'source_members.json') == complete['source_members_sha256']
    assert sha(scene['request']) == complete['request_sha256'] == scene['request_sha256']
    request_records = list(rows(scene['request']))
    assert len(request_records) == 1
    request = request_records[0]
    members = json.loads((folder/'source_members.json').read_text())['members']
    assert set(members) == set(request['sensors'])
    frontend = json.loads((folder/'frontend_receipt.json').read_text())
    assert set(frontend['detections']) == {g['views'][c]['camera'] for g in request['groups'] for c in CHANNELS}
    counts = Counter()
    histories = {name:dict(raw={},assignment={}) for name in METHODS}
    stream = iter(rows(folder/'frames.jsonl.gz'))
    for group in request['groups']:
        frame = next(stream)
        assert frame['scene'] == scene['scene'] and frame['log'] == scene['log_token']
        check_frame(frame, group, request, members, histories, counts)
    assert next(stream, None) is None
    assert counts['frames'] == scene['frames'] == complete['counts']['frames']
    assert counts['empty_depth_views'] == complete['counts'].get('empty_depth_views', 0)
    assert complete['counts']['actual_GPU_image_calls'] == scene['frames'] * 6
    assert frontend['counts']['actual_GPU_image_calls'] == scene['frames'] * 6
    for name in METHODS:
        assert counts[name+'_nodes'] == complete['counts'][name+'_predictions']
    result = dict(passed=True, scene=scene['scene'], counts=dict(counts), frames_sha256=complete['frames_sha256'],
        complete_receipt_sha256=fingerprint, request_sha256=scene['request_sha256'],
        audited_stream_stat=dict(bytes=(folder/'frames.jsonl.gz').stat().st_size, mtime_ns=(folder/'frames.jsonl.gz').stat().st_mtime_ns),
        audit_source_sha256=sha(__file__), method_manifest_sha256=sha(DATA/'method_manifest.json'),
        no_GPU=True, GT_opened=False, elapsed_s=time.monotonic()-started,
        scope='closed accepted scene, full frame/six-condition denominator, causal sources, stored-pixel reprojection and raw-state speed/gates',
        limits=['Stored pixel/depth provenance is checked for self-consistency; original raw mask was independently checked in the cached 20-group contract only.',
            'Unknown classifications are retained; this audit does not reconstruct lifecycle aliases or claim all unknown reasons are exhaustive.',
            'No mapping scores are computed before full850 prediction closure.'])
    write(destination, result)
    return result, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maximum-new-scenes', type=int, default=100)
    args = parser.parse_args()
    assert args.maximum_new_scenes > 0
    manifest = json.loads((DATA/'manifest.json').read_text())
    assert len(manifest['scenes']) == 850
    method_path = DATA/'method_manifest.json'
    method = json.loads(method_path.read_text()) if method_path.exists() else None
    if method:
        for path, expected in method['source_sha256'].items():
            assert sha(path) == expected, path
    statuses = []
    audited = reused = 0
    for scene in manifest['scenes']:
        folder = RUN/scene['scene']
        failed = list(folder.glob('attempts/*/failed.json')) if folder.exists() else []
        status = dict(scene=scene['scene'], expected_frames=scene['frames'], log=scene['log_token'],
            original_role=scene['role'], preserved_failed_attempts=[str(p) for p in failed])
        if (folder/'complete.json').exists() and method:
            complete_fingerprint = sha(folder/'complete.json')
            cached_receipt = REPORT/'nuscenes_scenes'/(scene['scene']+'__'+complete_fingerprint[:16]+'.json')
            if audited < args.maximum_new_scenes or cached_receipt.exists():
                result, cached = audit_scene(scene, manifest, method)
                reused += int(cached)
                audited += int(not cached)
                status.update(state='independently_audited', frames_sha256=result['frames_sha256'])
            else:
                status['state'] = 'closed_pending_next_audit_batch'
        else:
            status['state'] = 'failed_attempt_preserved_pending' if failed else 'not_closed_pending'
        statuses.append(status)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    result = dict(scope='all850 requests, no omitted pending/failed scenes', expected_scenes=850,
        expected_frames=sum(s['frames'] for s in manifest['scenes']), statuses=dict(Counter(s['state'] for s in statuses)),
        newly_audited_scenes=audited, reused_audit_receipts=reused, no_GPU=True, GT_opened=False,
        method_source_freeze_available=bool(method), utc=stamp, audit_source_sha256=sha(__file__), scenes=statuses)
    destination = REPORT/'nuscenes_batches'/(stamp+'.json')
    write(destination,result)
    print(json.dumps({k:v for k,v in result.items() if k!='scenes'},ensure_ascii=False))


if __name__ == '__main__':
    main()
