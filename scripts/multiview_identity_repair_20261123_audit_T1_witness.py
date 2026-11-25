"""独立重算T1唯一接受事件的7/16/74实际点双向投影，不重跑建图。"""
import gzip
import json
from pathlib import Path
import pickle

import numpy as np

from conceptgraph.project_paths import PROJECT_ROOT, experiment_directory
from scripts.multiview_identity_repair_20261123_prepare_validation import digest


def main():
    root = PROJECT_ROOT
    run = experiment_directory('nuscenes', 'validation', 'mvrepair_T1_witness_audit_20261123_v1', create=True)
    manifest_path = root/'dataset/processed/multiview_identity_repair_20261123_validation_v1/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    scene = next(s for s in manifest['scenes'] if s['scene'] == 'scene-0035')
    prefix = root/'outputs/results/nuscenes/validation/iterations/mvrepair_20261123_temporal_diagnosis_v2'
    source = prefix/'mvrepair_20261123_temporal_evidence.json.gz'
    release = json.loads((prefix/'mvrepair_20261123_release.json').read_text())
    assert release['exact_L4_prefix_groups'] == 12 and release['GT_read'] is False
    passive = {r['frame_idx']: r for r in json.loads(gzip.open(source, 'rt').read())}
    points = [next(s for s in passive[f]['states'] if s['raw_id'] == identity) for f, identity in [(10, 'track-00000056'), (10, 'track-00000039'), (11, 'track-00000039')]]
    assert [len(s['real_points']) for s in points] == [7, 16, 74]
    assert points[0]['hits'] == 1 and points[1]['hits'] >= 2
    assert points[0]['timestamp'] == points[1]['timestamp'] < points[2]['timestamp']
    keys = [s['source_keys'][-1] for s in points]
    assert keys[0][:2] == keys[1][:2] and keys[0][0] == keys[2][0] == 'CAM_FRONT_LEFT'
    raw_views = [scene['groups'][f]['views']['CAM_FRONT_LEFT'] for f in [10, 10, 11]]
    masks = []
    for key, view in zip(keys, raw_views):
        assert key[1] == view['camera_timestamp_s'] <= scene['groups'][11]['reference_timestamp_s']
        with gzip.open(view['detections'], 'rb') as stream:
            det = pickle.load(stream)
        masks.append(det['mask'][key[2]])

    def project(world_points, view, mask):
        world = np.asarray(world_points)
        pose = np.asarray(view['camera_pose'])
        camera = (world-pose[:3, 3]) @ pose[:3, :3]
        uvz = camera @ np.asarray(view['intrinsic']).T
        uv = np.rint(uvz[:, :2]/uvz[:, 2, None]).astype(int)
        valid = (camera[:, 2] > 1) & (camera[:, 2] < 65) & (uv[:, 0] >= 0) & (uv[:, 0] < view['width']) & (uv[:, 1] >= 0) & (uv[:, 1] < view['height'])
        inside = np.zeros(len(world), bool)
        inside[valid] = mask[uv[valid, 1], uv[valid, 0]]
        return dict(points=len(world), valid=int(valid.sum()), inside=int(inside.sum()), fraction=float(inside.mean()))

    values = dict(forward_weak=project(points[0]['real_points'], raw_views[2], masks[2]),
        forward_previous=project(points[1]['real_points'], raw_views[2], masks[2]),
        reverse_weak=project(points[2]['real_points'], raw_views[0], masks[0]),
        reverse_previous=project(points[2]['real_points'], raw_views[1], masks[1]))
    outputs = root/'outputs/results/nuscenes/validation/iterations/mvrepair_20261123_T1_crosslog_v1/scene-0035/mvrepair_20261123_predictions.jsonl.gz'
    prediction_rows = list(map(json.loads, gzip.open(outputs, 'rt')))
    accepted = [w for w in prediction_rows[11]['audit']['T1_three']['temporal_partial_surface'] if w['accepted']]
    assert len(accepted) == 1
    expected = accepted[0]
    for key, value in values.items():
        assert value == expected[key], (key, value, expected[key])
    iou = float((masks[0]&masks[1]).sum()/(masks[0]|masks[1]).sum())
    assert abs(iou-expected['previous_mask_IoU']) < 1e-15
    original_point_proofs = []
    ownership_file = root/'outputs/results/nuscenes/validation/iterations/mvrepair_lifecycle_20261123_O1_ownership_regression_v1/scene-0035/mvrepair_lifecycle_20261123_predictions.jsonl.gz'
    ownership = list(map(json.loads, gzip.open(ownership_file, 'rt')))
    for frame, state, key in zip([10, 10, 11], points, keys):
        candidates = [o for o in ownership[frame]['audit']['O1_ownership_three']['unique_support']['initial_trace_observations'] if o['source_key'] == key]
        exact = [o for o in candidates if np.array_equal(np.asarray(o['actual_retained_world_points']), np.asarray(state['real_points']))]
        proof = dict(source_key=key, frame_idx=frame, actual_points=len(state['real_points']), exact_O1_independently_audited_trace_match=len(exact) == 1)
        if len(exact) == 1:
            proof['selected_original_scan_winner_indices'] = exact[0]['selected_point_indices']
            proof['lidar_token'] = exact[0]['lidar_token']
        original_point_proofs.append(proof)
    result = dict(passed=True, event=dict(scene='scene-0035', frame_idx=11, weak='track-00000056', strong='track-00000039'),
        actual_point_counts=[7,16,74], projection=values, previous_raw_mask_IoU=iou, source_keys=keys,
        state_timestamps=[s['timestamp'] for s in points], actual_group_timestamp=scene['groups'][11]['reference_timestamp_s'],
        original_point_proofs=original_point_proofs, model_mapping_calls=0, GT_read=False, human_identity_verified=False,
        limitation='Reprojects actual saved selected voxel/sparse points from deep-copied passive exact-L4 first12 prefix. O1 original-scan trace link is asserted only where every saved world point is exactly equal. Geometric continuity remains an automatic hypothesis, not human identity truth.',
        source_sha256=digest(__file__), input_sha256={str(p):digest(p) for p in [manifest_path,source,outputs,ownership_file]}, original_states=points)
    (run/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    report = root/'outputs/reports/multiview_identity_repair_20261123/validation/T1_actual_points_bidirectional_audit.json'
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['original_states','input_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
