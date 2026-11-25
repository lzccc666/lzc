"""以保存的原始jitter输入独立重算voxel平均和原scan点归属，不调用建图。"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.spatial import cKDTree

from conceptgraph.project_paths import PROJECT_ROOT, experiment_directory
from scripts.multiview_identity_repair_20261123_prepare_validation import digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--method', default='O1_ownership_three')
    ap.add_argument('--out-id', required=True)
    args = ap.parse_args()
    output = experiment_directory('nuscenes', 'validation', 'mvrepair_ownership_audit_'+args.out_id, create=True)
    manifest = json.loads(args.manifest.read_text())
    records, all_claims, jitters = [], [], []
    counters = Counter()
    for scene in manifest['scenes']:
        path, = (args.run/scene['scene']).glob('*predictions*.jsonl.gz')
        rows = list(map(json.loads, gzip.open(path, 'rt')))
        assert len(rows) == len(scene['groups'])
        for group, row in zip(scene['groups'], rows):
            own = row['audit'][args.method]['unique_support']
            observations = own['initial_trace_observations']
            cache, claims = {}, defaultdict(list)
            for obs in observations:
                camera, timestamp, index = obs['source_key']
                view = group['views'][camera]
                assert timestamp == view['camera_timestamp_s'] and obs['lidar_token'] == view['lidar_token']
                assert timestamp <= group['reference_timestamp_s']
                if camera not in cache:
                    with np.load(view['projection_evidence']) as ev:
                        ev = {k: ev[k].copy() for k in ev.files}
                    cache[camera] = ev, {(int(y), int(x)): int(i) for y, x, i in zip(ev['pixel_y'], ev['pixel_x'], ev['point_index'])}, imageio.imread(view['depth'])/1000.
                ev, pixels, depth = cache[camera]
                selected = set(obs['selected_point_indices'])
                assert selected <= set(map(int, ev['point_index']))
                trace = obs.get('dense_trace')
                if obs['path'] == 'dense' and trace:
                    original = np.asarray(trace['original_input_world_points'])
                    voxels = np.asarray(trace['voxel_world_points'])
                    final = np.asarray(trace['final_world_points'])
                    contributors = trace['voxel_input_contributors']
                    y, x = np.asarray(trace['input_pixel_y'], int), np.asarray(trace['input_pixel_x'], int)
                    assert len(y) == len(x) == len(original)
                    assert sorted(i for c in contributors for i in c) == list(range(len(original)))
                    expected_voxels = np.asarray([original[np.asarray(c, int)].mean(0) for c in contributors])
                    assert np.max(np.abs(expected_voxels-voxels), initial=0) < 1e-8
                    lower = original.min(0)-trace['voxel_size_m']*.5
                    voxel_coordinates = np.floor((original-lower)/trace['voxel_size_m']).astype(int)
                    assert all(len({tuple(voxel_coordinates[i]) for i in c}) == 1 for c in contributors)
                    assert len({tuple(voxel_coordinates[c[0]]) for c in contributors}) == len(contributors)
                    distance, retained_voxels = cKDTree(voxels).query(final)
                    assert distance.max(initial=0) < 1e-7
                    retained_inputs = {int(i) for v in retained_voxels for i in contributors[int(v)]}
                    expected_indices = {pixels[int(y[i]), int(x[i])] for i in retained_inputs}
                    assert expected_indices == selected
                    assert np.array_equal(final, np.asarray(obs['actual_retained_world_points']))
                    pose, k = np.asarray(view['camera_pose']), np.asarray(view['intrinsic'])
                    z = depth[y, x]
                    measured = np.column_stack(((x-k[0, 2])*z/k[0, 0], (y-k[1, 2])*z/k[1, 1], z)) @ pose[:3, :3].T+pose[:3, 3]
                    camera_noise = (original-measured) @ pose[:3, :3]
                    jitters.extend(camera_noise.ravel().tolist())
                    counters['dense_exact_voxel_mean_contributor_pixel_scan_trace'] += 1
                elif obs['path'] == 'dense':
                    assert not selected
                    counters['dense_unavailable_trace_preserved'] += 1
                else:
                    actual = np.asarray(obs['actual_retained_world_points'])
                    pose, k = np.asarray(view['camera_pose']), np.asarray(view['intrinsic'])
                    camera_points = (actual-pose[:3, 3]) @ pose[:3, :3]
                    uvz = camera_points @ k.T
                    uv = np.rint(uvz[:, :2]/uvz[:, 2, None]).astype(int)
                    assert {pixels[int(v), int(u)] for u, v in uv} == selected
                    counters['sparse_rint_to_scan_trace'] += 1
                for point in selected:
                    claims[(*obs['source_key'][:2], obs['lidar_token'], point)].append(obs)
                records.append(dict(scene=scene['scene'], frame_idx=group['frame_idx'], source_key=obs['source_key'],
                    path=obs['path'], selected_original_points=len(selected), exact_dense_trace=bool(trace)))
            decisions = {tuple(r['source_key']): r for r in own['source_decisions']}
            for point, claimants in claims.items():
                winner = min(claimants, key=lambda o: (-o['confidence'], o['source_key'][2]))
                for obs in claimants:
                    expected = obs is winner
                    assert (point[-1] in decisions[tuple(obs['source_key'])]['retained_indices']) == expected
                if len(claimants) > 1:
                    all_claims.append(dict(scene=scene['scene'], frame_idx=group['frame_idx'], point_index=point[-1],
                        camera=point[0], timestamp_s=point[1], lidar_token=point[2], winner=winner['source_key'],
                        confidence=winner['confidence'], claimants=[o['source_key'] for o in claimants]))
            assert sum(len(v) > 1 for v in claims.values()) == own['collision_points']
            final_claims = Counter()
            for item in own['final_current_sources']:
                final_claims.update((*item['source_key'][:2], item.get('lidar_token', group['views'][item['source_key'][0]]['lidar_token']), p) for p in item['selected_point_indices'])
            assert sum(v > 1 for v in final_claims.values()) == own['final_remaining_collision_points']
            counters['remaining_collision_points'] += own['final_remaining_collision_points']
    result = dict(passed=True, method=args.method, counts=dict(counters), input=str(args.manifest), source_run=str(args.run),
        independently_reproduced_dense_voxel_means_and_exact_selected_scan_indices=True,
        saved_input_camera_noise_RMS_m=float(np.sqrt(np.mean(np.square(jitters)))) if jitters else None,
        original_support_ownership_by_detector_confidence_verified=True, scope_camera_exposure_scan=True,
        model_calls=0, mapping_calls=0, human_identity_verified=False, no_GT_read=True,
        source_sha256=digest(__file__), manifest_sha256=digest(args.manifest), records=records, collision_winners=all_claims)
    (output/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    report = PROJECT_ROOT/'outputs/reports/multiview_identity_repair_20261123/validation'
    (report/(args.out_id+'_ownership_trace_audit.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['records', 'collision_winners']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
