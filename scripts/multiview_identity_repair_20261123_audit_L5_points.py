"""独立核对L5保存的原扫描赢家索引/像素/sparse回投；不重跑建图。"""
from collections import Counter
import gzip
import json
from pathlib import Path
import pickle

import imageio.v2 as imageio
import numpy as np
from scipy.spatial import cKDTree

from conceptgraph.project_paths import PROJECT_ROOT, experiment_directory
from scripts.multiview_identity_repair_20261123_prepare_validation import digest


def main():
    base = PROJECT_ROOT
    run = experiment_directory('nuscenes', 'validation', 'mvrepair_L5_point_audit_20261123_v1', create=True)
    records, accepted, sources = [], [], {}
    for split, folder in [('pilot', 'multiview_20261123_pilot_v1'), ('regression', 'multiview_identity_repair_20261123_validation_v1')]:
        path = base/'dataset/processed'/folder/'manifest.json'
        manifest = json.loads(path.read_text())
        source_run = base/'outputs/results/nuscenes/validation/iterations'/('mvrepair_lifecycle_20261123_L5_sameview_'+split+'_v1')
        release_path = source_run/'mvrepair_lifecycle_20261123_release.json'
        release = json.loads(release_path.read_text())
        for name, value in release['source_sha256'].items():
            assert digest(name) == value
            assert digest(source_run/Path(name).name) == value
            sources[name] = value
        for scene in manifest['scenes']:
            file = source_run/scene['scene']/'mvrepair_lifecycle_20261123_predictions.jsonl.gz'
            for group, row in zip(scene['groups'], map(json.loads, gzip.open(file, 'rt'))):
                cache = {}
                for method in ['L5_sameview_three', 'L5_sameview_front', 'A_front_sameview_only']:
                    audit = row['audit'][method]['sameview']
                    observations = audit['observations']
                    for item in observations:
                        camera, timestamp, index = item['source_key']
                        view = group['views'][camera]
                        assert timestamp == view['camera_timestamp_s'] and item['lidar_token'] == view['lidar_token']
                        assert timestamp <= group['reference_timestamp_s']
                        if camera not in cache:
                            with gzip.open(view['detections'], 'rb') as stream:
                                det = pickle.load(stream)
                            with np.load(view['projection_evidence']) as ev:
                                ev = {k: ev[k].copy() for k in ev.files}
                            lookup = {int(p): i for i, p in enumerate(ev['point_index'])}
                            assert len(lookup) == len(ev['point_index'])
                            cache[camera] = det, ev, lookup, imageio.imread(view['depth'])/1000.
                        det, ev, lookup, depth = cache[camera]
                        ids = item['selected_point_indices']
                        assert len(ids) == len(set(ids)) and set(ids) <= set(lookup)
                        take = [lookup[p] for p in ids]
                        y, x = ev['pixel_y'][take], ev['pixel_x'][take]
                        assert det['mask'][index][y, x].all()
                        actual = np.asarray(item['actual_retained_world_points'], float).reshape(-1, 3)
                        assert len(actual) == item['actual_retained_points']
                        pose, k = np.asarray(view['camera_pose']), np.asarray(view['intrinsic'])
                        if item['path'] == 'sparse':
                            camera_points = (actual-pose[:3, 3]) @ pose[:3, :3]
                            projected = camera_points @ k.T
                            uv = np.rint(projected[:, :2]/projected[:, 2, None]).astype(int)
                            pixel_to_id = {(int(a), int(b)): int(i) for a, b, i in zip(ev['pixel_y'], ev['pixel_x'], ev['point_index'])}
                            expected = {pixel_to_id[int(v), int(u)] for u, v in uv}
                            assert expected == set(ids)
                            z = depth[uv[:, 1], uv[:, 0]]
                            expected_points = np.column_stack(((uv[:, 0]-k[0, 2])*z/k[0, 0], (uv[:, 1]-k[1, 2])*z/k[1, 1], z)) @ pose[:3, :3].T + pose[:3, 3]
                            assert np.max(np.abs(expected_points-actual), initial=0) < 1e-9
                            nearest = 0.
                        elif ids:
                            z = depth[y, x]
                            support = np.column_stack(((x-k[0, 2])*z/k[0, 0], (y-k[1, 2])*z/k[1, 1], z)) @ pose[:3, :3].T + pose[:3, 3]
                            nearest = float(cKDTree(support).query(actual)[0].max(initial=0))
                        else:
                            nearest = None
                        if item['point_provenance_status'] != 'exact_voxel_contributor_indices' and item['path'] == 'dense':
                            assert not ids
                            position = observations.index(item)
                            assert any(c['representative'] == position for c in audit['clusters'])
                        records.append(dict(split=split, scene=scene['scene'], frame_idx=group['frame_idx'], method=method,
                            source_key=item['source_key'], path=item['path'], status=item['point_provenance_status'],
                            selected_scan_winner_indices=len(ids), actual_voxel_or_sparse_points=len(actual),
                            maximum_voxel_to_selected_depth_support_m=nearest, raw_mask_subset_verified=True,
                            all_indices_projection_zbuffer_winners=True, sparse_exact_backprojection_verified=item['path']=='sparse'))
                    for witness in audit['pairs']:
                        if witness['accepted']:
                            a = next(o for o in observations if o['source_key'] == witness['source_key_a'])
                            b = next(o for o in observations if o['source_key'] == witness['source_key_b'])
                            ia, ib = set(a['selected_point_indices']), set(b['selected_point_indices'])
                            assert a['lidar_token'] == b['lidar_token'] == witness['lidar_token']
                            assert len(ia & ib) == witness['shared_original_points']
                            assert min(len(ia), len(ib)) == witness['smaller_selected_points']
                            accepted.append(dict(split=split, scene=scene['scene'], frame_idx=group['frame_idx'], method=method, **witness))
    summary = dict(passed=True, observations=len(records), statuses=dict(Counter(r['status'] for r in records)),
        all_recorded_original_indices_are_saved_zbuffer_winners=True, sparse_rint_projection_and_unperturbed_depth_coordinates_exact=True,
        unavailable_dense_trace_hypotheses_preserved=True, accepted_witnesses=accepted,
        dense_trace_scope='Source inspection confirms retained voxel contributors mapped to valid supported-mask row-major input pixels, then only saved zbuffer winner original point IDs. Saved artifacts independently validate mask membership, scan identity and voxel proximity; original jittered process inputs/contributor lists were not persisted, so exact dense contributor trace itself is not independently replayed.',
        historical_dense_coordinate_jitter_std_m=.004, no_new_mapping_or_model_calls=True,
        source_sha256=sources, audit_source_sha256=digest(__file__), records=records)
    for destination in [run/'audit.json', base/'outputs/reports/multiview_identity_repair_20261123/validation/L5_original_point_index_audit.json']:
        destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ['records', 'source_sha256', 'accepted_witnesses']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
