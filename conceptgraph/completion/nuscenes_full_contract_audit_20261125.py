"""Independent CPU replay of the frozen 20-group full-data adapter check."""
from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    from PIL import Image
    import torch
    from omegaconf import OmegaConf
    from conceptgraph.completion import nuscenes_full_20261125_run as adapter
    from conceptgraph.completion.stage9 import predict
    from conceptgraph.completion.vehicle_frontend import CLASSES
    from conceptgraph.scenegraph.node_motion_unified_20261125 import UnifiedR1Mapper, UnifiedS2Mapper
    from conceptgraph.scenegraph.relation_nav_unified_20261125 import full_current
    from conceptgraph.scenegraph.relations import RelationNode
    from conceptgraph.slam.streamlined_mapping import StreamlinedMapper
    from scripts.multiview_20261123_run_graph import load_frame
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    started = time.monotonic()
    report = ROOT / 'outputs/reports/nuscenes_full_20261125'
    summary = json.loads((report / 'CPU_contract_summary_v3.json').read_text())
    assert sha(adapter.__file__) == summary['main_draft_sha256']
    assert sha(ROOT / 'scripts/nuscenes_full_20261125_preflight_v3.py') == summary['source_sha256']
    assert sha(report / 'CPU_contract_frames_v3.jsonl.gz') == summary['frames_sha256']
    prereg = json.loads((adapter.DATA / 'cpu_contract_preregistration_v3.json').read_text())
    preseal = json.loads((adapter.DATA / 'phaseB_source_preseal_v1.json').read_text())
    for path, expected_hash in preseal['source_sha256'].items():
        assert sha(path) == expected_hash
    inputs = ROOT / 'dataset/processed/mv136_20261123_sixscene_v1/manifest.json'
    assert sha(inputs) == prereg['input_manifest_sha256']
    scene = json.loads(inputs.read_text())['scenes'][0]
    with gzip.open(report / 'CPU_contract_frames_v3.jsonl.gz', 'rt') as stream:
        observed = [json.loads(line) for line in stream]
    assert len(observed) == summary['frames'] == 20
    config, _, _ = adapter.configs()
    original = StreamlinedMapper(OmegaConf.create(config['original_fusion']), CLASSES)
    references = {'R1': UnifiedR1Mapper(OmegaConf.create(config['R1']), CLASSES)}
    references.update({name: UnifiedS2Mapper(OmegaConf.create(config['S2']), CLASSES)
        for name in ('S2_1', 'S2_3', 'S2_6')})
    counts = Counter()
    for group, saved in zip(scene['groups'][:20], observed):
        index = group['frame_idx']
        stamp = group['reference_timestamp_s']
        assert index == saved['frame_idx']
        frames = {camera: load_frame(group, camera) for camera in adapter.CHANNELS}
        front = frames['CAM_FRONT']
        projections = {}
        for camera in adapter.CHANNELS:
            with np.load(group['views'][camera]['projection_evidence']) as stream:
                projections[camera] = {key: stream[key].copy() for key in stream.files}
            projections[camera]['lidar_token'] = group['views'][camera]['lidar_token']
        np.random.seed(17 + index)
        original.update(copy.deepcopy(front))
        expected = {'original_fusion': predict(original.objects, CLASSES, pose=front.camera_pose, intrinsic=front.cam_K),
            'single_frame': predict(original.last_observations, CLASSES, pose=front.camera_pose, intrinsic=front.cam_K)}
        for row in expected['single_frame']:
            row['node_id'] = f"frame{index}:{row['node_id']}"
        for name, reference in references.items():
            reference.begin_evidence(scene['scene'], index)
            np.random.seed(17 + index)
            if name == 'R1':
                reference.capture_times = {'CAM_FRONT': group['views']['CAM_FRONT']['camera_timestamp_s']}
                reference.update(copy.deepcopy(front))
                result = reference.position_predictions(predict(reference.confirmed_objects(), CLASSES), front)
            else:
                cameras = adapter.CHANNELS[:int(name.split('_')[-1])]
                result = reference.update_group([(camera, copy.deepcopy(frames[camera])) for camera in cameras],
                    {camera: group['views'][camera]['camera_timestamp_s'] for camera in cameras},
                    {camera: projections[camera] for camera in cameras})
            expected[name] = adapter.clean(result)
            counts['independent_causal_mapper_comparisons'] += 1
        assert set(saved['methods']) == set(adapter.METHODS)
        for name, value in saved['methods'].items():
            assert adapter.clean(expected[name]) == value['predictions'], (index, name)
            counts['independent_all_method_prediction_comparisons'] += 1
            assert len(value['nodes']) == len(value['predictions'])
            counts['nonempty_method_frames'] += bool(value['nodes'])
            relation_nodes = {}
            registry = {source: int(local) for local, source in value['relation_node_registry'].items()}
            for prediction, node in zip(value['predictions'], value['nodes']):
                assert node['node_id'] == prediction['node_id']
                assert node['center_world_m'] == prediction['center']
                assert node['estimated_motion_state'] == node['legacy_motion_state']
                assert node['verified_motion_state'] == node['motion_state']
                assert node['vehicle_safety_authorized'] is False and node['body_heading_available'] is False
                local = registry[node['node_id']]
                relation_nodes[local] = RelationNode(local, prediction['center'], dynamic=node['motion_state'] == 'dynamic',
                    observation_count=prediction['num_detections'], last_seen_frame=prediction.get('last_seen_frame'),
                    metadata=dict(timestamp_s=prediction.get('timestamp_s'), motion_state=node['motion_state'],
                        shape_available=node['shape_available'], source_node_id=node['node_id']))
            edges = full_current(relation_nodes, frame_index=index, heading=front.camera_pose[:2, 2], timestamp_s=stamp).edges
            normalize = lambda rows: sorted(json.dumps(row, sort_keys=True) for row in rows)
            assert normalize(edge.to_dict() for edge in edges) == normalize(value['edges'])
            counts['full_current_graph_payload_comparisons'] += 1
            counts['edges_checked'] += len(edges)
            for raw in value['raw_observations']:
                camera, capture, _ = raw['source_key']
                view = group['views'][camera]
                assert capture == view['camera_timestamp_s'] <= stamp
                assert raw['input_sensor_tokens']['camera_token'] == view['camera_token']
                assert raw['input_sensor_tokens']['lidar_token'] == view['lidar_token']
                pixels = np.asarray(raw['pixels_flat'], dtype=int)
                depth_mm = np.asarray(raw['depth_mm'], dtype=int)
                assert len(pixels) == len(depth_mm) == raw['current_points']
                with Image.open(view['depth']) as image:
                    depth = np.asarray(image)
                assert np.array_equal(depth.ravel()[pixels], depth_mm)
                mask = np.asarray(frames[camera].detections['mask'][raw['original_detection_index']], dtype=bool)
                assert mask.shape == depth.shape and mask.ravel()[pixels].all()
                if len(pixels):
                    height, width = depth.shape
                    v, u = np.divmod(pixels, width)
                    z = depth_mm / 1000.
                    intrinsic = np.asarray(view['intrinsic'])
                    points = np.column_stack(((u - intrinsic[0, 2]) * z / intrinsic[0, 0],
                        (v - intrinsic[1, 2]) * z / intrinsic[1, 1], z))
                    pose = np.asarray(view['camera_pose'])
                    measured = np.median(points @ pose[:3, :3].T + pose[:3, 3], axis=0)
                    assert np.allclose(measured, raw['raw_center_world_m'], rtol=0., atol=1e-8)
                else:
                    assert raw['raw_center_world_m'] is None
                counts['current_raw_sources_verified'] += 1
                counts['actual_selected_pixels_verified'] += len(pixels)
    assert counts['independent_causal_mapper_comparisons'] == 80
    assert counts['independent_all_method_prediction_comparisons'] == 120
    assert not torch.cuda.is_initialized()
    for path, expected_hash in preseal['source_sha256'].items():
        assert sha(path) == expected_hash
    result = dict(passed=True, counts=dict(counts), no_GPU=True, GT_in_method=False,
        elapsed_s=time.monotonic() - started, audit_source_sha256=sha(__file__),
        input_summary_sha256=sha(report / 'CPU_contract_summary_v3.json'),
        presealed_sources_unchanged=len(preseal['source_sha256']),
        scope='20 already cached real groups; independent original mapper replay, original-mask pixels, full/cache relation payload, six output adapters',
        not_full850_results=True)
    target = ROOT / 'outputs/reports/local_scope_audit_20261125/nuscenes_CPU_contract_independent_v1.json'
    with target.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
