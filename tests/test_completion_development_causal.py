import gzip
import hashlib
import json
import pickle

import numpy as np
import pytest

from conceptgraph.completion.development_causal_audit import independent_depth, past_selection
from conceptgraph.completion.development_causal_inputs import bind_detection, development_cameras
from conceptgraph.completion import development_causal_statistics as causal_statistics
from conceptgraph.completion import retrieval_statistics


def test_independent_projection_nearest_collision_and_missing_pixels():
    # 平移后两个点占同一像素；深度1和65m边界及图外点都必须丢弃。
    transform = np.eye(4)
    transform[0, 3] = 1
    points = np.array([[-1, 0, 8], [-1, 0, 3], [-1, 0, 1],
                       [-1, 0, 65], [100, 0, 5], [np.nan, 0, 5]], dtype=float)
    intrinsic = np.array([[10, 0, 2], [0, 10, 1], [0, 0, 1]])
    depth = independent_depth(points, transform, intrinsic, 3, 5)
    assert depth.dtype == np.uint16
    assert depth[1, 2] == 3000
    assert np.count_nonzero(depth) == 1


def test_independent_selector_rejects_future_and_old_lidar():
    candidates = [dict(token='b', timestamp=1100000), dict(token='a', timestamp=900000)]
    assert past_selection(candidates, 1000000)['token'] == 'a'
    assert past_selection(candidates, 1000001) is None
    assert past_selection(candidates, 899999) is None


@pytest.mark.parametrize('failure', ['test_split', 'camera_reordered'])
def test_development_inputs_reject_split_leakage_or_reordered_rgb(tmp_path, failure):
    scene = 'scene-unit'
    (tmp_path/'converted'/scene).mkdir(parents=True)
    split = 'test' if failure == 'test_split' else 'development'
    (tmp_path/'split_manifest.json').write_text(json.dumps(dict(scene_to_split={scene: split})))
    (tmp_path/'scene_selection_manifest.json').write_text(json.dumps(dict(scenes=[dict(scene=scene, scene_token='s')])) )
    (tmp_path/'source_file_index.json').write_text(json.dumps({scene: dict(sequence=[dict(
        sample_token='sample', sensors=dict(CAM_FRONT=dict(filename='expected.jpg')))])}))
    (tmp_path/'converted'/scene/'frame_alignment.json').write_text(json.dumps([
        dict(camera_source='different.jpg', sample_token='sample')]))
    np.savetxt(tmp_path/'converted'/scene/'poses.txt', np.eye(4).reshape(1, -1))
    with pytest.raises(ValueError, match='非开发' if failure == 'test_split' else '帧顺序'):
        development_cameras([dict(scene=scene, root=str(tmp_path), log_token='log')])


def test_reuse_rejects_detection_from_another_rgb_even_with_valid_file_sha(tmp_path):
    rgb = tmp_path/'rgb.jpg'
    rgb.write_bytes(b'camera-frame-A')
    folder = tmp_path/'frontend/inputs/scene-unit'
    detections = folder/'exp_streamlined_yolo_outdoor_objects_v1/detections'
    detections.mkdir(parents=True)
    path = detections/'000000.pkl.gz'
    with gzip.open(path, 'wb') as handle:
        pickle.dump(dict(frame_provenance=dict(rgb_sha256='another-frame',
            image_size_wh=[1600, 900], sam_prompt_tensor_isolated=True)), handle)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (folder/'frontend_complete.json').write_text(json.dumps(dict(output_sha256={str(path): sha})))
    camera = dict(scene='scene-unit', frame_idx=0, old_rgb=str(rgb), width=1600, height=900)
    with pytest.raises(ValueError, match='非当前原图'):
        bind_detection(camera, tmp_path/'frontend', tmp_path/'destination')
    assert not (tmp_path/'destination').exists()


def test_causal_statistics_keeps_missing_lidar_truth_and_unknown_error(tmp_path, monkeypatch):
    monkeypatch.setattr(causal_statistics, 'OUTPUT', tmp_path)
    inputs = tmp_path/'data_manifest/input'
    mapping = tmp_path/'mapping_evaluation/mapping'
    inputs.mkdir(parents=True)
    (mapping/'scene-unit').mkdir(parents=True)
    def write(path, value):
        path.write_text(json.dumps(value))
    write(inputs/'complete.json', dict(scenes=1, cameras=2))
    write(inputs/'release.json', dict(evaluation_time_limit='GT time proxy'))
    write(inputs/'alignment_manifest.json', dict(plans=[dict(camera=dict(scene='scene-unit', frame_idx=i,
        official_sample_timestamp=1000000+i*500000, timestamp=999000+i*500000),
        lidar=None if i == 0 else dict(token='past')) for i in range(2)]))
    write(mapping/'summary.json', dict(scene_count=1, frames=2))
    write(mapping/'release.json', dict(scenes=[dict(scene='scene-unit', log_token='log')]))
    methods = ['original_fusion', 'one_to_one_depth_quality']
    results = dict(true_positive=1, truth=2, predictions=1, duplicate_nodes=0,
        identity_switches=0, adjacent_matched_transitions=0,
        nodes_associated_with_multiple_instances=0, center_error_m_mean=.25)
    write(mapping/'scene-unit/report.json', dict(methods={m: results for m in methods}, log_token='log', frame_count=2))
    with (mapping/'scene-unit/frames.jsonl').open('w') as handle:
        for i in range(2):
            metrics = dict(true_positive=i, truth=1, predictions=i, duplicate_nodes=0,
                matches=[dict(distance_m=.25)] if i else [])
            handle.write(json.dumps(dict(frame_idx=i, metrics={m: metrics for m in methods}))+'\n')
    causal_statistics.summarize('input', 'mapping', 'stats')
    report = json.loads((tmp_path/'statistics/stats/report.json').read_text())
    result = report['methods']['original_fusion']
    assert result['all']['truth'] == 2
    assert result['all']['recall']['value'] == .5
    assert result['missing_past_lidar']['truth'] == 1
    assert result['missing_past_lidar']['recall']['value'] == 0
    assert result['missing_past_lidar']['matched_prediction_precision']['value'] is None
    assert result['missing_past_lidar']['center_error_m_mean']['value'] is None
    assert result['missing_past_lidar']['identity_switches'] is None
    assert report['actual_arrivals'] == 0


def test_retrieval_statistics_negative_only_keeps_positive_rate_unknown():
    # 首帧缺测可只有负查询；不能制造100%正准确率或写出非法NaN。
    counts = np.array([[[0, 0, 0, 1], [0, 0, 1, 1]]])
    totals, estimates, samples = retrieval_statistics.bootstrap(counts)
    assert totals[:, 1].tolist() == [0, 0]
    positive = retrieval_statistics.nullable_number(estimates[0, 0])
    assert positive is None
    assert retrieval_statistics.interval(samples[:, 0, 0]) == dict(ci95=None, valid_resamples=0)
    assert retrieval_statistics.nullable_number(estimates[0, 1]) == 0
    assert retrieval_statistics.nullable_number(estimates[1, 1]) == 1
    assert json.dumps(dict(positive=positive), allow_nan=False) == '{"positive": null}'
