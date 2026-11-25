"""验证完整范围和预登记护栏，避免部分实验进入效果结论。"""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def summary():
    path = Path(__file__).resolve().parents[1] / 'scripts/summarize_association_refinement.py'
    spec = importlib.util.spec_from_file_location('association_summary', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complete_scope():
    report = dict(condition='R1', full_validation_scope=True, scenes=38, logs=11,
                  frames=1508, result=dict(counts=dict(truth=5723)))
    release = dict(threshold_m=3., seed=17, camera_shape_wh=[1600, 900], pilot=False,
                   ground_truth_in_inference=False, test_accessed=False,
                   scenes=[dict(scene=f's{i}', log_token=f'l{i % 11}') for i in range(38)],
                   config={}, inputs_sha256={})
    return report, release


def registration(summary):
    return dict(registered_comparisons=summary.COMPARISONS, primary_metric='IDF1',
                bootstrap_seed=20261123, bootstrap_replicates=10000,
                ground_truth_in_inference=False, test_accessed=False,
                baseline_run=summary.DEFAULT_RUNS['R1'])


def test_rejects_partial_scope_changed_denominator_or_duplicate_scenes(summary):
    report, release = complete_scope()
    assert len(summary.validate_scope(report, release, 'R1')) == 38
    partial = copy.deepcopy(report)
    partial['frames'] = 1507
    with pytest.raises(ValueError):
        summary.validate_scope(partial, release, 'R1')
    changed = copy.deepcopy(report)
    changed['result']['counts']['truth'] = 5722
    with pytest.raises(ValueError):
        summary.validate_scope(changed, release, 'R1')
    duplicate = copy.deepcopy(release)
    duplicate['scenes'][-1] = duplicate['scenes'][0]
    with pytest.raises(ValueError):
        summary.validate_scope(report, duplicate, 'R1')


def test_rejects_changed_primary_or_comparison_family(summary):
    manifest = registration(summary)
    summary.validate_manifest(manifest)
    changed = copy.deepcopy(manifest)
    changed['primary_metric'] = 'recall'
    with pytest.raises(ValueError):
        summary.validate_manifest(changed)
    changed = copy.deepcopy(manifest)
    changed['registered_comparisons'] = changed['registered_comparisons'][:1]
    with pytest.raises(ValueError):
        summary.validate_manifest(changed)


def test_missing_candidate_refuses_statistics_and_creates_no_report(summary, tmp_path, monkeypatch):
    root = tmp_path / 'runs'
    baseline = root / summary.DEFAULT_RUNS['R1']
    baseline.mkdir(parents=True)
    report, release = complete_scope()
    for name, value in [('report.json', report), ('release.json', release),
                        ('input_fingerprints.json', {f'input{i}': 'sha' for i in range(4600)})]:
        (baseline / name).write_text(json.dumps(value))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(registration(summary)))
    output = tmp_path / 'must_not_exist'
    monkeypatch.setattr(summary, 'RUN_ROOT', root)
    monkeypatch.setattr(summary.sys, 'argv', ['summary', '--manifest', str(manifest),
                                            '--output', str(output)])
    monkeypatch.setattr(summary, 'paired_metric_uncertainty',
                        lambda *a, **k: pytest.fail('不完整实验不能进入统计'))
    with pytest.raises(ValueError, match='R3尚未完成全范围'):
        summary.main()
    assert not output.exists()


def test_output_diagnostic_keeps_unmatched_and_unknown_measurement_fields(summary):
    method = 'R1'
    row = dict(frame_idx=5, camera_timestamp_s=10., evaluation_only_official=[
        dict(category='car', center=[0., 0., 0.], instance_token='gt')],
        predictions={method: [
            dict(node_id='a', category='car', center=[0., 0., 0.], num_detections=3,
                 last_seen_frame=5, prediction_age_s=0),
            dict(node_id='b', category='car', center=[1., 0., 0.], num_detections=1,
                 last_seen_frame=4, prediction_age_s=.5, timestamp_s=7., motion_state='unknown'),
            dict(node_id='sparse', category='truck', center=[20., 0., 0.], prediction_age_s=1.)]},
        metrics={method: dict(matches=[dict(prediction_index=0, truth_index=0)])})
    result = summary.describe_predictions({'scene': [row]}, method)
    assert result['output_observations'] == 3
    assert result['by_output_status'] == dict(matched=1, unmatched_duplicate_within_3m=1,
                                             unmatched_outside_same_class_gates=1)
    duplicate = result['by_status_and_field']['unmatched_duplicate_within_3m']
    assert duplicate['measurement_recency'] == {'older_measurement': 1}
    assert duplicate['detection_observations'] == {'one': 1}
    unknown = result['by_status_and_field']['unmatched_outside_same_class_gates']
    assert unknown['measurement_recency'] == {'unknown_last_seen_frame': 1}
    assert result['single_unknown_timestamp_over_2s_by_source_and_status'] == {
        'dense_pointcloud_backed': {'unmatched_duplicate_within_3m': 1}}
