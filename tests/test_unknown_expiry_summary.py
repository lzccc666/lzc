"""追加修复的比较家族、选择偏差声明与完整五臂护栏。"""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def summary():
    path = Path(__file__).resolve().parents[1]/'scripts/summarize_unknown_expiry_refinement.py'
    spec = importlib.util.spec_from_file_location('unknown_summary', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan(summary):
    return dict(registered_comparisons=summary.FAMILY, primary_metric='IDF1',
        bootstrap_seed=20261123, bootstrap_replicates=10000, adaptive_selection=True,
        comparisons_registered=7, ttl_s=2., ttl_parameter_changed=False,
        confirmation_changed=False, default_changed=False, threshold_m=3.,
        test_accessed=False, ground_truth_in_inference=False)


def test_expanded_family_widens_corrected_interval_without_changing_raw_interval(summary):
    base = []
    for i in range(11):
        identity = 30+i
        base.append(dict(scene=f's{i}', log=f'l{i}', frames=10, result=dict(counts=dict(
            true_positive=40, truth=100, predictions=100, IDTP=identity,
            IDFP=100-identity, IDFN=100-identity, duplicate_nodes=2, center_error_sum_m=40.))))
    new = copy.deepcopy(base)
    for i, row in enumerate(new):
        c = row['result']['counts']
        c.update(IDTP=c['IDTP']+2*i, IDFP=c['IDFP']-2*i, IDFN=c['IDFN']-2*i)
    old = summary.paired_metric_uncertainty(base, new, seed=20261123, replicates=10000)
    actual = summary.family_intervals(base, new)
    old_ci = old['intervals']['IDF1']['ci_bonferroni']
    new_ci = actual['intervals']['IDF1']['ci_bonferroni7']
    assert new_ci[0] < old_ci[0] and new_ci[1] > old_ci[1]
    np.testing.assert_allclose(actual['intervals']['IDF1']['ci95'], old['intervals']['IDF1']['ci95'])
    assert actual['comparisons_in_family'] == 7
    assert actual['adaptive_selection']


def test_plan_cannot_hide_adaptive_selection_or_change_original_ttl(summary):
    original = plan(summary)
    summary.validate_plan(original)
    for key, value in [('adaptive_selection', False), ('ttl_s', 3.), ('comparisons_registered', 4)]:
        invalid = copy.deepcopy(original)
        invalid[key] = value
        with pytest.raises(ValueError):
            summary.validate_plan(invalid)


def test_missing_fifth_arm_prevents_statistics_and_report_creation(summary, tmp_path, monkeypatch):
    root = tmp_path/'runs'
    scope = [dict(scene=f's{i}', log_token=f'l{i % 11}') for i in range(38)]
    for method, run in list(summary.RUNS.items())[:4]:
        folder = root/run
        folder.mkdir(parents=True)
        report = dict(condition=method, full_validation_scope=True, scenes=38,
            logs=11, frames=1508, result=dict(counts=dict(truth=5723)))
        release = dict(scenes=scope, threshold_m=3., seed=17, camera_shape_wh=[1600, 900],
            pilot=False, ground_truth_in_inference=False, test_accessed=False, config={}, inputs_sha256={})
        if method.endswith('_unknown_ttl'):
            release.update(comparisons_registered=7, bootstrap_seed=20261123,
                paired_bootstrap_replicates=10000, primary_metric='IDF1',
                comparison_family='association_with_adaptive_unknown_expiry_20261123',
                adaptive_after_R3_failure=True,
                algorithm_settings=dict(unknown_potential_dynamic_expiry=True, original_dynamic_ttl_s=2.))
        for filename, value in [('report.json', report), ('release.json', release),
                ('input_fingerprints.json', {f'i{i}': 'sha' for i in range(4600)})]:
            (folder/filename).write_text(json.dumps(value))
    registration = tmp_path/'plan.json'
    registration.write_text(json.dumps(plan(summary)))
    output = tmp_path/'no_report'
    monkeypatch.setattr(summary, 'RUN_ROOT', root)
    monkeypatch.setattr(summary.sys, 'argv', ['summary', '--plan', str(registration), '--output', str(output)])
    monkeypatch.setattr(summary, 'family_intervals', lambda *a, **k: pytest.fail('五臂不完整不应进入统计'))
    with pytest.raises(ValueError, match='R1_unknown_ttl尚未完整结束'):
        summary.main()
    assert not output.exists()
