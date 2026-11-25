"""核对附指标的方向、日志配对和固定主检验兼容。"""
import copy

import numpy as np
import pytest

from conceptgraph.completion.metric_acceptance import paired_metric_uncertainty


def rows():
    return [dict(scene=f's{i}', log=f'l{i}', frames=10, result=dict(counts=dict(
        true_positive=40, truth=100, predictions=80, IDTP=35, IDFP=45, IDFN=65,
        duplicate_nodes=4, center_error_sum_m=40.))) for i in range(11)]


def test_secondary_units_and_micro_average_keep_primary_intervals():
    base = rows()
    new = copy.deepcopy(base)
    for row in new:
        row['result']['counts'].update(true_positive=50, duplicate_nodes=2, center_error_sum_m=25.)
    result = paired_metric_uncertainty(base, new)
    expected = dict(recall=10., precision=12.5, F1=100 * 20 / 180,
                    IDF1=0., duplicate_nodes=-22., center_error_m=-.5)
    for key, value in expected.items():
        np.testing.assert_allclose(result['intervals'][key]['ci_bonferroni'], [value, value])
    assert result['primary_exactly_preserved']


def test_secondary_rejects_duplicate_scene_or_changed_denominator():
    base, candidate = rows(), rows()
    candidate[0]['result']['counts']['truth'] += 1
    with pytest.raises(ValueError):
        paired_metric_uncertainty(base, candidate)
    with pytest.raises(ValueError):
        paired_metric_uncertainty(base + [base[0]], rows())
