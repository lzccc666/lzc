import copy
import json

import numpy as np
import pytest

from conceptgraph.completion.camera_time_evaluation import interpolate_center, write_statistics


def evidence():
    obj = dict(annotation_token='current', instance_token='instance-a', center=[5., 2., 1.],
        size_wlh=[2., 4., 1.5], rotation_wxyz=[1., 0., 0., 0.])
    current = dict(token='current', instance_token='instance-a', sample_token='later',
        translation=[5., 2., 1.], prev='previous')
    previous = dict(token='previous', instance_token='instance-a', sample_token='earlier',
        translation=[0., 2., 1.])
    samples = dict(earlier=dict(timestamp=1000000, scene_token='scene-a'),
        later=dict(timestamp=1500000, scene_token='scene-a'))
    return obj, current, previous, samples


def test_center_interpolation_is_evaluation_only_and_preserves_original_box():
    obj, current, previous, samples = evidence()
    original = copy.deepcopy(obj)
    aligned, detail = interpolate_center(obj, current, previous, samples, 1465000)
    assert np.allclose(aligned['center'], [4.65, 2., 1.])
    assert obj == original
    assert aligned['size_wlh'] == obj['size_wlh']
    assert aligned['rotation_wxyz'] == obj['rotation_wxyz']
    assert detail['state'] == 'interpolated' and detail['evaluation_only']
    assert detail['interpolation_weight'] == .93
    assert np.isclose(detail['center_shift_m'], .35)


@pytest.mark.parametrize('camera_us', [900000, 1600000])
def test_unbracketed_reference_never_extrapolates_or_drops_object(camera_us):
    obj, current, previous, samples = evidence()
    aligned, detail = interpolate_center(obj, current, previous, samples, camera_us)
    assert aligned == obj
    assert detail['state'] == 'fallback'
    assert detail['reason'] == 'camera_not_bracketed_no_extrapolation'


def test_missing_previous_keeps_denominator_proxy_and_marks_fallback():
    obj, current, _, samples = evidence()
    aligned, detail = interpolate_center(obj, current, None, samples, 1465000)
    assert aligned == obj and detail['state'] == 'fallback'
    assert detail['reason'] == 'no_previous_annotation'


def test_exact_sample_does_not_require_previous_annotation():
    obj, current, _, samples = evidence()
    aligned, detail = interpolate_center(obj, current, None, samples, 1500000)
    assert aligned == obj and detail['state'] == 'exact'


@pytest.mark.parametrize('corruption', ['instance', 'previous_token', 'scene'])
def test_reference_cannot_interpolate_another_instance_or_scene(corruption):
    obj, current, previous, samples = evidence()
    if corruption == 'instance':
        previous['instance_token'] = 'other-instance'
    elif corruption == 'previous_token':
        previous['token'] = 'not-the-adjacent-annotation'
    else:
        samples['earlier']['scene_token'] = 'other-scene'
    with pytest.raises(ValueError):
        interpolate_center(obj, current, previous, samples, 1465000)


def test_gap_budget_and_nonincreasing_time_are_not_silently_accepted():
    obj, current, previous, samples = evidence()
    samples['later']['timestamp'] = 2100000
    aligned, detail = interpolate_center(obj, current, previous, samples, 2065000)
    assert aligned == obj and detail['reason'] == 'annotation_gap_exceeds_budget'
    samples['later']['timestamp'] = 1000000
    with pytest.raises(ValueError, match='不递增'):
        interpolate_center(obj, current, previous, samples, 999000)


def test_empty_GT_statistics_remain_null_and_never_get_safety_credit(tmp_path):
    methods = ['original_fusion', 'one_to_one_depth_quality', 'one_to_one_depth_robust_motion']
    result = write_statistics(tmp_path, np.zeros((1, 2, 3, 7)),
        np.zeros((1, 2, 3, 6, 3)), ['synthetic-unit-log'], methods)
    for conditions in result['conditions'].values():
        for metrics in conditions.values():
            assert metrics['recall']['value'] is None
            assert metrics['center_error_m_mean']['value'] is None
            assert metrics['recall']['valid_resamples'] == 0
    assert result['cohorts']['official_sample_proxy']['original_fusion']['fallback']['truth'] == 0
    json.dumps(result, allow_nan=False)
