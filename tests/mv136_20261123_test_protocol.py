import copy
import pytest
from conceptgraph.completion.mv136_20261123_protocol import SIX, clock_evidence


def group():
    return dict(reference_timestamp_s=10.05,official_sample_timestamp_s=10.08,
        views={c:dict(camera_timestamp_s=10+i*.01,lidar_timestamp_s=9.99) for i,c in enumerate(SIX)})


def test_same_clock_does_not_rewrite_subset_ready():
    evidence = clock_evidence(group())
    assert evidence['subset_ready_timestamp_s']['T3_1'] == 10
    assert evidence['subset_ready_timestamp_s']['T3_3'] == 10.02
    assert evidence['subset_ready_timestamp_s']['T3_6'] == 10.05


def test_lidar_before_common_but_after_own_camera_is_rejected():
    value = copy.deepcopy(group())
    value['views']['CAM_FRONT']['lidar_timestamp_s'] = 10.01
    with pytest.raises(ValueError):
        clock_evidence(value)


def test_missing_camera_and_non_max_clock_rejected():
    value = group()
    del value['views']['CAM_BACK']
    with pytest.raises(ValueError):
        clock_evidence(value)
    value = group()
    value['reference_timestamp_s'] = 10.06
    with pytest.raises(ValueError):
        clock_evidence(value)
