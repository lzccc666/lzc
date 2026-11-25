"""当前支持不足不能吞并、强前视位置和原身份必须保留。"""
import numpy as np

from conceptgraph.slam.mviter_20261123_front_anchor import canonicalize


def entry(source, node, *, shift=0., time=1., category='car'):
    return dict(source=source, prediction=dict(node_id=node, category=category,
        center=[shift, 0., 10.], position_source='reliable_current_dense_lidar'),
        evidence=dict(points=np.array([[shift, 0., 10.], [shift+.1, 0., 10.]]),
            feature=np.array([1., 0.]), capture_timestamp_s=time, state_timestamp_s=time,
            source_camera='CAM_FRONT' if source == 'front' else 'CAM_FRONT_RIGHT', confidence=.5))


def test_shared_current_observation_keeps_original_front_center_and_identity():
    a, b = entry('front', 'a'), entry('multi', 'b', shift=.03, time=1.01)
    output, audit = canonicalize([a, b], 1.02)
    assert len(output) == 1 and output[0]['node_id'] == 'front/a'
    assert output[0]['center'] == a['prediction']['center'] and audit['removed'] == 1


def test_close_but_distinct_current_points_and_stale_support_are_preserved():
    a, b = entry('front', 'a'), entry('multi', 'b', shift=.8)
    assert len(canonicalize([a, b], 1.02)[0]) == 2
    b = entry('multi', 'b', time=.8)
    assert len(canonicalize([a, b], 1.02)[0]) == 2  # >75ms exposure difference
    b = entry('multi', 'b')
    assert len(canonicalize([a, b], 2.)[0]) == 2  # old points never treated as current


def test_two_front_nodes_crossclass_and_unmatched_side_are_never_hidden():
    a, b = entry('front', 'a'), entry('front', 'b')
    assert len(canonicalize([a, b], 1.02)[0]) == 2
    b = entry('multi', 'b', category='pedestrian')
    assert len(canonicalize([a, b], 1.02)[0]) == 2
    b = entry('multi', 'b', shift=50.)
    assert len(canonicalize([a, b], 1.02)[0]) == 2
