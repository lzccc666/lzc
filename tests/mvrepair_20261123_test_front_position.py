"""前视辅助不能绕过确认/时效，也不能覆写可靠当前稠密位置。"""
from conceptgraph.slam.mvrepair_20261123_front_position import UnifiedSettings, position_decision


def front(**changes):
    return dict(_capture_timestamp_s=1., _groups={0, 1}, total_points=4, _dense_seen=False,
                confidence=.3, _reliable_dense=False, **changes)


def test_confirmed_recent_front_protects_from_only_sparse_side():
    current = [dict(_last_camera='CAM_FRONT_RIGHT', _last_observation_points=4, _last_reliable_dense=False)]
    assert position_decision(front(), current, 1.45, UnifiedSettings())[0]


def test_reliable_current_dense_kept_until_reliable_current_front_exists():
    current = [dict(_last_camera='CAM_FRONT_RIGHT', _last_observation_points=20, _last_reliable_dense=True)]
    assert not position_decision(front(), current, 1.45, UnifiedSettings())[0]
    f = front()
    f['_reliable_dense'] = True
    assert position_decision(f, current, 1.01, UnifiedSettings())[0]


def test_single_group_or_stale_front_is_not_output_evidence():
    f = front()
    f['_groups'], f['total_points'] = {0}, 2
    assert not position_decision(f, [], 1.01, UnifiedSettings())[0]
    assert not position_decision(front(), [], 1.8, UnifiedSettings())[0]
