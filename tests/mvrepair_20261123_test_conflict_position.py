"""关键干预边界：不得改变单前视或缺测持有位置；保留强当前dense。"""
from conceptgraph.slam.mvrepair_20261123_conflict_position import conflict_position_decision, UnifiedSettings


def front():
    return dict(_capture_timestamp_s=9.5, _dense_seen=True, confidence=.3,
        _groups={1, 2}, total_points=12, _reliable_dense=True)


def test_missing_and_front_native_measurements_are_preserved():
    settings = UnifiedSettings()
    assert not conflict_position_decision(front(), [], 10., settings)[0]
    assert not conflict_position_decision(front(), [dict(_last_camera='CAM_FRONT', _last_observation_points=2)], 10., settings)[0]


def test_strong_side_dense_and_mixed_dense_are_preserved():
    settings = UnifiedSettings()
    weak = dict(_last_camera='CAM_FRONT_RIGHT', _last_observation_points=2)
    strong = dict(_last_camera='CAM_FRONT_LEFT', _last_observation_points=12, _last_reliable_dense=True)
    assert not conflict_position_decision(front(), [weak, strong], 10., settings)[0]


def test_only_real_current_weak_side_can_use_confirmed_recent_front():
    settings = UnifiedSettings()
    weak = dict(_last_camera='CAM_FRONT_RIGHT', _last_observation_points=2)
    assert conflict_position_decision(front(), [weak], 10., settings)[0]
    assert not conflict_position_decision(front(), [weak], 11., settings)[0]
    assert not conflict_position_decision(front(), [weak], 9., settings)[0]
