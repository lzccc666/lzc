from conceptgraph.completion.natural_inputs import latest_past_lidar


def test_natural_instruction_alignment_rejects_future_and_stale_returns():
    future = dict(token='future',timestamp=1100000)
    past = dict(token='past',timestamp=950000)
    assert latest_past_lidar([future,past],1000000) is past
    assert latest_past_lidar([future],1000000) is None
    assert latest_past_lidar([past],1100001) is None
    assert latest_past_lidar([future,past],1100000) is future
