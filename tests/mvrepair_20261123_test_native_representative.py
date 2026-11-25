from conceptgraph.slam.mvrepair_20261123_native_representative import representative_rank

def test_native_continuity_precedes_more_partial_points():
    a = dict(reliable=True, points=15, confidence=.3)
    b = dict(reliable=True, points=21, confidence=.4)
    assert representative_rank(a, .80, 0) > representative_rank(b, .52, 1)

def test_weak_cannot_displace_current_reliable():
    a = dict(reliable=True, points=8, confidence=.2)
    b = dict(reliable=False, points=7, confidence=.9)
    assert representative_rank(a, .4, 0) > representative_rank(b, .9, 1)
