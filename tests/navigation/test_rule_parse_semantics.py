from conceptgraph.navigation.semantic import parse_navigation_query


def test_chinese_anchor_relation_and_static_are_not_reversed_or_replaced_by_near():
    query = parse_navigation_query('去公交车左侧停着的红色小汽车旁边', 8)
    assert query.target_categories == ['car']
    assert query.target_colors == ['red']
    assert '公交车' in query.anchor_text
    assert query.relation == 'left_of' and query.requires_static


def test_negated_object_does_not_enter_positive_target_or_color():
    query = parse_navigation_query('不要红色卡车，去最近的蓝色自行车附近', 8)
    assert query.target_categories == ['bicycle']
    assert query.target_colors == ['blue']
    assert query.excluded_categories == ['truck']
    assert query.excluded_colors == ['red']
    assert query.prefer_nearest and query.relation is None


def test_single_object_approach_and_front_preference_do_not_create_anchor():
    query = parse_navigation_query('靠近前方的小汽车', 8)
    assert query.relation is None and query.anchor_text is None
    assert query.prefer_front


def test_target_first_relation_and_anchor_front_do_not_mean_ego_front():
    query = parse_navigation_query('小汽车在卡车前方', 8)
    assert query.target_categories == ['car']
    assert query.anchor_text == '卡车' and query.relation == 'in_front_of'
    assert not query.prefer_front


def test_english_relation_keeps_target_first_order():
    query = parse_navigation_query('car left of truck', 8)
    assert query.target_categories == ['car'] and query.anchor_text == 'truck'
    assert query.relation == 'left_of'
