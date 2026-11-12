import numpy as np

from conceptgraph.completion.reserved_annotation import (
    derive_query, make_scene_queries, scoped_objects,
)


def obj(token, category, center, motion='static'):
    return dict(instance_token=token, category=category, center=center,
        official_motion=motion, visibility_token='4')


def test_dynamic_nearest_never_becomes_static_navigation_truth():
    frame = dict(scene='scene-0001', frame_idx=0, sample_token='sample', camera_timestamp_s=1,
        objects=[obj('moving', 'car', [2, 0, 0], 'dynamic'),
                 obj('parked', 'car', [10, 0, 0]), obj('unknown', 'car', [3, 0, 0], 'unknown')])
    row = derive_query(frame, np.eye(4), category='car', nearest=True,
        family=0, query_id='q')
    assert row['accepted_instance_tokens'] == ['parked']
    assert row['reachable_label'] is None


def test_relation_uses_driver_forward_not_world_axes_and_allows_moving_anchor():
    # 相机+Z在世界+X，驾驶员左方在世界+Y。
    pose = np.eye(4)
    pose[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    frame = dict(scene='scene-0001', frame_idx=0, sample_token='sample', camera_timestamp_s=1,
        objects=[obj('left', 'car', [10, 3, 0]), obj('right', 'car', [10, -3, 0]),
                 obj('anchor', 'truck', [10, 0, 0], 'dynamic')])
    row = derive_query(frame, pose, category='car', nearest=False,
        family=3, query_id='q', anchor='truck', relation='left_of')
    assert row['accepted_instance_tokens'] == ['left']


def test_official_scope_excludes_behind_camera_and_low_visibility():
    rows = [obj('front', 'car', [0, 0, 10]), obj('behind', 'car', [0, 0, -10]),
            dict(obj('hidden', 'car', [0, 0, 10]), visibility_token='1')]
    result = scoped_objects(rows, np.eye(4), np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]]), 100, 100)
    assert [r['instance_token'] for r in result] == ['front']


def test_absent_scene_retains_twenty_queries_and_six_replays_without_inventing_positives():
    frames = [dict(scene='scene-0001', frame_idx=i, sample_token=str(i),
        camera_timestamp_s=i / 2, objects=[]) for i in range(40)]
    main, challenge = make_scene_queries(frames, np.repeat(np.eye(4)[None], 40, axis=0))
    assert len(main) == 20
    assert sum(q['online_replay_selected'] for q in main) == 6
    assert all(q['no_matching_target'] for q in main + challenge)
    assert all(q['challenge_type'] != 'positive_relation' for q in challenge)
    assert len({(q['query_text'], q['frame_idx']) for q in main}) == 20
