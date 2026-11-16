"""检查参照/否定/静止约束的串扰和证据边界；不读取封存标签。"""
import copy
import unittest

from public_query_intake_055 import choose, parse, reference_available, strong_rule


def node(nid, category, xy, *, static=True):
    return dict(node_id=nid, category=category, center_xy=xy,
                distance_m=(xy[0]**2+xy[1]**2)**.5,
                motion_state='static' if static else 'dynamic', version=3,
                now_s=10, last_seen_s=10,
                evidence=dict(source='current_projected_lidar', version=3,
                              observations=3, lidar_pixels=8, support_fraction=.8, conflict=False))


class IntakeTests(unittest.TestCase):
    def test_anchor_category_does_not_become_target(self):
        q = parse('目标是在卡车右侧的静止行人，请接近它')
        self.assertEqual(q.targets, ('pedestrian',))
        self.assertEqual(q.anchors, ('truck',))
        self.assertEqual(q.relation, 'right_of')
        self.assertTrue(q.requires_static)

    def test_excluded_category_does_not_become_target(self):
        q = parse('本次接近观察范围内静止小汽车，而不是公交车')
        self.assertEqual(q.targets, ('car',))
        self.assertEqual(q.excluded, ('bus',))

    def test_dynamic_nearer_object_not_static_answer(self):
        nodes = [node(1, 'car', [1, 0], static=False), node(2, 'car', [5, 0])]
        self.assertEqual(choose('观察范围内若有最近的静止小汽车，请安全接近', nodes)['proposed_node_id'], 2)

    def test_unknown_not_assumed_static(self):
        n = node(1, 'car', [1, 0])
        n['motion_state'] = 'unknown'
        self.assertIsNone(choose('将车辆驶向观察范围内静止小汽车旁的合法区域', [n])['proposed_node_id'])

    def test_dynamic_reference_allowed_for_static_target(self):
        nodes = [node(1, 'truck', [5, 3], static=False), node(2, 'car', [5, 0])]
        result = strong_rule('目标是在卡车右侧的静止小汽车，请接近它', nodes, relation_radius_m=15)
        self.assertEqual(result['proposed_node_id'], 2)

    def test_reference_cannot_be_target_itself(self):
        q = parse('目标是在小汽车右侧的静止小汽车，请接近它')
        self.assertFalse(reference_available(q, [node(1, 'car', [5, 0])]))

    def test_proposal_not_control_authorization(self):
        n = node(1, 'car', [5, 0])
        n['evidence']['observations'] = 1
        result = choose('选择最近的小汽车', [n])
        self.assertEqual(result['proposed_node_id'], 1)
        self.assertIsNone(result['authorized_node_id'])

    def test_second_nearest_remains_complex(self):
        nodes = [node(1, 'car', [2, 0]), node(2, 'car', [6, 0])]
        text = '不要最近的小汽车，选择第二近的小汽车'
        self.assertEqual(strong_rule(text, nodes, relation_radius_m=12)['proposed_node_id'], 2)
        self.assertTrue(choose(text, nodes)['backend_output_used'])

    def test_unsupported_relation_abstains(self):
        q = parse('小汽车左侧最近的行人')
        self.assertFalse(q.supported)
        self.assertIsNone(choose('小汽车左侧最近的行人', [node(1, 'car', [2, 0])])['proposed_node_id'])

    def test_unknown_color_not_claimed_supported(self):
        self.assertFalse(parse('选择最近的红色小汽车').supported)

    def test_invalid_or_outside_model_id_rejected(self):
        text = '以卡车为参照，选择其右侧最近的小汽车'
        nodes = [node(1, 'car', [5, 0]), node(2, 'truck', [5, 3])]
        pred = dict(schema_valid=True, normalized_output=dict(selected_node_id=999))
        self.assertIsNone(choose(text, nodes, pred)['proposed_node_id'])

    def test_input_immutable(self):
        nodes = [node(1, 'car', [5, 0]), node(2, 'truck', [5, 3])]
        original = copy.deepcopy(nodes)
        strong_rule('目标是在卡车右侧的静止小汽车，请接近它', nodes, relation_radius_m=15)
        self.assertEqual(nodes, original)


if __name__ == '__main__':
    unittest.main()
