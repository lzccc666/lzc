"""缺参照物不能盲调模型、误报无匹配或沿旧目标路线驾驶。"""
import copy
from types import SimpleNamespace
import unittest

from carla_reference_readiness_047 import ReferenceReadyLaneBridge, reference_readiness
from conceptgraph.navigation.vehicle_geometry import VehicleState
from test_carla_real_graph_bridge_002 import agent

QUERY = '以卡车为参照，选择本车道内后方的小汽车'


def observed_node(identity, category):
    return dict(node_id=identity, category=category, motion_state='static',
        now_s=.6, last_seen_s=.6, version=4,
        evidence=dict(source='current_projected_lidar', version=4, observations=4,
            lidar_pixels=10, support_fraction=.8, conflict=False))


class ReferenceReadinessTests(unittest.TestCase):
    def test_missing_reference_is_unknown_not_proven_absence(self):
        r = reference_readiness(QUERY, {'candidate_nodes': [observed_node(100, 'car')]})
        self.assertFalse(r['ready'])
        self.assertFalse(r['absence_proven'])
        self.assertEqual(r['reason'], 'reference_not_in_observed_input')

    def test_supported_reference_and_distinct_target_can_proceed(self):
        nodes = [observed_node(100, 'car'), observed_node(101, 'truck')]
        self.assertTrue(reference_readiness(QUERY, {'candidate_nodes': nodes})['ready'])

    def test_expired_or_conflicting_reference_cannot_authorize(self):
        for change in ['age', 'conflict', 'depth', 'motion']:
            nodes = [observed_node(100, 'car'), observed_node(101, 'truck')]
            ref = nodes[1]
            if change == 'age':
                ref['last_seen_s'] = -2.
            elif change == 'conflict':
                ref['evidence']['conflict'] = True
            elif change == 'depth':
                ref['evidence']['lidar_pixels'] = 0
            else:
                ref['motion_state'] = 'unknown'
            with self.subTest(change=change):
                self.assertFalse(reference_readiness(QUERY, {'candidate_nodes': nodes})['ready'])

    def test_one_object_cannot_be_both_target_and_its_reference(self):
        r = reference_readiness('以小汽车为参照，选择后方的小汽车',
            {'candidate_nodes': [observed_node(100, 'car')]})
        self.assertFalse(r['ready'])

    def test_clear_query_does_not_change_existing_selection_policy(self):
        self.assertTrue(reference_readiness('靠近最近的小汽车', {'candidate_nodes': []})['ready'])

    def test_all_three_modes_refuse_missing_reference_without_call_or_plan(self):
        for mode in ['rule', 'base', 'adapter']:
            backend = SimpleNamespace(calls=0)
            b = ReferenceReadyLaneBridge(QUERY, mode,
                backend=None if mode == 'rule' else backend,
                planner=lambda *a: self.fail('missing reference must not invoke target planner'))
            try:
                a = agent()
                a.path_purpose = 'target_approach'
                a.semantic_bridge = b
                final, plans, audit = b.select(a, [], [], VehicleState(0., 0., 0., time_s=.6), 3)
                self.assertEqual(final, [])
                self.assertEqual(plans, {})
                self.assertEqual(a.path, [])
                self.assertEqual(backend.calls, 0)
                self.assertIsNone(b.pending)
                self.assertFalse(audit['reference_readiness']['absence_proven'])
            finally:
                b.close()

    def test_missing_reference_preserves_independently_checked_search_route(self):
        b = ReferenceReadyLaneBridge(QUERY, 'rule')
        try:
            a = agent()
            a.path_purpose = 'bounded_lane_search'
            old = copy.deepcopy(a.path)
            b.select(a, [], [], VehicleState(0., 0., 0., time_s=.6), 3)
            self.assertEqual(a.path, old)
            self.assertFalse(b.hold_for_pending_semantics)
        finally:
            b.close()

    def test_clear_query_uses_unchanged_rule_selection_and_planning(self):
        state = VehicleState(0., 0., 0., time_s=.6)
        plan = SimpleNamespace(states=[state], failure=None)
        b = ReferenceReadyLaneBridge('靠近最近的小汽车', 'rule',
            planner=lambda *args: (plan, {'passed': True}))
        try:
            a = agent()
            a.path_purpose = 'target_approach'
            final, plans, audit = b.select(a, [], [], state, 3)
            self.assertEqual(final[0].node_id, 0)
            self.assertIs(plans[0], plan)
            self.assertTrue(audit['reference_readiness']['ready'])
        finally:
            b.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
