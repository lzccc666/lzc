"""身份持续锁定仍须每帧有证据和安全路线；全部为注入后端软件测试。"""
from types import SimpleNamespace
import unittest

from conceptgraph.navigation.vehicle_geometry import VehicleState
from carla_persistent_intent_bridge_003 import (
    PersistentIntentBridge, hold_search_until_semantic_delivery)
import test_carla_real_graph_bridge_002 as previous


class PersistentIntentTests(previous.BridgeTests):
    def make(self, mode='adapter', backend=None, planner=None, **kwargs):
        if planner is None:
            planner = lambda a, i, s: (SimpleNamespace(states=[s], failure=None), {'passed': True})
        bridge = PersistentIntentBridge(previous.QUERY, mode, backend=backend,
            planner=planner, **kwargs)
        self.bridges.append(bridge)
        return bridge

    def committed(self, mode='rule', **kwargs):
        backend = previous.Backend() if mode != 'rule' else None
        bridge = self.make(mode=mode, backend=backend, **kwargs)
        a = previous.agent()
        a.machine = SimpleNamespace(locked_target_id=None)
        final, _, _ = self.select(bridge, a)
        if backend:
            previous.delivered(bridge)
            final, _, _ = self.select(bridge, a)
        self.assertEqual(final[0].node_id, 1)
        self.assertIsNone(bridge.intent)
        # 模拟真实observe在select之后确认锁定，不能仅凭提议跳过锁定。
        a.machine.locked_target_id = 1
        a.selected = 1
        final, _, audit = self.select(bridge, a)
        self.assertTrue(audit['intent_committed'])
        return bridge, a

    def test_proposal_without_machine_lock_cannot_commit_intent(self):
        bridge = self.make(mode='rule')
        a = previous.agent()
        a.machine = SimpleNamespace(locked_target_id=None)
        for _ in range(3):
            self.assertTrue(self.select(bridge, a)[0])
            self.assertIsNone(bridge.intent)

    def test_model_intent_survives_ego_motion_and_old_response_expiry(self):
        wall = [0.]
        bridge, a = self.committed(mode='adapter', clock=lambda: wall[0], ttl_s=1.)
        request = bridge.intent['backend_request_id']
        wall[0] = 5.
        self.state = VehicleState(3., 0., .2, time_s=.8)
        final, _, audit = self.select(bridge, a)
        self.assertEqual(final[0].node_id, 1)
        self.assertEqual(bridge.backend.calls, 1)
        self.assertFalse(audit['background_result_used'])
        self.assertEqual(audit['intent_origin_request_id'], request)
        self.assertTrue(audit['decision']['original_intent_from_model'])
        self.assertTrue(audit['model_result_delivery_deadline_unchanged'])

    def test_rule_keeps_committed_identity_when_distance_order_changes(self):
        bridge, a = self.committed()
        a.nodes[0].bbox_center[0] = 25.
        final, _, audit = self.select(bridge, a)
        self.assertEqual(final[0].node_id, 1)
        self.assertEqual(audit['decision']['source'], 'persistent_observed_identity')
        self.assertFalse(audit['decision']['original_intent_from_model'])

    def test_recycled_numeric_id_revokes_committed_identity(self):
        bridge, a = self.committed()
        a.nodes[1].metadata['object_identity'] = 'track-replacement'
        _, _, audit = self.select(bridge, a)
        self.assertIsNone(bridge.intent)
        self.assertTrue(any(e.get('reason') == 'identity_replaced'
            for e in audit['additional_identity_events']))
        # 可重新提出同编号的新对象，但必须重新由状态机确认。
        self.assertFalse(audit['intent_committed'])

    def test_missing_identity_revokes_and_cannot_keep_old_route(self):
        bridge, a = self.committed()
        del a.nodes[1]
        final, _, audit = self.select(bridge, a)
        self.assertFalse(final)
        self.assertIsNone(bridge.intent)
        self.assertEqual(a.path, [])
        self.assertTrue(any(e.get('reason') == 'identity_missing'
            for e in audit['additional_identity_events']))

    def test_changed_query_cannot_keep_previous_intent(self):
        bridge, a = self.committed()
        bridge.query = '最近的公交车'
        final, _, audit = self.select(bridge, a)
        self.assertFalse(final)
        self.assertIsNone(bridge.intent)
        self.assertTrue(any(e.get('reason') == 'query_changed'
            for e in audit['additional_identity_events']))

    def test_static_track_drift_requires_fresh_selection(self):
        bridge, a = self.committed()
        a.nodes[1].bbox_center[0] += 1.6
        _, _, audit = self.select(bridge, a)
        self.assertIsNone(bridge.intent)
        self.assertTrue(any(e.get('reason') == 'static_identity_position_drift'
            for e in audit['additional_identity_events']))

    def test_current_support_still_required_for_committed_intent(self):
        for change in ['motion', 'units', 'pixels', 'future_time', 'short_term']:
            with self.subTest(change=change):
                bridge, a = self.committed()
                node = a.nodes[1]
                if change == 'motion':
                    node.motion_state = 'dynamic'
                elif change == 'units':
                    node.metadata['depth_evidence'][-1]['units'] = 'mm'
                elif change == 'pixels':
                    node.metadata['depth_evidence'][-1]['retained_pixels'] = 0
                elif change == 'future_time':
                    node.metadata['timestamp_s'] = 5.
                else:
                    node.is_short_term = True
                final, plans, audit = self.select(bridge, a)
                self.assertFalse(final)
                self.assertFalse(plans)
                self.assertFalse(audit['current_evidence_passed'])
                self.assertEqual(a.path, [])

    def test_expired_observation_revokes_intent(self):
        bridge, a = self.committed()
        self.state = VehicleState(0., 0., 0., time_s=3.)
        final, _, audit = self.select(bridge, a)
        self.assertFalse(final)
        self.assertIsNone(bridge.intent)
        self.assertTrue(any(e.get('reason') == 'observation_expired'
            for e in audit['additional_identity_events']))

    def test_new_unsafe_route_blocks_committed_identity(self):
        bridge, a = self.committed()
        bridge.planner = lambda *args: (None, {'passed': False, 'reason': 'new obstacle'})
        final, plans, audit = self.select(bridge, a)
        self.assertTrue(audit['intent_committed'])
        self.assertTrue(audit['current_evidence_passed'])
        self.assertFalse(final)
        self.assertFalse(plans)
        self.assertEqual(a.path, [])

    def test_pending_backend_stops_search_when_observed_objects_exist(self):
        bridge = self.make(backend=previous.Backend(blocked=True))
        a = previous.agent()
        a.semantic_bridge = bridge
        a.path_purpose = 'bounded_lane_search'
        self.select(bridge, a)
        self.assertTrue(bridge.hold_for_pending_semantics)
        result = hold_search_until_semantic_delivery(a,
            dict(path_world=a.path, path_purpose=a.path_purpose))
        self.assertTrue(result['pending_semantics_search_suppressed'])
        self.assertFalse(a.path)
        self.assertFalse(result['path_world'])

    def test_missing_observed_category_does_not_suspend_search(self):
        bridge = self.make(backend=previous.Backend(blocked=True))
        a = previous.agent()
        for node in a.nodes.values():
            node.object_tag = 'pedestrian'
        a.semantic_bridge = bridge
        a.path_purpose = 'bounded_lane_search'
        self.select(bridge, a)
        result = hold_search_until_semantic_delivery(a,
            dict(path_world=a.path, path_purpose=a.path_purpose))
        self.assertFalse(result['pending_semantics_search_suppressed'])
        self.assertTrue(a.path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
