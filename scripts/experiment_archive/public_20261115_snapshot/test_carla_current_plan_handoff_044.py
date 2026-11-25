"""计划交接的行为与拒绝保护回归；不启动CARLA或GPU。"""
import copy
from dataclasses import asdict
from types import SimpleNamespace
import unittest

import numpy as np
from shapely.geometry import LineString, box

from carla_current_plan_handoff_044 import RecordingPlanBridge, apply_current_plan
from conceptgraph.navigation.vehicle_geometry import VehicleSpec, VehicleState, step_vehicle
from conceptgraph.navigation.vehicle_planner import VehiclePlan


class PlanHandoffTests(unittest.TestCase):
    def setUp(self):
        self.spec = VehicleSpec()
        self.state = VehicleState(0., 0., 0., speed_mps=.5, time_s=1.)
        self.states = [self.state]
        for _ in range(10):
            self.states.append(step_vehicle(self.states[-1], 0., 0., self.spec))
        self.goal = np.array([self.states[-1].x, self.states[-1].y])
        self.agent = SimpleNamespace(selected=1, plan_target=1, guard_latched=False,
            path=[[0., 0.], [8., 0.]], path_purpose='target_approach', goal=np.array([8., 0.]),
            nodes={1: SimpleNamespace(metadata={'object_identity': 'track-0001'})},
            road=box(-10., -3., 30., 3.), obstacles=[], line=LineString([(-10., 0.), (30., 0.)]),
            approach_goal=lambda identity: (self.goal.copy(), 0.))
        self.plan = VehiclePlan(self.states, 2, .03)
        self.delegate = SimpleNamespace(
            select=lambda *args: ([SimpleNamespace(node_id=1)], {1: self.plan},
                {'authorized_node_id': 1, 'current_evidence_passed': True}),
            backend=object())
        self.bridge = RecordingPlanBridge(self.delegate)
        self.bridge.select(self.agent, [], [], self.state, 5)
        self.result = dict(frame_idx=5, path_world=copy.deepcopy(self.agent.path),
            path_purpose='target_approach')

    def apply(self):
        return apply_current_plan(self.agent, self.bridge, self.result, self.state, self.spec)

    def refused(self, reason):
        r = self.apply()
        self.assertFalse(r['current_plan_handoff']['applied'])
        self.assertEqual(r['current_plan_handoff']['reason'], reason)
        self.assertEqual(self.agent.path, [])
        self.assertIsNone(self.agent.plan_target)
        self.assertIsNone(self.agent.goal)

    def test_new_plan_replaces_old_path_even_with_same_locked_identity(self):
        # 原路径分支因plan_target==selected不会接收prepared_plans。
        self.assertEqual(self.agent.plan_target, self.agent.selected)
        old_endpoint = self.agent.path[-1].copy()
        r = self.apply()
        self.assertTrue(r['current_plan_handoff']['applied'])
        self.assertNotEqual(old_endpoint, self.agent.path[-1])
        np.testing.assert_equal(self.agent.path[-1], self.goal)
        np.testing.assert_equal(self.agent.goal, self.goal)
        self.assertEqual(r['planning']['goal_xy'], self.goal.tolist())

    def test_existing_stop_latch_is_never_reset_by_new_plan(self):
        self.agent.guard_latched = True
        self.assertTrue(self.apply()['current_plan_handoff']['applied'])
        self.assertTrue(self.agent.guard_latched)

    def test_previous_frame_plan_cannot_authorize_current_motion(self):
        self.result['frame_idx'] = 6
        self.refused('different_or_missing_physical_snapshot')

    def test_changed_vehicle_pose_rejects_prepared_plan(self):
        self.state = VehicleState(.02, 0., 0., speed_mps=.5, time_s=1.)
        self.refused('different_or_missing_physical_snapshot')

    def test_new_obstacle_after_selection_revokes_route(self):
        self.agent.obstacles = [box(.1, -.1, .2, .1)]
        self.refused('current_obstacles_or_padded_body_invalid')

    def test_missing_evidence_cannot_keep_stale_route(self):
        self.bridge.last_selection.evidence_passed = False
        self.refused('current_target_not_authorized')

    def test_plan_for_another_object_does_not_apply(self):
        self.bridge.last_selection.authorized_id = 2
        self.refused('current_target_not_authorized')

    def test_reused_numeric_id_rejects_old_plan(self):
        self.agent.nodes[1].metadata['object_identity'] = 'replacement-object'
        self.refused('object_identity_changed')

    def test_changed_observed_goal_requires_new_plan(self):
        self.goal += [1., 0.]
        self.refused('observed_goal_changed_after_selection')

    def test_finite_valid_goal_does_not_authorize_nan_trajectory(self):
        plan = self.bridge.last_selection.plans[1]
        states = list(plan['states'])
        # 绕过冻结数据类构造检查，模拟跨模块传递后被破坏的数据。
        object.__setattr__(states[3], 'x', float('nan'))
        plan['states'] = tuple(states)
        self.refused('nonfinite_planned_state')

    def test_failed_plan_cannot_restore_path(self):
        self.bridge.last_selection.plans[1]['failure'] = 'PLANNING_TIME_BUDGET'
        self.refused('plan_does_not_start_at_current_state')

    def test_unlocked_safe_search_is_not_removed(self):
        self.agent.selected = None
        self.agent.path_purpose = 'bounded_lane_search'
        old = copy.deepcopy(self.agent.path)
        r = self.apply()
        self.assertFalse(r['current_plan_handoff']['applied'])
        self.assertEqual(self.agent.path, old)

    def test_recorder_does_not_change_backend_or_selection_result(self):
        self.assertIs(self.bridge.backend, self.delegate.backend)
        final, plans, audit = self.bridge.select(self.agent, [], [], self.state, 5)
        self.assertEqual(final[0].node_id, 1)
        self.assertIs(plans[1], self.plan)
        self.assertEqual(audit['authorized_node_id'], 1)
        self.assertEqual(self.bridge.last_selection.state, asdict(self.state))


if __name__ == '__main__':
    unittest.main(verbosity=2)
