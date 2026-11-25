"""原控制提前返回也执行保护；禁止把软件测试当物理安全证据。"""
from types import SimpleNamespace
import unittest

from shapely.geometry import box
from carla_guarded_control_001 import guarded_control
from conceptgraph.navigation.vehicle_geometry import VehicleState, VehicleSpec


class AlwaysGuardTests(unittest.TestCase):
    def setUp(self):
        self.state = VehicleState(0., 0., 0., speed_mps=.55)
        self.spec = VehicleSpec()
        self.agent = SimpleNamespace(road=box(-10., -3., 40., 3.),
            obstacles=[], path=[], goal=None, current_lidar_observation={'valid': True})

    def call(self):
        return guarded_control(self.agent, self.state, self.spec, -2., 0.,
            {'reason': 'waiting_for_evidence_or_plan'})

    def test_no_route_wait_does_not_skip_obstacle_check(self):
        self.agent.obstacles = [box(5., -1., 6., 1.)]
        _, _, tracking = self.call()
        self.assertTrue(tracking['always_on_motion_guard']['always_executed'])
        self.assertTrue(tracking['current_motion_guard_stop'])
        self.assertEqual(tracking['desired_speed_mps'], 0.)

    def test_missing_current_sensor_demands_stop(self):
        self.agent.current_lidar_observation = None
        _, _, tracking = self.call()
        self.assertTrue(tracking['current_motion_guard_stop'])

    def test_clear_road_preserves_controller_result_and_labels_scope(self):
        acceleration, steering, tracking = self.call()
        self.assertEqual((acceleration, steering), (-2., 0.))
        self.assertFalse(tracking['always_on_motion_guard']['stop_required'])
        self.assertFalse(tracking['always_on_motion_guard']['actual_physical_safety_proven'])
        self.assertEqual(tracking['reason'], 'waiting_for_evidence_or_plan')

    def test_road_edge_is_checked_without_route(self):
        self.agent.road = box(-10., -3., 4., 3.)
        self.assertTrue(self.call()[2]['current_motion_guard_stop'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
