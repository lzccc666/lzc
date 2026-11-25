"""并集加速不得漏掉单个障碍、改变调用方输入或接受不合法几何。"""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from shapely.geometry import Polygon, box, LineString
from carla_aggregated_obstacle_bridge_004 import AggregatedObstacleBridge
from carla_real_graph_bridge_002 import RealGraphNavigationBridge
from conceptgraph.navigation.vehicle_geometry import VehicleState


class AggregatedPlanTests(unittest.TestCase):
    def setUp(self):
        self.bridge = AggregatedObstacleBridge('靠近汽车', 'rule')
        self.state = VehicleState(0., 0., 0.)
        self.agent = SimpleNamespace(nodes={}, road=box(-10., -3., 30., 3.),
            line=LineString([(0.,0.), (20.,0.)]), obstacles=[box(8., -1., 10., 1.),
                box(9., 0., 11., 2.)], approach_goal=lambda i: ([5.,0.], 0.))

    def test_returned_plan_rechecked_against_original_obstacles(self):
        original = list(self.agent.obstacles)
        returned = SimpleNamespace(states=[self.state])
        def planner(bridge, proxy, identity, state):
            self.assertEqual(len(proxy.obstacles), 1)
            for obstacle in original:
                self.assertTrue(proxy.obstacles[0].covers(obstacle))
            return returned, {'passed':True}
        with patch.object(RealGraphNavigationBridge, 'current_plan', planner):
            plan, check = self.bridge.current_plan(self.agent, 0, self.state)
        self.assertIs(plan, returned)
        self.assertTrue(check['original_individual_obstacles_sweep_audit']['passed'])
        self.assertEqual(self.agent.obstacles, original)

    def test_bad_planner_flag_cannot_override_original_collision(self):
        self.agent.obstacles.append(box(0., -1., 1., 1.))
        returned = SimpleNamespace(states=[self.state])
        with patch.object(RealGraphNavigationBridge, 'current_plan', return_value=(returned, {'passed':True})):
            plan, check = self.bridge.current_plan(self.agent, 0, self.state)
        self.assertIsNone(plan)
        self.assertFalse(check['passed'])
        self.assertFalse(check['original_individual_obstacles_sweep_audit']['passed'])

    def test_invalid_geometry_is_rejected_before_planning(self):
        self.agent.obstacles = [Polygon([(0.,0.),(2.,2.),(0.,2.),(2.,0.)])]
        with patch.object(RealGraphNavigationBridge, 'current_plan') as backend:
            plan, check = self.bridge.current_plan(self.agent, 0, self.state)
        self.assertIsNone(plan)
        self.assertFalse(check['passed'])
        backend.assert_not_called()

    def test_empty_obstacle_set_remains_empty(self):
        self.agent.obstacles = []
        def planner(bridge, proxy, identity, state):
            self.assertEqual(proxy.obstacles, [])
            return None, {'passed':False, 'reason':'test no plan'}
        with patch.object(RealGraphNavigationBridge, 'current_plan', planner):
            plan, check = self.bridge.current_plan(self.agent, 0, self.state)
        self.assertIsNone(plan)
        self.assertFalse(check['passed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
