"""名义停车距离与速度上限的一致性；不把测试当成官方物理验证。"""
from dataclasses import replace
import math
import unittest

from carla_stock_stopping_profile_049 import StockStoppingProfile, observed_approach_profile
from conceptgraph.navigation.vehicle_geometry import VehicleState


class StoppingProfileTests(unittest.TestCase):
    def setUp(self):
        self.p = StockStoppingProfile()

    def test_stop_distance_uses_observed_speed_above_cruise_cap(self):
        self.assertGreater(self.p.distance(.8), self.p.distance(.55))
        self.assertGreater(self.p.distance(.5), 3.)

    def test_inverse_limit_uses_exact_same_distance_model(self):
        for v in [.01, .08, .149, .15, .151, .3, .5, .55]:
            with self.subTest(speed=v):
                self.assertAlmostEqual(self.p.speed_cap(self.p.distance(v)), v, places=10)

    def test_reduced_available_distance_cannot_increase_speed(self):
        values = [self.p.speed_cap(d) for d in [-1., 0., .3, .35, .5, 1., 2., 4., 10.]]
        self.assertEqual(values, sorted(values))
        self.assertEqual(values[:3], [0., 0., 0.])
        self.assertLessEqual(values[-1], .55)

    def test_distance_continuous_at_low_speed_brake_gate(self):
        self.assertAlmostEqual(self.p.distance(.15-1.e-10), self.p.distance(.15+1.e-10), places=7)

    def test_malformed_inputs_are_rejected(self):
        for v in [float('nan'), float('inf'), -1.]:
            with self.subTest(speed=v), self.assertRaises(ValueError):
                self.p.distance(v)
        for v in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                self.p.speed_cap(v)
        with self.assertRaises(ValueError):
            replace(self.p, coast_deceleration_mps2=0.)

    def test_missing_route_does_not_create_motion(self):
        r = observed_approach_profile(VehicleState(0., 0., 0.), [], None,
            dict(desired_speed_mps=.5), self.p)
        self.assertEqual(r['desired_speed_mps'], 0.)

    def test_speed_advice_does_not_raise_existing_stop_request(self):
        state = VehicleState(0., 0., 0., speed_mps=.5)
        r = observed_approach_profile(state, [[0., 0.], [10., 0.]], [10., 0.],
            dict(desired_speed_mps=0., current_motion_guard_stop=True), self.p)
        self.assertEqual(r['desired_speed_mps'], 0.)
        self.assertTrue(r['current_motion_guard_stop'])

    def test_speed_advice_keeps_caller_state_path_and_steering_unchanged(self):
        state = VehicleState(0., 0., 0., speed_mps=.5, steering_rad=.1)
        path = [[0., 0.], [1., 0.]]
        tracking = dict(desired_speed_mps=.55, steering_rad=.1)
        r = observed_approach_profile(state, path, [1., 0.], tracking, self.p)
        self.assertLess(r['desired_speed_mps'], .55)
        self.assertEqual(tracking['desired_speed_mps'], .55)
        self.assertEqual(path, [[0., 0.], [1., 0.]])
        self.assertEqual(r['steering_rad'], state.steering_rad)
        self.assertTrue(math.isfinite(r['stock_stopping_profile']['nominal_stopping_distance_m']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
