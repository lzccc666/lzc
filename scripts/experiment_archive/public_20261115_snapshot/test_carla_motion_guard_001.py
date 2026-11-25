"""回归：没有路线不应跳过当前障碍和车身保护；注入点云不计传感器实测。"""
import unittest

import numpy as np
from shapely.geometry import box

from carla_motion_guard_001 import current_full_lidar_obstacles,nominal_coasting_check
from conceptgraph.navigation.vehicle_geometry import VehicleSpec,VehicleState


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.spec=VehicleSpec()
        self.state=VehicleState(0.,0.,0.,speed_mps=.55)
        self.road=box(-10,-3,40,3)

    def check(self,obstacles,**kwargs):
        return nominal_coasting_check(self.state,self.road,obstacles,self.spec,
            observation={'valid':True},**kwargs)

    def test_no_path_still_checks_obstacle_ahead(self):
        result=self.check([box(5,-1,6,1)])
        self.assertTrue(result['stop_required'])
        self.assertTrue(result['checked_without_route'])

    def test_empty_clear_lane_does_not_stop_only_because_route_missing(self):
        result=self.check([])
        self.assertFalse(result['stop_required'])
        self.assertFalse(result['actual_physical_safety_proven'])

    def test_side_obstacle_on_body_sweep_is_checked(self):
        self.assertTrue(self.check([box(4,.8,5,1.7)])['stop_required'])

    def test_far_behind_obstacle_is_not_a_forward_collision(self):
        self.assertFalse(self.check([box(-8,-1,-7,1)])['stop_required'])

    def test_missing_observation_never_means_free_space(self):
        result=nominal_coasting_check(self.state,self.road,[],self.spec,observation=None)
        self.assertTrue(result['stop_required'])

    def test_road_boundary_checked_without_path(self):
        result=nominal_coasting_check(self.state,box(-10,-3,4,3),[],self.spec,observation={'valid':True})
        self.assertTrue(result['stop_required'])

    def test_stale_sensor_frame_is_rejected(self):
        points=np.ones((100,4))
        _,result=current_full_lidar_obstacles(points,np.eye(4),self.state,self.spec,
            sensor_frame=5,snapshot_frame=6)
        self.assertFalse(result['valid'])

    def test_full_circle_points_behind_camera_are_retained(self):
        # 原始CARLA坐标y向右，内部坐标翻转；负x在前置相机视野之外。
        points=np.tile(np.array([-5.,2.,1.,.5]),(100,1))
        points[:,:2]+=np.random.default_rng(17).normal(0,.01,(100,2))
        obstacles,result=current_full_lidar_obstacles(points,np.eye(4),self.state,self.spec,
            sensor_frame=5,snapshot_frame=5)
        self.assertTrue(result['valid'])
        self.assertTrue(obstacles)
        self.assertLess(obstacles[0].centroid.x,-4)
        self.assertLess(obstacles[0].centroid.y,-1)


if __name__=='__main__':
    unittest.main(verbosity=2)
