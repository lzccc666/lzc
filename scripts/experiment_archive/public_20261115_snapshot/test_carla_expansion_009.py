"""新道路候选不得绕过车身/障碍约束；证据消融必须确实影响授权；存储逐字节可逆。"""
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from shapely.geometry import LineString, box

from carla_expansion_truth_009 import barrier_certificate
from carla_lane_seed_bridge_006 import LaneSeedBridge
from carla_lossless_jsonl_001 import LosslessJsonl, read_rows
from carla_persistent_intent_bridge_004 import PersistentIntentBridge
from conceptgraph.navigation.vehicle_geometry import VehicleState


class ExpansionTests(unittest.TestCase):
    def agent(self, obstacles=()):
        return SimpleNamespace(nodes={0:SimpleNamespace(metadata={'object_identity':'test'})},
            obstacles=list(obstacles),road=box(-5.,-2.,30.,2.),line=LineString([(-5.,0.),(30.,0.)]),
            approach_goal=lambda identity:([7.,0.],0.))

    def test_lane_seed_is_dynamically_integrated_and_original_obstacles_audited(self):
        agent=self.agent();b=LaneSeedBridge('靠近小汽车','rule');start=VehicleState(0.,0.,0.)
        plan,check=b.current_plan(agent,0,start)
        self.assertIsNotNone(plan);self.assertGreater(len(plan.states),10)
        self.assertEqual(plan.states[0],start)
        self.assertTrue(check['original_individual_obstacles_sweep_audit']['passed'])
        self.assertEqual(agent.obstacles,[])

    def test_lane_seed_does_not_cross_blocking_vehicle(self):
        agent=self.agent([box(3.,-3.,5.,3.)]);b=LaneSeedBridge('靠近小汽车','rule')
        with patch('carla_aggregated_obstacle_bridge_005.AggregatedObstacleBridge.current_plan',return_value=(None,{'passed':False})):
            plan,check=b.current_plan(agent,0,VehicleState(0.,0.,0.))
        self.assertIsNone(plan);self.assertFalse(check['passed'])

    def test_barrier_truth_uses_connectivity_not_planning(self):
        road=box(-4.,-1.75,30.,1.75);goal=box(12.,-.5,14.,.5)
        self.assertTrue(barrier_certificate(road,goal,[0.,0.],box(8.,-3.,10.,3.))['independently_inaccessible'])
        self.assertFalse(barrier_certificate(road,goal,[0.,0.],box(8.,-.5,10.,.5))['independently_inaccessible'])

    def test_evidence_ablation_changes_consumed_control_gate(self):
        node=SimpleNamespace(object_tag='car',bbox_center=np.array([8.,0.,0.]),motion_state='unknown',is_short_term=True,
            metadata=dict(object_identity='observed-1',last_seen_frame=0,timestamp_s=0.,num_detections=1,
                depth_evidence=[dict(source='current_projected_lidar',units='m',retained_pixels=2,support_fraction=.2,sufficient_support=False)]))
        agent=SimpleNamespace(nodes={0:node},selected=None,path=[],plan_target=None,path_purpose=None)
        fake=lambda *args:(SimpleNamespace(states=[VehicleState(0.,0.,0.)]),{'passed':True})
        full=PersistentIntentBridge('靠近小汽车','rule',planner=fake)
        removed=PersistentIntentBridge('靠近小汽车','rule',planner=fake,evidence_check_enabled=False)
        good,_,audit=full.select(agent,[],[],VehicleState(0.,0.,0.),0)
        other,_,ablated=removed.select(agent,[],[],VehicleState(0.,0.,0.),0)
        self.assertEqual(good,[]);self.assertEqual(other[0].node_id,0)
        self.assertFalse(audit['current_evidence_passed']);self.assertFalse(ablated['current_evidence_passed'])
        self.assertTrue(audit['evidence_check_enabled']);self.assertFalse(ablated['evidence_check_enabled'])

    def test_lossless_multilingual_jsonl(self):
        parent=Path(__file__).resolve().parents[1]/'json'
        with tempfile.TemporaryDirectory(prefix='gzip_roundtrip_',dir=parent) as d:
            p=Path(d)/'execution.jsonl.gz';payload='{"中文":"逐字节","value":0.12345678912345678}\n'
            with LosslessJsonl(p) as stream:
                for _ in range(3):stream.write(payload);stream.flush()
            self.assertEqual(gzip.decompress(p.read_bytes()),(payload*3).encode())
            self.assertEqual(read_rows(Path(d),'execution.jsonl'),[json.loads(payload)]*3)


if __name__=='__main__':unittest.main(verbosity=2)
