"""异步模型输出不得越过当前对象证据或规划门闸；注入后端不计实际实验。"""
import copy
import json
from threading import Event
import time
from types import SimpleNamespace
import unittest

import numpy as np

from carla_real_graph_bridge_001 import RealGraphNavigationBridge, observed_row
from conceptgraph.navigation.types import SceneGraphNode
from conceptgraph.navigation.vehicle_geometry import VehicleState


QUERY = '不要最近的小汽车，选择第二近的小汽车'


def node(i, distance):
    return SceneGraphNode(i, 'car', ['car'], 'car', np.array([distance, 0., 1.]),
        np.array([2., 1., 1.]), motion_state='static', metadata=dict(
            object_identity=f'track-{i:08d}', num_detections=4, last_seen_frame=3,
            timestamp_s=.6, depth_evidence=[dict(source='current_projected_lidar',
            units='m', retained_pixels=10, support_fraction=.8, sufficient_support=True)]))


def agent():
    return SimpleNamespace(nodes={0: node(0, 10), 1: node(1, 20)}, selected=None,
        path=[[0., 0.], [2., 0.]], plan_target=0, obstacles=[])


class Backend:
    def __init__(self, *, blocked=False, invalid=False, missing=False):
        self.start, self.finish = Event(), Event()
        if not blocked:
            self.finish.set()
        self.invalid, self.missing = invalid, missing
        self.calls = 0
        self.rows = []

    def __call__(self, task, row):
        self.calls += 1
        self.rows.append(copy.deepcopy(row))
        self.start.set()
        if not self.finish.wait(5):
            raise RuntimeError('test backend was not released')
        if self.invalid:
            return 'not JSON'
        chosen = None if self.missing else 101
        order = [101, 100] if not self.missing else row['candidate_ids']
        return json.dumps(dict(schema_version='1.0.0', selected_node_id=chosen,
            ranking=order, no_match=self.missing, explanation='injected test only', confidence=.9))


def delivered(bridge):
    until = time.monotonic()+5
    while time.monotonic() < until:
        if any(e['kind'] == 'backend_completed' for e in bridge.queue.snapshot_events()):
            return
        Event().wait(.01)
    raise AssertionError('test backend did not finish')


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.bridges = []
        self.state = VehicleState(0, 0, 0, time_s=.6)

    def make(self, mode='adapter', backend=None, planner=None, **kwargs):
        if planner is None:
            planner = lambda a, i, s: (SimpleNamespace(states=[s], failure=None), {'passed': True})
        b = RealGraphNavigationBridge(QUERY, mode, backend=backend, planner=planner, **kwargs)
        self.bridges.append(b)
        return b

    def tearDown(self):
        for b in self.bridges:
            if b.backend:
                b.backend.finish.set()
            result = b.close(timeout_s=5)
            if result['worker']:
                self.assertTrue(result['worker']['worker_stopped'])

    def select(self, bridge, a):
        return bridge.select(a, [], [], self.state, 3)

    def test_second_nearest_pool_keeps_unreliable_distractor(self):
        a = agent()
        a.nodes[0].motion_state = 'unknown'
        b = self.make(mode='rule')
        final, _, audit = self.select(b, a)
        self.assertEqual({n['node_id'] for n in audit['input_row']['candidate_nodes']}, {100, 101})
        self.assertEqual(audit['proposed_node_id'], 1)
        self.assertEqual(final[0].node_id, 1)

    def test_late_result_cannot_refer_to_recycled_identity(self):
        backend = Backend(blocked=True)
        b = self.make(backend=backend)
        a = agent()
        self.assertFalse(self.select(b, a)[0])
        self.assertTrue(backend.start.wait(2))
        a.nodes[1].metadata['object_identity'] = 'track-another-object'
        backend.finish.set()
        delivered(b)
        final, _, audit = self.select(b, a)
        self.assertFalse(final)
        self.assertIsNone(audit['authorized_node_id'])
        self.assertTrue(any(e.get('reason') == 'binding_changed' for e in b.queue.snapshot_events()))

    def test_completed_proposal_requires_current_depth_again(self):
        backend = Backend()
        b = self.make(backend=backend)
        a = agent()
        self.select(b, a)
        delivered(b)
        a.nodes[1].metadata['depth_evidence'][-1]['source'] = 'unmeasured'
        final, _, audit = self.select(b, a)
        self.assertEqual(audit['proposed_node_id'], 1)
        self.assertFalse(audit['current_evidence_passed'])
        self.assertFalse(final)
        self.assertEqual(a.path, [])

    def test_invalid_JSON_never_falls_back_to_nearest(self):
        backend = Backend(invalid=True)
        b = self.make(backend=backend)
        a = agent()
        self.select(b, a)
        delivered(b)
        final, _, audit = self.select(b, a)
        self.assertFalse(final)
        self.assertIsNone(audit['proposed_node_id'])

    def test_valid_refusal_clears_old_path(self):
        backend = Backend(missing=True)
        b = self.make(backend=backend)
        a = agent()
        self.select(b, a)
        delivered(b)
        a.path = [[0., 0.], [2., 0.]]
        final, _, audit = self.select(b, a)
        self.assertTrue(audit['background_result_used'])
        self.assertFalse(final)
        self.assertEqual(a.path, [])

    def test_correct_proposal_does_not_bypass_failed_plan(self):
        backend = Backend()
        b = self.make(backend=backend,
            planner=lambda a, i, s: (None, {'passed': False, 'reason': 'new obstacle'}))
        a = agent()
        self.select(b, a)
        delivered(b)
        final, plans, audit = self.select(b, a)
        self.assertEqual(audit['proposed_node_id'], 1)
        self.assertTrue(audit['current_evidence_passed'])
        self.assertFalse(final)
        self.assertFalse(plans)

    def test_expired_output_is_not_reactivated_by_polling(self):
        wall = [0.]
        backend = Backend()
        b = self.make(backend=backend, clock=lambda: wall[0], ttl_s=1.)
        a = agent()
        self.select(b, a)
        delivered(b)
        wall[0] = 2.
        final, _, audit = self.select(b, a)
        self.assertFalse(final)
        self.assertFalse(audit['background_result_used'])
        self.assertTrue(any(e.get('reason') == 'expired_before_delivery' for e in b.queue.snapshot_events()))

    def test_one_inflight_request_does_not_accumulate_per_frame(self):
        backend = Backend(blocked=True)
        b = self.make(backend=backend)
        a = agent()
        for _ in range(20):
            self.select(b, a)
        self.assertTrue(backend.start.wait(2))
        self.assertEqual(backend.calls, 1)
        self.assertEqual(sum(e['kind'] == 'submitted' for e in b.queue.snapshot_events()), 1)

    def test_candidate_world_geometry_change_invalidates_response(self):
        backend = Backend(blocked=True)
        b = self.make(backend=backend)
        a = agent()
        self.select(b, a)
        self.assertTrue(backend.start.wait(2))
        a.nodes[1].bbox_center[0] += 2
        backend.finish.set()
        delivered(b)
        self.assertFalse(self.select(b, a)[0])

    def test_base_and_adapter_construct_identical_input(self):
        a = agent()
        b1, b2 = Backend(), Backend()
        base, adapter = self.make(mode='base', backend=b1), self.make(backend=b2)
        self.select(base, a)
        self.select(adapter, a)
        delivered(base)
        delivered(adapter)
        self.assertEqual(b1.rows, b2.rows)
        self.assertEqual(set(b1.rows[0]), {'task','sample_id','query_text','candidate_nodes',
                                         'candidate_ids','relation_metadata'})

    def test_missing_time_and_short_term_track_cannot_authorize(self):
        for field in ['timestamp', 'short_term', 'units']:
            a = agent()
            if field == 'timestamp':
                a.nodes[1].metadata['timestamp_s'] = -10
            elif field == 'short_term':
                a.nodes[1].is_short_term = True
            else:
                a.nodes[1].metadata['depth_evidence'][-1]['units'] = 'mm'
            b = self.make(mode='rule')
            self.assertFalse(self.select(b, a)[0], field)

    def test_coordinate_transform_uses_ego_heading(self):
        a = agent()
        row, _, _, _ = observed_row(QUERY, a.nodes, [],
            VehicleState(0, 0, np.pi/2, time_s=.6), 3)
        self.assertTrue(all(abs(n['center_xy'][0])<.001 and n['center_xy'][1]<0
                            for n in row['candidate_nodes']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
