"""核验移除实际生效、TTL覆盖及新旧闭环参数等价；不伪装成真实实验。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from carla_agent_isolated_002 import mapping_config
from carla_history_lifecycle_017 import Agent, lifecycle_config
from conceptgraph.slam.streamlined_mapping import MappingFrame, StreamlinedMapper
from conceptgraph.slam.utils import initialize_motion_metadata, update_motion_metadata_after_match, should_use_short_term_update

R = Path(__file__).resolve().parents[1]


class Box:
    def get_center(self):
        return np.array([1., 0., 1.])


class LifecycleTests(unittest.TestCase):
    def test_removal_does_not_invent_static_motion_but_original_confirms_stationary_observations(self):
        for enabled in [True, False]:
            cfg = lifecycle_config(mapping_config(), enabled)
            obj = dict(bbox=Box(), class_name=['car'], timestamp_s=0.)
            initialize_motion_metadata(cfg, obj, 0)
            for index in [1, 2]:
                det = dict(bbox=Box(), class_name=['car'], timestamp_s=float(index))
                update_motion_metadata_after_match(cfg, obj, det, 0., Box().get_center(), index)
            self.assertEqual(obj.get('motion_state', 'unknown'), 'static' if enabled else 'unknown')
            self.assertEqual(should_use_short_term_update(cfg, obj, det, 3.), enabled)
            self.assertEqual(len(obj.get('center_history', [])), 3 if enabled else 0)

    def test_off_retains_old_short_term_object_when_original_expiry_removes_it(self):
        counts = []
        for enabled in [True, False]:
            mapper = StreamlinedMapper(lifecycle_config(mapping_config(), enabled), ['car'])
            mapper.objects.append(dict(bbox=Box(), is_short_term=True, timestamp_s=0., num_detections=2))
            frame = MappingFrame(1, np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2)),
                                 np.eye(3), np.eye(4), {}, 'fixture', timestamp_s=3.)
            with patch('conceptgraph.slam.streamlined_mapping.gobs_to_detection_list', return_value=([], [])):
                mapper.update(frame)
            counts.append(len(mapper.objects))
        self.assertEqual(counts, [0, 1])

    def test_actual_agent_changes_only_lifecycle_parameters_and_keeps_safety_policy(self):
        prior = dict(centerline_xy=[[0., 0.], [50., 0.]], road_polygon_xy=[[-5., -2.], [55., -2.], [55., 2.], [-5., 2.]])
        with tempfile.TemporaryDirectory() as name:
            full = Agent('靠近小汽车', prior, ['car'], Path(name), 'full_rule_geometry')
            off = Agent('靠近小汽车', prior, ['car'], Path(name), 'without_history_maintenance')
            a, b = [OmegaConf.to_container(x.mapper.cfg) for x in [full, off]]
            self.assertEqual({k for k in a if a[k] != b[k]}, {'enable_dynamic_object_handling', 'dynamic_ttl_s'})
            self.assertEqual(full.evidence_policy, off.evidence_policy)
            self.assertEqual(full.road, off.road)
            self.assertFalse(off.guard_latched)

    def test_runtime_derivation_and_matched_episode_inputs_are_exact(self):
        receipt = json.loads((R/'carla_official/history_runtime_derivation_012.json').read_text())
        original = Path(receipt['source']).read_text()
        for item in receipt['exact_replacements']:
            self.assertEqual(original.count(item['before']), item['occurrences'])
            original = original.replace(item['before'], item['after'])
        self.assertEqual(original, Path(receipt['derived']).read_text())
        bank = json.loads((R/'carla_official/expansion_registration_012.json').read_text())
        parent = json.loads(Path(bank['original_frozen_registration']).read_text())
        for name in ['episodes', 'seeds', 'seed_perturbations', 'maximum_wall_s', 'graph_config', 'semantic_protocol', 'adapter', 'base_model_binding']:
            self.assertEqual(bank[name], parent[name], name)
        jobs = bank['execution_phases']['history_maintenance']
        self.assertEqual(len(jobs), 30)
        self.assertEqual([dict(copy.deepcopy(j), condition='full_rule_geometry') for j in jobs],
                         bank['full_reference_jobs_in_primary'])
        self.assertTrue(all(j in parent['execution_phases']['primary'] for j in bank['full_reference_jobs_in_primary']))


if __name__ == '__main__':
    unittest.main()
