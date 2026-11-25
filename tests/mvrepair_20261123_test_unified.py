"""物理点全键、桥接误并、组内确认与稠密稀疏连续ID。"""
import numpy as np

from conceptgraph.slam.mvrepair_20261123_unified import (
    UnifiedMapper, UnifiedSettings, group_observations, shared_witness,
)
from conceptgraph.slam.streamlined_mapping import MappingFrame


def observation(camera='CAM_FRONT', *, group=0, dense=False, shift=0., token='lidar0'):
    points = np.array([[shift, 0., 10.], [shift+.05, 0., 10.]])
    return dict(category='car', center=np.array([shift, 0., 10.]), feature=np.array([1., 0.]),
        confidence=.3, points=2 if not dense else 8, variance=np.ones(3), real_points=points,
        reliable_current_dense=dense, dense=dense, maximum_speed_m_s=20.,
        source_key=[camera, 1.+group*.5, 0], cameras=[camera], capture_timestamp_s=1.+group*.5,
        box=np.array([0., 0., 10., 10.]), lidar_point_keys=[[token, 1], [token, 2]])


def test_same_lidar_indices_from_different_sweeps_are_not_identical_points():
    a, b = observation(), observation('CAM_FRONT_RIGHT', shift=.5, token='other_lidar')
    result = shared_witness(a, b, UnifiedSettings())
    assert not result['accepted']
    b = observation('CAM_FRONT_RIGHT')
    b['feature'] = np.array([.72, np.sqrt(1-.72**2)])
    assert shared_witness(a, b, UnifiedSettings())['accepted']  # strong physical witness survives viewpoint appearance


def test_complete_link_prevents_side_bridge_merging_distinct_same_camera_objects():
    a, b, c = observation(), observation('CAM_FRONT_RIGHT'), observation()
    c['box'] = np.array([20., 0., 30., 10.])
    c['source_key'][-1] = 1
    grouped, audit = group_observations([a, b, c], UnifiedSettings())
    assert len(grouped) == 2 and audit['removed'] == 1


def test_sparse_dense_transition_keeps_one_id_and_one_hit_per_group():
    mapper = UnifiedMapper.__new__(UnifiedMapper)
    from conceptgraph.slam.dense_refinement import DenseProtectedPositionLayer
    mapper.settings, mapper.filter = UnifiedSettings(), DenseProtectedPositionLayer()
    mapper.states, mapper.next_id, mapper.timestamp = [], 0, None

    class Extractor:
        obs = []

        def extract_group(self, *args):
            return self.obs, []

    mapper.extractor = Extractor()
    output = []
    for group in range(3):
        timestamp = 1.+group*.5
        k = np.array([[10., 0., 10.], [0., 10., 10.], [0., 0., 1.]])
        frame = MappingFrame(group, np.zeros((20, 20, 3), dtype=np.uint8), np.zeros((20, 20)),
                             k, np.eye(4), {}, '', timestamp)
        mapper.extractor.obs = [observation(c, group=group, dense=group == 2) for c in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']]
        output.append(mapper.update_group([('CAM_FRONT', frame)], {}, {}))
    assert output[0] == [] and len(output[1]) == len(output[2]) == 1
    assert output[1][0]['node_id'] == output[2][0]['node_id']
    assert output[2][0]['num_detections'] == 3
    assert len(mapper.states) == 1 and mapper.states[0]['total_points'] == 12
