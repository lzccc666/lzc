"""纯接口反例：空间邻近不足、未来/弱支持不能触发部分观测归并。"""
import numpy as np

from conceptgraph.slam.mvrepair_20261123_partial_fragment import temporal_witness


def state(node, points, timestamp, hits):
    return dict(node_id=node, hits=hits, timestamp=timestamp, _last_world_points=np.tile([0.,0.,10.],(points,1)),
        _last_reliable_dense=points>=8, _observation_keys=[['CAM_FRONT_LEFT',timestamp,0]],
        feature=[1.,0.], last_measured_center=np.array([0.,0.,10.]))


def view(mask=True):
    return dict(pose=np.eye(4),intrinsic=np.eye(3),masks={0:np.full((4,4),mask)})


def test_current_strong_temporal_mutual_support_is_required():
    weak, old, current = state('w',7,1.,1),state('s',16,1.,2),state('s',30,1.5,3)
    assert temporal_witness(weak,old,current,view(),view(),1.5)['accepted']
    assert not temporal_witness(weak,old,current,view(),view(False),1.5)['accepted']


def test_old_multi_group_or_future_fragment_is_not_a_single_surface_fragment():
    weak, old, current = state('w',7,1.,2),state('s',16,1.,2),state('s',30,1.5,3)
    assert not temporal_witness(weak,old,current,view(),view(),1.5)['accepted']
    weak['hits']=1
    assert not temporal_witness(weak,old,current,view(),view(),.5)['accepted']


def test_independent_past_masks_cannot_merge_by_center_alone():
    weak, old, current = state('w',7,1.,1),state('s',16,1.,2),state('s',30,1.5,3)
    weak['_observation_keys']=[['CAM_FRONT_LEFT',1.,1]]
    past=view()
    past['masks'][1]=np.zeros((4,4),bool)
    assert not temporal_witness(weak,old,current,past,view(),1.5)['accepted']
