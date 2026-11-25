import numpy as np

from conceptgraph.slam.paper_graph_20261123_sparse_continuity import ImageContinuityLayer


def observation(x=0.,box=(10,10,30,30),category='car'):
    return dict(center=np.array([x,0.,0.]),feature=np.array([1.,0.]),variance=np.ones(3),
        category=category,points=2,confidence=.9,image_box=list(box))


def test_unique_image_continuity_rejoins_sparse_track_without_relaxing_confirmation():
    layer=ImageContinuityLayer()
    assert layer.update([observation()],[],{},0.)==[]
    output=layer.update([observation(3.)],[],{},.5)
    assert len(layer.tracks)==1
    assert layer.tracks[0]['hits']==2 and layer.tracks[0]['total_points']==4
    assert output[0]['node_id']=='sparse-00000000'
    assert layer.fallback_used[0]['selected_update']
    assert layer.settings.minimum_frames==2 and layer.settings.minimum_total_points==4


def test_two_nearby_same_class_image_candidates_veto_fallback():
    layer=ImageContinuityLayer()
    layer.update([observation(),observation(.1)],[],{},0.)
    layer.update([observation(3.)],[],{},.5)
    assert not layer.fallback_used
    assert len(layer.tracks)==3


def test_image_gap_does_not_extend_original_ttl_or_cross_class():
    layer=ImageContinuityLayer()
    layer.update([observation()],[],{},0.)
    layer.update([observation(3.,category='bus')],[],{},.5)
    assert not layer.fallback_used
    layer.update([observation(3.)],[],{},2.1)
    assert len(layer.tracks)==1 and layer.tracks[0]['hits']==1


def test_reliable_dense_measurement_still_directly_protects_center():
    layer=ImageContinuityLayer()
    state=layer._new_state(observation(),0.,'dense-1')
    measured=observation(4.)
    measured.update(reliable_current_dense=True,maximum_speed_m_s=20.)
    layer._update_state(state,measured,.5)
    np.testing.assert_array_equal(state['center'],measured['center'])
