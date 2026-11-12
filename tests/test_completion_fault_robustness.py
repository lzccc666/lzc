import numpy as np

from conceptgraph.completion.fault_robustness import perturb


def test_blackout_preserves_frame_metadata_and_does_not_mutate_original():
    gobs=dict(xyxy=np.zeros((2,4)),class_id=np.array([0,1]),mask=np.ones((2,3,3),bool),
        image_crops=['a','b'],classes=['car','bus'],frame_provenance={'frame':8})
    depth=np.ones((3,3))
    pose=np.eye(4)
    noisy,observed,g,a=perturb(depth,pose,gobs,'three_frame_perception_blackout',np.random.default_rng(17),8)
    assert len(g['xyxy'])==0 and g['mask'].shape==(0,3,3)
    assert g['classes']==['car','bus'] and g['frame_provenance']=={'frame':8}
    assert len(gobs['xyxy'])==2 and np.array_equal(noisy,depth) and np.array_equal(observed,pose)
    assert a['retained_detections']==0


def test_depth_noise_does_not_fabricate_measurements_in_unobserved_pixels():
    depth=np.array([[0.,20.],[1.,0.]])
    gobs=dict(xyxy=np.zeros((0,4)),classes=['car'],frame_provenance={})
    noisy,_,_,_=perturb(depth,np.eye(4),gobs,'depth_noise_020m',np.random.default_rng(17),0)
    assert np.array_equal(noisy>0,depth>0)
    assert noisy[0,1]!=depth[0,1]
    assert np.array_equal(depth,[[0,20],[1,0]])
