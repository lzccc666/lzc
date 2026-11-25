"""关联原型必须有真实互投连续，不因旧别名或空间邻近强制绑定。"""
import numpy as np

from conceptgraph.slam.mvrepair_20261123_prototype import prototype_witness, independent_competitors, merge_prototypes, current_parts_compatible


def fixtures():
    view=dict(pose=np.eye(4),intrinsic=np.eye(3))
    obs=dict(camera='CAM_FRONT_LEFT',category='truck',center=np.array([0.,0.,10.]),
        feature=[1.,0.],world_points=np.tile([0.,0.,10.],(10,1)),mask=np.ones((5,5),bool))
    context=dict(source_key=['CAM_FRONT_LEFT',1.,0],timestamp=1.,category='truck',
        feature=[1.,0.],center=np.array([0.,0.,10.]),points=obs['world_points'],view=view,mask=obs['mask'])
    return obs,view,context


def test_current_physical_support_can_continue_prototype():
    obs,view,context=fixtures()
    assert prototype_witness(obs,view,[context],1.5)['primary']


def test_neighbor_cannot_takeover_from_center_or_feature_alone():
    obs,view,context=fixtures()
    obs['mask']=np.zeros((5,5),bool)
    assert not prototype_witness(obs,view,[context],1.5)['physical_continuity']


def test_expired_prototype_has_no_binding():
    obs,view,context=fixtures()
    assert not prototype_witness(obs,view,[context],3.)['primary']


def test_physical_secondary_still_checks_other_canonical_competition():
    obs,view,context=fixtures()
    obs['feature']=[.77,np.sqrt(1-.77**2)]
    witness=prototype_witness(obs,view,[context],1.5)
    assert witness['physical_continuity'] and not witness['primary']
    other=dict(canonical='independent',category='truck',center=obs['center'],feature=obs['feature'])
    assert independent_competitors(obs,'original',[other]) == ['independent']


def test_canonical_union_keeps_unique_real_contexts_and_latest_real_clock():
    _,_,context=fixtures()
    second=dict(context,source_key=['CAM_FRONT_LEFT',1.4,1],timestamp=1.4)
    result=merge_prototypes({'a':dict(contexts=[context],last_seen=1.),
        'b':dict(contexts=[context,second],last_seen=1.4)},lambda _: 'a',1.5)
    assert len(result['a']['contexts']) == 2 and result['a']['last_seen'] == 1.4


def test_current_parts_need_complete_pixel_and_box_containment():
    a=dict(camera='front',mask=np.ones((5,5),bool),box=[0,0,5,5])
    b=dict(camera='front',mask=np.ones((5,5),bool),box=[1,1,4,4])
    assert current_parts_compatible(a,b)
    assert not current_parts_compatible(a,dict(b,box=[4,4,7,7]))
    assert not current_parts_compatible(a,dict(b,camera='left'))
