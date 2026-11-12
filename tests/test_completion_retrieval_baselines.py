import numpy as np
import pytest

from conceptgraph.completion.retrieval_baselines import SparseFeatureGrid, query_english, associated_identity


def test_sparse_grid_expiry_uses_real_seconds_and_never_future_or_duplicate_time():
    grid=SparseFeatureGrid(ttl_s=2.)
    detection=dict(category='car',feature=np.array([1.,0.]),points=np.array([[0.,0.,10.],[.1,.1,10.]]))
    grid.update([detection],1530000000.)
    assert len(grid.cells)==1
    grid.update([],1530000001.5)
    assert len(grid.cells)==1
    grid.update([],1530000002.1)
    assert grid.cells=={}
    with pytest.raises(ValueError):
        grid.update([detection],1530000001.)


def test_rule_translation_and_independent_ambiguous_identity_stay_explicit():
    text,parsed=query_english('去公交车左侧停着的小汽车旁边')
    assert text=='a parked car to the left of a bus'
    assert parsed.relation=='left_of'
    truth=[dict(category='car',center=[1.,0.,0.],instance_token='a'),
           dict(category='car',center=[1.1,0.,0.],instance_token='b')]
    identity,status=associated_identity(dict(category='car',center=np.array([1.,0.,0.])),truth)
    assert identity is None and status=='ambiguous_independent_association'
