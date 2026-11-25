import numpy as np
import pytest

from conceptgraph.completion.retrieval_statistics import aggregate, bootstrap


def test_missing_method_for_paired_query_is_not_silently_dropped():
    def row(index,method):
        return dict(scene='s',frame_idx=index,query_index=0,method=method,
            positive=True,selected_correct=False,query_text='车辆',acceptable_instances=['id'])
    with pytest.raises(ValueError,match='incomplete paired'):
        aggregate([row(0,'a'),row(0,'b'),row(1,'a')],{'s':'log'})


def test_same_cluster_draws_preserve_zero_difference_for_identical_methods():
    # 两log难度不同；相同方法的配对差应对每次重采样为0。
    counts=np.array([[[1,10,3,4],[1,10,3,4]],[[8,10,0,4],[8,10,0,4]]])
    totals,estimate,rates=bootstrap(counts)
    assert totals[0].tolist()==[9,20,3,8]
    assert np.allclose(estimate[0],[.45,.375])
    assert np.all(rates[:,0]-rates[:,1]==0)
    assert np.min(rates[:,0,0])==.1 and np.max(rates[:,0,0])==.8
