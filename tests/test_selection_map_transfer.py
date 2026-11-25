import numpy as np
import pytest

from conceptgraph.completion.evidence import target_evidence_decision
from conceptgraph.completion.selection_map_transfer import (
    EVIDENCE_POLICY, nodes_from_frozen_predictions, select_frozen_map,
)


def test_sparse_unknowns_remain_missing_and_cannot_receive_evidence_credit():
    prediction = dict(node_id='sparse-0001',category='car',center=[10.,0.,0.],
                      num_detections=4,shape_available=False)
    node = nodes_from_frozen_predictions([prediction])[0]
    assert node.bbox_extent is None and node.motion_state=='unknown'
    assert node.metadata['depth_evidence']==[]
    assert node.metadata['frozen_object_id']=='sparse-0001'
    assert not target_evidence_decision(node,EVIDENCE_POLICY)[0]
    pose=np.eye(4)
    pose[:2,2]=[1.,0.]
    assert select_frozen_map('去最近的停着的小汽车附近',[prediction],pose,4)==[]


def test_unknown_shape_is_explicit_and_nonfinite_center_does_not_disappear():
    with pytest.raises(ValueError):
        nodes_from_frozen_predictions([dict(node_id='x',category='car',center=[10.,0.,0.],num_detections=3)])
    with pytest.raises(ValueError):
        nodes_from_frozen_predictions([dict(node_id='x',category='car',center=[np.nan,0.,0.],
                                            num_detections=3,shape_available=False)])
