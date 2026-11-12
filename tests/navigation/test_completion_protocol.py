import copy
import math

import numpy as np
import pytest

from conceptgraph.completion.association import gated_assignment, gated_distance_match
from conceptgraph.completion.protocol import ArrivalEvaluator, GoalTruth
from conceptgraph.completion.quality import identity_metrics, object_frame_metrics, relations_from_centers
from conceptgraph.navigation.online import SnapshotNodeTracker
from tests.navigation.test_snapshot_replay import raw_object
from conceptgraph.completion.depth import depth_support


def test_depth_support_uses_measured_cluster_and_rejects_nonfinite_points():
    depth=np.array([[10,10.1,10.2,10.3,10.4,40,41,np.inf,np.nan,0]])
    mask,evidence=depth_support(depth,np.ones_like(depth,dtype=bool),robust=True)
    assert evidence['valid_pixels']==7 and evidence['retained_pixels']==5
    assert evidence['sufficient_support']
    assert np.all(depth[mask]<11)
    assert evidence['source']=='current_projected_lidar'


def test_new_motion_protocol_requires_temporal_evidence_for_static_vehicle():
    from types import SimpleNamespace
    from conceptgraph.slam.utils import initialize_motion_metadata, update_motion_metadata_after_match
    from tests.navigation.test_qwen_online_crops import _BBox
    cfg=SimpleNamespace(enable_dynamic_object_handling=True,initial_motion_unknown=True,
        static_confirmation_observations=3)
    obj={'bbox':_BBox(),'class_name':['car']}
    initialize_motion_metadata(cfg,obj,0)
    assert obj['motion_state']=='unknown'
    det={'bbox':_BBox(),'class_name':['car']}
    update_motion_metadata_after_match(cfg,obj,det,0,np.array([3,4,.5]),1)
    assert obj['motion_state']=='unknown'
    update_motion_metadata_after_match(cfg,obj,det,0,np.array([3,4,.5]),2)
    assert obj['motion_state']=='static'


def test_evidence_policy_keeps_supported_positive_and_rejects_unknown_or_shallow():
    from conceptgraph.completion.evidence import target_evidence_decision
    from tests.navigation.test_online_state_machine import node
    n=node()
    n.motion_state='static'
    n.metadata.update(num_detections=3,depth_evidence=[dict(source='current_projected_lidar',
        units='m',retained_pixels=6,support_fraction=.8)])
    policy={'enabled':True}
    assert target_evidence_decision(n,policy)[0]
    n.motion_state='unknown'
    assert not target_evidence_decision(n,policy)[0]
    n.motion_state='static'
    n.metadata['depth_evidence'][-1]['retained_pixels']=2
    assert not target_evidence_decision(n,policy)[0]
    assert target_evidence_decision(n,{})[0]


def test_robust_ground_velocity_uses_seconds_and_ignores_vertical_surface_jitter():
    from conceptgraph.completion.motion import robust_xy_speed
    centers=[[0,0,0],[1,0,20],[2,0,-5],[3,0,50]]
    assert robust_xy_speed(centers,[0,.5,1,1.5])==2.0
    assert robust_xy_speed([[0,0,0],[0,0,20]],[0,.5])==0.0
    with pytest.raises(ValueError):
        robust_xy_speed(centers,[0,.5,.5,1.5])


def truth():
    return GoalTruth('task', ('official-a',), ({'instance_id': 'official-a',
        'polygon_xy': [[1, -2], [4, -2], [4, 2], [1, 2]], 'heading_rad': 0},),
        '冻结局部道路', (0, 90), '独立几何标注', reachability='reachable')


def state(t, **kwargs):
    return dict(time_s=t, xy=[2, 0], yaw_rad=0, speed_mps=0,
                evaluated_instance_id='official-a', **kwargs)


def test_arrival_requires_identity_region_stop_heading_hold_and_safety():
    ev = ArrivalEvaluator(truth())
    assert not ev.update(**state(0))['actually_arrived']
    assert ev.update(**state(2))['actually_arrived']
    for key, value in [('evaluated_instance_id','wrong'), ('xy',[0,0]),
                       ('yaw_rad',math.pi), ('speed_mps',1)]:
        ev = ArrivalEvaluator(truth())
        for t in (0, 3):
            s = state(t)
            s[key] = value
            assert not ev.update(**s)['actually_arrived']
    ev = ArrivalEvaluator(truth())
    ev.update(**state(0, body_collision=True))
    assert not ev.update(**state(4))['actually_arrived']


def test_hold_reset_timeout_and_time_validation():
    ev = ArrivalEvaluator(truth())
    ev.update(**state(0))
    s = state(1)
    s['speed_mps'] = 1
    ev.update(**s)
    assert not ev.update(**state(2))['actually_arrived']
    assert ev.update(**state(92))['status'] == 'TIMEOUT'
    with pytest.raises(ValueError):
        ev.update(**state(92))


def test_matching_maximizes_legal_cardinality_before_distance():
    # 先做无门限Hungarian再剔除，会漏掉可合法匹配的第二对。
    pred = [{'category':'car','center':[0,0]}, {'category':'car','center':[2,0]}]
    gt = [{'category':'car','center':[0,0]}, {'category':'car','center':[0,2.9]}]
    assert len(gated_distance_match(pred, gt, 3)) == 2
    assert gated_assignment([[1, -np.inf], [0.9, -np.inf]]) in ({0:0}, {1:0})
    assert gated_assignment([[1], [0.9]], policy='greedy_many_to_one') == {0:0,1:0}
    assert gated_assignment(np.zeros((0, 2))) == {}


def test_no_predictions_is_not_perfect_safety_or_precision():
    r = object_frame_metrics([], [{'category':'car','center':[0,0]}])
    assert r['recall'] == 0 and r['precision'] is None


def test_identity_and_driver_frame_relations():
    r=identity_metrics([(0,[(1,'a')]), (1,[(2,'a')]), (3,[(3,'a')])])
    assert r['identity_switches'] == 1
    objects=[{'id':'a','center':[-3,0]}, {'id':'b','center':[0,0]}]
    assert ('a','left_of','b') in relations_from_centers(objects, forward_xy=[0,1])
    assert ('a','behind','b') in relations_from_centers(objects, forward_xy=[1,0])


def test_snapshot_speed_uses_elapsed_seconds_and_stable_identity():
    tr = SnapshotNodeTracker(['car'], frame_dt_s=0.5)
    raw = raw_object([0.1,0.2,0.3], [0,0,0])
    raw['stable_object_id']='a'
    a,_ = tr.update([raw], 0, timestamp_s=100.0)
    assert a[0].motion_state == 'unknown'
    moved = raw_object([0.9,0.2,0.3], [1,0,0])
    moved['stable_object_id']='a'
    b,_ = tr.update([moved], 2, timestamp_s=100.5)
    assert b[0].velocity == 2.0 and b[0].motion_state == 'dynamic'
    assert b[0].metadata['frame_displacement_m'] == 1
    with pytest.raises(ValueError):
        tr.update([copy.deepcopy(moved)], 3, timestamp_s=100.5)
