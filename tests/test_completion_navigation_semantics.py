import json
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
from shapely.geometry import LineString, box

from conceptgraph.completion.navigation_semantics import NavigationSemanticBridge
from conceptgraph.completion.stage9 import digest
from conceptgraph.navigation.semantic import parse_navigation_query
from conceptgraph.navigation.state_machine import NavigationStateMachine
from conceptgraph.navigation.types import CandidateTarget, SceneGraphNode
from conceptgraph.navigation.vehicle_geometry import VehicleState, step_vehicle
from conceptgraph.navigation.vehicle_planner import VehiclePlan


def envelope(output):
    return dict(raw_output=json.dumps(output),used_qwen=True,fallback_used=False)


def wait_response(bridge):
    deadline = time.monotonic()+2
    while bridge.queue.completed.empty():
        assert time.monotonic()<deadline
        time.sleep(.002)


def agent_fixture():
    nodes = {i:SceneGraphNode(i,'car',['car'],'car',np.array([float(i),5.,1.]),
        np.ones(3),motion_state='unknown',metadata=dict(object_identity=f'track-{i}',num_detections=3))
        for i in (7,9)}
    return SimpleNamespace(nodes=nodes,parsed=parse_navigation_query('去小汽车附近',8),
        machine=NavigationStateMachine(),selected=7,path=[[0,0],[10,0]],plan_target=7,
        approach_goal=lambda i:(np.array([float(i),0.]),0.),road=box(-10,-10,30,10),
        obstacles=[],line=LineString([(0,0),(30,0)]))


def test_only_current_safe_candidates_reach_model_and_no_match_clears_existing_lock(tmp_path,monkeypatch):
    calls = []
    class Delegate:
        def rerank_candidates(self,query,candidates,edges,**kw):
            calls.append(candidates)
            return envelope(dict(schema_version='1.0.0',selected_node_id=None,
                ranking=[7],no_match=True,explanation='no semantic match',confidence=.9))
    def planner(state,goal,*args,**kwargs):
        if goal[0]==9:
            return VehiclePlan([],1,.001,'PLANNING_TIME_BUDGET')
        return VehiclePlan([state,step_vehicle(state,1,0)],1,.001)
    monkeypatch.setattr('conceptgraph.completion.navigation_semantics.hybrid_astar',planner)
    bridge = NavigationSemanticBridge('去小汽车附近',tmp_path,Delegate(),tasks={'candidate_rerank'})
    agent = agent_fixture()
    candidates = [CandidateTarget(i,1,1,0,0) for i in (7,9)]
    try:
        selected,_,audit = bridge.select(agent,candidates,[],VehicleState(0,0,0),0)
        assert [c.node_id for c in selected]==[7]
        assert not audit['candidate_checks'][1]['passed']
        wait_response(bridge)
        bridge.prepare(agent,1)
        selected,_,_ = bridge.select(agent,candidates,[],VehicleState(0,0,0),1)
        assert [c['node_id'] for c in calls[0]]==[7]
        assert selected==[] and agent.path==[] and agent.selected is None
        assert bridge.applied['candidate_rerank']==1
    finally:
        bridge.close()


def test_geometry_change_rejects_previous_ranking_without_credit(tmp_path,monkeypatch):
    class Delegate:
        def rerank_candidates(self,*args,**kwargs):
            return envelope(dict(schema_version='1.0.0',selected_node_id=None,
                ranking=[7],no_match=True,explanation='old evidence',confidence=.9))
    monkeypatch.setattr('conceptgraph.completion.navigation_semantics.hybrid_astar',
        lambda state,*a,**k:VehiclePlan([state,step_vehicle(state,1,0)],1,.001))
    bridge = NavigationSemanticBridge('去小汽车附近',tmp_path,Delegate(),tasks={'candidate_rerank'})
    agent = agent_fixture()
    candidates = [CandidateTarget(7,1,1,0,0)]
    try:
        bridge.select(agent,candidates,[],VehicleState(0,0,0),0)
        wait_response(bridge)
        agent.nodes[7].bbox_center[0] += 2.
        bridge.prepare(agent,1)
        selected,_,_ = bridge.select(agent,candidates,[],VehicleState(0,0,0),1)
        assert [c.node_id for c in selected]==[7]
        assert bridge.applied['candidate_rerank']==0
        assert any(e['kind']=='ranking_context_checked' and not e['applied'] for e in bridge.events)
    finally:
        bridge.close()


def test_real_past_crop_attribute_never_overwrites_category_motion_or_geometry(tmp_path):
    class Delegate:
        def describe_object(self,paths,metadata,**kwargs):
            assert paths==[str(crop)]
            return envelope(dict(schema_version='1.0.0',summary='red bus',object_tag='bus',
                possible_tags=['bus'],colors=['red'],body_type='unknown',motion_state='static',confidence=.9))
    crop = tmp_path/'past_crop.png'
    Image.new('RGB',(28,28),'red').save(crop)
    bridge = NavigationSemanticBridge('去红色车旁边',tmp_path,Delegate(),tasks={'object_attributes'})
    agent = agent_fixture()
    try:
        bridge.queue.submit('object_attributes',dict(node_id=7,crop_paths=[str(crop)],crop_sha256=digest(crop),
            metadata=dict(detection_label='car'),frame_idx=0),{'object_7':['track-7','car']},ttl_s=2.)
        wait_response(bridge)
        before = agent.nodes[7].bbox_center.copy()
        bridge.prepare(agent,1)
        node = agent.nodes[7]
        assert node.metadata['attributes']['colors']==['red']
        assert node.object_tag=='car' and node.motion_state=='unknown'
        np.testing.assert_array_equal(node.bbox_center,before)
        assert node.metadata['qwen_category_advisory']=='bus'
        assert node.metadata['qwen_attributes']['crop_sha256']==digest(crop)
        # 即使复用同一节点编号，新的轨迹身份也不能继承旧属性。
        node.metadata['object_identity'] = 'track-new'
        node.metadata.pop('attributes')
        bridge.prepare(agent,2)
        assert 'attributes' not in node.metadata and 7 not in bridge.attributes
    finally:
        bridge.close()
