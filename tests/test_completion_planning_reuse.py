from shapely.geometry import LineString, box

from conceptgraph.completion.planning_reuse import revalidate_route
from conceptgraph.completion.kinematic_benchmark import SPEC
from conceptgraph.navigation.vehicle_geometry import VehicleState, audit_executed_states


def test_cached_route_uses_current_state_and_integrates_real_control():
    start=VehicleState(2,0,0,speed_mps=.5,time_s=10.)
    road=box(-10,-5,30,5)
    line=LineString([(0,0),(20,0)])
    plan,reason=revalidate_route(start,[[0,0],[12,0]],[12,0],0,road,[],line,SPEC)
    assert plan is not None and reason=='current_state_and_obstacles_revalidated'
    assert plan.states[0] is start and plan.states[-1].time_s>start.time_s
    assert plan.states[-1].speed_mps<=.15 and abs(plan.states[-1].x-12)<=.65
    assert audit_executed_states(plan.states,road,[],SPEC)['passed']


def test_new_obstacle_blocks_cached_route_and_changed_endpoint_rejects():
    start=VehicleState(0,0,0)
    road=box(-10,-5,30,5)
    line=LineString([(0,0),(20,0)])
    plan,reason=revalidate_route(start,[[0,0],[12,0]],[12,0],0,road,[box(6,-2,7,2)],line,SPEC)
    assert plan is None and reason=='current_body_or_swept_motion_invalid'
    plan,reason=revalidate_route(start,[[0,0],[12,0]],[14,0],0,road,[],line,SPEC)
    assert plan is None and reason=='route_endpoint_no_longer_matches_goal'
