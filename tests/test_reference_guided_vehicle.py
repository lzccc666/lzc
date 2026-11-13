import math

import numpy as np
import pytest
from shapely.geometry import LineString, box

from conceptgraph.completion.topology_vehicle import DirectedVehicleCorridor
from conceptgraph.navigation.reference_guided_vehicle import reference_guided_vehicle
from conceptgraph.navigation.vehicle_geometry import VehicleState, audit_executed_states


def straight_corridor(*, obstacles=(), drivable=None):
    def lane(token, lo, hi):
        return dict(token=token, polygon=list(box(lo, -2, hi, 2).exterior.coords),
                    centerline=[[lo, 0], [hi, 0]], start_pose=[lo, 0, 0], end_pose=[hi, 0, 0])
    lanes = [lane('entry', -10, 8), lane('exit', 7.9, 30)]
    return DirectedVehicleCorridor(lanes, {'entry': ['exit']},
                                   box(-10, -2, 30, 2) if drivable is None else drivable, list(obstacles))


def curved_corridor():
    def centerline(lo, hi):
        angles = np.linspace(lo, hi, 160)
        return np.column_stack([30*np.sin(angles), 30*(1-np.cos(angles))])
    def lane(token, lo, hi):
        points = centerline(lo, hi)
        return dict(token=token, polygon=list(LineString(points).buffer(2, cap_style=2).exterior.coords),
                    centerline=points.tolist(), start_pose=[*points[0], lo], end_pose=[*points[-1], hi])
    lanes = [lane('entry', -.4, .26), lane('exit', .24, 1.)]
    return (DirectedVehicleCorridor(lanes, {'entry': ['exit']}, box(-50, -50, 50, 50), []),
            LineString(centerline(-.4, 1.)))


def test_curved_route_uses_limited_kinematics_and_declared_connection():
    corridor, reference = curved_corridor()
    goal = (30*math.sin(.5), 30*(1-math.cos(.5)))
    plan = reference_guided_vehicle(VehicleState(0, 0, 0), goal, .5, corridor, reference,
                                   budget_s=5, max_expanded=1000)
    assert plan.failure is None and len(plan.states) > 50
    assert math.dist((plan.states[-1].x, plan.states[-1].y), goal) <= .65
    assert audit_executed_states(plan.states, corridor.road, corridor.obstacles)['passed']
    stages = {0}
    for before, after in zip(plan.states, plan.states[1:]):
        checked = corridor.validate_step(before, after, stages)
        assert checked['legal']
        stages = set(checked['next_stages'])
    assert 1 in stages and plan.states[-1].yaw_rad > .25


@pytest.mark.parametrize('kind', ['obstacle', 'road_hole'])
def test_partial_guidance_cannot_return_a_route_through_blocking_geometry(kind):
    barrier = box(6, -2, 7, 2) if kind == 'obstacle' else box(6, -.7, 7, .7)
    corridor = straight_corridor(obstacles=[barrier] if kind == 'obstacle' else [],
        drivable=box(-10, -2, 30, 2).difference(barrier) if kind == 'road_hole' else None)
    plan = reference_guided_vehicle(VehicleState(0, 0, 0), (15, 0), 0, corridor,
                                   LineString([[-10, 0], [30, 0]]), budget_s=.25)
    assert not plan.states and plan.failure in {'PLANNING_TIME_BUDGET', 'NO_FEASIBLE_PRIMITIVE_ROUTE'}
    assert plan.diagnostics['rejected_steps'] > 0


def test_invalid_start_body_cannot_be_accepted_from_reference_proximity():
    plan = reference_guided_vehicle(VehicleState(0, 1.9, 0), (15, 0), 0, straight_corridor(),
                                   LineString([[-10, 0], [30, 0]]))
    assert plan.failure == 'INVALID_INITIAL_BODY_OR_ENTRY_LANE' and not plan.states and plan.expanded == 0


def test_initial_polygon_overlap_cannot_skip_the_entry_identity():
    plan = reference_guided_vehicle(VehicleState(20, 0, 0), (22, 0), 0, straight_corridor(),
                                   LineString([[-10, 0], [30, 0]]))
    assert plan.failure == 'INVALID_INITIAL_BODY_OR_ENTRY_LANE' and not plan.states


def test_guidance_counts_toward_the_expansion_budget():
    plan = reference_guided_vehicle(VehicleState(0, 0, 0), (15, 0), 0, straight_corridor(),
                                   LineString([[-10, 0], [30, 0]]), max_expanded=1)
    assert plan.expanded == 1 and plan.failure == 'EXPANSION_BUDGET' and not plan.states


@pytest.mark.parametrize('budget,count', [(0, 1), (float('nan'), 1), (1, 0)])
def test_invalid_budget_is_rejected(budget, count):
    with pytest.raises(ValueError, match='预算'):
        reference_guided_vehicle(VehicleState(0, 0, 0), (15, 0), 0, straight_corridor(),
                                 LineString([[-10, 0], [30, 0]]), budget_s=budget, max_expanded=count)


def test_undeclared_chain_is_rejected_before_search():
    corridor = straight_corridor()
    with pytest.raises(ValueError, match='未声明'):
        DirectedVehicleCorridor(corridor.lanes, {}, corridor.road, [])
