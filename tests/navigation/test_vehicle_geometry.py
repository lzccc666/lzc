import math

import pytest
from shapely.geometry import box

from conceptgraph.navigation.vehicle_geometry import (
    VehicleSpec, VehicleState, step_vehicle, footprint_polygon,
    body_is_legal, swept_body_polygon, audit_executed_states,
)


def test_controls_respect_acceleration_steering_rate_and_forward_only():
    spec=VehicleSpec()
    initial=VehicleState(0,0,0)
    after=step_vehicle(initial,100,100,spec)
    assert after.speed_mps==pytest.approx(.1)
    assert after.steering_rad==pytest.approx(math.radians(2.5))
    assert after.x>0 and after.time_s==pytest.approx(.1)
    for _ in range(100):
        after=step_vehicle(after,-100,-100,spec)
    assert after.speed_mps==0
    assert abs(after.steering_rad)<=spec.max_steering_rad


def test_center_clear_does_not_hide_rectangular_body_collision():
    state=VehicleState(0,0,0)
    obstacle=box(3,.7,3.5,1.2)
    assert not obstacle.contains(footprint_polygon(state).centroid)
    assert not body_is_legal(state,box(-10,-10,10,10),[obstacle])
    assert not body_is_legal(state,box(-10,-.5,10,.5),[])


def test_sweep_catches_obstacle_between_clear_endpoint_bodies():
    spec=VehicleSpec(front_from_rear_axle_m=1,wheelbase_m=.8,rear_overhang_m=.2,width_m=.4)
    before=VehicleState(0,0,0)
    after=VehicleState(4,0,0,time_s=.1)
    obstacle=box(2,-.1,2.1,.1)
    assert not footprint_polygon(before,spec).intersects(obstacle)
    assert not footprint_polygon(after,spec).intersects(obstacle)
    assert swept_body_polygon(before,after,spec).intersects(obstacle)
    audit=audit_executed_states([before,after],box(-10,-10,10,10),[obstacle],spec)
    assert not audit['passed']
    assert {'position_jump','swept_collision_or_boundary'} <= {r['reason'] for r in audit['violations']}


def test_real_bicycle_updates_pass_independent_execution_audit():
    states=[VehicleState(0,0,0)]
    for _ in range(50):
        states.append(step_vehicle(states[-1],.5,.2))
    audit=audit_executed_states(states,box(-30,-30,30,30),[])
    assert audit['passed'] and audit['audited_intervals']==50
    assert states[-1].y>0 and states[-1].yaw_rad>0
