from dataclasses import asdict
import math
import time

import numpy as np
from shapely.geometry import LineString, Polygon, box, mapping

from conceptgraph.completion.reference_reachability import integrate_reference
from conceptgraph.completion.reference_reachability_audit import (
    audit_body_rows, audit_control_rows, audit_directed_rows, audit_terminal_rows,
)
from conceptgraph.navigation.vehicle_geometry import VehicleSpec, VehicleState, step_vehicle


def test_straight_reference_controls_reach_small_region_and_independent_replay_accepts():
    context = dict(drivable_geojson=mapping(box(-5, -2, 20, 2)),
        lane_union_geojson=mapping(box(-5, -2, 20, 2)), objects=[])
    region = dict(polygon_xy=list(box(7.8, -.1, 8.2, .1).exterior.coords),
        heading_rad=0., max_heading_error_rad=math.radians(5), nominal_rear_axle_xy=[8., 0.])
    rows, info = integrate_reference(VehicleState(0, 0, 0), LineString([[0, 0], [8, 0]]),
        region, context, time.monotonic()+10.)
    assert rows is not None and info['held_steps'] == 10
    assert not audit_control_rows(rows, asdict(VehicleSpec()), np.eye(4))
    assert not audit_body_rows(rows, asdict(VehicleSpec()), box(-5, -2, 20, 2),
        box(-5, -2, 20, 2), Polygon())


def test_independent_control_replay_rejects_teleport_and_wrong_rear_axle():
    a = VehicleState(0, 0, 0)
    b = step_vehicle(a, 1., 0.)
    rows = [dict(state=asdict(a), control=None)]
    for _ in range(12):
        rows.append(dict(state=asdict(b), control=dict(acceleration_mps2=1., steering_command_rad=0.)))
        a, b = b, step_vehicle(b, 1., 0.)
    rows[5]['state']['x'] += 1.
    assert 'saved_state_not_obtained_by_control_integration' in audit_control_rows(rows, asdict(VehicleSpec()), np.eye(4))
    pose = np.eye(4)
    pose[0, 3] = 1.
    assert 'initial_state_does_not_match_known_rear_axle_at_rest' in audit_control_rows(rows, asdict(VehicleSpec()), pose)


def test_full_body_audit_keeps_holes_and_all_non_target_obstacles():
    rows = [dict(state=asdict(VehicleState(0, 0, 0)), control=None)]
    outer = box(-5, -3, 10, 3)
    road = Polygon(outer.exterior.coords, [list(box(2., -.2, 2.2, .2).exterior.coords)])
    assert audit_body_rows(rows, asdict(VehicleSpec()), road, outer, Polygon())
    assert audit_body_rows(rows, asdict(VehicleSpec()), outer, outer, box(1., -.2, 1.2, .2))


def test_directed_audit_rejects_missing_connection_and_reverse_heading():
    lanes = dict(a=dict(polygon=box(-1, -2, 1, 2), samples=np.array([[0., 0., 0.]])),
        b=dict(polygon=box(1, -2, 3, 2), samples=np.array([[2., 0., 0.]])))
    rows = [dict(state=asdict(VehicleState(x, 0, 0)), control=None) for x in [0., .5, 1., 1.5, 2.]]
    assert not audit_directed_rows(rows, ['a', 'b'], lanes, {'a': {'outgoing': ['b']}}, .4)
    assert audit_directed_rows(rows, ['a', 'b'], lanes, {}, .4) == ['undeclared_official_lane_connection']
    rows[2]['state']['yaw_rad'] = math.pi
    assert audit_directed_rows(rows, ['a', 'b'], lanes, {'a': {'outgoing': ['b']}}, .4)


def test_terminal_duration_counts_intervals_and_respects_original_horizon():
    region = dict(polygon_xy=list(box(-.2, -.1, .2, .1).exterior.coords),
        heading_rad=0., max_heading_error_rad=.08)
    rows = [dict(state=asdict(VehicleState(0, 0, 0, time_s=i*.1))) for i in range(21)]
    policy = dict(hold_steps=21, minimum_hold_duration_s=2., maximum_reference_time_s=90.)
    assert not audit_terminal_rows(rows, region, policy)
    short_policy = dict(policy, hold_steps=20)
    assert 'terminal_hold_duration_too_short' in audit_terminal_rows(rows, region, short_policy)
    for row in rows:
        row['state']['time_s'] += 89.
    assert 'reference_exceeds_declared_horizon' in audit_terminal_rows(rows, region, policy)


def test_independent_replay_rejects_nonfinite_commands():
    rows = [dict(state=asdict(VehicleState(0, 0, 0, time_s=i*.1)),
        control=dict(acceleration_mps2=0., steering_command_rad=0.)) for i in range(21)]
    rows[5]['control']['acceleration_mps2'] = float('nan')
    assert 'nonfinite_control' in audit_control_rows(rows, asdict(VehicleSpec()), np.eye(4))
