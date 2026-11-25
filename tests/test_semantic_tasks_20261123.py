"""Safety boundary tests use synthetic providers, never real navigation scores."""
from copy import deepcopy
from dataclasses import replace
import math
import unittest

from conceptgraph.navigation.semantic_tasks_20261123 import (
    Entity, Evidence, MonitorState, footprint_in_region, monitor_task, parse_task_spec, propose_task,
)
from scripts.semantic_tasks_20261123_demo import fixture_evidence, synthetic_graph


def node(graph, identity):
    return next(n for n in graph.entities if n.entity_id == identity)


def parking():
    graph = synthetic_graph()
    spec = parse_task_spec("停到最近空闲车位")
    return graph, spec


def telemetry(now, **values):
    kinds = {"execution_observed": "telemetry", "collision_free": "telemetry_audit",
             "terminal_region_still_allowed_and_free": "terminal_audit", "tracking_identity": "tracking",
             "user_end": "user_command"}
    return Entity("observed", "telemetry", properties={
        name: Evidence(value, "synthetic_fixture:observed_test_not_real_vehicle", now, 0.1,
                       source_kind=kinds.get(name, "telemetry")) for name, value in values.items()})


def terminal(now, **changes):
    values = {"execution_observed": True, "collision_free": True, "position_xy": [22, 5],
              "heading_rad": 0.0, "speed_mps": 0.0, "dimensions_lw_m": [4.5, 1.8],
              "terminal_region_still_allowed_and_free": True}
    values.update(changes)
    return telemetry(now, **values)


def follow_observation(now, **changes):
    values = {"execution_observed": True, "collision_free": True, "tracking_identity": "track-verified-A",
              "same_lane_front": True, "bumper_gap_m": 20.5, "ego_speed_mps": 5.0,
              "lead_speed_mps": 5.0, "user_end": False}
    values.update(changes)
    return telemetry(now, **values)


def test_three_tasks_same_graph_different_target_representations():
    graph = synthetic_graph()
    results = [propose_task(parse_task_spec(s), graph) for s in ["靠边停车", "停到最近空闲车位", "跟车行驶"]]
    assert [r["target"]["entity_id"] for r in results] == ["pull-region", "space-min-path", "front-vehicle"]
    assert all(not r["execution_authorized"] for r in results)
    assert results[2]["target"]["fixed_endpoint"] is None


def test_nearest_filters_occupied_then_uses_path_cost_not_euclidean():
    graph, spec = parking()
    result = propose_task(spec, graph)
    assert result["target"]["entity_id"] == "space-min-path"
    assert "space-near-occupied" not in result["details"]["eligible_ids"]
    node(graph, "space-min-path").properties["reachable"] = fixture_evidence(False, "planner")
    assert propose_task(spec, graph)["target"]["entity_id"] == "space-short-euclidean"


def test_unknown_occupancy_cannot_create_a_parking_proposal():
    graph, spec = parking()
    for n in graph.entities:
        if n.kind == "parking_space":
            n.properties["occupancy"] = fixture_evidence(None, "perception")
    assert propose_task(spec, graph)["target"] is None


def test_string_false_nan_stale_conflict_and_language_permissions_rejected():
    graph, spec = parking()
    for bad in [fixture_evidence("false", "regulatory_map"), fixture_evidence("unknown", "regulatory_map"),
                Evidence(True, "old_regulatory_map", 90, 1, source_kind="regulatory_map"),
                Evidence(True, "qwen:invented_permission", 100, 5, source_kind="regulatory_map"),
                Evidence(True, "model_reply", 100, 5, source_kind="language"),
                Evidence(True, "conflicting_map", 100, 5, conflict=True, source_kind="regulatory_map")]:
        current = deepcopy(graph)
        for n in current.entities:
            if n.kind == "parking_space":
                n.properties["allowed"] = bad
        assert propose_task(spec, current)["target"] is None
    for n in graph.entities:
        if n.kind == "parking_space":
            n.properties["path_cost_m"] = fixture_evidence(float("nan"), "planner")
    assert propose_task(spec, graph)["target"] is None


def test_future_nan_and_boolean_evidence_times_rejected():
    for bad in [Evidence(True, "map", 101, 5), Evidence(True, "map", float("nan"), 5),
                Evidence(True, "map", True, 500), Evidence(True, "map", 100, 5, confidence=float("nan"))]:
        assert bad.read(100)[0] is None


def test_vehicle_size_and_nonconvex_terminal_geometry_are_not_silent_passes():
    graph, spec = parking()
    for n in graph.entities:
        if n.kind == "parking_space":
            n.properties["dimensions_lw_m"] = fixture_evidence([3, 1])
    assert propose_task(spec, graph)["target"] is None
    assert not footprint_in_region([0, 0], 0, [2, 1], [(-3, -3), (3, 3), (-3, 3), (3, -3)])


def test_declared_curb_adjacency_requires_actual_geometry():
    graph = synthetic_graph()
    curb = node(graph, "curb-A")
    curb.polygon_xy = [(100, 100), (102, 100), (102, 102), (100, 102)]
    assert propose_task(parse_task_spec("靠边停车"), graph)["target"] is None


def test_unsourced_geometry_cannot_become_safe_region_or_front_vehicle():
    graph, spec = parking()
    for n in graph.entities:
        if n.kind == "parking_space":
            n.geometry_evidence = None
    assert propose_task(spec, graph)["target"] is None
    graph = synthetic_graph()
    node(graph, "front-vehicle").geometry_evidence = None
    assert propose_task(parse_task_spec("跟车行驶"), graph)["target"] is None


def test_reverse_unknown_rear_and_unobserved_vehicle_sweep_reject_parking():
    graph, spec = parking()
    for n in graph.entities:
        if n.kind == "parking_space":
            n.properties["requires_reverse"] = fixture_evidence(True, "planner")
    assert propose_task(spec, graph)["target"] is None
    for n in graph.entities:
        if n.kind == "parking_space":
            n.properties["rear_observed_free"] = fixture_evidence(True, "perception_audit")
            n.properties["maneuver_observed_free"] = fixture_evidence(None, "perception_audit")
    assert propose_task(spec, graph)["target"] is None


def test_carpark_area_and_drivable_do_not_become_individual_spaces():
    graph, spec = parking()
    graph.entities = [n for n in graph.entities if n.kind != "parking_space"]
    graph.entities.append(Entity("a_drivable_area", "carpark_area", (10, 0)))
    assert propose_task(spec, graph)["target"] is None


def test_wrong_side_direction_and_unsourced_curb_rejected():
    graph = synthetic_graph()
    spec = parse_task_spec("右侧靠边停车")
    assert propose_task(spec, graph)["target"]["entity_id"] == "pull-region"
    pull = node(graph, "pull-region")
    pull.properties["travel_heading_rad"] = fixture_evidence(math.pi)
    assert propose_task(spec, graph)["target"] is None
    pull.properties["travel_heading_rad"] = fixture_evidence(0)
    pull.properties["side_of_travel"] = fixture_evidence("left")
    assert propose_task(spec, graph)["target"] is None
    pull.properties["side_of_travel"] = fixture_evidence("right")
    pull.properties["terminal_heading_rad"] = fixture_evidence(math.pi)
    assert propose_task(spec, graph)["target"] is None
    pull.properties["terminal_heading_rad"] = fixture_evidence(0.0)
    del pull.properties["allowed"]
    assert propose_task(spec, graph)["target"] is None


def test_follow_never_chases_farther_verified_car_past_unknown_lead():
    graph = synthetic_graph()
    lead = node(graph, "front-vehicle")
    nearer = deepcopy(lead)
    nearer.entity_id = "nearer-unknown-car"
    nearer.center_xy = (17, 0)
    nearer.properties["lane_station_m"] = fixture_evidence(37)
    nearer.properties["continuous_identity_verified"] = fixture_evidence(None, "tracking_audit")
    graph.entities.append(nearer)
    result = propose_task(parse_task_spec("跟车行驶"), graph)
    assert result["target"] is None
    assert result["details"]["nearest_observed_lead"] == nearer.entity_id


def test_follow_stationary_ego_does_not_complete_parking_or_follow_by_horizon():
    graph = synthetic_graph()
    spec = parse_task_spec("跟车行驶")
    decision = propose_task(spec, graph)
    state = MonitorState(100)
    assert monitor_task(spec, decision, state, follow_observation(100), 100)["status"] == "following_under_observation"
    assert monitor_task(spec, decision, state, follow_observation(220), 220)["status"] == "failed_timeout"
    assert spec.termination["parking_success"] is False


def test_parking_requires_position_heading_speed_and_continuous_hold():
    graph, spec = parking()
    decision = propose_task(spec, graph)
    state = MonitorState(100)
    assert monitor_task(spec, decision, state, terminal(100, speed_mps=1), 100)["status"] == "terminal_conditions_not_met"
    assert monitor_task(spec, decision, state, terminal(101, heading_rad=math.pi), 101)["status"] == "terminal_conditions_not_met"
    assert monitor_task(spec, decision, state, terminal(102, position_xy=[0, 0]), 102)["status"] == "terminal_conditions_not_met"
    assert monitor_task(spec, decision, state, terminal(103), 103)["status"] == "holding_terminal_conditions"
    assert monitor_task(spec, decision, state, terminal(104), 104)["status"] == "holding_terminal_conditions"
    assert monitor_task(spec, decision, state, terminal(105), 105)["status"] == "terminal_conditions_observed"


def test_missing_execution_or_gapped_observations_cannot_complete():
    graph, spec = parking()
    decision = propose_task(spec, graph)
    state = MonitorState(100)
    assert monitor_task(spec, decision, state, terminal(100, execution_observed=False), 100)["status"] == "need_execution_observation"
    monitor_task(spec, decision, state, terminal(101), 101)
    assert monitor_task(spec, decision, state, terminal(104), 104)["status"] == "holding_terminal_conditions"


def test_follow_identity_switch_and_gap_violations_are_separate_latched_failures():
    graph = synthetic_graph()
    spec = parse_task_spec("跟车行驶10秒")
    decision = propose_task(spec, graph)
    state = MonitorState(100)
    monitor_task(spec, decision, state, follow_observation(100), 100)
    assert monitor_task(spec, decision, state, follow_observation(101, tracking_identity="wrong-car"), 101)["status"] == "failed_identity_switch"
    assert monitor_task(spec, decision, state, follow_observation(102), 102)["status"] == "failed_identity_switch"
    state = MonitorState(100)
    assert monitor_task(spec, decision, state, follow_observation(100, bumper_gap_m=2), 100)["status"] == "failed_follow_distance_or_speed_constraint"


def test_lost_front_vehicle_is_not_a_parking_or_follow_success():
    graph = synthetic_graph()
    spec = parse_task_spec("跟车行驶3秒")
    decision = propose_task(spec, graph)
    state = MonitorState(100)
    monitor_task(spec, decision, state, follow_observation(100), 100)
    result = monitor_task(spec, decision, state, follow_observation(101, same_lane_front=None), 101)
    assert result["status"] == "lost_or_unknown_lead"
    assert state.valid_since_s is None


def test_requested_follow_duration_requires_all_observed_interval():
    graph = synthetic_graph()
    spec = parse_task_spec("跟车行驶3秒")
    decision = propose_task(spec, graph)
    state = MonitorState(100)
    for t in range(100, 103):
        assert monitor_task(spec, decision, state, follow_observation(t), t)["status"] == "following_under_observation"
    assert monitor_task(spec, decision, state, follow_observation(103), 103)["status"] == "requested_safe_follow_interval_observed"


def test_unsupported_negation_color_or_mutated_contract_needs_clarification():
    for text in ["不要跟车靠边停车", "停到白车旁", "跟车行驶0秒", "一直往前开"]:
        assert parse_task_spec(text).parse_status == "need_clarification"
    graph, spec = parking()
    mutated = replace(spec, constraints={**spec.constraints, "maximum_terminal_speed_mps": 100})
    assert propose_task(mutated, graph)["status"] == "need_clarification"


def load_tests(loader, standard_tests, pattern):
    """Also runnable with the standard library when pytest is not installed."""
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in sorted(globals().items())
                              if name.startswith("test_") and callable(value))
