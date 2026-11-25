"""Evidence-bounded semantic task contracts, offline proposals, and task monitoring.

This independent prototype does not modify the navigation pipeline and never
authorizes vehicle execution. Rule parsing is not a Qwen/model result. All
permission, occupancy, identity, and route checks must carry real provenance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any

ACTIONS = ("pull_over", "park_nearest", "follow_vehicle", "unsupported")
SPATIAL_RELATIONS = ("near", "left_of", "right_of", "in_front_of", "behind")


@dataclass(frozen=True)
class Evidence:
    value: Any
    source: str
    observed_s: float
    valid_for_s: float
    confidence: float = 1.0
    conflict: bool = False
    source_kind: str = "unknown"

    def read(self, now_s: float) -> tuple[Any, str | None]:
        numbers = (self.observed_s, self.valid_for_s, self.confidence, now_s)
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in numbers):
            return None, "invalid_evidence_time_or_confidence"
        if not self.source or self.conflict or not 0 < self.confidence <= 1:
            return None, "conflicting_or_unsourced_evidence"
        if self.observed_s > now_s + 1e-6 or self.valid_for_s < 0:
            return None, "future_or_invalid_evidence"
        if now_s - self.observed_s > self.valid_for_s + 1e-6:
            return None, "stale_evidence"
        if self.value is None:
            return None, "unknown_evidence"
        return self.value, None


@dataclass
class Entity:
    entity_id: str
    kind: str
    center_xy: tuple[float, float] | None = None
    polygon_xy: list[tuple[float, float]] = field(default_factory=list)
    properties: dict[str, Evidence] = field(default_factory=dict)
    geometry_evidence: Evidence | None = None

    def get(self, name: str, now_s: float) -> tuple[Any, str | None]:
        evidence = self.properties.get(name)
        return evidence.read(now_s) if evidence else (None, "missing_evidence")


@dataclass
class TaskGraph:
    now_s: float
    entities: list[Entity]
    ego: Entity
    source_scope: str

    def __post_init__(self):
        ids = [self.ego.entity_id, *(e.entity_id for e in self.entities)]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate entity identity")
        if not math.isfinite(self.now_s):
            raise ValueError("invalid graph time")


@dataclass(frozen=True)
class TaskSpec:
    action: str
    instruction: str
    target_kind: str
    goal_mode: str
    timeout_s: float
    requested_duration_s: float | None = None
    side: str | None = None
    parser: str = "deterministic_rule_prototype"
    parse_status: str = "supported"
    target_conditions: tuple[str, ...] = ()
    constraints: dict = field(default_factory=dict)
    termination: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def parse_task_spec(instruction: str) -> TaskSpec:
    """Supported phrases only; unsupported modifiers never silently disappear."""
    text = re.sub(r"[，。！？、\s]", "", instruction)
    text = re.sub(r"^(请|麻烦)", "", text)
    action = "unsupported"
    duration = None
    side = None
    if re.fullmatch(r"(向|在)?(左侧|右侧)?(安全)?靠边停车", text):
        action = "pull_over"
        side = "left" if "左侧" in text else "right" if "右侧" in text else None
    elif re.fullmatch(r"(停到|驶入|泊入)(最近的?)?(空闲)?(停车位|车位)", text):
        action = "park_nearest"
    elif re.fullmatch(r"(跟车行驶|跟随(本车道)?前车)(\d+(\.\d+)?秒)?", text):
        action = "follow_vehicle"
        match = re.search(r"(\d+(\.\d+)?)秒", text)
        duration = float(match[1]) if match else None
        if duration is not None and not 0 < duration <= 600:
            action = "unsupported"
    if action == "unsupported":
        return TaskSpec(action, instruction, "unknown", "clarification", 0,
                        parse_status="need_clarification",
                        target_conditions=("unsupported_or_ambiguous_instruction",))
    if action == "follow_vehicle":
        return TaskSpec(action, instruction, "vehicle", "persistent_relation_policy",
                        max(120.0, (duration or 0) + 10), duration, side,
                        target_conditions=("same_lane_front", "continuous_identity", "fresh_relative_state"),
                        constraints={"minimum_gap_m": 3.0, "minimum_headway_s": 1.8,
                                     "minimum_ttc_s": 3.0, "maximum_relative_speed_mps": 10.0,
                                     "maximum_observation_gap_s": 1.0},
                        termination={"mode": "requested_safe_duration_or_user_end",
                                     "fixed_endpoint": None, "parking_success": False})
    return TaskSpec(action, instruction, "stopping_region" if action == "pull_over" else "parking_space",
                    "terminal_region_pose", 120.0 if action == "pull_over" else 180.0,
                    side=side, target_conditions=("allowed", "free", "vehicle_fits", "qualified_path"),
                    constraints={"clearance_each_side_m": 0.25, "maximum_terminal_speed_mps": 0.2,
                                 "maximum_heading_error_rad": math.radians(10),
                                 "maximum_observation_gap_s": 1.0},
                    termination={"mode": "full_footprint_in_region_heading_speed_hold",
                                 "hold_s": 2.0, "actual_execution_observation_required": True})


def _number(value, minimum=None):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and (minimum is None or value >= minimum))


def _xy(value):
    return isinstance(value, (tuple, list)) and len(value) == 2 and all(_number(v) for v in value)


def _angle(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _convex(polygon):
    if len(polygon) < 3 or not all(_xy(p) for p in polygon):
        return False
    signs = [_cross(a, b, p) for a, b in zip(polygon, polygon[1:] + polygon[:1]) for p in polygon]
    return (all(t >= -1e-8 for t in signs) or all(t <= 1e-8 for t in signs)) and any(abs(t) > 1e-8 for t in signs)


def point_in_polygon(point, polygon):
    """Boundary is included; polygon must be finite and simple at provider boundary."""
    if not _xy(point) or len(polygon) < 3 or not all(_xy(p) for p in polygon):
        return False
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if abs(_cross(a, b, point)) < 1e-8 and all(min(a[i], b[i]) - 1e-8 <= point[i] <= max(a[i], b[i]) + 1e-8 for i in range(2)):
            return True
        if (a[1] > point[1]) != (b[1] > point[1]):
            x = (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]
            if point[0] < x:
                inside = not inside
    return inside


def footprint_in_region(center, yaw, dimensions, polygon):
    """Convex terminal regions only; concave shapes need verified decomposition."""
    if not (_convex(polygon) and _xy(center) and _number(yaw) and _xy(dimensions)
            and all(v > 0 for v in dimensions)):
        return False
    length, width = dimensions
    c, s = math.cos(yaw), math.sin(yaw)
    corners = [(center[0] + c * dx - s * dy, center[1] + s * dx + c * dy)
               for dx in (-length / 2, length / 2) for dy in (-width / 2, width / 2)]
    return all(point_in_polygon(p, polygon) for p in corners)


def polygon_distance(a, b):
    """Convex polygon separation; None means geometry is unsupported/unknown."""
    if not _convex(a) or not _convex(b):
        return None
    if any(point_in_polygon(p, b) for p in a) or any(point_in_polygon(p, a) for p in b):
        return 0.0
    distances = []
    for first, second in ((a, b), (b, a)):
        for point in first:
            for start, end in zip(second, second[1:] + second[:1]):
                dx, dy = end[0] - start[0], end[1] - start[1]
                squared = dx * dx + dy * dy
                t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / squared)) if squared else 0.0
                distances.append(math.dist(point, (start[0] + t * dx, start[1] + t * dy)))
    # Crossing convex edges with no contained vertex (e.g. a plus-shaped overlap).
    for a0, a1 in zip(a, a[1:] + a[:1]):
        for b0, b1 in zip(b, b[1:] + b[:1]):
            if _cross(a0, a1, b0) * _cross(a0, a1, b1) < 0 and _cross(b0, b1, a0) * _cross(b0, b1, a1) < 0:
                return 0.0
    return min(distances)


def _read(entity, name, now, missing):
    value, error = entity.get(name, now)
    evidence = entity.properties.get(name)
    allowed = {
        "allowed": {"regulatory_map", "human_verified_permission"},
        "occupancy": {"perception", "human_verified_occupancy"},
        "reachable": {"planner"}, "path_cost_m": {"planner"}, "path_qualified": {"planner_audit"},
        "vehicle_sweep_qualified": {"planner_audit"}, "maneuver_observed_free": {"perception_audit"},
        "requires_reverse": {"planner"}, "rear_observed_free": {"perception_audit"},
        "continuous_identity_verified": {"tracking_audit"}, "tracking_identity": {"tracking", "telemetry"},
        "execution_observed": {"telemetry"}, "collision_free": {"telemetry_audit"},
        "terminal_region_still_allowed_and_free": {"terminal_audit"},
        "user_end": {"user_command"},
    }.get(name, {"geometry", "map", "perception", "tracking", "telemetry", "human_verified_geometry"})
    if evidence and (evidence.source_kind not in allowed or evidence.source.lower().startswith(("qwen", "llm", "rule_parse"))):
        value, error = None, "source_not_authoritative_for_property"
    if error:
        missing.append(f"{entity.entity_id}.{name}:{error}")
    return value


def _geometry_available(entity, now_s):
    evidence = entity.geometry_evidence
    if evidence is None or evidence.source_kind not in {"geometry", "map", "perception", "human_verified_geometry"}:
        return False
    value, error = evidence.read(now_s)
    return error is None and value is True and not evidence.source.lower().startswith(("qwen", "llm", "rule_parse"))


def _lane_state(entity, graph):
    missing = []
    lane = _read(entity, "lane_id", graph.now_s, missing)
    station = _read(entity, "lane_station_m", graph.now_s, missing)
    return lane, station, missing


def task_relations(graph: TaskGraph) -> list[dict]:
    """Only evidence-backed relations; never infer occupied/reachable/allowed."""
    result = []
    for entity in [graph.ego, *graph.entities]:
        if not _geometry_available(entity, graph.now_s):
            continue
        lane_id, station, errors = _lane_state(entity, graph)
        lane = next((n for n in graph.entities if n.entity_id == lane_id and n.kind == "lane"), None)
        if (not errors and lane and _geometry_available(lane, graph.now_s) and _number(station)
                and entity.center_xy and point_in_polygon(entity.center_xy, lane.polygon_xy)):
            result.append({"source": entity.entity_id, "target": lane_id, "relation": "belongs_to_lane",
                           "evidence": [asdict(entity.properties["lane_id"]), asdict(entity.properties["lane_station_m"])],
                           "scope": "center_membership_not_full_body_or_permission"})
        curb_missing = []
        curb = _read(entity, "curb_region_id", graph.now_s, curb_missing)
        curb_node = next((n for n in graph.entities if n.entity_id == curb and n.kind == "curb_region"), None)
        adjacency = _read(entity, "curb_adjacency_verified", graph.now_s, curb_missing)
        separation = polygon_distance(entity.polygon_xy, curb_node.polygon_xy) if curb_node else None
        if (curb_node and _geometry_available(curb_node, graph.now_s) and not curb_missing and adjacency is True
                and separation is not None and separation <= 0.25):
            result.append({"source": entity.entity_id, "target": curb, "relation": "stopping_region_associated_curb",
                           "evidence": [asdict(entity.properties["curb_region_id"]), asdict(entity.properties["curb_adjacency_verified"])],
                           "scope": "association_not_parking_permission"})
        if entity.kind in {"stopping_region", "parking_space"}:
            for region in graph.entities:
                if (region.entity_id == entity.entity_id or region.kind not in {"lane", "road_region", "curb_region"}
                        or not _geometry_available(region, graph.now_s)):
                    continue
                separation = polygon_distance(entity.polygon_xy, region.polygon_xy)
                if separation is not None and separation <= 0.25:
                    result.append({"source": entity.entity_id, "target": region.entity_id,
                                   "relation": "region_adjacent_or_overlapping", "separation_m": separation,
                                   "source_scope": "computed_polygon_geometry_not_permission_or_reachability"})
    ego_lane, ego_s, errors = _lane_state(graph.ego, graph)
    memberships = {(r["source"], r["target"]) for r in result if r["relation"] == "belongs_to_lane"}
    if not errors and _number(ego_s) and (graph.ego.entity_id, ego_lane) in memberships:
        for entity in graph.entities:
            lane, station, missing = _lane_state(entity, graph)
            if (entity.kind == "vehicle" and not missing and lane == ego_lane and _number(station)
                    and station > ego_s and (entity.entity_id, lane) in memberships):
                result.append({"source": entity.entity_id, "target": graph.ego.entity_id,
                               "relation": "same_lane_ahead", "lane_id": lane,
                               "station_difference_m": station - ego_s,
                               "source_scope": "current_geometry_and_lane_assignment_not_identity_or_control"})
    return result


def _decision(spec, graph, status, missing=None, target=None, details=None):
    return {"task": spec.to_dict(), "graph_time_s": graph.now_s, "status": status,
            "target": target, "missing_or_rejection": missing or [], "details": details or {},
            "execution_authorized": False, "prototype_only": True,
            "model_calls": 0, "planner_calls": 0, "controller_calls": 0}


def propose_task(spec: TaskSpec, graph: TaskGraph) -> dict:
    if (spec.action not in ACTIONS or spec.action == "unsupported" or spec.parse_status != "supported"
            or spec != parse_task_spec(spec.instruction)):
        return _decision(spec, graph, "need_clarification", ["unsupported_or_ambiguous_task"])
    if spec.action == "follow_vehicle":
        return _propose_follow(spec, graph)
    return _propose_parking(spec, graph)


def _propose_parking(spec, graph):
    ego_missing = []
    dimensions = _read(graph.ego, "dimensions_lw_m", graph.now_s, ego_missing)
    heading = _read(graph.ego, "heading_rad", graph.now_s, ego_missing)
    ego_lane = _read(graph.ego, "lane_id", graph.now_s, ego_missing)
    if not _geometry_available(graph.ego, graph.now_s):
        ego_missing.append("ego.geometry_unknown_stale_or_unsourced")
    if ego_missing or not _xy(dimensions) or any(v <= 0 for v in dimensions) or not _number(heading):
        return _decision(spec, graph, "need_more_observation", ego_missing or ["invalid_ego_geometry"])
    candidates = [n for n in graph.entities if n.kind == spec.target_kind]
    if not candidates:
        return _decision(spec, graph, "need_more_observation", [f"no_evidenced_{spec.target_kind}"])
    accepted, failures = [], []
    for node in candidates:
        missing = []
        if not _geometry_available(node, graph.now_s):
            missing.append("candidate_geometry_unknown_stale_or_unsourced")
        values = {k: _read(node, k, graph.now_s, missing) for k in
                  ("allowed", "occupancy", "dimensions_lw_m", "reachable", "path_cost_m",
                   "path_qualified", "vehicle_sweep_qualified", "maneuver_observed_free", "requires_reverse",
                   "terminal_xy", "terminal_heading_rad")}
        if not missing:
            if values["allowed"] is not True:
                missing.append("parking_not_allowed")
            if values["occupancy"] != "free":
                missing.append("not_confirmed_free")
            if values["reachable"] is not True or values["path_qualified"] is not True:
                missing.append("no_qualified_reachable_path")
            if values["vehicle_sweep_qualified"] is not True or values["maneuver_observed_free"] is not True:
                missing.append("vehicle_sweep_or_full_maneuver_observation_not_verified")
            if values["requires_reverse"] is True:
                rear = _read(node, "rear_observed_free", graph.now_s, missing)
                if rear is not True:
                    missing.append("reverse_rear_occupancy_unknown_or_not_free")
            elif values["requires_reverse"] is not False:
                missing.append("maneuver_reverse_requirement_unknown")
            if not _number(values["path_cost_m"], 0):
                missing.append("invalid_path_cost")
            size = values["dimensions_lw_m"]
            margin = 2 * spec.constraints["clearance_each_side_m"]
            if not _xy(size) or any(size[i] < dimensions[i] + margin for i in range(2)):
                missing.append("vehicle_does_not_fit")
            if not footprint_in_region(values["terminal_xy"], values["terminal_heading_rad"],
                                       [d + margin for d in dimensions], node.polygon_xy):
                missing.append("terminal_footprint_not_verified_in_convex_region")
        if spec.action == "pull_over":
            curb = _read(node, "curb_region_id", graph.now_s, missing)
            adjacency = _read(node, "curb_adjacency_verified", graph.now_s, missing)
            side = _read(node, "side_of_travel", graph.now_s, missing)
            direction = _read(node, "travel_heading_rad", graph.now_s, missing)
            lane = _read(node, "approach_lane_id", graph.now_s, missing)
            curb_node = next((n for n in graph.entities if n.entity_id == curb and n.kind == "curb_region"), None)
            separation = polygon_distance(node.polygon_xy, curb_node.polygon_xy) if curb_node else None
            if (adjacency is not True or separation is None or separation > 0.25
                    or not _geometry_available(curb_node, graph.now_s)):
                missing.append("no_verified_curb_region_association")
            if side not in ("left", "right") or (spec.side and side != spec.side):
                missing.append("wrong_or_unknown_stopping_side")
            if lane != ego_lane or not _number(direction) or _angle(direction, heading) > math.pi / 4:
                missing.append("wrong_or_unknown_direction_of_travel")
            if (not _number(values["terminal_heading_rad"]) or not _number(direction)
                    or _angle(values["terminal_heading_rad"], direction) > spec.constraints["maximum_heading_error_rad"]):
                missing.append("terminal_heading_conflicts_with_direction_of_travel")
        if missing:
            failures.append({"entity_id": node.entity_id, "reasons": missing})
        else:
            accepted.append((values["path_cost_m"], node.entity_id, node, values))
    if not accepted:
        return _decision(spec, graph, "need_more_observation_or_no_feasible_candidate", failures)
    cost, _, node, values = min(accepted, key=lambda r: (r[0], r[1]))
    return _decision(spec, graph, "proposal_ready", target={"entity_id": node.entity_id,
                     "representation": "region_and_terminal_pose", "kind": node.kind,
                     "polygon_xy": node.polygon_xy, "terminal_xy": values["terminal_xy"],
                     "terminal_heading_rad": values["terminal_heading_rad"],
                     "maximum_terminal_speed_mps": spec.constraints["maximum_terminal_speed_mps"]},
                     details={"selection": "minimum_qualified_path_cost_after_all_filters", "path_cost_m": cost,
                              "eligible_ids": [r[1] for r in sorted(accepted, key=lambda r: (r[0], r[1]))],
                              "rejected_candidates": failures, "evidence": {k: asdict(v) for k, v in node.properties.items()},
                              "continuous_execution_safety_verified": False})


def _propose_follow(spec, graph):
    missing = []
    ego_speed = _read(graph.ego, "speed_mps", graph.now_s, missing)
    ego_length = _read(graph.ego, "length_m", graph.now_s, missing)
    ego_heading = _read(graph.ego, "heading_rad", graph.now_s, missing)
    ego_lane, ego_station, lane_missing = _lane_state(graph.ego, graph)
    missing += lane_missing
    relations = task_relations(graph)
    ahead = [r for r in relations if r["relation"] == "same_lane_ahead"]
    if missing or not _number(ego_speed, 0) or not _number(ego_length, 0.01) or not _number(ego_heading):
        return _decision(spec, graph, "need_more_observation", missing or ["invalid_ego_follow_state"])
    if not ahead:
        return _decision(spec, graph, "need_more_observation", ["no_confirmed_same_lane_front_vehicle"])
    # Closest observed lead is assessed first; never skip an uncertain near lead to chase a farther car.
    relation = min(ahead, key=lambda r: (r["station_difference_m"], r["source"]))
    node = next(n for n in graph.entities if n.entity_id == relation["source"])
    values = {k: _read(node, k, graph.now_s, missing) for k in
              ("continuous_identity_verified", "tracking_identity", "speed_mps", "length_m", "heading_rad")}
    if values["continuous_identity_verified"] is not True or not isinstance(values["tracking_identity"], str) or not values["tracking_identity"]:
        missing.append("lead_continuous_identity_not_verified")
    if not _number(values["speed_mps"], 0) or not _number(values["length_m"], 0.01):
        missing.append("lead_speed_or_size_unknown")
    if not _number(values["heading_rad"]) or _angle(values["heading_rad"], ego_heading) > math.pi / 4:
        missing.append("lead_wrong_or_unknown_direction")
    # Nearby vehicles without resolved lane membership may be the actual immediate lead.
    uncertain = []
    for other in graph.entities:
        if other.kind != "vehicle" or other.entity_id == node.entity_id or other.center_xy is None:
            continue
        lane, station, errors = _lane_state(other, graph)
        if ((errors or not _geometry_available(other, graph.now_s)) and graph.ego.center_xy
                and math.dist(other.center_xy, graph.ego.center_xy) <= relation["station_difference_m"]):
            uncertain.append(other.entity_id)
    if uncertain:
        missing.append("nearer_vehicle_lane_membership_unknown:" + ",".join(uncertain))
    if missing:
        return _decision(spec, graph, "need_more_observation", missing, details={"nearest_observed_lead": node.entity_id})
    gap = relation["station_difference_m"] - (ego_length + values["length_m"]) / 2
    desired_gap = spec.constraints["minimum_gap_m"] + spec.constraints["minimum_headway_s"] * ego_speed
    closing = ego_speed - values["speed_mps"]
    ttc = gap / closing if closing > 0 and gap > 0 else None
    if gap < desired_gap or (ttc is not None and ttc < spec.constraints["minimum_ttc_s"]):
        return _decision(spec, graph, "blocked_follow_gap_or_ttc", ["current_gap_constraint_violated"],
                         details={"gap_m": gap, "required_gap_m": desired_gap, "ttc_s": ttc})
    if abs(closing) > spec.constraints["maximum_relative_speed_mps"]:
        return _decision(spec, graph, "blocked_relative_speed", ["relative_speed_constraint_violated"])
    return _decision(spec, graph, "proposal_ready", target={"entity_id": node.entity_id,
                     "tracking_identity": values["tracking_identity"], "representation": "persistent_follow_relation",
                     "lane_id": ego_lane, "fixed_endpoint": None},
                     details={"policy": "receding_horizon_gap_and_relative_speed_monitor",
                              "gap_m": gap, "minimum_desired_gap_m": desired_gap,
                              "relative_speed_mps": closing, "ttc_s": ttc,
                              "constraints": spec.constraints, "task_relations": relations,
                              "control_command_generated": False})


@dataclass
class MonitorState:
    start_s: float
    last_observed_s: float | None = None
    valid_since_s: float | None = None
    bound_tracking_identity: str | None = None
    status: str = "pending"


def monitor_task(spec: TaskSpec, decision: dict, state: MonitorState, observation: Entity, now_s: float) -> dict:
    """Monitor actual telemetry supplied by a provider, never predicted ego arrival.

    Synthetic telemetry is allowed only as explicitly identified interface tests.
    A success here is an observed terminal predicate or bounded follow interval,
    not a certificate of controller safety or an experimental driving score.
    """
    def finish(status, reasons=None):
        state.status = status
        return {"status": status, "reasons": reasons or [], "state": asdict(state),
                "execution_authorized": False, "actual_arrival_claim": False}

    if state.status.startswith("failed_"):
        return finish(state.status, ["terminal_failure_is_latched"])
    if spec != parse_task_spec(spec.instruction) or decision.get("task") != spec.to_dict():
        return finish("blocked_invalid_or_mismatched_task_contract")
    if now_s < state.start_s or not math.isfinite(now_s):
        return finish("invalid_monitor_time")
    if state.last_observed_s is not None and now_s <= state.last_observed_s:
        return finish("invalid_nonmonotonic_monitor_time")
    if spec.action == "unsupported" or decision["status"] != "proposal_ready":
        state.valid_since_s = None
        return finish("blocked_no_task_proposal")
    missing = []
    observed = _read(observation, "execution_observed", now_s, missing)
    collision = _read(observation, "collision_free", now_s, missing)
    if collision is False:
        state.valid_since_s = None
        return finish("failed_collision")
    if now_s - state.start_s >= spec.timeout_s:
        state.valid_since_s = None
        return finish("failed_timeout")
    if observed is not True or missing or collision is not True:
        state.valid_since_s = None
        return finish("need_execution_observation", missing)
    if state.last_observed_s is None or now_s - state.last_observed_s > spec.constraints["maximum_observation_gap_s"]:
        state.valid_since_s = None
    state.last_observed_s = now_s
    if spec.action == "follow_vehicle":
        identity = _read(observation, "tracking_identity", now_s, missing)
        bound = decision["target"]["tracking_identity"]
        if state.bound_tracking_identity is None:
            state.bound_tracking_identity = bound
        if state.bound_tracking_identity != bound or (identity is not None and identity != state.bound_tracking_identity):
            state.valid_since_s = None
            return finish("failed_identity_switch")
        lane_ok = _read(observation, "same_lane_front", now_s, missing)
        gap = _read(observation, "bumper_gap_m", now_s, missing)
        ego_speed = _read(observation, "ego_speed_mps", now_s, missing)
        lead_speed = _read(observation, "lead_speed_mps", now_s, missing)
        if missing or identity is None or lane_ok is not True or not all(_number(v, 0) for v in (gap, ego_speed, lead_speed)):
            state.valid_since_s = None
            return finish("lost_or_unknown_lead", missing)
        required = spec.constraints["minimum_gap_m"] + spec.constraints["minimum_headway_s"] * ego_speed
        closing = ego_speed - lead_speed
        if (gap < required or abs(closing) > spec.constraints["maximum_relative_speed_mps"] or
                (closing > 0 and gap / closing < spec.constraints["minimum_ttc_s"])):
            state.valid_since_s = None
            return finish("failed_follow_distance_or_speed_constraint")
        state.valid_since_s = now_s if state.valid_since_s is None else state.valid_since_s
        user_end_missing = []
        user_end = _read(observation, "user_end", now_s, user_end_missing)
        if user_end is True:
            return finish("ended_by_user_after_current_valid_follow")
        if spec.requested_duration_s is not None and now_s - state.valid_since_s >= spec.requested_duration_s:
            return finish("requested_safe_follow_interval_observed")
        return finish("following_under_observation")
    position = _read(observation, "position_xy", now_s, missing)
    yaw = _read(observation, "heading_rad", now_s, missing)
    speed = _read(observation, "speed_mps", now_s, missing)
    dimensions = _read(observation, "dimensions_lw_m", now_s, missing)
    terminal_region_verified = _read(observation, "terminal_region_still_allowed_and_free", now_s, missing)
    if missing or not _xy(position) or not _number(yaw) or not _number(speed, 0) or not _xy(dimensions):
        state.valid_since_s = None
        return finish("need_terminal_observation", missing)
    target = decision["target"]
    satisfied = (terminal_region_verified is True and footprint_in_region(position, yaw, dimensions, target["polygon_xy"])
                 and _angle(yaw, target["terminal_heading_rad"]) <= spec.constraints["maximum_heading_error_rad"]
                 and speed <= spec.constraints["maximum_terminal_speed_mps"])
    if not satisfied:
        state.valid_since_s = None
        return finish("terminal_conditions_not_met")
    state.valid_since_s = now_s if state.valid_since_s is None else state.valid_since_s
    if now_s - state.valid_since_s >= spec.termination["hold_s"]:
        return finish("terminal_conditions_observed")
    return finish("holding_terminal_conditions")


def graph_from_dict(payload: dict) -> TaskGraph:
    def entity(value):
        return Entity(value["entity_id"], value["kind"], value.get("center_xy"), value.get("polygon_xy", []),
                      {k: Evidence(**v) for k, v in value.get("properties", {}).items()},
                      Evidence(**value["geometry_evidence"]) if value.get("geometry_evidence") else None)
    return TaskGraph(payload["now_s"], [entity(n) for n in payload["entities"]], entity(payload["ego"]), payload["source_scope"])


def task_spec_schema() -> dict:
    """Contract for semantics only: language/model output cannot assert safety."""
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "SemanticTaskSpec20261123",
            "type": "object", "additionalProperties": False,
            "required": ["action", "instruction", "target_kind", "goal_mode", "timeout_s", "parser", "parse_status",
                         "requested_duration_s", "side", "target_conditions", "constraints", "termination"],
            "properties": {"action": {"enum": list(ACTIONS)}, "instruction": {"type": "string"},
                           "target_kind": {"enum": ["stopping_region", "parking_space", "vehicle", "unknown"]},
                           "goal_mode": {"enum": ["terminal_region_pose", "persistent_relation_policy", "clarification"]},
                           "timeout_s": {"type": "number", "minimum": 0},
                           "requested_duration_s": {"type": ["number", "null"], "exclusiveMinimum": 0},
                           "side": {"enum": ["left", "right", None]}, "parser": {"type": "string"},
                           "parse_status": {"enum": ["supported", "need_clarification"]},
                           "target_conditions": {"type": "array", "items": {"type": "string"}},
                           "constraints": {"type": "object"}, "termination": {"type": "object"}},
            "$comment": "只由确定性策略补约束和终止；未来Qwen仅提语义槽位，不能填allowed/free/reachable或执行授权。"}


def language_intent_schema() -> dict:
    """Future language-model boundary; distinct from the policy-completed contract."""
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "LanguageIntentOnly20261123",
            "type": "object", "additionalProperties": False,
            "required": ["action", "target_description", "target_conditions", "requested_constraints",
                         "termination_request", "needs_clarification"],
            "properties": {"action": {"enum": list(ACTIONS)}, "target_description": {"type": "string"},
                           "target_conditions": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                               "required": ["attribute", "requested_value", "grounding_required"],
                               "properties": {"attribute": {"enum": ["category", "color", "lane_relation", "region_type",
                                   "free_occupancy_requirement", "parking_permission_requirement"]},
                                   "requested_value": {"type": ["string", "number", "boolean", "null"]},
                                   "grounding_required": {"const": True}}}},
                           "requested_constraints": {"type": "object", "additionalProperties": False,
                               "properties": {"side": {"enum": ["left", "right", None]},
                                   "maximum_speed_mps": {"type": ["number", "null"], "exclusiveMinimum": 0},
                                   "minimum_headway_s": {"type": ["number", "null"], "exclusiveMinimum": 0}}},
                           "termination_request": {"type": "object", "additionalProperties": False,
                               "required": ["mode", "duration_s"], "properties": {
                                   "mode": {"enum": ["stopped_in_allowed_region", "safe_follow_duration", "user_end", "unknown"]},
                                   "duration_s": {"type": ["number", "null"], "exclusiveMinimum": 0, "maximum": 600}}},
                           "needs_clarification": {"type": "boolean"}},
            "$comment": "无allowed/occupied/reachable/尺寸/坐标/安全阈值/执行授权字段；语言输出不提升证据权限。本轮未调用Qwen。"}
