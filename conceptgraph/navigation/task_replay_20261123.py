"""Causal recorded-data adapter for the frozen semantic task prototype.

This module consumes predicted objects, known road geometry, current camera
calibration/pose, and already observed pose history. It neither reads task GT nor
infers permission, free occupancy, vehicle dimensions, or verified identities.
"""
from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np

from conceptgraph.navigation.semantic_tasks_20261123 import (
    Entity, Evidence, MonitorState, TaskGraph, monitor_task, parse_task_spec,
    point_in_polygon, propose_task, task_relations,
)


def transform(translation, rotation_wxyz):
    q = np.asarray(rotation_wxyz, dtype=float)
    t = np.asarray(translation, dtype=float)
    if q.shape != (4,) or t.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(t).all() or np.linalg.norm(q) < 1e-10:
        raise ValueError("invalid calibration")
    w, x, y, z = q / np.linalg.norm(q)
    result = np.eye(4)
    result[:3, :3] = [[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                     [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                     [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]]
    result[:3, 3] = t
    return result


def ego_pose_from_camera(camera_world, calibration):
    matrix = np.asarray(camera_world, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("invalid camera pose")
    return matrix @ np.linalg.inv(transform(calibration["translation"], calibration["rotation"]))


class PastPoseMotion:
    """One backward difference, without future interpolation/extrapolation."""

    def __init__(self):
        self.previous = None

    def update(self, timestamp_s, world_from_ego, eligibility_s, source):
        matrix = np.asarray(world_from_ego, dtype=float)
        if (not all(math.isfinite(x) for x in (timestamp_s, eligibility_s)) or timestamp_s > eligibility_s + 1e-6
                or matrix.shape != (4, 4) or not np.isfinite(matrix).all()):
            raise ValueError("future or invalid ego observation")
        if self.previous is not None and timestamp_s <= self.previous[0]:
            raise ValueError("ego observations must be strictly increasing")
        heading = math.atan2(matrix[1, 0], matrix[0, 0])
        center = matrix[:2, 3]
        estimate = None
        interval = None
        previous_source = None
        if self.previous is not None:
            previous_time, previous_matrix, previous_source = self.previous
            dt = timestamp_s - previous_time
            delta = (center - previous_matrix[:2, 3]) / dt
            longitudinal = float(delta @ np.array([math.cos(heading), math.sin(heading)]))
            lateral = float(delta @ np.array([-math.sin(heading), math.cos(heading)]))
            estimate = {"world_velocity_xy_mps": delta.tolist(), "signed_longitudinal_mps": longitudinal,
                        "lateral_mps": lateral, "speed_norm_mps": float(np.linalg.norm(delta)),
                        "estimate_scope": "mean_over_past_pose_interval_not_instantaneous_telemetry"}
            interval = [previous_time, timestamp_s]
        self.previous = (timestamp_s, matrix.copy(), source)
        ev = lambda value: Evidence(value, source, timestamp_s, 1.0, source_kind="geometry")
        ego = Entity("ego", "ego", tuple(center), properties={"heading_rad": ev(heading)},
                     geometry_evidence=ev(True))
        # Negative longitudinal velocity is retained in the diagnostic; a forward
        # speed consumer receives unknown instead of silently taking abs().
        speed = estimate["signed_longitudinal_mps"] if estimate and estimate["signed_longitudinal_mps"] >= 0 else None
        ego.properties["speed_mps"] = ev(speed)
        diagnostic = {"pose_timestamp_s": timestamp_s, "available_at_s": eligibility_s,
                      "pose_age_s": eligibility_s - timestamp_s, "world_from_ego": matrix.tolist(),
                      "current_source": source, "past_source": previous_source,
                      "past_interval_s": interval, "motion": estimate, "future_pose_used": False,
                      "body_dimensions_measured": False, "ego_reference_to_body_center_verified": False}
        return ego, diagnostic


def station_on_centerline(point, line):
    prefix, best = 0.0, None
    for a, b in zip(line, line[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        u = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length**2))
        distance = math.dist(point, (a[0] + u * dx, a[1] + u * dy))
        candidate = (distance, prefix + u * length)
        best = min(best, candidate) if best is not None else candidate
        prefix += length
    return best[1] if best else None


class ObjectRoadProvider:
    def __init__(self, lane_payload, lane_source):
        self.geometries = lane_payload.get("geometries", {}) if lane_payload else {}
        self.lane_source = lane_source

    def _assign_lane(self, entity, now):
        possible = [(token, g) for token, g in self.geometries.items()
                    if entity.center_xy is not None and point_in_polygon(entity.center_xy, g["polygon"])]
        lane, station = None, None
        if len(possible) == 1:
            lane, geometry = possible[0]
            station = station_on_centerline(entity.center_xy, geometry["centerline"])
        evidence = lambda v: Evidence(v, self.lane_source, now, 1.0, source_kind="geometry")
        entity.properties.update(lane_id=evidence(lane), lane_station_m=evidence(station))
        return [token for token, _ in possible]

    def build(self, predictions, ego, now_s, prediction_source, condition):
        lanes = []
        for token, geometry in self.geometries.items():
            lanes.append(Entity(token, "lane", polygon_xy=geometry["polygon"],
                                geometry_evidence=Evidence(True, self.lane_source, now_s, 1.0, source_kind="map")))
        assignments = {"ego": self._assign_lane(ego, now_s)}
        objects = []
        for raw in predictions:
            center = raw["center"][:2]
            observed = float(raw["timestamp_s"])
            if not np.isfinite(center).all() or observed > now_s + 1e-6:
                raise ValueError("invalid or future predicted object")
            identity = condition + ":" + raw["node_id"]
            kind = "vehicle" if raw["category"] in {"car", "truck", "bus", "bicycle", "motorcycle"} else "object"
            evidence = lambda value, authority="perception": Evidence(value, prediction_source, observed, 0.75, source_kind=authority)
            properties = {"tracking_identity": evidence(identity, "tracking"),
                          "continuous_identity_verified": evidence(None, "tracking_audit"),
                          "predicted_category": evidence(raw["category"]),
                          "predicted_motion_state": evidence(raw.get("motion_state"), "tracking"),
                          "position_covariance_diag": evidence(raw.get("position_covariance_diag")),
                          "length_m": evidence(None, "geometry"), "heading_rad": evidence(None, "geometry"),
                          "speed_mps": evidence(None, "tracking")}
            entity = Entity(identity, kind, tuple(center), properties=properties, geometry_evidence=evidence(True))
            assignments[identity] = self._assign_lane(entity, now_s)
            objects.append(entity)
        graph = TaskGraph(now_s, [*lanes, *objects], ego,
                          "frozen_multiview_objects_and_known_static_road_prior_no_task_GT")
        relations = task_relations(graph)
        coverage = {"object_nodes": len(objects), "vehicle_nodes": sum(n.kind == "vehicle" for n in objects),
                    "lane_nodes": len(lanes), "road_input_available": bool(lanes),
                    "ego_lane_candidates": assignments["ego"],
                    "ego_unique_lane": len(assignments["ego"]) == 1,
                    "objects_unique_lane": sum(len(v) == 1 for k, v in assignments.items() if k != "ego"),
                    "ambiguous_lane_assignments": sum(len(v) > 1 for v in assignments.values()),
                    "relations": relations, "lane_assignment_candidates": assignments,
                    "stopping_regions": 0, "parking_spaces": 0, "rear_view_available": False,
                    "temporal_identity_is_only_internal_track": True}
        return graph, coverage


def provider_gaps(action, graph):
    """Complete downstream gap inventory, separate from the first runtime stop."""
    now = graph.now_s
    speed, speed_error = graph.ego.get("speed_mps", now)
    result = [
        {"field": "ego_actual_dimensions_and_reference_alignment", "status": "unknown",
         "needs": "vehicle calibration or verified ego body configuration; engineering VehicleSpec is diagnostic only"},
        {"field": "task_ground_truth_and_human_review", "status": "pending",
         "needs": "human task/target/region review; no correctness metrics this run"},
    ]
    if action in {"pull_over", "park_nearest"}:
        result += [{"field": field, "status": "unknown", "needs": need} for field, need in [
            ("parking_permission", "independent regulatory or verified permission evidence"),
            ("current_free_occupancy", "visible occupancy and occlusion/coverage audit"),
            ("full_maneuver_observation_and_rear_if_reversing", "front-three views are not 360 coverage"),
            ("qualified_route_cost_and_vehicle_sweep", "task-specific real planning and independent body audit")]]
        if action == "pull_over":
            result += [{"field": "curb_and_allowed_stopping_region", "status": "unknown",
                        "needs": "curb geometry, direction/side, and permission review; road edge is insufficient"}]
        else:
            result += [{"field": "individual_parking_space_geometry", "status": "unknown",
                        "needs": "individual space polygon/size/heading; carpark_area is not a space"}]
    else:
        result += [{"field": "ego_backward_pose_speed", "status": "unknown" if speed_error else "past_interval_estimate_available",
                    "value_mps": speed, "needs": "first frame lacks a past observation; estimate is not instantaneous telemetry"}]
        result += [{"field": field, "status": "unknown", "needs": need} for field, need in [
            ("verified_front_vehicle_continuous_identity", "internal tracker ID alone is not audited true continuity"),
            ("front_vehicle_longitudinal_speed_heading_length", "directional tracking and body-size evidence"),
            ("executed_follow_control_observations", "recorded nuScenes trajectory is not task-triggered control")]]
    return result


class EpisodeSession:
    """One issued task, one persistent monitor; no per-frame new instruction."""

    def __init__(self, episode, condition):
        self.episode = episode
        self.condition = condition
        self.spec = parse_task_spec(episode["instruction"])
        self.monitor = MonitorState(episode["issue_time_s"])
        self.last_frame = None
        self.issue_count = 1

    def step(self, frame_idx, graph):
        if graph.now_s < self.episode["issue_time_s"] or (self.last_frame is not None and frame_idx <= self.last_frame):
            raise ValueError("pre-issue or nonmonotonic episode frame")
        self.last_frame = frame_idx
        decision = propose_task(self.spec, graph)
        observation = Entity("recorded_replay", "telemetry", properties={
            "execution_observed": Evidence(False, "nuScenes_recording_not_task_execution", graph.now_s, 1.0, source_kind="telemetry")})
        monitored = monitor_task(self.spec, decision, self.monitor, observation, graph.now_s)
        return {"episode_id": self.episode["episode_id"], "scene": self.episode["scene"],
                "condition": self.condition, "frame_idx": frame_idx, "time_s": graph.now_s,
                "issue_time_s": self.episode["issue_time_s"], "instruction_issue_count": self.issue_count,
                "task_spec": self.spec.to_dict(), "task_decision": decision, "monitor": monitored,
                "provider_gaps": provider_gaps(self.spec.action, graph),
                "human_review_status": "pending", "task_correctness_evaluated": False,
                "parking_success_evaluated": False, "follow_success_evaluated": False,
                "GT_in_inference": False, "actual_vehicle_control": False}
