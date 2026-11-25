"""Run synthetic interface demonstrations and a real scene0012 support diagnostic.

No model, planning search, controller, or simulator is started. Synthetic fixture
outputs are interface demonstrations, never nuScenes experiment success rates.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from conceptgraph.navigation.semantic_tasks_20261123 import (  # noqa: E402
    Entity, Evidence, TaskGraph, parse_task_spec, point_in_polygon, propose_task,
    language_intent_schema, task_relations, task_spec_schema,
)

R1 = PROJECT / ("outputs/results/nuscenes/validation/iterations/"
                "recall_refinement_R1_full_20261120_v2/scene-0012/frames.jsonl.gz")
CONTEXT = PROJECT / ("outputs/project_v2/thesis_completion_v1/ablations/"
                     "causal_goals_full_001/contexts/scene-0012_f12/context.json")
MAP_ROOT = PROJECT / "dataset/nuscenes_map_expansion/expansion"


def dump(path, value):
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture_evidence(value, kind="geometry", now=100.0, ttl=5.0):
    return Evidence(value, "synthetic_fixture:unit_interface_only", now, ttl, source_kind=kind)


def synthetic_graph():
    """Entire scene is synthetic. Authority types describe mocked provider contracts."""
    ev = fixture_evidence
    lane = Entity("lane-A", "lane", (25, 0), [(-20, -2), (80, -2), (80, 2), (-20, 2)])
    ego = Entity("ego", "ego", (0, 0), properties={
        "lane_id": ev("lane-A"), "lane_station_m": ev(20.0), "speed_mps": ev(5.0, "telemetry"),
        "length_m": ev(4.5), "dimensions_lw_m": ev([4.5, 1.8]), "heading_rad": ev(0.0)})
    curb = Entity("curb-A", "curb_region", (12, -5), [(5, -5.2), (20, -5.2), (20, -5), (5, -5)])

    def region(identity, kind, center, cost, occupancy="free"):
        x, y = center
        properties = {"allowed": ev(True, "regulatory_map"), "occupancy": ev(occupancy, "perception"),
                      "dimensions_lw_m": ev([6.0, 3.0]), "reachable": ev(True, "planner"),
                      "path_qualified": ev(True, "planner_audit"), "path_cost_m": ev(cost, "planner"),
                      "vehicle_sweep_qualified": ev(True, "planner_audit"),
                      "maneuver_observed_free": ev(True, "perception_audit"), "requires_reverse": ev(False, "planner"),
                      "terminal_xy": ev(list(center)), "terminal_heading_rad": ev(0.0)}
        return Entity(identity, kind, center, [(x - 3, y - 1.5), (x + 3, y - 1.5),
                                             (x + 3, y + 1.5), (x - 3, y + 1.5)], properties)

    pull = region("pull-region", "stopping_region", (12, -3.5), 18.0)
    pull.properties.update({"curb_region_id": ev("curb-A"), "curb_adjacency_verified": ev(True),
                            "side_of_travel": ev("right"), "travel_heading_rad": ev(0.0),
                            "approach_lane_id": ev("lane-A")})
    near_blocked = region("space-near-occupied", "parking_space", (5, 5), 7.0, "occupied")
    euclidean_near = region("space-short-euclidean", "parking_space", (10, 5), 40.0)
    path_near = region("space-min-path", "parking_space", (22, 5), 26.0)
    lead = Entity("front-vehicle", "vehicle", (25, 0), properties={
        "lane_id": ev("lane-A"), "lane_station_m": ev(45.0),
        "continuous_identity_verified": ev(True, "tracking_audit"),
        "tracking_identity": ev("track-verified-A", "tracking"), "speed_mps": ev(5.0, "tracking"),
        "length_m": ev(4.5), "heading_rad": ev(0.0)})
    for entity in [ego, lane, curb, pull, near_blocked, euclidean_near, path_near, lead]:
        entity.geometry_evidence = ev(True)
    return TaskGraph(100.0, [lane, curb, pull, near_blocked, euclidean_near, path_near, lead], ego,
                     "synthetic_unit_interface_demo_not_real_driving")


def _station(point, centerline):
    best = None
    prefix = 0.0
    for a, b in zip(centerline, centerline[1:]):
        delta = (b[0] - a[0], b[1] - a[1])
        length = math.hypot(*delta)
        if length == 0:
            continue
        t = max(0.0, min(1.0, ((point[0] - a[0]) * delta[0] + (point[1] - a[1]) * delta[1]) / length**2))
        projection = (a[0] + t * delta[0], a[1] + t * delta[1])
        candidate = (math.dist(point, projection), prefix + t * length)
        best = min(best, candidate) if best is not None else candidate
        prefix += length
    return best[1] if best else None


def real_graph():
    context = json.loads(CONTEXT.read_text())
    with gzip.open(R1, "rt") as handle:
        frame = next(json.loads(line) for line in handle if json.loads(line)["frame_idx"] == 12)
    now = frame["camera_timestamp_s"]
    lanes = [Entity(token, "lane", polygon_xy=value["polygon"])
             for token, value in context["lane_geometries"].items()]

    def geometry(value, source, kind="geometry", observed=now):
        return Evidence(value, source, observed, 0.5, source_kind=kind)

    def lane_properties(center):
        possible = [n for n in lanes if point_in_polygon(center, n.polygon_xy)]
        if len(possible) != 1:
            return {"lane_id": geometry(None, str(CONTEXT)), "lane_station_m": geometry(None, str(CONTEXT))}
        lane = possible[0]
        station = _station(center, context["lane_geometries"][lane.entity_id]["centerline"])
        return {"lane_id": geometry(lane.entity_id, str(CONTEXT)),
                "lane_station_m": geometry(station, str(CONTEXT))}

    pose = context["camera_pose"]
    center = (pose[0][3], pose[1][3])
    ego = Entity("ego", "ego", center, properties=lane_properties(center))
    ego.geometry_evidence = geometry(True, str(CONTEXT))
    ego.properties["heading_rad"] = geometry(math.atan2(pose[1][2], pose[0][2]), str(CONTEXT))
    # Camera pose is not a calibrated ego body footprint or longitudinal speed.
    nodes = []
    for raw in frame["predictions"]["R1"]:
        category = raw["category"]
        kind = "vehicle" if category in {"car", "truck", "bus", "motorcycle", "bicycle"} else "object"
        center = tuple(raw["center"][:2])
        props = lane_properties(center)
        stamp = raw.get("timestamp_s", now)
        props["tracking_identity"] = geometry(raw["node_id"], str(R1), "tracking", stamp)
        props["continuous_identity_verified"] = Evidence(None, str(R1), now, 0.5, source_kind="tracking_audit")
        # R1 scalar velocity and axis-aligned extent are retained in provenance,
        # never promoted to verified longitudinal speed or vehicle body length.
        nodes.append(Entity(raw["node_id"], kind, center, properties=props,
                            geometry_evidence=geometry(True, str(R1), "perception", stamp)))
    for lane in lanes:
        lane.geometry_evidence = geometry(True, str(CONTEXT), "map")
    graph = TaskGraph(now, [*lanes, *nodes], ego, "real_scene0012_f12_R1_and_historical_road_geometry_diagnostic")
    support = {"scene": frame["scene"], "frame_idx": frame["frame_idx"], "timestamp_s": now,
               "R1_nodes": len(nodes), "R1_vehicle_nodes": sum(n.kind == "vehicle" for n in nodes),
               "road_lane_entities": len(lanes), "computed_task_relations": task_relations(graph),
               "graph_has_no_GT": True, "R1_snapshot_preserved_no_retracking": True,
               "historical_context_obstacles_not_reused_as_R1_occupancy": True,
               "missing": ["真实路缘几何与可停靠侧审核", "可停车许可及有效期", "单个停车位几何和尺寸",
                           "实时空闲及遮挡unknown", "针对本任务候选的合格路线和路径成本",
                           "整个机动扫掠区域的观测覆盖（倒车时另需后方空闲证据）",
                           "ego车身尺寸/速度与完整车辆航向", "真实前车连续身份/纵向速度与车身长度"],
               "boundary": "可计算当前中心属于车道与同车道前方关系，不证明车身全在车道、连续ID、可停车或可执行控制",
               "source": {"R1": str(R1), "context": str(CONTEXT)}}
    return graph, support


def audit_map_layers():
    result = []
    for path in sorted(MAP_ROOT.glob("*.json")):
        payload = json.loads(path.read_text())
        layers = {k: len(v) for k, v in payload.items() if isinstance(v, list)}
        fields = {k: sorted({field for r in payload[k] for field in r}) for k in
                  ("carpark_area", "road_segment", "road_block", "lane", "lane_connector", "drivable_area") if k in payload}
        result.append({"path": str(path), "sha256": sha(path), "layers": layers, "fields": fields,
                       "dedicated_curb_layer": any(k in payload for k in ("curb", "curb_region", "road_curb")),
                       "individual_parking_space_layer": any(k in payload for k in ("parking_space", "parking_slot")),
                       "live_occupancy_or_parking_permission_not_in_inspected_fields": True})
    return {"maps": result,
            "interpretation": "carpark_area是停车区域记录，未发现逐车位/占用/停车许可字段；边界/可行驶多边形不能升级为路缘或允许停车",
            "local_scope_only": True, "metadata_does_not_create_missing_perception_labels": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if shutil.disk_usage(PROJECT).free < 20 * 1024**3:
        raise OSError("磁盘低于20GiB")
    args.input_dir.mkdir(parents=True, exist_ok=False)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), PROJECT / "conceptgraph/navigation/semantic_tasks_20261123.py",
               R1, CONTEXT, *MAP_ROOT.glob("*.json")]
    fingerprints = {str(p): sha(p) for p in sources}
    dump(args.run_dir / "frozen_configuration.json", {
        "source_and_input_sha256": fingerprints, "new_model_calls": 0, "new_planner_calls": 0,
        "synthetic_fixture_is_not_dataset_score": True, "real_frame": "scene-0012/f12",
        "no_parameter_search": True, "randomness": None, "public_pipeline_modified": False})
    instructions = ["靠边停车", "停到最近空闲车位", "跟车行驶"]
    graph = synthetic_graph()
    dump(args.input_dir / "synthetic_interface_graph.json", asdict(graph))
    specs = [parse_task_spec(text) for text in instructions]
    dump(args.report_dir / "task_contracts.json", {"scope": "rule_parser_prototype_not_Qwen", "tasks": [s.to_dict() for s in specs]})
    dump(args.report_dir / "task_contract.schema.json", task_spec_schema())
    dump(args.report_dir / "language_intent.schema.json", language_intent_schema())
    demo = {"scope": "synthetic_unit_interface_only_not_empirical_metrics", "same_input_graph": True,
            "instructions": instructions, "decisions": [propose_task(s, graph) for s in specs],
            "execution_authorized": False}
    dump(args.run_dir / "synthetic_demo.json", demo)
    dump(args.report_dir / "synthetic_demo.json", demo)
    real, support = real_graph()
    dump(args.input_dir / "scene0012_f12_task_graph.json", asdict(real))
    support["task_decisions"] = [propose_task(s, real) for s in specs]
    dump(args.run_dir / "real_scene0012_diagnostic.json", support)
    dump(args.report_dir / "real_scene0012_diagnostic.json", support)
    dump(args.report_dir / "local_map_layer_support.json", audit_map_layers())
    if any(sha(Path(p)) != digest for p, digest in fingerprints.items()):
        raise AssertionError("输入运行中改变")
    snapshot = args.run_dir / "source_snapshot"
    for path in sources[:2]:
        dest = snapshot / path.relative_to(PROJECT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    print(json.dumps({"synthetic": [{"action": d["task"]["action"], "status": d["status"], "target": d["target"]} for d in demo["decisions"]],
                      "real": [{"action": d["task"]["action"], "status": d["status"], "missing": d["missing_or_rejection"]} for d in support["task_decisions"]],
                      "real_nodes": support["R1_nodes"], "real_lanes": support["road_lane_entities"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
