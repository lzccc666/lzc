"""Export a local evidence browser from immutable experiment artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from conceptgraph.navigation.lane_constraints import load_lane_topology, validate_lane_path
from conceptgraph.navigation.planner import OccupancyGrid2D, smooth_grid_path
from conceptgraph.nuscenes.benchmark_ground_truth import associate_nodes
from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT
from conceptgraph.scripts.lane_expansion_production import resolve_online_directory
from conceptgraph.scripts.run_incremental_semantic_navigation import write_json


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def rounded(value):
    return np.asarray(value, dtype=float).round(5).tolist()


class Exporter:
    def __init__(self, root):
        self.root = root
        self.output = root / "visualizations"
        (self.output / "data").mkdir(parents=True, exist_ok=True)
        (self.output / "grids").mkdir(exist_ok=True)
        self.sources = set()
        self.grids = {}
        self.rejected = []
        self.reranks = []
        self.representative_candidates = []

    def grid(self, path):
        path = Path(path)
        self.sources.add(path)
        if path in self.grids:
            return self.grids[path]
        with np.load(path) as data:
            occupancy = data["occupancy"]
            origin = data["origin_xy"]
            resolution = float(data["resolution"])
        digest = hashlib.sha256(occupancy.tobytes()+np.asarray(occupancy.shape).tobytes()).hexdigest()[:24]
        destination = self.output / "grids" / f"{digest}.png"
        if not destination.exists():
            pixels = np.zeros((*occupancy.shape, 4), dtype=np.uint8)
            pixels[occupancy != 0] = [105, 111, 118, 140]
            Image.fromarray(pixels).save(destination)
        result = {"image": f"grids/{digest}.png", "origin": rounded(origin), "resolution": resolution, "shape": list(occupancy.shape)}
        self.grids[path] = result
        return result

    def static(self, scene, topology, vehicle):
        runs = read(self.root / 'experiment_run_manifest.json', {}).get('active_runs', {})
        run = runs.get('static_test', 'static_test_v1') if scene['role'] == 'sealed_test' else runs.get('static_development', 'static_development_all_v3')
        complete_path = self.root / "experiments" / run / scene["scene"] / "complete.json"
        complete = read(complete_path)
        self.sources.add(complete_path)
        cases = []
        for row in complete["results"]:
            directory = Path(row["artifact"])
            result_path = directory / "target_selection_result.json"
            result = read(result_path)
            self.sources.add(result_path)
            grid_path = directory / "live_grid.npz"
            if not grid_path.exists():
                # A semantic no-candidate failure happens before grid creation.
                # Show its actual frozen input map; never invent a live grid.
                grid_path = directory.parent / "input_bev.npz"
            grid = self.grid(grid_path)
            plan = read(Path(result["plan_path"])) if result["success"] else None
            if plan:
                self.sources.add(Path(result["plan_path"]))
            path = plan["path"]["world_xyz"] if plan else []
            goals, rejected = [], []
            for candidate in result["evaluated_candidates"]:
                for goal in candidate["goal_candidates"]:
                    if goal.get("goal_xy"):
                        goals.append({"node_id": candidate["node_id"], "desired": goal.get("desired_goal_xy"), "goal": goal["goal_xy"], "success": goal["path_found"], "reason": goal["rejection_reason"], "projection_m": goal.get("debug", {}).get("goal_projection_distance_m", goal.get("fallback_distance_m"))})
                    if goal.get("debug", {}).get("smoothing_rejected") and not rejected:
                        with np.load(grid_path) as data:
                            occupancy = OccupancyGrid2D(data["occupancy"], float(data["resolution"]), data["origin_xy"])
                        smoothed = smooth_grid_path(occupancy, [tuple(rc) for rc in goal["path_grid_raw"]], chaikin_iterations=2)
                        validation = validate_lane_path(smoothed, topology, occupancy.occupancy, start_heading_xy=vehicle[3:5], direction_tolerance_deg=75)
                        stored = goal["debug"]["smoothing_validation"]
                        match = validation == stored
                        item = {"path": rounded([occupancy.grid_to_world(rc) for rc in smoothed]), "source": "deterministically reconstructed from saved raw A* path, actual occupancy and frozen smoothing=2; coordinates were not originally stored", "saved_validation_reproduced": match, "stored_validation": stored, "artifact": str(result_path)}
                        rejected.append(item)
                        self.rejected.append({"scene": scene["scene"], "query_id": row["query_id"], "mode": row["method"], **item})
            cases.append({"query_id": row["query_id"], "method": row["method"], "success": result["success"], "failure": result["failure_code"], "grid": grid, "path": rounded(np.asarray(path).reshape(-1, 3)[:, :2]), "selected": row["selected_node_id"], "goals": goals, "rejected": rejected, "artifact": "../"+str(result_path.relative_to(self.root)), "lane_validation": row.get("lane_validation"), "retrieval": row["retrieval"]})
            if scene["role"] == "sealed_test":
                self.representative_candidates.append({"scene": scene["scene"], "query_id": row["query_id"], "method": row["method"], "success": result["success"], "failure": result["failure_code"], "artifact": str(result_path.relative_to(self.root))})
        return cases

    def online(self, scene, queries, official):
        cases = []
        for query in queries:
            if not query["online_replay_selected"]:
                continue
            variants = ["off", "base", "adapter"] if scene["role"] == "sealed_test" or scene["scene"] in {"scene-0003", "scene-0916"} else ["off"]
            for variant in variants:
                for lane in (False, True):
                    method = f"{variant}_{'lane' if lane else 'drivable'}"
                    directory = resolve_online_directory(self.root, query, variant, lane)
                    complete = read(directory / "complete.json")
                    if complete is None:
                        continue
                    self.sources.add(directory / "complete.json")
                    for repeat in ("first", "cache_repeat"):
                        frames_path = directory / repeat / "end_to_end_replay" / scene["scene"] / "frames.jsonl"
                        self.sources.add(frames_path)
                        poses = np.loadtxt(self.root / "converted" / scene["scene"] / "poses.txt").reshape(-1, 4, 4)
                        frames = []
                        for line in frames_path.open():
                            frame = json.loads(line)
                            index = frame["frame_idx"]
                            audit = frame["experiment_audit"]
                            events = []
                            for event in frame["qwen_audit"]["events"]:
                                if event.get("task") != "candidate_rerank":
                                    continue
                                context = event.get("context", {})
                                before = [c["node_id"] for c in sorted(context.get("verified_candidate_payload", []), key=lambda c: (c.get("semantic_rank") or 10**9, c["node_id"]))]
                                request = next(iter((directory / repeat / "qwen_audit").glob(f"{event['task_id']:06d}_candidate_rerank.json")), None)
                                response = read(request, {}) if request else {}
                                if request:
                                    self.sources.add(request)
                                response = response.get("result", response)
                                after = (response.get("normalized_output") or {}).get("ranking")
                                item = {"frame": index, "submitted_frame": event.get("submitted_frame"), "before_verified_ids": before, "response_ranking": after, "selected_recommendation": event.get("recommendation"), "applied": event.get("applied", False), "diagnostic": event.get("diagnostic"), "cache_status": event.get("cache_status"), "request_artifact": "../"+str(request.relative_to(self.root)) if request else None}
                                events.append(item)
                                self.reranks.append({"scene": scene["scene"], "query_id": query["query_id"], "method": method, "repeat": repeat, **item})
                            nodes = frame.get("benchmark_nodes", [])
                            mapping = associate_nodes(nodes, official[index]["objects"])
                            frames.append({"frame": index, "grid": self.grid(audit["occupancy_artifact"]), "path": rounded(np.asarray(audit["path_world"]).reshape(-1, 3)[:, :2]), "vehicle": rounded(poses[index, :2, 3]), "nodes": nodes, "gt": official[index]["objects"], "matches": mapping["matches"], "selected": frame["selected_node_id"], "failure": frame["failure"], "state": frame["state_after"], "replanned": frame["replanned"], "rerank": events, "active_rerank": frame["qwen_audit"].get("active_rerank"), "path_collision_cells": audit["remaining_path_collision_cells"], "lane_validation": audit["lane_validation"], "source_image": "../"+str(Path(frame["source"]).relative_to(self.root))})
                        measured = complete[repeat]
                        cases.append({"query_id": query["query_id"], "method": method, "repeat": repeat, "frames": frames, "qwen_counts": measured["qwen_audit"], "determinism": complete["determinism_passed"], "cache_zero_calls": complete["cache_zero_new_backend_calls"], "artifact": "../"+str(frames_path.relative_to(self.root))})
        return cases

    def main(self):
        self.sources.update(self.root/name for name in ('scene_selection_manifest.json', 'benchmark_queries.jsonl', 'ground_truth_frames.jsonl', 'production_job_index.json', 'statistical_report.json', 'navigation_report.json'))
        if (self.root / 'experiment_run_manifest.json').exists():
            self.sources.add(self.root / 'experiment_run_manifest.json')
        runs = read(self.root / 'experiment_run_manifest.json', {}).get('active_runs', {})
        selected = read(self.root / "scene_selection_manifest.json")["scenes"]
        queries = [json.loads(line) for line in (self.root / "benchmark_queries.jsonl").open()]
        official = {}
        for line in (self.root / "ground_truth_frames.jsonl").open():
            row = json.loads(line)
            official.setdefault(row["scene"], {})[row["frame_idx"]] = row
        scene_index = []
        for scene in sorted(selected, key=lambda r: r["scene"]):
            name = scene["scene"]
            self.sources.update((self.root/'lane_maps'/name/filename) for filename in ('lane_topology.json', 'lane_constrained_bev.npz'))
            self.sources.add(self.root/'frame0_inputs'/name/'nodes.json')
            self.sources.add(self.root/'converted'/name/'poses.txt')
            fault_path = self.root / 'experiments' / runs.get('faults', 'faults_matrix_v2') / name / 'complete.json'
            if fault_path.exists():
                self.sources.add(fault_path)
            topology = load_lane_topology(self.root / "lane_maps" / name / "lane_constrained_bev.npz")
            matrix = np.loadtxt(self.root / "converted" / name / "poses.txt").reshape(-1, 4, 4)[0]
            vehicle = np.r_[matrix[:3, 3], matrix[:2, 2]]
            scene_queries = [q for q in queries if q["scene"] == name]
            nodes = read(self.root / "frame0_inputs" / name / "nodes.json")
            normalized_nodes = [{"node_id": n["node_id"], "category": n["category"], "center": n["center"]} for n in nodes]
            association = associate_nodes(normalized_nodes, official[name][0]["objects"])
            geometry = [{"token": token, "polygon": rounded(g.polygon), "centerline": rounded(g.centerline), "start": g.start_pose, "end": g.end_pose, "outgoing": topology.transitions.get(token, [])} for token, g in topology.geometries.items()]
            data = {"scene": name, "role": scene["role"], "description": scene["description"], "queries": scene_queries, "geometry": geometry, "vehicle": rounded(vehicle[:2]), "nodes": normalized_nodes, "gt": official[name][0]["objects"], "matches": association["matches"], "static": self.static(scene, topology, vehicle), "online": self.online(scene, scene_queries, official[name]), "faults": read(fault_path), "source_image": f"../converted/{name}/color/000000.jpg"}
            text = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            (self.output / "data" / f"{name}.js").write_text("window.installScene("+text+");\n")
            scene_index.append({"scene": name, "role": scene["role"], "online_pairs": len(data["online"])//2})
            print(name, len(text), "bytes", flush=True)
        stats = read(self.root / "statistical_report.json", {})
        navigation = read(self.root / "navigation_report.json", {})
        counts = read(self.root / "progress_report.json", {}).get('counts', {})
        summary = {"scenes": scene_index, "statistics": stats, "paired_rows": [{k: r[k] for k in ("scene", "query_id", "role", "method", "query_success", "target_confirmation_correct")} for r in navigation.get("rows", [])], "progress": {"counts": counts}}
        summary['report_href'] = '../../prospective_evaluation_report.md' if (self.root / 'sampling_plan.json').exists() else '../experiment_report.md'
        summary['cohort_notice'] = (
            f'事前分层补充组：{len(selected)}个新增场景；新开发与新独立测试分别统计，原11个固定回归场景另见主报告。'
            if (self.root / 'sampling_plan.json').exists()
            else f'原{len(selected)}场景诊断组：测试此前已被访问，历史结果与限制保留；新事前分层测试见补正评测报告。'
        )
        (self.output / "summary.js").write_text("window.benchmarkSummary="+json.dumps(summary, ensure_ascii=False, allow_nan=False)+";\n")
        template = Path(__file__).with_name("lane_expansion_browser.html")
        (self.output / "index.html").write_text(template.read_text())
        write_json(self.output / "rejected_smoothing_evidence.json", {"cases": self.rejected, "all_saved_validations_reproduced": all(c["saved_validation_reproduced"] for c in self.rejected), "not_an_accepted_path": True})
        write_json(self.output / "rerank_before_after.json", {"events": self.reranks, "event_count": len(self.reranks), "changed_order_count": sum(r["response_ranking"] is not None and r["response_ranking"] != r["before_verified_ids"] for r in self.reranks), "application_and_response_are_separate": True})
        source_manifest = [{"path": str(p.relative_to(self.root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(self.sources)]
        write_json(self.output / "source_manifest.json", {"sources": source_manifest, "all_geometry_from_actual_hd_map": True, "renderer_template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(), "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "progress_counts_sha256": hashlib.sha256(json.dumps(counts, sort_keys=True).encode()).hexdigest(), "progress_scope": "Only displayed execution counts are embedded; the mutable final acceptance state is not a visualization input."})
        candidates = sorted(self.representative_candidates, key=lambda c: (c['scene'], c['query_id'], c['method']))
        additional_order = candidates
        rejected_order = sorted(self.rejected, key=lambda r: (r['scene'], r['query_id'], r['mode']))
        if (self.root / 'sampling_plan.json').exists():
            # The supplementary release explicitly orders additional cases by artifact path.
            additional_order = sorted(candidates, key=lambda c: c['artifact'])
            rejected_order = sorted(self.rejected, key=lambda c: c['artifact'])
        write_json(self.output / "representative_cases.json", {"selection_policy": read(self.root/'test_release.json')['representative_case_policy'], "default": candidates[0], "first_failure": next((c for c in additional_order if not c['success']), None), "first_accepted_strict_lane": next((c for c in additional_order if c['success'] and c['method'] == 'lane_constrained'), None), "first_rejected_smoothing": next((c for c in rejected_order if c['scene'] in {s['scene'] for s in selected if s['role'] == 'sealed_test'}), None), "all_failures_remain_browsable": True})
        complete = navigation.get("all_complete", False) and read(self.root / "retrieval_report.json", {}).get("all_complete", False)
        write_json(self.output / "visualization_manifest.json", {"entry": str(self.output / "index.html"), "scene_count": len(scene_index), "all_experiments_complete": complete, "all_required_views_present": len(scene_index) == len(selected) and bool(self.reranks) and bool(self.rejected) and all(c["saved_validation_reproduced"] for c in self.rejected), "views": ["four_map_static", "ordinary_lane_replay", "actual_per_frame_occupancy", "lane_polygon_centerline_direction", "legal_transitions", "projections", "rejected_smoothing_reconstruction", "official_gt_node_association", "rerank_before_after", "first_cache_repeat", "synthetic_fault_fallback", "scene_paired_outcomes", "cluster_bootstrap_ci"], "representative_policy": read(self.root / "test_release.json")["representative_case_policy"], "default_scene": next(s["scene"] for s in scene_index if s["role"] == "sealed_test"), "human_reviewed": False})


if __name__ == "__main__":
    Exporter(DEFAULT_ROOT).main()
