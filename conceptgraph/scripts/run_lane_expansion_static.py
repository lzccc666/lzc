"""Four-map static trials and independent retrieval metrics on frozen queries."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from omegaconf import OmegaConf

from conceptgraph.navigation.lane_constraints import (
    load_lane_topology,
    validate_lane_path,
)
from conceptgraph.navigation.lane_geometry import continuous_collision_cells
from conceptgraph.navigation.pipeline import SemanticNavigationPipeline
from conceptgraph.nuscenes.benchmark_ground_truth import (
    associate_nodes,
    retrieval_metrics,
)
from conceptgraph.nuscenes.expansion_benchmark import content_sha256
from conceptgraph.nuscenes.benchmark_fingerprints import (
    implementation_manifest,
    scene_input_manifest,
    file_sha256,
)
from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT, PROJECT_ROOT
from conceptgraph.scripts.run_incremental_semantic_navigation import (
    make_navigation_config,
    resolve_scene_inputs,
    write_json,
)
from conceptgraph.scripts.prepare_lane_expansion_frame0 import load_frame0
from conceptgraph.scripts.run_lane_expansion_benchmark import require_test_release


def run_scene(root, run, scene):
    queries = [
        json.loads(line)
        for line in (root / "benchmark_queries.jsonl").read_text().splitlines()
        if json.loads(line)["scene"] == scene
    ]
    protocol = json.loads((root / "benchmark_protocol.json").read_text())
    require_test_release(
        root, {"role": queries[0]["role"], "protocol_sha256": protocol["sha256"]}
    )
    directory = run / scene
    material = {
        "scene": scene,
        "protocol": protocol["sha256"],
        "query_ids": [q["query_id"] for q in queries],
        "implementation": implementation_manifest(PROJECT_ROOT),
        "inputs": scene_input_manifest(root, scene),
        "frame0_nodes_sha256": file_sha256(
            root / "frame0_inputs" / scene / "nodes.pkl.gz"
        ),
    }
    expected = content_sha256(material)
    completed = directory / "complete.json"
    if completed.exists():
        if json.loads(completed.read_text())["input_sha256"] != expected:
            raise ValueError("static scene inputs changed")
        return
    if directory.exists():
        raise FileExistsError(f"partial static trial: {directory}")
    directory.mkdir(parents=True)
    write_json(directory / "input_manifest.json", material)
    inputs = resolve_scene_inputs(scene, dataset_root=root / "converted")
    nodes = load_frame0(root, scene)
    associations = [
        {
            "node_id": n.node_id,
            "category": n.object_tag,
            "center": n.bbox_center.tolist(),
        }
        for n in nodes.values()
    ]
    gt = next(
        json.loads(line)
        for line in (root / "ground_truth_frames.jsonl").open()
        if json.loads(line)["scene"] == scene and json.loads(line)["frame_idx"] == 0
    )
    association = associate_nodes(associations, gt["objects"])
    write_json(directory / "association.json", association)
    camera = np.loadtxt(inputs["poses"]).reshape(-1, 4, 4)[0]
    pose = np.r_[camera[:3, 3], camera[:2, 2]]
    topology = load_lane_topology(
        root / "lane_maps" / scene / "lane_constrained_bev.npz"
    )
    with np.load(root / "lane_maps" / scene / "lane_constrained_bev.npz") as source:
        arrays = {k: source[k] for k in ("occupancy", "origin_xy", "resolution")}
    results = []
    for mode in ("free_space", "drivable_area", "lane_area", "lane_constrained"):
        maps = directory / mode
        maps.mkdir()
        occupancy = arrays["occupancy"]
        if mode == "free_space":
            occupancy = np.zeros_like(occupancy)
        elif mode == "lane_area":
            occupancy = np.maximum(occupancy, ~topology.lane_mask)
        np.savez_compressed(
            maps / "input_bev.npz", **{**arrays, "occupancy": occupancy}
        )
        for query in queries:
            settings = OmegaConf.create(
                {
                    "lane_constrained": mode == "lane_constrained",
                    "lane_map_root": str(root / "lane_maps"),
                    "online": {"relation_coordinate_frame": "ego_heading"},
                }
            )
            cfg = make_navigation_config(scene, inputs, settings)
            cfg.bev_map_path = str(maps / "input_bev.npz")
            pipeline = SemanticNavigationPipeline(cfg)
            pipeline.set_in_memory_inputs(nodes=nodes, edges=[], occupancy_grid=None)
            result = pipeline.select(
                query["query_text"], pose, run_semantic_delegate=False
            )
            output = maps / query["query_id"]
            pipeline.save_selection_result(result, output, save_visualizations=False)
            grid = pipeline._occupancy_grid
            collision_cells = []
            validation = None
            if result.plan is not None:
                cells = np.asarray(result.plan.path_grid)
                for a, b in zip(cells[:-1], cells[1:]):
                    collision_cells.extend(
                        continuous_collision_cells(a, b, grid.occupancy)
                    )
                if len(cells) == 1:
                    collision_cells.extend(
                        continuous_collision_cells(cells[0], cells[0], grid.occupancy)
                    )
                lane_cells = (
                    (result.plan.path_world[:, :2] - topology.origin_xy)
                    / topology.resolution
                )[:, ::-1]
                validation = validate_lane_path(lane_cells, topology)
                if collision_cells or (
                    mode == "lane_constrained" and not validation["valid"]
                ):
                    raise ValueError(
                        f"accepted static path failed independent audit: {output}"
                    )
            if grid is not None:
                np.savez_compressed(
                    output / "live_grid.npz",
                    occupancy=grid.occupancy,
                    origin_xy=grid.origin_xy,
                    resolution=grid.resolution,
                )
            ranking = [
                c.node_id
                for c in sorted(
                    result.evaluated_candidates,
                    key=lambda c: (
                        c.semantic_rank if c.semantic_rank is not None else 10**9
                    ),
                )
            ]
            retrieval = retrieval_metrics(
                ranking, query["accepted_instance_tokens"], association
            )
            results.append(
                {
                    "scene": scene,
                    "query_id": query["query_id"],
                    "method": mode,
                    "role": query["role"],
                    "success": result.success,
                    "retrieval": retrieval,
                    "ranking_scope": "evaluated top-K semantic candidates, independent of planning success",
                    "lane_validation": validation,
                    "collision_count": len(collision_cells),
                    "selected_node_id": result.selected_candidate.node_id
                    if result.selected_candidate
                    else None,
                    "artifact": str(output),
                }
            )
    write_json(
        completed,
        {
            "input_sha256": expected,
            "results": results,
            "independent_accepted_path_audit_passed": True,
            "frame0_node_count": len(nodes),
        },
    )
    print(scene, len(results), "static trials complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description="多场景四地图独立静态评测")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scene", action="append")
    parser.add_argument(
        "--role",
        choices=("development", "legacy_regression", "sealed_test"),
        action="append",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-scene")
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id) is None or args.workers < 1:
        raise ValueError("invalid run ID/workers")
    root = DEFAULT_ROOT
    run = root / "experiments" / args.run_id
    rows = json.loads((root / "scene_selection_manifest.json").read_text())["scenes"]
    names = [
        r["scene"]
        for r in rows
        if (not args.role or r["role"] in args.role)
        and (not args.scene or r["scene"] in args.scene)
    ]
    if args.worker_scene:
        if args.worker_scene not in {r["scene"] for r in rows}:
            raise ValueError("scene outside manifest")
        run_scene(root, run, args.worker_scene)
        return
    run.mkdir(parents=True, exist_ok=True)

    def execute(name):
        with (run / f"{name}.log").open("a") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "conceptgraph.scripts.run_lane_expansion_static",
                    "--run-id",
                    args.run_id,
                    "--worker-scene",
                    name,
                ],
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=3600,
            )
        return {"scene": name, "returncode": result.returncode}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        statuses = list(executor.map(execute, names))
    write_json(run / "execution_status.json", {"statuses": statuses})
    if any(s["returncode"] for s in statuses):
        raise RuntimeError("static evaluation failed; see per-scene logs")


if __name__ == "__main__":
    main()
