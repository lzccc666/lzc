"""Nonblocking synthetic completion timeline, separate from CUDA frame barriers."""

from __future__ import annotations

import copy
import json
import threading
import time

import numpy as np
from omegaconf import OmegaConf

from conceptgraph.navigation.online import (
    OnlineSemanticNavigator,
    _node_crop_paths,
    _object_evidence_version,
)
from conceptgraph.nuscenes.expansion_benchmark import content_sha256
from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT
from conceptgraph.scripts.prepare_lane_expansion_frame0 import load_frame0
from conceptgraph.scripts.run_incremental_semantic_navigation import (
    make_navigation_config,
    resolve_scene_inputs,
    write_json,
)


def main():
    root = DEFAULT_ROOT
    destination = root / "asynchronous_fault_timeline.json"
    if destination.exists():
        raise FileExistsError(destination)
    records = []
    for row in json.loads((root / "scene_selection_manifest.json").read_text())[
        "scenes"
    ]:
        scene = row["scene"]
        if row["role"] != "sealed_test" and scene not in {"scene-0003", "scene-0916"}:
            continue
        nodes, inputs = (
            load_frame0(root, scene),
            resolve_scene_inputs(scene, dataset_root=root / "converted"),
        )
        for lane in (False, True):
            cfg = make_navigation_config(
                scene,
                inputs,
                OmegaConf.create(
                    {"lane_constrained": lane, "lane_map_root": str(root / "lane_maps")}
                ),
            )
            for fault in ("late_query", "expired_object_evidence"):
                events, gate = [], threading.Event()
                nav = OnlineSemanticNavigator(
                    cfg, online_config={"qwen_enabled": False}
                )
                try:
                    nav.set_query("前往当前画面中小汽车附近")
                    nav._nodes = copy.deepcopy(nodes)
                    run = (
                        "static_test_v1"
                        if row["role"] == "sealed_test"
                        else "static_development_all_v3"
                    )
                    case = (
                        root
                        / "experiments"
                        / run
                        / scene
                        / ("lane_constrained" if lane else "drivable_area")
                        / f"{scene}-q00"
                    )
                    selected = json.loads(
                        (case / "target_selection_result.json").read_text()
                    )
                    if selected["success"]:
                        path = json.loads(
                            (case / "semantic_navigation_plan.json").read_text()
                        )["path"]
                        nav._path_grid = path["grid_rc"]
                        nav._path_world = np.asarray(path["world_xyz"])
                    baseline = content_sha256(
                        {"path": nav._path_world.tolist(), "grid": nav._path_grid}
                    )
                    start = time.perf_counter()

                    def mark(event, **data):
                        events.append(
                            {
                                "event": event,
                                "elapsed_s": time.perf_counter() - start,
                                **data,
                            }
                        )

                    def delayed():
                        mark("backend_started_synthetic")
                        if not gate.wait(timeout=2.0):
                            raise TimeoutError("synthetic gate wait exceeded")
                        mark("backend_completed_synthetic")
                        return {
                            "used_qwen": True,
                            "normalized_output": {
                                "confidence": 0.9,
                                "target_categories": ["truck"],
                            },
                        }

                    if fault == "late_query":
                        task, binding, context = (
                            "query_parse",
                            {"query_version": nav._query_version},
                            {},
                        )
                    else:
                        if not nav._nodes:
                            records.append(
                                {
                                    "scene": scene,
                                    "lane": lane,
                                    "fault": fault,
                                    "status": "not_applicable_no_real_node",
                                }
                            )
                            continue
                        node_id = min(nav._nodes)
                        node = nav._nodes[node_id]
                        task, binding, context = (
                            "object_attributes",
                            {},
                            {
                                "node_id": node_id,
                                "detector_tag": node.object_tag,
                                "object_identity": node.metadata.get("object_identity"),
                                "evidence_version": _object_evidence_version(
                                    node, _node_crop_paths(node)
                                ),
                            },
                        )
                    mark("submitted", frame_idx=0)
                    nav._submit_qwen(
                        task,
                        delayed,
                        binding=binding,
                        submitted_frame=0,
                        context=context,
                    )
                    pending = next(iter(nav._pending_qwen.values()))
                    nonblocking = not pending.future.done()
                    mark("submission_returned", backend_still_pending=nonblocking)
                    if fault == "late_query":
                        nav._query_version += 1
                    else:
                        node.metadata["num_detections"] = (
                            int(node.metadata.get("num_detections", 1)) + 1
                        )
                    nav._last_frame_idx = 1
                    mark("input_version_advanced", frame_idx=1)
                    gate.set()
                    pending.future.result(timeout=3.0)
                    result = nav._poll_qwen()[0]
                    mark(
                        "polled_discarded"
                        if not result["applied"]
                        else "polled_applied",
                        frame_idx=1,
                        status=result["status"],
                        diagnostic=result.get("diagnostic"),
                        applied=result["applied"],
                    )
                    unchanged = baseline == content_sha256(
                        {"path": nav._path_world.tolist(), "grid": nav._path_grid}
                    )
                    passed = (
                        nonblocking
                        and not result["applied"]
                        and unchanged
                        and (
                            result["status"] == "stale_discarded"
                            if fault == "late_query"
                            else result["diagnostic"]
                            == "object_evidence_version_changed"
                        )
                    )
                    records.append(
                        {
                            "scene": scene,
                            "lane": lane,
                            "fault": fault,
                            "passed": passed,
                            "timeline": events,
                            "same_saved_safe_path": unchanged,
                            "baseline_legal_path_exists": selected["success"],
                            "real_model_calls": 0,
                        }
                    )
                finally:
                    gate.set()
                    nav.close()
    report = {
        "passed": all(r.get("passed", True) for r in records),
        "records": records,
        "scope": "real saved static inputs/path + synthetic asynchronous dispatcher, not CUDA model latency and not a real-time vehicle trial",
        "controlled_gpu_frame_barrier_is_separate": True,
    }
    write_json(destination, report)
    if not report["passed"]:
        raise RuntimeError("asynchronous version safety failed")
    print("asynchronous checks", len(records), "passed", flush=True)


if __name__ == "__main__":
    main()
