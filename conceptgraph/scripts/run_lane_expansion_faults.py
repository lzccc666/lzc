"""Bounded synthetic failure and delayed-version checks on frozen real scenes."""

from __future__ import annotations

import argparse
import copy
import json
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from conceptgraph.navigation.online import (
    OnlineSemanticNavigator,
    _PendingQwenTask,
    _node_crop_paths,
    _object_evidence_version,
)
from conceptgraph.navigation.pipeline import SemanticNavigationPipeline
from conceptgraph.nuscenes.expansion_benchmark import content_sha256
from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT
from conceptgraph.scripts.prepare_lane_expansion_frame0 import load_frame0
from conceptgraph.scripts.run_incremental_semantic_navigation import (
    make_navigation_config,
    resolve_scene_inputs,
    write_json,
)
from conceptgraph.scripts.run_lane_expansion_benchmark import require_test_release
from conceptgraph.vlm.qwen_semantic_delegate import (
    QwenDelegateConfig,
    QwenSemanticDelegate,
)


def safety_signature(result):
    return content_sha256(
        {
            "success": result.success,
            "failure_code": result.failure_code,
            "path": result.plan.path_grid if result.plan else [],
        }
    )


def run_scene(scene, run):
    root = DEFAULT_ROOT
    queries = [
        json.loads(line)
        for line in (root / "benchmark_queries.jsonl").open()
        if json.loads(line)["scene"] == scene
        and json.loads(line)["online_replay_selected"]
    ]
    protocol = json.loads((root / "benchmark_protocol.json").read_text())
    require_test_release(
        root, {"role": queries[0]["role"], "protocol_sha256": protocol["sha256"]}
    )
    directory = run / scene
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    inputs, nodes = (
        resolve_scene_inputs(scene, dataset_root=root / "converted"),
        load_frame0(root, scene),
    )
    matrix = np.loadtxt(inputs["poses"]).reshape(-1, 4, 4)[0]
    pose = np.r_[matrix[:3, 3], matrix[:2, 2]]
    cases = []
    for lane in (False, True):
        cfg = make_navigation_config(
            scene,
            inputs,
            OmegaConf.create(
                {
                    "lane_constrained": lane,
                    "lane_map_root": str(root / "lane_maps"),
                    "online": {"relation_coordinate_frame": "ego_heading"},
                }
            ),
        )
        for query in queries:
            pipeline = SemanticNavigationPipeline(cfg)
            pipeline.set_in_memory_inputs(
                nodes=copy.deepcopy(nodes), edges=[], occupancy_grid=None
            )
            baseline = pipeline.select(
                query["query_text"], pose, run_semantic_delegate=False
            )
            identity = (
                directory / f"{query['query_id']}_{'lane' if lane else 'ordinary'}"
            )
            pipeline.save_selection_result(
                baseline, identity / "baseline", save_visualizations=False
            )
            baseline_hash = safety_signature(baseline)
            for failure in ("exception", "timeout", "invalid_schema", "empty_output"):

                def backend(*args, _failure=failure, **kwargs):
                    if _failure == "exception":
                        raise RuntimeError("synthetic fault only; no CUDA model")
                    if _failure == "timeout":
                        time.sleep(0.03)
                    return (
                        "" if _failure == "empty_output" else '{"unexpected":"fault"}'
                    )

                delegate = QwenSemanticDelegate(
                    QwenDelegateConfig(
                        enabled=True,
                        use_qwen_query_parse=True,
                        use_qwen_rerank=True,
                        timeout_s=0.005,
                        model_revision="synthetic-fault-only",
                        log_dir=identity / failure / "audit",
                    ),
                    backend=backend,
                )
                try:
                    pipeline._semantic_delegate = delegate
                    pipeline._semantic_delegate_initialized = True
                    actual = pipeline.select(query["query_text"], pose)
                    audit = pipeline._last_query_audit
                    same = safety_signature(actual) == baseline_hash
                    cases.append(
                        {
                            "scene": scene,
                            "query_id": query["query_id"],
                            "lane": lane,
                            "fault": failure,
                            "baseline_legal_path_exists": baseline.success,
                            "baseline_failure": baseline.failure_code,
                            "same_safety_outcome": same,
                            "fallback_used": bool(audit.get("fallback_used")),
                            "passed": same and bool(audit.get("fallback_used")),
                            "backend_calls_synthetic": delegate.backend_call_count,
                            "failure_code": audit.get("failure_code"),
                            "path_sha256": safety_signature(actual),
                        }
                    )
                    pipeline.save_selection_result(
                        actual, identity / failure, save_visualizations=False
                    )
                finally:
                    delegate.close()
            # Actual async dispatch: hold a completion until its query binding expires.
            nav = OnlineSemanticNavigator(cfg, online_config={"qwen_enabled": False})
            try:
                nav.set_query(query["query_text"])
                nav._nodes = copy.deepcopy(nodes)
                nav._path_grid = list(baseline.plan.path_grid) if baseline.plan else []
                nav._path_world = (
                    baseline.plan.path_world.copy()
                    if baseline.plan
                    else np.empty((0, 3))
                )
                before = content_sha256(
                    {
                        "path": nav._path_world.tolist(),
                        "nodes": [(i, n.object_tag) for i, n in nav._nodes.items()],
                    }
                )
                gate = threading.Event()

                def delayed():
                    gate.wait(timeout=1.0)
                    return {
                        "used_qwen": True,
                        "normalized_output": {"confidence": 0.9, "object_tag": "truck"},
                    }

                nav._submit_qwen(
                    "query_parse",
                    delayed,
                    binding={"query_version": nav._query_version},
                    submitted_frame=0,
                )
                nav._query_version += 1
                gate.set()
                pending = list(nav._pending_qwen.values())[0]
                pending.future.result(timeout=2.0)
                events = nav._poll_qwen()
                after = content_sha256(
                    {
                        "path": nav._path_world.tolist(),
                        "nodes": [(i, n.object_tag) for i, n in nav._nodes.items()],
                    }
                )
                cases.append(
                    {
                        "scene": scene,
                        "query_id": query["query_id"],
                        "lane": lane,
                        "fault": "late_query_version",
                        "baseline_legal_path_exists": baseline.success,
                        "events": events,
                        "passed": before == after
                        and events[0]["status"] == "stale_discarded"
                        and not events[0]["applied"],
                        "model_loaded": False,
                    }
                )
                if nodes:
                    node_id = min(nodes)
                    node = nav._nodes[node_id]
                    version = _object_evidence_version(node, _node_crop_paths(node))
                    pending = _PendingQwenTask(
                        task_id=99,
                        task="object_attributes",
                        future=Future(),
                        binding={},
                        submitted_frame=0,
                        context={
                            "node_id": node_id,
                            "detector_tag": node.object_tag,
                            "object_identity": node.metadata.get("object_identity"),
                            "evidence_version": version,
                        },
                    )
                    node.metadata["num_detections"] = (
                        int(node.metadata.get("num_detections", 1)) + 1
                    )
                    applied, diagnostic = nav._accept_qwen_object_attributes(
                        pending,
                        {
                            "used_qwen": True,
                            "normalized_output": {"confidence": 0.9, "colors": ["red"]},
                        },
                    )
                    cases.append(
                        {
                            "scene": scene,
                            "query_id": query["query_id"],
                            "lane": lane,
                            "fault": "stale_object_evidence_version",
                            "baseline_legal_path_exists": baseline.success,
                            "passed": not applied
                            and diagnostic == "object_evidence_version_changed"
                            and content_sha256(nav._path_world.tolist())
                            == content_sha256(
                                baseline.plan.path_world.tolist()
                                if baseline.plan
                                else []
                            ),
                            "diagnostic": diagnostic,
                            "model_loaded": False,
                        }
                    )
                else:
                    cases.append(
                        {
                            "scene": scene,
                            "query_id": query["query_id"],
                            "lane": lane,
                            "fault": "stale_object_evidence_version",
                            "status": "not_applicable_no_real_node",
                            "passed": None,
                        }
                    )
            finally:
                nav.close()
    write_json(
        directory / "complete.json",
        {
            "cases": cases,
            "passed": all(c["passed"] is not False for c in cases),
            "real_qwen_backend_calls": 0,
            "source": "real frozen first-frame objects/maps; all fault backends explicitly synthetic",
        },
    )
    print(scene, len(cases), "fault checks complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="append")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("invalid run ID")
    run = DEFAULT_ROOT / "experiments" / args.run_id
    for scene in args.scene or []:
        run_scene(scene, run)


if __name__ == "__main__":
    main()
