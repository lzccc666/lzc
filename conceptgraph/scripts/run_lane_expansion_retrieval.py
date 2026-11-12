"""All frozen query retrieval and static confirmation, separate from online runs."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from dataclasses import fields

import numpy as np
import torch
from omegaconf import OmegaConf

from conceptgraph.navigation.pipeline import SemanticNavigationPipeline
from conceptgraph.nuscenes.benchmark_fingerprints import (
    implementation_manifest,
    scene_input_manifest,
)
from conceptgraph.nuscenes.benchmark_ground_truth import (
    associate_nodes,
    retrieval_metrics,
)
from conceptgraph.nuscenes.expansion_benchmark import content_sha256
from conceptgraph.scripts.lane_experiment_metrics import qwen_audit_metrics
from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT, PROJECT_ROOT
from conceptgraph.scripts.prepare_lane_expansion_frame0 import load_frame0
from conceptgraph.scripts.run_incremental_semantic_navigation import (
    DEFAULT_CONFIG,
    make_navigation_config,
    resolve_scene_inputs,
    write_json,
)
from conceptgraph.scripts.run_lane_expansion_benchmark import require_test_release
from conceptgraph.vlm.qwen_semantic_delegate import (
    QwenDelegateConfig,
    QwenSemanticDelegate,
)
from conceptgraph.vlm.qwen_vl_chat import QwenVLChat


def evaluate_query(root, run, query, variant, lane, backend):
    protocol = json.loads((root / "benchmark_protocol.json").read_text())
    require_test_release(
        root, {"role": query["role"], "protocol_sha256": protocol["sha256"]}
    )
    directory = run / query["query_id"] / f"{variant}_{'lane' if lane else 'drivable'}"
    material = {
        "query": query["query_id"],
        "variant": variant,
        "lane": lane,
        "protocol": protocol["sha256"],
        "code": implementation_manifest(PROJECT_ROOT),
        "inputs": scene_input_manifest(root, query["scene"]),
    }
    signature = content_sha256(material)
    complete = directory / "complete.json"
    if complete.exists():
        previous = json.loads(complete.read_text())
        if previous["input_sha256"] != signature:
            raise ValueError(f"completed retrieval input changed: {directory}")
        return
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    write_json(directory / "input_manifest.json", material)
    settings = OmegaConf.load(DEFAULT_CONFIG)
    values = {
        k: v
        for k, v in OmegaConf.to_container(settings.qwen_online, resolve=True).items()
        if k in {f.name for f in fields(QwenDelegateConfig)}
    }
    values.update(enabled=variant != "off", cache_dir=str(directory / "cache"))
    if variant == "base":
        values.update(adapter_path=None, adapter_revision="base-model-no-adapter")
    inputs = resolve_scene_inputs(query["scene"], dataset_root=root / "converted")
    original_nodes = load_frame0(root, query["scene"])
    gt = next(
        json.loads(line)
        for line in (root / "ground_truth_frames.jsonl").open()
        if json.loads(line)["scene"] == query["scene"]
        and json.loads(line)["frame_idx"] == 0
    )
    association = associate_nodes(
        [
            {
                "node_id": n.node_id,
                "category": n.object_tag,
                "center": n.bbox_center.tolist(),
            }
            for n in original_nodes.values()
        ],
        gt["objects"],
    )
    camera = np.loadtxt(inputs["poses"]).reshape(-1, 4, 4)[0]
    pose = np.r_[camera[:3, 3], camera[:2, 2]]
    cfg = make_navigation_config(
        query["scene"],
        inputs,
        OmegaConf.create(
            {
                "lane_constrained": lane,
                "lane_map_root": str(root / "lane_maps"),
                "online": {"relation_coordinate_frame": "ego_heading"},
            }
        ),
    )
    phases = {}
    for repeat in ("first", "cache_repeat"):
        started = time.perf_counter()
        values["log_dir"] = str(directory / repeat / "qwen_audit")
        delegate = (
            QwenSemanticDelegate(QwenDelegateConfig(**values), backend=backend)
            if variant != "off"
            else None
        )
        pipeline = SemanticNavigationPipeline(cfg, semantic_delegate=delegate)
        pipeline.set_in_memory_inputs(
            nodes=copy.deepcopy(original_nodes), edges=[], occupancy_grid=None
        )
        try:
            result = pipeline.select(
                query["query_text"], pose, run_semantic_delegate=variant != "off"
            )
            pipeline.save_selection_result(
                result, directory / repeat, save_visualizations=False
            )
            ranking = [
                c.node_id
                for c in sorted(
                    result.evaluated_candidates, key=lambda c: c.semantic_rank or 10**9
                )
            ]
            selected = (
                result.selected_candidate.node_id if result.selected_candidate else None
            )
            metrics = retrieval_metrics(
                ranking, query["accepted_instance_tokens"], association
            )
            mapped = {m["node_id"]: m["instance_token"] for m in association["matches"]}
            confirmation = (
                mapped.get(selected) in query["accepted_instance_tokens"]
                if query["accepted_instance_tokens"]
                else selected is None
            )
            audit = pipeline._last_query_audit or {}
            output = audit.get("normalized_output") or {}
            expected = protocol["templates"][int(query["query_id"].rsplit("q", 1)[1])]
            fields_correct = {
                "category": output.get("target_categories") == [expected["category"]],
                "nearest": bool(output.get("prefer_nearest")) == expected["nearest"],
                "relation": output.get("relation")
                == (expected.get("relation") or "near"),
            }
            canonical = {
                "success": result.success,
                "failure_code": result.failure_code,
                "selected": selected,
                "ranking": ranking,
                "path": result.plan.path_grid if result.plan else [],
                "parse": output if delegate else None,
                "rerank_recommendation": result.debug.get(
                    "qwen_candidate_rerank", {}
                ).get("normalized_output"),
            }
            phases[repeat] = {
                "retrieval": metrics,
                "target_confirmation_correct": confirmation,
                "query_parse_field_correct": fields_correct if delegate else None,
                "selected_node_id": selected,
                "planning_success": result.success,
                "failure_code": result.failure_code,
                "content_sha256": content_sha256(canonical),
                "backend_call_count": delegate.backend_call_count if delegate else 0,
                "qwen_audit": qwen_audit_metrics(
                    directory / repeat / "qwen_audit", delegate.backend_call_count
                )
                if delegate
                else None,
                "elapsed_s": time.perf_counter() - started,
                "static_rerank_applied_to_selection": False,
            }
        finally:
            if delegate:
                delegate.close()
    result = {
        "input_sha256": signature,
        "scene": query["scene"],
        "query_id": query["query_id"],
        "role": query["role"],
        "variant": variant,
        "lane": lane,
        "ground_truth": {
            "accepted_instances": query["accepted_instance_tokens"],
            "association": association,
        },
        "phases": phases,
        "determinism_passed": phases["first"]["content_sha256"]
        == phases["cache_repeat"]["content_sha256"],
        "cache_repeat_zero_new_backend_calls": phases["cache_repeat"][
            "backend_call_count"
        ]
        == 0,
        "scope": "frozen first-frame retrieval and static target confirmation; online results reported separately",
    }
    result["passed"] = (
        result["determinism_passed"] and result["cache_repeat_zero_new_backend_calls"]
    )
    write_json(complete, result)
    if not result["passed"]:
        raise RuntimeError(f"retrieval determinism/cache failed: {directory}")
    print(query["query_id"], variant, lane, "complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scene", action="append")
    parser.add_argument(
        "--variant", choices=["off", "base", "adapter"], action="append"
    )
    args = parser.parse_args()
    if __import__("pathlib").Path(args.run_id).name != args.run_id:
        raise ValueError("invalid run ID")
    root, backend = DEFAULT_ROOT, None
    run = root / "experiments" / args.run_id
    queries = [
        q
        for line in (root / "benchmark_queries.jsonl").open()
        if (q := json.loads(line))["role"] == "sealed_test"
        or q["scene"] in {"scene-0003", "scene-0916"}
    ]
    if args.scene:
        queries = [q for q in queries if q["scene"] in args.scene]
    for variant in args.variant or ["off", "adapter", "base"]:
        backend = None
        gc.collect()
        torch.cuda.empty_cache()
        if variant != "off":
            settings = OmegaConf.load(DEFAULT_CONFIG)
            backend = QwenVLChat(
                str(settings.qwen_online.model_name_or_path),
                adapter_path=str(settings.qwen_online.adapter_path)
                if variant == "adapter"
                else None,
                torch_dtype=getattr(torch, str(settings.qwen_online.torch_dtype)),
                max_image_side=int(settings.qwen_online.max_image_side),
            )
        for query in queries:
            for lane in (False, True):
                evaluate_query(root, run, query, variant, lane, backend)


if __name__ == "__main__":
    main()
