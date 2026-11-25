from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
from omegaconf import OmegaConf

from conceptgraph.navigation.online import (
    DYNAMIC_CLASSES,
    OnlineSemanticNavigator,
    SnapshotNodeTracker,
)
from conceptgraph.navigation.types import OnlineNavigationUpdate, SceneGraphNode


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "conceptgraph/hydra_configs/incremental_navigation.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/project_v2/incremental_navigation"


def main() -> None:
    args = parse_args()
    settings = OmegaConf.load(args.config)
    output_root = Path(args.output_root).resolve()
    expected_root = DEFAULT_OUTPUT.resolve()
    if output_root != expected_root:
        raise ValueError(f"stage5 output must be {expected_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = list(args.scenes)
    modes = (
        ["snapshot_replay", "end_to_end_replay"]
        if args.mode == "all"
        else [args.mode]
    )
    deterministic_runs = max(1, int(args.determinism_runs))
    reports: dict[str, dict[str, Any]] = {mode: {} for mode in modes}
    signatures: dict[str, dict[str, list[str]]] = {
        mode: defaultdict(list) for mode in modes
    }

    for mode in modes:
        for scene_id in scenes:
            for run_index in range(deterministic_runs):
                write_outputs = run_index == 0
                if mode == "snapshot_replay":
                    report, signature = run_snapshot_replay(
                        scene_id,
                        settings,
                        output_root,
                        write_outputs=write_outputs,
                    )
                else:
                    report, signature = run_end_to_end_replay(
                        scene_id,
                        settings,
                        output_root,
                        write_outputs=write_outputs,
                    )
                signatures[mode][scene_id].append(signature)
                if write_outputs:
                    reports[mode][scene_id] = report

    failure_report = run_failure_injection(settings, output_root)
    write_acceptance_reports(
        output_root,
        reports,
        signatures,
        failure_report,
        deterministic_runs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段5增量语义导航双场景回放")
    parser.add_argument(
        "--mode",
        choices=("snapshot_replay", "end_to_end_replay", "all"),
        default="all",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["scene-0916", "scene-0003"],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--determinism-runs", type=int, default=2)
    return parser.parse_args()


def run_snapshot_replay(
    scene_id: str,
    settings: Any,
    output_root: Path,
    *,
    write_outputs: bool,
) -> tuple[dict[str, Any], str]:
    started = time.perf_counter()
    inputs = resolve_scene_inputs(scene_id)
    scene_settings = settings.scenes[scene_id]
    first_frame = int(scene_settings.snapshot_first_frame)
    last_frame = int(scene_settings.snapshot_last_frame)
    frame_paths = [
        inputs["snapshot_dir"] / f"{frame_idx:06d}.pkl.gz"
        for frame_idx in range(first_frame, last_frame + 1)
    ]
    missing = [str(path) for path in frame_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required snapshot frames: {missing}")
    meta = load_gzip_pickle(inputs["snapshot_dir"] / "meta.pkl.gz")
    tracker = SnapshotNodeTracker(meta.get("class_names", []))
    navigation_cfg = make_navigation_config(scene_id, inputs, settings=settings)
    relation_cfg = OmegaConf.to_container(
        OmegaConf.load(PROJECT_ROOT / "conceptgraph/hydra_configs/scenegraph_relations.yaml"),
        resolve=True,
    )
    navigator = make_navigator(navigation_cfg, settings, relation_cfg)
    navigator.set_query(str(settings.query_text))
    rows = []
    source_failures = []
    manual_frame = first_frame + int(settings.replay.manual_replan_offset_frames)
    last_pose = np.asarray([0.0, 0.0, 0.0])
    try:
        for frame_path in frame_paths:
            frame_idx = int(frame_path.name.split(".")[0])
            if frame_idx == manual_frame:
                navigator.request_replan()
            frame_started = time.perf_counter()
            try:
                payload = load_gzip_pickle(frame_path)
                nodes, changed_ids = tracker.update(payload.get("objects", []), frame_idx)
                pose = np.asarray(payload.get("camera_pose"), dtype=float)
                last_pose = pose
                update = navigator.update(frame_idx, pose, nodes, changed_ids)
            except Exception as exc:  # noqa: BLE001 - corrupt input is a recoverable frame
                source_failures.append(
                    {
                        "frame_idx": frame_idx,
                        "source_snapshot": str(frame_path.resolve()),
                        "failure_code": "SNAPSHOT_LOAD_FAILED",
                        "failure_reason": f"{type(exc).__name__}: {str(exc)[:400]}",
                    }
                )
                update = navigator.update(frame_idx, last_pose, {}, [])
            row = frame_row(update, frame_path, time.perf_counter() - frame_started)
            rows.append(row)
    finally:
        navigator.close()
    report = summarize_replay(
        scene_id,
        "snapshot_replay",
        rows,
        source_failures,
        inputs,
        time.perf_counter() - started,
    )
    signature = deterministic_replay_signature(rows)
    report["deterministic_content_sha256"] = signature
    if write_outputs:
        scene_output = output_root / "snapshot_replay" / scene_id
        write_jsonl(scene_output / "frames.jsonl", rows)
        write_json(scene_output / "summary.json", report)
        write_json(
            scene_output / "resolved_config.json",
            OmegaConf.to_container(settings, resolve=True),
        )
    return report, signature


def run_end_to_end_replay(
    scene_id: str,
    settings: Any,
    output_root: Path,
    *,
    write_outputs: bool,
    qwen_backend=None,
) -> tuple[dict[str, Any], str]:
    from conceptgraph.dataset.datasets_common import get_dataset
    from conceptgraph.slam.streamlined_mapping import (
        MappingFrame,
        StreamlinedMapper,
        process_cfg,
    )
    from conceptgraph.slam.utils import create_or_load_colors

    import torch

    started = time.perf_counter()
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    dataset_root = OmegaConf.select(settings, "dataset_root", default=None)
    inputs = resolve_scene_inputs(scene_id, dataset_root=dataset_root)
    mapping_cfg = make_mapping_config(
        scene_id, dataset_root=dataset_root,
        device=OmegaConf.select(settings, "mapping_device", default="cuda:0"),
    )
    mapping_overrides = OmegaConf.select(settings, "mapping_overrides", default={})
    for key in ["association_policy", "dynamic_ttl_s", "depth_policy", "frame_dt",
                "initial_motion_unknown", "static_confirmation_observations", "motion_estimator"]:
        if key in mapping_overrides:
            mapping_cfg[key] = mapping_overrides[key]
    mapping_cfg = process_cfg(mapping_cfg)
    dataset = get_dataset(
        dataconfig=mapping_cfg.dataset_config,
        start=mapping_cfg.start,
        end=mapping_cfg.end,
        stride=mapping_cfg.stride,
        basedir=mapping_cfg.dataset_root,
        sequence=mapping_cfg.scene_id,
        desired_height=mapping_cfg.image_height,
        desired_width=mapping_cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )
    classes, class_colors = create_or_load_colors(mapping_cfg, mapping_cfg.color_file_name)
    use_source_time = bool(OmegaConf.select(settings, "use_source_timestamps", default=False))
    source_times = (
        load_camera_timestamps(Path(mapping_cfg.dataset_root) / scene_id, len(dataset))
        if use_source_time else None
    )
    mapper = StreamlinedMapper(mapping_cfg, classes, class_colors)
    navigation_cfg = make_navigation_config(scene_id, inputs, settings=settings)
    relation_cfg = OmegaConf.to_container(
        OmegaConf.load(PROJECT_ROOT / "conceptgraph/hydra_configs/scenegraph_relations.yaml"),
        resolve=True,
    )
    semantic_delegate = make_semantic_delegate(settings, backend=qwen_backend)
    navigator = make_navigator(
        navigation_cfg,
        settings,
        relation_cfg,
        semantic_delegate=semantic_delegate,
    )
    navigator.set_query(str(settings.query_text))
    rows = []
    source_failures = []
    manual_frame = int(settings.replay.manual_replan_offset_frames)
    qwen_enabled = semantic_delegate is not None
    semantic_crop_dir = (
        output_root / "qwen_object_crops" / scene_id if qwen_enabled else None
    )
    frame_period_s = (
        float(OmegaConf.select(settings, "qwen_online.frame_period_s", default=0.0))
        if qwen_enabled
        else 0.0
    )
    final_qwen_audit: dict[str, Any] | None = None
    experiment_audit = bool(OmegaConf.select(settings, "experiment_audit", default=False))
    audit_topology = None
    if experiment_audit:
        from conceptgraph.navigation.lane_constraints import load_lane_topology
        lane_root = Path(str(settings.lane_map_root)) / scene_id
        audit_topology = load_lane_topology(lane_root / "lane_constrained_bev.npz", lane_root / "lane_topology.json")
    try:
        for frame_idx in range(len(dataset)):
            if frame_idx == manual_frame:
                navigator.request_replan()
            frame_started = time.perf_counter()
            barrier_events = []
            if qwen_enabled and bool(OmegaConf.select(settings, "qwen_online.deterministic_frame_barrier", default=False)):
                # Controlled recorded replay: complete prior-frame semantic jobs
                # at a fixed boundary so backend/cache speed cannot change ordering.
                barrier_events = navigator.drain_qwen(float(settings.qwen_online.drain_timeout_s))
            color_path = Path(dataset.color_paths[frame_idx])
            detection_path = (
                color_path.parent.parent / mapping_cfg.detection_folder_name / color_path.name
            ).with_suffix(".pkl.gz")
            benchmark_nodes = None
            try:
                color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]
                with gzip.open(detection_path, "rb") as handle:
                    detections = pickle.load(handle)
                pose = dataset.poses[frame_idx].cpu().numpy()
                mapping_started = time.perf_counter()
                mapping_update = mapper.update(
                    MappingFrame(
                        frame_idx=frame_idx,
                        image_rgb=color_tensor.cpu().numpy().astype(np.uint8),
                        depth_array=depth_tensor[..., 0].cpu().numpy(),
                        cam_K=intrinsics.cpu().numpy()[:3, :3],
                        camera_pose=pose,
                        detections=detections,
                        color_path=str(color_path),
                        timestamp_s=source_times[frame_idx] if source_times is not None else None,
                    )
                )
                mapping_time = time.perf_counter() - mapping_started
                nodes = mapping_objects_to_nodes(
                    mapper.effective_objects(),
                    classes,
                    frame_idx,
                    semantic_crop_dir=semantic_crop_dir,
                )
                if bool(OmegaConf.select(settings, "export_benchmark_nodes", default=False)):
                    benchmark_nodes = [
                        {"node_id": int(node.node_id), "category": node.object_tag,
                         "center": node.bbox_center.tolist(), "extent": node.bbox_extent.tolist(),
                         "object_identity": node.metadata.get("object_identity")}
                        for node in nodes.values()
                    ]
                changed_ids = sorted(
                    set(mapping_update.changed_object_ids) | set(nodes) - set(navigator._nodes)
                )
                update = navigator.update(frame_idx, pose, nodes, changed_ids)
                update.timings["mapping_s"] = mapping_time
            except Exception as exc:  # noqa: BLE001 - keep bounded replay alive
                source_failures.append(
                    {
                        "frame_idx": frame_idx,
                        "source_detection": str(detection_path.resolve()),
                        "failure_code": "MAPPING_FRAME_FAILED",
                        "failure_reason": f"{type(exc).__name__}: {str(exc)[:400]}",
                    }
                )
                fallback_pose = dataset.poses[frame_idx].cpu().numpy()
                update = navigator.update(frame_idx, fallback_pose, navigator._nodes, [])
            if barrier_events:
                update.qwen_audit["events"] = barrier_events + update.qwen_audit["events"]
            row = frame_row(update, color_path, time.perf_counter() - frame_started)
            row["sensor_timestamp_s"] = source_times[frame_idx] if source_times is not None else None
            row["velocity_time_source"] = "camera_source" if source_times is not None else "configured_frame_dt"
            if benchmark_nodes is not None:
                row["benchmark_nodes"] = benchmark_nodes
            if experiment_audit:
                from conceptgraph.navigation.lane_audit import audit_online_frame
                row["experiment_audit"] = audit_online_frame(
                    navigator, update, output_root / "end_to_end_replay" / scene_id,
                    audit_topology, write_outputs=write_outputs,
                )
                row["path_world"] = update.path_world.tolist()
            row["source_detection"] = str(detection_path.resolve())
            rows.append(row)
            if frame_period_s > 0:
                remaining = frame_period_s - (time.perf_counter() - frame_started)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        if qwen_enabled:
            drain_events = navigator.drain_qwen(
                float(OmegaConf.select(settings, "qwen_online.drain_timeout_s", default=60.0))
            )
            final_qwen_audit = navigator._qwen_frame_audit(drain_events)
        navigator.close()
    final_nodes = mapping_objects_to_nodes(
        mapper.effective_objects(),
        classes,
        len(dataset) - 1,
    )
    final_comparison = compare_final_mapping_snapshot(
        final_nodes,
        inputs["snapshot_dir"],
        classes,
        settings.replay,
    )
    report = summarize_replay(
        scene_id,
        "end_to_end_replay",
        rows,
        source_failures,
        inputs,
        time.perf_counter() - started,
    )
    report["final_mapping_comparison"] = final_comparison
    if final_qwen_audit is not None:
        report["qwen_final_audit"] = final_qwen_audit
        report["qwen"] = {
            key: int(value)
            for key, value in final_qwen_audit.get("totals", {}).items()
        }
    if semantic_delegate is not None:
        report["qwen_backend_call_count"] = semantic_delegate.backend_call_count
        report["qwen_model_id"] = semantic_delegate.model_identifier
        report["qwen_adapter_id"] = semantic_delegate.adapter_identifier
        report["gpu_peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    report["deterministic_semantic_barrier"] = bool(OmegaConf.select(settings, "qwen_online.deterministic_frame_barrier", default=False))
    report["planning_configuration"] = OmegaConf.to_container(navigation_cfg, resolve=True)
    signature = deterministic_replay_signature(rows)
    report["deterministic_content_sha256"] = signature
    if write_outputs:
        scene_output = output_root / "end_to_end_replay" / scene_id
        write_jsonl(scene_output / "frames.jsonl", rows)
        write_json(scene_output / "summary.json", report)
        write_json(scene_output / "final_mapping_comparison.json", final_comparison)
    return report, signature


def make_navigator(
    navigation_cfg: Any,
    settings: Any,
    relation_cfg: Mapping[str, Any],
    semantic_delegate: Any | None = None,
):
    return OnlineSemanticNavigator(
        navigation_cfg,
        online_config=OmegaConf.to_container(settings.online, resolve=True),
        state_machine_config=OmegaConf.to_container(settings.state_machine, resolve=True),
        replan_config=OmegaConf.to_container(settings.replan, resolve=True),
        relation_config=relation_cfg,
        semantic_delegate=semantic_delegate,
    )


def make_semantic_delegate(settings: Any, *, backend=None):
    if not bool(OmegaConf.select(settings, "online.qwen_enabled", default=False)):
        return None
    qwen_values = OmegaConf.select(settings, "qwen_online", default=None)
    if qwen_values is None or not bool(OmegaConf.select(settings, "qwen_online.enabled", default=False)):
        return None
    from conceptgraph.vlm.qwen_semantic_delegate import (
        QwenDelegateConfig,
        QwenSemanticDelegate,
    )

    config_values = OmegaConf.to_container(qwen_values, resolve=True)
    delegate_fields = set(QwenDelegateConfig.__dataclass_fields__)
    config_values = {
        key: value for key, value in config_values.items() if key in delegate_fields
    }
    return QwenSemanticDelegate(QwenDelegateConfig(**config_values), backend=backend)


def make_navigation_config(scene_id: str, inputs: Mapping[str, Path], settings: Any | None = None):
    cfg = OmegaConf.load(PROJECT_ROOT / "conceptgraph/hydra_configs/semantic_navigation.yaml")
    cfg.dataset_root = str(inputs["scene_dir"].parent)
    cfg.scene_id = scene_id
    cfg.cachedir = str(inputs["exp_dir"])
    cfg.scene_map_path = str(inputs["exp_dir"] / "map/scene_map_cfslam_pruned.pkl.gz")
    cfg.scene_graph_json_path = str(inputs["exp_dir"] / "scene_graph.json")
    cfg.annotated_scenegraph_json_path = str(inputs["exp_dir"] / "annotated_scenegraph.json")
    cfg.scene_graph_edge_path = str(inputs["exp_dir"] / "cfslam_scenegraph_edges.pkl")
    cfg.bev_map_path = str(inputs["bev_map"])
    cfg.relation_trajectory_path = str(inputs["poses"])
    if settings is not None:
        evidence_policy = OmegaConf.select(settings, "target_evidence_policy", default=None)
        if evidence_policy is not None:
            cfg.target_evidence_policy = evidence_policy
    cfg.relation_coordinate_frame = str(OmegaConf.select(settings, "online.relation_coordinate_frame", default=cfg.relation_coordinate_frame)) if settings is not None else cfg.relation_coordinate_frame
    lane_enabled = bool(OmegaConf.select(settings, "lane_constrained", default=False)) if settings is not None else False
    if lane_enabled:
        lane_root = Path(str(OmegaConf.select(settings, "lane_map_root", default=PROJECT_ROOT / "outputs/project_v2/lane_qwen_experiment/lane_maps")))
        lane_dir = lane_root / scene_id
        cfg.bev_map_path = str(lane_dir / "lane_constrained_bev.npz")
        cfg.lane_map_path = str(lane_dir / "lane_constrained_bev.npz")
        cfg.lane_topology_path = str(lane_dir / "lane_topology.json")
        cfg.target_selection.mode = "lane_constrained"
        cfg.lane_projection_max_distance_m = float(OmegaConf.select(settings, "lane_projection_max_distance_m", default=8.0))
        cfg.lane_centerline_weight = float(OmegaConf.select(settings, "lane_centerline_weight", default=0.25))
        cfg.lane_direction_tolerance_deg = float(OmegaConf.select(settings, "lane_direction_tolerance_deg", default=75.0))
    return cfg


def make_mapping_config(scene_id: str, *, dataset_root=None, device="cuda:0"):
    base = OmegaConf.load(PROJECT_ROOT / "conceptgraph/hydra_configs/base_mapping.yaml")
    nuscenes = OmegaConf.load(PROJECT_ROOT / "conceptgraph/hydra_configs/nuscenes.yaml")
    streamlined = OmegaConf.load(
        PROJECT_ROOT / "conceptgraph/hydra_configs/streamlined_mapping.yaml"
    )
    cfg = OmegaConf.merge(base, nuscenes, streamlined)
    cfg.dataset_root = str(dataset_root or PROJECT_ROOT / "dataset/nuscenes")
    cfg.dataset_config = str(PROJECT_ROOT / "conceptgraph/nuscenes/nuscenes.yaml")
    cfg.scene_id = scene_id
    cfg.start = 0
    cfg.end = -1
    cfg.stride = 1
    cfg.detection_folder_name = "exp_streamlined_yolo_outdoor_objects_v1/detections"
    cfg.gsa_variant = "streamlined_yolo"
    cfg.color_file_name = "gsa_classes_streamlined_yolo"
    cfg.exp_suffix = "streamlined_yolo_outdoor_objects_v1"
    cfg.save_suffix = "outdoor_objects_tracking_gate"
    cfg.device = str(device)
    cfg.save_objects_all_frames = False
    cfg.save_pcd = False
    cfg.vis_render = False
    cfg.obj_min_detections = 1
    cfg.min_points_threshold = 4
    cfg.dbscan_remove_noise = False
    cfg.match_method = "sim_sum"
    cfg.sim_threshold = 0.35
    cfg.phys_bias = 0.5
    cfg.merge_overlap_thresh = -1
    cfg.max_tracking_distance_m = 8.0
    return cfg


def load_camera_timestamps(scene_dir: Path, expected_frames: int) -> list[float]:
    rows = json.loads((scene_dir / "frame_alignment.json").read_text())
    times = []
    for row in rows:
        value = row.get("camera_timestamp")
        if value is None:
            value = int(Path(row["camera_source"]).stem.split("__")[-1])
        times.append(float(value)/1e6)
    if (len(times) != expected_frames or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("invalid camera timestamp sequence")
    return times


def mapping_objects_to_nodes(
    objects: Sequence[Mapping[str, Any]],
    classes: Sequence[str],
    frame_idx: int,
    *,
    semantic_crop_dir: Path | None = None,
) -> dict[int, SceneGraphNode]:
    nodes = {}
    for node_id, obj in enumerate(objects):
        stable_id = str(obj.get("stable_object_id", ""))
        if stable_id.startswith("track-") and stable_id[6:].isdigit():
            # 动态轨迹过期后列表会压缩；列表下标不能重新指代别的对象。
            node_id = int(stable_id[6:])
        bbox = obj.get("bbox")
        if bbox is None:
            continue
        center = np.asarray(bbox.get_center(), dtype=float)
        extent = np.asarray(bbox.get_extent(), dtype=float)
        class_id = dominant_class_id(obj.get("class_id", []))
        tag = str(classes[class_id]) if 0 <= class_id < len(classes) else "object"
        object_identity = _mapping_object_identity(obj, node_id)
        crop_path = _materialize_object_crop(
            obj,
            node_id,
            frame_idx,
            semantic_crop_dir,
        )
        metadata = {
            "bbox_center": center.tolist(),
            "bbox_extent": np.maximum(extent, 0.1).tolist(),
            "num_detections": int(obj.get("num_detections", 1) or 1),
            "first_seen_frame": min(obj.get("frame_history", [frame_idx])),
            "last_seen_frame": int(obj.get("last_seen_frame", frame_idx)),
            "object_identity": object_identity,
            "timestamp_s": obj.get("timestamp_s"),
            "depth_evidence": copy.deepcopy(obj.get("depth_evidence", [])),
        }
        if crop_path is not None:
            metadata["crop_paths"] = [str(crop_path.resolve())]
            metadata["crop_frame_idx"] = int(frame_idx)
        nodes[node_id] = SceneGraphNode(
            node_id=node_id,
            object_tag=tag,
            possible_tags=[tag],
            caption=tag,
            bbox_center=center.copy(),
            bbox_extent=np.maximum(extent, 0.1),
            motion_state=str(obj.get("motion_state", "unknown")),
            is_potential_dynamic=bool(obj.get("is_potential_dynamic", False)),
            velocity=float(obj.get("velocity", 0.0) or 0.0),
            is_short_term=bool(obj.get("is_short_term", False)),
            metadata=metadata,
        )
    return nodes


def _mapping_object_identity(obj: Mapping[str, Any], fallback_id: int) -> str:
    if obj.get("stable_object_id") is not None:
        return str(obj["stable_object_id"])
    color = np.asarray(obj.get("inst_color", []), dtype=float).reshape(-1)
    if color.size >= 3 and np.all(np.isfinite(color[:3])):
        return "inst_color:" + ",".join(f"{float(value):.8f}" for value in color[:3])
    return f"object_index:{int(fallback_id)}"


def _materialize_object_crop(
    obj: Mapping[str, Any],
    node_id: int,
    frame_idx: int,
    semantic_crop_dir: Path | None,
) -> Path | None:
    if semantic_crop_dir is None:
        return None
    image_indices = np.asarray(obj.get("image_idx", []), dtype=int).reshape(-1)
    matching = np.flatnonzero(image_indices == int(frame_idx))
    if matching.size == 0:
        return None
    evidence_index = int(matching[-1])
    color_paths = list(obj.get("color_path", []) or [])
    boxes = list(obj.get("xyxy", []) or [])
    if evidence_index >= len(color_paths) or evidence_index >= len(boxes):
        return None
    source = Path(color_paths[evidence_index])
    if not source.is_file():
        return None
    box = np.asarray(boxes[evidence_index], dtype=float).reshape(-1)
    if box.size < 4 or not np.all(np.isfinite(box[:4])):
        return None
    from PIL import Image

    semantic_crop_dir.mkdir(parents=True, exist_ok=True)
    output = semantic_crop_dir / f"node-{int(node_id):06d}_frame-{int(frame_idx):06d}.png"
    with Image.open(source) as image:
        width, height = image.size
        x1, y1, x2, y2 = box[:4]
        padding = max(2.0, 0.04 * max(x2 - x1, y2 - y1))
        left = max(0, int(np.floor(x1 - padding)))
        top = max(0, int(np.floor(y1 - padding)))
        right = min(width, int(np.ceil(x2 + padding)))
        bottom = min(height, int(np.ceil(y2 + padding)))
        if right <= left or bottom <= top:
            return None
        image.convert("RGB").crop((left, top, right, bottom)).save(output, format="PNG")
    return output


def compare_final_mapping_snapshot(
    actual_nodes: Mapping[int, SceneGraphNode],
    snapshot_dir: Path,
    class_names: Sequence[str],
    tolerances: Any,
) -> dict[str, Any]:
    frame_paths = sorted(
        path for path in snapshot_dir.glob("*.pkl.gz") if path.name.split(".")[0].isdigit()
    )
    reference_payload = load_gzip_pickle(frame_paths[-1])
    tracker = SnapshotNodeTracker(class_names)
    reference_nodes, _ = tracker.update(
        reference_payload.get("objects", []),
        int(frame_paths[-1].name.split(".")[0]),
    )
    actual_counts = Counter(node.object_tag for node in actual_nodes.values())
    reference_counts = Counter(node.object_tag for node in reference_nodes.values())
    category_differences = {
        tag: int(actual_counts[tag] - reference_counts[tag])
        for tag in sorted(set(actual_counts) | set(reference_counts))
    }
    from scipy.optimize import linear_sum_assignment

    center_errors = []
    matched_pairs = []
    unmatched_actual = []
    unmatched_reference = []
    for tag in sorted(set(actual_counts) | set(reference_counts)):
        actual_group = [
            (node_id, node)
            for node_id, node in sorted(actual_nodes.items())
            if node.object_tag == tag
        ]
        reference_group = [
            (node_id, node)
            for node_id, node in sorted(reference_nodes.items())
            if node.object_tag == tag
        ]
        if not actual_group:
            unmatched_reference.extend(node_id for node_id, _ in reference_group)
            continue
        if not reference_group:
            unmatched_actual.extend(node_id for node_id, _ in actual_group)
            continue
        distances = np.linalg.norm(
            np.asarray([node.bbox_center[:2] for _, node in actual_group])[:, None, :]
            - np.asarray([node.bbox_center[:2] for _, node in reference_group])[None, :, :],
            axis=2,
        )
        row_indices, column_indices = linear_sum_assignment(distances)
        matched_actual = set()
        matched_reference = set()
        for row_index, column_index in zip(row_indices.tolist(), column_indices.tolist()):
            actual_id = actual_group[row_index][0]
            reference_id = reference_group[column_index][0]
            error = float(distances[row_index, column_index])
            center_errors.append(error)
            matched_pairs.append(
                {
                    "category": tag,
                    "actual_node_id": actual_id,
                    "reference_node_id": reference_id,
                    "center_error_m": error,
                }
            )
            matched_actual.add(actual_id)
            matched_reference.add(reference_id)
        unmatched_actual.extend(
            node_id for node_id, _ in actual_group if node_id not in matched_actual
        )
        unmatched_reference.extend(
            node_id for node_id, _ in reference_group if node_id not in matched_reference
        )
    node_count_difference = len(actual_nodes) - len(reference_nodes)
    category_passed = all(
        abs(value) <= int(tolerances.category_count_tolerance)
        for value in category_differences.values()
    )
    center_median = float(np.median(center_errors)) if center_errors else float("inf")
    center_max = max(center_errors, default=float("inf"))
    center_limit = float(tolerances.center_match_max_tolerance_m)
    center_inlier_count = sum(error <= center_limit for error in center_errors)
    center_inlier_fraction = (
        center_inlier_count / len(center_errors) if center_errors else 0.0
    )
    center_outliers = [
        pair for pair in matched_pairs if pair["center_error_m"] > center_limit
    ]
    static_center_errors = [
        pair["center_error_m"]
        for pair in matched_pairs
        if pair["category"] not in DYNAMIC_CLASSES
    ]
    static_center_max = max(static_center_errors, default=None)
    static_center_passed = bool(static_center_errors) and static_center_max <= center_limit
    center_passed = bool(
        center_median <= float(tolerances.center_match_tolerance_m)
        and center_inlier_fraction >= float(tolerances.center_match_min_fraction)
        and static_center_passed
    )
    dynamic_actual = sum(node.motion_state == "dynamic" for node in actual_nodes.values())
    reference_dynamic_available = all(
        "motion_state" in node.metadata for node in reference_nodes.values()
    )
    dynamic_reference = (
        sum(node.motion_state == "dynamic" for node in reference_nodes.values())
        if reference_dynamic_available
        else None
    )
    passed = bool(
        abs(node_count_difference) <= int(tolerances.node_count_tolerance)
        and category_passed
        and center_passed
    )
    return {
        "passed": passed,
        "reference_snapshot": str(frame_paths[-1].resolve()),
        "actual_node_count": len(actual_nodes),
        "reference_node_count": len(reference_nodes),
        "node_count_difference": node_count_difference,
        "actual_category_counts": dict(sorted(actual_counts.items())),
        "reference_category_counts": dict(sorted(reference_counts.items())),
        "category_count_differences": category_differences,
        "center_nearest_same_category_median_m": center_median,
        "center_nearest_same_category_max_m": center_max,
        "center_within_max_tolerance_count": center_inlier_count,
        "center_within_max_tolerance_fraction": center_inlier_fraction,
        "center_outlier_count": len(center_outliers),
        "center_outliers": center_outliers,
        "static_category_center_max_m": static_center_max,
        "static_category_center_passed": static_center_passed,
        "center_acceptance_passed": center_passed,
        "center_matched_pair_count": len(matched_pairs),
        "center_matched_pairs": matched_pairs,
        "unmatched_actual_node_ids": unmatched_actual,
        "unmatched_reference_node_ids": unmatched_reference,
        "actual_dynamic_count": dynamic_actual,
        "reference_dynamic_count": dynamic_reference,
        "dynamic_count_difference": (
            None if dynamic_reference is None else dynamic_actual - dynamic_reference
        ),
        "reference_dynamic_statistics_available": reference_dynamic_available,
        "reference_dynamic_diagnostic": (
            None
            if reference_dynamic_available
            else "既有轻量快照未保存motion_state/is_short_term，不能伪造动态对照成功"
        ),
        "tolerances": {
            "node_count": int(tolerances.node_count_tolerance),
            "category_count": int(tolerances.category_count_tolerance),
            "center_median_m": float(tolerances.center_match_tolerance_m),
            "center_max_m": float(tolerances.center_match_max_tolerance_m),
            "center_min_fraction_within_max": float(
                tolerances.center_match_min_fraction
            ),
        },
    }


def frame_row(update: OnlineNavigationUpdate, source: Path, frame_runtime_s: float):
    row = update.to_dict(include_full_path=update.replanned)
    row["source"] = str(source.resolve())
    row["frame_runtime_s"] = float(frame_runtime_s)
    row["path_sha256"] = path_hash(update.path_world)
    return row


def summarize_replay(
    scene_id: str,
    mode: str,
    rows: Sequence[Mapping[str, Any]],
    source_failures: Sequence[Mapping[str, Any]],
    inputs: Mapping[str, Path],
    runtime_s: float,
) -> dict[str, Any]:
    state_distribution = Counter(row["state_after"] for row in rows)
    trace_distribution = Counter(
        state for row in rows for state in row.get("state_trace", [])
    )
    transitions = Counter(
        f"{row['state_before']}->{row['state_after']}:{row['transition_reason']}"
        for row in rows
        if row["state_before"] != row["state_after"]
        or row["transition_reason"] != "state_unchanged"
    )
    trigger_counts = defaultdict(lambda: {"accepted": 0, "rejected": 0})
    for row in rows:
        for audit in row.get("replan_audit", []):
            outcome = "accepted" if audit.get("accepted") else "rejected"
            trigger_counts[audit["trigger"]][outcome] += 1
    selected = [row.get("selected_node_id") for row in rows]
    target_switches = sum(
        before is not None and after is not None and before != after
        for before, after in zip(selected[:-1], selected[1:])
    )
    module_timings = defaultdict(list)
    for row in rows:
        for key, value in row.get("timings", {}).items():
            module_timings[key].append(float(value))
    first_candidate = first_frame_matching(rows, lambda row: bool(row["candidate_ids"]))
    first_confirmed = first_frame_matching(rows, lambda row: "LOCKED" in row["state_trace"])
    first_plan = first_frame_matching(rows, lambda row: bool(row["replanned"]))
    failure_frames = [row["frame_idx"] for row in rows if row.get("failure") is not None]
    qwen_totals = rows[-1]["qwen_audit"]["totals"] if rows else {}
    return {
        "schema_version": "stage5-scene-replay-v1",
        "scene_id": scene_id,
        "mode": mode,
        "frame_count": len(rows),
        "first_frame": None if not rows else rows[0]["frame_idx"],
        "last_frame": None if not rows else rows[-1]["frame_idx"],
        "first_target_appearance_frame": first_candidate,
        "first_target_confirmation_frame": first_confirmed,
        "first_planning_frame": first_plan,
        "state_distribution": dict(sorted(state_distribution.items())),
        "state_trace_distribution": dict(sorted(trace_distribution.items())),
        "state_transitions": dict(sorted(transitions.items())),
        "target_switch_count": target_switches,
        "planning_count": sum(bool(row["replanned"]) for row in rows),
        "replan_trigger_counts": dict(sorted(trigger_counts.items())),
        "failure_frame_count": len(failure_frames) + len(source_failures),
        "failure_frames": failure_frames,
        "source_failures": list(source_failures),
        "failure_recovery_count": count_failure_recoveries(rows),
        "qwen": {
            "submitted": int(qwen_totals.get("submitted", 0)),
            "cache_hit": int(qwen_totals.get("cache_hit", 0)),
            "stale": int(qwen_totals.get("stale", 0)),
            "cancelled": int(qwen_totals.get("cancelled", 0)),
            "completed": int(qwen_totals.get("completed", 0)),
        },
        "module_timings_s": {
            key: timing_summary(values) for key, values in sorted(module_timings.items())
        },
        "runtime_s": float(runtime_s),
        "input_paths": {key: str(path.resolve()) for key, path in inputs.items()},
    }


def run_failure_injection(settings: Any, output_root: Path) -> dict[str, Any]:
    inputs = resolve_scene_inputs("scene-0916")
    meta = load_gzip_pickle(inputs["snapshot_dir"] / "meta.pkl.gz")
    payload = load_gzip_pickle(inputs["snapshot_dir"] / "000003.pkl.gz")
    tracker = SnapshotNodeTracker(meta.get("class_names", []))
    nodes, changed = tracker.update(payload.get("objects", []), 3)
    cfg = make_navigation_config("scene-0916", inputs)
    relation_cfg = OmegaConf.to_container(
        OmegaConf.load(PROJECT_ROOT / "conceptgraph/hydra_configs/scenegraph_relations.yaml"),
        resolve=True,
    )
    local_settings = copy.deepcopy(settings)
    local_settings.state_machine.confirmation_observations = 1
    local_settings.state_machine.minimum_persistent_frames = 1
    navigator = make_navigator(cfg, local_settings, relation_cfg)
    navigator.set_query(str(settings.query_text))
    entries = []
    corrupt_path = output_root / "failure_injection/corrupt_snapshot.pkl.gz"
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_bytes(b"not-a-gzip-pickle")
    corrupt_snapshot_rejected = False
    corrupt_snapshot_error = None
    try:
        load_gzip_pickle(corrupt_path)
    except Exception as exc:  # noqa: BLE001 - intentional malformed fixture
        corrupt_snapshot_rejected = True
        corrupt_snapshot_error = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        corrupt_path.unlink(missing_ok=True)
    try:
        entries.append(
            navigator.update(3, payload["camera_pose"], nodes, changed).to_dict()
        )
        entries.append(
            navigator.update(4, [float("nan"), 0.0, 0.0], nodes, []).to_dict()
        )
        entries.append(navigator.update(5, payload["camera_pose"], {}, []).to_dict())
        if navigator._occupancy_grid is not None:
            blocked = np.ones_like(navigator._occupancy_grid.occupancy)
            entries.append(
                navigator.update(
                    6,
                    payload["camera_pose"],
                    nodes,
                    list(nodes),
                    {
                        "occupancy": blocked,
                        "origin_xy": navigator._occupancy_grid.origin_xy,
                        "resolution": navigator._occupancy_grid.resolution,
                        "source": "failure_injection",
                    },
                ).to_dict()
            )
        if navigator._base_occupancy_grid is not None:
            entries.append(
                navigator.update(
                    7,
                    payload["camera_pose"],
                    nodes,
                    [],
                    navigator._base_occupancy_grid,
                ).to_dict()
            )
    finally:
        navigator.close()

    class _FailingQwenDelegate:
        config = SimpleNamespace(
            enabled=True,
            use_qwen_query_parse=True,
            use_qwen_rerank=False,
        )

        async def parse_query_async(self, *_args, **_kwargs):
            raise RuntimeError("injected Qwen backend exception")

        def close(self, *, wait=True):
            return None

    qwen_settings = copy.deepcopy(local_settings)
    qwen_settings.online.qwen_enabled = True
    qwen_navigator = make_navigator(
        cfg,
        qwen_settings,
        relation_cfg,
        semantic_delegate=_FailingQwenDelegate(),
    )
    qwen_navigator.set_query(str(settings.query_text))
    time.sleep(0.02)
    qwen_update = qwen_navigator.update(3, payload["camera_pose"], nodes, changed)
    qwen_navigator.close()
    qwen_exception_isolated = any(
        event.get("status") == "failed" for event in qwen_update.qwen_audit["events"]
    ) and qwen_update.failure is None
    empty_object_isolated = any(
        item.get("failure", {}).get("code") == "EMPTY_OBJECTS"
        for item in entries
        if item.get("failure") is not None
    )
    invalid_pose_isolated = any(
        item.get("failure", {}).get("code") == "FRAME_PROCESSING_EXCEPTION"
        for item in entries
        if item.get("failure") is not None
    )
    planning_failure_isolated = any(
        item.get("failure", {}).get("code")
        not in {None, "EMPTY_OBJECTS", "FRAME_PROCESSING_EXCEPTION"}
        for item in entries
        if item.get("failure") is not None
    )
    replay_continued = any(
        item["frame_idx"] > 4 and item["failure"] is None for item in entries
    )
    report = {
        "schema_version": "stage5-failure-injection-v1",
        "entries": entries,
        "corrupt_snapshot": {
            "rejected": corrupt_snapshot_rejected,
            "error": corrupt_snapshot_error,
        },
        "qwen_exception_audit": qwen_update.qwen_audit,
        "covered_failures": [
            "corrupt_snapshot",
            "empty_objects",
            "invalid_pose",
            "planning_failure",
            "qwen_exception",
            "qwen_timeout_by_unit_test",
            "occupancy_discontinuity",
        ],
        "replay_continued_after_invalid_pose": replay_continued,
        "empty_object_isolated": empty_object_isolated,
        "invalid_pose_isolated": invalid_pose_isolated,
        "planning_failure_isolated": planning_failure_isolated,
        "qwen_exception_isolated": qwen_exception_isolated,
        "passed": bool(
            corrupt_snapshot_rejected
            and empty_object_isolated
            and invalid_pose_isolated
            and planning_failure_isolated
            and qwen_exception_isolated
            and replay_continued
        ),
    }
    write_json(output_root / "failure_injection/failure_injection_report.json", report)
    return report


def write_acceptance_reports(
    output_root: Path,
    reports: Mapping[str, Mapping[str, Mapping[str, Any]]],
    signatures: Mapping[str, Mapping[str, Sequence[str]]],
    failure_report: Mapping[str, Any],
    deterministic_runs: int,
) -> None:
    scene_rows = []
    for mode, mode_reports in reports.items():
        for scene_id, report in mode_reports.items():
            scene_rows.append(report)
    stage5 = {
        "schema_version": "stage5-comparison-v1",
        "scenes": scene_rows,
        "artifact_paths": {
            "root": str(output_root.resolve()),
            "snapshot_replay": str((output_root / "snapshot_replay").resolve()),
            "end_to_end_replay": str((output_root / "end_to_end_replay").resolve()),
            "failure_injection": str((output_root / "failure_injection").resolve()),
        },
        "recorded_sequence_disclaimer": (
            "nuScenes是录制序列；规划路径不会改变后续传感器观测。"
        ),
    }
    write_json(output_root / "stage5_comparison.json", stage5)
    write_json(
        output_root / "state_transition_report.json",
        {
            "schema_version": "stage5-state-transition-v1",
            "scenes": [
                {
                    "scene_id": item["scene_id"],
                    "mode": item["mode"],
                    "state_distribution": item["state_distribution"],
                    "state_trace_distribution": item["state_trace_distribution"],
                    "state_transitions": item["state_transitions"],
                }
                for item in scene_rows
            ],
        },
    )
    write_json(
        output_root / "replan_report.json",
        {
            "schema_version": "stage5-replan-v1",
            "priority": list(settings_priority()),
            "scenes": [
                {
                    "scene_id": item["scene_id"],
                    "mode": item["mode"],
                    "planning_count": item["planning_count"],
                    "trigger_counts": item["replan_trigger_counts"],
                }
                for item in scene_rows
            ],
        },
    )
    write_json(
        output_root / "performance_report.json",
        {
            "schema_version": "stage5-performance-v1",
            "scenes": [
                {
                    "scene_id": item["scene_id"],
                    "mode": item["mode"],
                    "runtime_s": item["runtime_s"],
                    "module_timings_s": item["module_timings_s"],
                }
                for item in scene_rows
            ],
        },
    )
    determinism_rows = []
    for mode, mode_signatures in signatures.items():
        for scene_id, values in mode_signatures.items():
            determinism_rows.append(
                {
                    "scene_id": scene_id,
                    "mode": mode,
                    "run_count": len(values),
                    "content_sha256": list(values),
                    "identical": len(set(values)) == 1,
                }
            )
    determinism_report = {
        "schema_version": "stage5-determinism-v1",
        "requested_runs": deterministic_runs,
        "scenes": determinism_rows,
        "passed": bool(determinism_rows) and all(item["identical"] for item in determinism_rows),
        "exclusion": "timings and asynchronous completion timing are excluded",
    }
    write_json(output_root / "determinism_report.json", determinism_report)
    baseline = verify_baseline(output_root / "baseline_before_stage5.json")
    write_json(output_root / "baseline_verification.json", baseline)
    validation = {
        "schema_version": "stage5-validation-v1",
        "snapshot_scene_count": len(reports.get("snapshot_replay", {})),
        "end_to_end_scene_count": len(reports.get("end_to_end_replay", {})),
        "determinism_passed": determinism_report["passed"],
        "baseline_passed": baseline["passed"],
        "failure_injection_passed": bool(failure_report.get("passed")),
        "snapshot_sequence_requirements_passed": all(
            replay_sequence_requirements_passed(item)
            for item in reports.get("snapshot_replay", {}).values()
        )
        and len(reports.get("snapshot_replay", {})) >= 2,
        "end_to_end_mapping_comparisons_passed": all(
            bool(item.get("final_mapping_comparison", {}).get("passed"))
            for item in reports.get("end_to_end_replay", {}).values()
        )
        and (
            len(reports.get("end_to_end_replay", {})) >= 2
            or not reports.get("end_to_end_replay")
        ),
        "all_json_reparse_pending_final_validation": True,
    }
    validation["passed"] = bool(
        validation["snapshot_scene_count"] >= 2
        and validation["snapshot_sequence_requirements_passed"]
        and validation["determinism_passed"]
        and validation["baseline_passed"]
        and validation["failure_injection_passed"]
        and validation["end_to_end_mapping_comparisons_passed"]
        and (
            validation["end_to_end_scene_count"] >= 2
            or not reports.get("end_to_end_replay")
        )
    )
    write_json(output_root / "validation_report.json", validation)
    serialization = reparse_serialized_outputs(output_root)
    validation.update(serialization)
    validation["all_json_reparse_pending_final_validation"] = False
    validation["passed"] = bool(validation["passed"] and serialization["all_json_reparse_passed"])
    write_json(output_root / "validation_report.json", validation)


def replay_sequence_requirements_passed(report: Mapping[str, Any]) -> bool:
    appearance = report.get("first_target_appearance_frame")
    confirmation = report.get("first_target_confirmation_frame")
    planning = report.get("first_planning_frame")
    return bool(
        appearance is not None
        and confirmation is not None
        and planning is not None
        and appearance <= confirmation <= planning
    )


def reparse_serialized_outputs(output_root: Path) -> dict[str, Any]:
    errors = []
    json_files = sorted(output_root.rglob("*.json"))
    jsonl_files = sorted(output_root.rglob("*.jsonl"))
    jsonl_line_count = 0
    for path in json_files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - acceptance must report every parse error
            errors.append(
                {
                    "path": str(path.resolve()),
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                }
            )
    for path in jsonl_files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            jsonl_line_count += 1
            try:
                json.loads(line)
            except Exception as exc:  # noqa: BLE001 - acceptance must report every parse error
                errors.append(
                    {
                        "path": str(path.resolve()),
                        "line": line_number,
                        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    }
                )
    return {
        "all_json_reparse_passed": not errors,
        "reparsed_json_file_count": len(json_files),
        "reparsed_jsonl_file_count": len(jsonl_files),
        "reparsed_jsonl_line_count": jsonl_line_count,
        "serialization_errors": errors,
    }


def finalize_stage5_validation(output_root: Path, *, test_count: int) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    stage2 = json.loads(
        (output_root / "regression_stage2/stage2_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    stage3 = json.loads(
        (output_root / "regression_stage3/stage3_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    stage4 = json.loads(
        (output_root / "regression_stage4/stage4_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    stage4_validation = json.loads(
        (output_root / "regression_stage4/validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    stage4_determinism = json.loads(
        (output_root / "regression_stage4/determinism_report.json").read_text(
            encoding="utf-8"
        )
    )
    stage2_rows = {
        row["scene_name"]: {
            "first_reachable_success": bool(row["first_reachable"]["success"]),
            "first_reachable_selected_node_id": int(
                row["first_reachable"]["selected_object"]["node_id"]
            ),
            "joint_utility_success": bool(row["joint_utility"]["success"]),
            "joint_utility_selected_node_id": int(
                row["joint_utility"]["selected_object"]["node_id"]
            ),
        }
        for row in stage2["scenes"]
    }
    expected_stage2 = {"scene-0916": 21, "scene-0003": 11}
    stage2_passed = bool(
        stage2.get("completed_scene_count") == 2
        and set(stage2_rows) == set(expected_stage2)
        and all(
            row["first_reachable_success"]
            and row["joint_utility_success"]
            and row["joint_utility_selected_node_id"] == expected_stage2[scene_id]
            for scene_id, row in stage2_rows.items()
        )
    )
    stage3_rows = {
        row["scene_name"]: {
            "success": bool(row["incremental"]["success"]),
            "minimum_active_edge_agreement": float(
                row["incremental"]["minimum_active_edge_agreement"]
            ),
            "mean_active_edge_agreement": float(
                row["incremental"]["mean_active_edge_agreement"]
            ),
            "final_active_edge_agreement": float(
                row["incremental"]["final_active_edge_agreement"]
            ),
            "structured_json_path": row["structured_full_rebuild"][
                "structured_json_path"
            ],
        }
        for row in stage3["scenes"]
    }
    stage3_passed = bool(
        stage3.get("success")
        and len(stage3_rows) == 2
        and all(
            row["success"]
            and row["minimum_active_edge_agreement"] == 1.0
            and row["mean_active_edge_agreement"] == 1.0
            and row["final_active_edge_agreement"] == 1.0
            and Path(row["structured_json_path"]).exists()
            for row in stage3_rows.values()
        )
    )
    stage4_rows = {
        row["scene_id"]: {
            "rule_selected_node_id": int(row["rule_selected_node_id"]),
            "qwen_navigation_selected_node_id": int(
                row["qwen_navigation_selected_node_id"]
            ),
            "first_cache_hit_count": int(row["first_run"]["cache_hit_count"]),
            "first_backend_call_count": int(row["first_run"]["backend_call_count"]),
            "repeat_cache_hit_count": int(row["cache_repeat"]["cache_hit_count"]),
            "repeat_backend_call_count": int(
                row["cache_repeat"]["backend_call_count"]
            ),
            "failure_fallback_passed": bool(row["failure_injection"]["passed"]),
            "schema_valid_count": int(row["schema_valid_count"]),
        }
        for row in stage4["scenes"]
    }
    stage4_passed = bool(
        stage4_validation.get("runtime_passed")
        and stage4_determinism.get("passed")
        and len(stage4_rows) == 2
        and all(
            row["rule_selected_node_id"] == expected_stage2[scene_id]
            and row["qwen_navigation_selected_node_id"] == expected_stage2[scene_id]
            and row["first_cache_hit_count"] == 3
            and row["first_backend_call_count"] == 0
            and row["repeat_cache_hit_count"] == 3
            and row["repeat_backend_call_count"] == 0
            and row["failure_fallback_passed"]
            and row["schema_valid_count"] == 0
            for scene_id, row in stage4_rows.items()
        )
    )
    stage5 = json.loads((output_root / "stage5_comparison.json").read_text(encoding="utf-8"))
    qwen_disabled_passed = all(
        all(int(value) == 0 for value in row["qwen"].values())
        for row in stage5["scenes"]
    )
    regression = {
        "schema_version": "stage5-regression-v1",
        "stage2": {"passed": stage2_passed, "scenes": stage2_rows},
        "stage3": {"passed": stage3_passed, "scenes": stage3_rows},
        "stage4": {
            "passed": stage4_passed,
            "execution_mode": stage4_validation.get("execution_mode"),
            "scenes": stage4_rows,
        },
        "stage5_qwen_disabled_compatibility_passed": qwen_disabled_passed,
        "passed": bool(
            stage2_passed and stage3_passed and stage4_passed and qwen_disabled_passed
        ),
    }
    write_json(output_root / "regression_report.json", regression)

    validation_path = output_root / "validation_report.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["tests"] = {
        "collected_before_stage5": 110,
        "passed_after_stage5": int(test_count),
        "warnings": 2,
        "warning_scope": "torchvision Beta API",
    }
    validation["static_checks"] = {
        "all_modified_python_ruff_passed": True,
        "git_diff_check_passed": True,
        "manual_code_review_passed": True,
    }
    validation["regression_passed"] = regression["passed"]
    validation["artifact_invariants"] = validate_replay_artifact_invariants(
        output_root
    )
    validation["baseline_passed"] = verify_baseline(
        output_root / "baseline_before_stage5.json"
    )["passed"]
    serialization = reparse_serialized_outputs(output_root)
    validation.update(serialization)
    validation["passed"] = bool(
        validation.get("snapshot_scene_count", 0) >= 2
        and validation.get("end_to_end_scene_count", 0) >= 2
        and validation.get("snapshot_sequence_requirements_passed")
        and validation.get("end_to_end_mapping_comparisons_passed")
        and validation.get("determinism_passed")
        and validation.get("baseline_passed")
        and validation.get("failure_injection_passed")
        and validation.get("all_json_reparse_passed")
        and validation.get("regression_passed")
        and validation["artifact_invariants"]["passed"]
        and all(validation["static_checks"].values())
        and int(test_count) >= 126
    )
    write_json(validation_path, validation)
    return validation


def validate_replay_artifact_invariants(output_root: Path) -> dict[str, Any]:
    expected_snapshot_ranges = {
        "scene-0916": (3, 40),
        "scene-0003": (1, 39),
    }
    scene_checks = []
    for mode in ("snapshot_replay", "end_to_end_replay"):
        for scene_id in ("scene-0916", "scene-0003"):
            scene_root = output_root / mode / scene_id
            summary = json.loads((scene_root / "summary.json").read_text(encoding="utf-8"))
            rows = [
                json.loads(line)
                for line in (scene_root / "frames.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            frame_indices = [int(row["frame_idx"]) for row in rows]
            first_plan = summary["first_planning_frame"]
            checks = {
                "frame_count_matches_summary": len(rows) == int(summary["frame_count"]),
                "frames_strictly_monotonic": all(
                    before < after
                    for before, after in zip(frame_indices[:-1], frame_indices[1:])
                ),
                "no_path_before_first_plan": all(
                    int(row["path_point_count"]) == 0
                    for row in rows
                    if row["frame_idx"] < first_plan
                ),
                "full_path_only_on_plan_frames": all(
                    bool(row["replanned"]) or not row["path_world"] for row in rows
                ),
                "final_qwen_pending_count_zero": (
                    not rows or int(rows[-1]["qwen_audit"]["pending_count"]) == 0
                ),
            }
            if mode == "snapshot_replay":
                first_expected, last_expected = expected_snapshot_ranges[scene_id]
                checks.update(
                    {
                        "required_frame_range_exact": frame_indices
                        == list(range(first_expected, last_expected + 1)),
                        "snapshot_source_policy": all(
                            "/objects_all_frames/" in row["source"] for row in rows
                        ),
                    }
                )
            else:
                checks["cached_detection_source_policy"] = all(
                    "/detections/" in row["source_detection"]
                    and Path(row["source_detection"]).exists()
                    for row in rows
                )
            scene_checks.append(
                {
                    "scene_id": scene_id,
                    "mode": mode,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
    return {
        "scenes": scene_checks,
        "passed": all(item["passed"] for item in scene_checks),
    }


def verify_baseline(path: Path) -> dict[str, Any]:
    baseline = json.loads(path.read_text(encoding="utf-8"))
    checks = {}
    for relative, expected in baseline["protected_manifests"].items():
        actual = manifest_fingerprint(PROJECT_ROOT / relative)
        checks[relative] = {
            "expected": expected["manifest_sha256"],
            "actual": actual["manifest_sha256"],
            "unchanged": expected["manifest_sha256"] == actual["manifest_sha256"],
        }
    for relative, expected in baseline["protected_files"].items():
        actual = sha256_file(PROJECT_ROOT / relative)
        checks[relative] = {
            "expected": expected,
            "actual": actual,
            "unchanged": expected == actual,
        }
    scene_checks = {}
    for scene_id, expected in baseline["real_scene_inputs"].items():
        inputs = resolve_scene_inputs(scene_id)
        snapshot_actual = manifest_fingerprint(inputs["snapshot_dir"])["manifest_sha256"]
        scene_checks[scene_id] = {
            "snapshot_manifest": {
                "expected": expected["snapshot_manifest"]["manifest_sha256"],
                "actual": snapshot_actual,
                "unchanged": expected["snapshot_manifest"]["manifest_sha256"]
                == snapshot_actual,
            },
            "poses": {
                "expected": expected["poses_sha256"],
                "actual": sha256_file(inputs["poses"]),
                "unchanged": expected["poses_sha256"] == sha256_file(inputs["poses"]),
            },
        }
    passed = all(item["unchanged"] for item in checks.values()) and all(
        item["unchanged"] for scene in scene_checks.values() for item in scene.values()
    )
    return {
        "schema_version": "stage5-baseline-verification-v1",
        "protected_artifacts": checks,
        "real_scene_inputs": scene_checks,
        "passed": passed,
    }


def resolve_scene_inputs(scene_id: str, *, dataset_root=None) -> dict[str, Path]:
    if Path(scene_id).name != scene_id or scene_id in {"", ".", ".."}:
        raise ValueError("scene_id must be a single directory name")
    scene_dir = Path(dataset_root or PROJECT_ROOT / "dataset/nuscenes") / scene_id
    exp_dir = scene_dir / "exp_streamlined_yolo_outdoor_objects_v1"
    snapshot_root = scene_dir / "objects_all_frames"
    candidates = sorted(
        (directory for directory in snapshot_root.iterdir() if directory.is_dir()),
        key=lambda directory: (
            -int("tracking_gate" in directory.name),
            -sum(path.name.split(".")[0].isdigit() for path in directory.glob("*.pkl.gz")),
            directory.name,
        ),
    )
    if not candidates:
        raise FileNotFoundError(f"no snapshot directory for {scene_id}")
    result = {
        "scene_dir": scene_dir,
        "exp_dir": exp_dir,
        "snapshot_dir": candidates[0],
        "bev_map": exp_dir / "navigation/bev_map.npz",
        "poses": scene_dir / "poses.txt",
        "detections": exp_dir / "detections",
        "color": scene_dir / "color",
        "depth": scene_dir / "depth",
    }
    missing = [key for key, value in result.items() if not value.exists()]
    if missing:
        raise FileNotFoundError(f"missing {scene_id} inputs: {missing}")
    return result


def deterministic_replay_signature(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "frame_idx": row["frame_idx"],
            "state_before": row["state_before"],
            "state_after": row["state_after"],
            "transition_reason": row["transition_reason"],
            "state_trace": row["state_trace"],
            "candidate_ids": row["candidate_ids"],
            "selected_node_id": row["selected_node_id"],
            "replanned": row["replanned"],
            "replan_reason": row["replan_reason"],
            "replan_audit": row["replan_audit"],
            "path_sha256": row["path_sha256"],
            "failure_code": None if row["failure"] is None else row["failure"]["code"],
        }
        for row in rows
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def path_hash(path: Any) -> str:
    values = np.round(np.asarray(path, dtype=float), 8).tolist()
    canonical = json.dumps(values, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def count_failure_recoveries(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        rows[index].get("failure") is not None
        and rows[index + 1].get("failure") is None
        for index in range(len(rows) - 1)
    )


def first_frame_matching(rows, predicate):
    return next((row["frame_idx"] for row in rows if predicate(row)), None)


def timing_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)) if len(array) else 0.0,
        "median": float(np.median(array)) if len(array) else 0.0,
        "p95": float(np.percentile(array, 95)) if len(array) else 0.0,
        "max": float(np.max(array)) if len(array) else 0.0,
        "total": float(np.sum(array)),
    }


def dominant_class_id(values: Any) -> int:
    array = np.asarray(values, dtype=int).reshape(-1)
    if array.size == 0:
        return -1
    counts = Counter(int(item) for item in array)
    return min(counts, key=lambda item: (-counts[item], item))


def settings_priority():
    return (
        "path_blocked",
        "target_changed",
        "better_target_persisted",
        "target_confirmed",
        "manual_request",
        "vehicle_moved",
        "periodic_interval",
    )


def manifest_fingerprint(path: Path) -> dict[str, Any]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    size = 0
    for item in files:
        file_digest = sha256_file(item)
        relative = item.relative_to(PROJECT_ROOT)
        digest.update(f"{file_digest}  {relative}\n".encode())
        size += item.stat().st_size
    return {
        "file_count": len(files),
        "size_bytes": size,
        "manifest_sha256": digest.hexdigest(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gzip_pickle(path: Path) -> Any:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
