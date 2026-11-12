"""Materialize real first-frame nodes without relying on omitted init snapshots."""

from __future__ import annotations

import gzip
import json
import pickle

import numpy as np
import torch

from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.nuscenes.expansion_benchmark import content_sha256
from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT
from conceptgraph.scripts.run_incremental_semantic_navigation import (
    make_mapping_config,
    mapping_objects_to_nodes,
    write_json,
)
from conceptgraph.slam.streamlined_mapping import (
    MappingFrame,
    StreamlinedMapper,
    process_cfg,
)
from conceptgraph.slam.utils import create_or_load_colors


def load_frame0(root, scene):
    path = root / "frame0_inputs" / scene / "nodes.pkl.gz"
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def prepare_scene(root, scene):
    output = root / "frame0_inputs" / scene
    if (output / "complete.json").exists():
        return
    np.random.seed(0)
    torch.manual_seed(0)
    cfg = process_cfg(
        make_mapping_config(scene, dataset_root=root / "converted", device="cpu")
    )
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=1,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=scene,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )
    classes, colors = create_or_load_colors(cfg, cfg.color_file_name)
    mapper = StreamlinedMapper(cfg, classes, colors)
    color_path = dataset.color_paths[0]
    source = (
        root
        / "converted"
        / scene
        / cfg.detection_folder_name
        / __import__("pathlib").Path(color_path).name
    ).with_suffix(".pkl.gz")
    with gzip.open(source, "rb") as handle:
        detections = pickle.load(handle)
    color, depth, intrinsic, *_ = dataset[0]
    mapper.update(
        MappingFrame(
            frame_idx=0,
            image_rgb=color.cpu().numpy().astype(np.uint8),
            depth_array=depth[..., 0].cpu().numpy(),
            cam_K=intrinsic.cpu().numpy()[:3, :3],
            camera_pose=dataset.poses[0].cpu().numpy(),
            detections=detections,
            color_path=str(color_path),
        )
    )
    nodes = mapping_objects_to_nodes(
        mapper.effective_objects(),
        classes,
        0,
        semantic_crop_dir=root / "frame0_crops" / scene,
    )
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(output / "nodes.pkl.gz", "wb") as handle:
        pickle.dump(nodes, handle)
    rows = [
        {
            "node_id": int(n.node_id),
            "category": n.object_tag,
            "center": n.bbox_center.tolist(),
            "extent": n.bbox_extent.tolist(),
            "metadata": n.metadata,
        }
        for n in nodes.values()
    ]
    write_json(output / "nodes.json", rows)
    write_json(
        output / "complete.json",
        {
            "scene": scene,
            "frame_idx": 0,
            "node_count": len(nodes),
            "content_sha256": content_sha256(rows),
            "source": str(source),
            "source_sha256": __import__("hashlib")
            .sha256(source.read_bytes())
            .hexdigest(),
            "initialization_snapshot_bug_avoided": True,
        },
    )
    print(scene, len(nodes), flush=True)


def main():
    root = DEFAULT_ROOT
    for row in json.loads((root / "scene_selection_manifest.json").read_text())[
        "scenes"
    ]:
        if not (
            root / "preprocessing/detections" / row["scene"] / "complete.json"
        ).exists():
            continue
        prepare_scene(root, row["scene"])


if __name__ == "__main__":
    main()
