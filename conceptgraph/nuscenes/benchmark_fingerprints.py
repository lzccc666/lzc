"""Content fingerprints for benchmark inputs, implementation and resume checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

from conceptgraph.nuscenes.expansion_benchmark import content_sha256


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(paths, base):
    return [
        {
            "path": str(p.relative_to(base)),
            "bytes": p.stat().st_size,
            "sha256": file_sha256(p),
        }
        for p in sorted(set(paths))
        if p.is_file()
    ]


def implementation_manifest(project):
    paths = []
    for name in ("navigation", "scenegraph", "slam", "vlm", "dataset"):
        paths.extend((project / "conceptgraph" / name).rglob("*.py"))
    paths.extend((project / "conceptgraph/hydra_configs").rglob("*.yaml"))
    paths += [
        project / "conceptgraph/scripts" / name
        for name in (
            "run_incremental_semantic_navigation.py",
            "run_lane_expansion_benchmark.py",
            "run_lane_expansion_static.py",
            "run_lane_expansion_retrieval.py",
            "prepare_lane_expansion_frame0.py",
        )
    ]
    paths += list((project / "conceptgraph/nuscenes").glob("benchmark_*.py"))
    rows = file_manifest(paths, project)
    return {"files": rows, "sha256": content_sha256(rows)}


def scene_input_manifest(root, scene):
    directory = root / "converted" / scene
    paths = [
        p
        for name in (
            "color",
            "depth",
            "exp_streamlined_yolo_outdoor_objects_v1/detections",
        )
        for p in (directory / name).glob("*")
        if p.is_file()
    ]
    paths += [
        directory / name
        for name in (
            "poses.txt",
            "frame_alignment.json",
            "intrinsics.txt",
            "conversion_complete.json",
        )
    ]
    paths += list((root / "lane_maps" / scene).glob("*.npz")) + list(
        (root / "lane_maps" / scene).glob("*.json")
    )
    rows = file_manifest(paths, root)
    return {"files": rows, "sha256": content_sha256(rows)}
