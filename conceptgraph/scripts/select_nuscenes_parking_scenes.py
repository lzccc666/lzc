"""Backward-compatible entry point for structured nuScenes scene selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from conceptgraph.nuscenes.scene_selector import NuScenesSceneSelector, SelectorConfig
from conceptgraph.scripts.build_nuscenes_subset_manifest import (
    PROJECT_ROOT,
    PROJECT_V2_ROOT,
    validate_output_dir,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select parking-like nuScenes scenes using structured metadata."
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "nuscenes_trainval_raw",
    )
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument(
        "--data-roots",
        nargs="+",
        type=Path,
        default=[
            PROJECT_ROOT / "dataset" / "nuscenes_trainval_raw",
            PROJECT_ROOT / "dataset" / "nuscenes",
        ],
    )
    parser.add_argument("--num-scenes", type=int, default=12)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_V2_ROOT / "subset" / "selected_parking_scenes_legacy.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_scenes <= 0:
        raise ValueError("--num-scenes must be greater than zero")
    output_path = validate_output_dir(args.out.parent) / args.out.name
    result = NuScenesSceneSelector(
        SelectorConfig(
            metadata_root=args.metadata_root,
            version=args.version,
            data_roots=tuple(path.expanduser().resolve() for path in args.data_roots),
        )
    ).score_scenes()
    selected = result.select(args.num_scenes)
    for row in selected:
        print(
            row["scene_name"],
            f"score={row['total_score']:.3f}",
            f"frames={row['cam_front_lidar_top_complete_frames']}/{row['nbr_samples']}",
            row["selection_reason"],
        )
    if args.dry_run:
        print("Dry run complete; no output file was written.")
        return
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
    write_json_atomic(output_path, selected)
    print(f"Saved {len(selected)} selected scenes to {output_path}")


def load_json(path: Path) -> list[dict]:
    """Retained for callers that imported the old script helper."""
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return value


if __name__ == "__main__":
    main()
