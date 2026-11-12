"""Build deterministic nuScenes parking-scene manifests under outputs/project_v2."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from conceptgraph.nuscenes.scene_selector import (
    MetadataValidationError,
    NuScenesSceneSelector,
    SelectorConfig,
    build_scene_scores_payload,
    build_selected_scenes_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_V2_ROOT = PROJECT_ROOT / "outputs" / "project_v2"
DEFAULT_OUT_DIR = PROJECT_V2_ROOT / "subset"
OUTPUT_LAYOUT = (
    "tests",
    "subset",
    "offline_navigation",
    "incremental_navigation",
    "qwen_eval",
    "performance",
)
OUTPUT_FILENAMES = (
    "scene_scores.json",
    "selected_scenes.json",
    "integrity_report.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score nuScenes scenes from structured metadata and build a local-data-safe subset manifest."
        )
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "nuscenes_trainval_raw",
        help="nuScenes dataset root or version directory",
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
        help="roots searched for sample files and converted scene directories",
    )
    parser.add_argument("--num-scenes", type=int, default=12)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output directory, which must be under outputs/project_v2",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="score real metadata and print the selection without writing files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing stage-1 manifest files atomically",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_scenes <= 0:
        parser.error("--num-scenes must be greater than zero")

    try:
        out_dir = validate_output_dir(args.out_dir)
        output_paths = [out_dir / filename for filename in OUTPUT_FILENAMES]
        existing = [path for path in output_paths if path.exists()]
        if existing and not args.overwrite and not args.dry_run:
            names = ", ".join(str(path) for path in existing)
            parser.error(
                "output files already exist; pass --overwrite to replace them: "
                f"{names}"
            )
        data_roots = tuple(path.expanduser().resolve() for path in args.data_roots)
        result = NuScenesSceneSelector(
            SelectorConfig(
                metadata_root=args.metadata_root,
                version=args.version,
                data_roots=data_roots,
            )
        ).score_scenes()
    except (MetadataValidationError, ValueError) as exc:
        parser.error(str(exc))

    selected_payload = build_selected_scenes_payload(result, args.num_scenes)
    _print_summary(
        result.scene_scores, selected_payload["scenes"], result.integrity_report
    )
    if args.dry_run:
        print("Dry run complete; no output files were written.")
        return 0

    ensure_output_layout()
    payloads = (
        build_scene_scores_payload(result),
        selected_payload,
        result.integrity_report,
    )
    for path, payload in zip(output_paths, payloads):
        write_json_atomic(path, payload)
        print(f"Wrote {path}")
    return 0


def validate_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    allowed = PROJECT_V2_ROOT.resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(
            f"--out-dir must be {allowed} or one of its descendants; got {resolved}"
        ) from exc
    return resolved


def ensure_output_layout() -> None:
    for relative in OUTPUT_LAYOUT:
        (PROJECT_V2_ROOT / relative).mkdir(parents=True, exist_ok=True)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _print_summary(
    scene_scores: Sequence[dict[str, Any]],
    selected: Sequence[dict[str, Any]],
    integrity_report: dict[str, Any],
) -> None:
    ready_count = integrity_report["ready_for_local_conversion_scene_count"]
    print(
        f"Scored {len(scene_scores)} scenes; {ready_count} have complete local source data."
    )
    print(f"Selected {len(selected)} scene(s):")
    for row in selected:
        print(
            f"  {row['selection_rank']:02d} {row['scene_name']} "
            f"score={row['total_score']:.3f} "
            f"frames={row['cam_front_lidar_top_complete_frames']}/{row['nbr_samples']} "
            f"parked={row['vehicle_parked_count']} "
            f"dynamic_ratio={row['dynamic_target_ratio']:.3f}"
        )
    coverage = integrity_report["sensor_file_coverage"]
    print(
        "Integrity: "
        f"metadata={integrity_report['metadata_status']}, "
        f"invalid_refs={integrity_report['invalid_reference_count']}, "
        f"missing_or_invalid_sensor_files={coverage['missing_or_invalid_count']}, "
        f"converted_frame_mismatches={integrity_report['converted_frame_mismatch_count']}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
