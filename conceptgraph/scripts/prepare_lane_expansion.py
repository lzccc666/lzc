"""Build a fresh multi-scene inventory and log-isolated selection manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from conceptgraph.nuscenes.expansion_benchmark import (
    inventory,
    select_scenes,
    summarize_inventory,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "outputs/project_v2/lane_qwen_expansion_v1"


def main():
    parser = argparse.ArgumentParser(description="盘点真实输入并生成多场景独立划分")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--archive-members", type=Path)
    parser.add_argument("--target", type=int, default=50)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if not root.is_relative_to(DEFAULT_ROOT):
        raise ValueError("output must be within the new expansion root")
    destinations = [
        root / name
        for name in (
            "data_inventory.json",
            "scene_selection_manifest.json",
            "source_file_index.json",
            "split_manifest.json",
            "leakage_audit.json",
            "missing_data_manifest.json",
        )
    ]
    if any(p.exists() for p in destinations):
        raise FileExistsError(
            "preparation already exists; use a new output subdirectory"
        )
    members = (
        set(args.archive_members.read_text().splitlines())
        if args.archive_members
        else set()
    )
    rows, sources = inventory(
        PROJECT_ROOT / "dataset/nuscenes_trainval_raw/v1.0-trainval",
        [
            PROJECT_ROOT / "dataset/nuscenes_trainval_raw",
            PROJECT_ROOT / "dataset/v1.0-mini",
            root / "raw",
        ],
        PROJECT_ROOT / "dataset/nuscenes",
        members,
    )
    legacy_path = (
        PROJECT_ROOT / "outputs/project_v2/qwen_eval/dataset/split_manifest.json"
    )
    legacy = json.loads(legacy_path.read_text())["scene_to_split"]
    selection = select_scenes(rows, legacy, args.target)
    selected_names = {r["scene"] for r in selection["scenes"]}
    selected_sources = {name: sources[name] for name in sorted(selected_names)}
    missing = []
    for name, source in selected_sources.items():
        for filename, path in source["source_files"].items():
            if path is None:
                missing.append(
                    {
                        "scene": name,
                        "filename": filename,
                        "in_available_archive": filename in members,
                    }
                )
    payloads = [
        {
            "summary": summarize_inventory(rows),
            "scenes": rows,
            "archive_member_source": str(args.archive_members)
            if args.archive_members
            else None,
        },
        selection,
        selected_sources,
        {
            "scene_to_split": selection["scene_to_split"],
            "selection_sha256": selection["sha256"],
            "original_scene_to_split": legacy,
            "test_labels_opened": False,
        },
        selection["audit"],
        {
            "missing_raw_files": missing,
            "missing_raw_file_count": len(missing),
            "minimum_scene_count": 30,
            "selected_count": len(selected_names),
            "source_eligibility_is_not_evaluation_completion": True,
        },
    ]
    root.mkdir(parents=True, exist_ok=True)
    for path, value in zip(destinations, payloads):
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(
        json.dumps(
            {"inventory": summarize_inventory(rows), "selection": selection["audit"]},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
