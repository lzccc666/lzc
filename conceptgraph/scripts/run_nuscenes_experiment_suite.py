from __future__ import annotations

import argparse
import csv
import json
import subprocess
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from conceptgraph.navigation.evaluation import save_scene_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or resume a multi-scene nuScenes ConceptGraphs experiment suite.")
    parser.add_argument("--raw-nuscenes-root", type=Path, default=Path("/root/autodl-pub/nuScenes"))
    parser.add_argument("--minimal-root", type=Path, default=Path("/root/autodl-tmp/lizichong/dataset/nuscenes"))
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--scene-list", nargs="*", default=None)
    parser.add_argument("--num-scenes", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--exp-suffix", default="qwen2vl_semantics_v1")
    parser.add_argument("--baseline-suffix", default="streamlined_yolo_outdoor_objects_v1")
    parser.add_argument("--map-root", type=Path, default=Path("/root/autodl-tmp/lizichong/dataset/nuScenes-map-expansion-v1.3"))
    parser.add_argument("--summary-dir", type=Path, default=Path("/root/autodl-tmp/lizichong/outputs/experiment_summary"))
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--run-preprocess", action="store_true")
    parser.add_argument("--run-navigation", action="store_true", default=True)
    parser.add_argument("--run-evaluation-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    selected = select_scenes(args)
    failures: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    nav_rows: list[dict[str, Any]] = []

    for scene in selected:
        scene_name = scene["scene_name"]
        scene_dir = args.minimal_root / scene_name
        qwen_exp = scene_dir / f"exp_{args.exp_suffix}"
        baseline_exp = scene_dir / f"exp_{args.baseline_suffix}"
        row = dict(scene)
        try:
            if args.run_preprocess and not args.run_evaluation_only:
                run_stage(
                    "preprocess",
                    [
                        "python",
                        "conceptgraph/scripts/preprocess_nuscenes_to_minimal.py",
                        "--nuscenes-root",
                        str(args.raw_nuscenes_root),
                        "--version",
                        args.version,
                        "--scene-name",
                        scene_name,
                        "--output-root",
                        str(args.minimal_root),
                        "--stride",
                        str(args.stride),
                        "--max-frames",
                        str(args.max_frames),
                    ],
                    failures,
                    scene_name,
                )
            if not args.run_evaluation_only:
                ensure_scene_artifacts(scene_dir, qwen_exp, baseline_exp, args, failures)
            if not qwen_exp.exists():
                raise FileNotFoundError(f"Missing experiment directory: {qwen_exp}")
            eval_dir = save_scene_evaluation(scene_dir=scene_dir, exp_dir=qwen_exp, navigation_dir=qwen_exp / "navigation_examples")
            row.update(load_json(eval_dir / "scene_metrics.json", {}))
            nav_cases = load_json(eval_dir / "navigation_cases.json", [])
            for case in nav_cases:
                flat = {"scene_id": scene_name}
                flat.update(case.get("target_retrieval", {}))
                flat.update(case.get("reachability", {}))
                flat.update(case.get("path_planning", {}))
                nav_rows.append(flat)
            row["completed"] = True
            row["failed_stage"] = None
        except Exception as exc:
            row["completed"] = False
            row["failed_stage"] = "suite"
            row["error"] = str(exc)
            failures.append(
                {
                    "scene_id": scene_name,
                    "failed_stage": "suite",
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=3),
                }
            )
        scene_rows.append(row)

    write_json(args.summary_dir / "selected_scenes.json", selected)
    write_json(args.summary_dir / "failures.json", failures)
    write_csv(args.summary_dir / "all_scene_summary.csv", scene_rows)
    write_csv(args.summary_dir / "all_navigation_cases.csv", nav_rows)
    write_report(args.summary_dir / "experiment_report.md", scene_rows, nav_rows, failures)
    print(f"Saved experiment summary to {args.summary_dir}")


def select_scenes(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.scene_list:
        names = args.scene_list
    else:
        names = select_parking_like_scene_names(args)
    selected = []
    for name in names[: args.num_scenes]:
        scene_dir = args.minimal_root / name
        frame_count = count_frames(scene_dir)
        selected.append(
            {
                "scene_name": name,
                "frame_count": int(frame_count),
                "converted": bool(frame_count > 0),
                "selection_reason": "existing minimal scene with poses/color/depth" if frame_count else "selected from raw nuScenes table",
                "initial_target_count": None,
            }
        )
    return selected


def select_parking_like_scene_names(args: argparse.Namespace) -> list[str]:
    existing = sorted(path.name for path in args.minimal_root.glob("scene-*") if (path / "poses.txt").exists())
    scene_meta_path = _find_scene_meta_path(args)
    if not scene_meta_path:
        return existing[: args.num_scenes]
    scenes = load_json(scene_meta_path, [])
    by_name = {item["name"]: item for item in scenes if "name" in item}
    parking_keywords = ("parking", "parked", "parking lot", "park ")
    dynamic_keywords = ("high speed", "busy intersection", "overtaking", "jaywalker")

    def score(name: str) -> tuple[int, int, str]:
        desc = str(by_name.get(name, {}).get("description", "")).lower()
        parking_score = sum(desc.count(keyword) for keyword in parking_keywords)
        dynamic_penalty = sum(desc.count(keyword) for keyword in dynamic_keywords)
        return parking_score, -dynamic_penalty, name

    known = [name for name in existing if name in by_name]
    known.sort(key=score, reverse=True)
    raw = [item["name"] for item in scenes if item.get("name") not in known]
    raw.sort(key=score, reverse=True)
    return (known + raw)[: args.num_scenes]


def _find_scene_meta_path(args: argparse.Namespace) -> Path | None:
    candidates = [
        args.raw_nuscenes_root / args.version / "scene.json",
        args.minimal_root / args.version / "scene.json",
        Path("/root/autodl-tmp/lizichong/dataset/nuscenes_trainval_raw") / args.version / "scene.json",
        Path("/root/autodl-tmp/lizichong/dataset/nuscenes_trainval_meta") / args.version / "scene.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def ensure_scene_artifacts(scene_dir: Path, qwen_exp: Path, baseline_exp: Path, args: argparse.Namespace, failures: list[dict[str, Any]]) -> None:
    if args.run_navigation and qwen_exp.exists():
        if not (baseline_exp / "navigation" / "bev_map.npz").exists() and (scene_dir / "poses.txt").exists():
            map_output = baseline_exp / "navigation" / "bev_map.npz"
            run_stage(
                "build_bev_map",
                [
                    "python",
                    "conceptgraph/scripts/build_nuscenes_bev_map.py",
                    "--scene-dir",
                    str(scene_dir),
                    "--map-root",
                    str(args.map_root),
                    "--scene-name",
                    scene_dir.name,
                    "--output",
                    str(map_output),
                ],
                failures,
                scene_dir.name,
                allow_fail=True,
            )
        run_stage(
            "navigation_examples",
            [
                "python",
                "conceptgraph/scripts/run_nuscenes_navigation_examples.py",
                "--scene-dir",
                str(scene_dir),
                "--qwen-exp-dir",
                str(qwen_exp),
                "--baseline-exp-dir",
                str(baseline_exp),
                "--out-dir",
                str(qwen_exp / "navigation_examples"),
                "--visualize",
            ],
            failures,
            scene_dir.name,
            allow_fail=True,
        )
    if not qwen_exp.exists():
        raise FileNotFoundError(f"Missing experiment directory: {qwen_exp}")


def run_stage(stage: str, cmd: list[str], failures: list[dict[str, Any]], scene_name: str, allow_fail: bool = False) -> None:
    try:
        subprocess.run(cmd, cwd="/root/autodl-tmp/lizichong", check=True)
    except Exception as exc:
        failures.append({"scene_id": scene_name, "failed_stage": stage, "error": str(exc), "command": cmd})
        if not allow_fail:
            raise


def count_frames(scene_dir: Path) -> int:
    color_dir = scene_dir / "color"
    if color_dir.exists():
        return len(list(color_dir.glob("*.jpg"))) + len(list(color_dir.glob("*.png")))
    poses = scene_dir / "poses.txt"
    if poses.exists():
        data = np.loadtxt(poses)
        return int(1 if data.ndim == 1 else len(data))
    return 0


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path: Path, scene_rows: list[dict[str, Any]], nav_rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    lines = [
        "# nuScenes Semantic Navigation Experiment Report",
        "",
        f"- Scenes: {len(scene_rows)}",
        f"- Navigation cases: {len(nav_rows)}",
        f"- Failures: {len(failures)}",
        "",
        "## Scene Summary",
        "",
    ]
    for row in scene_rows:
        lines.append(
            f"- {row.get('scene_name')}: frames={row.get('num_frames', row.get('frame_count'))}, "
            f"nodes={row.get('num_scenegraph_nodes')}, edges={row.get('num_scenegraph_edges')}, "
            f"dynamic_nodes={row.get('num_dynamic_nodes')}"
        )
    lines.extend(["", "## Failure Summary", ""])
    for failure in failures[:20]:
        lines.append(f"- {failure.get('scene_id')}: {failure.get('failed_stage')} - {failure.get('error')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
