"""Bounded, resumable preprocessing over the frozen expansion manifest."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from conceptgraph.nuscenes.expansion_benchmark import content_sha256
from conceptgraph.scripts.prepare_lane_expansion import DEFAULT_ROOT, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(
        description="按冻结场景清单构建地图、检测及建图输入"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--phase", choices=("maps", "detections", "mapping"), required=True
    )
    parser.add_argument("--scene", action="append")
    parser.add_argument("--reuse-detection-models", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(DEFAULT_ROOT):
        raise ValueError("preprocessing must remain in the expansion root")
    phase_directory = root / "preprocessing" / args.phase
    phase_directory.mkdir(parents=True, exist_ok=True)
    phase_lock = (phase_directory / "phase.lock").open("a")
    fcntl.flock(phase_lock, fcntl.LOCK_EX)
    selection = json.loads((root / "scene_selection_manifest.json").read_text())
    names = sorted(r["scene"] for r in selection["scenes"])
    if args.scene:
        if set(args.scene) - set(names):
            raise ValueError("unknown scene")
        names = sorted(args.scene)
    env = dict(os.environ)
    env.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HOME=str(root / "hf_cache"),
    )
    os.environ.update(env)
    detection_models = None
    for scene in names:
        converted = root / "converted" / scene
        completion = converted / "conversion_complete.json"
        if not completion.exists():
            raise FileNotFoundError(f"incomplete conversion: {scene}")
        if args.phase == "maps":
            command = [
                sys.executable,
                "-m",
                "conceptgraph.scripts.build_lane_constrained_bev",
                "--scene-dir",
                str(converted),
                "--scene-name",
                scene,
                "--map-root",
                str(PROJECT_ROOT / "dataset/nuscenes_map_expansion"),
                "--nuscenes-meta-root",
                str(PROJECT_ROOT / "dataset/nuscenes_trainval_raw"),
                "--version",
                "v1.0-trainval",
                "--output-dir",
                str(root / "lane_maps" / scene),
            ]
        elif args.phase == "detections":
            command = [
                sys.executable,
                "-m",
                "conceptgraph.scripts.streamlined_detections",
                f"dataset_root={root / 'converted'}",
                f"scene_id={scene}",
                f"dataset_config={PROJECT_ROOT / 'conceptgraph/nuscenes/nuscenes.yaml'}",
                f"classes_file={PROJECT_ROOT / 'conceptgraph/vehicle_classes.txt'}",
                "exp_suffix=streamlined_yolo_outdoor_objects_v1",
                "save_video=false",
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "conceptgraph.slam.streamlined_mapping",
                f"dataset_root={root / 'converted'}",
                f"scene_id={scene}",
                f"dataset_config={PROJECT_ROOT / 'conceptgraph/nuscenes/nuscenes.yaml'}",
                "detection_folder_name=exp_streamlined_yolo_outdoor_objects_v1/detections",
                "gsa_variant=streamlined_yolo",
                "+exp_suffix=streamlined_yolo_outdoor_objects_v1",
                "save_suffix=outdoor_objects_tracking_gate",
                "device=cuda:0",
                "save_objects_all_frames=true",
                "save_pcd=false",
                "vis_render=false",
                "obj_min_detections=1",
                "min_points_threshold=4",
                "dbscan_remove_noise=false",
                "match_method=sim_sum",
                "sim_threshold=0.35",
                "phys_bias=0.5",
                "merge_overlap_thresh=-1",
                "max_tracking_distance_m=8.0",
            ]
        material = {
            "command": command,
            "selection": selection["sha256"],
            "conversion": json.loads(completion.read_text()),
            "frame_alignment_sha256": content_sha256(
                json.loads((converted / "frame_alignment.json").read_text())
            ),
        }
        signature = content_sha256(material)
        ledger = root / "preprocessing" / args.phase / scene
        ledger.mkdir(parents=True, exist_ok=True)
        completed = ledger / "complete.json"
        if completed.exists():
            previous = json.loads(completed.read_text())
            if previous["input_sha256"] != signature:
                raise ValueError(f"completed job input changed: {ledger}")
            print("VERIFIED_EXISTING", args.phase, scene, flush=True)
            continue
        attempt = len(list(ledger.glob("attempt_*.json"))) + 1
        log = ledger / f"attempt_{attempt:03}.log"
        executed_command = list(command)
        if args.phase in {"detections", "mapping"}:
            # Logging destinations are isolated per attempt and do not affect
            # the semantic input signature used by already completed jobs.
            executed_command.append(f"hydra.run.dir={ledger / f'hydra_attempt_{attempt:03}'}")
        started = time.time()
        with log.open("x") as handle:
            try:
                if args.phase == "detections" and args.reuse_detection_models:
                    from hydra import compose, initialize_config_dir
                    from conceptgraph.scripts.streamlined_detections import (
                        load_detection_models,
                        run_detections,
                    )

                    with (
                        contextlib.redirect_stdout(handle),
                        contextlib.redirect_stderr(handle),
                    ):
                        with initialize_config_dir(
                            version_base=None,
                            config_dir=str(PROJECT_ROOT / "conceptgraph/hydra_configs"),
                        ):
                            cfg = compose(
                                config_name="streamlined_detections",
                                overrides=command[3:],
                            )
                        if detection_models is None:
                            detection_models = load_detection_models(cfg)
                        run_detections(cfg, detection_models, skip_existing=True)
                    returncode = 0
                else:
                    result = subprocess.run(
                        executed_command,
                        cwd=PROJECT_ROOT,
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        timeout=3600,
                    )
                    returncode = result.returncode
            except subprocess.TimeoutExpired:
                returncode = -1
            except Exception:
                traceback.print_exc(file=handle)
                returncode = 1
        record = {
            "phase": args.phase,
            "scene": scene,
            "command": command,
            "executed_command": executed_command,
            "input_sha256": signature,
            "started_unix": started,
            "runtime_s": time.time() - started,
            "returncode": returncode,
            "passed": returncode == 0,
            "log": str(log),
        }
        record["model_reuse_enabled"] = (
            args.phase == "detections" and args.reuse_detection_models
        )
        (ledger / f"attempt_{attempt:03}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n"
        )
        if returncode == 0:
            completed.write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n"
            )
        print(json.dumps(record), flush=True)
        if returncode:
            raise RuntimeError(f"preprocessing failed: {ledger}; retained attempt log")


if __name__ == "__main__":
    main()
