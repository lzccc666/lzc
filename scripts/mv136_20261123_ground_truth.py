"""隔离导出确认集官方GT；独立齐次变换核验视域，不读取方法预测。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from conceptgraph.nuscenes.benchmark_ground_truth import official_category
from conceptgraph.nuscenes.expansion_benchmark import iter_records
from conceptgraph.project_paths import PROJECT_ROOT
from multiview_identity_repair_20261123_prepare_validation import digest, dump


def in_view(center, view):
    xyz = np.linalg.solve(np.asarray(view["camera_pose"]), np.r_[center, 1.])[:3]
    uvz = np.asarray(view["intrinsic"]) @ xyz
    return bool(1 < xyz[2] < 65 and 0 <= uvz[0]/uvz[2] < view["width"] and 0 <= uvz[1]/uvz[2] < view["height"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "dataset/processed/mv136_20261123_sixscene_v1/manifest.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    groups = [(s, g) for s in manifest["scenes"] for g in s["groups"]]
    selected = {g["sample_token"]: [] for _, g in groups}
    meta = PROJECT_ROOT / "dataset/raw/nuscenes_trainval/v1.0-trainval"
    categories = {r["token"]: r["name"] for r in iter_records(meta / "category.json")}
    instances = {r["token"]: categories[r["category_token"]] for r in iter_records(meta / "instance.json")}
    for row in iter_records(meta / "sample_annotation.json"):
        if row["sample_token"] not in selected:
            continue
        category = official_category(instances[row["instance_token"]])
        if category and int(row["visibility_token"]) >= 2:
            selected[row["sample_token"]].append(dict(annotation_token=row["token"], instance_token=row["instance_token"],
                category=category, center=row["translation"], size_wlh=row["size"], rotation_wxyz=row["rotation"],
                visibility_token=row["visibility_token"], num_lidar_pts=row["num_lidar_pts"]))
    target = args.manifest.parent / "mv136_20261123_evaluation_ground_truth.jsonl"
    counts = dict(groups=len(groups), union_GT=0, front_GT=0, side_only_GT=0, front3_GT=0, rear_only_GT=0)
    with target.open("x") as stream:
        for scene, group in groups:
            objects = []
            for obj in selected[group["sample_token"]]:
                visible = [c for c, v in group["views"].items() if in_view(obj["center"], v)]
                if visible:
                    objects.append(dict(obj, visible_cameras=visible))
            objects.sort(key=lambda o: o["instance_token"])
            assert len(objects) == len({o["instance_token"] for o in objects})
            front = sum("CAM_FRONT" in o["visible_cameras"] for o in objects)
            counts["union_GT"] += len(objects)
            counts["front_GT"] += front
            f3=sum(bool(set(o["visible_cameras"]) & {"CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT"}) for o in objects)
            counts["front3_GT"] += f3
            counts["rear_only_GT"] += len(objects)-f3
            counts["side_only_GT"] += len(objects) - front
            stream.write(json.dumps(dict(scene=scene["scene"], frame_idx=group["frame_idx"], sample_token=group["sample_token"],
                official_sample_timestamp_s=group["official_sample_timestamp_s"], objects=objects)) + "\n")
    release = dict(counts=counts, input_sha256={str(args.manifest): digest(args.manifest), **{str(meta/(n+".json")): digest(meta/(n+".json")) for n in ("category", "instance", "sample_annotation")}},
        output=str(target), output_sha256=digest(target), source_sha256=digest(__file__),
        criteria="same six classes, visibility>=2, 1<camera_z<65 and center in real image FOV; instance deduplicated per group",
        timestamp_limit="official sample-time positions, not motion compensated; camera/group offsets retained",
        role="pre-existing development/debug confirmation data, not unseen test", GT_in_inference=False,
        method_results_accessed=False, new_test_opened=False)
    dump(args.manifest.parent / "mv136_20261123_ground_truth_release.json", release)
    print(json.dumps(dict(GT_isolation_complete=True, groups=len(groups), output=str(target))), flush=True)


if __name__ == "__main__":
    main()
