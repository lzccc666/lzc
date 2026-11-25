"""独立回查官方时钟/标定/pose及最近过去LiDAR选择；仅读原始metadata。"""
import json
from pathlib import Path

import numpy as np

from conceptgraph.nuscenes.expansion_benchmark import iter_records
from conceptgraph.project_paths import PROJECT_ROOT
from multiview_identity_repair_20261123_prepare_validation import digest, dump


def transform(row):
    w,x,y,z = np.asarray(row["rotation"],float) / np.linalg.norm(row["rotation"])
    matrix=np.eye(4)
    matrix[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    matrix[:3,3]=row["translation"]
    return matrix


def main():
    root=PROJECT_ROOT / "dataset/processed/multiview_identity_repair_20261123_nextlogs_v1"
    manifest=json.loads((root/"manifest.json").read_text())
    plans=json.loads((root/"raw_sensor_plan.json").read_text())["plans"]
    scene_tokens={s["scene_token"]:s["scene"] for s in manifest["scenes"]}
    meta=PROJECT_ROOT / "dataset/raw/nuscenes_trainval/v1.0-trainval"
    samples={r["token"]:r for r in iter_records(meta/"sample.json") if r["scene_token"] in scene_tokens}
    wanted={p[k]["token"] for p in plans for k in ("camera","lidar") if p[k]}
    actual, lidar_candidates={}, {s:[] for s in scene_tokens.values()}
    for r in iter_records(meta/"sample_data.json"):
        if r["token"] in wanted:
            actual[r["token"]]=r
        sample=samples.get(r["sample_token"])
        if sample and "/LIDAR_TOP/" in r["filename"]:
            lidar_candidates[scene_tokens[sample["scene_token"]]].append(r)
    assert set(actual)==wanted
    pose_ids={r["ego_pose_token"] for r in actual.values()}
    poses={r["token"]:r for r in iter_records(meta/"ego_pose.json") if r["token"] in pose_ids}
    calibs={r["token"]:r for r in iter_records(meta/"calibrated_sensor.json")}
    views={(s["scene"],g["frame_idx"],c):v for s in manifest["scenes"] for g in s["groups"] for c,v in g["views"].items()}
    maximum_pose_error=maximum_K_error=0.
    for plan in plans:
        camera=actual[plan["camera"]["token"]]
        assert camera == plan["camera"]
        choices=[r for r in lidar_candidates[plan["scene"]] if 0<=camera["timestamp"]-r["timestamp"]<=100000]
        expected=max(choices,key=lambda r:r["timestamp"]) if choices else None
        assert (expected["token"] if expected else None)==(plan["lidar"]["token"] if plan["lidar"] else None)
        v=views[plan["scene"],plan["frame_idx"],plan["channel"]]
        assert camera["timestamp"]/1e6==v["camera_timestamp_s"] and camera["filename"]==v["camera_source_filename"]
        assert camera["width"]==v["width"] and camera["height"]==v["height"]
        c=calibs[camera["calibrated_sensor_token"]]
        cw=transform(poses[camera["ego_pose_token"]]) @ transform(c)
        maximum_pose_error=max(maximum_pose_error,float(np.max(np.abs(cw-np.asarray(v["camera_pose"])))))
        maximum_K_error=max(maximum_K_error,float(np.max(np.abs(np.asarray(c["camera_intrinsic"])-v["intrinsic"]))))
        if expected:
            assert expected==plan["lidar"] and expected["timestamp"]/1e6==v["lidar_timestamp_s"]
            lw=transform(poses[expected["ego_pose_token"]]) @ transform(calibs[expected["calibrated_sensor_token"]])
            maximum_pose_error=max(maximum_pose_error,float(np.max(np.abs(lw-np.asarray(v["lidar_world_pose"])))))
        else:
            assert v["lidar_timestamp_s"] is None
    assert maximum_pose_error<1e-10 and maximum_K_error==0
    sample_reference=[g["official_sample_timestamp_s"]-g["reference_timestamp_s"] for s in manifest["scenes"] for g in s["groups"]]
    sample_camera=[g["official_sample_timestamp_s"]-v["camera_timestamp_s"] for s in manifest["scenes"] for g in s["groups"] for v in g["views"].values()]
    out=dict(passed=True,views=len(plans),official_sensor_records=len(actual),latest_past_selection_verified=True,
        camera_timestamps_and_filenames_equal_official=True,intrinsic_max_abs_error=maximum_K_error,
        independent_quaternion_pose_max_abs_error=maximum_pose_error,
        min_GT_sample_minus_reference_s=min(sample_reference),max_GT_sample_minus_reference_s=max(sample_reference),
        min_GT_sample_minus_camera_s=min(sample_camera),max_GT_sample_minus_camera_s=max(sample_camera),
        source_sha256=digest(__file__),manifest_sha256=digest(root/"manifest.json"),GT_in_inference=False,
        new_test_opened=False,new_model_calls=0)
    dump(PROJECT_ROOT/"outputs/reports/multiview_identity_repair_20261123/validation/nextlogs/official_source_audit.json",out)
    print(json.dumps(out),flush=True)


if __name__=="__main__":
    main()
