"""独立重投影与真实图像解码检查，不读取GT或方法结果。"""
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from conceptgraph.project_paths import PROJECT_ROOT
from conceptgraph.scripts.convert_lane_expansion import nearest_depth_image
from multiview_20261123_prepare_inputs import digest, dump


def main():
    root = PROJECT_ROOT / "dataset/processed/mv136_20261123_sharedscan_v1"
    manifest = json.loads((root / "manifest.json").read_text())
    checked = json.loads((root / "input_verification.json").read_text())
    frozen = {(r["scene"], r["frame_idx"], r["channel"]): r for r in checked}
    records = []
    for scene in manifest["scenes"]:
        for group in scene["groups"]:
            assert group["reference_timestamp_s"] == max(v["camera_timestamp_s"] for v in group["views"].values())
            for channel, view in group["views"].items():
                old = frozen[(scene["scene"], group["frame_idx"], channel)]
                assert digest(view["rgb"]) == old["rgb_sha256"]
                with Image.open(view["rgb"]) as image:
                    image.load()
                    assert image.size == (view["width"], view["height"])
                raw_depth = imageio.imread(view["depth"])
                evidence = np.load(view["projection_evidence"])
                y, x, ids = [evidence[k] for k in ("pixel_y", "pixel_x", "point_index")]
                assert len(set(zip(y.tolist(), x.tolist()))) == len(y)
                if view["lidar_raw"]:
                    assert view["lidar_timestamp_s"] <= view["camera_timestamp_s"]
                    assert 0 <= view["lidar_age_s"] <= .1
                    points = np.fromfile(view["lidar_raw"], np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
                    camera = np.asarray(view["camera_pose"], dtype=np.float64)
                    lidar = np.asarray(view["lidar_world_pose"], dtype=np.float64)
                    # Independent homogeneous matrix solve rather than generator inverse_transform.
                    transform = np.linalg.solve(camera, lidar)
                    pc = (transform @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
                    expected = nearest_depth_image(pc, view["intrinsic"], view["height"], view["width"])
                    expected_mm = np.rint(expected * 1000).astype(np.uint16)
                    mismatched = int(np.count_nonzero(raw_depth != expected_mm))
                    assert mismatched == 0
                    assert np.allclose(pc[ids, 2], evidence["depth_m"], atol=4e-6, rtol=1e-7)
                    uvz = (np.asarray(view["intrinsic"]) @ pc[ids].T).T
                    uv = np.rint(uvz[:, :2] / uvz[:, 2, None]).astype(int)
                    assert np.array_equal(uv[:, 0], x) and np.array_equal(uv[:, 1], y)
                    assert np.array_equal(np.rint(evidence["depth_m"].astype(float) * 1000).astype(np.uint16), raw_depth[y, x]) or np.max(np.abs(evidence["depth_m"].astype(float) - raw_depth[y, x] / 1000)) <= .000504
                else:
                    mismatched = 0
                    assert not raw_depth.any() and len(ids) == 0
                records.append(dict(scene=scene["scene"], frame_idx=group["frame_idx"], channel=channel,
                    valid_pixels=len(ids), image_decoded=True, future_lidar=False,
                    independent_projection_mismatched_pixels=mismatched,
                    actual_lidar_age_s=view["lidar_age_s"],
                    synchronization_wait_s=view["synchronization_wait_s"]))
    ages = [r["actual_lidar_age_s"] for r in records if r["actual_lidar_age_s"] is not None]
    report = dict(passed=True, views=len(records), actual_images_decoded=len(records),
        future_lidar_views=0, missing_past_lidar_views=sum(r["actual_lidar_age_s"] is None for r in records),
        independent_projection_mismatched_pixels=sum(r["independent_projection_mismatched_pixels"] for r in records),
        maximum_lidar_age_s=max(ages), minimum_lidar_age_s=min(ages),
        maximum_synchronization_wait_s=max(r["synchronization_wait_s"] for r in records),
        manifest_sha256=digest(root / "manifest.json"), source_sha256=digest(__file__),
        ground_truth_accessed=False, new_test_opened=False, records=records)
    dump(PROJECT_ROOT / "outputs/reports/multiview_viewcount_20261123/sensors/sharedscan/independent_input_audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
