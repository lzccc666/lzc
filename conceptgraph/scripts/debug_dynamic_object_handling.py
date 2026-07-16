from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import open3d as o3d
import torch

from conceptgraph.slam.mapping import merge_detections_to_objects
from conceptgraph.slam.slam_classes import DetectionList, MapObjectList
from conceptgraph.slam.utils import merge_objects


def make_cfg(**overrides):
    cfg = SimpleNamespace(
        enable_dynamic_object_handling=True,
        dynamic_classes=["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"],
        static_classes=["traffic sign", "traffic light", "road", "sidewalk", "building", "tree", "fence", "pole"],
        velocity_threshold=0.5,
        frame_dt=1.0,
        center_history_size=4,
        max_tracking_distance_m=20.0,
        downsample_voxel_size=0.05,
        dbscan_remove_noise=False,
        dbscan_eps=0.05,
        dbscan_min_points=2,
        spatial_sim_type="iou",
        merge_overlap_thresh=0.1,
        merge_visual_sim_thresh=0.0,
        merge_text_sim_thresh=0.0,
        device="cpu",
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_obj(class_name: str, center: tuple[float, float, float], frame: int, class_id: int = 0) -> dict:
    points = np.asarray(
        [
            [center[0] - 0.2, center[1] - 0.2, center[2] - 0.2],
            [center[0] + 0.2, center[1] - 0.2, center[2] - 0.2],
            [center[0] - 0.2, center[1] + 0.2, center[2] + 0.2],
            [center[0] + 0.2, center[1] + 0.2, center[2] + 0.2],
            [center[0], center[1], center[2]],
        ],
        dtype=float,
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.ones_like(points) * 0.5)
    bbox = pcd.get_axis_aligned_bounding_box()
    return {
        "image_idx": [frame],
        "mask_idx": [0],
        "color_path": [""],
        "class_name": [class_name],
        "class_id": [class_id],
        "num_detections": 1,
        "mask": [np.ones((4, 4), dtype=bool)],
        "xyxy": [[0, 0, 4, 4]],
        "conf": [1.0],
        "n_points": [len(points)],
        "pixel_area": [16],
        "contain_number": [None],
        "inst_color": np.asarray([0.2, 0.4, 0.6]),
        "is_background": False,
        "pcd": pcd,
        "bbox": bbox,
        "clip_ft": torch.nn.functional.normalize(torch.ones(8), dim=0),
        "text_ft": torch.nn.functional.normalize(torch.ones(8), dim=0),
    }


def run_case(name: str, first: dict, second: dict, expected_dynamic: bool, expected_points_change: bool) -> None:
    cfg = make_cfg()
    objects = MapObjectList([first])
    detections = DetectionList([second])
    before_points = len(objects[0]["pcd"].points)
    agg_sim = torch.tensor([[1.0]], dtype=torch.float32)
    objects = merge_detections_to_objects(cfg, detections, objects, agg_sim)
    after_points = len(objects[0]["pcd"].points)
    is_dynamic = objects[0]["motion_state"] == "dynamic" and bool(objects[0]["is_short_term"])
    points_changed = after_points > before_points
    assert is_dynamic == expected_dynamic, f"{name}: expected dynamic={expected_dynamic}, got {objects[0]['motion_state']}"
    assert points_changed == expected_points_change, f"{name}: expected points_changed={expected_points_change}, got {before_points}->{after_points}"
    print(f"{name}: ok motion_state={objects[0]['motion_state']} velocity={objects[0]['velocity']:.3f} points={before_points}->{after_points}")


def main() -> None:
    run_case("static_car_long_term_merge", make_obj("car", (0, 0, 0), 0), make_obj("car", (0.1, 0, 0), 1), False, True)
    run_case("moving_car_short_term", make_obj("car", (0, 0, 0), 0), make_obj("car", (3.0, 0, 0), 1), True, False)
    run_case("traffic_sign_static_merge", make_obj("traffic sign", (0, 0, 0), 0, 1), make_obj("traffic sign", (3.0, 0, 0), 1, 1), False, True)

    cfg = make_cfg(merge_overlap_thresh=0.01)
    dynamic = make_obj("car", (0, 0, 0), 1)
    dynamic["motion_state"] = "dynamic"
    dynamic["is_short_term"] = True
    dynamic["is_potential_dynamic"] = True
    dynamic["center_history"] = [[0, 0, 0], [3, 0, 0]]
    dynamic["frame_history"] = [0, 1]
    dynamic["last_seen_frame"] = 1
    static = make_obj("car", (0, 0, 0), 0)
    static["motion_state"] = "static"
    static["center_history"] = [[0, 0, 0]]
    static["frame_history"] = [0]
    objects = MapObjectList([dynamic, static])
    merged = merge_objects(cfg, objects)
    assert len(merged) == 2, "dynamic short-term object should not be globally merged"
    print("dynamic_merge_guard: ok")


if __name__ == "__main__":
    main()
