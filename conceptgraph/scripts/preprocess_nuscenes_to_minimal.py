import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw nuScenes data into a minimal CAM_FRONT + LIDAR_TOP subset for ConceptGraphs."
    )
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--scene-name", type=str, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-depth-m", type=float, default=1.0)
    parser.add_argument("--max-depth-m", type=float, default=80.0)
    return parser.parse_args()


def load_table(root: Path, version: str, table_name: str):
    with open(root / version / f"{table_name}.json", "r") as handle:
        return json.load(handle)


def index_by_token(records):
    return {record["token"]: record for record in records}


def build_transform(translation, rotation) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Quaternion(rotation).rotation_matrix
    transform[:3, 3] = np.array(translation, dtype=np.float64)
    return transform


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = transform[:3, :3].T
    inv[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inv


def load_lidar_points(file_path: Path) -> np.ndarray:
    points = np.fromfile(file_path, dtype=np.float32)
    if points.size % 5 != 0:
        raise ValueError(f"Unexpected point count in {file_path}")
    return points.reshape(-1, 5)[:, :3]


def project_lidar_to_camera_depth(
    lidar_points: np.ndarray,
    lidar_calib: dict,
    lidar_pose: dict,
    cam_calib: dict,
    cam_pose: dict,
    camera_intrinsic: np.ndarray,
    image_height: int,
    image_width: int,
    min_depth_m: float,
    max_depth_m: float,
):
    depth = np.zeros((image_height, image_width), dtype=np.float32)
    if lidar_points.size == 0:
        return depth

    lidar_to_ego = build_transform(lidar_calib["translation"], lidar_calib["rotation"])
    ego_lidar_to_global = build_transform(lidar_pose["translation"], lidar_pose["rotation"])
    cam_to_ego = build_transform(cam_calib["translation"], cam_calib["rotation"])
    ego_cam_to_global = build_transform(cam_pose["translation"], cam_pose["rotation"])

    global_to_ego_cam = inverse_transform(ego_cam_to_global)
    ego_to_cam = inverse_transform(cam_to_ego)
    lidar_to_cam = ego_to_cam @ global_to_ego_cam @ ego_lidar_to_global @ lidar_to_ego

    lidar_points_h = np.concatenate(
        [lidar_points, np.ones((lidar_points.shape[0], 1), dtype=np.float64)], axis=1
    )
    cam_points = (lidar_to_cam @ lidar_points_h.T).T[:, :3]

    z = cam_points[:, 2]
    valid = (z > min_depth_m) & (z < max_depth_m)
    cam_points = cam_points[valid]
    z = z[valid]
    if cam_points.shape[0] == 0:
        return depth

    projected = (camera_intrinsic @ cam_points.T).T
    u = np.round(projected[:, 0] / projected[:, 2]).astype(np.int32)
    v = np.round(projected[:, 1] / projected[:, 2]).astype(np.int32)

    in_bounds = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds]

    order = np.argsort(z)
    u = u[order]
    v = v[order]
    z = z[order]
    depth[v, u] = z
    return depth


def collect_scene_samples(scene_record: dict, sample_by_token: dict):
    ordered = []
    sample_token = scene_record["first_sample_token"]
    while sample_token:
        sample_record = sample_by_token[sample_token]
        ordered.append(sample_record)
        sample_token = sample_record["next"]
    return ordered


def collect_sample_data_for_scene(scene_record: dict, sample_by_token: dict, sample_data_by_token: dict):
    """Collect CAM_FRONT and LIDAR_TOP data for a scene"""
    ordered_samples = collect_scene_samples(scene_record, sample_by_token)

    direct_result = []
    for sample_record in ordered_samples:
        data = sample_record.get("data", {})
        cam_token = data.get("CAM_FRONT")
        lidar_token = data.get("LIDAR_TOP")
        if cam_token in sample_data_by_token and lidar_token in sample_data_by_token:
            direct_result.append((sample_record, {"CAM_FRONT": cam_token, "LIDAR_TOP": lidar_token}))
    if direct_result:
        return direct_result

    # Build a mapping from sample_token to its data
    sample_data_by_sample_token = {}
    for sd_token, sd_record in sample_data_by_token.items():
        sample_token = sd_record["sample_token"]
        if sample_token not in sample_data_by_sample_token:
            sample_data_by_sample_token[sample_token] = {}
        filename = sd_record.get("filename", "")
        parts = Path(filename).parts
        if not sd_record.get("is_key_frame", False):
            continue
        if len(parts) < 3 or parts[0] != "samples":
            continue
        if len(parts) >= 2 and parts[-2] == "CAM_FRONT" and "CAM_FRONT" not in sample_data_by_sample_token[sample_token]:
            sample_data_by_sample_token[sample_token]["CAM_FRONT"] = sd_token
        elif len(parts) >= 2 and parts[-2] == "LIDAR_TOP" and "LIDAR_TOP" not in sample_data_by_sample_token[sample_token]:
            sample_data_by_sample_token[sample_token]["LIDAR_TOP"] = sd_token
    
    # Match samples with their data
    result = []
    for sample_record in ordered_samples:
        sample_token = sample_record["token"]
        if sample_token in sample_data_by_sample_token:
            data_tokens = sample_data_by_sample_token[sample_token]
            if "CAM_FRONT" in data_tokens and "LIDAR_TOP" in data_tokens:
                result.append((sample_record, data_tokens))
    
    return result


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()

    scenes = load_table(args.nuscenes_root, args.version, "scene")
    samples = load_table(args.nuscenes_root, args.version, "sample")
    sample_data = load_table(args.nuscenes_root, args.version, "sample_data")
    calibrated_sensors = load_table(args.nuscenes_root, args.version, "calibrated_sensor")
    ego_poses = load_table(args.nuscenes_root, args.version, "ego_pose")

    scene_by_name = {scene["name"]: scene for scene in scenes}
    if args.scene_name not in scene_by_name:
        raise KeyError(f"Scene {args.scene_name} not found in {args.version}")

    sample_by_token = index_by_token(samples)
    sample_data_by_token = index_by_token(sample_data)
    calibrated_sensor_by_token = index_by_token(calibrated_sensors)
    ego_pose_by_token = index_by_token(ego_poses)

    ordered_samples = collect_scene_samples(scene_by_name[args.scene_name], sample_by_token)
    
    # Collect CAM_FRONT and LIDAR_TOP data for the scene
    scene_data = collect_sample_data_for_scene(
        scene_by_name[args.scene_name], 
        sample_by_token, 
        sample_data_by_token
    )
    
    if args.stride > 1:
        scene_data = scene_data[::args.stride]
    if args.max_frames is not None:
        scene_data = scene_data[:args.max_frames]

    output_scene_dir = args.output_root / args.scene_name
    color_dir = output_scene_dir / "color"
    depth_dir = output_scene_dir / "depth"
    ensure_dir(color_dir)
    ensure_dir(depth_dir)

    written_poses = []
    intrinsics = None

    for index, (sample_record, data_tokens) in enumerate(tqdm(scene_data, desc="Converting nuScenes")):
        cam_token = data_tokens.get("CAM_FRONT")
        lidar_token = data_tokens.get("LIDAR_TOP")
        if cam_token is None or lidar_token is None:
            continue

        cam_sd = sample_data_by_token[cam_token]
        lidar_sd = sample_data_by_token[lidar_token]
        cam_calib = calibrated_sensor_by_token[cam_sd["calibrated_sensor_token"]]
        lidar_calib = calibrated_sensor_by_token[lidar_sd["calibrated_sensor_token"]]
        cam_pose = ego_pose_by_token[cam_sd["ego_pose_token"]]
        lidar_pose = ego_pose_by_token[lidar_sd["ego_pose_token"]]

        image_path = args.nuscenes_root / cam_sd["filename"]
        lidar_path = args.nuscenes_root / lidar_sd["filename"]
        if not image_path.exists() or not lidar_path.exists():
            continue

        image = imageio.imread(image_path)
        image_height, image_width = image.shape[:2]
        camera_intrinsic = np.array(cam_calib["camera_intrinsic"], dtype=np.float64)
        if intrinsics is None:
            intrinsics = camera_intrinsic.copy()

        lidar_points = load_lidar_points(lidar_path)
        depth_m = project_lidar_to_camera_depth(
            lidar_points=lidar_points,
            lidar_calib=lidar_calib,
            lidar_pose=lidar_pose,
            cam_calib=cam_calib,
            cam_pose=cam_pose,
            camera_intrinsic=camera_intrinsic,
            image_height=image_height,
            image_width=image_width,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
        )

        depth_mm = np.round(depth_m * 1000.0).astype(np.uint16)
        frame_name = f"{index:06d}"
        imageio.imwrite(color_dir / f"{frame_name}.jpg", image)
        imageio.imwrite(depth_dir / f"{frame_name}.png", depth_mm)

        cam_to_ego = build_transform(cam_calib["translation"], cam_calib["rotation"])
        ego_to_global = build_transform(cam_pose["translation"], cam_pose["rotation"])
        cam_to_global = ego_to_global @ cam_to_ego
        written_poses.append(cam_to_global.reshape(-1))

    if not written_poses:
        raise RuntimeError("No frames were converted. Check the scene name and downloaded data.")

    np.savetxt(output_scene_dir / "poses.txt", np.stack(written_poses), fmt="%.8f")
    np.savetxt(output_scene_dir / "intrinsics.txt", intrinsics, fmt="%.8f")

    print(f"Wrote {len(written_poses)} frames to {output_scene_dir}")


if __name__ == "__main__":
    main()
