import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw KITTI-360 data into a minimal single-camera subset for ConceptGraphs."
    )
    parser.add_argument("--kitti360-root", type=Path, required=True)
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cam-id", type=int, default=0, choices=[0])
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-depth-m", type=float, default=80.0)
    parser.add_argument("--min-depth-m", type=float, default=1.0)
    return parser.parse_args()


def read_named_matrix(file_path: Path, name: str, rows: int, cols: int) -> np.ndarray:
    with open(file_path, "r") as handle:
        for line in handle:
            if line.startswith(f"{name}:"):
                values = [float(value) for value in line.replace(f"{name}:", "").split()]
                if len(values) != rows * cols:
                    raise ValueError(f"Expected {rows * cols} values for {name} in {file_path}")
                return np.array(values, dtype=np.float64).reshape(rows, cols)
    raise KeyError(f"Could not find {name} in {file_path}")


def load_perspective_calibration(calibration_dir: Path, cam_id: int):
    perspective_path = calibration_dir / "perspective.txt"
    p_rect = read_named_matrix(perspective_path, f"P_rect_{cam_id:02d}", 3, 4)
    r_rect = read_named_matrix(perspective_path, f"R_rect_{cam_id:02d}", 3, 3)

    cam_to_velo = np.loadtxt(calibration_dir / "calib_cam_to_velo.txt", dtype=np.float64).reshape(3, 4)
    cam_to_velo = np.vstack([cam_to_velo, [0.0, 0.0, 0.0, 1.0]])

    return p_rect, r_rect, cam_to_velo


def load_cam0_to_world(poses_path: Path) -> dict[int, np.ndarray]:
    pose_entries = np.loadtxt(poses_path, dtype=np.float64)
    if pose_entries.ndim == 1:
        pose_entries = pose_entries[None, :]

    poses = {}
    for row in pose_entries:
        frame_id = int(row[0])
        pose = row[1:].reshape(4, 4)
        poses[frame_id] = pose
    return poses


def project_velodyne_to_rectified_depth(
    velo_points: np.ndarray,
    image_height: int,
    image_width: int,
    p_rect: np.ndarray,
    r_rect: np.ndarray,
    cam_to_velo: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    depth = np.zeros((image_height, image_width), dtype=np.float32)

    if velo_points.size == 0:
        return depth

    velo_h = np.concatenate(
        [velo_points[:, :3], np.ones((velo_points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )

    velo_to_cam = np.linalg.inv(cam_to_velo)
    cam_points = (velo_to_cam @ velo_h.T).T[:, :3]
    rect_points = (r_rect @ cam_points.T).T

    z = rect_points[:, 2]
    valid = (z > min_depth_m) & (z < max_depth_m)
    rect_points = rect_points[valid]
    z = z[valid]

    if rect_points.shape[0] == 0:
        return depth

    projected = (p_rect[:, :3] @ rect_points.T).T
    u = np.round(projected[:, 0] / projected[:, 2]).astype(np.int32)
    v = np.round(projected[:, 1] / projected[:, 2]).astype(np.int32)

    in_bounds = (
        (u >= 0)
        & (u < image_width)
        & (v >= 0)
        & (v < image_height)
    )
    u = u[in_bounds]
    v = v[in_bounds]
    z = z[in_bounds]

    order = np.argsort(z)
    u = u[order]
    v = v[order]
    z = z[order]

    depth[v, u] = z
    return depth


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    calibration_dir = args.kitti360-root / "calibration"
    sequence_dir = args.kitti360-root / "data_2d_raw" / args.sequence / f"image_{args.cam_id:02d}" / "data_rect"
    velodyne_dir = args.kitti360-root / "data_3d_raw" / args.sequence / "velodyne_points" / "data"
    poses_path = args.kitti360-root / "data_poses" / args.sequence / "cam0_to_world.txt"

    if not sequence_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {sequence_dir}")
    if not velodyne_dir.exists():
        raise FileNotFoundError(f"Missing velodyne directory: {velodyne_dir}")
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing pose file: {poses_path}")

    p_rect, r_rect, cam_to_velo = load_perspective_calibration(calibration_dir, args.cam_id)
    poses = load_cam0_to_world(poses_path)

    available_pose_frames = sorted(poses.keys())
    selected_frames = [
        frame_id
        for frame_id in available_pose_frames
        if (args.start_frame is None or frame_id >= args.start_frame)
        and (args.end_frame is None or frame_id <= args.end_frame)
    ]

    if args.stride > 1:
        selected_frames = selected_frames[:: args.stride]

    if args.max_frames is not None:
        selected_frames = selected_frames[: args.max_frames]

    output_scene_dir = args.output_root / args.sequence
    color_dir = output_scene_dir / "color"
    depth_dir = output_scene_dir / "depth"
    ensure_dir(color_dir)
    ensure_dir(depth_dir)

    intrinsics_path = output_scene_dir / "intrinsics.txt"
    poses_out_path = output_scene_dir / "poses.txt"

    intrinsics = p_rect[:3, :3].copy()
    np.savetxt(intrinsics_path, intrinsics, fmt="%.8f")

    written_poses = []
    num_written = 0
    num_skipped = 0

    for frame_id in tqdm(selected_frames, desc="Converting KITTI-360"):
        image_path = sequence_dir / f"{frame_id:010d}.png"
        velo_path = velodyne_dir / f"{frame_id:010d}.bin"

        if not image_path.exists() or not velo_path.exists():
            num_skipped += 1
            continue

        image = imageio.imread(image_path)
        image_height, image_width = image.shape[:2]

        velo_points = np.fromfile(velo_path, dtype=np.float32)
        if velo_points.size % 4 != 0:
            num_skipped += 1
            continue
        velo_points = velo_points.reshape(-1, 4)

        depth_m = project_velodyne_to_rectified_depth(
            velo_points=velo_points,
            image_height=image_height,
            image_width=image_width,
            p_rect=p_rect,
            r_rect=r_rect,
            cam_to_velo=cam_to_velo,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
        )

        depth_mm = np.round(depth_m * 1000.0).astype(np.uint16)
        imageio.imwrite(color_dir / f"{frame_id:010d}.png", image)
        imageio.imwrite(depth_dir / f"{frame_id:010d}.png", depth_mm)

        written_poses.append(poses[frame_id].reshape(-1))
        num_written += 1

    if written_poses:
        np.savetxt(poses_out_path, np.stack(written_poses), fmt="%.8f")
    else:
        raise RuntimeError("No frames were converted. Check the selected sequence and paths.")

    print(f"Wrote {num_written} frames to {output_scene_dir}")
    print(f"Skipped {num_skipped} frames due to missing image or LiDAR data")
    print(f"Saved intrinsics to {intrinsics_path}")
    print(f"Saved poses to {poses_out_path}")


if __name__ == "__main__":
    main()
