"""
nuScenes to RGB-D Dataset Converter for ConceptGraphs

This script converts nuScenes dataset to the format required by ConceptGraphs:
- RGB images
- Depth maps (generated from LiDAR point clouds)
- Camera poses
- Camera intrinsics

Usage:
    python convert_nuscenes_to_rgbd.py --nuscenes_root /path/to/nuscenes --output_root /path/to/output
"""

import os
import json
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from scipy.spatial.transform import Rotation as R
import struct


def quaternion_to_rotation_matrix(quaternion: List[float]) -> np.ndarray:
    """
    Convert quaternion [qw, qx, qy, qz] to rotation matrix.
    
    Args:
        quaternion: List of [qw, qx, qy, qz]
        
    Returns:
        3x3 rotation matrix
    """
    qw, qx, qy, qz = quaternion
    rotation = R.from_quat([qx, qy, qz, qw])
    return rotation.as_matrix()


def load_nuscenes_json(json_path: str) -> List[Dict]:
    """Load JSON file from nuScenes dataset."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def create_depth_map_from_lidar(
    lidar_data: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    image_width: int = 1600,
    image_height: int = 900
) -> np.ndarray:
    """
    Create depth map by projecting LiDAR points onto camera plane.
    
    Args:
        lidar_data: Nx5 array of LiDAR points [x, y, z, intensity, ring]
        intrinsic: Camera intrinsic matrix 3x3
        extrinsic: Camera extrinsic matrix 4x4 (transform from LiDAR to camera)
        image_width: Image width
        image_height: Image height
        
    Returns:
        Depth map of shape (image_height, image_width)
    """
    depth_map = np.zeros((image_height, image_width), dtype=np.float32)
    
    # Extract points (x, y, z)
    points = lidar_data[:, :3]
    
    # Transform points from LiDAR frame to camera frame
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]
    
    # Apply extrinsic transformation
    points_cam = (rotation @ points.T).T + translation
    
    # Filter points in front of camera (z > 0)
    valid_mask = points_cam[:, 2] > 0
    points_cam = points_cam[valid_mask]
    
    # Project to image plane
    points_img = (intrinsic @ points_cam.T).T
    points_img = points_img[:, :2] / points_img[:, 2:3]
    
    # Filter points within image bounds
    u = points_img[:, 0].astype(int)
    v = points_img[:, 1].astype(int)
    
    valid_mask = (
        (u >= 0) & (u < image_width) &
        (v >= 0) & (v < image_height)
    )
    
    u = u[valid_mask]
    v = v[valid_mask]
    depths = points_cam[:, 2][valid_mask]
    
    # Fill depth map (keep closest point for each pixel)
    for i in range(len(u)):
        current_depth = depth_map[v[i], u[i]]
        if current_depth == 0 or depths[i] < current_depth:
            depth_map[v[i], u[i]] = depths[i]
    
    return depth_map


def load_lidar_pointcloud(lidar_path: str) -> np.ndarray:
    """
    Load LiDAR point cloud from binary file.
    
    Args:
        lidar_path: Path to .pcd.bin file
        
    Returns:
        Nx5 array of points [x, y, z, intensity, ring]
    """
    scan = np.fromfile(lidar_path, dtype=np.float32)
    # Each point is 5 floats (x, y, z, intensity, ring)
    points = scan.reshape((-1, 5))
    return points


class NuScenesToRGBDConverter:
    """
    Converter class for transforming nuScenes data to RGB-D format.
    """
    
    def __init__(
        self,
        nuscenes_root: str,
        output_root: str,
        camera_channel: str = 'CAM_FRONT',
        stride: int = 1
    ):
        """
        Initialize the converter.
        
        Args:
            nuscenes_root: Path to nuScenes mini dataset root
            output_root: Path to save converted RGB-D dataset
            camera_channel: Camera channel to use (e.g., 'CAM_FRONT', 'CAM_BACK')
            stride: Sample every N frames
        """
        self.nuscenes_root = Path(nuscenes_root)
        self.output_root = Path(output_root)
        self.camera_channel = camera_channel
        self.stride = stride
        
        # Paths to JSON files
        self.json_root = self.nuscenes_root / 'v1.0-mini'
        
        # Load all JSON data
        self._load_json_data()
        
        # Create output directories
        self._create_output_dirs()
    
    def _load_json_data(self):
        """Load all necessary JSON files from nuScenes."""
        print("Loading nuScenes JSON data...")
        
        self.samples = load_nuscenes_json(
            self.json_root / 'v1.0-mini' / 'sample.json'
        )
        self.scenes = load_nuscenes_json(
            self.json_root / 'v1.0-mini' / 'scene.json'
        )
        self.sample_data = load_nuscenes_json(
            self.json_root / 'v1.0-mini' / 'sample_data.json'
        )
        self.calibrated_sensors = load_nuscenes_json(
            self.json_root / 'v1.0-mini' / 'calibrated_sensor.json'
        )
        self.ego_poses = load_nuscenes_json(
            self.json_root / 'v1.0-mini' / 'ego_pose.json'
        )
        self.sensors = load_nuscenes_json(
            self.json_root / 'v1.0-mini' / 'sensor.json'
        )
        
        print(f"Loaded {len(self.samples)} samples from {len(self.scenes)} scenes")
    
    def _create_output_dirs(self):
        """Create output directories for converted dataset."""
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_root / 'color').mkdir(exist_ok=True)
        (self.output_root / 'depth').mkdir(exist_ok=True)
        (self.output_root / 'pose').mkdir(exist_ok=True)
        
        print(f"Created output directories in {self.output_root}")
    
    def _get_camera_intrinsics(self, calibrated_sensor_token: str) -> np.ndarray:
        """Get camera intrinsic matrix from calibrated sensor token."""
        calibrated_sensor = next(
            cs for cs in self.calibrated_sensors
            if cs['token'] == calibrated_sensor_token
        )
        
        intrinsic = np.array(calibrated_sensor['camera_intrinsic'])
        return intrinsic
    
    def _get_lidar_to_camera_transform(
        self,
        calibrated_sensor_token: str
    ) -> np.ndarray:
        """Get transformation matrix from LiDAR to camera."""
        calibrated_sensor = next(
            cs for cs in self.calibrated_sensors
            if cs['token'] == calibrated_sensor_token
        )
        
        translation = np.array(calibrated_sensor['translation'])
        rotation = quaternion_to_rotation_matrix(calibrated_sensor['rotation'])
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        
        return transform
    
    def _get_ego_pose(self, ego_pose_token: str) -> np.ndarray:
        """Get ego vehicle pose from token."""
        ego_pose = next(
            ep for ep in self.ego_poses
            if ep['token'] == ego_pose_token
        )
        
        translation = np.array(ego_pose['translation'])
        rotation = quaternion_to_rotation_matrix(ego_pose['rotation'])
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        
        return transform
    
    def _get_sample_data_for_channel(
        self,
        sample_token: str,
        channel: str
    ) -> Optional[Dict]:
        """Get sample data for a specific camera channel."""
        for sd in self.sample_data:
            if sd['sample_token'] == sample_token and channel in sd['filename']:
                return sd
        return None
    
    def _save_camera_params(self, intrinsic: np.ndarray):
        """Save camera intrinsic parameters."""
        camera_params = {
            'intrinsic': intrinsic.tolist(),
            'image_width': 1600,
            'image_height': 900,
            'camera_channel': self.camera_channel
        }
        
        with open(self.output_root / 'camera_params.json', 'w') as f:
            json.dump(camera_params, f, indent=2)
    
    def convert_scene(self, scene_token: str):
        """
        Convert a single scene to RGB-D format.
        
        Args:
            scene_token: Token of the scene to convert
        """
        # Get scene
        scene = next(s for s in self.scenes if s['token'] == scene_token)
        scene_name = scene['name']
        
        print(f"\nConverting scene: {scene_name}")
        
        # Create scene output directory
        scene_output = self.output_root / scene_name
        scene_output.mkdir(exist_ok=True)
        (scene_output / 'color').mkdir(exist_ok=True)
        (scene_output / 'depth').mkdir(exist_ok=True)
        (scene_output / 'pose').mkdir(exist_ok=True)
        
        # Get all samples for this scene
        current_sample_token = scene['first_sample_token']
        frame_idx = 0
        saved_idx = 0
        
        while current_sample_token:
            # Get sample
            sample = next(
                s for s in self.samples
                if s['token'] == current_sample_token
            )
            
            # Only process every stride frames
            if frame_idx % self.stride == 0:
                # Get camera sample data
                camera_data = self._get_sample_data_for_channel(
                    sample['token'],
                    self.camera_channel
                )
                
                if camera_data is None:
                    print(f"Warning: No camera data for sample {sample['token']}")
                    current_sample_token = sample['next']
                    frame_idx += 1
                    continue
                
                # Get LiDAR sample data
                lidar_data = self._get_sample_data_for_channel(
                    sample['token'],
                    'LIDAR_TOP'
                )
                
                if lidar_data is None:
                    print(f"Warning: No LiDAR data for sample {sample['token']}")
                    current_sample_token = sample['next']
                    frame_idx += 1
                    continue
                
                try:
                    # Load RGB image
                    camera_path = self.nuscenes_root / camera_data['filename']
                    rgb_image = cv2.imread(str(camera_path))
                    
                    # Load LiDAR point cloud
                    lidar_path = self.nuscenes_root / lidar_data['filename']
                    lidar_points = load_lidar_pointcloud(str(lidar_path))
                    
                    # Get transformations
                    intrinsic = self._get_camera_intrinsics(
                        camera_data['calibrated_sensor_token']
                    )
                    
                    lidar_to_cam = self._get_lidar_to_camera_transform(
                        camera_data['calibrated_sensor_token']
                    )
                    
                    ego_pose = self._get_ego_pose(
                        camera_data['ego_pose_token']
                    )
                    
                    # Create depth map from LiDAR
                    depth_map = create_depth_map_from_lidar(
                        lidar_points,
                        intrinsic,
                        lidar_to_cam,
                        image_width=1600,
                        image_height=900
                    )
                    
                    # Save RGB image
                    rgb_output_path = scene_output / 'color' / f'frame_{saved_idx:06d}.jpg'
                    cv2.imwrite(str(rgb_output_path), rgb_image)
                    
                    # Save depth map as 16-bit PNG
                    depth_output_path = scene_output / 'depth' / f'frame_{saved_idx:06d}.png'
                    depth_uint16 = (depth_map * 1000).astype(np.uint16)  # Scale to mm
                    cv2.imwrite(str(depth_output_path), depth_uint16)
                    
                    # Save pose as 4x4 matrix
                    pose_output_path = scene_output / 'pose' / f'frame_{saved_idx:06d}.txt'
                    np.savetxt(str(pose_output_path), ego_pose)
                    
                    # Save camera parameters for first frame
                    if saved_idx == 0:
                        self._save_camera_params(intrinsic)
                    
                    saved_idx += 1
                    print(f"  Frame {saved_idx}: Saved RGB, depth, and pose")
                    
                except Exception as e:
                    print(f"Error processing sample {sample['token']}: {e}")
            
            current_sample_token = sample['next']
            frame_idx += 1
        
        # Save scene trajectory
        print(f"Scene {scene_name} conversion complete: {saved_idx} frames saved")
    
    def convert_all_scenes(self):
        """Convert all scenes in the dataset."""
        print("Starting conversion of all scenes...\n")
        
        for scene in self.scenes:
            self.convert_scene(scene['token'])
        
        print("\n✓ All scenes converted successfully!")
        print(f"Output location: {self.output_root}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert nuScenes dataset to RGB-D format for ConceptGraphs'
    )
    parser.add_argument(
        '--nuscenes_root',
        type=str,
        default='/home/lzc/PycharmProjects/concept-graphs/dataset/v1.0-mini',
        help='Path to nuScenes mini dataset root'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='/home/lzc/PycharmProjects/concept-graphs/dataset/nuscenes_rgbd',
        help='Path to save converted RGB-D dataset'
    )
    parser.add_argument(
        '--camera_channel',
        type=str,
        default='CAM_FRONT',
        choices=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
        help='Camera channel to use for conversion'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Sample every N frames (for faster processing)'
    )
    parser.add_argument(
        '--scene_name',
        type=str,
        default=None,
        help='Convert specific scene by name (default: convert all)'
    )
    
    args = parser.parse_args()
    
    # Create converter
    converter = NuScenesToRGBDConverter(
        nuscenes_root=args.nuscenes_root,
        output_root=args.output_root,
        camera_channel=args.camera_channel,
        stride=args.stride
    )
    
    # Convert scenes
    if args.scene_name:
        # Convert specific scene
        scene = next(
            (s for s in converter.scenes if s['name'] == args.scene_name),
            None
        )
        if scene is None:
            print(f"Error: Scene '{args.scene_name}' not found!")
            print("Available scenes:")
            for s in converter.scenes:
                print(f"  - {s['name']}")
            return
        converter.convert_scene(scene['token'])
    else:
        # Convert all scenes
        converter.convert_all_scenes()
    
    print("\n✓ Conversion complete!")


if __name__ == '__main__':
    main()