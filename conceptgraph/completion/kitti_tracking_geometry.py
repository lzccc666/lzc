"""KITTI Tracking的相机2平移、IMU世界位姿和隔离轨迹标签转换。"""
from __future__ import annotations

import math
import numpy as np

from conceptgraph.completion.kitti_geometry import labels


def homogeneous(value):
    matrix = np.eye(4)
    matrix[:3, :np.asarray(value).shape[1]] = value
    return matrix


def calibration(text):
    rows = {}
    for line in text.splitlines():
        fields = line.split(maxsplit=1)
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError('标定行缺少数值')
        rows[fields[0].removesuffix(':')] = np.fromstring(fields[1], sep=' ')
    def get(names, shape):
        key = next((key for key in names if key in rows), None)
        if key is None:
            raise ValueError('缺少KITTI Tracking标定：'+str(names))
        value = rows[key].reshape(shape)
        if not np.isfinite(value).all():
            raise ValueError('标定含非有限值')
        return value
    projection = get(['P2'], (3, 4))
    rect = get(['R_rect', 'R0_rect'], (3, 3))
    velo = get(['Tr_velo_cam', 'Tr_velo_to_cam'], (3, 4))
    imu = get(['Tr_imu_velo', 'Tr_imu_to_velo'], (3, 4))
    intrinsic = projection[:, :3]
    shift = np.linalg.solve(intrinsic, projection[:, 3])
    move = np.eye(4)
    move[:3, 3] = shift
    camera0_from_velo = homogeneous(rect)@homogeneous(velo)
    camera2_from_velo = move@camera0_from_velo
    imu_from_camera0 = np.linalg.inv(homogeneous(imu))@np.linalg.inv(camera0_from_velo)
    imu_from_camera2 = imu_from_camera0@np.linalg.inv(move)
    return dict(projection=projection, intrinsic=intrinsic, camera2_shift=shift,
        camera0_from_velo=camera0_from_velo, camera2_from_velo=camera2_from_velo,
        imu_from_camera0=imu_from_camera0, imu_from_camera2=imu_from_camera2)


def oxts_poses(text):
    values = np.atleast_2d(np.loadtxt(text.splitlines()))
    if values.shape[1] < 6 or not np.isfinite(values).all():
        raise ValueError('OXTS记录不完整或非有限')
    if (np.abs(values[:, 0]) >= 85).any() or (np.abs(values[:, 1]) > 180).any():
        raise ValueError('非法经纬度')
    scale, radius = math.cos(math.radians(values[0, 0])), 6378137.
    positions = np.column_stack([scale*np.radians(values[:, 1])*radius,
        scale*radius*np.log(np.tan(np.radians(90+values[:, 0])/2)), values[:, 2]])
    poses = []
    for row, position in zip(values, positions):
        roll, pitch, yaw = row[3:6]
        cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
        rx = np.array([[1,0,0], [0,cr,-sr], [0,sr,cr]])
        ry = np.array([[cp,0,sp], [0,1,0], [-sp,0,cp]])
        rz = np.array([[cy,-sy,0], [sy,cy,0], [0,0,1]])
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rz@ry@rx, position
        poses.append(pose)
    origin_inverse = np.linalg.inv(poses[0])
    return np.asarray([origin_inverse@pose for pose in poses])


def project_lidar(points, calib, image_wh):
    xyz = np.asarray(points, dtype=float)[:, :3]
    transform = calib['camera2_from_velo']
    camera = xyz@transform[:3, :3].T+transform[:3, 3]
    projected = camera@calib['intrinsic'].T
    valid = np.isfinite(camera).all(axis=1)&(camera[:, 2] > 0)&(camera[:, 2] <= 65.535)
    uv = np.zeros((len(xyz), 2))
    uv[valid] = projected[valid, :2]/projected[valid, 2, None]
    pixels = np.floor(uv).astype(int)
    width, height = image_wh
    valid &= (pixels[:, 0] >= 0)&(pixels[:, 0] < width)&(pixels[:, 1] >= 0)&(pixels[:, 1] < height)
    indices = np.flatnonzero(valid)
    indices = indices[np.argsort(camera[indices, 2], kind='stable')]
    flat = pixels[indices, 1]*width+pixels[indices, 0]
    _, first = np.unique(flat, return_index=True)
    keep = indices[first]
    depth = np.zeros((height, width), dtype=np.uint16)
    depth[pixels[keep, 1], pixels[keep, 0]] = np.rint(camera[keep, 2]*1000).astype(np.uint16)
    return depth, dict(input_lidar_points=len(points), valid_projected_points=len(indices),
        occupied_pixels=len(keep), invalid_depth_zero=True, rounding='floor pixels / nearest mm depth',
        P2_translation_absorbed_in_camera2=True)


def tracking_labels(text, sequence, poses_world_imu, calib):
    result = []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 17:
            raise ValueError('tracking训练标签必须17列')
        frame, track = int(fields[0]), int(fields[1])
        entry = labels(' '.join(fields[2:]))[0]
        world = poses_world_imu[frame]@calib['imu_from_camera0']
        center = np.asarray(entry['center_camera0'])
        center_world = world[:3, :3]@center+world[:3, 3]
        center_camera2 = center+calib['camera2_shift']
        entry.update(frame_idx=frame, track_id=track, instance_token=f'{sequence}:{track}',
            center=center_world.tolist(), center_camera2=center_camera2.tolist(),
            coordinate_frame='initial_imu_world', geometry_provenance='official_3D_box_and_OXTS',
            evaluation_only=True)
        result.append(entry)
    return result
