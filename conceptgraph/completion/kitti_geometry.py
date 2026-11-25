"""KITTI Object 相机0坐标几何；保留 P2 第四列及其深度偏置。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def calibration(text):
    rows = {k: np.fromstring(v, sep=' ') for k, v in
            (line.split(':', 1) for line in text.splitlines() if ':' in line)}
    return {k: rows[k].reshape(shape) for k, shape in
            [('P2', (3, 4)), ('R0_rect', (3, 3)), ('Tr_velo_to_cam', (3, 4))]}


def project_lidar(points, calib, image_wh):
    """每像素保留最近实测点，depth 为 rectified camera0 的 z，未做稠密补全。"""
    xyz = np.asarray(points, dtype=float)[:, :3]
    camera = (xyz @ calib['Tr_velo_to_cam'][:, :3].T
              + calib['Tr_velo_to_cam'][:, 3]) @ calib['R0_rect'].T
    projected = camera @ calib['P2'][:, :3].T + calib['P2'][:, 3]
    valid = (np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)
             & np.isfinite(projected).all(axis=1) & (projected[:, 2] > 0))
    uv = np.zeros((len(xyz), 2), dtype=float)
    uv[valid] = projected[valid, :2] / projected[valid, 2, None]
    # floor 与像素单元约定一致，避免刚超出图像的负坐标被截断到0。
    pixels = np.floor(uv).astype(int)
    w, h = image_wh
    valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
    valid &= (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    order = np.flatnonzero(valid)
    order = order[np.argsort(camera[order, 2], kind='stable')]
    flat = pixels[order, 1] * w + pixels[order, 0]
    _, first = np.unique(flat, return_index=True)
    keep = order[first]
    depth = np.zeros((h, w), dtype=np.float32)
    coordinates = np.zeros((h, w, 3), dtype=np.float32)
    depth[pixels[keep, 1], pixels[keep, 0]] = camera[keep, 2]
    coordinates[pixels[keep, 1], pixels[keep, 0]] = camera[keep]
    return depth, coordinates, dict(lidar_points=len(xyz), projected_points=len(order),
                                   occupied_pixels=len(keep), quantization='floor')


def backproject(uv, z_camera0, projection):
    """准确逆投影，用于标定回归核验；不假设 P2[:,3]==0。"""
    uv = np.atleast_2d(uv).astype(float)
    p = np.asarray(projection, dtype=float)
    inv = np.linalg.inv(p[:, :3])
    rays = np.column_stack([uv, np.ones(len(uv))]) @ inv.T
    offset = inv @ p[:, 3]
    scale = (np.asarray(z_camera0) + offset[2]) / rays[:, 2]
    return rays * np.atleast_1d(scale)[:, None] - offset


def labels(text):
    output = []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 15:
            raise ValueError('KITTI training label 必须含15列')
        x = np.asarray(fields[1:], dtype=float)
        h, w, length = x[7:10]
        center = x[10:13].copy()
        center[1] -= h / 2  # 官方 location 为底面中心。
        output.append(dict(label_index=i, category=fields[0], truncation=float(x[0]),
                           occlusion=int(x[1]), bbox=x[3:7].tolist(),
                           dimensions_hwl=[h, w, length], center_camera0=center.tolist(),
                           bottom_center_camera0=x[10:13].tolist(), yaw=float(x[13])))
    return output


def box_overlap(a, b, *, ioa=False):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    intersection = np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])))
    area_a = float(np.prod(np.maximum(0, a[2:] - a[:2])))
    area_b = float(np.prod(np.maximum(0, b[2:] - b[:2])))
    denominator = area_a if ioa else area_a + area_b - intersection
    return float(intersection / denominator) if denominator > 0 else 0.0


def assignment(predictions, truth, *, metric='iou', gate=.5):
    """类别一致且通过门限的最大基数匹配，再优化 IoU 或距离。"""
    if not predictions or not truth:
        return []
    costs = np.full((len(predictions), len(truth)), 1e6)
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(truth):
            if pred['evaluation_category'] != gt['evaluation_category']:
                continue
            if metric == 'iou':
                value = box_overlap(pred['bbox'], gt['bbox'])
                if value >= gate:
                    costs[i, j] = 1 - value
            elif metric == 'distance':
                value = float(np.linalg.norm(np.asarray(pred['center_camera0']) - gt['center_camera0']))
                if value <= gate:
                    costs[i, j] = value / gate
            else:
                raise ValueError(metric)
    a, b = linear_sum_assignment(costs)
    return [(int(i), int(j)) for i, j in zip(a, b) if costs[i, j] < 1e6]
