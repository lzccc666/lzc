"""可交互三维传感器环境；对象真值只在环境和独立评价端使用。"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh
from conceptgraph.completion.render_probe import camera_pose, car_parts, sparse_lidar
import pyrender
from conceptgraph.completion.stage9 import digest, dump
from conceptgraph.navigation.vehicle_geometry import step_vehicle, VehicleSpec

ASSET = Path('/root/autodl-fs/third_party/urban_scene_assets/ToyCar_002.glb')
ASSET_SHA = '01a60862de55cd4b9f3acfab0b0def86451800f9c42467fcd61052c16cb9838c'
COLORS = {'red': (.85,.025,.015,1), 'blue': (.015,.06,.85,1), 'white': (.92,.92,.92,1)}


class InteractiveRoadEnvironment:
    """控制经自行车模型积分，RGB和16束LiDAR由新位姿重新渲染。"""

    intrinsic = np.array([[320.,0,320],[0,320.,180],[0,0,1]])

    def __init__(self, config, task, seed, initial_state):
        if digest(ASSET) != ASSET_SHA:
            raise ValueError('third-party asset changed')
        self.state = initial_state
        self.scene = pyrender.Scene(bg_color=[.55,.75,.95,1],ambient_light=[.65,.65,.65])
        self.scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=1.8))
        self._add_box([160,160,.1],[20,0,-.18],[88,118,72,255])
        points = np.asarray(config['centerline_xy'])
        tangent = np.gradient(points,axis=0)
        tangent /= np.linalg.norm(tangent,axis=1)[:,None]
        normal = np.column_stack([-tangent[:,1],tangent[:,0]])
        vertices = np.vstack([np.column_stack([points-1.8*normal,np.zeros(len(points))]),
                              np.column_stack([points+1.8*normal,np.zeros(len(points))])])
        n = len(points)
        faces = [[i,i+1,n+i] for i in range(n-1)] + [[i+1,n+i+1,n+i] for i in range(n-1)]
        mesh = trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
        mesh.visual.vertex_colors = [65,65,70,255]
        self.scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=False))
        parts = car_parts(ASSET)
        for target in config['targets']:
            pose = trimesh.transformations.rotation_matrix(target['yaw_rad'],[0,0,1])
            pose[:3,3] = [*target['center_xy'],.015]
            for j,part in enumerate(parts):
                material = (pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=COLORS[target['color']],metallicFactor=.15,roughnessFactor=.6)
                    if j == 0 else None)
                self.scene.add(pyrender.Mesh.from_trimesh(part,smooth=True,material=material),pose=pose)
        if task['obstacle_mode'] == 'full_width_barrier':
            from shapely.geometry import LineString
            from conceptgraph.completion.kinematic_benchmark import pose_on
            xy,yaw = pose_on(LineString(points),18)
            self._add_box([2,100,2.5],[*xy,1.25],[175,90,30,255],yaw)
        for i in range(0,n,40):
            xy = points[i]+8*normal[i]
            self._add_box([.3,.3,3],[*xy,1.5],[100,70,40,255])
            self._add_box([3,3,3],[*xy,4],[40,120,45,255])
        self.camera = self.scene.add(pyrender.IntrinsicsCamera(320,320,320,180,znear=.1,zfar=100))
        self.renderer = pyrender.OffscreenRenderer(640,360)
        self.seed = seed

    def _add_box(self, extents, position, color, yaw=0):
        mesh = trimesh.creation.box(extents=extents)
        mesh.visual.vertex_colors = color
        pose = trimesh.transformations.rotation_matrix(yaw,[0,0,1])
        pose[:3,3] = position
        self.scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=False),pose=pose)

    def act(self, acceleration, steering, spec=VehicleSpec()):
        self.state = step_vehicle(self.state,acceleration,steering,spec)
        return self.state

    def observe(self, dest, index):
        cv,gl = camera_pose(self.state)
        self.scene.set_pose(self.camera,pose=gl)
        rgb,_ = self.renderer.render(self.scene)
        lidar_cv,lidar_gl = camera_pose(self.state,height=1.8)
        self.scene.set_pose(self.camera,pose=lidar_gl)
        _,surface = self.renderer.render(self.scene)
        depth,count = sparse_lidar(surface,self.intrinsic,lidar_cv,cv)
        color = dest/'color'/f'{index:06d}.jpg'
        depthpath = dest/'depth'/f'{index:06d}.png'
        color.parent.mkdir(parents=True,exist_ok=True)
        depthpath.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(rgb).save(color,quality=95)
        Image.fromarray(depth).save(depthpath)
        evidence = dict(frame_idx=index,timestamp_s=self.state.time_s,state=asdict(self.state),
                        rgb=str(color),rgb_sha256=digest(color),depth=str(depthpath),
                        depth_sha256=digest(depthpath),lidar_surface_points=count,
                        lidar_pixels=int((depth>0).sum()),camera_pose_cv=cv.tolist())
        dump(dest/'sensor_evidence'/f'{index:06d}.json',evidence)
        # The agent receives sensors, calibration and measured ego state only.
        return np.asarray(Image.open(color).convert('RGB')),depth.astype(float)/1000,cv,color,evidence

    def close(self):
        self.renderer.delete()


def observed_mask_color(rgb, mask):
    """简单颜色基线，仅读取当前SAM前景像素；不是人工或场景GT标签。"""
    import cv2
    pixels = rgb[np.asarray(mask,dtype=bool)]
    if len(pixels) < 20:
        return 'unknown',{}
    hsv = cv2.cvtColor(pixels.reshape(-1,1,3),cv2.COLOR_RGB2HSV).reshape(-1,3)
    h,s,v = hsv.T
    counts = dict(red=int((((h<12)|(h>165))&(s>90)&(v>45)).sum()),
                  blue=int(((h>90)&(h<135)&(s>90)&(v>45)).sum()),
                  white=int(((s<50)&(v>150)).sum()))
    name = max(counts,key=counts.get)
    return (name if counts[name]/len(pixels) >= .12 else 'unknown'),dict(
        counts=counts,pixels=len(pixels),source='current RGB + actual SAM mask; deterministic HSV baseline')
