"""真实三维离屏渲染和控制改变传感器的依赖试验；不是完整导航验收。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import os
from pathlib import Path
import pickle
import time

os.environ.setdefault('PYOPENGL_PLATFORM','egl')

import numpy as np
from PIL import Image
import pyglet

pyglet.options['shadow_window']=False

import pyrender  # noqa: E402 - configure headless pyglet before importing its viewer
import trimesh  # noqa: E402

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, dump, digest  # noqa: E402
from conceptgraph.navigation.vehicle_geometry import VehicleState, step_vehicle  # noqa: E402


def camera_pose(state,height=1.45):
    c,s=np.cos(state.yaw_rad),np.sin(state.yaw_rad)
    forward=np.array([c,s,0.])
    right=np.array([s,-c,0.])
    down=np.array([0.,0.,-1.])
    cv=np.eye(4)
    cv[:3,:3]=np.column_stack([right,down,forward])
    cv[:3,3]=np.array([state.x,state.y,height])+.5*forward
    gl=cv.copy()
    gl[:3,1:3]*=-1
    return cv,gl


def car_parts(asset):
    source=trimesh.load(asset,force='scene')
    parts=[]
    rotation=np.array([[0,0,1],[1,0,0],[0,1,0]])
    for name in ['ToyCar','Glass']:
        transform,geometry=source.graph[name]
        mesh=source.geometry[geometry].copy()
        mesh.apply_transform(transform)
        mesh.vertices=np.asarray(mesh.vertices)@rotation.T
        parts.append(mesh)
    all_vertices=np.vstack([m.vertices for m in parts])
    lo,hi=all_vertices.min(axis=0),all_vertices.max(axis=0)
    scale=4.6/(hi[0]-lo[0])
    shift=np.array([(hi[0]+lo[0])/2,(hi[1]+lo[1])/2,lo[2]])
    for mesh in parts:
        mesh.vertices=(np.asarray(mesh.vertices)-shift)*scale
    return parts


def scene_for(asset):
    scene=pyrender.Scene(bg_color=[.55,.75,.95,1],ambient_light=[.65,.65,.65])
    def add_box(extents,position,color):
        mesh=trimesh.creation.box(extents=extents)
        mesh.visual.vertex_colors=color
        pose=np.eye(4)
        pose[:3,3]=position
        scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=False),pose=pose)
    add_box([100,100,.1],[30,0,-.16],[88,118,72,255])
    add_box([65,3.6,.1],[25,0,-.05],[65,65,70,255])
    for x in range(-6,55,4):
        for y in [-1.75,1.75]:
            add_box([2,.08,.015],[x,y,.01],[235,235,225,255])
    for index,(x,y) in enumerate([(14,3.2),(26,3.2)]):
        for mesh in car_parts(asset):
            pose=np.eye(4)
            pose[:3,3]=[x,y,0]
            scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=True),pose=pose,name=f'parked-car-{index}')
    for x in [8,20,32,44]:
        add_box([.3,.3,3],[x,7,1.5],[100,70,40,255])
        add_box([3,3,3],[x,7,4],[40,120,45,255])
    light_pose=np.eye(4)
    light_pose[:3,:3]=trimesh.transformations.euler_matrix(-.5,.2,.2)[:3,:3]
    scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2),pose=light_pose)
    return scene


def sparse_lidar(dense_depth,intrinsic,lidar_pose,camera_cv):
    """16条俯仰束的真实表面交点，转换至相机后用最近深度z-buffer投影。"""
    height,width=dense_depth.shape
    points=[]
    fx,fy,cx,cy=intrinsic[0,0],intrinsic[1,1],intrinsic[0,2],intrinsic[1,2]
    for elevation in np.linspace(-25,10,16):
        row=round(cy-fy*np.tan(np.radians(elevation)))
        if not 0<=row<height:
            continue
        for azimuth in np.arange(-44,44,.4):
            col=round(cx+fx*np.tan(np.radians(azimuth)))
            if 0<=col<width:
                z=float(dense_depth[row,col])
                if 1<z<65:
                    points.append([(col-cx)*z/fx,(row-cy)*z/fy,z])
    image=np.full(height*width,np.inf)
    if points:
        points=np.asarray(points)
        world=points@lidar_pose[:3,:3].T+lidar_pose[:3,3]
        camera=(world-camera_cv[:3,3])@camera_cv[:3,:3]
        uv=camera@intrinsic.T
        u=np.rint(uv[:,0]/uv[:,2]).astype(int)
        v=np.rint(uv[:,1]/uv[:,2]).astype(int)
        valid=(camera[:,2]>1)&(camera[:,2]<65)&(u>=0)&(u<width)&(v>=0)&(v<height)
        np.minimum.at(image,v[valid]*width+u[valid],camera[valid,2])
    image[~np.isfinite(image)]=0
    return np.rint(image.reshape(height,width)*1000).astype('uint16'),len(points)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    dest=OUTPUT/'interactive_simulation'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    try:
        run(dest)
    except Exception as exc:
        dump(dest/'failure.json',dict(error=repr(exc),actual_navigation_complete=False,goal_complete=False))
        raise


def run(dest):
    asset=Path('/root/autodl-fs/third_party/urban_scene_assets/ToyCar_002.glb')
    if digest(asset)!='01a60862de55cd4b9f3acfab0b0def86451800f9c42467fcd61052c16cb9838c':
        raise ValueError('asset fingerprint changed')
    dump(dest/'release.json',dict(asset=str(asset),asset_sha256=digest(asset),source_sha256=digest(Path(__file__)),
        real_frontend=True,gt_objects_sent_to_detector=False,gt_objects_sent_to_mapper=False,
        rendering='3D mesh rasterization with actual camera pose',
        scope='依赖和传感器/前端试验；没有完整导航、异步语义或到达信用'))
    scene=scene_for(asset)
    intrinsic=np.array([[320.,0,320],[0,320.,180],[0,0,1]])
    camera=pyrender.IntrinsicsCamera(320,320,320,180,znear=.1,zfar=100)
    node=scene.add(camera,pose=np.eye(4))
    renderer=pyrender.OffscreenRenderer(640,360)
    states=[VehicleState(0,0,0)]
    for _ in range(50):
        states.append(step_vehicle(states[-1],1.,0.))
    scene_root=dest/'inputs'/'synthetic-probe'
    (scene_root/'color').mkdir(parents=True)
    (scene_root/'depth').mkdir()
    poses=[]
    evidence=[]
    started=time.monotonic()
    for index,state in enumerate([states[0],states[25],states[50]]):
        cv,gl=camera_pose(state)
        scene.set_pose(node,pose=gl)
        rgb,_=renderer.render(scene)
        lidar_cv,lidar_gl=camera_pose(state,height=1.8)
        scene.set_pose(node,pose=lidar_gl)
        _,lidar_depth=renderer.render(scene)
        depth,count=sparse_lidar(lidar_depth,intrinsic,lidar_cv,cv)
        rgbpath=scene_root/'color'/f'{index:06d}.jpg'
        depthpath=scene_root/'depth'/f'{index:06d}.png'
        Image.fromarray(rgb).save(rgbpath,quality=95)
        Image.fromarray(depth).save(depthpath)
        poses.append(cv)
        evidence.append(dict(frame_idx=index,state=asdict(state),rgb_sha256=digest(rgbpath),
            depth_sha256=digest(depthpath),lidar_surface_points=count,projected_lidar_pixels=int((depth>0).sum()),
            timestamp_s=state.time_s,camera_pose_cv=cv.tolist(),lidar_pose_cv=lidar_cv.tolist()))
    renderer.delete()
    if len({r['rgb_sha256'] for r in evidence})!=3 or len({r['depth_sha256'] for r in evidence})!=3:
        raise AssertionError('control did not change camera and LiDAR observations')
    np.savetxt(scene_root/'poses.txt',np.asarray(poses).reshape(-1,16))
    np.savetxt(scene_root/'intrinsics.txt',intrinsic)
    dump(scene_root/'frame_alignment.json',[dict(frame_idx=r['frame_idx'],timestamp=int(r['timestamp_s']*1e6),
        camera_timestamp=int(r['timestamp_s']*1e6)) for r in evidence])
    dump(dest/'sensor_evidence.json',dict(frames=evidence,physics_states=[asdict(s) for s in states],
        actual_controls_applied=True,actual_rgb_changed=True,actual_lidar_changed=True,
        renderer_duration_s=time.monotonic()-started))
    config_path=dest/'sensor_config.yaml'
    config_path.write_text("dataset_name: nuscenes\ncamera_params:\n  image_height: 360\n  image_width: 640\n  fx: 320.0\n  fy: 320.0\n  cx: 320.0\n  cy: 180.0\n  png_depth_scale: 1000.0\n  crop_edge: 0\n")
    from hydra import compose, initialize_config_dir
    from conceptgraph.scripts.streamlined_detections import run_detections, load_detection_models
    with initialize_config_dir(version_base=None,config_dir=str(PROJECT/'conceptgraph/hydra_configs')):
        cfg=compose(config_name='streamlined_detections',overrides=[
            f'dataset_root={dest/"inputs"}','scene_id=synthetic-probe',f'dataset_config={config_path}',
            f'classes_file={PROJECT/"conceptgraph/vehicle_classes.txt"}',
            'exp_suffix=streamlined_yolo_outdoor_objects_v1','save_video=false','device=cuda:0',
            'desired_width=640','desired_height=360'])
    models=load_detection_models(cfg)
    run_detections(cfg,models)
    detections=[]
    for path in sorted(scene_root.glob('exp_*/detections/*.pkl.gz')):
        with gzip.open(path,'rb') as f:
            gobs=pickle.load(f)
        detections.append(dict(path=str(path),sha256=digest(path),
            classes=[gobs['classes'][int(i)] for i in gobs['class_id']],
            boxes=gobs['xyxy'].tolist(),mask_count=len(gobs['mask']),
            actual_clip_crop_count=len(gobs['image_crops'])))
    dump(dest/'report.json',dict(real_3d_rendering=True,real_vehicle_control_updates=True,
        actions_changed_rgb_and_projected_lidar=True,real_yolo_sam_clip=True,detections=detections,
        actual_navigation_complete=False,actual_full_perception_arrivals=0,
        limitations=['仅一配置、三帧的传感器与真实前端依赖试验',
            '低复杂度合成环境不能代表真实道路、真实物理轮胎或天气域泛化',
            '没有完整语义/规划/控制闭环任务，没有最低功能到达信用'],goal_complete=False))


if __name__=='__main__':
    main()
