"""导出两幅真实 R1 失败诊断图：原RGB/原mask/实测点与官方几何，非论文成图。"""
from __future__ import annotations

import gzip
import json
import pickle
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


BASE = PROJECT / 'outputs/results/nuscenes/validation/iterations'
FINAL = BASE / 'r1_failure_attribution_final_20261123_v1'
STATES = BASE / 'r1_failure_states_20261123_v1'
R1 = BASE / 'recall_refinement_R1_full_20261120_v2'
REPORT = PROJECT / 'outputs/reports/metric_optimization_20261123/r1_failure_attribution/cases_v2'
CASES = [('surface_center_static_truck', 'scene-0054', 24, 3),
         ('crowded_pedestrian_shared_support', 'scene-0019', 31, 9)]


def read(path):
    return list(map(json.loads, gzip.open(path, 'rt')))


def font(size=16):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)


def draw_case(label, scene, idx, ti):
    row = next(r for r in read(FINAL/scene/'gt_attribution.jsonl.gz') if r['frame_idx']==idx and r['truth_index']==ti)
    frame = next(r for r in read(R1/scene/'frames.jsonl.gz') if r['frame_idx']==idx)
    state = next(r for r in read(STATES/scene/'states.jsonl.gz') if r['frame_idx']==idx)
    source_path = OUTPUT/'mapping_evaluation/causal_mapping_full_001'/scene/'frames.jsonl'
    raw = next(r for r in map(json.loads, source_path.open()) if r['frame_idx']==idx)
    truth = raw['evaluation_only_official'][ti]
    with gzip.open(row['source_records']['detections'], 'rb') as stream:
        detections = pickle.load(stream)
    image = Image.open(row['source_records']['rgb']).convert('RGBA')
    overlay = Image.new('RGBA', image.size)
    marks = ImageDraw.Draw(overlay)
    indices = {c['raw_index'] for c in row['candidates']}
    for j in indices:
        mask = Image.fromarray(detections['mask'][j].astype(np.uint8)*65)
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        tint = Image.new('RGBA', image.size, (0, 180, 255, 0))
        tint.putalpha(mask)
        overlay = Image.alpha_composite(overlay, tint)
        marks = ImageDraw.Draw(overlay)
        box = detections['xyxy'][j]
        marks.rectangle(tuple(map(float,box)), outline=(0,220,255,255), width=3)
        marks.text((float(box[0]), max(float(box[1])-22,0)),f'raw det {j}',font=font(18),fill='cyan',stroke_width=1,stroke_fill='black')
    gt = row['gt_projected_box']
    marks.rectangle(tuple(gt), outline='yellow', width=4)
    marks.text((gt[0],gt[1]),f'FN GT {ti}',font=font(22),fill='yellow',stroke_width=2,stroke_fill='black')
    if row.get('competition'):
        other_i = row['competition'][0]['assigned_truth_index']
        other = next(r for r in read(FINAL/scene/'gt_attribution.jsonl.gz') if r['frame_idx']==idx and r['truth_index']==other_i)
        marks.rectangle(tuple(other['gt_projected_box']), outline='lime', width=4)
        marks.text((other['gt_projected_box'][0],other['gt_projected_box'][1]+25),f'matched GT {other_i}',font=font(20),fill='lime',stroke_width=1,stroke_fill='black')
    observations = [o for o in state['observations'] if o['raw_index'] in indices]
    pose, intrinsic = np.asarray(raw['camera_pose']), np.asarray(raw['intrinsic'])
    for observation in observations:
        points = np.asarray(observation['points']).reshape(-1,3)
        camera = (points-pose[:3,3]) @ pose[:3,:3]
        uvz = camera @ intrinsic.T
        for u,v in uvz[:,:2]/uvz[:,2:]:
            marks.ellipse((u-3,v-3,u+3,v+3),fill=(255,90,20,255))
    annotated = Image.alpha_composite(image,overlay).convert('RGB').resize((1120,630))
    canvas = Image.new('RGB',(1720,780),'white')
    canvas.paste(annotated,(0,90))
    draw = ImageDraw.Draw(canvas)
    draw.text((15,12),f'{label} | {scene} frame {idx} | category={row["category"]}',font=font(21),fill='black')
    draw.text((15,42),'Yellow: official FN box | Cyan: original detection/mask | Orange: measured selected LiDAR points',font=font(16),fill='black')
    draw.text((15,735),f'R1 frozen failure; XY range={row["distance_xy_m"]:.1f}m, official motion={row["official_motion"]}; projected boxes are support proxies.',font=font(15),fill='black')
    center = np.asarray(truth['center'])
    scale = 19.
    origin = np.array([1420.,410.])

    def pixel(point):
        delta = (np.asarray(point)[:2]-center[:2])*scale
        return (float(origin[0]+delta[0]),float(origin[1]-delta[1]))

    draw.text((1140,95),'World XY: real geometry (m)',font=font(18),fill='black')
    for off in [-10,-5,0,5,10]:
        x = origin[0]+off*scale
        y = origin[1]-off*scale
        draw.line((x,160,x,650),fill='#eeeeee')
        draw.line((1170,y,1680,y),fill='#eeeeee')
        draw.text((x,655),str(off),font=font(13),fill='gray')
    w,x,y,z = truth['rotation_wxyz']
    rotation=Rotation.from_quat([x,y,z,w]).as_matrix()
    width,length,height=truth['size_wlh']
    corners=np.array([[-length/2,-width/2,0],[length/2,-width/2,0],[length/2,width/2,0],[-length/2,width/2,0]]) @ rotation.T+center
    draw.polygon([pixel(p) for p in corners],outline='#bb9900',width=3)
    gx,gy=pixel(center)
    radius=3*scale
    draw.ellipse((gx-radius,gy-radius,gx+radius,gy+radius),outline='#bbbbbb',width=2)
    draw.ellipse((gx-4,gy-4,gx+4,gy+4),fill='#bb9900')
    for observation in observations:
        for p in observation['points']:
            u,v=pixel(p)
            draw.ellipse((u-2,v-2,u+2,v+2),fill='#ff6611')
        u,v=pixel(observation['center'])
        draw.rectangle((u-5,v-5,u+5,v+5),fill='red')
    for pred in frame['predictions']['R1']:
        if pred['category'] != row['category'] or np.linalg.norm(np.asarray(pred['center'][:2])-center[:2])>12:
            continue
        u,v=pixel(pred['center'])
        draw.ellipse((u-5,v-5,u+5,v+5),outline='green',width=3)
    if row.get('competition'):
        other=raw['evaluation_only_official'][row['competition'][0]['assigned_truth_index']]
        u,v=pixel(other['center'])
        draw.line((u-6,v,u+6,v),fill='blue',width=3)
        draw.line((u,v-6,u,v+6),fill='blue',width=3)
    draw.text((1140,690),'Gold: GT body/center | Grey circle: 3m',font=font(15),fill='black')
    draw.text((1140,712),'Red square: current observation center',font=font(15),fill='black')
    draw.text((1140,734),'Green circle: R1 outputs | Blue +: other GT',font=font(15),fill='black')
    path=REPORT/(label+'.png')
    canvas.save(path)
    dump(REPORT/(label+'.json'),dict(evidence=row, RGB_sha256=digest(row['source_records']['rgb']),
        source_frame=str(source_path), frozen_states=str(STATES/scene/'states.jsonl.gz'),
        output_sha256=digest(path), actual_RGB=True, synthetic_scene=False,
        limitations='原始mask展示；3D投影框/框内点是几何代理，无实例像素真值；图为诊断非新增实验'))
    return str(path)


def main():
    REPORT.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,REPORT/'exporter_source_snapshot.py')
    paths=[draw_case(*case) for case in CASES]
    dump(REPORT/'manifest.json',dict(cases=CASES, paths=paths, source_sha256=digest(__file__),
        selection='静止长车体表面中心偏差、拥挤共享当前检测支持；按投影框面积选清晰失败例，不代表发生率',
        new_inference=False, real_RGB_and_existing_masks_and_points=True))
    print(json.dumps(paths))


if __name__=='__main__':
    main()
