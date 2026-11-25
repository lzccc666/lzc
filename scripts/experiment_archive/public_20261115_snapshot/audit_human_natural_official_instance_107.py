"""对一条人工自然指令做独立官方实例投影核验；只做事后审计，不把GT送入推理。"""
from __future__ import annotations
import hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
OUT=R/'annotation/offline_imports/human_natural_lzc_20261116_001/official_instance_audit_107';OUT.mkdir(exist_ok=True)
DATA=Path('/root/autodl-tmp/lizichong/dataset/nuscenes_trainval_raw/v1.0-trainval')
def load(name): return json.loads((DATA/(name+'.json')).read_text())
def qmat(q):
 w,x,y,z=q; return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],[2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],[2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]],dtype=float)
def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
record=json.loads((R/'annotation/offline_imports/human_natural_lzc_20261116_001/original_user_export.json').read_text())['records'][0]
sd=next(x for x in load('sample_data') if x['token']==record['camera_token']); cs=next(x for x in load('calibrated_sensor') if x['token']==sd['calibrated_sensor_token']); ep=next(x for x in load('ego_pose') if x['token']==sd['ego_pose_token']); anns=load('sample_annotation'); inst=load('instance'); cats={x['token']:x['name'] for x in load('category')}; itoc={x['token']:x['category_token'] for x in inst}
Rg=qmat(ep['rotation']);tg=np.array(ep['translation']);Rs=qmat(cs['rotation']);ts=np.array(cs['translation']);K=np.array([[1252.8131021185304,0,826.588114781398],[0,1252.8131021185304,469.9846626224581],[0,0,1.]])
click=np.array([record['author_target_point_normalized'][0]*1600,record['author_target_point_normalized'][1]*900])
# Official annotation rows for the exact sample at the camera anchor.
sample_token=sd['sample_token']; rows=[]
for a in anns:
 if a['sample_token']!=sample_token: continue
 cat=cats[itoc[a['instance_token']]]
 if not cat.startswith('vehicle.'): continue
 Ra=qmat(a['rotation']); center=np.array(a['translation']); size=np.array(a['size'])
 corners=[]
 for sx in (-1,1):
  for sy in (-1,1):
   for sz in (-1,1):
    local=np.array([sx*size[0]/2,sy*size[1]/2,sz*size[2]/2]); pg=center+Ra@local
    ps=Rs.T@(Rg.T@(pg-tg)-ts); corners.append(ps)
 corners=np.array(corners); valid=corners[:,2]>0.05
 if not valid.any(): continue
 uv=(corners[valid]@K.T); uv=uv[:,:2]/uv[:,2,None]
 bbox=[float(uv[:,0].min()),float(uv[:,1].min()),float(uv[:,0].max()),float(uv[:,1].max())]
 center_sensor=Rs.T@(Rg.T@(center-tg)-ts); center_uv=None
 if center_sensor[2]>0: center_uv=(K@center_sensor)[:2]/center_sensor[2]
 inside=bool(center_sensor[2]>0.05 and bbox[0]<=click[0]<=bbox[2] and bbox[1]<=click[1]<=bbox[3])
 dist=float(np.linalg.norm(center_uv-click)) if center_uv is not None else None
 rows.append({'annotation_token':a['token'],'category':cat,'visibility_token':a['visibility_token'],'num_lidar_pts':a['num_lidar_pts'],'bbox_2d_xyxy':bbox,'center_uv':center_uv.tolist() if center_uv is not None else None,'center_distance_px':float(dist) if dist is not None else None,'click_inside_projected_box':bool(inside)})
rows.sort(key=lambda r:(not r['click_inside_projected_box'],float('inf') if r['center_distance_px'] is None else r['center_distance_px']))
report={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'goal_complete':False,'audit_only':True,'model_inference_received_gt':False,'scene':record['scene'],'frame_idx':record['frame_idx'],'camera_token':record['camera_token'],'rgb_sha256':record['rgb_sha256'],'official_dataset_root':str(DATA),'official_source_sha256':{f:'sha256_unavailable_large_json' for f in ['sample_data.json','calibrated_sensor.json','ego_pose.json','sample_annotation.json','instance.json','category.json']},'click_px':click.tolist(),'image_size':[1600,900],'official_vehicle_annotations_in_exact_sample':len(rows),'projected_candidates':rows,'independent_label_status':'unresolved_no_conservative_car_match' if not any(r['click_inside_projected_box'] and r['category']=='vehicle.car' for r in rows) else 'candidate_requires_human_review','reason':'点击位置未落入官方vehicle.car的保守三维投影框；图像中远处目标过小且作者在提交前查看过后续帧，因此不自动赋予实例/颜色/运动/可达性标签。','future_observations_excluded_from_inference':True}
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'vehicle_annotations':len(rows),'car_box_match':any(r['click_inside_projected_box'] and r['category']=='vehicle.car' for r in rows),'status':report['independent_label_status']},ensure_ascii=False))
