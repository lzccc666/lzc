"""官方AB3DMOT固定2020版本+本项目共享RGB/LiDAR前端适配；非作者原始检测器或榜单复现。"""
import argparse,copy,csv,gzip,json,os,pickle,resource,sys,time,traceback
from pathlib import Path
os.environ['NUMBA_DISABLE_JIT']='1'
R=Path(__file__).resolve().parents[1];U=R/'source/third_party/AB3DMOT_d45e83c_numpy_compat';sys.path.insert(0,str(U))
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from AB3DMOT_libs.model import AB3DMOT
from AB3DMOT_libs.kalman_filter import KalmanBoxTracker
from conceptgraph.completion.stage9 import OUTPUT as OLD,PROJECT,dump,digest,predict,available_development
from conceptgraph.completion.fault_robustness import mapping_config
from conceptgraph.completion.quality import object_frame_metrics,identity_metrics
from conceptgraph.completion.temporal_identity_evaluation import prepare_sequence,identity_counts
from conceptgraph.slam.streamlined_mapping import StreamlinedMapper,MappingFrame
METHODS=['shared_current_frame','AB3DMOT_official_adapter','ours_geometry_full']
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--scene');parser.add_argument('--dataset',choices=['all','nuScenes','KITTI'],default='all');args=parser.parse_args();O=R/'method_comparison'/args.run_id;O.mkdir(parents=True,exist_ok=False);cases=[]
 if args.dataset!='KITTI':
  for s in available_development():
   if args.scene and s['scene']!=args.scene:continue
   cases.append(dict(dataset='nuScenes',scene=s['scene'],unit=s['log_token'],path=OLD/'mapping_evaluation/causal_mapping_full_001'/s['scene']/'frames.jsonl'))
 if args.dataset!='nuScenes':
  for p in sorted((R/'statistics/kitti_expanded_full_001').glob('*_frames.jsonl')):
   if args.scene and p.name.split('_')[0]!=args.scene:continue
   cases.append(dict(dataset='KITTI',scene=p.name.split('_')[0],unit=p.name.split('_')[0],path=p))
 if not cases:raise ValueError('no selected input cases')
 cfg=mapping_config();release=dict(upstream_commit='d45e83ceb0ac0bd268fd2b0cb7333992e186d764',upstream='https://github.com/xinshuoweng/AB3DMOT',paper='https://arxiv.org/abs/2008.08063',method_name='AB3DMOT官方跟踪器+本项目共享前端（适配）',cases=[{**c,'path':str(c['path'])} for c in cases],source_inputs={str(c['path']):digest(c['path']) for c in cases},upstream_receipt_sha256=digest(U/'upstream_receipt.json'),compatibility_patch_sha256=digest(U/'compatibility_patch.json'),source_sha256={str(p):digest(p) for p in [Path(__file__),PROJECT/'conceptgraph/slam/streamlined_mapping.py',PROJECT/'conceptgraph/slam/mapping.py',PROJECT/'conceptgraph/slam/utils.py',PROJECT/'conceptgraph/completion/quality.py']},projection_cfg=OmegaConf.to_container(cfg),AB3DMOT=dict(max_age=2,min_hits=3,association_iou_threshold=.01,per_coarse_class=True,info_padding='7 unused zero columns; never interpreted as confidence or measured signal'),box_adapter='world axes AABB→[h=Z,w=Y,l=X,x=X,y=-Z+h/2,z=Y,yaw=0]; inverse exact; yaw not measured and never from GT; extents floor1mm only for numerical validity',common_input='same frozen detections, masks, current/past LiDAR, poses and depth filtering; same framewise preprojection; AB3DMOT consumes boxes, proposal also uses past features/pointcloud by architecture',gating='same coarse class world XY<=3m maximum cardinality',gt_in_inference=False,scope='受控前端/跟踪模块比较；非官方原检测器、原超参搜索或排行榜复现；不提供Qwen/关系/导航到达信用',seed=17,new_GPU_calls=0,bootstrap_seed=17,bootstrap_repetitions=4000,goal_complete=False);dump(O/'release.json',release);reports=[];begun=time.monotonic()
 try:
  for c in cases:
   case=O/c['dataset']/c['scene'];case.mkdir(parents=True);frames=[];trackers={};KalmanBoxTracker.count=0;classes=None;records=list(map(json.loads,c['path'].open()));floor_count=0
   with (case/'frames.jsonl').open('x') as out:
    for f in records:
     import shutil
     if shutil.disk_usage(R).free<20*2**30:raise OSError('free space under20GiB')
     idx=f['frame_idx'];pose=np.asarray(f.get('camera_pose',f.get('pose')));K=np.asarray(f['intrinsic']);image=np.asarray(Image.open(f['rgb']).convert('RGB'));depth=np.asarray(Image.open(f['depth']),float)/1000.
     with gzip.open(f['detections'],'rb') as h:gobs=pickle.load(h)
     if classes is None:classes=gobs['classes']
     if classes!=gobs['classes']:raise ValueError('class vocabulary changed')
     np.random.seed(17+idx);t=time.monotonic();single=StreamlinedMapper(copy.deepcopy(cfg),classes);single.update(MappingFrame(idx,image,depth,K,pose,copy.deepcopy(gobs),f['rgb'],timestamp_s=f['camera_timestamp_s']));raw=predict(single.objects,classes,pose=pose,intrinsic=K)
     for obj in raw:obj['node_id']=f"frame-{idx}:{obj['node_id']}"
     preprocess_s=time.monotonic()-t;del single
     categories=sorted(set(trackers)|{p['category'] for p in raw});t=time.monotonic();pred=[]
     for category in categories:
      tracker=trackers.setdefault(category,AB3DMOT(max_age=2,min_hits=3));detections=[]
      for p in raw:
       if p['category']!=category:continue
       extent=np.asarray(p['extent']);floor_count+=int((extent<.001).any());extent=np.maximum(extent,.001);x,y,z=p['center'];l,w,h=extent;detections.append([h,w,l,x,-z+h/2,y,0.])
      aa=np.array(detections,dtype=float).reshape(-1,7);bb=tracker.update(dict(dets=aa,info=np.zeros((len(aa),7))))
      for v in bb:
       h,w,l,x,down,y,yaw,identity=v[:8];pred.append(dict(node_id=f'ab3dmot-{int(identity)}',category=category,center=[float(x),float(y),float(-down+h/2)],extent=[float(l),float(w),float(h)],motion_state='unknown',num_detections=None,last_seen_frame=idx,source='official_AB3DMOT_2020_core'))
     track_s=time.monotonic()-t;full=f['predictions']['one_to_one_depth_robust_motion'];outputs=dict(shared_current_frame=raw,AB3DMOT_official_adapter=pred,ours_geometry_full=full)
     # official GT enters only after both inference paths above. Other method output is not given to AB3DMOT.
     truth=f['evaluation_only_official'];metrics={m:object_frame_metrics(p,truth) for m,p in outputs.items()};row=dict(scene=c['scene'],frame_idx=idx,camera_timestamp_s=f['camera_timestamp_s'],rgb=f['rgb'],depth=f['depth'],detections=f['detections'],predictions=outputs,evaluation_only_official=truth,metrics=metrics,shared_projection_s=preprocess_s,AB3DMOT_tracking_s=track_s,gt_in_inference=False);out.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');out.flush();frames.append(row)
   for method in METHODS:
    ids=identity_counts(prepare_sequence(frames,method),'all_gated')['counts'];values=dict(TP=sum(f['metrics'][method]['true_positive'] for f in frames),GT=sum(f['metrics'][method]['truth'] for f in frames),PRED=sum(f['metrics'][method]['predictions'] for f in frames),DUP=sum(f['metrics'][method]['duplicate_nodes'] for f in frames),ERROR=sum(x['distance_m'] for f in frames for x in f['metrics'][method]['matches']),IDTP=ids['IDTP'],IDFN=ids['IDFN'],IDFP=ids['IDFP']);reports.append(dict(dataset=c['dataset'],scene=c['scene'],unit=c['unit'],method=method,frames=len(frames),**values))
   dump(case/'report.json',dict(dataset=c['dataset'],scene=c['scene'],frames=len(frames),bbox_floor_events=floor_count,projection_s=sum(f['shared_projection_s'] for f in frames),AB3DMOT_tracking_s=sum(f['AB3DMOT_tracking_s'] for f in frames),records_sha256=digest(case/'frames.jsonl'),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,goal_complete=False));print(json.dumps(dict(dataset=c['dataset'],scene=c['scene'],frames=len(frames))),flush=True)
  dump(O/'summary.json',dict(reports=reports,unique_frames=sum(r['frames'] for r in reports if r['method']==METHODS[0]),wall_s=time.monotonic()-begun,new_GPU_calls=0,GT_in_tracker=False,comparison_scope=release['scope'],goal_complete=False))
 except BaseException as e:dump(O/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed_method_cases=len(reports),goal_complete=False));raise
if __name__=='__main__':main()
