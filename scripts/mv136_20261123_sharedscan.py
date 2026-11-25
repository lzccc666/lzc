"""严格scan控制：同组复用一份早于最早六路曝光的最近LiDAR；仅重投影。"""
import copy,json,time
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from conceptgraph.project_paths import PROJECT_ROOT,experiment_directory
from conceptgraph.scripts.preprocess_nuscenes_to_minimal import inverse_transform
from scripts.multiview_identity_repair_20261123_prepare_nextlogs import digest,dump,projection

def main():
 start=time.monotonic();root=PROJECT_ROOT;source=root/'dataset/processed/mv136_20261123_sixscene_v1/manifest.json';main=json.loads(source.read_text());m=copy.deepcopy(main);p=root/'dataset/processed/mv136_20261123_sharedscan_v1';p.mkdir(parents=True,exist_ok=False);r=root/'outputs/reports/multiview_viewcount_20261123/sensors/sharedscan';r.mkdir(parents=True,exist_ok=False)
 run=experiment_directory('nuscenes','validation','mv136_20261123_sharedscan_inputs_v1',create=True);(run/'source_snapshot').mkdir();import shutil;shutil.copy2(__file__,run/'source_snapshot'/Path(__file__).name)
 dump(r/'registration.json',dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_manifest=str(source),source_manifest_sha256=digest(source),source_sha256=digest(__file__),protocol='for each group latest scan strictly before earliest of six exposures; same raw scan all cameras, each view age<=100ms else depth missing; earliest view absence leaves entire group empty',common_GT_and_images_and_features_reused=True,no_new_frontend_calls=True,no_method_scores_accessed=True,budget_CPU_s=300,new_bytes=128*2**20))
 choices=[]
 for s in m['scenes']:
  for g in s['groups']:
   earliest=min(g['views'].values(),key=lambda v:v['camera_timestamp_s']);li=earliest['lidar_timestamp_s'];assert li is None or li<earliest['camera_timestamp_s']
   choices.append(dict(scene=s['scene'],frame_idx=g['frame_idx'],earliest_camera_timestamp_s=earliest['camera_timestamp_s'],lidar_timestamp_s=li,lidar_token=earliest['lidar_token']))
   for c,v in g['views'].items():
    folder=p/s['scene']/f'{g["frame_idx"]:06d}'/c;folder.mkdir(parents=True);age=v['camera_timestamp_s']-li if li is not None else None;eligible=age is not None and 0<age<=.1
    v['main_protocol_lidar_token']=v['lidar_token'];v['shared_scan_requested_token']=earliest['lidar_token'];v['shared_scan_age_exceeded']=age is not None and age>.1
    for k in ('lidar_timestamp_s','lidar_token','lidar_raw','lidar_world_pose','lidar_source_filename'):v[k]=copy.deepcopy(earliest[k]) if eligible else None
    v['lidar_age_s']=age if eligible else None;v['depth']=str(folder/'depth.png');v['projection_evidence']=str(folder/'projection_evidence.npz');v['reused_frontend']=True
 m['schema_version']='mv136-20261123-sharedscan-v1';m['depth_rule']='same past scan strictly before earliest six exposure, per-view age<=100ms; missing remains empty';m['cache_reuse_only']=True;m['source_main_manifest']=str(source);m['source_main_manifest_sha256']=digest(source);m['source_sha256']=digest(__file__);m['frontend_cache_manifest']=str(source)
 dump(p/'manifest.json',m);dump(p/'shared_scan_choices.json',choices)
 checks=[]
 for s in m['scenes']:
  for g in s['groups']:
   for c,v in g['views'].items():
    if v['lidar_raw']:
     pts=np.fromfile(v['lidar_raw'],np.float32).reshape(-1,5);depth,e=projection(pts,inverse_transform(np.asarray(v['camera_pose']))@np.asarray(v['lidar_world_pose']),v['intrinsic'],v['height'],v['width'])
    else:depth=np.zeros((v['height'],v['width']),np.uint16);e=dict(pixel_y=np.empty(0,np.int32),pixel_x=np.empty(0,np.int32),point_index=np.empty(0,np.int32),depth_m=np.empty(0,np.float32))
    imageio.imwrite(v['depth'],depth);np.savez_compressed(v['projection_evidence'],**e);checks.append(dict(scene=s['scene'],frame_idx=g['frame_idx'],channel=c,reused=True,rgb_sha256=digest(v['rgb']),depth_sha256=digest(v['depth']),projection_sha256=digest(v['projection_evidence']),detections_sha256=digest(v['detections']),lidar_sha256=digest(v['lidar_raw']) if v['lidar_raw'] else None,lidar_age_s=v['lidar_age_s']))
 dump(p/'input_verification.json',checks);out=dict(manifest_sha256=digest(p/'manifest.json'),views=720,groups=120,new_frontend_calls=0,reused_frontend_calls=720,missing_past_lidar=sum(x['lidar_age_s'] is None for x in checks),future_lidar_views=0,elapsed_s=time.monotonic()-start,cache_reuse_manifest_sha256=digest(source),ground_truth_reused=str(source.parent/'mv136_20261123_evaluation_ground_truth.jsonl'))
 dump(p/'inputs_complete.json',out);dump(r/'inputs_complete.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':main()
