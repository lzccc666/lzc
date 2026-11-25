"""固定六scene×20时刻，只补后三路真实输入；前3路物理缓存只读复用。"""
import copy,json,shutil,tarfile,time
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from conceptgraph.nuscenes.expansion_benchmark import iter_records
from conceptgraph.project_paths import PROJECT_ROOT,experiment_directory
from conceptgraph.scripts.preprocess_nuscenes_to_minimal import build_transform,inverse_transform
from scripts.multiview_identity_repair_20261123_prepare_nextlogs import digest,dump,projection
FRONT=('CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT');BACK=('CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT');CHANNELS=FRONT+BACK
NAME='mv136_20261123_sixscene_v1';REPORT=PROJECT_ROOT/'outputs/reports/multiview_viewcount_20261123/sensors'
def main():
 start=time.monotonic();root=PROJECT_ROOT;REPORT.mkdir(parents=True,exist_ok=True)
 raw=root/'dataset/raw'/NAME;processed=root/'dataset/processed'/NAME;raw.mkdir(parents=True,exist_ok=False);processed.mkdir(parents=True,exist_ok=False)
 run=experiment_directory('nuscenes','validation','mv136_20261123_frontend_v1',create=True)
 sources=[root/'dataset/processed'/n/'manifest.json' for n in ['multiview_20261123_pilot_v1','multiview_identity_repair_20261123_validation_v1','multiview_identity_repair_20261123_nextlogs_v1']]
 old=[json.loads(p.read_text()) for p in sources];scenes=copy.deepcopy([s for m in old for s in m['scenes']]);assert [s['scene'] for s in scenes]==['scene-0012','scene-0013','scene-0019','scene-0035','scene-0042','scene-0047']
 reg=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),role='seen original development/debug scenes, regression; no test',fixed_scenes=[dict(scene=s['scene'],log_token=s['log_token'],frames=20) for s in scenes],input_sources={str(p):digest(p) for p in sources},channels=list(CHANNELS),new_views=360,reused_views=360,seed=17,training=False,method='frozen T3v2; no new rules or threshold changes',budget=dict(deadline_utc='2026-11-23T15:24:00Z',wall_s=2700,GPU_s=300,new_bytes=640*2**20,min_free_bytes=20*2**30),common_time='max six actual camera exposure timestamps; same for 1/3/6; subset readiness recorded; camera-specific latest past LiDAR <= own exposure and <=100ms; no later points via common clock',source_sha256=digest(__file__),free_before_bytes=shutil.disk_usage(root).free)
 dump(REPORT/'registration.json',reg)
 meta=root/'dataset/raw/nuscenes_trainval/v1.0-trainval';scene_names={s['scene_token']:s['scene'] for s in scenes};wanted={g['sample_token'] for s in scenes for g in s['groups']}
 samples={r['token']:r for r in iter_records(meta/'sample.json') if r['scene_token'] in scene_names};cameras={};lidars={s['scene']:[] for s in scenes}
 for r in iter_records(meta/'sample_data.json'):
  sample=samples.get(r['sample_token'])
  if sample is None:continue
  channel=r['filename'].split('/')[1]
  if channel=='LIDAR_TOP':lidars[scene_names[sample['scene_token']]].append(r)
  elif r['sample_token'] in wanted and r['is_key_frame'] and channel in CHANNELS:cameras[r['sample_token'],channel]=r
 plans=[]
 for s in scenes:
  assert len(s['groups'])==20
  for g in s['groups']:
   for c in CHANNELS:
    cam=cameras.get((g['sample_token'],c));assert cam is not None,('missing actual camera',s['scene'],g['frame_idx'],c)
    past=[r for r in lidars[s['scene']] if 0<=cam['timestamp']-r['timestamp']<=100000];li=max(past,key=lambda r:r['timestamp']) if past else None
    plans.append(dict(scene=s['scene'],frame_idx=g['frame_idx'],sample=samples[g['sample_token']],channel=c,camera=cam,lidar=li,reused=c in FRONT))
 poseids={p[k]['ego_pose_token'] for p in plans for k in ('camera','lidar') if p[k]};poses={r['token']:r for r in iter_records(meta/'ego_pose.json') if r['token'] in poseids};calibs={r['token']:r for r in iter_records(meta/'calibrated_sensor.json')}
 existing={v['camera_source_filename']:Path(v['rgb']) for s in scenes for g in s['groups'] for v in g['views'].values()};existing.update({v['lidar_source_filename']:Path(v['lidar_raw']) for s in scenes for g in s['groups'] for v in g['views'].values() if v['lidar_raw']})
 missing={};rawsources={}
 for p in plans:
  if p['reused']:continue
  for key in ('camera','lidar'):
   sensor=p[key]
   if not sensor or sensor['filename'] in rawsources:continue
   fn=sensor['filename'];candidates=[existing.get(fn),root/'dataset/raw/nuscenes_trainval'/fn,root/'outputs/project_v2/thesis_completion_v1/data_manifest/development_causal_inputs_full_001/raw_past_lidar'/f'{sensor["token"]}.bin' if key=='lidar' else None]
   path=next((x for x in candidates if x and x.is_file()),raw/fn);rawsources[fn]=path
   if not path.is_file():missing[fn]=path
 for s in scenes:
  s['data_role']='seen_development_regression_fixed_viewcount'
  for g in s['groups']:
   oldref=g['reference_timestamp_s'];relevant=[p for p in plans if p['scene']==s['scene'] and p['frame_idx']==g['frame_idx']];latest=max(p['camera']['timestamp']/1e6 for p in relevant);g['original_front3_reference_timestamp_s']=oldref;g['reference_timestamp_s']=latest;g['common_reference_timestamp_s']=latest;g['subset_ready_timestamp_s']={n:max(p['camera']['timestamp']/1e6 for p in relevant if p['channel'] in cc) for n,cc in [('1',FRONT[:1]),('3',FRONT),('6',CHANNELS)]};g['additional_wait_from_front3_s']=latest-oldref
   for p in relevant:
    c,cam,li=p['channel'],p['camera'],p['lidar']
    if p['reused']:
     v=g['views'][c];assert v['camera_token']==cam['token'];assert v['lidar_token']==(li['token'] if li else None);v['original_synchronization_wait_s']=v['synchronization_wait_s'];v['reused_frontend']=True;v['synchronization_wait_s']=latest-v['camera_timestamp_s'];continue
    folder=processed/s['scene']/f'{g["frame_idx"]:06d}'/c;folder.mkdir(parents=True)
    cc=calibs[cam['calibrated_sensor_token']];cw=build_transform(poses[cam['ego_pose_token']]['translation'],poses[cam['ego_pose_token']]['rotation'])@build_transform(cc['translation'],cc['rotation']);lw=None
    if li:
     lp,lc=poses[li['ego_pose_token']],calibs[li['calibrated_sensor_token']];lw=build_transform(lp['translation'],lp['rotation'])@build_transform(lc['translation'],lc['rotation'])
    g['views'][c]=dict(rgb=str(rawsources[cam['filename']]),depth=str(folder/'depth.png'),depth_scale=1000,projection_evidence=str(folder/'projection_evidence.npz'),intrinsic=cc['camera_intrinsic'],camera_pose=cw.tolist(),camera_timestamp_s=cam['timestamp']/1e6,lidar_timestamp_s=li['timestamp']/1e6 if li else None,lidar_age_s=(cam['timestamp']-li['timestamp'])/1e6 if li else None,lidar_token=li['token'] if li else None,lidar_raw=str(rawsources[li['filename']]) if li else None,lidar_world_pose=lw.tolist() if lw is not None else None,camera_token=cam['token'],camera_source_filename=cam['filename'],lidar_source_filename=li['filename'] if li else None,width=cam['width'],height=cam['height'],synchronization_wait_s=latest-cam['timestamp']/1e6,detections=str(run/'detections'/s['scene']/f'{g["frame_idx"]:06d}'/f'{c}.pkl.gz'),reused_frontend=False)
 archive=Path('/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval01_blobs.tgz')
 manifest=dict(schema_version='mv136-20261123-v1',selection_rule='fixed previous 6 scenes and first20 sample tokens; independent of method scores',channels=list(CHANNELS),scenes=scenes,frontend_run=str(run),source_sha256=digest(__file__),source_manifest_sha256=reg['input_sources'],inference_time_rule=reg['common_time'],depth_rule='latest same-scene past LiDAR <= own camera <=100ms, missing retains empty depth; ego compensated, no object GT motion compensation',evaluation_ground_truth_separate=True,gt_in_inference=False,new_test_opened=False,budget=reg['budget'],shared_archive=dict(path=str(archive),bytes=archive.stat().st_size,mtime_ns=archive.stat().st_mtime_ns))
 dump(processed/'manifest.json',manifest);dump(processed/'first_group_example.json',scenes[0]['groups'][0]);dump(processed/'raw_sensor_plan.json',dict(plans=plans,needed_archive_members=sorted(missing)))
 print(json.dumps(dict(manifest=str(processed/'manifest.json'),new_archive_members=len(missing),scenes=6,groups=120,new_views=360,reused_views=360)),flush=True)
 prov=[]
 if missing:
  with tarfile.open(archive,'r|gz') as tf:
   for member in tf:
    if time.monotonic()-start>900:raise TimeoutError('bounded archive extraction exceeded 15min; retain partial')
    fn=member.name.removeprefix('./')
    if fn not in missing:continue
    if shutil.disk_usage(root).free<20*2**30:raise OSError('disk threshold')
    assert member.isfile() and member.size<20_000_000
    dest=missing.pop(fn);dest.parent.mkdir(parents=True,exist_ok=True);data=tf.extractfile(member).read()
    with dest.open('xb') as f:f.write(data)
    prov.append(dict(member=fn,path=str(dest),sha256=digest(dest),bytes=len(data),archive=str(archive)))
    if len(prov)%60==0:print(json.dumps(dict(extracted=len(prov),remaining=len(missing),wall_s=time.monotonic()-start)),flush=True)
    if not missing:break
 dump(processed/'extraction_provenance.json',prov)
 if missing:dump(processed/'missing_members.json',sorted(missing));raise FileNotFoundError('archive missing input, do not replace scenes')
 checks=[]
 for s in scenes:
  for g in s['groups']:
   for c,v in g['views'].items():
    if not v['reused_frontend']:
     if v['lidar_raw']:
      pts=np.fromfile(v['lidar_raw'],np.float32).reshape(-1,5);depth,evidence=projection(pts,inverse_transform(np.asarray(v['camera_pose']))@np.asarray(v['lidar_world_pose']),v['intrinsic'],v['height'],v['width'])
     else:depth=np.zeros((v['height'],v['width']),np.uint16);evidence=dict(pixel_y=np.empty(0,np.int32),pixel_x=np.empty(0,np.int32),point_index=np.empty(0,np.int32),depth_m=np.empty(0,np.float32))
     imageio.imwrite(v['depth'],depth);np.savez_compressed(v['projection_evidence'],**evidence)
    checks.append(dict(scene=s['scene'],frame_idx=g['frame_idx'],channel=c,reused=v['reused_frontend'],rgb_sha256=digest(v['rgb']),depth_sha256=digest(v['depth']),projection_sha256=digest(v['projection_evidence']),detections_sha256=digest(v['detections']) if v['reused_frontend'] else None,lidar_sha256=digest(v['lidar_raw']) if v['lidar_raw'] else None,lidar_age_s=v['lidar_age_s']))
 dump(processed/'input_verification.json',checks);complete=dict(manifest_sha256=digest(processed/'manifest.json'),groups=120,views=720,new_views=360,reused_views=360,missing_past_lidar=sum(v['lidar_age_s'] is None for s in scenes for g in s['groups'] for v in g['views'].values()),future_lidar_views=0,elapsed_s=time.monotonic()-start,extracted_bytes=sum(p['bytes'] for p in prov),frontend_executed=False)
 dump(processed/'inputs_complete.json',complete);dump(REPORT/'input_preparation.json',complete);print(json.dumps(complete),flush=True)
if __name__=='__main__':main()
