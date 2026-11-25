"""只读汇总图像模型实际调用/复用成本与异步LiDAR扫描混杂。"""
import json,statistics
from collections import Counter
from pathlib import Path
from conceptgraph.project_paths import PROJECT_ROOT
from scripts.multiview_identity_repair_20261123_prepare_nextlogs import digest,dump
from scripts.multiview_identity_repair_20261123_independent_scores import write_csv
F=['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT'];B=['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
def main():
 root=PROJECT_ROOT;folder=root/'dataset/processed/mv136_20261123_sixscene_v1';m=json.loads((folder/'manifest.json').read_text());r=root/'outputs/reports/multiview_viewcount_20261123/sensors'
 oldpaths=list(m['source_manifest_sha256']);old=[json.loads(Path(p).read_text()) for p in oldpaths];oldviews={(s['scene'],g['frame_idx'],c):v for om in old for s in om['scenes'] for g in s['groups'] for c,v in g['views'].items()};count=0
 fields=['rgb','depth','projection_evidence','intrinsic','camera_pose','camera_timestamp_s','lidar_timestamp_s','lidar_age_s','lidar_token','lidar_raw','lidar_world_pose','camera_token','camera_source_filename','lidar_source_filename','width','height','detections']
 for s in m['scenes']:
  for g in s['groups']:
   for c in F:
    v=g['views'][c];before=oldviews[s['scene'],g['frame_idx'],c]
    assert all(v[k]==before[k] for k in fields);count+=1
 assert count==360
 records=[];releases=[]
 for om in old+[m]:
  run=Path(om['frontend_run']);release=json.loads((run/'frontend_release.json').read_text());releases.append(release)
  records.extend(json.loads(x) for x in (run/'frontend_frames.jsonl').read_text().splitlines())
 keys={(r['scene'],r['frame_idx'],r['channel']):r for r in records};assert len(keys)==len(records)==720
 assert all(rel['weights_sha256']==releases[0]['weights_sha256'] for rel in releases)
 assert all(rel['detection_image_size']==960 and rel['detector_confidence']==.1 and rel['seed']==17 for rel in releases)
 modelcost=[]
 for n,channels in [('1',F[:1]),('3',F),('6',F+B)]:
  rr=[v for (s,f,c),v in keys.items() if c in channels];assert len(rr)==120*int(n)
  modelcost.append(dict(views=n,frames=120,required_total_image_calls=len(rr),historical_front_cache_reused=sum(c in F for s,f,c in keys if c in channels),new_rear_image_calls=sum(c in B for s,f,c in keys if c in channels),sum_actual_per_image_model_wall_s=sum(x['model_wall_s'] for x in rr),mean_per_group_sequential_model_wall_s=sum(x['model_wall_s'] for x in rr)/120,cost_scope='sum of recorded actual per-image infer wall times, excludes model load/mapper/io; not measured end-to-end or parallel realtime latency'))
 scans=[]
 for s in m['scenes']:
  for g in s['groups']:
   tokens={n:{g['views'][c]['lidar_token'] for c in cc if g['views'][c]['lidar_token']} for n,cc in [('1',F[:1]),('3',F),('6',F+B)]}
   row=dict(scene=s['scene'],log_token=s['log_token'],frame_idx=g['frame_idx'],extra_scan_tokens_vs_front3=sorted(tokens['6']-tokens['3']),front3_all_depth_missing=not bool(tokens['3']),rear_available_when_front3_missing=not bool(tokens['3']) and bool(tokens['6']))
   for n,cc in [('1',F[:1]),('3',F),('6',F+B)]:
    ts=[g['views'][c]['lidar_timestamp_s'] for c in cc if g['views'][c]['lidar_timestamp_s'] is not None];row.update({f'view{n}_unique_scans':len(tokens[n]),f'view{n}_scan_tokens':sorted(tokens[n]),f'view{n}_missing_views':sum(g['views'][c]['lidar_token'] is None for c in cc),f'view{n}_earliest_scan_s':min(ts) if ts else None,f'view{n}_latest_scan_s':max(ts) if ts else None,f'view{n}_ready_s':g['subset_ready_timestamp_s'][n],f'view{n}_common_wait_s':g['reference_timestamp_s']-g['subset_ready_timestamp_s'][n]})
   scans.append(row)
 summary=dict(passed=True,manifest_sha256=digest(folder/'manifest.json'),old_front3_physical_fields_equal_views=count,original_manifest_unchanged={p:digest(p)==sha for p,sha in m['source_manifest_sha256'].items()},models_same_frozen_weights=True,models_weights_sha256=releases[0]['weights_sha256'],image_calls=modelcost,new_rear_frontend_complete=json.loads((Path(m['frontend_run'])/'frontend_complete.json').read_text()),groups=120,groups_with_extra_scan_vs_front3=sum(bool(x['extra_scan_tokens_vs_front3']) for x in scans),front3_all_depth_missing_groups=sum(x['front3_all_depth_missing'] for x in scans),groups_rear_depth_when_front3_missing=sum(x['rear_available_when_front3_missing'] for x in scans),max_common_wait_after_front_s=max(x['view1_common_wait_s'] for x in scans),max_common_wait_after_front3_s=max(x['view3_common_wait_s'] for x in scans),interpretation='main comparison estimates combined practical camera setup with own causal LiDAR and asynchronous capture; not pure camera-count coverage effect. Separate sharedscan control holds raw scan fixed across cameras; no new image model calls.',GT_timestamp_limit='official sample GT, exposure poses ego compensated only; no object motion compensation',source_sha256=digest(__file__))
 assert all(summary['original_manifest_unchanged'].values());write_csv(r/'subset_model_cost.csv',modelcost);write_csv(r/'subset_scan_clock_by_group.csv',scans);dump(r/'sensor_cost_and_scan_summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ['models_weights_sha256','new_rear_frontend_complete','original_manifest_unchanged','image_calls']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
