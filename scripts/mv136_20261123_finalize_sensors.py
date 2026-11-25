"""六路真实输入的最终核验/证据入口；不运行模型或方法。"""
import json,shutil
from pathlib import Path
from conceptgraph.project_paths import PROJECT_ROOT,experiment_directory
from scripts.multiview_identity_repair_20261123_prepare_nextlogs import digest,dump

def main():
 root=PROJECT_ROOT;p=root/'dataset/processed/mv136_20261123_sixscene_v1';q=root/'dataset/processed/mv136_20261123_sharedscan_v1';r=root/'outputs/reports/multiview_viewcount_20261123/sensors';m=json.loads((p/'manifest.json').read_text());strict=json.loads((q/'manifest.json').read_text());gt=list(map(json.loads,(p/'mv136_20261123_evaluation_ground_truth.jsonl').read_text().splitlines()))
 oldpaths=[root/'dataset/processed'/n/f for n,f in [('multiview_20261123_pilot_v1','multiview_20261123_evaluation_ground_truth.jsonl'),('multiview_identity_repair_20261123_validation_v1','multiview_identity_repair_20261123_evaluation_ground_truth.jsonl'),('multiview_identity_repair_20261123_nextlogs_v1','multiview_identity_repair_20261123_evaluation_ground_truth.jsonl')]];old={(x['scene'],x['frame_idx']):x for path in oldpaths for x in map(json.loads,path.read_text().splitlines())};same=0;F={'CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT'}
 for row in gt:
  rows=[o for o in row['objects'] if set(o['visible_cameras'])&F];before=old[row['scene'],row['frame_idx']]['objects'];assert len(rows)==len(before)
  for a,b in zip(rows,before):
   assert {k:v for k,v in a.items() if k!='visible_cameras'}=={k:v for k,v in b.items() if k!='visible_cameras'}
   assert [c for c in a['visible_cameras'] if c in F]==b['visible_cameras'];same+=1
 choices=0
 for ss,st in zip(m['scenes'],strict['scenes']):
  for a,b in zip(ss['groups'],st['groups']):
   ev=min(a['views'].values(),key=lambda v:v['camera_timestamp_s']);token=ev['lidar_token'];points=set()
   for c,v in b['views'].items():
    original=a['views'][c]
    for k in ['rgb','intrinsic','camera_pose','camera_timestamp_s','detections']:assert v[k]==original[k]
    age=original['camera_timestamp_s']-ev['lidar_timestamp_s'] if token else None;expected=token if token and 0<age<=.1 else None;assert v['lidar_token']==expected
    if expected:assert v['lidar_raw']==ev['lidar_raw'];points.add(expected)
   assert len(points)<=1;choices+=1
 out=dict(passed=True,front3_GT_exact_old_objects=same,GT_groups=120,front_GT=sum('CAM_FRONT' in o['visible_cameras'] for row in gt for o in row['objects']),front3_GT=same,union6_GT=sum(len(row['objects']) for row in gt),sharedscan_groups_verified=choices,sharedscan_image_K_pose_detection_unchanged_views=720,main_manifest_sha256=digest(p/'manifest.json'),sharedscan_manifest_sha256=digest(q/'manifest.json'),note='original GT fields and front3 visibility exact; rear only adds visibility or extra union objects. Earliest camera past scan independently verified against official metadata in main source audit.')
 assert (same,out['front_GT'],out['union6_GT'])==(687,385,1181);dump(r/'GT_and_sharedscan_independent_audit.json',out)
 audits={n:json.loads((r/name).read_text()) for n,name in [('projection','independent_input_audit.json'),('official','official_source_audit.json'),('sharedscan_projection','sharedscan/independent_input_audit.json'),('cost','sensor_cost_and_scan_summary.json')]};assert all(x['passed'] for x in audits.values())
 oldreg=json.loads((root/'outputs/reports/multiview_identity_repair_20261123/validation/registration.json').read_text());protected={path:digest(path)==sha for path,sha in oldreg['protected_sha256'].items()};assert all(protected.values())
 run=experiment_directory('nuscenes','validation','mv136_20261123_sensors_delivery_v1',create=True);sources={}
 for path in sorted((root/'scripts').glob('mv136_20261123_*.py')):
  target=run/'source_snapshot'/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);sources[str(path)]=dict(sha256=digest(path),snapshot=str(target))
 scoped=[root/'dataset/raw/mv136_20261123_sixscene_v1',p,q,Path(m['frontend_run']),r,run];files={f.resolve() for base in scoped for f in base.rglob('*') if f.is_file()};size=sum(f.stat().st_size for f in files);free=shutil.disk_usage(root).free
 verification=dict(passed=True,source_snapshots=sources,protected_unchanged=protected,protected_count=len(protected),old_manifest_unchanged=audits['cost']['original_manifest_unchanged'],manifest_sha256={str(p/'manifest.json'):digest(p/'manifest.json'),str(q/'manifest.json'):digest(q/'manifest.json')},GT_sha256=digest(p/'mv136_20261123_evaluation_ground_truth.jsonl'),scoped_new_bytes=size,scoped_new_MiB=size/2**20,budget_new_MiB=640,free_GiB=free/2**30,new_real_frontend_calls=360,reused_frontend_views=360,sharedscan_reused_frontend_views=720,new_frontend_GPU_s=audits['cost']['new_rear_frontend_complete']['gpu_wall_s'],model_training_calls=0,new_test_opened=False,actual_new_raw_extracted_bytes=json.loads((p/'inputs_complete.json').read_text())['extracted_bytes'])
 assert size<640*2**20 and free>20*2**30;dump(r/'delivery_verification.json',verification)
 lines=['# 1/3/6视角真实输入与独立传感器核验','','固定原六个已用验证/回归scene、各前20时刻，共120时刻/5log；未打开新测试、未训练。共同评价时钟为六路最新实际曝光，各subset实际ready单列；未把晚时钟用于给早相机拿未来LiDAR。','','主输入含720图：原前3缓存360图只读复用，仅新增后3真实推理360图。YOLO-World960/conf0.1、MobileSAM、CLIP ViT-H-14及全部权重SHA、seed17与旧前端相同；新GPU47.86秒，含加载14.21秒、逐图模型30.84秒。复用成本并非零，各subset全量历史逐图成本见subset_model_cost.csv；与mapperCPU及数据加载分列，不能宣称车辆实时闭环。','','六路GT1181对象帧，front385/front3687/rear-only494；旧front3的687对象及全部原标注字段精确一致。类别/visibility/3m协议不变。官方GT在sample时刻，相对共同clock约−1.674到3.124ms，相对相机曝光最大45.703ms；只做ego补偿，未对动态对象补偿。','','主对照按每camera自身曝光选最近past LiDAR且≤100ms，34路缺深度保留，未来点0。720真实图均解码，独立全部zbuffer重投影0像素差，原点索引和round投影全等；官方K误差0、独立四元数pose最大误差1.14e−13。原前3的19项物理字段/路径/hash全等，只补clock与复用标记。','','23/120组六路比前3多一个scan，6首组前3无深度，其中2组后向有深度。因此主结果估计实际相机配置组合，不把额外时间采样误归纯相机覆盖。另预登记严格同scan控制：同组最早曝光前的最近严格past scan供六路共同重投影，每路仍≤100ms，缺失全保留（36路）。该控制0新图像模型调用，720图/特征/GT/pose完全复用，只有深度与原点投影独立生成；主与控制成绩分表，未看结果调方法。','','证据入口：','']
 for label,path in [('主manifest',p/'manifest.json'),('严格同scan manifest',q/'manifest.json'),('隔离官方GT',p/'mv136_20261123_evaluation_ground_truth.jsonl'),('逐组LiDAR/时钟',r/'subset_scan_clock_by_group.csv'),('各subset模型完整成本',r/'subset_model_cost.csv'),('独立投影',r/'independent_input_audit.json'),('官方源核验',r/'official_source_audit.json'),('同scan独立投影',r/'sharedscan/independent_input_audit.json'),('GT/复用控制核验',r/'GT_and_sharedscan_independent_audit.json'),('源码保护及预算',r/'delivery_verification.json')]:lines.append(f'- [{label}]({path})')
 lines+=['',f'本线传感器/缓存/审计新增{size/2**20:.2f}MiB（上限640MiB），空闲{free/2**30:.2f}GiB，13项注册保护未变。方法结果由冻结runner另存，独立评分另表。']
 (r/'六视角传感器输入交付.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(passed=True,new_MiB=size/2**20,free_GiB=free/2**30,report=str(r/'六视角传感器输入交付.md')),ensure_ascii=False))
if __name__=='__main__':main()
