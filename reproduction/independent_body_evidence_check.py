"""独立核对固定总体、几何输入、逐区间账目与证据链；不是第三套连续证书算法。"""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,math
from shapely.geometry import Polygon,box
from shapely.affinity import rotate,translate
PROJECT=Path('/root/autodl-tmp/lizichong');OUT=PROJECT/'outputs/project_v2/thesis_completion_v1'
DEST=OUT/'vehicle_execution/independent_body_evidence_check_001';DEST.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
taskpath=OUT/'interactive_simulation/synthetic_tasks_v1.json';release=read(taskpath)
configs={c['config_id']:c for c in release['configs']};tasks={t['task_id']:t for t in release['tasks']}
files={};totals=Counter();maxima=Counter();depth=0
checks=[]
for cohort,run in [('rule','independent_body_rule_full_001'),('fault_cpu','independent_body_fault_cpu_full_001')]:
 root=OUT/'vehicle_execution'/run;r=read(root/'report.json');frozen=read(root/'release.json')
 assert not r['is_smoke'] and r['cases']==r['expected_cases']==(150 if cohort=='rule' else 6)
 assert sha(taskpath)==frozen['task_release_sha256']
 for p,h in frozen['source_sha256'].items():assert sha(PROJECT/p)==h
 cases=[json.loads(x) for x in (root/'cases.jsonl').open()];assert len(cases)==r['cases']
 keys={(c['task'],c['seed'],c['method']) for c in cases};assert len(keys)==len(cases)
 if cohort=='rule':
  expected={(t,seed,'rule_logical_clock') for t in tasks for seed in release['seeds']}
 else:
  parent=read(OUT/'faults/realtime_fault_cpu_smoke_002/release.json')
  expected={(parent['task'],parent['seed'],x['condition']) for x in parent['plan']}
 assert keys==expected
 status=Counter();intervals=0;moving=0;oldarrivals=0
 for index,c in enumerate(cases):
  key=f"{index:04d}_{c['task']}_seed{c['seed']}_{c['method']}";folder=root/key;rawcase=Path(c['raw_case'])
  assert read(folder/'report.json')==c
  for p,h in c['source_sha256'].items():assert sha(p)==h;files[str(p)]=h
  geometry=read(folder/'geometry.json');config=configs[c['config_id']]
  assert geometry['road_polygon_xy']==config['road_polygon_xy']
  assert Polygon(geometry['road_polygon_xy']).is_valid
  for i,target in enumerate(config['targets']):
   # 验证器用GEOS矩形及独立affine API，不复用被审核模块的标量矩形函数。
   reference=translate(rotate(box(-target['length_m']/2,-target['width_m']/2,target['length_m']/2,target['width_m']/2),
     target['yaw_rad'],origin=(0,0),use_radians=True),target['center_xy'][0],target['center_xy'][1])
   actual=Polygon(geometry['obstacles_xy'][i]);assert actual.hausdorff_distance(reference)<1e-12
   assert abs(actual.area-target['length_m']*target['width_m'])<1e-11
   totals['parked_obstacle_polygons_verified']+=1
  if tasks[c['task']]['kind']=='inaccessible':
   barrier=Polygon(geometry['obstacles_xy'][-1]);assert len(geometry['obstacles_xy'])==4
   assert abs(barrier.area-200.)<1e-9
   # 中心线18m位置与几何屏障中心由边长重算，不依赖原setup。
   line=config['centerline_xy'];length=0.;expected_center=None
   for a,b in zip(line,line[1:]):
    d=math.dist(a,b)
    if length+d>=18.:
     fraction=(18.-length)/d;expected_center=[a[k]+fraction*(b[k]-a[k]) for k in (0,1)];break
    length+=d
   assert expected_center is not None and math.dist(expected_center,[barrier.centroid.x,barrier.centroid.y])<1e-12
   totals['barrier_polygons_verified']+=1
  else:assert len(geometry['obstacles_xy'])==3
  audit=read(folder/'interval_audit.json');raw=[json.loads(x) for x in (rawcase/'execution.jsonl').open()]
  first=json.loads(next((rawcase/'perception.jsonl').open()));assert first['sensor']['state']['time_s']==0.
  assert len(audit['intervals'])==len(raw)==audit['audited_intervals']==c['audited_intervals']
  assert [x['index'] for x in audit['intervals']]==list(range(len(raw)))
  enumerated=Counter(x['status'] for x in audit['intervals'])
  for name,value in enumerated.items():assert audit['counts'][name]==value
  assert audit['passed']==(enumerated['certificate_safe']==len(raw) and len(raw)>0)
  assert c['passed']==audit['passed'] and c['status']==audit['status']
  for interval in audit['intervals']:
   for field,value in interval['state_residuals'].items():
    assert 0<=value<=audit['tolerances'][field];maxima[field]=max(maxima[field],value)
   depth=max(depth,interval.get('deepest_subdivision',0))
  assert audit['numerical_padding_m']==2e-6 and audit['maximum_subdivision_depth']==12
  status[c['status']]+=1;intervals+=len(raw);moving+=sum(x['state']['speed_mps']>1e-6 for x in raw);oldarrivals+=c['prior_reported_arrival']
  for p in [folder/'interval_audit.json',folder/'geometry.json',folder/'report.json']:files[str(p)]=sha(p)
 assert dict(status)==r['status_counts'] and intervals==r['audited_intervals'] and moving==r['moving_control_records']
 assert oldarrivals==r['previously_reported_arrivals']==r['previous_arrivals_with_continuous_body_certificate']
 assert r['new_arrivals']==r['new_executions']==r['actual_model_calls']==0
 checks.append(dict(cohort=cohort,cases=len(cases),intervals=intervals,moving_control_records=moving,old_arrivals=oldarrivals,status_counts=dict(status)))
 totals['cases']+=len(cases);totals['intervals']+=intervals;totals['moving_controls']+=moving
 files[str(root/'report.json')]=sha(root/'report.json');files[str(root/'release.json')]=sha(root/'release.json')
result=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),cohorts=checks,
 totals=dict(totals),max_state_residuals=dict(maxima),maximum_observed_subdivision_depth=depth,
 input_and_artifact_SHA_files=len(files),files_manifest_sha256=hashlib.sha256((json.dumps(files,sort_keys=True)+'\n').encode()).hexdigest(),
 independent_scope=['全部50任务×3种子/六故障条件集合不漏不重','目标矩形以独立affine变换及面积/位置核对','静态屏障尺寸与18m位置','332? 实际区间以totals为准，逐条编号/分母与原执行长度','全部控制残差/数值余量/证书状态汇总','原始执行/感知/结果SHA及派生产物SHA'],
 limits=['没有第三次独立重写连续证书或重跑模型；这是输入与账目审计','未独立重评语言身份/区域保持/拓扑方向；不增加到达','双精度GEOS，不是形式化区间算术'],goal_complete=False)
result['independent_scope'][3]='33076个区间逐条编号/分母与原执行长度核对'
assert totals['cases']==156 and totals['intervals']==33076
save(DEST/'files.json',files);save(DEST/'report.json',result)
print(json.dumps(result,ensure_ascii=False))
