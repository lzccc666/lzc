"""独立连续车身审核的固定开发/故障/完整主矩阵入口。"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,hashlib,json,math,os,shutil,time
from shapely.geometry import Polygon
from conceptgraph.completion.independent_continuous_body import audit_trajectory,rectangle

PROJECT=Path('/root/autodl-tmp/lizichong');OUTPUT=PROJECT/'outputs/project_v2/thesis_completion_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 def reject(x):raise ValueError('非法JSON常量 '+x)
 return json.loads(Path(p).read_text(),parse_constant=reject)
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as h:json.dump(x,h,ensure_ascii=False,indent=2,allow_nan=False);h.write('\n')
def interpolate(points,progress):
 remaining=max(0.,progress)
 for a,b in zip(points,points[1:]):
  length=math.hypot(b[0]-a[0],b[1]-a[1])
  if remaining<=length:return [a[0]+(b[0]-a[0])*remaining/length,a[1]+(b[1]-a[1])*remaining/length]
  remaining-=length
 return points[-1]
def geometry(config,task):
 road=Polygon(config['road_polygon_xy'])
 obstacles=[rectangle(t['center_xy'],t['yaw_rad'],t['length_m']/2,t['length_m']/2,t['width_m']) for t in config['targets']]
 if task['kind']=='inaccessible':
  points=config['centerline_xy'];xy=interpolate(points,18.);a=interpolate(points,17.9);b=interpolate(points,18.1)
  yaw=math.atan2(b[1]-a[1],b[0]-a[0]);obstacles.append(rectangle(xy,yaw,1.,1.,100.))
 return road,obstacles

def plans(cohort):
 if cohort=='rule':
  root=OUTPUT/'interactive_simulation/interactive_rule_full_002';release=read(root/'release.json');summary=read(root/'summary.json')
  assert len(summary['results'])==150 and summary['trials']==150
  rows=[dict(cohort=cohort,task=r['task_id'],config_id=r['config_id'],seed=r['seed'],method='rule_logical_clock',
   raw_case=str(root/r['task_id']/f"seed{r['seed']}"),result=r) for r in summary['results']]
 elif cohort=='fault_cpu':
  root=OUTPUT/'faults/realtime_fault_cpu_smoke_002';release=read(root/'release.json');summary=read(root/'summary.json')
  assert summary['attempted']==summary['expected']==6
  rows=[]
  for recordpath in sorted(root.glob('*_record.json')):
   r=read(recordpath);result=r['result'];assert sha(Path(r['raw_case'])/'result.json')==r['result_sha256']
   rows.append(dict(cohort=cohort,task=result['task_id'],config_id=result['config_id'],seed=result['seed'],
    method=r['condition'],raw_case=r['raw_case'],result=result))
  assert len(rows)==6
 else:
  root=OUTPUT/'interactive_simulation/realtime_three_model_full_001';release=read(root/'release.json');summary=read(root/'summary.json')
  assert sha(root/'trial_records.jsonl')==summary['trial_records_sha256']
  raw=[json.loads(x) for x in (root/'trial_records.jsonl').open()];assert len(raw)==len(release['trials'])==450
  lookup={(r['variant'],r['task'],r['seed']):r for r in raw};assert len(lookup)==450
  assert set(lookup)=={(r['variant'],r['task'],r['seed']) for r in release['trials']}
  rows=[dict(cohort=cohort,task=r['task'],config_id=r['config_id'],seed=r['seed'],method=r['variant'],
   raw_case=r['raw_case'],result=r['result']) for r in raw]
 return rows,root,release

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--cohort',choices=['rule','fault_cpu','main'],required=True)
 parser.add_argument('--run-id',required=True);parser.add_argument('--limit',type=int);args=parser.parse_args()
 started=time.monotonic();assert shutil.disk_usage(OUTPUT).free>6*1024**3
 rows,original,release=plans(args.cohort)
 taskpath=OUTPUT/'interactive_simulation/synthetic_tasks_v1.json';tasks=read(taskpath)
 if 'task_release_sha256' in release:assert sha(taskpath)==release['task_release_sha256']
 if 'task_release_sha256' not in release:
  original_synthetic=read(OUTPUT/'interactive_simulation/realtime_fault_cpu_smoke_002_baseline/release.json')
  assert sha(taskpath)==original_synthetic['task_release_sha256']
 task_lookup={t['task_id']:t for t in tasks['tasks']};configs={c['config_id']:c for c in tasks['configs']}
 root=OUTPUT/'vehicle_execution'/args.run_id;root.mkdir(exist_ok=False)
 frozen={str(p.relative_to(PROJECT)):sha(p) for p in [Path(__file__).resolve(),PROJECT/'conceptgraph/completion/independent_continuous_body.py']}
 save(root/'release.json',dict(cohort=args.cohort,original_release_sha256=sha(original/'release.json'),
  original_summary_sha256=sha(original/'summary.json'),task_release_sha256=sha(taskpath),
  source_sha256=frozen,protocol_sha256=sha(OUTPUT/'task_protocol/independent_continuous_body_001.md'),
  vehicle=tasks['vehicle'],expected_cohort_cases=len(rows),limit=args.limit,budget_wall_s=900,budget_bytes=1024**3,
  CUDA_visible='',shared_CPU=True,independent_test=False,goal_complete=False))
 def note(state):
  with (OUTPUT/'interactive_simulation/realtime_three_model_full_001/resource_events.jsonl').open('a') as h:
   h.write(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(),event=args.run_id,status=state,shared_CPU=True,CUDA_visible=''))+'\n')
 output=[];counts=Counter();note('start')
 try:
  for index,plan in enumerate(rows[:args.limit]):
   assert time.monotonic()-started<900 and shutil.disk_usage(OUTPUT).free>6*1024**3
   assert all(sha(PROJECT/p)==v for p,v in frozen.items())
   key=f"{index:04d}_{plan['task']}_seed{plan['seed']}_{plan['method']}";case=Path(plan['raw_case']);task=task_lookup[plan['task']];config=configs[plan['config_id']]
   input_hashes={};recorded=plan['result'];resultpath=case/'result.json'
   if recorded is not None:
    assert read(resultpath)==recorded;input_hashes[str(resultpath)]=sha(resultpath)
   road,obstacles=geometry(config,task)
   save(root/key/'geometry.json',dict(road_polygon_xy=list(road.exterior.coords),obstacles_xy=[list(o.exterior.coords) for o in obstacles],
    source='冻结合成任务的静态评价几何；独立标量重建，不调用原setup/rectangle'))
   record=dict(**{k:v for k,v in plan.items() if k!='result'},prior_reported_arrival=bool(recorded and recorded['actual_full_perception_arrival']),
    original_execution_passed=recorded['independent_execution_audit']['passed'] if recorded else None)
   execution=case/'execution.jsonl';perception=case/'perception.jsonl'
   if not execution.exists() or not perception.exists() or recorded is None:
    record.update(status='unknown_missing_complete_evidence',passed=False,audited_intervals=0)
   else:
    input_hashes[str(execution)]=sha(execution);input_hashes[str(perception)]=sha(perception)
    raw=[json.loads(x) for x in execution.open()];first=json.loads(next(perception.open()))
    initial=first['sensor']['state']
    if initial['time_s']!=0:
     record.update(status='unknown_missing_initial_state',passed=False,audited_intervals=0,initial_observed_time_s=initial['time_s'])
    else:
     audited=audit_trajectory(initial,raw,tasks['vehicle'],road,obstacles)
     save(root/key/'interval_audit.json',audited)
     assert len(audited['intervals'])==len(raw)==audited['audited_intervals']
     record.update({k:v for k,v in audited.items() if k!='intervals'})
     record['saved_control_records']=len(raw)
     record['moving_control_records']=sum(x['state']['speed_mps']>1e-6 for x in raw)
     if recorded['independent_execution_audit'].get('audited_states') is not None:
      assert recorded['independent_execution_audit']['audited_states']==len(raw)+1
     for field in ['x','y','yaw_rad','speed_mps','steering_rad','time_s']:
      assert raw[-1]['state'][field]==recorded['final_state'][field]
   record['source_sha256']=input_hashes
   assert all(sha(p)==v for p,v in input_hashes.items())
   save(root/key/'report.json',record);output.append(record);counts[record['status']]+=1
   with (root/'cases.jsonl').open('a') as h:h.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
   print(json.dumps(dict(case=index+1,status=record['status'],intervals=record['audited_intervals']),ensure_ascii=False),flush=True)
  report=dict(cohort=args.cohort,cases=len(output),expected_cases=len(rows),is_smoke=args.limit is not None,
   status_counts=dict(counts),certificate_safe_cases=sum(r['passed'] for r in output),
   audited_intervals=sum(r['audited_intervals'] for r in output),
   moving_cases=sum(r.get('moving_control_records',0)>0 for r in output),
   moving_control_records=sum(r.get('moving_control_records',0) for r in output),
   previously_reported_arrivals=sum(r['prior_reported_arrival'] for r in output),
   previous_arrivals_with_continuous_body_certificate=sum(r['prior_reported_arrival'] and r['passed'] for r in output),
   unresolved_or_invalid_previous_arrivals=[dict(task=r['task'],seed=r['seed'],method=r['method'],status=r['status']) for r in output if r['prior_reported_arrival'] and not r['passed']],
   all_original_results_unchanged=True,source_summary_sha256=sha(original/'summary.json'),
   wall_duration_CPU_shared_s=time.monotonic()-started,source_sha256=frozen,actual_model_calls=0,new_executions=0,new_arrivals=0,
   limitations=['静态矩形/固定几何与分段恒定平均运动，双精度保守数值证书，不是形式化算术或真实动态交通证明',
    '复用GEOS但不复用原积分/车身/扫掠实现；不独立重评语言身份/到达保持或拓扑方向',
    '后验开发审核，不是新封存方法；未决/缺证据保留，原结果不覆盖'],goal_complete=False)
  save(root/'report.json',report)
  save(root/'complete.json',dict(cases=len(output),expected_cases=len(rows),complete_cohort=args.limit is None and len(output)==len(rows),report_sha256=sha(root/'report.json'),goal_complete=False))
  print(json.dumps(report,ensure_ascii=False),flush=True)
 except BaseException as exc:
  save(root/'interrupted.json',dict(error=repr(exc),completed_cases=len(output),goal_complete=False));raise
 finally:note('end')
if __name__=='__main__':main()
