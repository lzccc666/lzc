"""冻结主矩阵全部结束后，仅CPU审核其全部实际控制；不并发任何模型。"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess,sys,time
PROJECT=Path('/root/autodl-tmp/lizichong');OUT=PROJECT/'outputs/project_v2/thesis_completion_v1'
ROOT=OUT/'vehicle_execution/independent_body_after_main_001';ROOT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,value):
 with path.open('x') as h:json.dump(value,h,ensure_ascii=False,indent=2,allow_nan=False)
def start_tick(pid):
 try:return int(Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19])
 except FileNotFoundError:return None
main=OUT/'interactive_simulation/realtime_three_model_full_001';driver=OUT/'source_bundle/independent_body_audit_001.py'
frozen={str(p.relative_to(PROJECT)):sha(p) for p in [Path(__file__).resolve(),driver,PROJECT/'conceptgraph/completion/independent_continuous_body.py',OUT/'task_protocol/independent_continuous_body_001.md']}
release=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=frozen,main_release_sha256=sha(main/'release.json'),
 main_pid=26469,main_start_ticks=start_tick(26469),wait_budget_s=176400,child_budget_s=900,storage_budget_bytes=1024**3,
 minimum_free_bytes=6*1024**3,run_id='independent_body_main_full_001',CUDA_visible='',CPU_shared=True,
 queued_is_not_completed=True,actual_model_calls=0,new_executions=0,goal_complete=False)
save(ROOT/'release.json',release);print(json.dumps(dict(pid=os.getpid(),waiting_for=str(main/'summary.json'),CUDA_visible='',goal_complete=False)),flush=True)
started=time.monotonic()
try:
 while not (main/'summary.json').exists():
  if time.monotonic()-started>176400:raise TimeoutError('主矩阵未在预算内完整，不伪造审核')
  if start_tick(26469)!=release['main_start_ticks']:
   raise RuntimeError('原主进程结束且尚无完整summary，保留失败，不重复启动')
  time.sleep(5)
 assert all(sha(PROJECT/p)==v for p,v in frozen.items())
 assert sha(main/'release.json')==release['main_release_sha256']
 for p,v in json.loads((main/'release.json').read_text())['source_sha256'].items():assert sha(PROJECT/p)==v
 summary=json.loads((main/'summary.json').read_text());assert summary['attempted']==summary['expected']==450
 assert sha(main/'trial_records.jsonl')==summary['trial_records_sha256']
 assert shutil.disk_usage(OUT).free>6*1024**3
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 with (ROOT/'audit.log').open('x') as log:
  child=subprocess.run([sys.executable,str(driver),'--cohort','main','--run-id',release['run_id']],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
 assert child.returncode==0,'独立主矩阵车身审核失败，保留原结果，不重试'
 destination=OUT/'vehicle_execution'/release['run_id'];report=json.loads((destination/'report.json').read_text())
 assert report['cases']==report['expected_cases']==450 and not report['is_smoke']
 assert report['actual_model_calls']==report['new_executions']==report['new_arrivals']==0
 save(ROOT/'complete.json',dict(cases=450,report_sha256=sha(destination/'report.json'),status_counts=report['status_counts'],
  previous_arrivals=report['previously_reported_arrivals'],previous_arrivals_with_certificate=report['previous_arrivals_with_continuous_body_certificate'],
  still_requires_independent_input_ledger_verification=True,new_arrivals=0,goal_complete=False))
except BaseException as exc:
 save(ROOT/'interrupted.json',dict(error=repr(exc),goal_complete=False));raise
