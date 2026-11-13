from pathlib import Path
import subprocess,os,json,time,hashlib,datetime
P=Path('/root/autodl-tmp/lizichong');O=P/'outputs/project_v2/thesis_completion_v1';D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):
 with (D/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2)
def live(pid,tick):
 try:
  s=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return s[0]!='Z' and s[19]==tick
 except FileNotFoundError:return False
base=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(P),LD_LIBRARY_PATH='/root/miniconda3/envs/conceptgraph/lib',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2');base.pop('CUDA_VISIBLE_DEVICES',None)
helpers={str(p):sha(p) for p in D.glob('*.py')}
originals={}
for rel in ['ablations/semantic_navigation_chain_003/release.json','faults/realtime_fault_after_semantics_001/release.json','performance/runtime_profile_after_faults_001/release.json','reproduction/isolated_environment_after_GPU_001/release.json']:
 j=json.loads((O/rel).read_text())
 for n,h in j['source_sha256'].items():
  p=Path(n);p=p if p.is_absolute() else P/p;assert sha(p)==h,str(p);originals[str(p)]=h
save('supervisor_release.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),helpers=helpers,original_frozen_sources=originals,original_runs_preserved=True,goal_complete=False))
def run(label,args,timeout,cpu=False):
 for n,h in {**originals,**helpers}.items():assert sha(n)==h,n
 env=dict(base)
 if cpu:env['CUDA_VISIBLE_DEVICES']=''
 with (D/(label+'.log')).open('x') as log:
  try:r=subprocess.run([os.sys.executable]+args,cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout);code=r.returncode
  except subprocess.TimeoutExpired:code=124
 save(label+'_exit.json',dict(returncode=code,goal_complete=False));return code
try:
 r=json.loads((D/'resume_started.json').read_text());started=time.monotonic()
 while live(r['pid'],r['start_tick']):
  assert time.monotonic()-started<86400;time.sleep(10)
 summary=O/'interactive_simulation/realtime_semantic_minus_full_001/summary.json'
 full=summary.exists() and json.loads(summary.read_text())['attempted']==450
 codes={}
 if full:
  codes['compatibility']=run('01_compatibility',[str(D/'semantic_source_compatibility_004.py'),'--resume'],3600,True)
  codes['execution_audit']=run('02_execution_audit',[str(O/'source_bundle/semantic_minus_execution_audit_001.py'),'--run-id','semantic_minus_execution_full_002'],4000,True)
  if codes['compatibility']==0:codes['faults']=run('03_faults',[str(D/'realtime_fault_after_semantics_002.py')],50000)
 else:save('minus_incomplete.json',dict(reason='恢复消融未获得450条完整记账，不运行依赖故障',goal_complete=False))
 # 前置失败不冒充成功；独立性能及新环境仍按原协议执行。
 codes['performance']=run('04_performance',[str(D/'runtime_profile_after_faults_002.py')],11500)
 codes['environment']=run('05_environment',[str(D/'isolated_environment_after_GPU_002.py')],7800,True)
 save('finished.json',dict(jobs=codes,all_jobs_passed=full and len(codes)==5 and all(v==0 for v in codes.values()),goal_complete=False))
except BaseException as e:
 save('interrupted.json',dict(error=repr(e),goal_complete=False));raise
