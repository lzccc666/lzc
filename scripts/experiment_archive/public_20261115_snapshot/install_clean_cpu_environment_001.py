"""独立venv真实安装；不借用旧site-packages，持续资源守卫，只停止自身子进程。"""
import datetime,hashlib,json,os,shutil,signal,subprocess,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'reproduction/clean_cpu_install_001';POLICY=R/'json/resource_policy_002.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,data):
 with (O/name).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')
def guard(reserve=0):
 pol=json.loads(POLICY.read_text());size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink());free=shutil.disk_usage(R).free
 if size+reserve>=pol['new_project_artifact_budget_bytes'] or free<pol['minimum_physical_free_GiB']*2**30:raise OSError(f'25GiB/20GiB守卫：project={size},free={free}')
 envsize=sum(p.stat().st_size for p in O.rglob('*') if p.is_file() and not p.is_symlink())
 if envsize>=9*2**30:raise OSError('本次安装自身9GiB峰值预算触发，保留现场')
 return dict(project_bytes=size,free_bytes=free,environment_and_temporary_bytes=envsize)
def step(name,args,env):
 log=O/f'{name}.log';begun=time.monotonic();child=None
 try:
  with log.open('x') as out:
   child=subprocess.Popen(args,stdout=out,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   while child.poll() is None:
    guard(8*2**20)
    if time.monotonic()-begun>2400:raise TimeoutError(f'{name}超过40分钟，终止本次安装子进程')
    time.sleep(10)
  result=dict(name=name,command=args,exit_code=child.returncode,wall_s=time.monotonic()-begun,log_sha256=sha(log),resources=guard());save(f'{name}_receipt.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
  if child.returncode:raise RuntimeError(f'{name}退出{child.returncode}；日志及下载前缀保留')
  return result
 finally:
  if child is not None and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=8)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
def main():
 frozen={str(p):sha(p) for p in [Path(__file__),POLICY,O/'registration.json',O/'requirements.txt',O/'constraints.txt']};save('execution_binding.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources_sha256=frozen,goal_complete=False));guard(6*2**30)
 env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None);env['CUDA_VISIBLE_DEVICES']='';env['PYTHONDONTWRITEBYTECODE']='1';env['PIP_NO_CACHE_DIR']='1';env['PIP_DISABLE_PIP_VERSION_CHECK']='1';env['PIP_CONFIG_FILE']='/dev/null';env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
 for key,sub in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('YOLO_CONFIG_DIR','yolo_config')]:
  p=O/sub;p.mkdir(exist_ok=False);env[key]=str(p)
 py=str(O/'venv/bin/python');receipts=[]
 receipts.append(step('create_venv',['/root/miniconda3/bin/python','-m','venv',str(O/'venv')],env))
 receipts.append(step('install_cpu_torch',[py,'-m','pip','install','--no-cache-dir','--timeout','30','--retries','2','--index-url','https://download.pytorch.org/whl/cpu','--report',str(O/'torch_pip_report.json'),'torch==2.0.1+cpu','torchvision==0.15.2+cpu'],env))
 receipts.append(step('install_project_packages',[py,'-m','pip','install','--no-cache-dir','--timeout','30','--retries','2','--index-url','https://pypi.org/simple','--extra-index-url','https://download.pytorch.org/whl/cpu','--prefer-binary','--report',str(O/'project_pip_report.json'),'-c',str(O/'constraints.txt'),'-r',str(O/'requirements.txt')],env))
 save('installation_report.json',dict(steps=receipts,clean_venv_core_installation_passed=True,local_gradslam_and_SAM_still_required=True,whole_chain_runtime_passed=False,GPU_reproduction_passed=False,sources_sha256=frozen,resources=guard(),goal_complete=False))
 print('clean CPU core pip install finished; full runtime not yet claimed',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as exc:save('failure_001.json',dict(error=repr(exc),traceback=traceback.format_exc(),resources=guard(),goal_complete=False));raise
