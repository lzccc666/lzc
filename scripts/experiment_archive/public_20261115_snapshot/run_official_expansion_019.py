"""等待已有GPU及其汇总结束，再启动本任务自己的官方服务器执行冻结小容量标定。"""
import argparse,datetime,hashlib,json,os,resource,shutil,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'json/official_expansion_execution_019';SHARED=Path('/autodl-fs/data/thesis_completion_v1_bulk/carla_runtime_001');INSTALL=Path('/autodl-fs/data/third_party/carla_0.9.16/install');API=INSTALL.parent/'python_site';PILOT=R/'source/run_carla_expansion_batch_011.py';DONE=R/'json/guarded_motion_audit_jobs_001/report.json'

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while data:=f.read(1024*1024):h.update(data)
 return h.hexdigest()

def save(path,value):
 with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def start_tick(pid):
 try:
  fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return None if fields[0]=='Z' else fields[19]
 except FileNotFoundError:return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',required=True,type=int);args=ap.parse_args();O.mkdir(exist_ok=False);watched=start_tick(args.wait_pid);begun=time.monotonic();server=None;logfile=None;stops=[]
 registration=R/'carla_official/expansion_registration_011.json';frozen={str(PILOT):digest(PILOT),str(Path(__file__)):digest(__file__),str(registration):digest(registration)}
 save(O/'registration.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),wait_PID=args.wait_pid,wait_start_tick=watched,source_sha256=frozen,required_project_reserve_bytes=512*1024**2,no_other_GPU_job_parallel=True,only_own_server_terminated=True,scope='官方50任务三种子开发扩展，完整传感器、真实合作语义与当前运动保护',goal_complete=False))
 try:
  while watched is not None and start_tick(args.wait_pid)==watched:
   if time.monotonic()-begun>10800:raise TimeoutError('等待原作业超过3h；没有结束该作业')
   time.sleep(20)
  completed=json.loads(DONE.read_text());assert completed['all_three_complete_and_guard_audited']
  assert all(digest(p)==h for p,h in frozen.items());size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink());free=shutil.disk_usage(R).free
  policy=json.loads((R/'json/resource_policy_002.json').read_text())
  if size+512*1024**2>=policy['new_project_artifact_budget_bytes'] or free<policy['minimum_physical_free_GiB']*2**30:
   save(O/'blocked_storage.json',dict(project_bytes=size,physical_free_bytes=free,required_reserve=512*1024**2,actual_new_physical_calibrations=0,goal_complete=False));print('storage guard blocked pilot; no server started',flush=True);return
  gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True)
  if gpu.stdout.strip():raise RuntimeError('另有GPU计算进程，未启动CARLA：'+gpu.stdout.strip())
  with socket.socket() as probe:
   probe.settimeout(2)
   if probe.connect_ex(('127.0.0.1',2010))==0:raise RuntimeError('2010已有服务，未重载未知服务器')
  stdout=SHARED/'stdout_graphics_expansion_021.log';graphic=SHARED/'carla_graphics_expansion_021.log';assert not graphic.exists();logfile=stdout.open('x');env=os.environ.copy();env.pop('LD_PRELOAD',None);env['PATH']='/usr/bin:/bin';env['CUDA_VISIBLE_DEVICES']='0';env['XDG_CONFIG_HOME']=str(SHARED/'home/.config');env['XDG_CACHE_HOME']=str(SHARED/'home/.cache');env['XDG_DATA_HOME']=str(SHARED/'home/.local/share');env['XDG_RUNTIME_DIR']=str(SHARED/'xdg');env['LD_LIBRARY_PATH']=str(SHARED/'deps/usr/lib/x86_64-linux-gnu')+':/usr/lib/x86_64-linux-gnu';resource.setrlimit(resource.RLIMIT_CORE,(0,0))
  command=['/usr/sbin/runuser','-u','nobody','--','env',f"XDG_RUNTIME_DIR={env['XDG_RUNTIME_DIR']}",f"LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}",str(INSTALL/'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'),'CarlaUE4',f'-UserDir={SHARED}/home/.config/Epic/CarlaUE4/','-RenderOffScreen','-quality-level=Low','-carla-rpc-port=2010','-nosound','-stdout','-FullStdOutLogOutput',f'-abslog={graphic}']
  server=subprocess.Popen(command,cwd=INSTALL,env=env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True);save(O/'server_started.json',dict(PID=server.pid,command=command,stdout=str(stdout),graphics_log=str(graphic),started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),project_bytes_before=size,goal_complete=False));sys.path.insert(0,str(API));import carla
  client=None;ready=False;tick=time.monotonic();attempts=0;probe_errors=[]
  while time.monotonic()-tick<600:
   if server.poll() is not None:raise RuntimeError(f'官方服务器启动退出{server.returncode}')
   try:
    attempts+=1;client=carla.Client('127.0.0.1',2010);client.set_timeout(10)
    version=client.get_server_version();town=client.get_world().get_map().name
    if version!='0.9.16' or 'Town10HD_Opt' not in town:raise ValueError(f'官方版本/地图不符{version}/{town}')
    ready=True;break
   except RuntimeError as exc:
    if len(probe_errors)<8:probe_errors.append(dict(attempt=attempts,elapsed_s=time.monotonic()-tick,error=str(exc)))
    time.sleep(2)
  save(O/'readiness_probe.json',dict(attempts=attempts,ready=ready,elapsed_s=time.monotonic()-tick,first_errors=probe_errors,fresh_client_each_retry=True,timeout_per_call_s=10,goal_complete=False))
  if not ready:raise TimeoutError('官方服务器600s未就绪')
  runenv=os.environ.copy();runenv['PYTHONPATH']=str(API)+os.pathsep+runenv.get('PYTHONPATH','');runenv['CUDA_VISIBLE_DEVICES']='0';log=O/'pilot.log'
  with log.open('x') as f:result=subprocess.run([sys.executable,str(PILOT)],stdout=f,stderr=subprocess.STDOUT,env=runenv,timeout=604800,check=False)
  if result.returncode:raise RuntimeError(f'真实标定退出{result.returncode}；已保存前缀')
  report=R/'json/expansion_jobs_011_primary/report.json';actual=json.loads(report.read_text());assert actual['all_execution_and_audits_closed'] and actual['actual_launched_episodes']==150
  save(O/'report.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=result.returncode,server_version=version,client_version=client.get_client_version(),town=town,actual_new_physical_calibrations=0,source_report_sha256=digest(report),log_sha256=digest(log),actual_new_navigation_episodes=actual['actual_launched_episodes'],strict_arrival_must_read_independent_reports=True,goal_complete=False));print('registered expansion primary episodes and audits complete',flush=True)
 except BaseException as exc:
  save(O/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),goal_complete=False));raise
 finally:
  if server is not None:
   for sig,timeout in [(signal.SIGINT,8),(signal.SIGTERM,5),(signal.SIGKILL,5)]:
    if server.poll() is not None:break
    os.killpg(server.pid,sig);stops.append(sig.name)
    try:server.wait(timeout=timeout)
    except subprocess.TimeoutExpired:pass
   if logfile is not None:logfile.close()
   external={str(p):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in [SHARED/'stdout_graphics_expansion_021.log',SHARED/'carla_graphics_expansion_021.log'] if p.exists()}
   save(O/'server_shutdown.json',dict(PID=server.pid,exit_code=server.poll(),own_process_group_signals=stops,shared_official_runtime_logs=external,goal_complete=False))

if __name__=='__main__':main()
