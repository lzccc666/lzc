"""只接续正在运行的冻结GPU作业；成功结束后顺序汇总，不重启推理、不覆盖。"""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'nuscenes_ablation/full_semantics_near6_002';O=R/'json/full_semantics_near6_followup_002'

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while data:=f.read(1024*1024):h.update(data)
 return h.hexdigest()

def save(path,data):
 with path.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')

def start_tick(pid):
 try:
  fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return None if fields[0]=='Z' else fields[19]
 except FileNotFoundError:return None

def guard(reserve):
 total=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
 policy=json.loads((R/'json/resource_policy_002.json').read_text())
 if total+reserve>=policy['new_project_artifact_budget_bytes'] or shutil.disk_usage(R).free<policy['minimum_physical_free_GiB']*2**30:raise OSError('不能越过用户25GiB或物理空闲20GiB')
 return total

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pid',required=True,type=int);a=ap.parse_args();O.mkdir(exist_ok=False)
 jobs=[('aggregate_six_conditions',R/'source/nuscenes_full_semantics_near6_002.py',['summarize'],256*1024),('route_and_rule_details',R/'source/summarize_full_semantics_near6_details_002.py',[],96*1024),('editable_tables',R/'source/export_full_semantics_near6_office_002.py',[],160*1024)]
 sources={str(p):digest(p) for _,p,_,_ in jobs};sources[str(Path(__file__))]=digest(__file__);initial=start_tick(a.pid)
 save(O/'registration.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),watched_PID=a.pid,watched_process_start_tick=initial,source_sha256=sources,actions=[x[0] for x in jobs],no_GPU_restart=True,no_existing_output_overwrite=True,goal_complete=False))
 try:
  began=time.monotonic()
  while initial is not None and start_tick(a.pid)==initial:
   if time.monotonic()-began>10800:raise TimeoutError('等待原GPU进程超过3小时；未终止该进程')
   time.sleep(20)
  run=json.loads((D/'run_report.json').read_text());release=json.loads((D/'release.json').read_text())
  assert run['actual_CPU_cooperation_decisions']==9048 and run['actual_new_GPU_requests']==sum(release['planned_actual_GPU_requests'].values())
  assert digest(D/'decisions.jsonl.gz')==run['raw_sha256']
  steps=[]
  for name,path,args,reserve in jobs:
   used=guard(reserve);assert all(digest(p)==h for p,h in sources.items());log=O/f'{name}.log';tick=time.monotonic()
   with log.open('x') as f:
    completed=subprocess.run([sys.executable,str(path),*args],env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,check=False)
   steps.append(dict(action=name,exit_code=completed.returncode,wall_s=time.monotonic()-tick,log=str(log),log_sha256=digest(log),project_bytes_before=used))
   if completed.returncode:raise RuntimeError(f'{name} exited {completed.returncode}; existing outputs retained')
   print(json.dumps(steps[-1],ensure_ascii=False),flush=True)
  save(O/'report.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),steps=steps,actual_GPU_requests=run['actual_new_GPU_requests'],actual_CPU_cooperation=9048,project_bytes_after=guard(4096),new_sealed_test=False,thesis_prose_written=False,goal_complete=False))
  print('full semantic inference follow-up finished',flush=True)
 except BaseException as exc:
  save(O/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),goal_complete=False));raise

if __name__=='__main__':main()
