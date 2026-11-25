"""按冻结清单串行执行15个官方CARLA episode，失败保留并继续其他条件。"""
import json,os,subprocess,time,traceback
from pathlib import Path
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];O=R/'json/carla_ablation_jobs_002';PY='/root/miniconda3/envs/conceptgraph/bin/python'
def main():
 O.mkdir(exist_ok=False);bank=json.loads((R/'carla_official/ablation_registration_002.json').read_text());env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',PYTHONPATH='/autodl-fs/data/third_party/carla_0.9.16/python_site:/root/autodl-tmp/lizichong');jobs=[]
 import sys
 sys.path.insert(0,'/autodl-fs/data/third_party/carla_0.9.16/python_site')
 import carla
 ready=False
 for attempt in range(100):
  try:
   client=carla.Client('127.0.0.1',2010);client.set_timeout(3);version=client.get_server_version();world=client.get_world();mapname=world.get_map().name;ready=version=='0.9.16' and 'Town10HD_Opt' in mapname
   if ready:break
  except RuntimeError:pass
  if attempt%5==0:print(json.dumps(dict(waiting_for_official_server=True,attempt=attempt)),flush=True)
  time.sleep(3)
 if not ready:raise RuntimeError('official server not ready; no episodes started')
 dump(O/'server_ready.json',dict(server_version=version,map=mapname,registration_sha256=digest(R/'carla_official/ablation_registration_002.json')))
 for ep in bank['episodes']:
  for condition in bank['conditions']:
   episode=ep['episode'];log=R/f'carla_official/ablation_{episode}_{condition}_002.log';start=time.monotonic()
   with log.open('x') as f:p=subprocess.run([PY,str(R/'source/carla_ablation_episode_002.py'),'--episode',episode,'--condition',condition],env=env,stdout=f,stderr=subprocess.STDOUT)
   row=dict(episode=episode,condition=condition,returncode=p.returncode,wall_s=time.monotonic()-start,log=str(log));jobs.append(row);dump(O/f'job_{len(jobs):02d}.json',row);print(json.dumps(row),flush=True)
   E=R/'carla_official/ablation_batch_002'/episode/condition
   if (E/'summary.json').exists():
    auditlog=R/f'carla_official/ablation_{episode}_{condition}_audit_002.log'
    with auditlog.open('x') as f:q=subprocess.run([PY,str(R/'source/audit_carla_arrival_003.py'),'--episode',f'ablation_batch_002/{episode}/{condition}'],env={**env,'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT)
    row['audit_returncode']=q.returncode
   if p.returncode:
    # 资源守卫是停止条件，普通场景失败继续以免只剩正例。
    failure=(E/'failure.json').read_text() if (E/'failure.json').exists() else log.read_text()
    dump(O/'subprocess_failure.json',dict(episode=episode,condition=condition,failure=failure,goal_complete=False));break
  else:continue
  break
 dump(O/'summary.json',dict(jobs=jobs,completed_subprocesses=len(jobs),expected=bank['expected_episodes'],all_processes_succeeded=len(jobs)==bank['expected_episodes'] and all(j['returncode']==0 and j.get('audit_returncode')==0 for j in jobs),source_sha256=digest(__file__),goal_complete=False))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  dump(O/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),goal_complete=False));raise
