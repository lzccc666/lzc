"""实际串行资源调度：闭环、独立审计、停止本Goal的CARLA实例，再同预算逐模型评测。"""
import datetime,json,os,signal,subprocess,time
from pathlib import Path
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];PY='/root/miniconda3/envs/conceptgraph/bin/python';O=R/'json/serial_gpu_jobs_001';O.mkdir(exist_ok=False);env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0');records=[]
def run(script,args,log,carla=False):
 e=env.copy();e['PYTHONPATH']=('/autodl-fs/data/third_party/carla_0.9.16/python_site:' if carla else '')+'/root/autodl-tmp/lizichong';start=time.monotonic();cmd=[PY,str(R/'source'/script)]+args
 with log.open('x') as h:result=subprocess.run(cmd,env=e,stdout=h,stderr=subprocess.STDOUT)
 r=dict(script=script,args=args,returncode=result.returncode,wall_s=time.monotonic()-start,log=str(log),script_sha256=digest(R/'source'/script));records.append(r);dump(O/f'job_{len(records):02d}.json',r);print(json.dumps(r),flush=True);return result.returncode
run('carla_pipeline_episode_010.py',[],R/'carla_official/pipeline_episode_010.log',True)
if (R/'carla_official/pipeline_episode_010/execution.jsonl').exists():run('audit_carla_arrival_003.py',['--episode','pipeline_episode_010'],R/'carla_official/pipeline_episode_010_audit.log')
# 仅关闭本Goal启动并验证可执行文件的具体服务器PID，保留所有安装和日志。
pid=14124
try:
 command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
 if '/autodl-fs/data/third_party/carla_0.9.16/install/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping' not in command:raise RuntimeError('server pid does not match; refuse signal')
 os.kill(pid,signal.SIGTERM);dump(O/'server_shutdown.json',dict(pid=pid,signal='SIGTERM',verified_command=command,reason='GPU独占语义评测；安装和日志保留，可按原命令重启'))
 for _ in range(30):
  if not Path(f'/proc/{pid}').exists():break
  time.sleep(.5)
except FileNotFoundError:dump(O/'server_shutdown.json',dict(pid=pid,already_exited=True))
for name in ['base','old_adapter','historical_three_task_seed17','new_graph_seed17','new_graph_seed29','new_graph_seed43']:
 run('evaluate_official_graph_models_001.py',['--variant',name],R/f'qwen_adapter_fair_comparison/fair_gpu_{name}_001.log')
required=[R/f'qwen_adapter_fair_comparison/official_graph_fair_gpu_001/{n}/report.json' for n in ['base','old_adapter','historical_three_task_seed17','new_graph_seed17','new_graph_seed29','new_graph_seed43']]
if all(p.exists() for p in required):run('summarize_official_graph_semantics_001.py',[],R/'qwen_adapter_fair_comparison/official_graph_decisions_001.log')
dump(O/'summary.json',dict(jobs=records,all_subprocesses_succeeded=all(r['returncode']==0 for r in records),goal_complete=False))
