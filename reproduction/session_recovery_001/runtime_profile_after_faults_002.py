"""全部既有GPU作业退出后顺序测量；失败前置不被改成成功，也不阻止独立性能证据。"""
from pathlib import Path
from datetime import datetime,timezone
import json,os,shutil,subprocess,sys,time
from conceptgraph.completion.stage9 import PROJECT,OUTPUT,digest,dump
from conceptgraph.completion.realtime_batch import verify_sources,pipeline_sources


def start_tick(pid):
 try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None


def GPU_pids():
 return subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15).stdout.strip()


def hardware():
 result={}
 for name,args in [('GPU',['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total','--format=csv,noheader']),('CPU',['lscpu','-J'])]:
  r=subprocess.run(args,capture_output=True,text=True,timeout=30);result[name]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 result['thread_environment']={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']}
 result['time_utc']=datetime.now(timezone.utc).isoformat();return result


def main():
 root=OUTPUT/'performance/runtime_profile_after_faults_002';root.mkdir(exist_ok=False)
 path=OUTPUT/'performance/runtime_profile_protocol_002/plan.json';plan=json.loads(path.read_text())
 auditpath=OUTPUT/'performance/runtime_profile_cache_key_audit_001/report.json';audit=json.loads(auditpath.read_text())
 assert audit['passed'] and audit['plan_sha256']==digest(path) and audit['actual_GPU_calls']==0
 smoke=OUTPUT/'performance/runtime_profile_cpu_smoke_001/complete.json'
 assert json.loads(smoke.read_text())['actual_CPU_frontend_calls']==12
 prior={} # 串行恢复监督器已等待恢复故障子进程终止；GPU空闲检查仍保留
 sources=set(pipeline_sources()+[PROJECT/p for p in plan['source_sha256']]+[Path(__file__).resolve(),
  OUTPUT/'source_bundle/runtime_profile_cache_key_audit_001.py'])
 frozen={str(p.relative_to(PROJECT)):digest(p) for p in sorted(sources)}
 release=dict(plan_sha256=digest(path),source_sha256=frozen,prior_process_start_ticks=prior,
  prior_results=['interactive_simulation/realtime_three_model_full_001/summary.json','ablations/semantic_navigation_chain_001/complete.json','faults/realtime_fault_after_semantics_002/complete.json'],
  CPU_smoke_sha256=digest(smoke),cache_key_audit_sha256=digest(auditpath),
  wait_budget_s=176400,total_execution_budget_s=10800,child_timeout_s=3600,
  storage_budget_bytes=2000000000,minimum_free_bytes=6000000000,
  preceding_failure_policy='三个既有原进程均退出且GPU无计算进程，才开始独立性能；前置失败原样保留，不能推断其完成',
  model_execution_policy='GPU前端、原Qwen、旧Adapter、新Adapter逐子进程；同一时刻只有一个计算子进程。录制CPU最后单列',
  queued_requests_not_executed=True,goal_complete=False)
 dump(root/'release.json',release);dump(root/'hardware_queued.json',hardware())
 begun=time.monotonic()
 while any(start_tick(int(pid))==tick for pid,tick in prior.items()):
  if time.monotonic()-begun>176400:
   dump(root/'wait_failed.json',dict(reason='既有GPU作业未在等待预算内终止',goal_complete=False));raise TimeoutError('previous jobs still live')
  time.sleep(5)
 verify_sources(frozen);assert digest(path)==release['plan_sha256']
 if GPU_pids():
  dump(root/'gpu_busy.json',dict(pids=GPU_pids(),goal_complete=False));raise RuntimeError('CUDA occupied; no concurrent model')
 dump(root/'preceding_terminal_snapshot.json',dict(results={p:dict(exists=(OUTPUT/p).exists(),sha256=digest(OUTPUT/p) if (OUTPUT/p).exists() else None) for p in release['prior_results']},
  actual_original_processes_exited=True,all_success_not_assumed=True,goal_complete=False))
 for variant in plan['semantic_variants'].values():
  if variant['adapter']:assert digest(Path(variant['adapter'])/'adapter_model.safetensors')==variant['weights_sha256']
 jobs=[dict(mode='frontend',variant='new',device='cuda:0',run='runtime_profile_gpu_frontend_001')]
 jobs += [dict(mode='semantic',variant=v,device='cuda:0',run='runtime_profile_gpu_'+v+'_001') for v in ['base','old','new']]
 jobs += [dict(mode='recorded',variant='new',device='cpu',run='runtime_profile_post_GPU_recorded_001')]
 base_env=dict(os.environ,LD_LIBRARY_PATH='/root/autodl-fs/third_party/graphics_runtime/lib:/root/miniconda3/envs/conceptgraph/lib',
  OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 base_env.pop('PYTHONPATH',None);base_env.pop('CUDA_VISIBLE_DEVICES',None)
 rows=[];started=time.monotonic()
 def bytes_used():
  return sum(p.stat().st_size for j in jobs for p in (OUTPUT/'performance'/j['run']).rglob('*') if p.is_file())
 try:
  for index,job in enumerate(jobs):
   verify_sources(frozen)
   remaining=10800-(time.monotonic()-started)
   assert remaining>10 and shutil.disk_usage(OUTPUT).free>6000000000 and bytes_used()<2000000000
   if GPU_pids():raise RuntimeError('CUDA busy between benchmark subprocesses')
   env=dict(base_env)
   if job['device']=='cpu':
    runtime=OUTPUT/'source_bundle/software_render_runtime_001'
    env.update(CUDA_VISIBLE_DEVICES='',PYOPENGL_PLATFORM='osmesa',PYGLET_SHADOW_WINDOW='false',LP_NUM_THREADS='2',
      PYTHONPATH=str(runtime/'python_deps'),LD_LIBRARY_PATH=f'{runtime}/extracted/usr/lib/x86_64-linux-gnu:'+base_env['LD_LIBRARY_PATH'])
   dump(root/f'{index:02d}_hardware_before.json',hardware())
   command=[sys.executable,'-m','conceptgraph.completion.runtime_profiling','--plan',str(path),'--run-id',job['run'],'--mode',job['mode'],'--variant',job['variant'],'--device',job['device']]
   with (root/f'{index:02d}.log').open('x') as log:
    child=subprocess.run(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=min(3600,remaining))
   reportpath=OUTPUT/'performance'/job['run']/'report.json'
   rows.append(dict(**job,returncode=child.returncode,report_sha256=digest(reportpath) if reportpath.exists() else None))
   dump(root/f'{index:02d}_result.json',rows[-1])
   if child.returncode or not reportpath.exists():raise RuntimeError('performance child failed; no automatic retry')
   report=json.loads(reportpath.read_text())
   assert report['unique_CUDA_process_verified']==(job['device']=='cuda:0')
   assert bytes_used()<2000000000
  reports={j['run']:json.loads((OUTPUT/'performance'/j['run']/'report.json').read_text()) for j in jobs}
  actual=[r['actual'] for r in reports.values()]
  assert sum(r.get('actual_Qwen_calls',0) for r in actual)==216
  assert sum(r.get('replay_added_calls',0) for r in actual)==0
  dump(root/'complete.json',dict(jobs=rows,reports=reports,elapsed_s_including_wait=time.monotonic()-begun,
   execution_elapsed_s=time.monotonic()-started,saved_bytes=bytes_used(),actual_Qwen_calls=216,
   Qwen_cache_requests=216,cache_added_calls=0,actual_GPU_frontend_calls=12,
   benchmark_is_not_end_to_end_navigation_throughput=True,new_test_opened=False,goal_complete=False))
 except BaseException as exc:
  dump(root/'interrupted.json',dict(error=repr(exc),completed_jobs=len(rows),goal_complete=False));raise

if __name__=='__main__':main()
