from pathlib import Path
from datetime import datetime,timezone
import json,os,shutil,subprocess,sys,time
from conceptgraph.completion.stage9 import PROJECT,OUTPUT,digest,dump
from conceptgraph.completion.realtime_batch import verify_sources
root=OUTPUT/'performance/runtime_profile_cpu_smoke_001';root.mkdir(exist_ok=False)
planpath=OUTPUT/'performance/runtime_profile_protocol_001/plan.json';plan=json.loads(planpath.read_text());verify_sources(plan['source_sha256'])
runtime=OUTPUT/'source_bundle/software_render_runtime_001'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',LP_NUM_THREADS='2',
 PYOPENGL_PLATFORM='osmesa',PYGLET_SHADOW_WINDOW='false',PYTHONPATH=str(runtime/'python_deps'),
 LD_LIBRARY_PATH=f'{runtime}/extracted/usr/lib/x86_64-linux-gnu:/root/autodl-fs/third_party/graphics_runtime/lib:/root/miniconda3/envs/conceptgraph/lib',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
dump(root/'release.json',dict(plan_sha256=digest(planpath),source_sha256=digest(__file__),CUDA_visible='',
 total_wall_budget_s=1200,child_timeout_s=600,storage_budget_bytes=1000000000,minimum_free_bytes=6000000000,
 CPU_shared=True,GPU_credit=False,goal_complete=False))
def note(state):
 with (OUTPUT/'interactive_simulation/realtime_three_model_full_001/resource_events.jsonl').open('a') as h:
  h.write(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(),event='runtime_profile_cpu_smoke_001',status=state,CUDA_visible='',shared_CPU=True))+'\n')
rows=[];begun=time.monotonic();note('start')
try:
 for mode in ['frontend','recorded']:
  assert shutil.disk_usage(OUTPUT).free>6000000000 and time.monotonic()-begun<1200
  verify_sources(plan['source_sha256']);run='runtime_profile_cpu_'+mode+'_001'
  with (root/(mode+'.log')).open('x') as log:
   child=subprocess.run([sys.executable,'-m','conceptgraph.completion.runtime_profiling','--plan',str(planpath),'--run-id',run,'--mode',mode,'--device','cpu'],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
  result=OUTPUT/'performance'/run/'report.json'
  row=dict(mode=mode,returncode=child.returncode,report_sha256=digest(result) if result.exists() else None)
  rows.append(row);dump(root/(mode+'_result.json'),row)
  if child.returncode or not result.exists():raise RuntimeError('CPU performance smoke failed; preserve and stop expansion')
 dump(root/'complete.json',dict(results=rows,elapsed_s=time.monotonic()-begun,actual_CPU_frontend_calls=12,
  formal_GPU_exclusive_performance_completed=False,goal_complete=False))
except BaseException as exc:
 dump(root/'interrupted.json',dict(error=repr(exc),completed_modes=len(rows),goal_complete=False));raise
finally:note('end')
