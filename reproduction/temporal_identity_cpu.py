from datetime import datetime, timezone
import json,os,shutil,subprocess,sys,time
from conceptgraph.completion.stage9 import OUTPUT,PROJECT,digest,dump
root=OUTPUT/'statistics/temporal_identity_queue_001';root.mkdir(exist_ok=False)
dump(root/'release.json',dict(source_sha256=digest(PROJECT/'conceptgraph/completion/temporal_identity_evaluation.py'), protocol_sha256=digest(OUTPUT/'task_protocol/temporal_identity_001.md'),budget_wall_s=900,budget_bytes=1024**3,minimum_free_bytes=6*1024**3,CUDA_visible='',shared_CPU=True,goal_complete=False))
def note(state):
 with (OUTPUT/'interactive_simulation/realtime_three_model_full_001/resource_events.jsonl').open('a') as h:
  h.write(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(),event='temporal_identity_cpu_001',status=state,CUDA_visible='',shared_CPU=True))+'\n')
started=time.monotonic();note('start')
try:
 for run,limit in [('temporal_identity_smoke_001',1),('temporal_identity_full_001',None)]:
  assert shutil.disk_usage(OUTPUT).free>6*1024**3
  args=[sys.executable,'-m','conceptgraph.completion.temporal_identity_evaluation','--mapping-run','causal_mapping_full_001','--run-id',run]
  if limit:args+=['--scene-limit',str(limit)]
  with (root/(run+'.log')).open('x') as log:
   result=subprocess.run(args,cwd=PROJECT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,timeout=900-(time.monotonic()-started))
  if result.returncode:raise RuntimeError(f'{run} failed; stop expansion and preserve')
 dump(root/'complete.json',dict(wall_s=time.monotonic()-started,actual_model_calls=0,goal_complete=False))
except BaseException as exc:
 dump(root/'interrupted.json',dict(error=repr(exc),goal_complete=False));raise
finally:note('end')
