"""固定一个损失加权开发pilot，候选逆序实验结束后串行训练和同108条实际评测。"""
import json,os,shutil,subprocess,time
from pathlib import Path
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];Q=R/'qwen_adapter_fair_comparison';O=R/'json/weighted_pilot_jobs_001';PY='/root/miniconda3/envs/conceptgraph/bin/python'
def main():
 O.mkdir(exist_ok=False)
 while not (Q/'decision_weighted_pilot_registration_001.json').exists():time.sleep(5)
 registration=json.loads((Q/'decision_weighted_pilot_registration_001.json').read_text());sources=dict(registration['source_sha256']);sources[str(R/'source/evaluate_weighted_pilot_001.py')]=digest(R/'source/evaluate_weighted_pilot_001.py');dump(O/'release.json',dict(source_sha256=sources,registration_sha256=digest(Q/'decision_weighted_pilot_registration_001.json'),wait_for='json/real_graph_permutation_jobs_002/summary.json',new_GPU_eval_requests=108,training_seed=17,training_max_steps=240,model_selection='same18 validation only',goal_complete=False))
 while not (R/'json/real_graph_permutation_jobs_002/summary.json').exists():
  if (R/'json/real_graph_permutation_jobs_002/failure.json').exists():raise RuntimeError('permutation upstream failure; no pilot GPU started')
  time.sleep(20)
 if shutil.disk_usage(R).free<20*2**30 or sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())+16*2**20>10*2**30:raise OSError('pilot reserve16MiB unavailable; do not write past10GiB')
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',PYTHONPATH='/root/autodl-tmp/lizichong');jobs=[]
 for script,args in [('train_real_graph_decision_weighted_001.py',['--dataset','real_graph_dataset_003','--run-id','decision_weighted_pilot_seed17_001','--seed','17','--max-steps','240','--eval-steps','80','--grad-accum','4']),('evaluate_weighted_pilot_001.py',['--prepare']),('evaluate_weighted_pilot_001.py',['--variant','decision_weighted_pilot_seed17'])]:
  assert all(digest(p)==h for p,h in sources.items());log=O/f'job_{len(jobs)+1:02d}.log'
  with log.open('x') as f:proc=subprocess.run([PY,str(R/'source'/script)]+args,env=env,stdout=f,stderr=subprocess.STDOUT)
  jobs.append(dict(script=script,args=args,returncode=proc.returncode));dump(O/f'job_{len(jobs):02d}.json',jobs[-1])
  if proc.returncode:raise RuntimeError('pilot failed; preserve prefix '+str(log))
 dump(O/'summary.json',dict(jobs=jobs,new_actual_GPU_eval_requests=108,single_seed_pilot_not_formal_three_seeds=True,goal_complete=False))
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  import traceback
  dump(O/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),goal_complete=False));raise
