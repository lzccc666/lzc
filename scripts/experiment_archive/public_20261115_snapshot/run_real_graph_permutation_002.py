"""七模型主评测完成后才执行固定候选逆序，不并发GPU。"""
import json,os,subprocess,time
from pathlib import Path
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];O=R/'json/real_graph_permutation_jobs_002';PY='/root/miniconda3/envs/conceptgraph/bin/python';S=R/'source/real_graph_permutation_002.py';J=R/'json/real_graph_gpu_series_003'
def main():
 O.mkdir(exist_ok=False);env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',PYTHONPATH='/root/autodl-tmp/lizichong');h=digest(S);jobs=[]
 def run(args):
  assert digest(S)==h;name='prepare' if args==['--prepare'] else 'summary' if args==['--summarize'] else args[-1];log=O/(name+'.log')
  with log.open('x') as f:p=subprocess.run([PY,str(S)]+args,env=env,stdout=f,stderr=subprocess.STDOUT)
  jobs.append(dict(args=args,returncode=p.returncode));dump(O/f'job_{len(jobs):02d}.json',jobs[-1])
  if p.returncode:raise RuntimeError('permutation request failed; prefix preserved: '+str(log))
 # 选择和源码先冻结，不根据逆序成绩再挑样本。
 run(['--prepare']);dump(O/'release.json',dict(script_sha256=h,wait_for=str(J/'summary.json'),max_new_GPU_requests=126,goal_complete=False))
 while not (J/'summary.json').exists():
  if (J/'failure.json').exists():raise RuntimeError('upstream main comparison failed; no permutation GPU started')
  time.sleep(20)
 for model in ['base','old_adapter','historical_three_task_seed17','oracle_graph_seed17','real_graph_seed17','real_graph_seed29','real_graph_seed43']:run(['--variant',model])
 run(['--summarize']);dump(O/'summary.json',dict(jobs=jobs,new_actual_GPU_requests=126,goal_complete=False))
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  import traceback
  dump(O/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),goal_complete=False));raise
