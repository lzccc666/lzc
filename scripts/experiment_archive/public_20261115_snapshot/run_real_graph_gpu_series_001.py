"""等待官方CARLA批次释放GPU，再实际训练三种子与同输入七模型推理；全过程不并发训练。"""
import datetime,json,os,signal,subprocess,time
from pathlib import Path
from conceptgraph.completion.stage9 import digest,dump
R=Path(__file__).resolve().parents[1];O=R/'json/real_graph_gpu_series_001';PY='/root/miniconda3/envs/conceptgraph/bin/python'
def size():
 return sum(p.lstat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
def main():
 O.mkdir(exist_ok=False);source=[R/'source'/n for n in ['train_real_graph_adapter_003.py','evaluate_real_graph_models_001.py','real_graph_contract_003.py','build_real_graph_dataset_003.py']];D=R/'qwen_adapter_fair_comparison/real_graph_dataset_003';audit=json.loads((D/'independent_audit_003.json').read_text());assert audit['all_within_2048'];release=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),dataset_manifest_sha256=digest(D/'manifest.json'),independent_audit_sha256=digest(D/'independent_audit_003.json'),contract_migration_sha256=digest(D/'contract_migration_001.json'),seeds=[17,29,43],max_steps=240,gradient_accumulation=4,learning_rate=1e-4,lora_r=8,lora_alpha=16,lora_dropout=.05,lora_modules=['q_proj','v_proj'],max_input_tokens=2048,max_new_tokens=256,training_query_strata_uniform=True,ordinary_covered_proposal_sampling_probability=.75,validation_rule='每层固定hash顺序，先取1条非null目标(若有)再取另一条，共18；只按此验证选step，不按留出选种子',evaluation_models=['base','old_adapter','historical_three_task_seed17','oracle_graph_seed17','real_graph_seed17','real_graph_seed29','real_graph_seed43'],evaluation_shared_data='development_holdout.jsonl; all108 rows; same prompt/candidates/token budget; no truncation or cache',model_seed_for_future_live_integration=17,live_model_selection='seed17预先固定，不以本次留出挑最好种子',ordinary_proposal_is_not_actuation_authorization=True,source_sha256={str(p):digest(p) for p in source},waiting_for_batch='json/carla_ablation_jobs_003/summary.json',goal_complete=False);dump(O/'release.json',release)
 while not (R/'json/carla_ablation_jobs_003/summary.json').exists():
  print(json.dumps(dict(waiting_for_CARLA_to_release_GPU=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())),flush=True);time.sleep(30)
 batch=json.loads((R/'json/carla_ablation_jobs_003/summary.json').read_text());dump(O/'prior_batch_receipt.json',dict(completed=batch['completed_subprocesses'],expected=batch['expected'],processing_succeeded=batch['all_processes_succeeded'],note='process completed不等于arrival或各限值通过，独立指标另看'))
 pid=28195
 try:
  cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
  if '/autodl-fs/data/third_party/carla_0.9.16/install/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping' not in cmd:raise RuntimeError('server PID mismatch; refuse to signal')
  os.kill(pid,signal.SIGTERM);dump(O/'server_shutdown.json',dict(pid=pid,verified_command=cmd,signal='SIGTERM',reason='official batch finished; exclusive training/evaluation GPU'))
  for _ in range(30):
   if not Path(f'/proc/{pid}').exists():break
   time.sleep(.5)
 except FileNotFoundError:dump(O/'server_shutdown.json',dict(pid=pid,already_exited=True))
 import shutil
 if shutil.disk_usage(R).free<20*2**30 or size()>9.6*2**30:raise OSError('20GiB free / 10GiB project output reserve not satisfied; no training started')
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',PYTHONPATH='/root/autodl-tmp/lizichong');jobs=[]
 def run(name,args,log):
  assert all(digest(p)==h for p,h in release['source_sha256'].items()),'frozen source changed'
  t=time.monotonic()
  with log.open('x') as f:p=subprocess.run([PY,str(R/'source'/name)]+args,env=env,stdout=f,stderr=subprocess.STDOUT)
  record=dict(script=name,args=args,returncode=p.returncode,wall_s=time.monotonic()-t,log=str(log));jobs.append(record);dump(O/f'job_{len(jobs):02d}.json',record);print(json.dumps(record),flush=True)
  if p.returncode:raise RuntimeError('subprocess failed; retained raw prefix, see '+str(log))
 for seed in release['seeds']:
  run('train_real_graph_adapter_003.py',['--dataset','real_graph_dataset_003','--run-id',f'real_graph_seed{seed}_001','--seed',str(seed),'--max-steps','240','--eval-steps','80','--grad-accum','4'],R/f'qwen_adapter_fair_comparison/train_real_graph_seed{seed}_001.log')
 run('evaluate_real_graph_models_001.py',['--prepare'],R/'qwen_adapter_fair_comparison/real_graph_fair_prepare_001.log')
 for model in release['evaluation_models']:run('evaluate_real_graph_models_001.py',['--variant',model],R/f'qwen_adapter_fair_comparison/real_graph_fair_{model}_001.log')
 dump(O/'summary.json',dict(jobs=jobs,seeds=3,expected_real_GPU_requests=7*108,source_sha256=digest(__file__),goal_complete=False))
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  import traceback
  dump(O/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),goal_complete=False));raise
