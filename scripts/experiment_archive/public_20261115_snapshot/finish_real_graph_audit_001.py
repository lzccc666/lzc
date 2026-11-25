"""真实训练/推理完成后核验全部检查点和同输入七模型，再执行预冻结离线合作评分。"""
import ctypes
ctypes.CDLL('/root/miniconda3/envs/conceptgraph/lib/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
import json,os,subprocess,time
from pathlib import Path
import torch
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];Q=R/'qwen_adapter_fair_comparison';O=Q/'real_graph_completion_audit_001';J=R/'json/real_graph_gpu_series_003'
def main():
 O.mkdir(exist_ok=False);begun=time.monotonic()
 while not (J/'summary.json').exists():
  if (J/'failure.json').exists():dump(O/'upstream_failed.json',dict(failure_sha256=digest(J/'failure.json'),raw_prefix_preserved=True,goal_complete=False));return
  if time.monotonic()-begun>5400:dump(O/'waiting_timeout.json',dict(upstream=str(J),resume='完成上游后用新run-id重启审计，不覆盖本记录',goal_complete=False));return
  time.sleep(20)
 registration=json.loads((Q/'real_graph_cooperation_registration_001.json').read_text())
 for p,h in registration['source_sha256'].items():assert digest(p)==h,'preregistered cooperation or scorer changed'
 training=[];sources={};val_ids=None
 for seed in [17,29,43]:
  d=Q/f'training/real_graph_seed{seed}_001';report=json.loads((d/'training_report.json').read_text());cfg=json.loads((d/'release.json').read_text());sources[str(d/'training_report.json')]=digest(d/'training_report.json');assert report['seed']==seed and not report['independent_test_used'] and not cfg['test_used'];assert report['best_checkpoint']==max(report['checkpoints'],key=lambda x:x['score'])
  if val_ids is None:val_ids=cfg['validation_ids']
  assert cfg['validation_ids']==val_ids and len(val_ids)==18 and cfg['grad_accum']==4 and cfg['learning_rate']==1e-4
  for checkpoint in report['checkpoints']:
   p=Path(checkpoint['path']);assert p.resolve().is_relative_to(d.resolve());assert digest(p/'adapter_model.safetensors')==checkpoint['adapter_sha256'];ref=json.loads((p/'processor_reference.json').read_text())
   for name,h in ref['files_sha256'].items():assert digest(name)==h
   state=torch.load(p/'trainer_state.pt',map_location='cpu');manifest=json.loads((p/'checkpoint_manifest.json').read_text());assert state['completed_steps']==manifest['completed_steps'];assert set(['optimizer','scheduler','rng_state','curve']).issubset(state);assert len(state['curve'])==state['completed_steps'];assert set(['python','numpy','torch_cpu','torch_cuda']).issubset(state['rng_state']);del state
  training.append(dict(seed=seed,steps=report['completed_steps'],best_checkpoint=report['best_checkpoint'],checkpoints=len(report['checkpoints']),processor_reference_verified=True,all_adapter_and_resume_states_verified=True))
 g=Q/'real_graph_fair_gpu_001';release=json.loads((g/'release.json').read_text());expected=set(release['sample_ids']);tokens={};models=[]
 for model in release['variants']:
  p=g/model/'raw_records.jsonl';raw=list(map(json.loads,p.open()));report=json.loads((g/model/'report.json').read_text());assert len(raw)==108 and {r['sample_id'] for r in raw}==expected;assert all(r['backend_success'] and not r['fallback_used'] and r['input_tokens']<=2048 for r in raw);assert report['new_GPU_requests']==108 and digest(p)==report['records_sha256'];current={r['sample_id']:r['input_tokens'] for r in raw}
  if tokens:assert tokens==current,'same request unexpectedly has different tokens'
  tokens=current;models.append(dict(model=model,actual_GPU_calls=108,strict_correct=report['correct']));sources[str(p)]=digest(p)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH='/root/autodl-tmp/lizichong')
 with (O/'cooperation_summary.log').open('x') as f:proc=subprocess.run(['/root/miniconda3/envs/conceptgraph/bin/python',str(R/'source/summarize_real_graph_models_001.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
 dump(O/'report.json',dict(training=training,models=models,actual_upstream_GPU_requests=7*108,same_input_token_count_each_sample=True,validation_sample_ids_same_all_seeds=True,cooperation_summary_returncode=proc.returncode,source_sha256=sources,evaluator_source_sha256=digest(__file__),cooperation_registration_sha256=digest(Q/'real_graph_cooperation_registration_001.json'),no_human_annotation_credit=True,no_lock_or_arrival_credit=True,whole_project_sealed_test=False,goal_complete=False));print(json.dumps(dict(GPU_requests=756,cooperation_summary_returncode=proc.returncode)),flush=True)
 if proc.returncode:raise RuntimeError('cooperation summary failed; original outputs preserved')
if __name__=='__main__':main()
