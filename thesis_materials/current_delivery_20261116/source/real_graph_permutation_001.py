"""相同节点ID的候选逆序开发压力测试：固定18条，七模型均真实推理；无性能筛选。"""
import ctypes
ctypes.CDLL('/root/miniconda3/envs/conceptgraph/lib/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
import argparse,copy,gc,hashlib,json,os,resource,shutil,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from transformers import AutoProcessor
from conceptgraph.completion.stage9 import PROJECT,dump,digest
from scripts.train_qwen_lora_semantic_delegate_v2 import _load_model
from evaluate_real_graph_models_002 import variants
from train_real_graph_adapter_004 import inputs_for
from real_graph_contract_003 import validate
from semantic_target_only_audit_001 import parse_target
R=Path(__file__).resolve().parents[1];Q=R/'qwen_adapter_fair_comparison';O=Q/'real_graph_permutation_001';D=Q/'real_graph_dataset_003/development_holdout.jsonl';G=Q/'real_graph_fair_gpu_001';MODEL=PROJECT/'checkpoint/Qwen2-VL-2B-Instruct'
def selection():
 rows=list(map(json.loads,D.open()));out=[]
 for kind in sorted({r['query_kind'] for r in rows}):out.extend(sorted([r for r in rows if r['query_kind']==kind],key=lambda r:hashlib.sha256(('permutation-fixed-v1|'+r['sample_id']).encode()).hexdigest())[:2])
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--variant');p.add_argument('--summarize',action='store_true');args=p.parse_args();rows=selection();V=variants();source=[Path(__file__),R/'source/train_real_graph_adapter_004.py',R/'source/real_graph_contract_003.py'];release=dict(sample_ids=[r['sample_id'] for r in rows],rows=18,strata=9,selection='每层按固定hash取2条；不看任何模型输出/候选覆盖/对错，开发压力诊断不是新封存测试',permutation='逆序candidate_nodes及candidate_ids，节点ID/属性/关系/标签不变；相同变换用于所有模型',variants={m:str(v) if v else None for m,v in V.items()},dataset_sha256=digest(D),source_sha256={str(x):digest(x) for x in source},max_input_tokens=2048,max_new_tokens=256,seed=17,do_sample=False,torch_dtype='bfloat16',new_GPU_requests=18*7,original_outputs_reused=True,whole_project_sealed_test=False,goal_complete=False)
 if args.prepare:O.mkdir(exist_ok=False);dump(O/'release.json',release);print('frozen18 rows x7 variants',flush=True);return
 assert json.loads((O/'release.json').read_text())==release
 if args.summarize:
  table=[]
  for model in V:
   raw=list(map(json.loads,(O/model/'raw_records.jsonl').open()));original={r['sample_id']:r for r in map(json.loads,(G/model/'raw_records.jsonl').open())};assert len(raw)==18
   for scope,rr in [('all',raw),('ordinary_positive',[x for x in raw if x['ordinary_positive']])]:
    if not rr:continue
    strict_consistent=0;aux_consistent=0;both_valid=0;aux_valid=0
    for x in rr:
     a=original[x['sample_id']];v=x['schema_valid'] and a['schema_valid'];both_valid+=bool(v);strict_consistent+=bool(v and x['normalized_output']['selected_node_id']==a['normalized_output']['selected_node_id']);row=next(r for r in rows if r['sample_id']==x['sample_id']);aok,aid=parse_target(a['raw_output'],row['candidate_ids']);bok,bid=parse_target(x['raw_output'],row['candidate_ids']);aux_valid+=bool(aok and bok);aux_consistent+=bool(aok and bok and aid==bid)
    table.append(dict(model=model,scope=scope,queries=len(rr),strict_both_format_valid=both_valid,strict_ID_consistent=strict_consistent,strict_consistency_rate=strict_consistent/len(rr),target_only_aux_both_valid=aux_valid,target_only_aux_consistent=aux_consistent,target_only_aux_consistency_rate=aux_consistent/len(rr),original_correct=sum(original[x['sample_id']]['semantic_correct'] for x in rr),reversed_correct=sum(x['semantic_correct'] for x in rr),new_GPU_requests=len(raw)))
  import csv
  with (O/'comparison.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
  dump(O/'report.json',dict(rows_per_model=18,models=7,new_actual_GPU_requests=126,original_requests_reused=126,new_independent_frames=0,results=table,scope='固定18条候选逆序的开发稳定性诊断；格式/编号后验辅助和普通正查询分开，不能把两次非法输出一致当成功',goal_complete=False));print(json.dumps(table,ensure_ascii=False),flush=True);return
 if args.variant not in V:raise ValueError('unknown variant')
 if shutil.disk_usage(R).free<20*2**30 or sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())+4*2**20>10*2**30:raise OSError('storage reserve insufficient; no new GPU request')
 case=O/args.variant;case.mkdir(exist_ok=False);torch.manual_seed(17);np.random.seed(17);model=_load_model(MODEL,V[args.variant],trainable=False);processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True);model.eval();torch.cuda.reset_peak_memory_stats();records=[]
 with (case/'raw_records.jsonl').open('x') as f:
  for row in rows:
   changed=copy.deepcopy(row);changed['candidate_nodes'].reverse();changed['candidate_ids'].reverse();x=inputs_for(processor,changed).to(model.device);assert x['input_ids'].shape[-1]<=2048;torch.cuda.synchronize();t=time.monotonic()
   with torch.inference_mode():y=model.generate(**x,max_new_tokens=256,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=processor.tokenizer.eos_token_id)
   torch.cuda.synchronize();seconds=time.monotonic()-t;raw=processor.tokenizer.decode(y[0,x['input_ids'].shape[-1]:],skip_special_tokens=True).strip();v=validate('candidate_rerank',raw,allowed_candidate_ids=row['candidate_ids']);out=v.normalized_output if v.valid else None;record=dict(sample_id=row['sample_id'],kind=row['query_kind'],ordinary_positive=row['controlled_fault'] is None and row['official_positive'],raw_output=raw,normalized_output=out,schema_valid=v.valid,semantic_correct=bool(v.valid and out['selected_node_id'] in row['acceptable_node_ids']),latency_s=seconds,input_tokens=int(x['input_ids'].shape[-1]),backend_success=True,fallback_used=False);f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();records.append(record)
 dump(case/'report.json',dict(model=args.variant,records=18,new_GPU_requests=18,mean_latency_s=float(np.mean([x['latency_s'] for x in records])),peak_GPU_mib=torch.cuda.max_memory_allocated()/2**20,peak_RSS_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,raw_sha256=digest(case/'raw_records.jsonl'),goal_complete=False));print(args.variant,'18 done',flush=True)
if __name__=='__main__':main()
