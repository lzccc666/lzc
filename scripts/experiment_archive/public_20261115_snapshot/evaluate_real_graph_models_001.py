"""同一108条真实感知候选开发诊断的真实GPU推理；普通提议和证据授权区分，非自然指令/封存整课题成绩。"""
import ctypes
# 固定使用本Conda环境已安装的C++运行库，避免torch先加载系统旧库后SciPy ABI不匹配。
ctypes.CDLL('/root/miniconda3/envs/conceptgraph/lib/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
import argparse,gc,hashlib,json,resource,time,traceback
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,torch
from transformers import AutoProcessor
from conceptgraph.completion.stage9 import PROJECT,OUTPUT as OLD,dump,digest
from scripts.train_qwen_lora_semantic_delegate_v2 import _load_model
from train_real_graph_adapter_003 import inputs_for
from real_graph_contract_003 import validate
R=Path(__file__).resolve().parents[1];D=R/'qwen_adapter_fair_comparison/real_graph_dataset_003';O=R/'qwen_adapter_fair_comparison/real_graph_fair_gpu_001';MODEL=PROJECT/'checkpoint/Qwen2-VL-2B-Instruct'
def variants():
 out={'base':None,'old_adapter':PROJECT/'outputs/project_v2/qwen_eval/finetune_v2/best_adapter'}
 p=OLD/'training/three_task_v2_formal_seed17_001/training_report.json';out['historical_three_task_seed17']=Path(json.loads(p.read_text())['best_checkpoint']['path'])
 p=R/'qwen_adapter_fair_comparison/training/graph_v2_seed17_001/training_report.json';out['oracle_graph_seed17']=Path(json.loads(p.read_text())['best_checkpoint']['path'])
 for seed in [17,29,43]:
  p=R/f'qwen_adapter_fair_comparison/training/real_graph_seed{seed}_001/training_report.json';out[f'real_graph_seed{seed}']=Path(json.loads(p.read_text())['best_checkpoint']['path'])
 return out

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--variant');args=parser.parse_args();rows=list(map(json.loads,(D/'development_holdout.jsonl').open()));V=variants()
 sources=[Path(__file__),R/'source/train_real_graph_adapter_003.py',R/'source/real_graph_contract_003.py',PROJECT/'scripts/train_qwen_lora_semantic_delegate_v2.py']
 release=dict(dataset_manifest_sha256=digest(D/'manifest.json'),input_sha256=digest(D/'development_holdout.jsonl'),sample_ids=[r['sample_id'] for r in rows],variants={n:dict(adapter_path=str(p) if p else None,adapter_sha256=digest(p/'adapter_model.safetensors') if p else None) for n,p in V.items()},same_candidate_evidence=True,proposal_vs_authorization_separate=True,empty_candidate_contract=True,prompt_sha256=digest(R/'source/real_graph_contract_003.py'),source_sha256={str(p):digest(p) for p in sources},cpp_runtime='/root/miniconda3/envs/conceptgraph/lib/libstdc++.so.6',max_new_tokens=256,max_input_tokens=2048,do_sample=False,torch_dtype='bfloat16',cache=False,seed=17,split='development_holdout_2_logs_not_whole_project_sealed_test',no_human_color_credit=True,perception_coupled_object_proposal_diagnostic=True,actual_lock_or_arrival_credit=False,goal_complete=False)
 if args.prepare:
  O.mkdir(parents=True,exist_ok=False);dump(O/'release.json',release);print('frozen',len(rows),len(V));return
 if json.loads((O/'release.json').read_text())!=release:raise ValueError('frozen release changed')
 if args.variant not in V:raise ValueError('unknown variant')
 case=O/args.variant;case.mkdir(exist_ok=False);begun=time.monotonic();adapter=V[args.variant];records=[]
 try:
  np.random.seed(17);torch.manual_seed(17);model=_load_model(MODEL,adapter,trainable=False);processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True);model.eval();model.config.use_cache=True;load_s=time.monotonic()-begun;torch.cuda.reset_peak_memory_stats()
  with (case/'raw_records.jsonl').open('x') as out:
   for i,row in enumerate(rows):
    x=inputs_for(processor,row).to(model.device)
    if x['input_ids'].shape[-1]>2048:raise ValueError('input too long; refuse truncation')
    torch.cuda.synchronize();t=time.monotonic()
    with torch.inference_mode():y=model.generate(**x,max_new_tokens=256,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=processor.tokenizer.eos_token_id)
    torch.cuda.synchronize();latency=time.monotonic()-t;raw=processor.tokenizer.decode(y[0,x['input_ids'].shape[-1]:],skip_special_tokens=True).strip();v=validate('candidate_rerank',raw,allowed_candidate_ids=row['candidate_ids']);pred=v.normalized_output if v.valid else None
    r=dict(sample_id=row['sample_id'],scene_id=row['scene_id'],log_token=row['log_token'],query_kind=row['query_kind'],input_tokens=int(x['input_ids'].shape[-1]),raw_output=raw,normalized_output=pred,schema_valid=v.valid,semantic_correct=bool(v.valid and pred['selected_node_id'] in row['acceptable_node_ids']),official_positive=row['official_positive'],ordinary_proposal_only=row['ordinary_proposal_only'],candidate_target_coverage=row['candidate_target_coverage'],controlled_fault=row['controlled_fault'],validation_errors=v.validation_errors,backend_success=True,fallback_used=False,latency_s=latency,cold_first_request=i==0);out.write(json.dumps(r,ensure_ascii=False)+'\n');out.flush();records.append(r)
    if (i+1)%15==0:print(json.dumps(dict(variant=args.variant,completed=i+1,correct=sum(r['semantic_correct'] for r in records))),flush=True)
  groups={k:[r for r in records if r['query_kind']==k] for k in sorted({r['query_kind'] for r in records})};summary={k:dict(correct=sum(r['semantic_correct'] for r in rr),denominator=len(rr),format_valid=sum(r['schema_valid'] for r in rr),accuracy=sum(r['semantic_correct'] for r in rr)/len(rr)) for k,rr in groups.items()};dump(case/'report.json',dict(variant=args.variant,count=len(records),correct=sum(r['semantic_correct'] for r in records),strata=summary,model_load_s=load_s,mean_latency_s=float(np.mean([r['latency_s'] for r in records])),warm_median_latency_s=float(np.median([r['latency_s'] for r in records[1:]])),peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,new_GPU_requests=len(records),records_sha256=digest(case/'raw_records.jsonl'),performance='同一GPU、每模型新进程；若同期运行训练或前端须另标共享负载，不据此宣称实时',goal_complete=False));print(json.dumps(dict(variant=args.variant,done=len(records))),flush=True)
 except BaseException as e:dump(case/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),records=len(records),goal_complete=False));raise
if __name__=='__main__':main()
