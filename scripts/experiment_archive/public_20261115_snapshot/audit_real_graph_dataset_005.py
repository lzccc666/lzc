"""实际感知训练准备审计：原输入哈希、官方实例隔离、完整token预算和提议/授权分母。"""
import ctypes
ctypes.CDLL('/root/miniconda3/envs/conceptgraph/lib/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
import json,sys
from pathlib import Path
from collections import Counter
import numpy as np
from transformers import AutoProcessor
from conceptgraph.completion.stage9 import PROJECT,digest,dump
from real_graph_contract_003 import audit_rows
from train_real_graph_adapter_003 import inputs_for
from real_graph_contract_003 import reliable,build_completion_prompt
R=Path(__file__).resolve().parents[1];D=R/'qwen_adapter_fair_comparison/real_graph_dataset_003'
def main():
 manifest=json.loads((D/'manifest.json').read_text());rows={role:list(map(json.loads,(D/f'{role}.jsonl').open())) for role in ['train','validation','development_holdout']};lengths=[];processor=AutoProcessor.from_pretrained(PROJECT/'checkpoint/Qwen2-VL-2B-Instruct',local_files_only=True,trust_remote_code=True)
 for name,h in manifest['files'].items():assert digest(D/name)==h
 audit=audit_rows(rows['train']+rows['validation']);logs={role:{r['log_token'] for r in rr} for role,rr in rows.items()};instances={role:{v for r in rr for v in r['official_match_binding_evaluation_only'].values()} for role,rr in rows.items()};counts={}
 for a,b in [('train','validation'),('train','development_holdout'),('validation','development_holdout')]:assert not logs[a]&logs[b] and not instances[a]&instances[b],(a,b)
 for role,rr in rows.items():
  for row in rr:
   supported=[n['node_id'] for n in row['candidate_nodes'] if reliable(n)];binding=row['official_match_binding_evaluation_only'];gt=row['official_acceptable_instances'];available=[n['node_id'] for n in row['candidate_nodes'] if binding.get(str(n['node_id'])) in gt]
   assert row['candidate_target_coverage']==(bool(available) if gt else None)
   assert row['reliable_target_coverage']==(bool(set(available)&set(supported)) if gt else None)
   if row['controlled_fault'] is None and gt:assert set(row['acceptable_node_ids'])==set(available)
   else:assert row['acceptable_node_ids']==[None]
   assert len(row['candidate_ids'])<=8
   prompt=build_completion_prompt('candidate_rerank',row)
   assert 'official_acceptable_instances' not in prompt and 'official_match_binding_evaluation_only' not in prompt
   n=int(inputs_for(processor,row,training=False)['input_ids'].shape[-1]);from train_real_graph_adapter_003 import messages
   text=processor.apply_chat_template(messages(row,None,True),tokenize=False,add_generation_prompt=False);training_n=int(processor(text=[text],images=None,padding=False,return_tensors='pt')['input_ids'].shape[-1])
   lengths.append(dict(sample_id=row['sample_id'],split=role,prompt_tokens=n,train_tokens=training_n))
  normal=[r for r in rr if r['controlled_fault'] is None and r['official_positive']];counts[role]=dict(rows=len(rr),frames=len({(r['scene_id'],r['frame_idx']) for r in rr}),logs=len(logs[role]),scenes=len({r['scene_id'] for r in rr}),normal_positive=len(normal),normal_with_correct_candidate=sum(bool(r['candidate_target_coverage']) for r in normal),normal_with_supported_static_target=sum(bool(r['reliable_target_coverage']) for r in normal),supervision_null=sum(r['response']['no_match'] for r in rr),empty_candidate_rows=sum(not r['candidate_ids'] for r in rr),normal_proposal_perception_upper_bound=sum(bool(r['candidate_target_coverage']) for r in normal)/len(normal),strata=dict(Counter(r['query_kind'] for r in rr)))
 dump(D/'independent_audit_003.json',dict(counts=counts,train_validation_audit=audit,official_predicted_instance_binding_disjoint=True,log_splits_disjoint=True,max_prompt_tokens=max(r['prompt_tokens'] for r in lengths),max_train_tokens=max(r['train_tokens'] for r in lengths),all_within_2048=max(r['train_tokens'] for r in lengths)<=2048,truncation=False,lengths=lengths,source_sha256={str(p):digest(p) for p in [Path(__file__),R/'source/train_real_graph_adapter_003.py',R/'source/real_graph_contract_003.py',D/'manifest.json']},ordinary_proposal_is_not_safety_authorization=True,controlled_faults_separately_identified=True,new_GPU_calls=0,new_sealed_test_opened=False,goal_complete=False));print(json.dumps(counts,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
