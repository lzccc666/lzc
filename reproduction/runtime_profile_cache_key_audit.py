"""在真实cache.get处截获键后抛专用异常；无模型尝试、无伪造语义响应。"""
from pathlib import Path
from collections import Counter
import json
from conceptgraph.completion.stage9 import OUTPUT,PROJECT,digest,dump
from conceptgraph.vlm.qwen_cache import QwenDiskCache
from conceptgraph.vlm.qwen_semantic_delegate import QwenDelegateConfig,QwenSemanticDelegate

class KeyCaptured(Exception):pass
class CapturingCache(QwenDiskCache):
 def get(self,key):self.last=key;raise KeyCaptured()

root=OUTPUT/'performance/runtime_profile_cache_key_audit_001';root.mkdir(exist_ok=False)
planpath=OUTPUT/'performance/runtime_profile_protocol_002/plan.json';plan=json.loads(planpath.read_text())
old=json.loads((OUTPUT/'performance/runtime_profile_protocol_001/plan.json').read_text())
assert old['frontend_inputs']==plan['frontend_inputs'] and old['recorded_frames']==plan['recorded_frames']
records=[]
for variant,model in plan['semantic_variants'].items():
 cache=CapturingCache(root/variant/'empty_cache')
 delegate=QwenSemanticDelegate(QwenDelegateConfig(enabled=True,
  model_name_or_path=str(PROJECT/'checkpoint/Qwen2-VL-2B-Instruct'),adapter_path=model['adapter'],
  model_revision=plan['base_revision'],adapter_revision=model['weights_sha256'],device='cuda:0',
  prompt_profile='completion_v3',use_qwen_query_parse=True,use_qwen_object_attributes=True,use_qwen_rerank=True,
  max_new_tokens=256,max_images=1,torch_dtype='bfloat16',max_image_side=448,timeout_s=60),cache=cache)
 try:
  for r in plan['semantic_inputs']:
   try:
    if r['task']=='query_parse':delegate.parse_query(r['query_text'])
    elif r['task']=='object_attributes':delegate.describe_object([r['crop_path']],r['metadata'],object_version=r['crop_sha256'])
    else:delegate.rerank_candidates(r['query_text'],r['candidate_nodes'],r['relation_metadata'])
   except KeyCaptured:pass
   else:raise AssertionError('cache key was not intercepted before model')
   records.append(dict(variant=variant,task=r['task'],sample_id=r['sample_id'],actual_cache_key=cache.last))
  assert delegate.backend_call_count==0
 finally:delegate.close()
 subset=[r for r in records if r['variant']==variant]
 assert len({r['actual_cache_key'] for r in subset})==72
 assert Counter(r['task'] for r in subset)==Counter(query_parse=24,object_attributes=24,candidate_rerank=24)
report=dict(passed=True,plan_sha256=digest(planpath),records=records,models=3,unique_keys_per_model=72,
 actual_backend_attempts=0,actual_GPU_calls=0,semantic_responses_written=0,correctness_credit=False,
 earlier_CPU_frontend_and_recorded_inputs_unchanged=True,new_test_opened=False,goal_complete=False)
dump(root/'report.json',report)
print(json.dumps(dict(passed=True,keys=len(records),actual_GPU_calls=0,semantic_responses_written=0,goal_complete=False)))
