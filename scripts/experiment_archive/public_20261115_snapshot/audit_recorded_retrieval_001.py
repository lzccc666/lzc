"""重核历史真实前端的四方法检索结果，新增统计而不冒充重新推理。"""
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import OUTPUT as OLD,available_development,digest,dump
R=Path(__file__).resolve().parents[1];S=OLD/'main_experiments/causal_retrieval_full_001';O=R/'method_comparison/recorded_retrieval_audit_001'
def metric(rows):
 p=[r for r in rows if r['positive']];n=[r for r in rows if not r['positive']]
 return dict(positive_target_accuracy=sum(bool(r['selected_correct']) for r in p)/len(p),no_match_false_selection_rate=sum(bool(r['false_selection_on_no_match']) for r in n)/len(n))
def main():
 O.mkdir(exist_ok=False);rows=list(map(json.loads,(S/'queries.jsonl').open()));official=json.loads((S/'report.json').read_text());by_scene={r['scene']:r for r in available_development()};methods=sorted({r['method'] for r in rows});ids={m:{(r['scene'],r['frame_idx'],r['query_index']) for r in rows if r['method']==m} for m in methods};assert all(v==ids[methods[0]] for v in ids.values());assert len(ids[methods[0]])==1115
 for r in rows:
  assert r['gt_in_inference'] is False and r['actual_arrival_evaluated'] is False
  if r['positive']:assert r['selected_correct']==(r['evaluation_identity'] in r['acceptable_instances'])
  else:assert r['false_selection_on_no_match']==r['selected']
 logs=sorted({by_scene[r['scene']]['log_token'] for r in rows});draw=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)));table=[];counts=[]
 for method in methods:
  rr=[r for r in rows if r['method']==method];c=dict(queries=len(rr),positive_queries=sum(r['positive'] for r in rr),negative_queries=sum(not r['positive'] for r in rr),positive_selected_correct=sum(bool(r['selected_correct']) for r in rr),no_match_false_selection=sum(bool(r['false_selection_on_no_match']) for r in rr));old=official['methods'][method]
  for k,v in c.items():assert old[k]==v,(method,k)
  counts.append(dict(method=method,**c));bylog={l:[r for r in rr if by_scene[r['scene']]['log_token']==l] for l in logs};boots=defaultdict(list)
  for choice in draw:
   z=metric([r for i in choice for r in bylog[logs[i]]])
   for k,v in z.items():boots[k].append(v)
  for k,v in metric(rr).items():
   lo,hi=np.percentile(boots[k],[2.5,97.5]);table.append(dict(method=method,metric=k,value=v,ci95_low=lo,ci95_high=hi,scenes=len(by_scene),logs=len(logs),source_frames=official['frames'],query_frames=len({(r['scene'],r['frame_idx']) for r in rr}),**c,new_GPU_calls=0,new_text_encoder_calls=0,independent_per_method_time_s=None))
 for name,rr in [('comparison.csv',table),('counts.csv',counts)]:
  with (O/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
 dump(O/'report.json',dict(original_source=str(S),source_sha256={str(p):digest(p) for p in [S/'queries.jsonl',S/'release.json',S/'report.json',S/'actual_clip_text_embeddings.npz',Path(__file__)]},verified_all_four_methods_same_queries=True,same_sensor_frontend=True,candidate_representation='当前检测crop/历史稀疏格/增量对象图，表示不同带来的候选差异属于架构；非强制相同节点集合的Qwen公平试验',new_GPU_calls=0,new_encoder_calls=0,bootstrap=dict(repetitions=4000,seed=17,unit='log',units=len(logs)),scope='真实公开记录数据上的开发检索诊断，非新封存测试、无车辆执行、不是完整VLMaps复现',limits=['对象身份来自官方类别和中心门限关联，含unknown；没有逐实例人工核验','稀疏CLIP格是明确受控简化，无LSeg密集像素特征/完整原三维建图器/原规划器','方法任务覆盖不同：简单类别规则与CLIP不提供可靠运动状态和拒答，不伪称复杂能力相同','历史逐方法独立耗时无法从总耗时拆分，保留空值','不与新135条标注图语义诊断混合分母或宣称端到端Qwen获益'],goal_complete=False))
 print(json.dumps(counts,ensure_ascii=False))
if __name__=='__main__':main()
