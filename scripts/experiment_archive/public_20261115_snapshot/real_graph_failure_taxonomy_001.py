"""按固定顺序分解普通正查询失败；候选缺失和模型判断失败分别计数。"""
import csv,json
from collections import Counter
from pathlib import Path
from conceptgraph.completion.stage9 import digest,dump
R=Path(__file__).resolve().parents[1];Q=R/'qwen_adapter_fair_comparison';D=Q/'real_graph_dataset_003'
G=Q/'real_graph_fair_gpu_001';P=Q/'real_graph_weighted_pilot_gpu_001';O=Q/'real_graph_failure_taxonomy_001'

def main():
 O.mkdir(exist_ok=False);metadata={r['sample_id']:r for r in map(json.loads,(D/'development_holdout.jsonl').open())};ids={sid for sid,r in metadata.items() if r['controlled_fault'] is None and r['official_positive']};assert len(ids)==60
 models=list(json.loads((G/'release.json').read_text())['variants'])+['decision_weighted_pilot_seed17'];table=[];details=[];sources={}
 for model in models:
  path=(P if model=='decision_weighted_pilot_seed17' else G)/model/'raw_records.jsonl';sources[str(path)]=digest(path);raw={r['sample_id']:r for r in map(json.loads,path.open())};counts=Counter()
  for sid in sorted(ids):
   row=metadata[sid];a=raw[sid]
   if not row['candidate_target_coverage']:category='correct_target_absent_from_fixed_candidate_pool'
   elif not a['schema_valid']:category='candidate_present_but_full_output_invalid'
   elif a['semantic_correct']:category='correct_target_proposed'
   elif a['normalized_output']['selected_node_id'] is None:category='candidate_present_but_model_refused'
   else:category='candidate_present_but_wrong_target_proposed'
   counts[category]+=1;details.append(dict(model=model,sample_id=sid,kind=row['query_kind'],source_scene=row['scene_id'],source_frame=row['frame_idx'],log=row['log_token'],category=category,candidate_covered=row['candidate_target_coverage'],output_valid=a['schema_valid'],strict_semantic_correct=a['semantic_correct']))
  assert sum(counts.values())==60
  for category in ['correct_target_absent_from_fixed_candidate_pool','candidate_present_but_full_output_invalid','correct_target_proposed','candidate_present_but_model_refused','candidate_present_but_wrong_target_proposed']:table.append(dict(model=model,category=category,count=counts[category],denominator=60,fraction=counts[category]/60))
 for name,rr in [('counts.csv',table),('sample_index.csv',details)]:
  with (O/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
 split=[]
 for role in ['train','validation','development_holdout']:
  path=D/(role+'.jsonl');sources[str(path)]=digest(path);rr=list(map(json.loads,path.open()));normal=[r for r in rr if r['controlled_fault'] is None and r['official_positive']]
  split.append(dict(split=role,rows=len(rr),ordinary_positive=len(normal),positive_candidate_covered=sum(bool(r['candidate_target_coverage']) for r in normal),positive_current_evidence_supported=sum(bool(r['reliable_target_coverage']) for r in normal),null_response_labels=sum(r['response']['selected_node_id'] is None for r in rr),source_frames=len({(r['scene_id'],r['frame_idx']) for r in rr}),scenes=len({r['scene_id'] for r in rr}),logs=len({r['log_token'] for r in rr})))
 dump(O/'report.json',dict(source_sha256={**sources,str(Path(__file__)):digest(__file__)},counts=table,split_diagnosis=split,partition_order=['正查询候选缺失优先','有候选时格式失败','正确提议','有候选但拒答','有候选但选错'],limitations=['候选缺失优先归因，缺失时同时发生的输出问题不再重复计数；多原因交叉可查逐样本表','候选覆盖和标签来自冻结的官方类别/XY距离配对，不是人工确认的像素级真实身份','正查询主分母60不缩到28，覆盖上限不是已经实现的准确率','原始行的自动标签和受控故障不能替代真实人工颜色/细属性审核','这只是开发失败审计，不改标签、不调低基线、不授予锁定/到达信用'],actual_new_GPU_calls=0,new_sealed_test=False,goal_complete=False));print(json.dumps(split))

if __name__=='__main__':main()
