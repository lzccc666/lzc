"""统一的后验辅助审计：区分格式错误和编号选择错误；不替代冻结严格主指标。"""
import csv,json,math,re
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];G=R/'qwen_adapter_fair_comparison/official_graph_fair_gpu_001';D=R/'qwen_adapter_fair_comparison/official_graph_dataset_002';O=R/'qwen_adapter_fair_comparison/target_only_auxiliary_001'
def parse_target(text,allowed):
 s=text.strip()
 if s.startswith('```'):
  lines=s.splitlines()
  if lines[-1].strip()=='```':s='\n'.join(lines[1:-1]).strip()
 try:v=json.loads(s)
 except (ValueError,TypeError):return False,None
 if not isinstance(v,dict) or 'selected_node_id' not in v:return False,None
 target=v['selected_node_id']
 if target is None:return True,None
 if isinstance(target,bool) or not isinstance(target,int) or target not in allowed:return False,None
 return True,target
def main():
 O.mkdir(exist_ok=False);release=json.loads((G/'release.json').read_text());rows=list(map(json.loads,(D/'development_holdout.jsonl').open()));source={r['sample_id']:r for r in rows};records=[];table=[];hashes={}
 for model in release['variants']:
  p=G/model/'raw_records.jsonl';assert (p.parent/'report.json').exists();hashes[str(p)]=digest(p);raw=list(map(json.loads,p.open()));assert {r['sample_id'] for r in raw}==set(source) and len(raw)==len(source)
  for r in raw:
   row=source[r['sample_id']];valid,selected=parse_target(r['raw_output'],row['candidate_ids']);records.append(dict(model=model,sample_id=r['sample_id'],query_kind=row['query_kind'],log_token=row['log_token'],strict_format=r['schema_valid'],strict_correct=r['semantic_correct'],target_field_valid=valid,target_only_correct=bool(valid and selected in row['acceptable_node_ids']),explicit_abstention=bool(valid and selected is None),selected=selected,latency_s=r['latency_s']))
  for kind in ['all']+sorted({r['query_kind'] for r in rows}):
   rr=[r for r in records if r['model']==model and (kind=='all' or r['query_kind']==kind)];logs=sorted({r['log_token'] for r in rr});draw=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)));bylog={l:[r for r in rr if r['log_token']==l] for l in logs};b=[]
   for choice in draw:
    items=[r for i in choice for r in bylog[logs[i]]];b.append(sum(r['target_only_correct'] for r in items)/len(items))
   lo,hi=np.percentile(b,[2.5,97.5]);table.append(dict(model=model,stratum=kind,denominator=len(rr),strict_correct=sum(r['strict_correct'] for r in rr),strict_format=sum(r['strict_format'] for r in rr),target_only_correct=sum(r['target_only_correct'] for r in rr),target_field_valid=sum(r['target_field_valid'] for r in rr),target_only_accuracy=sum(r['target_only_correct'] for r in rr)/len(rr),ci95_low=lo,ci95_high=hi,mean_actual_GPU_latency_s=float(np.mean([r['latency_s'] for r in rr])),format_only_lost_correct=sum(r['target_only_correct'] and not r['strict_correct'] for r in rr)))
 with (O/'raw_auxiliary.jsonl').open('x') as f:
  for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 with (O/'comparison.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
 dump(O/'report.json',dict(purpose='后验统一辅助诊断，不能替代原严格主指标，也不能用于重选模型、改提示或重划测试',requests_per_model=len(rows),source_frames=len({(r['scene_id'],r['frame_idx']) for r in rows}),same_rule_all_models=True,unparsable_output_is_failure_not_correct_abstention=True,explicit_null_is_abstention=True,candidate_membership_required=True,bootstrap=dict(repetitions=4000,seed=17,unit='log',units=2,warning='两个开发日志，置信区间离散'),source_sha256={**hashes,str(Path(__file__)):digest(__file__),str(D/'development_holdout.jsonl'):digest(D/'development_holdout.jsonl')},new_GPU_calls=0,goal_complete=False))
 print(json.dumps([r for r in table if r['stratum']=='all'],ensure_ascii=False),flush=True)
if __name__=='__main__':main()
