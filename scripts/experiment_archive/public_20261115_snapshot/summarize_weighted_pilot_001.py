"""决策加权开发试验：全量/正目标/拒答分开，与全部冻结对照共同报告。"""
import csv,json,time
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import digest,dump
from real_graph_contract_003 import reliable
from real_graph_cooperation_001 import choose,CONDITIONS
from semantic_target_only_audit_001 import parse_target

R=Path(__file__).resolve().parents[1];Q=R/'qwen_adapter_fair_comparison';D=Q/'real_graph_dataset_003/development_holdout.jsonl'
G=Q/'real_graph_fair_gpu_001';P=Q/'real_graph_weighted_pilot_gpu_001';O=Q/'real_graph_weighted_pilot_analysis_001'

def save(name,rows):
 with (O/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def main():
 O.mkdir(exist_ok=False);rows=list(map(json.loads,D.open()));meta={r['sample_id']:r for r in rows};logs=sorted({r['log_token'] for r in rows});draw=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)))
 variants=list(json.loads((G/'release.json').read_text())['variants']);pilot='decision_weighted_pilot_seed17';variants.append(pilot)
 assert digest(D)==json.loads((P/'release.json').read_text())['input_sha256']
 records={};sources={str(D):digest(D)};models=[]
 for model in variants:
  directory=(P if model==pilot else G)/model;path=directory/'raw_records.jsonl';sources[str(path)]=digest(path)
  rr=list(map(json.loads,path.open()));assert len(rr)==108 and {r['sample_id'] for r in rr}==set(meta);records[model]={r['sample_id']:r for r in rr}
  report=json.loads((directory/'report.json').read_text());sources[str(directory/'report.json')]=digest(directory/'report.json')
  pos=[r for r in rows if r['controlled_fault'] is None and r['official_positive']];aux=[]
  for row in rows:
   a=records[model][row['sample_id']];valid,target=parse_target(a['raw_output'],row['candidate_ids']);aux.append(dict(sid=row['sample_id'],correct=bool(valid and target in row['acceptable_node_ids']),positive=row['controlled_fault'] is None and row['official_positive']))
  models.append(dict(model=model,model_role='single_seed_development_pilot' if model==pilot else 'frozen_formal_or_historical_comparator',strict_correct=sum(r['semantic_correct'] for r in rr),queries=108,normal_positive_correct=sum(records[model][r['sample_id']]['semantic_correct'] for r in pos),normal_positive_queries=60,format_valid=sum(r['schema_valid'] for r in rr),target_only_aux_correct=sum(r['correct'] for r in aux),target_only_aux_positive_correct=sum(r['correct'] for r in aux if r['positive']),mean_actual_GPU_latency_s=float(np.mean([r['latency_s'] for r in rr])),actual_GPU_requests=108,actual_locks=0,actual_arrivals=0))
 assert all(records[pilot][sid]['input_tokens']==records['base'][sid]['input_tokens'] for sid in meta)
 decisions=[]
 for row in rows:
  raw=records[pilot][row['sample_id']];selected=raw['normalized_output']['selected_node_id'] if raw['schema_valid'] else None;node=next((n for n in row['candidate_nodes'] if n['node_id']==selected),None)
  methods={pilot:dict(proposed_node_id=selected,authorized_node_id=selected if node and reliable(node) else None,valid=raw['schema_valid'],explicit_authorization_query=bool(row['controlled_fault']))}
  for condition in CONDITIONS:
   t=time.perf_counter();decision=choose(row['query_text'],row['candidate_nodes'],row['relation_metadata'],raw,condition);decision['offline_CPU_decision_s']=time.perf_counter()-t;methods['pilot_cooperation_'+condition]=decision
  for method,a in methods.items():
   proposal=a['proposed_node_id'];answer=a['authorized_node_id'] if method!=pilot and a.get('explicit_authorization_query') else proposal;node=next((n for n in row['candidate_nodes'] if n['node_id']==proposal),None)
   decisions.append(dict(method=method,sample_id=row['sample_id'],kind=row['query_kind'],log=row['log_token'],normal_positive=row['controlled_fault'] is None and row['official_positive'],valid=a['valid'],answer=answer,proposed=proposal,authorized=a['authorized_node_id'],task_correct=bool(a['valid'] and answer in row['acceptable_node_ids']),independently_supported_correct_proposal=bool(a['valid'] and proposal in row['acceptable_node_ids'] and node and reliable(node)),decision=a,actual_lock_credit=False,actual_arrival_credit=False))
 with (O/'pilot_decisions.jsonl').open('x') as f:
  for row in decisions:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
 table=[]
 for method in sorted({r['method'] for r in decisions}):
  rr=[r for r in decisions if r['method']==method]
  for scope in ['all','ordinary_positive']+sorted({r['kind'] for r in rr}):
   selected=[r for r in rr if scope=='all' or (scope=='ordinary_positive' and r['normal_positive']) or r['kind']==scope]
   a=np.asarray([[sum(r['task_correct'] for r in selected if r['log']==log),sum(r['log']==log for r in selected)] for log in logs]);b=a[draw].sum(1);v=np.divide(b[:,0],b[:,1],out=np.full(len(draw),np.nan),where=b[:,1]>0);low,high=np.nanpercentile(v,[2.5,97.5]);num=sum(r['task_correct'] for r in selected)
   table.append(dict(method=method,scope=scope,correct=num,denominator=len(selected),accuracy=num/len(selected),ci95_low=float(low),ci95_high=float(high),logs=len(logs),actual_locks=0,actual_arrivals=0))
 paired=[]
 for model in variants[:-1]:
  for scope in ['all','ordinary_positive']:
   ids=[sid for sid,row in meta.items() if scope=='all' or row['controlled_fault'] is None and row['official_positive']]
   a=np.asarray([[sum(int(records[pilot][sid]['semantic_correct'])-int(records[model][sid]['semantic_correct']) for sid in ids if meta[sid]['log_token']==log),sum(meta[sid]['log_token']==log for sid in ids)] for log in logs]);b=a[draw].sum(1);v=np.divide(b[:,0],b[:,1],out=np.full(len(draw),np.nan),where=b[:,1]>0);low,high=np.nanpercentile(v,[2.5,97.5]);paired.append(dict(pilot=pilot,reference=model,scope=scope,denominator=len(ids),absolute_delta=float(a[:,0].sum()/len(ids)),paired_log_ci95_low=float(low),paired_log_ci95_high=float(high),logs=len(logs)))
 save('all_eight_models.csv',models);save('pilot_strata_and_cooperation.csv',table);save('pilot_paired_differences.csv',paired)
 dump(O/'report.json',dict(sources_sha256={**sources,str(Path(__file__)):digest(__file__)},new_actual_GPU_eval_requests=108,new_actual_GPU_eval_requests_in_analysis=0,new_actual_offline_cooperation_decisions=540,queries_per_method=108,ordinary_positive_queries=60,positive_candidate_covered=28,unique_frames=12,scenes=2,logs=2,same108samples_same_input_token_counts=True,pilot_seed=17,pilot_not_formal_three_seeds=True,pilot_retains_only_adapters_optimizer_resume=False,final_adapter_selection_only_fixed_validation18=True,new_sealed_test=False,models=models,bootstrap=dict(unit='log',units=2,repeats=4000,seed=17),limitations=['在看过正式开发负结果后提出的单种子试验，不构成独立确认','训练损失权重16/4，仅本方法变化；未改变任何基线输入/标签/任务/推理预算','全任务改善必须同时检查60条普通有目标查询；48条无匹配或故障拒答不能代表找到目标','五开关合作为保存真实GPU输出后的新CPU计算，没有在线锁定/物理到达信用','仅2个已开发日志，CI离散；正式新封存测试尚未开启','全部旧实验、三个正式种子及失败结果均保留'],goal_complete=False));print(json.dumps(models[-1],ensure_ascii=False))

if __name__=='__main__':main()
