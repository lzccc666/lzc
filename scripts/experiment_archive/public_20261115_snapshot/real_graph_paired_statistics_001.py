"""真实图模型同请求的配对差异；严格输出与统一编号辅助分开，CI仅按2日志。"""
import csv,json
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];Q=R/'qwen_adapter_fair_comparison';D=Q/'real_graph_dataset_003/development_holdout.jsonl';S=Q/'real_graph_decisions_001';O=Q/'real_graph_paired_statistics_001'
def main():
 O.mkdir(exist_ok=False);metadata={r['sample_id']:r for r in map(json.loads,D.open())};decisions=list(map(json.loads,(S/'decisions.jsonl').open()));aux=list(map(json.loads,(S/'target_only_auxiliary.jsonl').open()));logs=sorted({r['log_token'] for r in metadata.values()});draw=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)));models=sorted({r['model'] for r in aux});table=[]
 def compare(a,b,records,score,kind,scope):
  aa={r['sample_id']:r for r in records if r.get('method',r.get('model'))==a};bb={r['sample_id']:r for r in records if r.get('method',r.get('model'))==b};ids=sorted(set(aa)&set(bb));ids=[sid for sid in ids if scope=='all' or (metadata[sid]['controlled_fault'] is None and metadata[sid]['official_positive'])];assert ids
  counts=np.asarray([[sum(bool(aa[sid][score]) for sid in ids if metadata[sid]['log_token']==log),sum(bool(bb[sid][score]) for sid in ids if metadata[sid]['log_token']==log),sum(metadata[sid]['log_token']==log for sid in ids)] for log in logs]);total=counts.sum(0);sampled=counts[draw].sum(1);delta=(total[0]-total[1])/total[2];boots=np.divide(sampled[:,0]-sampled[:,1],sampled[:,2],out=np.full(len(draw),np.nan),where=sampled[:,2]>0);lo,hi=np.nanpercentile(boots,[2.5,97.5]);base=total[1]/total[2];table.append(dict(method=a,reference=b,score_kind=kind,scope=scope,method_correct=int(total[0]),reference_correct=int(total[1]),denominator=int(total[2]),absolute_delta=float(delta),relative_change=float(delta/base) if base else None,paired_ci95_low=float(lo),paired_ci95_high=float(hi),paired_logs=2,bootstrap_repeats=4000))
 for seed in [17,29,43]:
  for ref in ['base','old_adapter','oracle_graph_seed17']:
   for scope in ['all','ordinary_positive']:
    compare(f'real_graph_seed{seed}',ref,decisions,'task_correct','strict_full_output_primary',scope);compare(f'real_graph_seed{seed}',ref,aux,'target_only_correct','uniform_target_ID_only_posthoc_auxiliary',scope)
 for ref in ['restricted_complete_geometry_rule','class_nearest_proposal_only','cooperation_seed17_without_candidate_reranking','cooperation_seed17_without_rule_query_resolution','cooperation_seed17_without_attribute_check','cooperation_seed17_without_evidence_check']:
  for scope in ['all','ordinary_positive']:compare('cooperation_seed17_full',ref,decisions,'task_correct','offline_decision_contract',scope)
 with (O/'paired_differences.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
 strata=[]
 for method in sorted({r['method'] for r in decisions}):
  for kind in sorted({r['kind'] for r in decisions}):
   rr=[r for r in decisions if r['method']==method and r['kind']==kind];a=np.asarray([[sum(r['task_correct'] for r in rr if r['log']==log),sum(r['log']==log for r in rr)] for log in logs]);b=a[draw].sum(1);boot=np.divide(b[:,0],b[:,1],out=np.full(len(draw),np.nan),where=b[:,1]>0);lo,hi=np.nanpercentile(boot,[2.5,97.5]);strata.append(dict(method=method,stratum=kind,correct=sum(r['task_correct'] for r in rr),denominator=len(rr),accuracy=sum(r['task_correct'] for r in rr)/len(rr),ci95_low=float(lo),ci95_high=float(hi),logs=2))
 with (O/'stratified_accuracy.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=strata[0]);w.writeheader();w.writerows(strata)
 summary=[]
 for scope in ['all','ordinary_positive']:
  values=[]
  for seed in [17,29,43]:
   rows=[r for r in decisions if r['method']==f'real_graph_seed{seed}' and (scope=='all' or r['normal_positive'])];values.append(sum(r['task_correct'] for r in rows)/len(rows))
  summary.append(dict(scope=scope,seed_accuracy=values,mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1))))
 dump(O/'report.json',dict(paired_comparisons=len(table),seeds_summary=summary,source_sha256={str(p):digest(p) for p in [Path(__file__),D,S/'decisions.jsonl',S/'target_only_auxiliary.jsonl']},bootstrap=dict(repeats=4000,seed=17,unit='log',units=2),limitations=['仅2个已开发日志，不据此声称充分道路泛化或精确显著性','原始模型使用严格全输出为主、编号后验辅助为辅；合作与规则使用决策契约，不能把规则有效率叫作LLM格式率','正任务候选缺失仍保留失败，分母不缩到覆盖子集；拒答率和全任务正确率不能替代60条正目标选择率','三种子不是3个独立道路数据集，均值标准差与日志CI含义不同','GPU为单模型进程，期间有只读CPU审计；不是独占整机实时性能测试'],goal_complete=False));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
