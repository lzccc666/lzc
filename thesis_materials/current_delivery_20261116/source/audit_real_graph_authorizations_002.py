"""独立重查提议的实际证据；关掉门闸后被放行不等于证据充分。"""
import csv,json
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import digest,dump
from real_graph_contract_003 import reliable
R=Path(__file__).resolve().parents[1];Q=R/'qwen_adapter_fair_comparison'
D=Q/'real_graph_dataset_003/development_holdout.jsonl'
S=Q/'real_graph_decisions_001/decisions.jsonl'
O=Q/'real_graph_authorization_audit_002'

def main():
 O.mkdir(exist_ok=False);metadata={r['sample_id']:r for r in map(json.loads,D.open())}
 records=list(map(json.loads,S.open()));logs=sorted({r['log'] for r in records})
 draw=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)));output=[]
 for method in sorted({r['method'] for r in records}):
  rr=[r for r in records if r['method']==method and r['normal_positive']];data=[]
  for r in rr:
   row=metadata[r['sample_id']];node=next((n for n in row['candidate_nodes'] if n['node_id']==r['proposed']),None)
   supported=node is not None and reliable(node)
   data.append(dict(log=r['log'],correct_proposal_and_gate_granted=int(r['proposal_correct'] and r['authorized'] is not None),correct_proposal_with_independently_sufficient_evidence=int(r['proposal_correct'] and supported),gate_granted_but_evidence_insufficient=int(r['authorized'] is not None and not supported)))
  for metric in [k for k in data[0] if k!='log']:
   counts=np.asarray([[sum(r[metric] for r in data if r['log']==log),sum(r['log']==log for r in data)] for log in logs]);b=counts[draw].sum(1);v=np.divide(b[:,0],b[:,1],out=np.full(len(draw),np.nan),where=b[:,1]>0);lo,hi=np.nanpercentile(v,[2.5,97.5]);num=sum(r[metric] for r in data)
   output.append(dict(method=method,metric=metric,numerator=num,denominator=len(data),value=num/len(data),log_ci95_low=float(lo),log_ci95_high=float(hi),logs=len(logs),actual_locks=0,actual_arrivals=0))
 with (O/'comparison.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=output[0]);w.writeheader();w.writerows(output)
 dump(O/'report.json',dict(source_sha256={str(p):digest(p) for p in [Path(__file__),D,S,R/'source/real_graph_contract_003.py']},old_metric='current_evidence_supported_correct_proposals',old_metric_actual_meaning='proposal_correct AND authorized is not None：对关闭证据检查条件只代表策略放行，不代表实际证据可靠',corrected_meaning='correct_proposal_with_independently_sufficient_evidence统一对全部方法再次执行冻结reliable，不因去证据门闸而放宽',old_results_unchanged=True,strict_task_accuracy_changed=False,new_GPU_calls=0,new_offline_locked_targets=0,new_arrivals=0,bootstrap=dict(unit='log',units=2,repeats=4000,seed=17),goal_complete=False));print(json.dumps(dict(rows=len(output),audited_records=len(records))))

if __name__=='__main__':main()
