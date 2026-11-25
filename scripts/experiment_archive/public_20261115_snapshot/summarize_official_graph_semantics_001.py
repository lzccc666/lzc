"""冻结同请求GPU输出上的真实合作决策复算；不把离线复算冒充闭环语义增益。"""
import csv,hashlib,json,re,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import dump,digest
from official_graph_cooperation_001 import select,reliable,C,CONDITIONS
R=Path(__file__).resolve().parents[1];D=R/'qwen_adapter_fair_comparison/official_graph_dataset_002';G=R/'qwen_adapter_fair_comparison/official_graph_fair_gpu_001';O=R/'qwen_adapter_fair_comparison/official_graph_decisions_001'
QWEN=['base','old_adapter','historical_three_task_seed17','new_graph_seed17','new_graph_seed29','new_graph_seed43']
def rules(row,complete=True):
 q=row['query_text'];nodes=row['candidate_nodes'];requested={c for c,w in C.items() if w in q};nodes=[n for n in nodes if (not complete or reliable(n)) and (not requested or n['category'] in requested)];nodes=sorted(nodes,key=lambda n:(n['distance_m'],n['node_id']))
 if complete:
  anchor=re.search(r'编号(\d+)',q)
  if anchor:
   a=next(n for n in row['candidate_nodes'] if n['node_id']==int(anchor.group(1)));axis=1 if '左侧' in q or '右侧' in q else 0;positive='左侧' in q or '前方' in q;nodes=[n for n in nodes if n['node_id']!=a['node_id'] and abs(n['center_xy'][axis]-a['center_xy'][axis])>1 and (n['center_xy'][axis]>a['center_xy'][axis])==positive]
  if '第二' in q:nodes=nodes[1:]
 selected=nodes[0]['node_id'] if nodes else None
 return dict(selected_node_id=selected,no_match=selected is None,valid=True,source='restricted_grammar_rule' if complete else 'class_nearest_only')
def main():
 rows=list(map(json.loads,(D/'development_holdout.jsonl').open()));O.mkdir(exist_ok=False);raw={m:{r['sample_id']:r for r in map(json.loads,(G/m/'raw_records.jsonl').open())} for m in QWEN};ids={r['sample_id'] for r in rows};assert all(set(v)==ids for v in raw.values());release=json.loads((G/'release.json').read_text());assert release['input_sha256']==digest(D/'development_holdout.jsonl');results=[]
 for row in rows:
  sid=row['sample_id'];decisions={m:dict(selected_node_id=raw[m][sid]['normalized_output']['selected_node_id'] if raw[m][sid]['schema_valid'] else None,valid=raw[m][sid]['schema_valid'],source='real_GPU_Qwen',backend_latency_s=raw[m][sid]['latency_s']) for m in QWEN};decisions.update(restricted_complete_rule=rules(row),class_nearest_only=rules(row,False))
  for seed in [17,29,43]:
   for condition in CONDITIONS:
    t=time.monotonic();answer=select(row['query_text'],row['candidate_nodes'],row['relation_metadata'],raw[f'new_graph_seed{seed}'][sid],condition);answer['offline_decision_wall_s']=time.monotonic()-t;decisions[f'cooperation_seed{seed}_{condition}']=answer
  for method,answer in decisions.items():
   correct=bool(answer['valid'] and answer['selected_node_id'] in row['acceptable_node_ids']);results.append(dict(method=method,sample_id=sid,log=row['log_token'],scene=row['scene_id'],query_kind=row['query_kind'],correct=correct,expected_abstention=row['acceptable_node_ids']==[None],selected=answer['selected_node_id'],valid=answer['valid'],decision=answer))
 with (O/'decisions.jsonl').open('x') as f:
  for r in results:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 methods=sorted({r['method'] for r in results});kinds=sorted({r['query_kind'] for r in rows});logs=sorted({r['log_token'] for r in rows});rng=np.random.default_rng(17);draw=rng.integers(0,len(logs),(4000,len(logs)));table=[]
 for method in methods:
  for kind in ['all']+kinds:
   rr=[r for r in results if r['method']==method and (kind=='all' or r['query_kind']==kind)];count=len(rr);success=sum(r['correct'] for r in rr);bylog={l:[r for r in rr if r['log']==l] for l in logs};bootstrap=[]
   for choice in draw:
    items=[r for idx in choice for r in bylog[logs[idx]]];bootstrap.append(sum(r['correct'] for r in items)/len(items))
   low,high=np.percentile(bootstrap,[2.5,97.5]);table.append(dict(method=method,stratum=kind,correct=success,denominator=count,accuracy=success/count,ci95_low=low,ci95_high=high,format_valid=sum(r['valid'] for r in rr),false_selection_on_expected_abstention=sum(r['selected'] is not None for r in rr if r['expected_abstention']),expected_abstention_denominator=sum(r['expected_abstention'] for r in rr),source_logs=len(logs),new_GPU_calls_this_decision_summary=0))
 with (O/'comparison.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
 dump(O/'report.json',dict(methods=methods,counts=dict(rows=len(rows),source_frames=len({(r['scene_id'],r['frame_idx']) for r in rows}),logs=len(logs),scenes=len({r['scene_id'] for r in rows})),bootstrap=dict(unit='log',repetitions=4000,seed=17,warning='仅2日志，CI离散且不能代表自然道路泛化'),conditions=CONDITIONS,source_sha256={str(p):digest(p) for p in [Path(__file__),R/'source/official_graph_cooperation_001.py',D/'manifest.json',D/'development_holdout.jsonl',G/'release.json']},backend_hashes={m:digest(G/m/'raw_records.jsonl') for m in QWEN},new_GPU_calls=0,actual_offline_cooperation_decisions=15*len(rows),scope='同一135条标注图/构造语言诊断，非自然指令、视觉目标选择、真实异步队列或CARLA闭环语义增益',limitations=['受限完整语法规则是该受控语言的强基线，不冒充通用语言能力','颜色细属性无人工审核，不计成绩','所有模型输入、候选、完整提示、预算一致，未故意降低基础Qwen','三个新种子全保留，不选择性仅报最好种子','离线合作基于已保存的同请求真实GPU输出，不宣称新GPU调用或完整模块实测平均延迟'],goal_complete=False))
 print(json.dumps([r for r in table if r['stratum']=='all'],ensure_ascii=False),flush=True)
if __name__=='__main__':main()
