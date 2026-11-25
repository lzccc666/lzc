"""复杂语义CPU汇总补丁：开发查询元数据使用query_kind；不改GPU输入、模型、标签或推理预算。"""
import csv,gzip,json
import numpy as np
from conceptgraph.completion.stage9 import dump,digest
from real_graph_cooperation_001 import choose
from nuscenes_complex_semantics_001 import O,read_rows,CONDITIONS

def summarize():
 with gzip.open(O/'raw_GPU_records.jsonl.gz','rt') as f:raw=list(map(json.loads,f))
 rows=read_rows();by={(r['mapping_condition'],r['sample_id']):r for r in rows};assert len(raw)==648 and len({(a['mapping_condition'],a['sample_id']) for a in raw})==648;decisions=[]
 for a in raw:
  row=by[a['mapping_condition'],a['sample_id']];decision=choose(row['query_text'],row['candidate_nodes'],row['relation_metadata'],a);answer=decision['authorized_node_id'] if decision['explicit_authorization_query'] else decision['proposed_node_id'];decisions.append(dict(condition=row['mapping_condition'],sample_id=row['sample_id'],log=row['log_token'],kind=row['query_kind'],normal_positive=row['controlled_fault'] is None and row['official_positive'],candidate_covered=bool(row['candidate_target_coverage']),raw_correct=a['semantic_correct'],cooperation_correct=bool(answer in row['acceptable_node_ids']),raw_format_valid=a['schema_valid'],relation_count=len(row['relation_metadata']),proposed_node_id=decision['proposed_node_id'],authorized_node_id=decision['authorized_node_id'],decision=decision,latency_s=a['latency_s'],actual_lock_credit=False,actual_arrival_credit=False))
 with gzip.open(O/'cooperation_decisions.jsonl.gz','xt',compresslevel=6) as f:
  for d in decisions:f.write(json.dumps(d,ensure_ascii=False,separators=(',',':'))+'\n')
 logs=sorted({d['log'] for d in decisions});draw=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)));table=[];paired=[]
 for condition in CONDITIONS:
  for scope in ['all','ordinary_positive']+sorted({r['query_kind'] for r in rows}):
   rr=[r for r in decisions if r['condition']==condition and (scope=='all' or (scope=='ordinary_positive' and r['normal_positive']) or r['kind']==scope)]
   for metric in ['raw_correct','cooperation_correct','raw_format_valid','candidate_covered']:
    counts=np.asarray([[sum(r[metric] for r in rr if r['log']==l),sum(r['log']==l for r in rr)] for l in logs]);b=counts[draw].sum(1);v=np.divide(b[:,0],b[:,1],out=np.full(len(draw),np.nan),where=b[:,1]>0);lo,hi=np.nanpercentile(v,[2.5,97.5]);num=sum(r[metric] for r in rr);table.append(dict(condition=condition,scope=scope,metric=metric,numerator=num,denominator=len(rr),value=num/len(rr),ci95_low=float(lo),ci95_high=float(hi),logs=2,frames_each=12,scenes_each=2,actual_locks=0,actual_arrivals=0))
 for condition in CONDITIONS[1:]:
  for scope in ['all','ordinary_positive']:
   aa={r['sample_id']:r for r in decisions if r['condition']=='full' and (scope=='all' or r['normal_positive'])};bb={r['sample_id']:r for r in decisions if r['condition']==condition and (scope=='all' or r['normal_positive'])};ids=sorted(aa);assert set(aa)==set(bb)
   for metric in ['raw_correct','cooperation_correct']:
    count=np.asarray([[sum(int(aa[s][metric])-int(bb[s][metric]) for s in ids if aa[s]['log']==l),sum(aa[s]['log']==l for s in ids)] for l in logs]);b=count[draw].sum(1);v=np.divide(b[:,0],b[:,1],out=np.full(len(draw),np.nan),where=b[:,1]>0);lo,hi=np.nanpercentile(v,[2.5,97.5]);paired.append(dict(method='full',reference=condition,scope=scope,metric=metric,denominator=len(ids),absolute_delta=float(count[:,0].sum()/len(ids)),paired_log_ci95_low=float(lo),paired_log_ci95_high=float(hi)))
 for name,rr in [('metrics.csv',table),('paired_differences.csv',paired)]:
  with (O/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
 dump(O/'report.json',dict(actual_new_GPU_requests=648,actual_new_CPU_cooperation_decisions=648,conditions=6,requests_each=108,ordinary_positive_each=60,source_frames_each=12,source_scenes_each=2,source_logs_each=2,actual_new_mapper_calls=0,actual_maintained_edges_used=True,release_sha256=digest(O/'release.json'),inputs_sha256=digest(O/'inputs.jsonl.gz'),raw_sha256=digest(O/'raw_GPU_records.jsonl.gz'),cooperation_sha256=digest(O/'cooperation_decisions.jsonl.gz'),source_sha256=digest(__file__),bootstrap=dict(unit='log',units=2,repeats=4000,seed=17),limitations=['仅已固定108条开发子集，不等于38场景1508帧全量复杂语义验收','各建图候选真实变化；相同查询/官方目标，但非不同语言模型同候选公平比较表','候选ID/顺序按GT无关新协议生成，实际维护器边与此前候选坐标派生边输入分布不同，不能直接与旧108原始分数作因果差异','无关系条件仅去实际关系边且候选/标签逐条相同；模型仍能读取坐标，结果相同不能声称关系维护有贡献','三个故障层是明确注入压力测试，普通提议与静止授权分开','未进行新的跨帧锁定或控制，无实际到达信用；仅单个事前固定种子17','仅2个已开发日志，CI离散，非新封存道路测试'],new_sealed_test=False,goal_complete=False));print('648 GPU +648 CPU complete',flush=True)


if __name__=='__main__':summarize()
