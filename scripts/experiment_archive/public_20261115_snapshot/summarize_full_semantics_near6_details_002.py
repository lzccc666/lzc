"""冻结在全量GPU结束前的补充汇总：路由格式/拒答、逐日志计数及规则配对比较。"""
import csv,gzip,json
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];D=R/'nuscenes_ablation/full_semantics_near6_002';O=D/'detail_statistics_001'

def write(name,rows):
 with (O/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def main():
 report=json.loads((D/'run_report.json').read_text());release=json.loads((D/'release.json').read_text());assert report['actual_CPU_cooperation_decisions']==9048
 with gzip.open(D/'decisions.jsonl.gz','rt') as f:rows=list(map(json.loads,f))
 assert len(rows)==9048 and digest(D/'decisions.jsonl.gz')==report['raw_sha256']
 O.mkdir(exist_ok=False);usage=[];strata=[];bylog=[];paired=[]
 for role in ['all','train','validation','development_holdout']:
  rr=[r for r in rows if role=='all' or r['role']==role];logs=sorted({r['log'] for r in rr});draw=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)))
  for condition in release['conditions']:
   selected=[r for r in rr if r['condition']==condition];gpu=[r for r in selected if r['Qwen_called']];lat=np.array([r['GPU_latency_s'] for r in gpu]);fault=[r for r in selected if r['fault']];positive=[r for r in selected if r['positive']];negative=[r for r in selected if r['ordinary_no_match']]
   usage.append(dict(role=role,condition=condition,requests=len(selected),Qwen_calls=len(gpu),rule_without_Qwen=len(selected)-len(gpu),Qwen_valid_formats=sum(r['Qwen_schema_valid'] for r in gpu),Qwen_format_denominator=len(gpu),Qwen_format_rate=sum(r['Qwen_schema_valid'] for r in gpu)/len(gpu) if gpu else None,conditional_GPU_latency_mean_s=float(lat.mean()) if len(lat) else None,conditional_GPU_latency_p95_s=float(np.percentile(lat,95)) if len(lat) else None,conditional_GPU_latency_std_s=float(lat.std(ddof=1)) if len(lat)>1 else None,ordinary_positive_correct=sum(r['correct'] for r in positive),ordinary_positive_denominator=len(positive),controlled_fault_refusals=sum(r['answer'] is None for r in fault),controlled_fault_denominator=len(fault),ordinary_no_match_false_proposals=sum(r['no_match_false_proposal'] for r in negative),ordinary_no_match_denominator=len(negative)))
   for kind in release['strata']:
    zz=[r for r in selected if r['kind']==kind]
    if zz:strata.append(dict(role=role,condition=condition,kind=kind,requests=len(zz),ordinary_positive=sum(r['positive'] for r in zz),ordinary_no_match=sum(r['ordinary_no_match'] for r in zz),controlled_fault=sum(r['fault'] for r in zz),Qwen_calls=sum(r['Qwen_called'] for r in zz),correct=sum(r['correct'] for r in zz),strong_rule_correct=sum(r['strong_rule_correct'] for r in zz),refusals=sum(r['answer'] is None for r in zz)))
   for log in logs:
    zz=[r for r in selected if r['log']==log];pp=[r for r in zz if r['positive']];bylog.append(dict(role=role,condition=condition,log=log,requests=len(zz),correct=sum(r['correct'] for r in zz),strong_rule_correct=sum(r['strong_rule_correct'] for r in zz),ordinary_positive=len(pp),ordinary_positive_correct=sum(r['correct'] for r in pp),strong_rule_positive_correct=sum(r['strong_rule_correct'] for r in pp),positive_candidate_covered=sum(r['covered'] for r in pp)))
   for scope in ['all','ordinary_positive','complex_relation','negation','similar_instances','multiple_targets']:
    zz=[r for r in selected if scope=='all' or scope=='ordinary_positive' and r['positive'] or r['kind']==scope]
    if not zz:continue
    for reference in ['strong_rule_correct','nearest_correct']:
     a=np.array([[sum(int(r['correct'])-int(r[reference]) for r in zz if r['log']==log),sum(r['log']==log for r in zz)] for log in logs]);b=a[draw].sum(1);v=np.divide(b[:,0],b[:,1],out=np.full(len(draw),np.nan),where=b[:,1]>0);finite=v[np.isfinite(v)];low,high=np.percentile(finite,[2.5,97.5]);paired.append(dict(role=role,condition=condition,scope=scope,reference=reference,denominator=len(zz),cooperation_correct=sum(r['correct'] for r in zz),reference_correct=sum(r[reference] for r in zz),cooperation_minus_reference=float(a[:,0].sum()/len(zz)),ci95_low=float(low),ci95_high=float(high),logs=len(logs)))
 write('routing_and_latency.csv',usage);write('query_strata.csv',strata);write('log_counts.csv',bylog);write('paired_rule_comparison.csv',paired)
 dump(O/'report.json',dict(actual_CPU_cooperation_decisions=9048,actual_GPU_calls=sum(r['Qwen_called'] for r in rows),actual_new_GPU_calls_here=0,source_sha256={str(Path(__file__)):digest(__file__),str(D/'run_report.json'):digest(D/'run_report.json'),str(D/'release.json'):digest(D/'release.json'),str(D/'decisions.jsonl.gz'):digest(D/'decisions.jsonl.gz')},output_hashes={p.name:digest(p) for p in O.glob('*.csv')},bootstrap=dict(unit='log',resamples=4000,seed=17),latency_scope='只计算实际路由至Qwen的GPU生成耗时；规则分支未调用模型，没有把其正确率作模型格式率。不是完整链路延迟或实时吞吐。',metric_scope='ordinary_positive才是选对真实目标；故障/无匹配正确拒答另列。controlled fault为人工可复现注入，非人工自然指令/颜色真值。',comparison_scope='相同输入的完整几何规则与类别最近同时汇总；最近规则无拒答能力，不用混合故障总体优势宣称语言模型选目标提升。含训练/验证日志，开发留出2日志另列，仍非新封存。',goal_complete=False))
 print('routing, latency, strata, log counts and paired rule comparison complete',flush=True)

if __name__=='__main__':main()
