"""六模型、三种子、五语义开关的同分母交付表，主指标和后验辅助指标并列。"""
import csv,json
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];Q=R/'qwen_adapter_fair_comparison';O=R/'tables/semantic_summary_001'
def write_csv(name,rows):
 with (O/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 O.mkdir(parents=True,exist_ok=False);G=Q/'official_graph_fair_gpu_001';A=Q/'target_only_auxiliary_001';D=Q/'official_graph_decisions_001';release=json.loads((G/'release.json').read_text());aux={r['model']:r for r in csv.DictReader((A/'comparison.csv').open()) if r['stratum']=='all'};table=[];training=[]
 for model in release['variants']:
  rep=json.loads((G/model/'report.json').read_text());a=aux[model];table.append(dict(model=model,strict_correct=rep['correct'],denominator=rep['count'],strict_accuracy=rep['correct']/rep['count'],format_valid=int(a['strict_format']),format_validity=int(a['strict_format'])/rep['count'],target_only_correct=int(a['target_only_correct']),target_only_accuracy=float(a['target_only_accuracy']),target_only_is_posthoc_auxiliary=True,mean_GPU_latency_s=rep['mean_latency_s'],peak_GPU_mib=rep['peak_gpu_allocated_mib'],peak_process_RSS_mib=rep['peak_process_rss_mib'],new_GPU_requests=rep['new_GPU_requests'],source_frames=15,scenes=2,logs=2,raw_records=str(G/model/'raw_records.jsonl'),report_sha256=digest(G/model/'report.json')))
 for seed in [17,29,43]:
  p=Q/f'training/graph_v2_seed{seed}_001/training_report.json';z=json.loads(p.read_text());training.append(dict(seed=seed,training_report=str(p),training_report_sha256=digest(p),best_checkpoint=json.dumps(z['best_checkpoint'],ensure_ascii=False),note='只按训练内验证选步；保留三个种子，不按开发留出挑最好种子'))
 rows=list(map(json.loads,(D/'decisions.jsonl').open()));kinds=sorted({r['query_kind'] for r in rows});logs=sorted({r['log'] for r in rows});conditions=['full','without_rule_query_resolution','without_attribute_check','without_candidate_reranking','without_evidence_check'];summary=[];draw=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)))
 for cond in conditions:
  for kind in ['all']+kinds:
   perseed=[[r for r in rows if r['method']==f'cooperation_seed{s}_{cond}' and (kind=='all' or r['query_kind']==kind)] for s in [17,29,43]];values=np.array([np.mean([r['correct'] for r in rr]) for rr in perseed]);b=[]
   for choice in draw:
    permodel=[]
    for rr in perseed:
     sampled=[r for idx in choice for r in rr if r['log']==logs[idx]];permodel.append(np.mean([r['correct'] for r in sampled]))
    b.append(np.mean(permodel))
   lo,hi=np.percentile(b,[2.5,97.5]);summary.append(dict(condition=cond,stratum=kind,accuracy_mean_3seeds=float(values.mean()),accuracy_sample_std_3seeds=float(values.std(ddof=1)),log_cluster_ci95_low=lo,log_cluster_ci95_high=hi,denominator_per_seed=len(perseed[0]),number_seeds=3,number_unique_logs=2,number_unique_frames=15,scope='controlled_given_graph_development_diagnostic_only'))
 write_csv('qwen_six_models.csv',table);write_csv('training_three_seeds.csv',training);write_csv('semantic_ablation_three_seeds.csv',summary)
 inputs=[G/'release.json',A/'comparison.csv',D/'comparison.csv',D/'decisions.jsonl',Path(__file__)];dump(O/'report.json',dict(actual_new_GPU_requests=sum(r['new_GPU_requests'] for r in table),actual_offline_cooperation_decisions=2025,unique_requests_per_model=135,unique_source_frames=15,scenes=2,logs=2,independent_whole_project_sealed_test=False,source_sha256={str(p):digest(p) for p in inputs},positive_findings=['新Adapter三个种子在统一编号辅助正确率上73/135、73/135、74/135，基础/旧Adapter均30/135；不只格式差异','三个训练都实际完成240步并保存旧/新checkpoint'],negative_findings=['完整合作模块约75%至78%，固定顺序消融84.44%；Qwen复杂重排在此诊断降低结果','受限完整语法规则135/135，强于当前合作模块，不能称Qwen优于强规则','历史三任务Adapter跨新契约明显退化，严格4/135、辅助9/135；保留','该数据使用官方标注几何/构造的证据字段，不是实际感知/自然指令/人工颜色标注'],limitations=['全部方法输入、候选、任务、预算相同，未故意削弱基线','主指标为冻结严格契约正确率，编号指标为结果开启后的统一辅助审计','两个开发日志的聚类CI离散且较窄，不能作充分泛化或显著性证明','合作逻辑离线重放同请求GPU原始输出，不宣称异步实时端到端吞吐','颜色、细属性和歧义人工记录缺失，未授予此类信用'],goal_complete=False))
 (O/'结果说明.md').write_text('这组实验验证“给定正确对象图时能否按受控语言选对对象”。新训练的三个种子在目标编号上确有提升，但没有改善所有模块：完整合作低于固定顺序消融，且低于覆盖本受限语法的完整规则。主表与编号辅助指标同时保留，不能只挑正结果。\n\n135条请求来自15帧、2场景、2日志，同一帧九类查询不独立。官方几何、人工构造的证据协议字段不等于真实感知。训练540条、验证180条、开发留出135条；训练验证日志及官方实例与留出日志隔离，但这些日志已用于建图开发，不能称整课题封存测试。\n',encoding='utf-8')
 print(json.dumps(dict(GPU_requests=sum(r['new_GPU_requests'] for r in table),seeds=3,cooperation_decisions=2025,goal_complete=False)))
if __name__=='__main__':main()
