"""相同真实感知候选的七模型及三种子五开关；普通编号提议、证据授权、真实锁定/到达分开。"""
import csv,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import dump,digest
from real_graph_cooperation_001 import choose,complete_geometry_rule,categories,CONDITIONS
from real_graph_contract_003 import reliable
from semantic_target_only_audit_001 import parse_target
R=Path(__file__).resolve().parents[1];Q=R/'qwen_adapter_fair_comparison';D=Q/'real_graph_dataset_003';G=Q/'real_graph_fair_gpu_001';O=Q/'real_graph_decisions_001'
def metric(rr):
 normal=[r for r in rr if r['normal_positive']];represented=[r for r in normal if r['candidate_covered']];negative=[r for r in rr if r['kind']=='no_match'];faults={k:[r for r in rr if r['kind']==k] for k in ['insufficient_evidence','stale_information','conflicting_information']}
 ratio=lambda numerator,denominator:numerator/denominator if denominator else None
 v=dict(task_accuracy=ratio(sum(r['task_correct'] for r in rr),len(rr)),target_proposal_accuracy_all_positive=ratio(sum(r['task_correct'] for r in normal),len(normal)),target_proposal_accuracy_given_candidate_covered=ratio(sum(r['task_correct'] for r in represented),len(represented)),no_match_false_proposal_rate=ratio(sum(r['proposed'] is not None for r in negative),len(negative)),current_evidence_supported_correct_proposals=ratio(sum(r['proposal_correct'] and r['authorized'] is not None for r in normal),len(normal)),strict_output_validity=ratio(sum(r['valid'] for r in rr),len(rr)))
 for kind,items in faults.items():v[kind+'_refusal_rate']=ratio(sum(r['valid'] and r['answer'] is None for r in items),len(items))
 return v
def main():
 rows=list(map(json.loads,(D/'development_holdout.jsonl').open()));release=json.loads((G/'release.json').read_text());assert len(rows)==108 and digest(D/'development_holdout.jsonl')==release['input_sha256'];models=list(release['variants']);raw={m:{r['sample_id']:r for r in map(json.loads,(G/m/'raw_records.jsonl').open())} for m in models};assert all((G/m/'report.json').exists() and set(x)=={r['sample_id'] for r in rows} for m,x in raw.items());O.mkdir(exist_ok=False);results=[];auxiliary=[]
 for row in rows:
  sid=row['sample_id'];normal=row['controlled_fault'] is None and row['official_positive'];decisions={}
  for model in models:
   a=raw[model][sid];selected=a['normalized_output']['selected_node_id'] if a['schema_valid'] else None;node=next((n for n in row['candidate_nodes'] if n['node_id']==selected),None);decisions[model]=dict(proposed_node_id=selected,authorized_node_id=selected if node is not None and reliable(node) else None,valid=a['schema_valid'],source='real_GPU_raw_model',explicit_authorization_query=bool(row['controlled_fault']),backend_actual_latency_s=a['latency_s'],authorization_is_uniform_postcheck_reference_only=True)
   valid,target=parse_target(a['raw_output'],row['candidate_ids']);auxiliary.append(dict(model=model,sample_id=sid,kind=row['query_kind'],log=row['log_token'],target_field_valid=valid,target_only_correct=bool(valid and target in row['acceptable_node_ids']),target=target,strict_correct=a['semantic_correct'],posthoc_auxiliary=True))
  for seed in [17,29,43]:
   for cond in CONDITIONS:
    t=time.perf_counter();a=choose(row['query_text'],row['candidate_nodes'],row['relation_metadata'],raw[f'real_graph_seed{seed}'][sid],cond);a['offline_CPU_decision_s']=time.perf_counter()-t;decisions[f'cooperation_seed{seed}_{cond}']=a
  t=time.perf_counter();a=complete_geometry_rule(row['query_text'],row['candidate_nodes'],row['relation_metadata']);a['offline_CPU_decision_s']=time.perf_counter()-t;decisions['restricted_complete_geometry_rule']=a
  requested=categories(row['query_text']);candidates=[n for n in row['candidate_nodes'] if not requested or n['category'] in requested];nearest=min(candidates,key=lambda n:(n['distance_m'],n['node_id'])) if candidates else None;decisions['class_nearest_proposal_only']=dict(proposed_node_id=nearest['node_id'] if nearest else None,authorized_node_id=None,valid=True,source='class_nearest_rule_no_authorization',explicit_authorization_query=False)
  for method,a in decisions.items():
   proposal=a['proposed_node_id'];authorized=a['authorized_node_id'];is_coop=method.startswith('cooperation') or method=='restricted_complete_geometry_rule';answer=authorized if is_coop and a.get('explicit_authorization_query') else proposal;correct=bool(a['valid'] and answer in row['acceptable_node_ids']);proposal_correct=bool(a['valid'] and proposal in row['acceptable_node_ids']);results.append(dict(method=method,sample_id=sid,log=row['log_token'],scene=row['scene_id'],frame=row['frame_idx'],kind=row['query_kind'],normal_positive=normal,candidate_covered=bool(row['candidate_target_coverage']),answer=answer,proposed=proposal,authorized=authorized,valid=a['valid'],task_correct=correct,proposal_correct=proposal_correct,decision=a,actual_lock_credit=False,actual_arrival_credit=False))
 for name,rr in [('decisions.jsonl',results),('target_only_auxiliary.jsonl',auxiliary)]:
  with (O/name).open('x') as f:
   for r in rr:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 logs=sorted({r['log_token'] for r in rows});draw=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)));table=[]
 for method in sorted({r['method'] for r in results}):
  rr=[r for r in results if r['method']==method];v=metric(rr);bylog={l:[r for r in rr if r['log']==l] for l in logs};b=defaultdict(list)
  for choice in draw:
   z=metric([r for i in choice for r in bylog[logs[i]]])
   for key,value in z.items():b[key].append(np.nan if value is None else value)
  for key,value in v.items():
   lo,hi=np.nanpercentile(b[key],[2.5,97.5]);table.append(dict(method=method,metric=key,value=value,ci95_low=lo,ci95_high=hi,total_queries=len(rr),normal_positive_queries=sum(r['normal_positive'] for r in rr),normal_candidate_covered=sum(r['normal_positive'] and r['candidate_covered'] for r in rr),no_match_queries=sum(r['kind']=='no_match' for r in rr),source_frames=12,scenes=2,logs=2,strict_correct=sum(r['task_correct'] for r in rr),new_GPU_calls_in_summary=0,actual_locks=0,actual_arrivals=0))
 aux_table=[]
 for model in models:
  rr=[r for r in auxiliary if r['model']==model];normal=[r for r in rr if r['kind'] in ['simple_category','complex_relation','negation','similar_instances','multiple_targets']];aux_table.append(dict(model=model,all_queries=len(rr),strict_correct=sum(r['strict_correct'] for r in rr),target_only_correct=sum(r['target_only_correct'] for r in rr),target_only_field_valid=sum(r['target_field_valid'] for r in rr),normal_positive_queries=len(normal),normal_positive_target_only_correct=sum(r['target_only_correct'] for r in normal),posthoc_same_rule_all_models=True))
 for name,rr in [('comparison.csv',table),('qwen_target_only_auxiliary.csv',aux_table)]:
  with (O/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
 dump(O/'report.json',dict(queries_per_method=108,ordinary_positive_proposals=60,perception_covered_positive_proposals=28,positive_proposal_upper_bound=28/60,source_frames=12,logs=2,scenes=2,models=models,actual_upstream_GPU_requests=sum(json.loads((G/m/'report.json').read_text())['new_GPU_requests'] for m in models),actual_offline_cooperation_decisions=15*108,new_GPU_calls_in_summary=0,conditions=CONDITIONS,source_sha256={str(p):digest(p) for p in [Path(__file__),R/'source/real_graph_cooperation_001.py',R/'source/real_graph_contract_003.py',D/'manifest.json',D/'development_holdout.jsonl',G/'release.json']},raw_output_sha256={m:digest(G/m/'raw_records.jsonl') for m in models},bootstrap=dict(repetitions=4000,seed=17,unit='log',units=2,warning='仅2开发日志，CI离散，不能作充分道路泛化或显著性证据'),scope='真实保存的因果RGB/LiDAR感知候选，受控语言与三个注入故障；普通编号提议和证据授权分开，非自然人工指令/封存整系统/真实锁定或到达',limitations=['正常有目标但正确候选缺失的查询保留且永远不能算选对；候选覆盖条件下准确率仅作辅助，主指标分母60不缩小','普通编号提议可包含动态目标，不能据此授权接近；合作模块授权另做静止/来源/版本/观测/时间/深度检查','三种故障为受控压力测试，不作真实道路故障发生率；删证据门闸在这些确定性标签上变差不能单独当创新证据','强规则覆盖该套受控语法，不冒充通用自然语言方法或外部论文复现','原始Qwen的authorized字段是统一事后门闸参考值，不冒称模型自己已经通过锁定或控制','合作CPU时间来自离线实际决策，后台GPU输出来自相同真实请求；不把查表重放延迟冒充在线端到端延迟','全量输出格式主指标与只看ID的后验辅助指标同时保留，无输出解析失败偷算拒答正确','训练/验证日志与此留出隔离，但此2日志已参与项目开发，不是整课题新封存测试'],goal_complete=False));print(json.dumps(aux_table,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
