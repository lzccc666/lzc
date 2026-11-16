"""真实感知语义实验可编辑表：正式三种子、单种子试验和负结果均标明。"""
import csv,html,json,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
from openpyxl import Workbook,load_workbook
from openpyxl.styles import Font,Alignment
from conceptgraph.completion.stage9 import dump,digest
from export_completed_tables_002 import p,table

R=Path(__file__).resolve().parents[1];Q=R/'qwen_adapter_fair_comparison';O=R/'tables/real_graph_office_001';sources={}
def read(path):
 sources[str(path)]=digest(path);return list(csv.DictReader(path.open()))
def percent(x):return f'{float(x)*100:.2f}%'
def main():
 O.mkdir(exist_ok=False);sections=[]
 rows=read(Q/'real_graph_weighted_pilot_analysis_001/all_eight_models.csv')
 sections.append(('真实感知图：七个冻结模型与一个开发试验',['模型','全任务正确/108','正目标正确/60','格式有效/108','编号辅助正确/108','GPU均秒'],[[r['model'],r['strict_correct'],r['normal_positive_correct'],r['format_valid'],r['target_only_aux_correct'],f"{float(r['mean_actual_GPU_latency_s']):.3f}"] for r in rows],'相同108请求、12帧、2场景、2已开发日志；60条普通正查询中28条含正确候选。最后一行为决策损失加权单种子开发试验，不构成独立三种子确认。全量提升主要来自拒答，不能替代找目标率。严格格式主指标和统一编号辅助同时保留。'))
 dpath=Q/'real_graph_decisions_001/decisions.jsonl';sources[str(dpath)]=digest(dpath);raw=list(map(json.loads,dpath.open()));logs=sorted({r['log'] for r in raw});draw=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)));stats=[]
 conditions=['full','without_rule_query_resolution','without_attribute_check','without_candidate_reranking','without_evidence_check']
 for condition in conditions:
  for scope in ['all','ordinary_positive']:
   values=[];boots=[]
   for seed in [17,29,43]:
    rr=[r for r in raw if r['method']==f'cooperation_seed{seed}_{condition}' and (scope=='all' or r['normal_positive'])];values.append(sum(r['task_correct'] for r in rr)/len(rr));a=np.asarray([[sum(r['task_correct'] for r in rr if r['log']==log),sum(r['log']==log for r in rr)] for log in logs]);b=a[draw].sum(1);boots.append(b[:,0]/b[:,1])
   low,high=np.percentile(np.mean(boots,axis=0),[2.5,97.5]);stats.append(dict(condition=condition,scope=scope,denominator_each_seed=108 if scope=='all' else 60,mean_accuracy=float(np.mean(values)),seed_sample_std=float(np.std(values,ddof=1)),log_ci95_low=float(low),log_ci95_high=float(high),seeds=3,logs=2))
 with (O/'three_seed_ablation.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=stats[0]);w.writeheader();w.writerows(stats)
 a={(r['condition'],r['scope']):r for r in stats}
 sections.append(('真实感知图：正式三种子合作模块消融',['配置','全任务均正确率','种子标准差','正目标均正确率','种子标准差','正目标日志CI'],[[c,percent(a[c,'all']['mean_accuracy']),f"{a[c,'all']['seed_sample_std']:.4f}",percent(a[c,'ordinary_positive']['mean_accuracy']),f"{a[c,'ordinary_positive']['seed_sample_std']:.4f}",percent(a[c,'ordinary_positive']['log_ci95_low'])+'—'+percent(a[c,'ordinary_positive']['log_ci95_high'])] for c in conditions],'每种子108查询/60正目标；相同真实模型输出后的1620次新CPU合作决策。三种子样本标准差描述训练波动，日志bootstrap4000次描述2日志数据波动，不能当作3个独立道路数据集。固定顺序消融优于完整配置，负结果保留。无在线控制或实际到达信用。'))
 a=read(Q/'real_graph_decisions_001/comparison.csv');by={}
 for r in a:by.setdefault(r['method'],{})[r['metric']]=r
 methods=['class_nearest_proposal_only','restricted_complete_geometry_rule','cooperation_seed17_without_candidate_reranking','cooperation_seed17_full']
 sections.append(('相同真实图：内部强规则与完整合作对照',['方法','正确/108','正目标正确率','正目标日志CI','无匹配误选率/12'],[[m,by[m]['task_accuracy']['strict_correct'],percent(by[m]['target_proposal_accuracy_all_positive']['value']),percent(by[m]['target_proposal_accuracy_all_positive']['ci95_low'])+'—'+percent(by[m]['target_proposal_accuracy_all_positive']['ci95_high']),percent(by[m]['no_match_false_proposal_rate']['value'])] for m in methods],'集成种子17在评测前固定，未按结果挑最好种子。规则均为本项目内部受控对照，不冒充外部公开算法。强规则13/60、固定顺序14/60、完整合作9/60；当前结果不支持Qwen重排提升找目标能力。'))
 a=[r for r in read(Q/'real_graph_permutation_002/comparison.csv') if r['scope']=='all'];sections.append(('候选顺序变化：开发压力测试',['模型','严格编号一致/18','编号辅助一致/18','原序正确/18','逆序正确/18'],[[r['model'],r['strict_ID_consistent'],r['target_only_aux_consistent'],r['original_correct'],r['reversed_correct']] for r in a],'固定18条来自9帧、2场景、2日志，每层2条，七模型新增126次真实GPU推理。只逆序节点列表，许可ID列表与输出示例不变；两个无效输出不计一致。稳定拒答不等于找对目标，CI及正查询10条分层见real_graph_permutation_statistics_001/。不是新封存测试。'))
 a=[r for r in read(Q/'real_graph_weighted_pilot_analysis_001/pilot_strata_and_cooperation.csv') if r['scope'] in ['all','ordinary_positive']];by={}
 for r in a:by.setdefault(r['method'],{})[r['scope']]=r
 sections.append(('单种子加权试验：合作诊断',['方法','正确/108','正目标正确/60','正目标日志CI'],[[m,v['all']['correct'],v['ordinary_positive']['correct'],percent(v['ordinary_positive']['ci95_low'])+'—'+percent(v['ordinary_positive']['ci95_high'])] for m,v in by.items()],'训练输入/标签/基线不变，只提高最终编号与拒答值的损失权重；459训练、171验证、240步，验证固定18条选240步。保留80/160/240步Adapter，试验不保存optimizer恢复状态。新增108次GPU评测、540次离线CPU合作。未改善正目标选择，不替换正式方案。'))
 wb=Workbook();wb.remove(wb.active);body=[]
 for i,(title,headers,rr,note) in enumerate(sections,1):
  ws=wb.create_sheet(f'{i:02d}');ws.append([title]);ws.append(headers)
  for row in rr:ws.append(row)
  ws.append([note]);ws.freeze_panes='B3';ws.auto_filter.ref=f'A2:{ws.cell(2,len(headers)).column_letter}{2+len(rr)}'
  for col in ws.columns:
   ws.column_dimensions[col[0].column_letter].width=28
   for cell in col:cell.font=Font(name='Noto Serif CJK SC',size=10);cell.alignment=Alignment(vertical='top',wrap_text=True)
  for cell in ws[2]:cell.font=Font(name='Noto Serif CJK SC',size=10,bold=True)
  if i>1:body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
  body.extend([p(f'表{i} '+title),table(headers,rr),p(note)])
 wb.save(O/'真实感知语义实验补充表.xlsx')
 base=R/'tables/completed_office_002/已完成实验交付表.docx'
 sect='<w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/><w:pgMar w:top="850" w:right="700" w:bottom="850" w:left="700"/></w:sectPr>'
 doc='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'+''.join(body)+sect+'</w:body></w:document>'
 ET.fromstring(doc)
 with zipfile.ZipFile(base) as source,zipfile.ZipFile(O/'真实感知语义实验补充表.docx','x',compression=zipfile.ZIP_DEFLATED) as target:
  for name in source.namelist():target.writestr(name,doc.encode() if name=='word/document.xml' else source.read(name))
 assert len(load_workbook(O/'真实感知语义实验补充表.xlsx').sheetnames)==5
 dump(O/'receipt.json',dict(tables=5,sources_sha256={**sources,str(Path(__file__)):digest(__file__)},word_sha256=digest(O/'真实感知语义实验补充表.docx'),excel_sha256=digest(O/'真实感知语义实验补充表.xlsx'),word_XML_parsed=True,excel_reopened=True,thesis_prose_written=False,whole_project_sealed_test=False,goal_complete=False));print(json.dumps(dict(tables=5,goal_complete=False)))

if __name__=='__main__':main()
