"""冻结在新GPU结果结束前的语义图：不混淆正目标、覆盖和拒答。"""
import csv,json
from pathlib import Path
import plot_public_evidence_004 as plot
R=Path(__file__).resolve().parents[1];D=R/'nuscenes_ablation/full_semantics_near6_002';O=R/'figures/corrected_semantics_001'
def main():
 plot.O=O;plot.SOURCES={};contract=plot.load(O/'figure_contract.json');assert contract['backend']=='Python';report=plot.load(D/'run_report.json');assert report['actual_CPU_cooperation_decisions']==9048;data=plot.read(D/'metrics.csv');index={(x['condition'],x['role'],x['scope'],x['metric']):x for x in data};rows=[]
 for c,label in [('full','完整合作'),('without_depth_quality','去深度质量'),('without_one_to_one','去一对一关联'),('without_dynamic_maintenance','去动态维护'),('without_short_term_expiry','去短期过期'),('without_relations','去关系边')]:
  row=dict(condition=c,label=label,frames=1508,scenes=38,logs=11,positive_denominator=627,no_match_denominator=425)
  for key,scope,metric in [('cooperation','ordinary_positive','correct'),('strong_rule','ordinary_positive','strong_rule_correct'),('coverage','ordinary_positive','covered'),('false_selection','ordinary_no_match','no_match_false_proposal')]:
   v=index[c,'all',scope,metric];assert int(v['denominator'])==(425 if key=='false_selection' else 627);row[key]=float(v['value']);row[key+'_low']=float(v['ci95_low']);row[key+'_high']=float(v['ci95_high']);row[key+'_count']=int(v['numerator'])
  rows.append(row)
 plot.writecsv('corrected_semantics_source.csv',rows);plot.make(rows,['cooperation','strong_rule','coverage','false_selection'],['合作模块：正目标选对率 ↑','同输入几何规则：正目标选对率 ↑','候选包含正确对象的比例 ↑','无匹配误选率 ↓'],'corrected_semantics','38开发场景 / 11日志，每配置1508条；正目标627条，无匹配425条，故障拒答另报，不计入找对目标。\n误差线：日志聚类95%区间。含训练/验证日志，单seed17，实际邻近半径6米；不授予锁定或实际到达信用。',[.5,.5,.65,.5])
 with (O/'automated_qa_001.json').open('x') as f:json.dump(dict(figures=plot.QA,exports=plot.EXPORTS,sources_sha256=plot.SOURCES,source_sha256=plot.sha(__file__),whole_project_sealed_test=False,goal_complete=False),f,ensure_ascii=False,indent=2)
 print('corrected actual semantic figure exported',flush=True)
if __name__=='__main__':main()
