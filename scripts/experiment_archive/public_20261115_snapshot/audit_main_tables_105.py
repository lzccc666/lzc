"""核对封存主表的条件、方法、分母和可重复统计；不修改主结果。"""
from __future__ import annotations
import csv, json, math, hashlib
from datetime import datetime, timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1]
OUT=R/'json/main_table_audit_105'; OUT.mkdir(exist_ok=True)
def read_csv(rel):
 p=R/rel
 rows=list(csv.DictReader(p.open(encoding='utf-8-sig')))
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 return rows,h
retrieval,rhash=read_csv('figures/sealed_public_078/retrieval_all_metrics.csv')
paired,phash=read_csv('figures/sealed_public_078/retrieval_paired_differences.csv')
mapping,mhash=read_csv('figures/sealed_public_078/mapping_metrics.csv')
mpaired,mphash=read_csv('figures/sealed_public_078/mapping_paired_differences.csv')
methods=sorted({r['method'] for r in retrieval}); cohorts=sorted({r['cohort'] for r in retrieval})
metrics=sorted({r['metric'] for r in retrieval})
conditions=sorted({r['condition'] for r in mapping}); mapmetrics=sorted({r['metric'] for r in mapping})
assert len(retrieval)==84 and len(methods)==14 and len(cohorts)==2 and len(metrics)==3
assert len(mapping)==60 and len(conditions)==10 and len(mapmetrics)==6
assert len(paired)==78 and len(mpaired)==54
assert {int(r['denominator']) for r in retrieval if r['metric']=='task_correct'} == {600,278}
assert {int(r['denominator']) for r in retrieval if r['metric']=='positive_target_correct'} == {264,158}
assert {int(r['denominator']) for r in retrieval if r['metric']=='no_match_false_selection'} == {336,120}
assert {int(r['frames']) for r in mapping} == {1196} and {int(r['scenes']) for r in mapping} == {30} and {int(r['logs']) for r in mapping} == {14}
for row in retrieval+mapping+paired+mpaired:
 for k in row:
  if k in {'method','cohort','condition','label','reference','comparison','metric','desirable_direction'}: continue
  if row[k] and row[k] not in {'True','False'} and not math.isfinite(float(row[k])): raise AssertionError((k,row[k]))
report={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'goal_complete':False,'retrieval_rows':len(retrieval),'retrieval_methods':methods,'retrieval_cohorts':cohorts,'retrieval_metrics':metrics,'mapping_rows':len(mapping),'mapping_conditions':conditions,'mapping_metrics':mapmetrics,'retrieval_denominators':{'distribution_preserving':{'task_correct':600,'positive_target_correct':264,'no_match_false_selection':336},'challenge':{'task_correct':278,'positive_target_correct':158,'no_match_false_selection':120}},'mapping_shape':{'frames_per_condition':1196,'scenes':30,'logs':14},'paired_rows':{'retrieval':len(paired),'mapping':len(mpaired)},'source_sha256':{'retrieval_all_metrics.csv':rhash,'retrieval_paired_differences.csv':phash,'mapping_metrics.csv':mhash,'mapping_paired_differences.csv':mphash},'interpretation_limits':['模板主集和挑战集不是人工自然指令','表中没有实际锁定或到达信用','14方法中的受控稀疏网格和内部规则/规划是适配基线，不是完整外部方法复现'],'checks_passed':True}
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'retrieval_rows':84,'methods':14,'mapping_rows':60,'checks_passed':True},ensure_ascii=False))
