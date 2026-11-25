"""从已保存的完整运动评价重算分母、条件分数和逐log，记录负项。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,gzip,hashlib,json
from collections import Counter,defaultdict
from conceptgraph.project_paths import PROJECT_ROOT as ROOT

RUN=ROOT/'outputs/results/nuscenes/validation/iterations/node_motion_unified_20261125_v2'
REPORT=ROOT/'outputs/reports/node_motion_unified_20261125'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def div(a,b):return a/b if b else None

def rates(conf,scope,kind,method,full_gt):
 rows={g:conf[('legacy_' if method=='legacy' else '')+kind+':'+g] for g in ['static','dynamic']}
 total=sum(sum(r.values()) for r in rows.values());known=sum(r.get('static',0)+r.get('dynamic',0) for r in rows.values())
 missing=sum(r.get('missing_node',0) for r in rows.values());unknown=sum(r.get('unknown',0) for r in rows.values());correct=sum(rows[g].get(g,0) for g in rows)
 f1=[];per={}
 for g in rows:
  tp=rows[g].get(g,0);fp=sum(r.get(g,0) for k,r in rows.items() if k!=g);fn=sum(rows[g].values())-tp
  f=div(2*tp,2*tp+fp+fn);f1.append(f or 0.)
  per[g]=dict(TP=tp,FP_known_wrong_label=fp,FN_including_unknown_missing=fn,precision_on_reference_labeled_GT=div(tp,tp+fp),recall_on_all_reference_class_GT=div(tp,tp+fn),F1=f)
 return dict(scope=scope,reference=kind,method=method,all_GT=full_gt,reference_labeled_GT=total,reference_unknown_GT=full_gt-total,
   matched_reference_labeled_GT=total-missing,known_prediction_labeled_GT=known,unknown_prediction_labeled_GT=unknown,missing_node_labeled_GT=missing,
   correct=correct,joint_correct_over_all_labeled_GT=div(correct,total),known_coverage_over_all_labeled_GT=div(known,total),
   correct_over_matched_labeled_GT=div(correct,total-missing),conditional_accuracy_given_known_prediction=div(correct,known),
   verified_correct_over_all_GT=div(correct,full_gt),macro_F1_all_reference_GT=sum(f1)/2,per_class=per)


def main():
 source=RUN/'motion_evaluation_summary.json';d=json.loads(source.read_text());rows=[]
 for scope,v in d['scopes'].items():
  for ref in ['attribute','past_displacement']:
   for method in ['legacy','raw_evidence']:
    rows.append(rates(v['confusion'],scope,ref,method,v['counts']['GT']))
 bylog=defaultdict(Counter)
 with gzip.open(RUN/'motion_evaluation_rows.jsonl.gz','rt') as f:
  for line in f:
   r=json.loads(line)
   for ref,field in [('attribute','attribute_motion'),('past_displacement','past_displacement_motion')]:
    label=r[field]
    for method,predfield in [('legacy','legacy_predicted_motion'),('raw_evidence','predicted_motion')]:
     c=bylog[(r['scope'],r['log'],ref,method)];c['all_GT']+=1
     if label=='unknown':c['unknown_reference']+=1;continue
     c['labeled_GT']+=1;state=r[predfield];c['correct']+=int(label==state);c['known_prediction']+=int(state in ['static','dynamic']);c['unknown_or_missing']+=int(state in ['unknown','missing_node'])
 result=dict(passed=True,source_summary_sha256=sha(source),source_rows_sha256=sha(RUN/'motion_evaluation_rows.jsonl.gz'),report_source_sha256=sha(__file__),
   rates=rows,by_log=[dict(scope=k[0],log=k[1],reference=k[2],method=k[3],**dict(v)) for k,v in bylog.items()],
   denominator_note='Reference unknown remains recorded but cannot be treated as a correct/incorrect binary class. Joint score includes missing nodes/unknown prediction as no credit; allGTverified score also retains unlabeled denominator. Conditional accuracy never replaces joint score.',
   negative_result='Raw evidence method reduces known coverage and joint correct count relative to legacy on both scopes; do not promote default or claim motion-recognition improvement')
 (RUN/'motion_rates.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 flat=[{k:v for k,v in r.items() if k!='per_class'} for r in rows]
 with (RUN/'motion_rates.csv').open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 logs=result['by_log']
 with (RUN/'motion_log_rates.csv').open('x') as f:
  keys=['scope','log','reference','method','all_GT','unknown_reference','labeled_GT','correct','known_prediction','unknown_or_missing'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(logs)
 coverage={s:json.loads((RUN/(s+'_node_coverage.json')).read_text()) for s in ['R1','S2']}
 first={s:dict(sum((Counter(v) for v in coverage[s]['unknown_first_reason'].values()),Counter())) for s in coverage}
 (RUN/'unknown_first_reason_totals.json').write_text(json.dumps(dict(totals=first,ordering='as frozen motion_evidence function; first failures mutually exclusive; all_reasons in node_coverage are overlapping'),indent=2)+'\n')
 print(json.dumps(dict(passed=True,rates=[{k:r[k] for k in ['scope','reference','method','reference_labeled_GT','correct','known_prediction_labeled_GT','joint_correct_over_all_labeled_GT','conditional_accuracy_given_known_prediction']} for r in rows],unknown_first=first)),flush=True)

if __name__=='__main__':main()
