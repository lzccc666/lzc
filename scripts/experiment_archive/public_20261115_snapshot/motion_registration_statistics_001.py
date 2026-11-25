"""完整同输入建图对照+事后统一的静止/证据覆盖辅助诊断，不能仅报告静止数量增加。"""
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import OUTPUT as OLD,digest,dump
R=Path(__file__).resolve().parents[1];O=R/'statistics/motion_registration_001';S=R/'nuscenes_ablation'
KEYS=['TP','GT','PRED','DUP','ERR','IDTP','IDFP','IDFN','STATIC_GT','STATIC_MATCH_TP','MATCHED_PRED_STATIC_WITH_KNOWN_GT','WRONG_STATIC_ON_DYNAMIC_GT','EVIDENCE_STATIC_TP','PRED_STATIC','PRED_ELIGIBLE']
def values(rows):
 c={k:sum(r[k] for r in rows) for k in KEYS}
 return dict(object_recall=c['TP']/c['GT'],matched_prediction_fraction=c['TP']/c['PRED'],center_error_m=c['ERR']/c['TP'],IDF1=2*c['IDTP']/(2*c['IDTP']+c['IDFP']+c['IDFN']),duplicates=c['DUP'],static_GT_recall=c['STATIC_MATCH_TP']/c['STATIC_GT'] if c['STATIC_GT'] else None,conditional_matched_static_precision=c['STATIC_MATCH_TP']/c['MATCHED_PRED_STATIC_WITH_KNOWN_GT'] if c['MATCHED_PRED_STATIC_WITH_KNOWN_GT'] else None,wrong_static_on_dynamic_truth=c['WRONG_STATIC_ON_DYNAMIC_GT'],current_supported_static_GT_recall=c['EVIDENCE_STATIC_TP']/c['STATIC_GT'] if c['STATIC_GT'] else None,predicted_static_count=c['PRED_STATIC'],predicted_supported_count=c['PRED_ELIGIBLE'])
def main():
 assert (S/'development_icp_motion_full_001/summary.json').exists(),'run incomplete'
 O.mkdir(parents=True,exist_ok=False);label=OLD/'annotation/official_motion_dev_001/evaluation_only_motion.jsonl';motion={(r['scene'],r['frame_idx']):{o['annotation_token']:o['official_motion'] for o in r['objects']} for r in map(json.loads,label.open())};reports=[];hashes={}
 for method,folder in [('robust_centroid_full','strict_full_001'),('translation_icp_motion','development_icp_motion_full_001')]:
  root=S/folder;summary=json.loads((root/'summary.json').read_text());assert summary['scenes']==38 and summary['frames']==1508
  for sc in summary['reports']:
   p=root/sc['scene']/'frames.jsonl';hashes[str(p)]=digest(p);c={k:0 for k in KEYS};c.update(IDTP=sc['identity']['IDTP'],IDFP=sc['identity']['IDFP'],IDFN=sc['identity']['IDFN']);frames=list(map(json.loads,p.open()));cond=summary['condition']
   for f in frames:
    gt=f['evaluation_only_official'];pred=f['predictions'][cond];metrics=f['metrics'][cond];labels=motion[(sc['scene'],f['frame_idx'])]
    c['TP']+=metrics['true_positive'];c['GT']+=len(gt);c['PRED']+=len(pred);c['DUP']+=metrics['duplicate_nodes'];c['ERR']+=sum(m['distance_m'] for m in metrics['matches']);c['STATIC_GT']+=sum(labels.get(g['annotation_token'])=='static' for g in gt);eligible=[]
    for n in pred:
     e=(n.get('depth_evidence') or [{}])[-1];age=f['camera_timestamp_s']-(n.get('timestamp_s') or -1);ok=n['motion_state']=='static' and n['num_detections']>=3 and e.get('source')=='current_projected_lidar' and e.get('retained_pixels',0)>=5 and e.get('support_fraction',0)>=.4 and e.get('sufficient_support') is True and 0<=age<=2;eligible.append(ok);c['PRED_STATIC']+=n['motion_state']=='static';c['PRED_ELIGIBLE']+=ok
    for m in metrics['matches']:
     pi,gi=m['prediction_index'],m['truth_index'];labelvalue=labels.get(gt[gi]['annotation_token']);isstatic=pred[pi]['motion_state']=='static';c['STATIC_MATCH_TP']+=isstatic and labelvalue=='static';c['MATCHED_PRED_STATIC_WITH_KNOWN_GT']+=isstatic and labelvalue in ['static','dynamic'];c['WRONG_STATIC_ON_DYNAMIC_GT']+=isstatic and labelvalue=='dynamic';c['EVIDENCE_STATIC_TP']+=eligible[pi] and labelvalue=='static'
   reports.append(dict(method=method,scene=sc['scene'],unit=sc['log'],frames=len(frames),**c))
 logs=sorted({r['unit'] for r in reports});draw=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)));table=[];all_boot={};all_values={}
 for method in ['robust_centroid_full','translation_icp_motion']:
  rr=[r for r in reports if r['method']==method];v=values(rr);bylog={u:[r for r in rr if r['unit']==u] for u in logs};b=defaultdict(list)
  for choice in draw:
   for k,x in values([r for idx in choice for r in bylog[logs[idx]]]).items():b[k].append(np.nan if x is None else x)
  all_values[method]=v;all_boot[method]=b
  for k,x in v.items():
   lo,hi=np.nanpercentile(b[k],[2.5,97.5]);table.append(dict(method=method,metric=k,value=x,ci95_low=lo,ci95_high=hi,frames=1508,scenes=38,logs=11,GT=sum(r['GT'] for r in rr),static_GT=sum(r['STATIC_GT'] for r in rr),predictions=sum(r['PRED'] for r in rr),new_GPU_calls=0,auxiliary_metric_registered_after_development_started=k not in ['object_recall','matched_prediction_fraction','center_error_m','IDF1','duplicates']))
 diffs=[]
 for k,x in all_values['translation_icp_motion'].items():
  lo,hi=np.nanpercentile(np.asarray(all_boot['translation_icp_motion'][k])-np.asarray(all_boot['robust_centroid_full'][k]),[2.5,97.5]);baseline=all_values['robust_centroid_full'][k];diffs.append(dict(metric=k,absolute_delta=x-baseline if x is not None and baseline is not None else None,paired_ci95_low=lo,paired_ci95_high=hi,units=11))
 for name,rr in [('comparison.csv',table),('paired_differences.csv',diffs),('source_data_by_scene.csv',reports)]:
  with (O/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
 dump(O/'report.json',dict(source_sha256={**hashes,str(label):digest(label),str(Path(__file__)):digest(__file__)},bootstrap=dict(repetitions=4000,seed=17,unit='log',units=11,paired=True),scope='38开发场景完整同输入运动估计变体；不替换原六条件，不授予新封存、Qwen或CARLA信用',native_policy_note='current_supported辅助额外要求观测age<=2sec和depth sufficient_support；两个方法相同，不能冒称原native门限完全同义',static_precision_scope='仅条件于类别/3m匹配到已知官方运动标签的静止预测；未匹配静止节点未记入此条件精度，全部数量另列',labels_scope='官方标注轨迹推导运动状态，仅评价使用；不当人工颜色/细属性标注',posthoc_motion_metrics=True,shared_CPU_timing_not_realtime=True,goal_complete=False));print(json.dumps(table,ensure_ascii=False))
if __name__=='__main__':main()
