"""由六个真实重建产物计算按采集log配对的指标与区间。"""
import csv,json
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import digest,dump
ROOT=Path(__file__).resolve().parents[1];CONDITIONS=['full','without_depth_quality','without_one_to_one','without_dynamic_maintenance','without_short_term_expiry','without_relations']
FIELDS=['TP','GT','PRED','ERROR_SUM','DUP','IDTP','IDFN','IDFP','ID_SWITCH','ADJ_TRANS','REL_TP','REL_PRED','REL_GT','REL_UNKNOWN','TIME_SUM','FRAMES']
NAMES=['object_recall','matched_prediction_fraction','center_error_m','duplicate_nodes_per_GT','IDF1','ID_switches_per_transition','relation_precision_resolved','relation_recall','relation_F1','relation_precision_unknown_as_wrong','mean_mapping_relation_s']
def rates(a):
 def div(x,y):return np.divide(x,y,out=np.full_like(x,np.nan,dtype=float),where=y>0)
 x=np.moveaxis(a,-1,0)
 return np.stack([div(x[0],x[1]),div(x[0],x[2]),div(x[3],x[0]),div(x[4],x[1]),div(2*x[5],2*x[5]+x[6]+x[7]),div(x[8],x[9]),div(x[10],x[11]),div(x[10],x[12]),div(2*x[10],x[11]+x[12]),div(x[10],x[11]+x[13]),div(x[14],x[15])],axis=-1)
def finite(v):return float(v) if np.isfinite(v) else None
def main():
 dest=ROOT/'nuscenes_ablation/statistics_001';dest.mkdir(exist_ok=False);datasets=[];releases=[];hashes={}
 for c in CONDITIONS:
  source=ROOT/'nuscenes_ablation'/f'strict_{c}_001';summary=json.loads((source/'summary.json').read_text());release=json.loads((source/'release.json').read_text());assert summary['scenes']==38 and summary['frames']==1508
  for r in summary['reports']:assert digest(source/r['scene']/'frames.jsonl')==r['records_sha256']
  datasets.append(summary);releases.append(release)
  for name in ['release.json','summary.json']:hashes[str(source/name)]=digest(source/name)
 logs=sorted({r['log'] for r in datasets[0]['reports']});assert len(logs)==11
 A=np.zeros((len(logs),len(CONDITIONS),len(FIELDS)));scene_rows=[];isolation=[]
 for j,(c,data,release) in enumerate(zip(CONDITIONS,datasets,releases)):
  assert release['frames']==releases[0]['frames'] and release['source_sha256']==releases[0]['source_sha256']
  diff={k:[releases[0]['config'].get(k),v] for k,v in release['config'].items() if releases[0]['config'].get(k)!=v};assert len(diff)==(0 if c in ['full','without_relations'] else 1)
  isolation.append(dict(condition=c,config_differences=diff,graph_removed=release['removed_graph_module'],inputs_same=True,sources_same=True))
  for r in data['reports']:
   o,i,t=r['object'],r['identity'],r['relations'];times=r['mapping_relation_time_s'];v=[o['true_positive'],o['truth'],o['predictions'],o['center_error_sum_m'],o['duplicate_nodes'],i['IDTP'],i['IDFN'],i['IDFP'],o['identity_switches'],o['adjacent_matched_transitions'],t['true_positive'],t['predicted'],t['truth'],t['unresolved_edges'],sum(times),r['frames']];A[logs.index(r['log']),j]+=v;scene_rows.append(dict(condition=c,scene=r['scene'],log=r['log'],**dict(zip(FIELDS,v)),peak_process_rss_mib=r['peak_process_rss_mib']))
 totals=A.sum(axis=0);points=rates(totals);rng=np.random.default_rng(17);boot=rates(A[rng.integers(0,len(logs),(4000,len(logs)))].sum(axis=1));rows=[];estimates={}
 for j,c in enumerate(CONDITIONS):
  estimates[c]=dict(counts=dict(zip(FIELDS,totals[j].tolist())),metrics={},peak_process_rss_mib=max(r['peak_process_rss_mib'] for r in datasets[j]['reports']),frame_time_p50_s=float(np.median([v for r in datasets[j]['reports'] for v in r['mapping_relation_time_s']])),frame_time_p95_s=float(np.quantile([v for r in datasets[j]['reports'] for v in r['mapping_relation_time_s']],.95)))
  for k,name in enumerate(NAMES):
   sample=boot[:,j,k];sample=sample[np.isfinite(sample)];diff=boot[:,j,k]-boot[:,0,k];diff=diff[np.isfinite(diff)];row=dict(condition=c,metric=name,value=finite(points[j,k]),ci95_low=finite(np.quantile(sample,.025)) if len(sample) else None,ci95_high=finite(np.quantile(sample,.975)) if len(sample) else None,delta_vs_full=finite(points[j,k]-points[0,k]),delta_ci95_low=finite(np.quantile(diff,.025)) if len(diff) else None,delta_ci95_high=finite(np.quantile(diff,.975)) if len(diff) else None,relative_change_vs_full=finite((points[j,k]-points[0,k])/abs(points[0,k])) if points[0,k]!=0 else None,frames=1508,scenes=38,logs=11);rows.append(row);estimates[c]['metrics'][name]=row
 for name,data in [('metrics.csv',rows),('source_data_by_scene.csv',scene_rows)]:
  with (dest/name).open('x') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 dump(dest/'report.json',dict(conditions=estimates,isolation_checks=isolation,source_sha256=hashes,statistics_source_sha256=digest(__file__),scenes=38,logs=11,frames_per_condition=1508,actual_new_mapper_calls=1508*6,bootstrap=dict(unit='采集log',paired=True,repetitions=4000,seed=17),interpretation='对象/关系图模块单因素消融；全体预测ID与位置由相同推理链产生；官方中心门限身份代理，非人工GT',limitations=['不是Qwen语义消融或新封存测试','并行共享CPU；耗时和进程峰值内存为诊断，不能当独占实时性能','关系身份未知单列，下界把未知全计错误','full保留深度+基数+运动+TTL+增量关系；无Qwen调用，不宣称完整语义导航架构验收'],goal_complete=False))
 lines=['# nuScenes 单模块对象/关系图消融结果','', '38场景、11采集log、每条件1508帧。括号区间见metrics.csv；这些实验不包含Qwen语义模块或实际到达。','', '|配置|对象找全率|匹配预测比例|中心误差m|重复节点|IDF1|关系F1|单帧建图关系均值s|','|---|---|---|---|---|---|---|---|']
 for c in CONDITIONS:
  d=estimates[c];m=d['metrics'];lines.append(f"|{c}|{m['object_recall']['value']:.4f}|{m['matched_prediction_fraction']['value']:.4f}|{m['center_error_m']['value']:.4f}|{int(d['counts']['DUP'])}|{m['IDF1']['value']:.4f}|{m['relation_F1']['value']:.4f}|{m['mean_mapping_relation_s']['value']:.4f}|")
 (dest/'结果说明.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
