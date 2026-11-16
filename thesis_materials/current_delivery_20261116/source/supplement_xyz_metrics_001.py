"""已有公开数据输出的三维中心补充诊断；不覆盖原XY指标，不冒充官方3D检测AP。"""
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];O=R/'statistics/xyz_center_supplement_001'
NS=['full','without_depth_quality','without_one_to_one','without_dynamic_maintenance','without_short_term_expiry','without_relations']
FIELDS=['TP_XY','TP_XYZ','GT','PRED','XYZ_SUM_AT_XY','Z_SUM_AT_XY','XYZ_SUM_AT_XYZ','FRAMES']
NAMES=['center_XYZ_MAE_given_XY_match_m','center_Z_MAE_given_XY_match_m','recall_XYZ_3m','matched_prediction_fraction_XYZ_3m','center_XYZ_MAE_given_XYZ_match_m']
def evaluate(frame,method):
 p=frame['predictions'][method];g=frame['evaluation_only_official'];m=frame['metrics'][method];pc=np.array([x['center'] for x in p]).reshape(-1,3);gc=np.array([x['center'] for x in g]).reshape(-1,3);d=np.linalg.norm(pc[:,None,:]-gc[None,:,:],axis=-1);allowed=(d<=3)&(np.array([x['category'] for x in p],object)[:,None]==np.array([x['category'] for x in g],object)[None,:]);a,b=linear_sum_assignment(np.where(allowed,d,1e6));use=allowed[a,b];a,b=a[use],b[use];errors=[pc[x['prediction_index']]-gc[x['truth_index']] for x in m['matches']]
 return np.array([m['true_positive'],len(a),len(g),len(p),sum(np.linalg.norm(x) for x in errors),sum(abs(x[2]) for x in errors),float(d[a,b].sum()),1.])
def rates(a):
 out=[]
 with np.errstate(divide='ignore',invalid='ignore'):
  for x,y in [(4,0),(5,0),(1,2),(1,3),(6,1)]:out.append(a[...,x]/a[...,y])
 return np.stack(out,axis=-1)
def main():
 O.mkdir(parents=True,exist_ok=False);sources=[];units=[]
 for condition in NS:
  for p in sorted((R/'nuscenes_ablation'/f'strict_{condition}_001').glob('scene-*/frames.jsonl')):
   report=json.loads((p.parent/'report.json').read_text());sources.append(dict(dataset='nuScenes',condition=condition,path=p,unit=report['log'],scene=report['scene']))
 for p in sorted((R/'statistics/kitti_expanded_full_001').glob('*_frames.jsonl')):
  for method in ['single_frame','original_fusion','one_to_one_depth_robust_motion']:sources.append(dict(dataset='KITTI',condition=method,path=p,unit=p.name.split('_')[0],scene=p.name.split('_')[0]))
 dump(O/'release.json',dict(source_sha256={str(s['path']):digest(s['path']) for s in sources},new_gate='same coarse class, Euclidean XYZ<=3m, maximum cardinality then minimum distance',conditional_error='original frozen XY assignments are retained for first two errors; XYZ assignments separate',posthoc_supplement=True,bootstrap_seed=17,bootstrap_repetitions=4000,GT_in_inference=False,new_inference_calls=0,source_code_sha256=digest(__file__),goal_complete=False))
 bydataset=defaultdict(list);rows=[]
 for s in sources:
  total=np.zeros(len(FIELDS))
  for frame in map(json.loads,s['path'].open()):total+=evaluate(frame,s['condition'])
  row=dict(dataset=s['dataset'],method=s['condition'],unit=s['unit'],scene=s['scene'],**dict(zip(FIELDS,total.tolist())));rows.append(row);bydataset[s['dataset']].append(row)
 for dataset,rr in bydataset.items():
  names=sorted({r['method'] for r in rr});logs=sorted({r['unit'] for r in rr});A=np.zeros((len(logs),len(names),len(FIELDS)))
  for r in rr:A[logs.index(r['unit']),names.index(r['method'])]+=[r[k] for k in FIELDS]
  point=rates(A.sum(axis=0));rng=np.random.default_rng(17);boot=rates(A[rng.integers(0,len(logs),(4000,len(logs)))].sum(axis=1))
  for j,method in enumerate(names):
   for k,name in enumerate(NAMES):
    lo,hi=np.nanpercentile(boot[:,j,k],[2.5,97.5]);units.append(dict(dataset=dataset,method=method,metric=name,value=point[j,k],ci95_low=lo,ci95_high=hi,cluster_units=len(logs),frames=int(A[:,j,-1].sum()),GT_observations=int(A[:,j,2].sum()),predictions=int(A[:,j,3].sum()),TP_XY=int(A[:,j,0].sum()),TP_XYZ=int(A[:,j,1].sum())))
 for name,data in [('source_data_by_scene.csv',rows),('comparison.csv',units)]:
  with (O/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
 dump(O/'report.json',dict(rows=units,bootstrap='nuScenes11 acquisition logs; KITTI10 selected sequences, seed17/4000',scope='三维中心后验补充诊断；不改变主表XY匹配，不作为3D检测IoU/AP，不授予语义/规划/实际到达信用',new_inference_calls=0,source_sha256=digest(__file__),goal_complete=False));print(json.dumps(units,ensure_ascii=False))
if __name__=='__main__':main()
