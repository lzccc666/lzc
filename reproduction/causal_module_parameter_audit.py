"""核对完整动态/TTL和关联门限真实记录；独立重算关联及log聚类区间，不跑模型。"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

out=Path(__file__).resolve().parents[1]
dest=out/'source_bundle/causal_module_parameter_audit_001'
dest.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
cache={}
files={}
metrics={'recall':(0,1),'precision':(0,2),'center_error_m_mean':(3,0),
    'identity_switch_rate':(4,5),'duplicates_per_gt_observation':(6,1)}
inputs={}
for scene in read(out/'ablations/causal_modules_full_001/release.json')['scene_logs']:
 p=out/'mapping_evaluation/causal_mapping_full_001'/scene/'frames.jsonl'
 inputs[scene]=[json.loads(line) for line in p.read_text().splitlines()]
 files[str(p)]=sha(p)
def case(path):
 if path in cache:return cache[path]
 report=read(path);raw=path.parent/'frames.jsonl';assert sha(raw)==report['records_sha256']
 files[str(path)]=sha(path);files[str(raw)]=sha(raw)
 rows=[json.loads(line) for line in raw.read_text().splitlines()]
 original=inputs[report['scene']];assert len(rows)==len(original)==report['frames']
 counts=defaultdict(int);error=0.;last={};node_instances=defaultdict(set)
 for row,source in zip(rows,original):
  assert row['frame_idx']==source['frame_idx'] and row['camera_timestamp_s']==source['camera_timestamp_s']
  assert row['scene']==source['scene'] and row['gt_sent_to_mapper'] is False
  pred,gt=row['predictions'],source['evaluation_only_official'];m,n=len(pred),len(gt)
  distances=np.full((m,n),np.inf)
  for i,p in enumerate(pred):
   for j,g in enumerate(gt):
    if p['category']==g['category']:distances[i,j]=np.hypot(p['center'][0]-g['center'][0],p['center'][1]-g['center'][1])
  allowed=distances<=3.;costs=np.zeros((m,n+m));costs[:,:n]=np.where(allowed,distances-4*(m+1),1e9)
  pairs=[]
  if m and n:
   ii,jj=linear_sum_assignment(costs);pairs=[(int(i),int(j)) for i,j in zip(ii,jj) if j<n and allowed[i,j]]
  saved=row['metrics'];assert set(pairs)=={(p['prediction_index'],p['truth_index']) for p in saved['matches']}
  selected={i for i,j in pairs}
  values=dict(truth=n,predictions=m,true_positive=len(pairs),false_negative=n-len(pairs),
   unmatched_predictions=m-len(pairs),duplicate_nodes=sum(bool(allowed[i].any()) for i in range(m) if i not in selected))
  for k,v in values.items():assert saved[k]==v;counts[k]+=v
  errors=[distances[i,j] for i,j in pairs]
  assert saved['center_error_m_mean'] is None if not errors else np.isclose(saved['center_error_m_mean'],np.mean(errors))
  error+=sum(errors)
  for i,j in pairs:
   node,instance=pred[i]['node_id'],gt[j]['instance_token']
   if instance in last and last[instance][0]==row['frame_idx']-1:
    counts['adjacent_matched_transitions']+=1;counts['identity_switches']+=last[instance][1]!=node
   last[instance]=(row['frame_idx'],node);node_instances[node].add(instance)
 counts['nodes_associated_with_multiple_instances']=sum(len(s)>1 for s in node_instances.values())
 for k,v in counts.items():assert report['metrics'][k]==v,(path,k)
 tp=counts['true_positive'];assert report['metrics']['center_error_m_mean'] is None if not tp else np.isclose(report['metrics']['center_error_m_mean'],error/tp)
 vector=np.array([tp,counts['truth'],counts['predictions'],error,counts['identity_switches'],
  counts['adjacent_matched_transitions'],counts['duplicate_nodes']])
 cache[path]=(vector,len(rows));return cache[path]
def ratios(v):
 return np.stack([np.divide(v[...,i],v[...,j],out=np.full(v.shape[:-1],np.nan),where=v[...,j]>0) for i,j in metrics.values()],axis=-1)
results=[]
for kind,folder,source_field,base in [('modules','ablations/causal_modules_full_001','condition','full'),
 ('parameters','performance/causal_mapping_parameters_full_001','threshold',0.35)]:
 source=out/folder;release=read(source/'release.json');summary=read(source/'summary.json')
 rows=[json.loads(line) for line in (source/'cases.jsonl').read_text().splitlines()]
 stats=read(out/'statistics'/source.name/'report.json');conditions=list(stats['conditions']);logs=sorted(set(release['scene_logs'].values()))
 assert len(rows)==summary['cases']==114
 for name,digest in release.get('source_frames_sha256',release.get('source_frames',{})).items():assert sha(name)==digest
 observed=set();counts=np.zeros((len(logs),len(conditions),7));frames=0
 for row in rows:
  condition=str(row[source_field]);key=(row['scene'],condition);assert key not in observed;observed.add(key)
  path=Path(row['report_path']);assert sha(path)==row['report_sha256']
  vector,length=case(path);frames+=length
  counts[logs.index(row['log_token']),conditions.index(condition)]+=vector
 assert observed=={(scene,cond) for scene in release['scene_logs'] for cond in conditions}
 points=ratios(counts.sum(axis=0));samples=ratios(counts[np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)))].sum(axis=1))
 for j,condition in enumerate(conditions):
  for k,metric in enumerate(metrics):
   s=stats['conditions'][condition][metric];assert np.isclose(points[j,k],s['estimate'])
   assert np.allclose(np.quantile(samples[:,j,k],[.025,.975]),s['ci95'])
 for row in stats['paired_differences']:
  j=conditions.index(str(row[source_field]));b=conditions.index(str(base));k=list(metrics).index(row['metric'])
  assert np.isclose(points[j,k]-points[b,k],row['difference'])
  assert np.allclose(np.quantile(samples[:,j,k]-samples[:,b,k],[.025,.975]),row['ci95'])
 results.append(dict(kind=kind,case_records=len(rows),frame_condition_records=frames,all_frame_assignments_recomputed=True,
  all_case_counts_and_ID_transitions_recomputed=True,estimates_and_intervals=15,paired_differences_and_intervals=10,
  source_summary_sha256=sha(source/'summary.json'),statistics_sha256=sha(out/'statistics'/source.name/'report.json')))
(dest/'files.json').write_text(json.dumps(files,ensure_ascii=False,indent=2)+'\n')
report=dict(passed=True,updated_utc=datetime.now(timezone.utc).isoformat(),results=results,unique_mapping_case_files=len(cache),
 unique_mapping_frame_records=sum(v[1] for v in cache.values()),source_files_checked=len(files),
 files_manifest_sha256=sha(dest/'files.json'),source_sha256=sha(__file__),
 model_calls=0,new_test_opened=False,actual_arrivals=0,
 limitations=['两批都复用同38个seed17完整基线，不能当成独立方法或重复增加样本',
 '独立重算同类3m XY代理关联，共享SciPy优化器，不能代替真实人工实例审核',
 'GT较相机晚约33–42ms，关闭动态模块导致运动标记unknown不作运动分类改善证据'],goal_complete=False)
(dest/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
