"""独立重算固定GT中心时间敏感性：不用生产插值、质量或统计函数。"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from conceptgraph.nuscenes.expansion_benchmark import iter_records

out=Path(__file__).resolve().parents[1]
project=out.parents[2]
source=out/'mapping_evaluation/camera_time_gt_dev_full_001'
dest=source/'audit_001';dest.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
events=out/'interactive_simulation/realtime_three_model_full_001/resource_events.jsonl'
def event(name):
 with events.open('a') as h:h.write(json.dumps(dict(event=name,run_id='camera_time_gt_audit_001',time_unix_s=time.time(),device='cpu',GPU_calls=0,performance_credit=False))+'\n')
event('camera_time_gt_independent_audit_start')
try:
 release=load(source/'release.json');report=load(source/'report.json');complete=load(source/'complete.json')
 assert sha(source/'report.json')==complete['report_sha256']
 assert sha(source/'frames.jsonl')==report['raw_records_sha256']
 assert sha(source/'release.json')==report['release_sha256']
 for field in ['source_sha256','input_sha256']:
  for name,value in release[field].items():assert sha(name)==value,(field,name)
 inputs={};all_objects={}
 for name in release['input_sha256']:
  if name.endswith('/frames.jsonl'):
   rows=[json.loads(line) for line in Path(name).read_text().splitlines()]
   for row in rows:inputs[(row['scene'],row['frame_idx'])]=row
   for row in rows:
    for obj in row['evaluation_only_official']:all_objects[obj['annotation_token']]=obj
 assert len(inputs)==1508 and sum(len(x['evaluation_only_official']) for x in inputs.values())==5723
 annotations_path=Path(next(name for name in release['input_sha256'] if name.endswith('/sample_annotation.json')))
 current={r['token']:r for r in iter_records(annotations_path) if r['token'] in all_objects}
 wanted={a['prev'] for a in current.values() if a['prev']}
 previous={r['token']:r for r in iter_records(annotations_path) if r['token'] in wanted}
 sample_path=Path(next(name for name in release['input_sha256'] if name.endswith('/sample.json')))
 wanted_samples={r['sample_token'] for r in [*current.values(),*previous.values()]}
 samples={r['token']:r for r in iter_records(sample_path) if r['token'] in wanted_samples}
 alignment=out/'data_manifest/development_causal_inputs_full_001/alignment_manifest.json'
 plans={(p['camera']['scene'],p['camera']['frame_idx']):p['camera'] for p in load(alignment)['plans']}
 split_paths=set();scene_logs={}
 for key in inputs:
  camera=plans[key];root=Path(camera['original_root']);split=root/'split_manifest.json';selection=root/'scene_selection_manifest.json';split_paths.update([split,selection])
  assert load(split)['scene_to_split'][key[0]]=='development'
  row=next(r for r in load(selection)['scenes'] if r['scene']==key[0]);scene_logs[key[0]]=row['log_token']
 # 原split/selection早于本轮实验，逐项对照初始受保护SHA，不伪称事后新冻结。
 expected={str(p.relative_to(project)):sha(p) for p in split_paths}
 found={}
 for line in (out/'source_bundle/protected_manifest.jsonl').open():
  r=json.loads(line)
  if r['path'] in expected:found[r['path']]=r['sha256']
 assert found==expected
 methods=list(next(iter(inputs.values()))['predictions']);logs=sorted(set(scene_logs.values()))
 conditions=['official_sample_proxy','camera_center_interpolation_proxy']
 groups=['interpolated','exact','fallback','motion_static','motion_dynamic','motion_unknown']
 motion_path=out/'annotation/official_motion_dev_001/evaluation_only_motion.jsonl'
 motion={(r['scene'],r['frame_idx']):r['objects'] for r in map(json.loads,motion_path.open()) if r['scene'] in release['scenes']}
 counts=np.zeros((len(logs),2,len(methods),7));cohorts=np.zeros((len(logs),2,len(methods),6,3))
 assignments=defaultdict(dict);states=Counter();reasons=Counter();shifts=[];shift_groups=defaultdict(list);seen=set();method_frames=0;changed=Counter()
 for line in (source/'frames.jsonl').open():
  row=json.loads(line);key=row['scene'],row['frame_idx'];assert key in inputs and key not in seen;seen.add(key)
  old=inputs[key];camera=plans[key];assert row['camera_timestamp_us']==camera['timestamp'] and row['GT_sent_to_inference'] is False
  truth=old['evaluation_only_official'];aligned=row['aligned_evaluation_only'];details=row['annotation_center_alignment']
  assert len(truth)==len(aligned)==len(details)==len(motion[key])
  for obj,new,detail,label in zip(truth,aligned,details,motion[key]):
   assert obj['annotation_token']==new['annotation_token']==detail['annotation_token']==label['annotation_token']
   annotation=current[obj['annotation_token']];cur=samples[annotation['sample_token']];camera_us=camera['timestamp'];position=np.asarray(obj['center'],float)
   assert np.array_equal(position,annotation['translation']) and annotation['instance_token']==obj['instance_token']
   assert annotation['sample_token']==camera['sample_token'] and cur['scene_token']==camera['scene_token']
   if camera_us==cur['timestamp']:state,reason='exact','camera_equals_official_sample'
   elif not annotation['prev']:state,reason='fallback','no_previous_annotation'
   else:
    prev=previous[annotation['prev']];p_sample=samples[prev['sample_token']];gap=cur['timestamp']-p_sample['timestamp'];assert prev['instance_token']==obj['instance_token'] and p_sample['scene_token']==cur['scene_token'] and gap>0
    if gap>1000000:state,reason='fallback','annotation_gap_exceeds_budget'
    elif not p_sample['timestamp']<=camera_us<=cur['timestamp']:state,reason='fallback','camera_not_bracketed_no_extrapolation'
    else:
     state,reason='interpolated','linear_centers_between_adjacent_official_annotations'
     # 独立写成当前中心减去标注速度乘时间差，避免复用生产的凸组合实现。
     lag=cur['timestamp']-camera_us;position=position-(np.asarray(annotation['translation'])-prev['translation'])*lag/gap
     assert np.isclose(detail['interpolation_weight'],1-lag/gap,atol=1e-12)
   assert detail['state']==state and detail['reason']==reason and detail['evaluation_only'] is True and detail['center_only'] is True
   assert np.allclose(position,new['center'],atol=1e-9,rtol=0)
   assert {k:v for k,v in obj.items() if k!='center'}=={k:v for k,v in new.items() if k!='center'}
   shift=float(np.linalg.norm(position-obj['center']));assert np.isclose(shift,detail['center_shift_m'],atol=1e-9)
   assert np.isclose(detail['current_annotation_lag_ms'],(cur['timestamp']-camera_us)/1000.)
   states[state]+=1;reasons[reason]+=1;shifts.append(shift);shift_groups['category_'+obj['category']].append(shift);shift_groups['motion_'+label['official_motion']].append(shift)
  for c,condition in enumerate(conditions):
   target=truth if c==0 else aligned
   for m,method in enumerate(methods):
    pred=old['predictions'][method];n_pred,n_gt=len(pred),len(target);distances=np.full((n_pred,n_gt),np.inf)
    for i,p in enumerate(pred):
     for j,g in enumerate(target):
      if p['category']==g['category']:distances[i,j]=np.hypot(p['center'][0]-g['center'][0],p['center'][1]-g['center'][1])
    allowed=distances<=3.;cost=np.zeros((n_pred,n_gt+n_pred));cost[:,:n_gt]=np.where(allowed,distances-4*(n_pred+1),1e9)
    pairs=[]
    if n_pred and n_gt:
     ii,jj=linear_sum_assignment(cost);pairs=[(int(i),int(j)) for i,j in zip(ii,jj) if j<n_gt and allowed[i,j]]
    saved=row['methods'][condition][method];assert set(pairs)=={(p['prediction_index'],p['truth_index']) for p in saved['matches']}
    if c==0:assert saved==old['metrics'][method]
    else:changed[method]+={(p['prediction_index'],p['truth_index']) for p in row['methods'][conditions[0]][method]['matches']}!=set(pairs)
    matched_pred={i for i,j in pairs};dup=sum(bool(allowed[i].any()) for i in range(n_pred) if i not in matched_pred)
    for name,value in dict(predictions=n_pred,truth=n_gt,true_positive=len(pairs),duplicate_nodes=dup,false_negative=n_gt-len(pairs),unmatched_predictions=n_pred-len(pairs)).items():assert saved[name]==value
    error=sum(distances[i,j] for i,j in pairs)
    assert saved['center_error_m_mean'] is None if not pairs else np.isclose(saved['center_error_m_mean'],error/len(pairs))
    l=logs.index(scene_logs[key[0]]);counts[l,c,m]+=[len(pairs),n_gt,n_pred,error,0,0,dup]
    assignments[(key[0],c,m)][key[1]]=[(pred[i]['node_id'],target[j]['instance_token']) for i,j in pairs]
    matched={j:distances[i,j] for i,j in pairs}
    for j,detail in enumerate(details):
     for group in [detail['state'],'motion_'+motion[key][j]['official_motion']]:cohorts[l,c,m,groups.index(group)]+=[1,int(j in matched),matched.get(j,0)]
    method_frames+=1
 assert seen==set(inputs) and method_frames==21112
 for (scene,c,m),rows in assignments.items():
  last={};switches=transitions=0
  for index,pairs in sorted(rows.items()):
   for node,instance in pairs:
    if instance in last and last[instance][0]==index-1:transitions+=1;switches+=last[instance][1]!=node
    last[instance]=(index,node)
  counts[logs.index(scene_logs[scene]),c,m,4:6]+=[switches,transitions]
 assert dict(states)==report['alignment_states'] and dict(reasons)==report['fallback_reasons'] and dict(changed)==report['changed_assignment_frames']
 ratio_columns={'recall':(0,1),'matched_prediction_precision':(0,2),'center_error_m_mean':(3,0),'identity_switch_rate':(4,5),'duplicate_nodes_per_GT':(6,1)}
 def ratios(v):return np.stack([np.divide(v[...,i],v[...,j],out=np.full(v.shape[:-1],np.nan),where=v[...,j]>0) for i,j in ratio_columns.values()],axis=-1)
 draws=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)));points=ratios(counts.sum(axis=0));samples=ratios(counts[draws].sum(axis=1))
 def check(estimate,point,values):
  valid=values[np.isfinite(values)];assert estimate['valid_resamples']==len(valid)
  if np.isfinite(point):assert np.isclose(estimate['value'],point,atol=1e-11)
  else:assert estimate['value'] is None
  if len(valid):assert np.allclose(estimate['ci95'],np.quantile(valid,[.025,.975]),atol=1e-11)
  else:assert estimate['ci95'] is None
 for c,condition in enumerate(conditions):
  for m,method in enumerate(methods):
   for k,name in enumerate(ratio_columns):check(report['conditions'][condition][method][name],points[c,m,k],samples[:,c,m,k])
 for row in report['temporal_differences']:
  m=methods.index(row['method']);k=list(ratio_columns).index(row['metric']);check(row,points[1,m,k]-points[0,m,k],samples[:,1,m,k]-samples[:,0,m,k])
 for row in report['method_contrasts']:
  c=conditions.index(row['condition']);a=methods.index(row['method']);b=methods.index(row['baseline']);k=list(ratio_columns).index(row['metric']);check(row,points[c,a,k]-points[c,b,k],samples[:,c,a,k]-samples[:,c,b,k])
 cohort_points=cohorts.sum(axis=0);cohort_samples=cohorts[draws].sum(axis=1)
 for c,condition in enumerate(conditions):
  for m,method in enumerate(methods):
   for g,group in enumerate(groups):
    truth,matched,error=cohort_points[c,m,g];s=cohort_samples[:,c,m,g];saved=report['cohorts'][condition][method][group];assert saved['truth']==truth and saved['matched']==matched
    for name,numerator,denominator in [('recall',1,0),('center_error_m_mean',2,1)]:
     value=cohort_points[c,m,g,numerator]/cohort_points[c,m,g,denominator] if cohort_points[c,m,g,denominator] else np.nan
     vals=np.divide(s[:,numerator],s[:,denominator],out=np.full(len(s),np.nan),where=s[:,denominator]>0);check(saved[name],value,vals)
 distributions={name:dict(observations=len(v),median=float(np.median(v)),p95=float(np.quantile(v,.95)),maximum=float(max(v))) for name,v in shift_groups.items()}
 audit=dict(passed=True,updated_utc=datetime.now(timezone.utc).isoformat(),source_report_sha256=sha(source/'report.json'),source_records_sha256=sha(source/'frames.jsonl'),audit_source_sha256=sha(__file__),frames=1508,fixed_GT_observations=5723,interpolated=5311,fallback_retained=412,method_frame_conditions_recomputed=method_frames,
   all_centers_and_source_chains_independently_recomputed=True,all_original_frame_metrics_exact=True,
   all_assignments_and_ID_transitions_independently_recomputed=True,all_point_CI_and_paired_contrasts_recomputed=True,
   split_sources_match_preexisting_protected_baseline=expected,center_shift_m_by_group=distributions,
   frontend_model_or_navigation_calls=0,new_test_opened=False,limitations=['共享SciPy优化器，不调用生产插值/质量/统计函数；独立重算不等于人工实例真值','中心插值为离线参考模型，保留412原中心；不是完整相机时刻GT或新的主表','原可见GT集合和预测不变，不补标缺失对象，也不计锁定/路线/到达'],goal_complete=False)
 (dest/'report.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
 text=['# 相机时刻中心插值：固定开发集合敏感性','', '38场景11log1508帧，原5723个GT观测全部保留；5311可插值、412缺前驱仍用原中心。尺寸、姿态、可见集合不变。这是中心参考模型敏感性，不是精确相机时刻真值或封存主表。','', '|评价中心|方法|召回|匹配精确率|中心误差(m)|','|---|---|---:|---:|---:|']
 for c,condition in enumerate(conditions):
  for method in ['original_fusion','one_to_one_depth_quality','one_to_one_depth_robust_motion']:
   values=report['conditions'][condition][method];text.append(f"|{condition}|{method}|{100*values['recall']['value']:.3f}%|{100*values['matched_prediction_precision']['value']:.3f}%|{values['center_error_m_mean']['value']:.4f}|")
 text+=['', '深度质量相对原融合在插值参考下召回差+12.983pp（11log配对95%CI[8.205,18.200]pp）、中心误差差−0.37094m（CI[−0.52945,−0.24141]m）。该定位差异在这一敏感性条件仍保留；身份变化率差区间跨0。', '', '同一深度质量方法改评价中心后召回只+0.227pp（CI[−0.023,0.727]pp），中心误差−0.02953m（CI[−0.05285,0.00374]m），区间跨0，不声称实测提高。不能据此把412fallback或非线性运动近似消除。', '', '全部21112个方法×帧×评价条件、逐对象中心/来源、指派/身份和区间独立重算通过。未新调用模型或改变冻结标签，真实人工仍不足，goal_complete=false。']
 (dest/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps({k:v for k,v in audit.items() if k not in ['center_shift_m_by_group','split_sources_match_preexisting_protected_baseline']},ensure_ascii=False))
finally:event('camera_time_gt_independent_audit_end')
