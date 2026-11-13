"""官方TrackEval隔离参考与独立门限、返回事件、聚类统计复核。"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import csv,difflib,hashlib,importlib,json,math,os,shutil,sys,time,types
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'source_bundle/identity_reference_001'
RUN=ROOT/'statistics/temporal_identity_full_001'
DEST=RUN/'audit_002';DEST.mkdir(exist_ok=False)
start=time.monotonic()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,data):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as h:json.dump(data,h,ensure_ascii=False,indent=2,allow_nan=False);h.write('\n')
def note(state):
 with (ROOT/'interactive_simulation/realtime_three_model_full_001/resource_events.jsonl').open('a') as h:
  h.write(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(),event='temporal_identity_audit_002',status=state,shared_CPU=True,CUDA_visible=''))+'\n')
manifest=json.loads((REF/'manifest.json').read_text())
compat=DEST/'reference_compatibility'
changes=[]
for row in manifest['files']:
 path=REF/'original'/row['path'];assert sha(path)==row['sha256']
 raw=path.read_text();converted=raw.replace('.astype(np.int)', '.astype(int)') if row['path']=='trackeval/metrics/identity.py' else raw
 target=compat/row['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_text(converted)
 if raw!=converted:
  diff=''.join(difflib.unified_diff(raw.splitlines(True),converted.splitlines(True),fromfile='original/'+row['path'],tofile='compatibility/'+row['path']))
  (DEST/'numpy_compatibility.patch').write_text(diff)
  changes.append(dict(path=row['path'],replacements=raw.count('.astype(np.int)'),original_sha256=sha(path),compatibility_sha256=sha(target)))
for name,rel in [('trackeval','trackeval'),('trackeval.metrics','trackeval/metrics')]:
 module=types.ModuleType(name);module.__path__=[str(compat/rel)];sys.modules[name]=module
Identity=importlib.import_module('trackeval.metrics.identity').Identity
reference=Identity(dict(THRESHOLD=.5,PRINT_CONFIG=False))
save(DEST/'release.json',dict(official_commit=manifest['commit'],download_manifest_sha256=sha(REF/'manifest.json'),compatibility_changes=changes,audit_source_sha256=sha(__file__),budget_wall_s=900,budget_bytes=1024**3,minimum_free_bytes=6*1024**3,independent_gate='math.hypot，同粗类别XY<=3m',independent_assignment='官方TrackEval扩展FN/FP矩阵',event_source='原保存的逐帧匹配；独立枚举返回和尾段，未独立重做逐帧Hungarian',goal_complete=False))
report=json.loads((RUN/'report.json').read_text())
release=json.loads((RUN/'release.json').read_text())
mapping=ROOT/'mapping_evaluation'/release['mapping_run']
assert sha(mapping/'release.json')==release['mapping_release_sha256']
assert sha(mapping/'summary.json')==release['mapping_summary_sha256']
methods=sorted(report['methods']);supports=['all_gated','mutually_unique_gated']
rows=[json.loads(x) for x in (RUN/'scene_method_support.jsonl').open()]
rowmap={(r['scene'],r['method'],r['support']):r for r in rows};assert len(rowmap)==len(rows)
by_log=defaultdict(Counter);checked=Counter();buckets=defaultdict(Counter)

def identity(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'))
def pid(obj,method,index):
 x=identity(obj['node_id']);return f'frame={index};id={x}' if method=='single_frame' else x

def recheck_returns(frames,method):
 visible=defaultdict(list);matches=[];legacy=[]
 for i,f in enumerate(frames):
  gt=f['evaluation_only_official'];pred=f['predictions'][method]
  for obj in gt:visible[identity(obj['instance_token'])].append(i)
  matched={};old={}
  for match in f['metrics'][method]['matches']:
   g,p=gt[match['truth_index']],pred[match['prediction_index']]
   assert g['category']==p['category'] and math.hypot(g['center'][0]-p['center'][0],g['center'][1]-p['center'][1])<=3
   token=identity(g['instance_token']);assert token not in matched
   matched[token]=pid(p,method,i);old[token]=identity(p['node_id'])
  matches.append(matched);legacy.append(old)
 count=Counter();events={};tails={}
 for token,indices in visible.items():
  count['GT_instances']+=1
  matched_indices=[i for i in indices if token in matches[i]]
  for a,b in zip(indices,indices[1:]):
   if b>a+1:
    left,right=matches[a].get(token),matches[b].get(token)
    status='unknown' if left is None or right is None else ('same' if left==right else 'changed')
    gap=frames[b]['camera_timestamp_s']-frames[a]['camera_timestamp_s']
    bucket='over_2s' if gap>2 else 'at_most_2s'
    count['official_return_opportunities']+=1;count['official_return_both_matched']+=status!='unknown'
    count[f'official_return_{status}_id']+=1
    events['official_list_return',token,a,b]=dict(identity_status=status,previous_matched=left is not None,
      returned_matched=right is not None,previous_prediction_identity=left,returned_prediction_identity=right,
      endpoint_gap_s=gap,intervening_frames=b-a-1,gap_bucket=bucket)
    buckets[method,bucket]['opportunities']+=1;buckets[method,bucket][status]+=1
  for a,b in zip(matched_indices,matched_indices[1:]):
   if b==a+1:
    count['adjacent_matched_transitions']+=1;count['adjacent_identity_changes']+=legacy[a][token]!=legacy[b][token]
   else:
    same=matches[a][token]==matches[b][token]
    count['method_gap_returns']+=1;count['method_gap_same_id' if same else 'method_gap_changed_id']+=1
    absent=sum(k not in indices for k in range(a+1,b))
    events['method_matched_gap_return',token,a,b]=dict(identity_status='same' if same else 'changed',
     previous_prediction_identity=matches[a][token],returned_prediction_identity=matches[b][token],
     endpoint_gap_s=frames[b]['camera_timestamp_s']-frames[a]['camera_timestamp_s'],intervening_frames=b-a-1,
     intervening_GT_absent_frames=absent,gap_cause='GT_list_absence_included' if absent else 'GT_list_continuously_present')
  if not matched_indices:
   count['never_matched_instances']+=1;count['never_matched_GT_observations']+=len(indices)
  elif matched_indices[-1]<len(frames)-1:
   last=matched_indices[-1];unreturned=sum(i>last for i in indices);absent=indices[-1]<len(frames)-1
   count['censored_tails']+=1;count['tails_GT_absent_at_scene_end']+=absent
   count['unreturned_visible_tail_observations']+=unreturned
   tails[token]=dict(last_matched_frame=last,last_GT_visible_frame=indices[-1],scene_end_frame=len(frames)-1,
      unmatched_visible_observations=unreturned,GT_absent_at_scene_end=absent)
 return count,events,tails

def gate(a,b):
 return a['category']==b['category'] and math.hypot(a['center'][0]-b['center'][0],a['center'][1]-b['center'][1])<=3

def norm(c):
 pairs={'IDR':('IDTP','GT_observations'),'IDP':('IDTP','prediction_observations'),
 'adjacent_identity_change_rate':('adjacent_identity_changes','adjacent_matched_transitions'),
 'official_return_both_matched_rate':('official_return_both_matched','official_return_opportunities'),
 'official_return_same_id_rate':('official_return_same_id','official_return_opportunities'),
 'official_return_changed_id_rate':('official_return_changed_id','official_return_opportunities'),
 'official_return_unknown_id_rate':('official_return_unknown_id','official_return_opportunities')}
 r={k:c[a]/c[b] if c[b] else None for k,(a,b) in pairs.items()}
 n=c['GT_observations']+c['prediction_observations'];r['IDF1']=2*c['IDTP']/n if n else None
 return r

def assert_close(a,b):
 if a is None or b is None:assert a is None and b is None
 else:assert math.isclose(a,b,abs_tol=1e-12,rel_tol=0),(a,b)

note('start')
try:
 assert shutil.disk_usage(ROOT).free>6*1024**3
 for scene in json.loads((mapping/'release.json').read_text())['scenes']:
  assert time.monotonic()-start<900
  path=mapping/scene['scene']/'frames.jsonl';assert sha(path)==report['source_frames_sha256'][str(path)]
  frames=[json.loads(x) for x in path.open()];checked['frames']+=len(frames);checked['scenes']+=1
  for method in methods:
   returned,events,tails=recheck_returns(frames,method)
   recorded=json.loads((RUN/scene['scene']/f'{method}_returns.json').read_text())
   for k,v in recorded['counts'].items():assert v==returned[k],(scene['scene'],method,k,v,returned[k])
   actual_events={(e['kind'],e['GT_instance'],e['left_frame'],e['right_frame']):e for e in recorded['events']}
   assert set(events)==set(actual_events)
   for key,fields in events.items():
    for k,v in fields.items():assert actual_events[key][k]==v,(key,k,v,actual_events[key][k])
   actual_tails={e['GT_instance']:e for e in recorded['censored_tails']};assert set(tails)==set(actual_tails)
   for key,fields in tails.items():
    for k,v in fields.items():assert actual_tails[key][k]==v
   checked['return_events']+=len(events);checked['censored_tails']+=len(tails)
   gids=sorted({identity(o['instance_token']) for f in frames for o in f['evaluation_only_official']})
   pids=sorted({pid(o,method,f['frame_idx']) for f in frames for o in f['predictions'][method]})
   gi={v:i for i,v in enumerate(gids)};pi={v:i for i,v in enumerate(pids)}
   for support in supports:
    gt_indices=[];pred_indices=[];scores=[];support_count=Counter(dict.fromkeys(['IDTP','IDFN','IDFP','GT_observations','prediction_observations','ambiguous_GT_observations','ambiguous_prediction_observations','gated_edges','supported_edges'],0));potential=Counter()
    for f in frames:
     gt=f['evaluation_only_official'];pred=f['predictions'][method]
     valid=[[gate(g,p) for p in pred] for g in gt]
     dg=[sum(x) for x in valid];dp=[sum(valid[i][j] for i in range(len(gt))) for j in range(len(pred))]
     support_count['GT_observations']+=len(gt);support_count['prediction_observations']+=len(pred)
     support_count['ambiguous_GT_observations']+=sum(x>1 for x in dg)
     support_count['ambiguous_prediction_observations']+=sum(x>1 for x in dp)
     support_count['gated_edges']+=sum(dg)
     similarity=np.zeros((len(gt),len(pred)))
     for i,g in enumerate(gt):
      for j,p in enumerate(pred):
       if valid[i][j] and (support=='all_gated' or dg[i]==dp[j]==1):
        similarity[i,j]=1;support_count['supported_edges']+=1
        potential[identity(g['instance_token']),pid(p,method,f['frame_idx'])]+=1
     gt_indices.append(np.array([gi[identity(g['instance_token'])] for g in gt],dtype=int))
     pred_indices.append(np.array([pi[pid(p,method,f['frame_idx'])] for p in pred],dtype=int))
     scores.append(similarity)
    data=dict(num_gt_dets=support_count['GT_observations'],num_tracker_dets=support_count['prediction_observations'],
      num_gt_ids=len(gids),num_tracker_ids=len(pids),gt_ids=gt_indices,tracker_ids=pred_indices,similarity_scores=scores)
    official=reference.eval_sequence(data)
    for k in ('IDTP','IDFN','IDFP'):support_count[k]=int(official[k])
    actual=json.loads((RUN/scene['scene']/f'{method}_{support}.json').read_text())
    assert dict(support_count)==actual['counts']
    assert len({a['GT_instance'] for a in actual['assignments']})==len(actual['assignments'])
    assert len({a['prediction_identity'] for a in actual['assignments']})==len(actual['assignments'])
    assert sum(a['overlap_observations'] for a in actual['assignments'])==support_count['IDTP']
    for a in actual['assignments']:
     assert potential[a['GT_instance'],a['prediction_identity']]==a['overlap_observations']
    row=rowmap[scene['scene'],method,support];expected=dict(**dict(support_count),**recorded['counts'])
    assert expected==row['counts'];by_log[scene['log_token'],method,support].update(expected)
    checked['identity_cases']+=1
  assert sha(path)==report['source_frames_sha256'][str(path)]
 assert checked['scenes']==38 and checked['frames']==1508 and checked['identity_cases']==532
 for entry in report['official_return_gap_groups']:
  assert {k:v for k,v in entry.items() if k not in ('method','gap_bucket')}==dict(buckets[entry['method'],entry['gap_bucket']])
 logs=sorted({x[0] for x in by_log});assert len(logs)==11
 draw=np.random.default_rng(17).integers(11,size=(4000,11));bootstrap={};totals={};metrics={}
 keys=list(rows[0]['counts'])
 for method in methods:
  for support in supports:
   values=np.array([[by_log[log,method,support][k] for k in keys] for log in logs])
   total=dict(zip(keys,map(int,values.sum(0))));assert total==report['methods'][method][support]['counts']
   if support=='all_gated':assert total['GT_observations']==5723 and total['official_return_opportunities']==167
   totals[method,support]=total;rates=norm(total);metrics[method,support]=rates
   samples=defaultdict(list)
   for vals in values[draw].sum(1):
    for k,v in norm(dict(zip(keys,vals))).items():samples[k].append(np.nan if v is None else v)
   for k,v in rates.items():
    actual=report['methods'][method][support]['metrics'][k];assert_close(actual['value'],v)
    bootstrap[method,support,k]=np.array(samples[k]);finite=np.array(samples[k]);finite=finite[np.isfinite(finite)]
    assert actual['valid_resamples']==len(finite)
    ci=np.quantile(finite,[.025,.975]).tolist() if len(finite) else None
    assert ci==actual['ci95'];checked['estimate_intervals']+=1
 for diff in report['paired_differences']:
  m,s,k=diff['method'],diff['support'],diff['metric'];a,b=metrics[m,s][k],metrics['original_fusion',s][k]
  assert_close(diff['difference'],a-b if a is not None and b is not None else None)
  samples=bootstrap[m,s,k]-bootstrap['original_fusion',s,k];samples=samples[np.isfinite(samples)]
  assert diff['valid_resamples']==len(samples)
  assert diff['ci95']==(np.quantile(samples,[.025,.975]).tolist() if len(samples) else None)
  checked['paired_difference_intervals']+=1
 with (RUN/'by_log.csv').open() as handle:
  logrows=list(csv.DictReader(handle))
 for row in logrows:
  expected=by_log[row['log_token'],row['method'],row['support']]
  for k in keys:assert int(row[k])==expected[k]
 assert len(logrows)==len(by_log)==154
 result=dict(passed=True,checked=dict(checked),by_log_rows=len(logrows),official_commit=manifest['commit'],
  original_reference_files_preserved=True,compatibility_changes=changes,source_report_sha256=sha(RUN/'report.json'),
  source_scene_rows_sha256=sha(RUN/'scene_method_support.jsonl'),audit_source_sha256=sha(__file__),
  independent_checks=['math.hypot中心门限与所有合法边','官方TrackEval扩展FN/FP身份代价矩阵','保存逐帧匹配上的独立返回/尾段枚举','11log固定重采样与全部估计/差值区间','逐log CSV'],
  limits=['原逐帧一对一匹配复用，未宣称第二份独立Hungarian关联实现','全序列评价用未来GT只在离线评价，不进入因果推理','不是人工身份/遮挡审核，也不是官方nuScenes成绩'],
  CPU_shared=True,elapsed_wall_s=time.monotonic()-start,model_calls=0,goal_complete=False)
 save(DEST/'report.json',result);print(json.dumps(result,ensure_ascii=False))
except BaseException as exc:
 save(DEST/'interrupted.json',dict(error=repr(exc),checked=dict(checked),goal_complete=False));raise
finally:note('end')
