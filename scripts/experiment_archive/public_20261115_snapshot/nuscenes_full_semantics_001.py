"""38场景六条件逐帧语义：源文件引用+输入指纹，真实路由Qwen，官方标签仅评价。"""
import ctypes
ctypes.CDLL('/root/miniconda3/envs/conceptgraph/lib/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
import argparse,csv,gzip,hashlib,json,resource,shutil,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import PROJECT,OUTPUT as OLD,available_development,dump,digest
from real_graph_contract_003 import build_completion_prompt,reliable,validate
from real_graph_cooperation_001 import choose,complete_geometry_rule,categories

R=Path(__file__).resolve().parents[1];Q=R/'qwen_adapter_fair_comparison';O=R/'nuscenes_ablation/full_semantics_001'
C={'car':'小汽车','truck':'卡车','bus':'公交车','pedestrian':'行人','bicycle':'自行车','motorcycle':'摩托车'}
KINDS=['simple_category','complex_relation','negation','similar_instances','multiple_targets','no_match','insufficient_evidence','stale_information','conflicting_information']
CONDITIONS=['full','without_depth_quality','without_one_to_one','without_dynamic_maintenance','without_short_term_expiry','without_relations']
MODEL=PROJECT/'checkpoint/Qwen2-VL-2B-Instruct'

def guard(reserve=0):
 size=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
 if size+reserve>=10*2**30 or shutil.disk_usage(R).free<20*2**30:raise OSError('遵守10GiB新增产物和20GiB物理空闲守卫')

def source_paths():
 return [OLD/'mapping_evaluation/causal_mapping_full_001'/s['scene']/'frames.jsonl' for s in available_development()]+[R/'nuscenes_ablation'/f'strict_{c}_001'/s['scene']/'frames.jsonl' for c in CONDITIONS for s in available_development()]

def task(frame,scene):
 pose=np.asarray(frame['camera_pose']);origin=pose[:2,3];forward=pose[:2,2].copy();forward/=np.linalg.norm(forward);left=np.array([-forward[1],forward[0]])
 def xy(g):
  d=np.asarray(g['center'][:2])-origin;return np.array([d@forward,d@left])
 truth=[g for g in frame['evaluation_only_official'] if g['category'] in C];truth=sorted(truth,key=lambda g:(float(np.linalg.norm(xy(g))),g['instance_token']));counts=Counter(g['category'] for g in truth);kind=KINDS[frame['frame_idx']%9];seed=int(hashlib.sha256(f"full-semantic-v1/{scene['scene']}/{frame['frame_idx']}".encode()).hexdigest()[:12],16);cat=sorted(counts)[seed%len(counts)] if counts else 'car';accepted=[];fault=kind if kind in KINDS[6:] else None
 if kind in ['simple_category']+KINDS[6:]:
  query='选择最近的'+C[cat];same=[g for g in truth if g['category']==cat];accepted=[same[0]['instance_token']] if same else []
  if fault:query='只有当前证据可靠且目标静止时才授权'+query
 elif kind=='complex_relation':
  pairs=[]
  for g in truth:
   for a in truth:
    if g['instance_token']==a['instance_token']:continue
    d=xy(g)-xy(a)
    if np.linalg.norm(d)>12:continue
    for axis in [0,1]:
     if abs(d[axis])>1:pairs.append((g['category'],a['category'],axis,bool(d[axis]>0)))
  target,anchor,axis,positive=pairs[seed%len(pairs)] if pairs else ('car','bus',1,True);direction=(('左侧' if positive else '右侧') if axis==1 else ('前方' if positive else '后方'));query=f'以{C[anchor]}为参照，选择其{direction}最近的{C[target]}'
  candidates=[g for g in truth if g['category']==target and any(a['category']==anchor and a['instance_token']!=g['instance_token'] and np.linalg.norm(xy(g)-xy(a))<=12 and abs((xy(g)-xy(a))[axis])>1 and bool((xy(g)-xy(a))[axis]>0)==positive for a in truth)];accepted=[candidates[0]['instance_token']] if candidates else []
 elif kind in ['negation','similar_instances']:
  repeated=sorted(c for c,n in counts.items() if n>=2);cat=repeated[seed%len(repeated)] if repeated else 'car';same=[g for g in truth if g['category']==cat];accepted=[same[1]['instance_token']] if len(same)>=2 else [];query=f'不要最近的{C[cat]}，选择第二近的{C[cat]}' if kind=='negation' else f'在所有{C[cat]}中按距离选择第二近的一辆'
 elif kind=='multiple_targets':
  cats=sorted(counts);cats=(cats[seed%len(cats):]+cats[:seed%len(cats)])[:2] if cats else ['car','bus'];query='在'+'或'.join(C[c] for c in cats)+'中选择最近的对象';same=[g for g in truth if g['category'] in cats];accepted=[same[0]['instance_token']] if same else []
 else:
  missing=sorted(set(C)-set(counts));cat=missing[seed%len(missing)] if missing else 'motorcycle';query='选择最近的'+C[cat];same=[g for g in truth if g['category']==cat];accepted=[same[0]['instance_token']] if same else []
 return dict(query_kind=kind,query_text=query,controlled_fault=fault,official_positive=bool(accepted),official_acceptable_instances=accepted),origin,forward,left

def iter_rows():
 split=json.loads((Q/'official_graph_dataset_002/release.json').read_text())['split_logs']
 for condition in CONDITIONS:
  for scene in available_development():
   original={f['frame_idx']:f for f in map(json.loads,(OLD/'mapping_evaluation/causal_mapping_full_001'/scene['scene']/'frames.jsonl').open())};registry={};role=next(k for k,logs in split.items() if scene['log_token'] in logs)
   for f in map(json.loads,(R/'nuscenes_ablation'/f'strict_{condition}_001'/scene['scene']/'frames.jsonl').open()):
    idx=f['frame_idx'];base=original[idx];assert f['evaluation_only_official']==base['evaluation_only_official'] and not f['gt_in_inference'];objects=f['predictions'][condition]
    for obj in objects:registry.setdefault(obj['node_id'],len(registry))
    assert set(map(int,f['nodes']))=={registry[obj['node_id']] for obj in objects}
    query,origin,forward,left=task(base,scene);now=f['camera_timestamp_s'];nodes=[];ids={};used=set()
    for obj in sorted(objects,key=lambda x:x['node_id']):
     nid=100+int(hashlib.sha256(f"full-semantic-ID-v1/{scene['scene']}/{idx}/{obj['node_id']}".encode()).hexdigest()[:8],16)%900
     while nid in used:nid=100+(nid-99)%900
     used.add(nid);ids[registry[obj['node_id']]]=nid;delta=np.asarray(obj['center'][:2])-origin;e=(obj.get('depth_evidence') or [{}])[-1];assert np.allclose(f['nodes'][str(registry[obj['node_id']])]['center'],obj['center'])
     nodes.append(dict(node_id=nid,category=obj['category'],center_xy=[round(float(delta@forward),3),round(float(delta@left),3)],distance_m=round(float(np.linalg.norm(delta)),3),motion_state=obj['motion_state'],version=idx+1,now_s=now,last_seen_s=obj.get('timestamp_s') if obj.get('timestamp_s') is not None else -1,evidence=dict(source=e.get('source','unknown'),version=idx+1,observations=obj['num_detections'],lidar_pixels=int(e.get('retained_pixels',0)),support_fraction=float(e.get('support_fraction',0)),conflict=not bool(e.get('sufficient_support',False)))))
    requested={c for c,w in C.items() if w in query['query_text']};pool=sorted(nodes,key=lambda n:(n['category'] not in requested,n['distance_m'],n['node_id']))[:8];kept={n['node_id'] for n in pool};pool=sorted(pool,key=lambda n:hashlib.sha256(f"full-semantic-pool-v1/{scene['scene']}/{idx}/{n['node_id']}".encode()).hexdigest())
    fault=query['controlled_fault']
    for node in pool:
     if fault=='insufficient_evidence':node['evidence'].update(observations=1,lidar_pixels=0)
     elif fault=='stale_information':node['last_seen_s']=now-10
     elif fault=='conflicting_information':node['evidence']['conflict']=True
    edges=[e for e in f['edges'] if e['source'] in ids and e['target'] in ids and ids[e['source']] in kept and ids[e['target']] in kept];edges=sorted(edges,key=lambda e:(ids[e['source']],ids[e['target']],e['relation']))[:16];relations=[dict(source=ids[e['source']],target=ids[e['target']],relation=e['relation']) for e in edges]
    binding={ids[registry[objects[m['prediction_index']]['node_id']]]:f['evaluation_only_official'][m['truth_index']]['instance_token'] for m in f['metrics'][condition]['matches']};covered=[n['node_id'] for n in pool if binding.get(n['node_id']) in query['official_acceptable_instances']];acceptable=[None] if fault or not query['official_positive'] else covered
    yield dict(task='candidate_rerank',condition=condition,sample_id=f"full_semantic_{scene['scene']}_{idx:03d}",scene_id=scene['scene'],frame_idx=idx,log_token=scene['log_token'],source_role=role,**query,candidate_nodes=pool,candidate_ids=[n['node_id'] for n in pool],relation_metadata=relations,acceptable_node_ids=acceptable,candidate_target_coverage=bool(covered),actual_authorization_credit=False,actual_arrival_evaluated=False)

def prompt_hash(row):return hashlib.sha256(build_completion_prompt('candidate_rerank',row).encode()).hexdigest()

def prepare():
 from transformers import AutoProcessor
 from train_real_graph_adapter_004 import inputs_for
 guard(1500*1024);O.mkdir(exist_ok=False);sources={str(p):digest(p) for p in source_paths()};processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True);counts=Counter();routes=Counter();positive=Counter();max_tokens=0;relation_pairs={};token_counts=[]
 with gzip.open(O/'input_fingerprints.jsonl.gz','xt',compresslevel=6) as out:
  for row in iter_rows():
   size=int(inputs_for(processor,row)['input_ids'].shape[-1]);assert size<=2048;max_tokens=max(max_tokens,size);pre=choose(row['query_text'],row['candidate_nodes'],row['relation_metadata'],None);needed=pre['backend_output_used'];out.write(json.dumps(dict(condition=row['condition'],sample_id=row['sample_id'],prompt_sha256=prompt_hash(row),input_tokens=size,needs_Qwen=needed),separators=(',',':'))+'\n');counts[row['condition']]+=1;routes[row['condition']]+=needed;positive[row['condition']]+=bool(row['official_positive'] and not row['controlled_fault'])
   if row['condition'] in ['full','without_relations']:
    signature=json.dumps({k:row[k] for k in ['candidate_nodes','candidate_ids','acceptable_node_ids','query_text']},sort_keys=True,separators=(',',':'));h=hashlib.sha256(signature.encode()).hexdigest()
    if row['condition']=='full':relation_pairs[row['sample_id']]=h
    else:assert not row['relation_metadata'] and h==relation_pairs[row['sample_id']]
 assert all(counts[c]==1508 for c in CONDITIONS) and len(relation_pairs)==1508
 train=Q/'training/real_graph_seed17_001/training_report.json';adapter=Path(json.loads(train.read_text())['best_checkpoint']['path']);files=[Path(__file__),R/'source/real_graph_contract_003.py',R/'source/real_graph_cooperation_001.py',R/'source/train_real_graph_adapter_004.py',Q/'official_graph_dataset_002/release.json',train,adapter/'adapter_model.safetensors'];sources.update({str(p):digest(p) for p in files})
 dump(O/'release.json',dict(scenes=38,logs=11,frames_each=1508,conditions=CONDITIONS,requests_total=9048,query_assignment='每帧一条，frame_idx%9循环九层；GT仅生成共同指令/评价目标，不按模型成绩或候选覆盖挑帧；全1508帧无排除',strata=KINDS,common_query_GT_scope='六条件同一帧查询及官方目标一致；需关系/第二近时仅依据官方几何选择词汇，缺失目标保留',candidate_limit=8,relation_limit=16,relations='实际增量维护器保存的边，不从答案重造；去边与完整候选/标签1508帧逐条同一',actual_route='冻结choose清晰规则/无可用候选先判断，仅需要复杂分支才调用Qwen；不是响应缓存或模型调用失败回退',planned_actual_GPU_requests=dict(routes),ordinary_positive_each=dict(positive),adapter=str(adapter),seed=17,max_input_tokens=2048,max_new_tokens=256,do_sample=False,torch_dtype='bfloat16',response_cache=False,max_input_tokens_preflight=max_tokens,input_fingerprints_sha256=digest(O/'input_fingerprints.jsonl.gz'),sources_sha256=sources,storage='复用冻结已有帧，只存可重构输入的逐请求指纹，不复制节点输入/图像/模型',scope='38开发场景全帧，不是新封存；含训练/验证日志，按source_role另报开发留出2日志；仅种子17',whole_project_sealed_test=False,goal_complete=False));print(json.dumps(dict(frames=dict(counts),GPU_routes=dict(routes),max_tokens=max_tokens)),flush=True)

def run():
 import torch
 from transformers import AutoProcessor
 from scripts.train_qwen_lora_semantic_delegate_v2 import _load_model
 from train_real_graph_adapter_004 import inputs_for
 release=json.loads((O/'release.json').read_text());assert all(digest(p)==h for p,h in release['sources_sha256'].items());assert digest(O/'input_fingerprints.jsonl.gz')==release['input_fingerprints_sha256'];guard(1024*1024);torch.manual_seed(17);np.random.seed(17);processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True);model=_load_model(MODEL,Path(release['adapter']),trainable=False);model.eval();torch.cuda.reset_peak_memory_stats();begun=time.monotonic();done=0;calls=0
 with gzip.open(O/'input_fingerprints.jsonl.gz','rt') as ref,gzip.open(O/'decisions.jsonl.gz','xt',compresslevel=6) as output:
  for row in iter_rows():
   if done%100==0:guard(64*1024)
   fingerprint=json.loads(next(ref));assert fingerprint['condition']==row['condition'] and fingerprint['sample_id']==row['sample_id'] and fingerprint['prompt_sha256']==prompt_hash(row);pre=choose(row['query_text'],row['candidate_nodes'],row['relation_metadata'],None);needed=pre['backend_output_used'];assert needed==fingerprint['needs_Qwen'];prediction=None;raw=None;seconds=None
   if needed:
    x=inputs_for(processor,row).to(model.device);assert x['input_ids'].shape[-1]==fingerprint['input_tokens'];torch.cuda.synchronize();tick=time.monotonic()
    with torch.inference_mode():y=model.generate(**x,max_new_tokens=256,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=processor.tokenizer.eos_token_id)
    torch.cuda.synchronize();seconds=time.monotonic()-tick;raw=processor.tokenizer.decode(y[0,x['input_ids'].shape[-1]:],skip_special_tokens=True).strip();valid=validate('candidate_rerank',raw,allowed_candidate_ids=row['candidate_ids']);prediction=dict(schema_valid=valid.valid,normalized_output=valid.normalized_output if valid.valid else None);calls+=1
   decision=choose(row['query_text'],row['candidate_nodes'],row['relation_metadata'],prediction);answer=decision['authorized_node_id'] if decision['explicit_authorization_query'] else decision['proposed_node_id'];rule=complete_geometry_rule(row['query_text'],row['candidate_nodes'],row['relation_metadata']);rule_answer=rule['authorized_node_id'] if rule['explicit_authorization_query'] else rule['proposed_node_id'];requested=categories(row['query_text']);pool=[n for n in row['candidate_nodes'] if not requested or n['category'] in requested];nearest=min(pool,key=lambda n:(n['distance_m'],n['node_id']))['node_id'] if pool else None
   record=dict(condition=row['condition'],sample_id=row['sample_id'],scene=row['scene_id'],log=row['log_token'],role=row['source_role'],kind=row['query_kind'],positive=bool(row['official_positive'] and not row['controlled_fault']),ordinary_no_match=bool(not row['official_positive'] and not row['controlled_fault']),fault=bool(row['controlled_fault']),covered=row['candidate_target_coverage'],answer=answer,authorized=decision['authorized_node_id'],correct=answer in row['acceptable_node_ids'],strong_rule_correct=rule_answer in row['acceptable_node_ids'],nearest_correct=nearest in row['acceptable_node_ids'],no_match_false_proposal=bool(not row['official_positive'] and not row['controlled_fault'] and decision['proposed_node_id'] is not None),Qwen_called=needed,raw_output=raw,Qwen_schema_valid=prediction['schema_valid'] if prediction else None,GPU_latency_s=seconds,prompt_sha256=fingerprint['prompt_sha256'],actual_lock_credit=False,actual_arrival_credit=False)
   output.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');output.flush();done+=1
   if done%100==0:print(json.dumps(dict(done=done,total=9048,GPU_calls=calls,condition=row['condition'])),flush=True)
  assert next(ref,None) is None and done==9048
 dump(O/'run_report.json',dict(actual_CPU_cooperation_decisions=done,actual_new_GPU_requests=calls,preplanned_GPU_requests=sum(release['planned_actual_GPU_requests'].values()),peak_GPU_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,wall_s=time.monotonic()-begun,raw_sha256=digest(O/'decisions.jsonl.gz'),evaluator_source_sha256=digest(__file__),performance='单模型进程；路由调用数随真实候选变化；不是每帧都调用Qwen、不是端到端实时吞吐或物理执行',goal_complete=False));print('all9048 done',flush=True)

def summarize():
 with gzip.open(O/'decisions.jsonl.gz','rt') as f:rows=list(map(json.loads,f))
 assert len(rows)==9048;table=[];paired=[]
 for role in ['all','train','validation','development_holdout']:
  rr=[r for r in rows if role=='all' or r['role']==role];logs=sorted({r['log'] for r in rr});draw=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)))
  for condition in CONDITIONS:
   for scope in ['all','ordinary_positive','ordinary_no_match']+KINDS:
    chosen=[r for r in rr if r['condition']==condition and (scope=='all' or scope=='ordinary_positive' and r['positive'] or scope=='ordinary_no_match' and r['ordinary_no_match'] or r['kind']==scope)]
    if not chosen:continue
    for metric in ['correct','strong_rule_correct','nearest_correct','covered','no_match_false_proposal']:
     a=np.array([[sum(r[metric] for r in chosen if r['log']==log),sum(r['log']==log for r in chosen)] for log in logs]);b=a[draw].sum(1);values=np.divide(b[:,0],b[:,1],out=np.full(len(draw),np.nan),where=b[:,1]>0);valid=values[np.isfinite(values)];low,high=np.percentile(valid,[2.5,97.5]) if len(valid) else (None,None);num=sum(r[metric] for r in chosen);table.append(dict(condition=condition,role=role,scope=scope,metric=metric,numerator=num,denominator=len(chosen),value=num/len(chosen),ci95_low=float(low) if low is not None else None,ci95_high=float(high) if high is not None else None,logs=len(logs)))
  for condition in CONDITIONS[1:]:
   for scope in ['all','ordinary_positive']:
    aa={r['sample_id']:r for r in rr if r['condition']=='full' and (scope=='all' or r['positive'])};bb={r['sample_id']:r for r in rr if r['condition']==condition and (scope=='all' or r['positive'])};assert set(aa)==set(bb);ids=sorted(aa)
    if not ids:continue
    a=np.array([[sum(int(aa[s]['correct'])-int(bb[s]['correct']) for s in ids if aa[s]['log']==log),sum(aa[s]['log']==log for s in ids)] for log in logs]);b=a[draw].sum(1);v=np.divide(b[:,0],b[:,1],out=np.full(len(draw),np.nan),where=b[:,1]>0);finite=v[np.isfinite(v)];low,high=np.percentile(finite,[2.5,97.5]);paired.append(dict(reference=condition,role=role,scope=scope,denominator=len(ids),full_minus_reference=float(a[:,0].sum()/len(ids)),ci95_low=float(low),ci95_high=float(high),logs=len(logs)))
 for name,data in [('metrics.csv',table),('paired_differences.csv',paired)]:
  with (O/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
 dump(O/'report.json',dict(actual_requests=9048,conditions=6,frames_each=1508,scenes=38,logs=11,actual_new_mapper_calls=0,run=json.loads((O/'run_report.json').read_text()),release_sha256=digest(O/'release.json'),source_sha256=digest(__file__),bootstrap=dict(unit='log',repeats=4000,seed=17),limitations=['38开发场景包含训练/验证日志，不能作独立整系统新封存；source_role另列，开发留出仅2日志','每帧固定循环九层只产生一条查询，非每帧九条；所有6条件共享同一帧查询/目标，未按结果筛样','无匹配层若6类全出现则如实按官方目标标签报告，不能强造负标签','官方类别/XY配对和受控语法/故障，不是人工细属性/自然指令；当前帧版本不是全局真实身份保障','路由按真实规则/候选需求调用Qwen，未调用的分支不记GPU，不把内部规则有效率说成语言模型格式率','输入由冻结已有帧和此源码可重构，逐请求指纹核验，不重复写节点图像','单个固定种子17实际整链语义；旧三种子同108请求语义消融另列','无新的在线锁定、控制或到达信用，物理闭环仍看官方CARLA独立证据'],whole_project_sealed_test=False,goal_complete=False));print('full six condition semantic statistics complete',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','summarize']);args=p.parse_args()
 try:globals()[args.action]()
 except BaseException as exc:
  import traceback
  if O.exists():dump(O/f'failure_{args.action}.json',dict(error=repr(exc),traceback=traceback.format_exc(),goal_complete=False))
  raise
