"""六条件实际维护关系边接冻结真实图Qwen；108条已开发请求，非新封存或物理闭环。"""
import ctypes
ctypes.CDLL('/root/miniconda3/envs/conceptgraph/lib/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
import argparse,copy,csv,gzip,hashlib,json,resource,shutil,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import PROJECT,OUTPUT as OLD,dump,digest
from real_graph_cooperation_001 import choose,CONDITIONS as SEMANTIC_CONDITIONS
from real_graph_contract_003 import validate,reliable

R=Path(__file__).resolve().parents[1];Q=R/'qwen_adapter_fair_comparison';O=R/'nuscenes_ablation/complex_semantics_development_001';D=Q/'real_graph_dataset_003/development_holdout.jsonl'
CONDITIONS=['full','without_depth_quality','without_one_to_one','without_dynamic_maintenance','without_short_term_expiry','without_relations']
MODEL=PROJECT/'checkpoint/Qwen2-VL-2B-Instruct'

def guard(reserve=0):
 total=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
 if total+reserve>10*2**30 or shutil.disk_usage(R).free<20*2**30:raise OSError('遵守10GiB产物/20GiB空闲守卫，不开启新请求')

def read_rows():
 with gzip.open(O/'inputs.jsonl.gz','rt') as f:return list(map(json.loads,f))

def prepare():
 guard(1024**2);O.mkdir(exist_ok=False);original=list(map(json.loads,D.open()));byframe=defaultdict(list)
 for row in original:byframe[row['scene_id'],row['frame_idx']].append(row)
 sources={str(D):digest(D)};poses={};base_truth={}
 for scene in sorted({s for s,_ in byframe}):
  p=OLD/'mapping_evaluation/causal_mapping_full_001'/scene/'frames.jsonl';sources[str(p)]=digest(p)
  for f in map(json.loads,p.open()):
   if (scene,f['frame_idx']) in byframe:poses[scene,f['frame_idx']]=f['camera_pose'];base_truth[scene,f['frame_idx']]=f['evaluation_only_official']
 allrows=[]
 for condition in CONDITIONS:
  for scene in sorted({s for s,_ in byframe}):
   p=R/'nuscenes_ablation'/f'strict_{condition}_001'/scene/'frames.jsonl';sources[str(p)]=digest(p);registry={}
   for f in map(json.loads,p.open()):
    objects=f['predictions'][condition]
    for obj in objects:registry.setdefault(obj['node_id'],len(registry))
    key=scene,f['frame_idx']
    if key not in byframe:continue
    assert f['evaluation_only_official']==base_truth[key] and not f['gt_in_inference']
    assert set(map(int,f['nodes']))=={registry[o['node_id']] for o in objects}
    pose=np.asarray(poses[key]);origin=pose[:2,3];forward=pose[:2,2].copy();forward/=np.linalg.norm(forward);left=np.asarray([-forward[1],forward[0]]);now=f['camera_timestamp_s'];ids={};used=set();nodes=[]
    for obj in sorted(objects,key=lambda o:o['node_id']):
     # 身份仅为该条件的预测轨迹编号；哈希不读GT/标签/模型结果，碰撞确定性递增。
     nid=100+int(hashlib.sha256(f"six-actual-edge-v1/{scene}/{f['frame_idx']}/{obj['node_id']}".encode()).hexdigest()[:8],16)%900
     while nid in used:nid=100+(nid-99)%900
     used.add(nid);ids[registry[obj['node_id']]]=nid;delta=np.asarray(obj['center'][:2])-origin;e=(obj.get('depth_evidence') or [{}])[-1]
     assert np.allclose(f['nodes'][str(registry[obj['node_id']])]['center'],obj['center'])
     nodes.append(dict(node_id=nid,category=obj['category'],center_xy=[round(float(delta@forward),3),round(float(delta@left),3)],distance_m=round(float(np.linalg.norm(delta)),3),motion_state=obj['motion_state'],version=f['frame_idx']+1,now_s=now,last_seen_s=obj.get('timestamp_s') if obj.get('timestamp_s') is not None else -1,evidence=dict(source=e.get('source','unknown'),version=f['frame_idx']+1,observations=obj['num_detections'],lidar_pixels=int(e.get('retained_pixels',0)),support_fraction=float(e.get('support_fraction',0)),conflict=not bool(e.get('sufficient_support',False)))))
    bypred={m['prediction_index']:m['truth_index'] for m in f['metrics'][condition]['matches']};binding={ids[registry[objects[pi]['node_id']]]:f['evaluation_only_official'][ti]['instance_token'] for pi,ti in bypred.items()}
    for source in byframe[key]:
     row=copy.deepcopy(source);row['mapping_condition']=condition;row['candidate_nodes']=copy.deepcopy(nodes);requested={c for c,w in [('car','小汽车'),('truck','卡车'),('bus','公交车'),('pedestrian','行人'),('bicycle','自行车'),('motorcycle','摩托车')] if w in row['query_text']};preferred=sorted(row['candidate_nodes'],key=lambda n:(n['category'] not in requested,n['distance_m'],n['node_id']))[:8];kept={n['node_id'] for n in preferred}
     row['candidate_nodes']=[n for n in row['candidate_nodes'] if n['node_id'] in kept];row['candidate_nodes'].sort(key=lambda n:hashlib.sha256(f"six-pool-v1/{scene}/{f['frame_idx']}/{n['node_id']}".encode()).hexdigest())
     fault=source['controlled_fault']
     for node in row['candidate_nodes']:
      if fault=='insufficient_evidence':node['evidence'].update(observations=1,lidar_pixels=0)
      elif fault=='stale_information':node['last_seen_s']=now-10
      elif fault=='conflicting_information':node['evidence']['conflict']=True
     # 直接转接图维护器保存的边；不从GT或候选坐标重新创造方向边。
     indexed=[(i,e) for i,e in enumerate(f['edges']) if e['source'] in ids and e['target'] in ids and ids[e['source']] in kept and ids[e['target']] in kept]
     indexed=sorted(indexed,key=lambda t:(ids[t[1]['source']],ids[t[1]['target']],t[1]['relation']))[:16]
     row['relation_metadata']=[dict(source=ids[e['source']],target=ids[e['target']],relation=e['relation']) for _,e in indexed];row['maintained_edge_indices_evaluation_only']=[i for i,_ in indexed];row['candidate_ids']=[n['node_id'] for n in row['candidate_nodes']]
     accepted=source['official_acceptable_instances'];covered=[n['node_id'] for n in row['candidate_nodes'] if binding.get(n['node_id']) in accepted]
     row['acceptable_node_ids']=[None] if fault or not accepted else covered;row['candidate_target_coverage']=bool(covered) if accepted else None;row['official_match_binding_evaluation_only']=binding;row['reliable_target_coverage']=any(n['node_id'] in covered and reliable(n) for n in row['candidate_nodes']) if accepted else None
     # 不携带原候选的训练答案/旧身份映射：推理只看contract指定的三个输入字段。
     row.pop('response',None);row.pop('predicted_identity_binding',None);row.pop('expected_proposal_or_explicit_authorization_answer',None);row['source_frame_file']=str(p);allrows.append(row)
 assert len(allrows)==648
 paired={(r['mapping_condition'],r['sample_id']):r for r in allrows}
 for source in original:
  a=paired['full',source['sample_id']];b=paired['without_relations',source['sample_id']]
  for k in ['candidate_nodes','candidate_ids','acceptable_node_ids','query_text']:assert a[k]==b[k],('relation isolation mismatch',source['sample_id'],k)
  assert b['relation_metadata']==[]
 with gzip.open(O/'inputs.jsonl.gz','xt',compresslevel=6) as f:
  for row in allrows:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
 train=Q/'training/real_graph_seed17_001/training_report.json';checkpoint=Path(json.loads(train.read_text())['best_checkpoint']['path']);sources[str(train)]=digest(train);sources[str(checkpoint/'adapter_model.safetensors')]=digest(checkpoint/'adapter_model.safetensors')
 dump(O/'release.json',dict(conditions=CONDITIONS,requests_each=108,total_requests=648,source_frames_each=12,source_scenes_each=2,source_logs_each=2,same_existing_queries_and_official_targets=True,selection='原108全部保留，不按六组模型成绩/候选覆盖选择；不同建图条件真实候选变化正是被测中介',candidate_limit=8,actual_maintainer_edges=True,relation_limit=16,relation_selection='按预测source/target/relation排序取前16，单条无向near不补成虚构有向边',candidate_ID='GT无关scene/frame/raw预测轨迹哈希100—999，碰撞递增；跨条件相同数字不代表真实身份相同',adapter=str(checkpoint),sources_sha256=sources,source_sha256={str(p):digest(p) for p in [Path(__file__),R/'source/real_graph_contract_003.py',R/'source/real_graph_cooperation_001.py',R/'source/train_real_graph_adapter_004.py']},max_input_tokens=2048,max_new_tokens=256,seed=17,do_sample=False,torch_dtype='bfloat16',response_cache=False,new_sealed_test=False,goal_complete=False));print('registered648; relation-only candidates identical',flush=True)

def gpu():
 import torch
 from transformers import AutoProcessor
 from scripts.train_qwen_lora_semantic_delegate_v2 import _load_model
 from train_real_graph_adapter_004 import inputs_for
 guard(512*1024);release=json.loads((O/'release.json').read_text());assert all(digest(p)==h for p,h in release['source_sha256'].items());assert all(digest(p)==h for p,h in release['sources_sha256'].items());rows=read_rows();processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
 # 全部输入先验预算核查，不截断、不挑掉超长样本；失败时不开始GPU。
 lengths=[int(inputs_for(processor,row)['input_ids'].shape[-1]) for row in rows];assert max(lengths)<=2048
 dump(O/'input_audit.json',dict(rows=len(rows),max_input_tokens=max(lengths),no_truncation=True,input_sha256=digest(O/'inputs.jsonl.gz'),goal_complete=False))
 torch.manual_seed(17);np.random.seed(17);model=_load_model(MODEL,Path(release['adapter']),trainable=False);model.eval();torch.cuda.reset_peak_memory_stats();begun=time.monotonic()
 with gzip.open(O/'raw_GPU_records.jsonl.gz','xt',compresslevel=6) as f:
  for i,row in enumerate(rows):
   if i%27==0:guard(128*1024)
   x=inputs_for(processor,row).to(model.device);torch.cuda.synchronize();t=time.monotonic()
   with torch.inference_mode():y=model.generate(**x,max_new_tokens=256,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=processor.tokenizer.eos_token_id)
   torch.cuda.synchronize();latency=time.monotonic()-t;raw=processor.tokenizer.decode(y[0,x['input_ids'].shape[-1]:],skip_special_tokens=True).strip();v=validate('candidate_rerank',raw,allowed_candidate_ids=row['candidate_ids']);out=v.normalized_output if v.valid else None
   rec=dict(mapping_condition=row['mapping_condition'],sample_id=row['sample_id'],raw_output=raw,normalized_output=out,schema_valid=v.valid,semantic_correct=bool(v.valid and out['selected_node_id'] in row['acceptable_node_ids']),input_tokens=lengths[i],latency_s=latency,backend_success=True,fallback_used=False);f.write(json.dumps(rec,ensure_ascii=False,separators=(',',':'))+'\n');f.flush()
   if (i+1)%27==0:print(json.dumps(dict(completed=i+1,total=648,condition=row['mapping_condition'])),flush=True)
 dump(O/'GPU_report.json',dict(actual_new_GPU_requests=len(rows),wall_s=time.monotonic()-begun,peak_GPU_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,raw_sha256=digest(O/'raw_GPU_records.jsonl.gz'),sequential_one_process=True,performance_scope='同一冻结模型、单GPU；非独立逐条件模型进程/非端到端实时吞吐',goal_complete=False))

def summarize():
 with gzip.open(O/'raw_GPU_records.jsonl.gz','rt') as f:raw=list(map(json.loads,f))
 rows=read_rows();by={(r['mapping_condition'],r['sample_id']):r for r in rows};assert len(raw)==648 and len({(a['mapping_condition'],a['sample_id']) for a in raw})==648;decisions=[]
 for a in raw:
  row=by[a['mapping_condition'],a['sample_id']];decision=choose(row['query_text'],row['candidate_nodes'],row['relation_metadata'],a);answer=decision['authorized_node_id'] if decision['explicit_authorization_query'] else decision['proposed_node_id'];decisions.append(dict(condition=row['mapping_condition'],sample_id=row['sample_id'],log=row['log_token'],kind=row['query_kind'],normal_positive=row['controlled_fault'] is None and row['official_positive'],candidate_covered=bool(row['candidate_target_coverage']),raw_correct=a['semantic_correct'],cooperation_correct=bool(answer in row['acceptable_node_ids']),raw_format_valid=a['schema_valid'],relation_count=len(row['relation_metadata']),proposed_node_id=decision['proposed_node_id'],authorized_node_id=decision['authorized_node_id'],decision=decision,latency_s=a['latency_s'],actual_lock_credit=False,actual_arrival_credit=False))
 with gzip.open(O/'cooperation_decisions.jsonl.gz','xt',compresslevel=6) as f:
  for d in decisions:f.write(json.dumps(d,ensure_ascii=False,separators=(',',':'))+'\n')
 logs=sorted({d['log'] for d in decisions});draw=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)));table=[];paired=[]
 for condition in CONDITIONS:
  for scope in ['all','ordinary_positive']+sorted({r['kind'] for r in rows}):
   rr=[r for r in decisions if r['condition']==condition and (scope=='all' or (scope=='ordinary_positive' and r['normal_positive']) or r['kind']==scope)]
   for metric in ['raw_correct','cooperation_correct','raw_format_valid','candidate_covered']:
    counts=np.asarray([[sum(r[metric] for r in rr if r['log']==l),sum(r['log']==l for r in rr)] for l in logs]);b=counts[draw].sum(1);v=np.divide(b[:,0],b[:,1],out=np.full(len(draw),np.nan),where=b[:,1]>0);lo,hi=np.nanpercentile(v,[2.5,97.5]);num=sum(r[metric] for r in rr);table.append(dict(condition=condition,scope=scope,metric=metric,numerator=num,denominator=len(rr),value=num/len(rr),ci95_low=float(lo),ci95_high=float(hi),logs=2,frames_each=12,scenes_each=2,actual_locks=0,actual_arrivals=0))
 for condition in CONDITIONS[1:]:
  for scope in ['all','ordinary_positive']:
   aa={r['sample_id']:r for r in decisions if r['condition']=='full' and (scope=='all' or r['normal_positive'])};bb={r['sample_id']:r for r in decisions if r['condition']==condition and (scope=='all' or r['normal_positive'])};ids=sorted(aa);assert set(aa)==set(bb)
   for metric in ['raw_correct','cooperation_correct']:
    count=np.asarray([[sum(int(aa[s][metric])-int(bb[s][metric]) for s in ids if aa[s]['log']==l),sum(aa[s]['log']==l for s in ids)] for l in logs]);b=count[draw].sum(1);v=np.divide(b[:,0],b[:,1],out=np.full(len(draw),np.nan),where=b[:,1]>0);lo,hi=np.nanpercentile(v,[2.5,97.5]);paired.append(dict(method='full',reference=condition,scope=scope,metric=metric,denominator=len(ids),absolute_delta=float(count[:,0].sum()/len(ids)),paired_log_ci95_low=float(lo),paired_log_ci95_high=float(hi)))
 for name,rr in [('metrics.csv',table),('paired_differences.csv',paired)]:
  with (O/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
 dump(O/'report.json',dict(actual_new_GPU_requests=648,actual_new_CPU_cooperation_decisions=648,conditions=6,requests_each=108,ordinary_positive_each=60,source_frames_each=12,source_scenes_each=2,source_logs_each=2,actual_new_mapper_calls=0,actual_maintained_edges_used=True,release_sha256=digest(O/'release.json'),inputs_sha256=digest(O/'inputs.jsonl.gz'),raw_sha256=digest(O/'raw_GPU_records.jsonl.gz'),cooperation_sha256=digest(O/'cooperation_decisions.jsonl.gz'),source_sha256=digest(__file__),bootstrap=dict(unit='log',units=2,repeats=4000,seed=17),limitations=['仅已固定108条开发子集，不等于38场景1508帧全量复杂语义验收','各建图候选真实变化；相同查询/官方目标，但非不同语言模型同候选公平比较表','候选ID/顺序按GT无关新协议生成，实际维护器边与此前候选坐标派生边输入分布不同，不能直接与旧108原始分数作因果差异','无关系条件仅去实际关系边且候选/标签逐条相同；模型仍能读取坐标，结果相同不能声称关系维护有贡献','三个故障层是明确注入压力测试，普通提议与静止授权分开','未进行新的跨帧锁定或控制，无实际到达信用；仅单个事前固定种子17','仅2个已开发日志，CI离散，非新封存道路测试'],new_sealed_test=False,goal_complete=False));print('648 GPU +648 CPU complete',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','gpu','summarize']);args=p.parse_args()
 try:globals()[args.action]()
 except BaseException as exc:
  import traceback
  if O.exists():dump(O/f'failure_{args.action}.json',dict(error=repr(exc),traceback=traceback.format_exc(),goal_complete=False))
  raise
