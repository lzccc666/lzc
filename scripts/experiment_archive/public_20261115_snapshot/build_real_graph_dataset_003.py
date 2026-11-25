"""实际感知候选与独立官方目标标签；受控故障单列，不把给定GT图成绩转用。"""
import copy,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import OUTPUT as OLD,digest,dump,available_development
from real_graph_contract_002 import reliable
R=Path(__file__).resolve().parents[1];O=R/'qwen_adapter_fair_comparison/real_graph_dataset_003'
C={'car':'小汽车','truck':'卡车','bus':'公交车','pedestrian':'行人','bicycle':'自行车','motorcycle':'摩托车'}
KINDS=['simple_category','complex_relation','negation','similar_instances','multiple_targets','no_match','insufficient_evidence','stale_information','conflicting_information']
def make(frame,scene,role,motion):
 pose=np.asarray(frame['camera_pose']);origin=pose[:2,3];forward=pose[:2,2];forward/=np.linalg.norm(forward);left=np.array([-forward[1],forward[0]]);now=frame['camera_timestamp_s'];observed_gt=[(i,g) for i,g in enumerate(frame['evaluation_only_official']) if g['category'] in C];counts=Counter(g['category'] for _,g in observed_gt)
 repeated=[c for c,n in sorted(counts.items()) if n>=2];missing=[c for c in C if not counts[c]]
 if not repeated or not missing:return []
 seed=int(hashlib.sha256(f"real_graph003/{scene['scene']}/{frame['frame_idx']}".encode()).hexdigest()[:12],16);rng=np.random.default_rng(seed)
 def distance(g):return float(np.linalg.norm(np.asarray(g['center'][:2])-origin))
 def relative(g):
  v=np.asarray(g['center'][:2])-origin;return np.array([v@forward,v@left])
 # 候选源为真实预测；每条查询统一做类别/参照类别优先的前8预选，所有模型共用，不读取GT。
 full=frame['predictions']['one_to_one_depth_robust_motion'];indices=sorted(range(len(full)),key=lambda i:(distance(full[i]),str(full[i]['node_id'])));ids=rng.choice(np.arange(100,999),len(indices),replace=False);nodes=[];truth_binding={};raw_identity={};by_pred={m['prediction_index']:m['truth_index'] for m in frame['metrics']['one_to_one_depth_robust_motion']['matches']}
 for pi,identity in zip(indices,ids):
  p=full[pi];e=(p.get('depth_evidence') or [{}])[-1];nid=int(identity);xy=relative(p);version=frame['frame_idx']+1
  node=dict(node_id=nid,category=p['category'],center_xy=[round(float(t),3) for t in xy],distance_m=round(distance(p),3),motion_state=p['motion_state'],version=version,now_s=now,last_seen_s=p['timestamp_s'] if p.get('timestamp_s') is not None else -1,evidence=dict(source=e.get('source','unknown'),version=version,observations=p['num_detections'],lidar_pixels=int(e.get('retained_pixels',0)),support_fraction=float(e.get('support_fraction',0)),conflict=not bool(e.get('sufficient_support',False))))
  nodes.append(node);raw_identity[nid]=p['node_id']
  if pi in by_pred:truth_binding[nid]=frame['evaluation_only_official'][by_pred[pi]]['instance_token']
 nodes=[nodes[i] for i in rng.permutation(len(nodes))];category=sorted(counts)[int(rng.integers(len(counts)))];same=repeated[int(rng.integers(len(repeated)))];gt_sorted=sorted(observed_gt,key=lambda x:(distance(x[1]),x[1]['instance_token']));pairs=[]
 for ti,t in gt_sorted:
  for ai,a in enumerate(frame['evaluation_only_official']):
   if ti==ai:continue
   delta=relative(t)-relative(a)
   if np.linalg.norm(delta)>12:continue
   for axis in [0,1]:
    if abs(delta[axis])>1:pairs.append((t['category'],a['category'],axis,delta[axis]>0))
 if not pairs:return []
 targetcat,anchorcat,axis,positive=pairs[int(rng.integers(len(pairs)))];out=[]
 for kind in KINDS:
  values=copy.deepcopy(nodes);facts=[];accepted=[];fault=None
  if kind in ['simple_category','insufficient_evidence','stale_information','conflicting_information']:
   query=f'选择最近的{C[category]}';choices=[g for _,g in gt_sorted if g['category']==category];accepted=[choices[0]['instance_token']]
   if kind!='simple_category':
    query='只有当前证据可靠且目标静止时才授权'+query;fault=kind
    for n in values:
     if kind=='insufficient_evidence':n['evidence'].update(observations=1,lidar_pixels=0)
     elif kind=='stale_information':n['last_seen_s']=now-10
     else:n['evidence']['conflict']=True
  elif kind=='complex_relation':
   direction=('左侧' if positive else '右侧') if axis==1 else ('前方' if positive else '后方');query=f'以{C[anchorcat]}为参照，选择其{direction}最近的{C[targetcat]}'
   anchors=[g for g in frame['evaluation_only_official'] if g['category']==anchorcat];choices=[]
   for _,g in gt_sorted:
    if g['category']!=targetcat:continue
    if any(a['instance_token']!=g['instance_token'] and abs((relative(g)-relative(a))[axis])>1 and ((relative(g)-relative(a))[axis]>0)==positive and np.linalg.norm(relative(g)-relative(a))<=12 for a in anchors):choices.append(g)
   if choices:accepted=[choices[0]['instance_token']]
   relation=('left_of' if positive else 'right_of') if axis==1 else ('in_front_of' if positive else 'behind')
   for n in values:
    for a in values:
     delta=np.asarray(n['center_xy'])-a['center_xy']
     if n['node_id']!=a['node_id'] and abs(delta[axis])>1 and (delta[axis]>0)==positive and np.linalg.norm(delta)<=12:facts.append(dict(source=n['node_id'],target=a['node_id'],relation=relation))
  elif kind in ['negation','similar_instances']:
   choices=[g for _,g in gt_sorted if g['category']==same];accepted=[choices[1]['instance_token']]
   query=(f'不要最近的{C[same]}，选择第二近的{C[same]}' if kind=='negation' else f'在所有{C[same]}中按距离选择第二近的一辆')
  elif kind=='multiple_targets':
   cats=list(rng.choice(sorted(counts),size=min(2,len(counts)),replace=False));query='在'+('或'.join(C[c] for c in cats))+'中选择最近的对象';accepted=[next(g for _,g in gt_sorted if g['category'] in cats)['instance_token']]
  else:query=f'选择最近的{C[missing[int(rng.integers(len(missing)))]]}'
  # 标签只在候选构建完成后关联；正查询缺失正确候选必须保留，不按“拒答正确”掩盖感知漏检。
  requested={c for c,word in C.items() if word in query};preferred=sorted(values,key=lambda n:(n['category'] not in requested,n['distance_m'],n['node_id']))[:8];kept={n['node_id'] for n in preferred};values=[n for n in values if n['node_id'] in kept];facts=[f for f in facts if f['source'] in kept and f['target'] in kept]
  available=[n['node_id'] for n in values if truth_binding.get(n['node_id']) in accepted];allowed=[n['node_id'] for n in values if reliable(n)];correct=[i for i in available if i in allowed] if fault else available
  answer=min((n for n in values if n['node_id'] in correct),key=lambda n:(n['distance_m'],n['node_id']))['node_id'] if correct and not fault else None
  # 严格任务指标：正常有目标但漏检/证据不足的查询不可算目标选择成功；安全动作可另计合理拒答。
  acceptable=[None] if fault or not accepted else correct
  order=sorted(values,key=lambda n:(n['node_id']!=answer,n['distance_m'],n['node_id']));response=dict(schema_version='1.0.0',selected_node_id=answer,ranking=[n['node_id'] for n in order],no_match=answer is None,explanation='真实感知观测与独立官方目标监督；不提供颜色细属性',confidence=1.)
  out.append(dict(task='candidate_rerank',sample_id=f"real_graph_{role}_{scene['scene']}_{frame['frame_idx']:03d}_{kind}",split=role,scene_id=scene['scene'],frame_idx=frame['frame_idx'],log_token=scene['log_token'],query_kind=kind,query_text=query,candidate_nodes=values,candidate_ids=[n['node_id'] for n in values],relation_metadata=facts,response=response,acceptable_node_ids=acceptable,expected_proposal_or_explicit_authorization_answer=answer,official_positive=bool(accepted),official_acceptable_instances=accepted,official_match_binding_evaluation_only=truth_binding,predicted_identity_binding=raw_identity,candidate_target_coverage=bool(available) if accepted else None,ordinary_proposal_only=fault is None,actual_authorization_credit=False,reliable_target_coverage=bool(set(available)&set(allowed)) if accepted else None,controlled_fault=fault,manual_reviewed=False,source_kind='real_saved_causal_RGB_LiDAR_detection_mapping_predictions',version_scope='frame context binding; not guaranteed global real instance identity',conflict_scope='measured depth sufficient_support failure only; controlled conflict injection separately marked',actual_arrival_evaluated=False))
 return out
def main():
 O.mkdir(parents=True,exist_ok=False);split=json.loads((R/'qwen_adapter_fair_comparison/official_graph_dataset_002/release.json').read_text())['split_logs'];frame_indices={'train':[2,10,18,26,34],'validation':[6,22],'development_holdout':[2,6,10,14,18,22,26,30,34,38]};lp=OLD/'annotation/official_motion_dev_001/evaluation_only_motion.jsonl';motion={(r['scene'],r['frame_idx']):{o['annotation_token']:o['official_motion'] for o in r['objects']} for r in map(json.loads,lp.open())};sources={};rows=[];excluded=[]
 release=dict(split_logs=split,frame_indices=frame_indices,candidate_limit=8,source_method='one_to_one_depth_robust_motion',selection='按预定帧号，仅要求官方GT有至少两个同类对象及一个可形成的关系；不按检测/模型结果选样，零候选/漏检保留；候选前8统一按查询类别优先再距离，所有模型同输入',controlled_language=True,proposal_and_authorization_separate=True,ordinary_queries_scope='普通5类只评价编号提议，不声明静止锁定/实际控制；三类故障指令明确要求可靠静止证据授权，实际控制仍必须独立检查',controlled_fault_strata=['insufficient_evidence','stale_information','conflicting_information'],nominal_strata=KINDS,seed='sha256 real_graph001 scene/frame',same_future_comparison_inputs=True,goal_complete=False,generator_sha256=digest(__file__),contract_sha256=digest(R/'source/real_graph_contract_002.py'),note='开发实验：2个留出日志已在此前标注图诊断开启过，不能称新封存测试；003修正002辅助字段名称和reliable_target_coverage含义，不按模型结果筛样')
 dump(O/'release.json',release)
 for scene in available_development():
  role=next(k for k,v in split.items() if scene['log_token'] in v);p=OLD/'mapping_evaluation/causal_mapping_full_001'/scene['scene']/'frames.jsonl';sources[str(p)]=digest(p)
  for frame in map(json.loads,p.open()):
   if frame['frame_idx'] not in frame_indices[role]:continue
   new=make(frame,scene,role,motion[(scene['scene'],frame['frame_idx'])]);rows+=new
   if not new:excluded.append(dict(scene=scene['scene'],frame_idx=frame['frame_idx'],split=role,reason='official GT insufficient same-class objects or relation pair; not detector/model outcome'))
 for role in split:
  with (O/f'{role}.jsonl').open('x') as f:
   for row in rows:
    if row['split']==role:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
 counts={k:dict(Counter(r['query_kind'] for r in rows if r['split']==k)) for k in split};dump(O/'manifest.json',dict(counts=counts,split_logs=split,files={p.name:digest(p) for p in O.glob('*.jsonl')},unique_frames={k:len({(r['scene_id'],r['frame_idx']) for r in rows if r['split']==k}) for k in split},sources_sha256={**sources,str(lp):digest(lp)},excluded=excluded,manual_review_count=0,new_sealed_test_opened=False,goal_complete=False));print(json.dumps(counts,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
