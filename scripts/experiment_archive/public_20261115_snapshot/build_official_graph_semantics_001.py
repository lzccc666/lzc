"""真实官方标注的几何图 + 自动构造语言/证据故障；不声称人工指令或端到端感知。"""
import copy,json,hashlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import OUTPUT as OLD,dump,digest,available_development
R=Path(__file__).resolve().parents[1];O=R/'qwen_adapter_fair_comparison/official_graph_dataset_001'
C={'car':'小汽车','truck':'卡车','bus':'公交车','pedestrian':'行人','bicycle':'自行车','motorcycle':'摩托车'}
KINDS=['simple_category','complex_relation','negation','similar_instances','multiple_targets','no_match','insufficient_evidence','stale_information','conflicting_information']
def make_rows(frame,scene,role):
 gt=[g for g in frame['evaluation_only_official'] if g['category'] in C]
 pose=np.asarray(frame['camera_pose']);origin=pose[:3,3];f=pose[:2,2];f=f/np.linalg.norm(f);left=np.array([-f[1],f[0]]);gt=sorted(gt,key=lambda g:np.linalg.norm(np.asarray(g['center'])[:2]-origin[:2]))[:4]
 if len(gt)<2:return []
 seed=int(hashlib.sha256(f"official_graph_v1/{scene['scene']}/{frame['frame_idx']}".encode()).hexdigest()[:12],16);rng=np.random.default_rng(seed);ids=rng.choice(np.arange(100,999),len(gt),replace=False);now=float(frame['camera_timestamp_s']);nodes=[];mapping={}
 for i,g in zip(ids,gt):
  delta=np.asarray(g['center'])[:2]-origin[:2];nid=int(i);mapping[nid]=g['instance_token'];nodes.append(dict(node_id=nid,category=g['category'],center_xy=[round(float(delta@f),3),round(float(delta@left),3)],distance_m=round(float(np.linalg.norm(delta)),3),version=1,now_s=now,last_seen_s=now,evidence=dict(source='official_annotation_geometry',version=1,observations=3,lidar_pixels=5,conflict=False)))
 # 3次观测/5像素为受控协议字段，不是本帧真实检测支持；绝不向感知评价转用。
 candidates=[nodes[i] for i in rng.permutation(len(nodes))];nearest=sorted(nodes,key=lambda n:(n['distance_m'],n['node_id']));category=nearest[0]['category'];name=C[category];rows=[]
 for kind in KINDS:
  values=copy.deepcopy(candidates);facts=[];acceptable=[];prefix='请' if role=='validation' else '请将' if role=='development_holdout' else ''
  if kind=='simple_category':query=f'{prefix}选择最近的{name}';acceptable=[min((n for n in values if n['category']==category),key=lambda n:(n['distance_m'],n['node_id']))['node_id']]
  elif kind=='complex_relation':
   anchor=nearest[-1];options=[n for n in nearest if n['node_id']!=anchor['node_id']];target=options[0];axis=1 if abs(target['center_xy'][1]-anchor['center_xy'][1])>1 else 0;positive=target['center_xy'][axis]>anchor['center_xy'][axis];direction=('左侧' if positive else '右侧') if axis==1 else ('前方' if positive else '后方');relation=('left_of' if positive else 'right_of') if axis==1 else ('in_front_of' if positive else 'behind');query=f"{prefix}以编号{anchor['node_id']}为参照，选择其{direction}最近的对象";options=[n for n in options if (n['center_xy'][axis]>anchor['center_xy'][axis])==positive];acceptable=[min(options,key=lambda n:(n['distance_m'],n['node_id']))['node_id']];facts=[dict(source=n['node_id'],target=anchor['node_id'],relation=relation) for n in options]
  elif kind in ['negation','similar_instances']:
   query=(f'{prefix}不要最近的对象，选择第二近的对象' if kind=='negation' else f'{prefix}按到自车的距离从近到远，选择第二个对象');acceptable=[nearest[1]['node_id']]
  elif kind=='multiple_targets':
   cats=sorted({n['category'] for n in values});query=f"{prefix}在{'或'.join(C[x] for x in cats)}中选择最近的一个";acceptable=[nearest[0]['node_id']]
  elif kind=='no_match':
   missing=next((c for c in C if c not in {n['category'] for n in values}),None);query=f'{prefix}选择最近的{C[missing]}';acceptable=[]
  else:
   query=f'{prefix}选择证据可靠且最近的{name}'
   for n in values:
    if kind=='insufficient_evidence':n['evidence']['observations']=1;n['evidence']['lidar_pixels']=0
    elif kind=='stale_information':n['last_seen_s']=now-10;n['evidence']['version']=0
    else:n['evidence']['conflict']=True
  ranked=sorted(values,key=lambda n:(n['node_id'] not in acceptable,n['distance_m'],n['node_id']));response=dict(schema_version='1.0.0',selected_node_id=ranked[0]['node_id'] if acceptable else None,ranking=[n['node_id'] for n in ranked],no_match=not bool(acceptable),explanation='official geometry and controlled evidence protocol',confidence=1.)
  rows.append(dict(task='candidate_rerank',sample_id=f"official_graph_{role}_{scene['scene']}_{frame['frame_idx']:03d}_{kind}",split=role,scene_id=scene['scene'],log_token=scene['log_token'],query_kind=kind,query_text=query,candidate_nodes=values,candidate_ids=[n['node_id'] for n in values],relation_metadata=facts,acceptable_node_ids=acceptable or [None],response=response,official_identity_evaluation_only=mapping,source_kind='official_annotation_oracle_graph_controlled_language',manual_reviewed=False,independent_real_instruction_credit=0,perception_accuracy_credit=0,evidence_fields='人工设计的协议诊断值，非真实检测或LiDAR支持计数'))
 return rows

def main():
 O.mkdir(parents=True,exist_ok=False);old=json.loads((OLD/'semantic_dataset/structured_supervision_002/manifest.json').read_text());known=available_development();split={k:{x for x in v if not x.startswith('symbolic')} for k,v in old['audit']['independent_logs'].items()};split['development_holdout']={s['log_token'] for s in known}-split['train']-split['validation'];assert not(split['train']&split['validation'] or split['train']&split['development_holdout'] or split['validation']&split['development_holdout']);rows=[];sources={}
 release=dict(split_logs={k:sorted(v) for k,v in split.items()},frame_indices={'train':[2,10,18,26,34],'validation':[6,22],'development_holdout':[2,6,10,14,18,22,26,30,34,38]},candidate_cap=4,minimum_gt_candidates=2,selection_rule='first4 official eligible GT objects by distance; no detector outcome used to select',kinds=KINDS,seed='sha256(scene/frame)',contract_sha256=digest(R/'source/semantic_graph_contract_001.py'),generator_sha256=digest(__file__),sealed_test_opened=False,task_scope='标注图推理上限诊断；证据故障受控生成；不是公开图像真实目标检索或自然人工指令',goal_complete=False);dump(O/'release.json',release)
 for scene in known:
  role=next(k for k,v in split.items() if scene['log_token'] in v);p=OLD/'mapping_evaluation/causal_mapping_full_001'/scene['scene']/'frames.jsonl';sources[str(p)]=digest(p)
  for f in map(json.loads,p.open()):
   if f['frame_idx'] in release['frame_indices'][role]:rows.extend(make_rows(f,scene,role))
 for role in split:
  with (O/f'{role}.jsonl').open('x') as handle:
   for row in rows:
    if row['split']==role:handle.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
 counts={k:dict(Counter(r['query_kind'] for r in rows if r['split']==k)) for k in split};dump(O/'manifest.json',dict(counts=counts,unique_source_frames={k:len({(r['scene_id'],r['sample_id'].split('_')[-2]) for r in rows if r['split']==k}) for k in split},split_logs=release['split_logs'],files={f.name:digest(f) for f in O.glob('*.jsonl')},source_sha256=sources,manual_review_count=0,new_test_opened=False,independent_real_instruction_credit=0,perception_accuracy_credit=0,goal_complete=False));print(json.dumps(counts,ensure_ascii=False))
if __name__=='__main__':main()
