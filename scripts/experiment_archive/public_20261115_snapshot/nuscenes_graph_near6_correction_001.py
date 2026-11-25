"""修正未被配置类接受的邻近半径键；真实重跑图维护，不重跑或改写对象建图。"""
import ctypes
ctypes.CDLL('/root/miniconda3/envs/conceptgraph/lib/libstdc++.so.6',mode=ctypes.RTLD_GLOBAL)
import copy,csv,gzip,hashlib,json,resource,shutil,time,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import OUTPUT as OLD,available_development,digest,dump
from conceptgraph.scenegraph.relations import RelationConfig,RelationNode
from conceptgraph.scenegraph.incremental_graph import IncrementalSceneGraph
from conceptgraph.completion.quality import relations_from_centers,relation_metrics
from conceptgraph.completion.relations_audit import expand
R=Path(__file__).resolve().parents[1];O=R/'nuscenes_ablation/graph_near6_correction_001';CONDITIONS=['full','without_depth_quality','without_one_to_one','without_dynamic_maintenance','without_short_term_expiry','without_relations'];POLICY=R/'json/resource_policy_002.json'

def guard(reserve=0):
 p=json.loads(POLICY.read_text());size=sum(f.stat().st_size for f in R.rglob('*') if f.is_file() and not f.is_symlink())
 if size+reserve>=p['new_project_artifact_budget_bytes'] or shutil.disk_usage(R).free<p['minimum_physical_free_GiB']*2**30:raise OSError('遵守用户新25GiB预算和20GiB物理空闲')

def totals(x):
 tp,pred,truth=(x[k] for k in ['true_positive','predicted','truth']);return dict(precision=tp/pred if pred else 0.,recall=tp/truth if truth else 0.,f1=2*tp/(pred+truth) if pred+truth else 0.)

def main():
 guard(64*2**20);O.mkdir(exist_ok=False);scenes=available_development();old_release=R/'nuscenes_ablation/strict_full_001/release.json';declared=json.loads(old_release.read_text())['graph_config'];old_config=RelationConfig.from_mapping(declared).to_dict();new_config=copy.deepcopy(old_config);new_config['relation_distance_thresholds_m']['near']=6.;assert old_config['relation_distance_thresholds_m']['near']==8.;check=copy.deepcopy(new_config);check['relation_distance_thresholds_m']['near']=8.;assert check==old_config
 files=[Path(__file__),POLICY,old_release,R/'source/nuscenes_strict_ablation_002.py',Path(__import__('conceptgraph.scenegraph.relations',fromlist=['x']).__file__),Path(__import__('conceptgraph.scenegraph.incremental_graph',fromlist=['x']).__file__)];sources={str(p):digest(p) for p in files};input_paths=[R/'nuscenes_ablation'/f'strict_{c}_001'/s['scene']/'frames.jsonl' for c in CONDITIONS for s in scenes];inputs={str(p):digest(p) for p in input_paths}
 dump(O/'release.json',dict(source_sha256=sources,input_files_sha256=inputs,conditions=CONDITIONS,scenes=scenes,old_declared_graph_config=declared,old_actual_graph_config=old_config,new_actual_graph_config=new_config,isolated_change='只有relation_distance_thresholds_m.near从实际8m改6m；其他已解析字段逐项相同；原max_distance_m/near_distance_m键不被RelationConfig使用',GT_near_radius_m=6.,directional_radius_m=12.,bootstrap=dict(unit='log',repeats=4000,seed=17),object_predictions_changed=False,new_mapper_calls=0,new_GPU_calls=0,new_test_opened=False,goal_complete=False))
 started=time.monotonic();updates=0;reports=[];rows_count=0;outside_old=0
 try:
  for condition in CONDITIONS:
   for scene in scenes:
    original={f['frame_idx']:f for f in map(json.loads,(OLD/'mapping_evaluation/causal_mapping_full_001'/scene['scene']/'frames.jsonl').open())};src=R/'nuscenes_ablation'/f'strict_{condition}_001'/scene['scene']/'frames.jsonl';case=O/condition/scene['scene'];case.mkdir(parents=True);graph=IncrementalSceneGraph(new_config);registry={};acc={name:dict(true_positive=0,predicted=0,truth=0) for name in ['old','corrected']};count=0;times=[];oldsix=0;new_edges_count=0
    with gzip.open(case/'frames.jsonl.gz','xt',compresslevel=6) as output:
     for f in map(json.loads,src.open()):
      if rows_count%200==0:guard(4*2**20)
      idx=f['frame_idx'];base=original[idx];assert base['evaluation_only_official']==f['evaluation_only_official'];objects=f['predictions'][condition]
      for obj in objects:registry.setdefault(obj['node_id'],len(registry))
      nodes={registry[obj['node_id']]:RelationNode(registry[obj['node_id']],obj['center'],dynamic=obj['motion_state']=='dynamic',observation_count=obj['num_detections'],last_seen_frame=obj['last_seen_frame'],metadata={'timestamp_s':obj.get('timestamp_s')}) for obj in objects};assert set(nodes)==set(map(int,f['nodes']))
      for k,v in nodes.items():assert np.allclose(v.center,f['nodes'][str(k)]['center'])
      forward=np.asarray(base['camera_pose'])[:2,2];tick=time.monotonic();edges=[] if condition=='without_relations' else graph.update_snapshot(nodes,frame_index=idx,ego_heading=forward).edges;times.append(time.monotonic()-tick);updates+=condition!='without_relations';assert all(e.distance_m<=6.+1e-8 for e in edges if e.relation=='near')
      truth=f['evaluation_only_official'];binding={registry[objects[m['prediction_index']]['node_id']]:truth[m['truth_index']]['instance_token'] for m in f['metrics'][condition]['matches']};expanded=expand(edges);resolved={(binding[a],rel,binding[b]) for a,rel,b in expanded if a in binding and b in binding};target=relations_from_centers([dict(id=g['instance_token'],center=g['center']) for g in truth],forward_xy=forward,max_distance_m=12.,near_m=6.);metrics=relation_metrics(resolved,target);metrics['unresolved_edges']=sum(a not in binding or b not in binding for a,rel,b in expanded)
      for name,m in [('old',f['relations']),('corrected',metrics)]:
       for k in acc[name]:acc[name][k]+=m[k]
      remap={};used=set()
      for obj in sorted(objects,key=lambda x:x['node_id']):
       nid=100+int(hashlib.sha256(f"full-semantic-ID-v1/{scene['scene']}/{idx}/{obj['node_id']}".encode()).hexdigest()[:8],16)%900
       while nid in used:nid=100+(nid-99)%900
       used.add(nid);remap[registry[obj['node_id']]]=nid
      oldsix+=sum(e['relation']=='near' and e['distance_m']>6.+1e-8 for e in f['edges']);new_edges_count+=len(edges);output.write(json.dumps(dict(scene=scene['scene'],frame_idx=idx,condition=condition,semantic_node_ids=remap,edges=[e.to_dict() for e in edges],relations=metrics,old_relations=f['relations'],source_frame_file=str(src),object_signature_sha256=hashlib.sha256(json.dumps(objects,sort_keys=True,separators=(',',':')).encode()).hexdigest()),ensure_ascii=False,separators=(',',':'))+'\n');count+=1;rows_count+=1
    outside_old+=oldsix;rep=dict(condition=condition,scene=scene['scene'],log=scene['log_token'],frames=count,old_counts=acc['old'],corrected_counts=acc['corrected'],old_metrics=totals(acc['old']),corrected_metrics=totals(acc['corrected']),old_near_edges_distance_above_6m=oldsix,new_edges=new_edges_count,actual_graph_update_calls=count if condition!='without_relations' else 0,graph_update_wall_s=sum(times),raw_sha256=digest(case/'frames.jsonl.gz'),source_frame_sha256=inputs[str(src)]);dump(case/'report.json',rep);reports.append(rep);print(json.dumps(dict(condition=condition,scene=scene['scene'],frames=count)),flush=True)
  assert rows_count==9048 and updates==7540 and all(digest(p)==h for p,h in sources.items());table=[];logs=sorted({r['log'] for r in reports});draw=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)))
  for condition in CONDITIONS:
   selected=[r for r in reports if r['condition']==condition];counts={name:{k:sum(r[name+'_counts'][k] for r in selected) for k in ['true_positive','predicted','truth']} for name in ['old','corrected']}
   for metric in ['precision','recall','f1']:
    estimates={};bootstrap={}
    for name in ['old','corrected']:
     bylog=np.array([[sum(r[name+'_counts'][k] for r in selected if r['log']==log) for k in ['true_positive','predicted','truth']] for log in logs]);sample=bylog[draw].sum(1);den=(sample[:,1] if metric=='precision' else sample[:,2] if metric=='recall' else sample[:,1]+sample[:,2]);num=sample[:,0]*(2 if metric=='f1' else 1);values=np.divide(num,den,out=np.zeros(len(draw),float),where=den>0);lo,hi=np.percentile(values,[2.5,97.5]);bootstrap[name]=values;estimates[name]=totals(counts[name])[metric];table.append(dict(condition=condition,metric=metric,version=name,value=estimates[name],ci95_low=float(lo),ci95_high=float(hi),frames=1508,scenes=38,logs=11,**counts[name]))
    delta=bootstrap['corrected']-bootstrap['old'];lo,hi=np.percentile(delta,[2.5,97.5]);table.append(dict(condition=condition,metric=metric,version='corrected_minus_old',value=estimates['corrected']-estimates['old'],ci95_low=float(lo),ci95_high=float(hi),frames=1508,scenes=38,logs=11,true_positive='',predicted='',truth=''))
  with (O/'comparison.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
  dump(O/'report.json',dict(output_records=rows_count,actual_new_graph_update_calls=updates,actual_new_mapper_calls=0,actual_new_GPU_calls=0,conditions=6,scenes=38,logs=11,frames_each=1508,old_near_edges_above_6m=outside_old,old_configs_and_results_preserved=True,new_graph_requires_new_semantic_inference=True,wall_s=time.monotonic()-started,peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,release_sha256=digest(O/'release.json'),comparison_sha256=digest(O/'comparison.csv'),limitations=['仅重跑图维护/关系评价；对象预测和对象指标不变，不冒称重跑检测/融合/Qwen/控制','原6m声明但实际8m的图/语义结果均保留为旧配置；新图接语言模块需重新冻结并真实推理','不是新的最终封存测试，未产生在线锁定或到达'],goal_complete=False))
  print('all near6 graph corrections completed',flush=True)
 except BaseException as exc:dump(O/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),saved_records=rows_count,goal_complete=False));raise

if __name__=='__main__':main()
