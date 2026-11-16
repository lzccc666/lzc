"""六组冻结建图输出接同一清晰类别合作规则；新CPU决策，不授予复杂Qwen或锁定/到达信用。"""
import csv,gzip,json,math,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import OUTPUT as OLD,available_development,dump,digest
from conceptgraph.completion.quality import object_frame_metrics
from real_graph_cooperation_001 import choose
R=Path(__file__).resolve().parents[1];O=R/'nuscenes_ablation/category_semantics_001'
CONDITIONS=['full','without_depth_quality','without_one_to_one','without_dynamic_maintenance','without_short_term_expiry','without_relations']
CATEGORIES={'car':'小汽车','bus':'公交车','pedestrian':'行人'}

def main():
 O.mkdir(exist_ok=False);scenes=available_development();sources={};rows=[]
 release=dict(source_sha256={str(p):digest(p) for p in [Path(__file__),R/'source/real_graph_cooperation_001.py',R/'source/real_graph_contract_003.py']},conditions=CONDITIONS,scenes=scenes,categories=CATEGORIES,queries='每帧固定3条“选择最近的<类别>”，按官方当前评估范围中最近同类实例评价；不根据模型成绩挑帧',candidate_limit=8,common_preselection='预测类别优先再相机距离，池内按预测ID；只用当前过去预测，不用GT',bootstrap=dict(unit='log',repeats=4000,seed=17),scope='六组同输入建图输出上的完整合作逻辑清晰类别分支；复杂Qwen分支未激活，仅离线普通编号提议，当前证据授权参考另列',actual_new_mapping_calls=0,actual_new_GPU_calls=0,actual_lock_credit=False,actual_arrival_credit=False,goal_complete=False)
 dump(O/'release.json',release)
 with gzip.open(O/'decisions.jsonl.gz','xt',encoding='utf-8',compresslevel=6) as output:
  for scene in scenes:
   source=OLD/'mapping_evaluation/causal_mapping_full_001'/scene['scene']/'frames.jsonl';sources[str(source)]=digest(source);base={f['frame_idx']:f for f in map(json.loads,source.open())}
   for condition in CONDITIONS:
    p=R/'nuscenes_ablation'/f'strict_{condition}_001'/scene['scene']/'frames.jsonl';sources[str(p)]=digest(p)
    frames=list(map(json.loads,p.open()));assert set(base)=={f['frame_idx'] for f in frames}
    for f in frames:
     idx=f['frame_idx'];original=base[idx];pose=np.asarray(original['camera_pose']);origin=pose[:2,3];forward=pose[:2,2];forward=forward/np.linalg.norm(forward);left=np.asarray([-forward[1],forward[0]]);now=f['camera_timestamp_s'];predictions=f['predictions'][condition];truth=f['evaluation_only_official'];assert truth==original['evaluation_only_official'];metrics=object_frame_metrics(predictions,truth);assert metrics==f['metrics'][condition]
     binding={m['prediction_index']+1:truth[m['truth_index']]['instance_token'] for m in metrics['matches']};nodes=[]
     for pi,pred in enumerate(predictions):
      xy=np.asarray(pred['center'][:2])-origin;e=(pred.get('depth_evidence') or [{}])[-1];nodes.append(dict(node_id=pi+1,category=pred['category'],center_xy=[float(xy@forward),float(xy@left)],distance_m=float(np.linalg.norm(xy)),motion_state=pred['motion_state'],version=idx+1,now_s=now,last_seen_s=pred.get('timestamp_s') if pred.get('timestamp_s') is not None else -1,evidence=dict(source=e.get('source','unknown'),version=idx+1,observations=pred['num_detections'],lidar_pixels=int(e.get('retained_pixels',0)),support_fraction=float(e.get('support_fraction',0)),conflict=not bool(e.get('sufficient_support',False)))))
     for category,label in CATEGORIES.items():
      query='选择最近的'+label;pool=sorted(nodes,key=lambda n:(n['category']!=category,n['distance_m'],n['node_id']))[:8];pool=sorted(pool,key=lambda n:n['node_id']);t=time.perf_counter();decision=choose(query,pool,[],None);elapsed=time.perf_counter()-t
      official=sorted([g for g in truth if g['category']==category],key=lambda g:(float(np.linalg.norm(np.asarray(g['center'][:2])-origin)),g['instance_token']));target=official[0]['instance_token'] if official else None;proposal=decision['proposed_node_id'];correct=bool(proposal is not None and binding.get(proposal)==target) if target is not None else proposal is None;covered=bool(target is not None and any(binding.get(n['node_id'])==target for n in pool));row=dict(condition=condition,scene=scene['scene'],log=scene['log_token'],frame_idx=idx,query=query,positive=target is not None,correct=correct,candidate_covered=covered,false_no_match_proposal=target is None and proposal is not None,proposed_node_id=proposal,authorized_node_id=decision['authorized_node_id'],current_evidence_supported_correct_proposal=bool(target is not None and correct and decision['authorized_node_id'] is not None),offline_decision_s=elapsed,source_frame_file=str(p),source_frame_idx=idx,GT_in_decision=False,actual_lock_credit=False,actual_arrival_credit=False)
      output.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n');rows.append(row)
  assert len(rows)==6*1508*3
 logs=sorted({r['log'] for r in rows});draw=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)));metrics=['task_accuracy','positive_proposal_accuracy','no_match_false_proposal_rate','candidate_coverage_positive','current_evidence_supported_correct_proposal_rate'];table=[];by_scene=[]
 def counts(rr):
  pos=[r for r in rr if r['positive']];neg=[r for r in rr if not r['positive']]
  return dict(task_accuracy=(sum(r['correct'] for r in rr),len(rr)),positive_proposal_accuracy=(sum(r['correct'] for r in pos),len(pos)),no_match_false_proposal_rate=(sum(r['false_no_match_proposal'] for r in neg),len(neg)),candidate_coverage_positive=(sum(r['candidate_covered'] for r in pos),len(pos)),current_evidence_supported_correct_proposal_rate=(sum(r['current_evidence_supported_correct_proposal'] for r in pos),len(pos)))
 for condition in CONDITIONS:
  selected=[r for r in rows if r['condition']==condition];by_log={l:counts([r for r in selected if r['log']==l]) for l in logs};totals=counts(selected)
  for scene in scenes:
   z=counts([r for r in selected if r['scene']==scene['scene']]);by_scene.append(dict(condition=condition,scene=scene['scene'],log=scene['log_token'],**{k+'_numerator':v[0] for k,v in z.items()},**{k+'_denominator':v[1] for k,v in z.items()}))
  for metric in metrics:
   a=np.asarray([by_log[l][metric] for l in logs]);sampled=a[draw].sum(axis=1);values=np.divide(sampled[:,0],sampled[:,1],out=np.full(len(draw),np.nan),where=sampled[:,1]>0);lo,hi=np.nanpercentile(values,[2.5,97.5]);num,den=totals[metric];table.append(dict(condition=condition,metric=metric,numerator=num,denominator=den,value=num/den if den else None,ci95_low=float(lo),ci95_high=float(hi),source_frames=1508,scenes=38,logs=11,queries=4524,actual_new_GPU_calls=0,actual_locks=0,actual_arrivals=0))
 for name,data in [('comparison.csv',table),('scene_counts.csv',by_scene)]:
  with (O/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
 dump(O/'report.json',dict(actual_new_CPU_decisions=len(rows),conditions=6,frames_per_condition=1508,queries_per_condition=4524,distinct_source_scenes=38,distinct_logs=11,shared_queries_per_frame=3,source_sha256=sources,decision_source_sha256=release['source_sha256'],results_sha256=digest(O/'decisions.jsonl.gz'),elapsed_scope='offline actual cooperation branch only; perception and model time not fabricated',limitations=['普通类别提议不是静止目标控制授权，不给锁定或到达信用','三种固定类别模板不是4524条独立自然指令，独立统计按11日志','清晰类别逻辑没有调用Qwen；关系关闭条件在这类查询无预期贡献，不冒称复杂语义模块已完成','候选随不同建图模块真实变化；所有条件共用相同原始前端帧与查询，语义模型公平比较另有同候选108条协议','缺失正确候选的正查询保留为失败；无匹配仅限当前已声明官方评估范围','候选版本是当前帧上下文绑定，非人工证明的全局真实身份','新决策仅复用已保存冻结建图输出，非重新运行前端或在线控制'],goal_complete=False))
 print(json.dumps(dict(decisions=len(rows),metrics=table),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
