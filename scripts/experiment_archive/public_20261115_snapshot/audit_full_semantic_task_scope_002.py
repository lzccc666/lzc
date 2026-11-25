"""冻结查询设计的覆盖与歧义审计，不看模型成绩，不改标签/输入。"""
from collections import Counter,defaultdict
from pathlib import Path
import json
from conceptgraph.completion.stage9 import available_development,OUTPUT as OLD,dump,digest
from nuscenes_full_semantics_001 import task,iter_rows,C
from real_graph_cooperation_001 import categories
R=Path(__file__).resolve().parents[1];D=R/'nuscenes_ablation/full_semantics_001';O=D/'task_scope_audit_002'

def main():
 release=json.loads((D/'release.json').read_text());factory=R/'source/nuscenes_full_semantics_001.py';assert digest(factory)==release['sources_sha256'][str(factory)];O.mkdir(exist_ok=False);counts=Counter();kinds=defaultdict(Counter);examples=defaultdict(list);texts=defaultdict(set);roles=Counter();byid={}
 for scene in available_development():
  for f in map(json.loads,(OLD/'mapping_evaluation/causal_mapping_full_001'/scene['scene']/'frames.jsonl').open()):
   q,origin,forward,left=task(f,scene);kind=q['query_kind'];counts['frames']+=1;kinds[kind]['queries']+=1;texts[kind].add(q['query_text']);gt=[g for g in f['evaluation_only_official'] if g['category'] in C];sid=f"full_semantic_{scene['scene']}_{f['frame_idx']:03d}";byid[sid]=q
   if kind=='complex_relation':
    anchor_text=q['query_text'].split('为参照')[0][1:];target_text=q['query_text'].split('选择')[-1];anchors={c for c,w in C.items() if w in anchor_text};targets={c for c,w in C.items() if w in target_text};n=sum(g['category'] in anchors for g in gt);kinds[kind]['multiple_official_reference_instances']+=n>1;kinds[kind]['reference_and_target_same_category']+=bool(anchors&targets)
    if n>1 and len(examples['multiple_reference_instances'])<5:examples['multiple_reference_instances'].append(dict(sample_id=sid,query=q['query_text'],official_reference_instances=n))
   if kind=='similar_instances' and '行人' in q['query_text']:
    kinds[kind]['vehicle_classifier_used_for_pedestrian']+=1
    if len(examples['awkward_classifier'])<5:examples['awkward_classifier'].append(dict(sample_id=sid,query=q['query_text']))
   if kind=='no_match':kinds[kind]['actually_positive_official_label']+=q['official_positive']
 for row in iter_rows():
  if row['condition']!='full':break
  q=byid[row['sample_id']];assert q['query_text']==row['query_text'];kind=row['query_kind'];roles[row['source_role']]+=1;wanted=categories(row['query_text']);compatible=[n for n in row['candidate_nodes'] if n['category'] in wanted];kinds[kind]['official_positive_before_controlled_fault']+=row['official_positive'];kinds[kind]['correct_target_in_predicted_pool']+=row['candidate_target_coverage'];kinds[kind]['any_predicted_compatible_candidate']+=bool(compatible);kinds[kind]['no_predicted_candidates']+=not row['candidate_nodes']
 assert counts['frames']==1508 and sum(roles.values())==1508
 dump(O/'report.json',dict(source_sha256={str(Path(__file__)):digest(__file__),str(factory):digest(factory),str(D/'release.json'):digest(D/'release.json')},actual_frames_audited=1508,conditions_under_audit='同查询工厂六组共享；只审计一次全体查询和完整配置候选覆盖，非六次独立自然指令',source_roles=dict(roles),query_kind_counts={k:dict(v,distinct_query_texts=len(texts[k])) for k,v in kinds.items()},distinct_query_texts_total=len(set.union(*texts.values())),examples=dict(examples),frozen_GT_predicate='复杂关系使用官方范围内任一同类参照，满足相对轴距离>1m、两点XY距<=12m；在满足条件的目标中取离自车最近的实例。该显式几何谓词不同于选离参照物最近者。否定/相似实例均按自车距离取第二近。',design_limitations=['同类别多参照可能导致自然语言歧义，当前只按冻结的存在性几何谓词评价；未获真实人工消歧确认','相似实例模板对行人使用一辆等量词可能不自然；原实验输入标签不修改，不据此删样','每帧由frame_idx%9确定任务层，文本重复，不能把1508当1508条独立人工自然指令','所有故障层均受控注入；应同时看有兼容候选和正确目标覆盖数量，空候选的拒答不足以证明复杂语义理解','本审计不读取模型答案、未重算或改写主成绩，不做结果筛样；新封存前需要明确自然语句及真实核验'],new_GPU_requests=0,new_semantic_decisions=0,new_annotation_credit=False,new_lock_or_arrival_credit=False,goal_complete=False))
 print(json.dumps(dict(frames=1508,distinct_texts=len(set.union(*texts.values())),query_kind_counts={k:dict(v) for k,v in kinds.items()}),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
