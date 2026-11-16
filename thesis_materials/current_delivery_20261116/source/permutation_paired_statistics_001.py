"""逆序压力测试的配对日志重采样；同一请求的逆序不计作新场景。"""
import csv,json
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import dump,digest
from semantic_target_only_audit_001 import parse_target

R=Path(__file__).resolve().parents[1];Q=R/'qwen_adapter_fair_comparison'
P=Q/'real_graph_permutation_002';G=Q/'real_graph_fair_gpu_001'
D=Q/'real_graph_dataset_003/development_holdout.jsonl'
O=Q/'real_graph_permutation_statistics_001'

def main():
 O.mkdir(exist_ok=False)
 metadata={r['sample_id']:r for r in map(json.loads,D.open())}
 release=json.loads((P/'release.json').read_text());ids=release['sample_ids']
 logs=sorted({metadata[s]['log_token'] for s in ids})
 draw=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)))
 values=[];sources={str(D):digest(D),str(P/'release.json'):digest(P/'release.json')}
 for model in release['variants']:
  a_path=G/model/'raw_records.jsonl';b_path=P/model/'raw_records.jsonl'
  sources.update({str(a_path):digest(a_path),str(b_path):digest(b_path)})
  original={r['sample_id']:r for r in map(json.loads,a_path.open())}
  reverse={r['sample_id']:r for r in map(json.loads,b_path.open())}
  assert set(reverse)==set(ids)
  for scope in ['all','ordinary_positive']:
   chosen=[s for s in ids if scope=='all' or reverse[s]['ordinary_positive']]
   scores={}
   for sid in chosen:
    a=original[sid];b=reverse[sid];allowed=metadata[sid]['candidate_ids']
    av,aid=parse_target(a['raw_output'],allowed);bv,bid=parse_target(b['raw_output'],allowed)
    scores[sid]=dict(strict_consistency=int(bool(a['schema_valid'] and b['schema_valid'] and a['normalized_output']['selected_node_id']==b['normalized_output']['selected_node_id'])),target_only_consistency=int(bool(av and bv and aid==bid)),strict_original_correct=int(a['semantic_correct']),strict_reversed_correct=int(b['semantic_correct']),strict_correct_paired_change=int(b['semantic_correct'])-int(a['semantic_correct']))
   for metric in next(iter(scores.values())):
    counts=np.asarray([[sum(scores[s][metric] for s in chosen if metadata[s]['log_token']==log),sum(metadata[s]['log_token']==log for s in chosen)] for log in logs])
    boot=counts[draw].sum(1);rates=np.divide(boot[:,0],boot[:,1],out=np.full(len(draw),np.nan),where=boot[:,1]>0);low,high=np.nanpercentile(rates,[2.5,97.5])
    values.append(dict(model=model,scope=scope,metric=metric,numerator=sum(scores[s][metric] for s in chosen),denominator=len(chosen),value=sum(scores[s][metric] for s in chosen)/len(chosen),log_ci95_low=float(low),log_ci95_high=float(high),logs=len(logs),bootstrap_repeats=4000))
 with (O/'metrics.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=values[0]);w.writeheader();w.writerows(values)
 dump(O/'report.json',dict(source_sha256={**sources,str(Path(__file__)):digest(__file__)},source_requests=len(ids),models=len(release['variants']),actual_new_GPU_requests=126,actual_new_GPU_requests_this_statistics=0,distinct_source_frames=len({(metadata[s]['scene_id'],metadata[s]['frame_idx']) for s in ids}),distinct_source_scenes=len({metadata[s]['scene_id'] for s in ids}),distinct_source_logs=len(logs),new_independent_frames=0,bootstrap=dict(seed=17,repeats=4000,unit='log'),limitations=['18条固定开发请求，不是新封存测试；逆序不增加独立场景数','一致拒答可能稳定但并不选对正目标，必须与正查询正确率同时读','仅逆序节点列表，候选ID许可列表和输出示例不变；双重无效输出一致不计成功','仅2个开发日志，CI离散且证据量小，不作充分泛化或显著性结论'],goal_complete=False))
 print(json.dumps(dict(rows=len(values),frames=len({(metadata[s]['scene_id'],metadata[s]['frame_idx']) for s in ids}),logs=len(logs))))

if __name__=='__main__':main()
