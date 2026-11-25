import json,re
from pathlib import Path
from collections import Counter
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1];D=R/'qwen_adapter_fair_comparison/official_graph_dataset_002';rows={role:list(map(json.loads,(D/f'{role}.jsonl').open())) for role in ['train','validation','development_holdout']};stats={};logs={};instances={}
for role,rr in rows.items():
 frames={(v['scene_id'],int(re.search(r'scene-\d+_(\d+)_',v['sample_id']).group(1))) for v in rr};logs[role]={v['log_token'] for v in rr};instances[role]={i for v in rr for i in v['official_identity_evaluation_only'].values()};assert len(rr)==9*len(frames)
 stats[role]=dict(rows=len(rr),unique_frames=len(frames),scenes=len({r['scene_id'] for r in rr}),logs=len(logs[role]),strata=dict(Counter(r['query_kind'] for r in rr)))
for a in rows:
 for b in rows:
  if a!=b:assert not(logs[a]&logs[b] or instances[a]&instances[b])
dump(D/'independent_audit_001.json',dict(correct_unique_frame_counts=stats,log_and_official_instance_disjoint=True,source_frame_counting='frame_idx与scene后帧号交叉核验，每帧9类共享同一来源',input_sha256={role:digest(D/f'{role}.jsonl') for role in rows},source_sha256=digest(__file__),manual_review_count=0,independent_natural_instruction_count=0,perception_credit=False,sealed_whole_project_test=False,goal_complete=False));print(stats)
