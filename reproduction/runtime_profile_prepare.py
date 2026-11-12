"""事前冻结开发输入、时钟口径、资源预算和独占GPU流水线；不打开封存测试。"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,tarfile
from conceptgraph.completion.stage9 import OUTPUT,PROJECT,digest,dump
from conceptgraph.completion.semantic_training import validation_subset
from conceptgraph.completion.semantic_evaluation import base_revision

root=OUTPUT/'performance/runtime_profile_protocol_002';root.mkdir(exist_ok=False)
index=json.loads((OUTPUT/'annotation/causal_dev_evidence_001/index.json').read_text())
scenes=sorted({r['scene'] for r in index['cases']})[:4]
frontend=[]
for scene in scenes:
 frames=list(map(json.loads,(OUTPUT/'mapping_evaluation/causal_mapping_full_001'/scene/'frames.jsonl').open()))
 for i in sorted({0,len(frames)//2,len(frames)-1}):
  r=frames[i];frontend.append(dict(input_id=f'{scene}/{i:06d}',scene=scene,rgb=r['rgb'],rgb_sha256=digest(r['rgb'])))
data=OUTPUT/'semantic_dataset/structured_supervision_002/validation.jsonl'
all_rows=validation_subset(list(map(json.loads,data.open())),0)
rows=[];seen=set();counts={}
from conceptgraph.vlm.qwen_semantic_delegate import _semantic_metadata,_semantic_edge
for row in all_rows:
 task=row['task']
 if counts.get(task,0)>=24:continue
 if task=='query_parse':key_input=dict(task=task,query_text=' '.join(row['query_text'].split()))
 elif task=='object_attributes':key_input=dict(task=task,crop_sha256=row['crop_sha256'],metadata=_semantic_metadata(row['metadata']))
 else:key_input=dict(task=task,query_text=' '.join(row['query_text'].split()),candidate_nodes=row['candidate_nodes'],relation_metadata=[_semantic_edge(e) for e in row['relation_metadata']])
 key=hashlib.sha256(json.dumps(key_input,sort_keys=True).encode()).hexdigest()
 if key in seen:continue
 seen.add(key);rows.append(row);counts[task]=counts.get(task,0)+1
assert counts==dict(candidate_rerank=24,object_attributes=24,query_parse=24)
# 性能执行不携带正确答案/可接受身份，不计算准确率。
rows=[{k:v for k,v in row.items() if k in {'task','sample_id','scene_id','log_token','query_text','crop_path','crop_sha256','metadata','candidate_nodes','relation_metadata'}} for row in rows]
assert len(rows)==72
for row in rows:
 if row.get('crop_path'):assert digest(row['crop_path'])==row['crop_sha256']
new=json.loads((OUTPUT/'training/three_task_v2_formal_seed17_001/training_report.json').read_text())['best_checkpoint']['path']
variants={}
for name,path in [('base',None),('old',PROJECT/'outputs/project_v2/qwen_eval/finetune_v2/best_adapter'),('new',Path(new))]:
 variants[name]=dict(adapter=str(path) if path else None,weights_sha256=digest(path/'adapter_model.safetensors') if path else None)
tasks_path=OUTPUT/'interactive_simulation/synthetic_tasks_v1.json';tasks=json.loads(tasks_path.read_text())
task=next(t for t in tasks['tasks'] if t['task_id']=='config00-positive0');config=next(c for c in tasks['configs'] if c['config_id']=='config00')
case=OUTPUT/'interactive_simulation/realtime_fault_cpu_smoke_002_baseline/config00-positive0/seed17'
recorded=[]
for p in sorted((case/'sensor_evidence').glob('*.json')):
 frame=json.loads(p.read_text());i=frame['frame_idx']
 recorded.append(dict(frame_idx=i,sensor_sha256=digest(p),detections_sha256=digest(case/'detections'/f'{i:06d}.pkl.gz')))
plan=dict(version='runtime-profile-v1',created_utc=datetime.now(timezone.utc).isoformat(),
 frontend_inputs=frontend,semantic_inputs=rows,semantic_variants=variants,base_revision=base_revision(),
 classes=[s.strip() for s in (PROJECT/'conceptgraph/vehicle_classes.txt').read_text().splitlines() if s.strip()],
 recorded_case=str(case),recorded_query=task['query_text'],recorded_prior={k:config[k] for k in ['centerline_xy','road_polygon_xy']},
 recorded_frames=recorded,recorded_execution_sha256=digest(case/'execution.jsonl'),
 task_release_sha256=digest(tasks_path),semantic_validation_sha256=digest(data),
 formal_GPU_pipeline=['frontend12','base72+72cache','old72+72cache','new72+72cache','recorded10frames391statesCPU'],
 total_GPU_fresh_requests=216,CPU_frontend_smoke_requests=12,
 GPU_child_timeout_s=3600,total_GPU_wall_budget_s=10800,CPU_smoke_wall_budget_s=1200,
 CPU_child_maximum_RSS_not_enforced=True,storage_budget_bytes=2000000000,minimum_free_bytes=6000000000,
 warm_policy='每任务首次请求/每模型首次预测与暖态分开，失败仍保留；不按成功或快慢筛选',
 cold_policy='全新子进程/模型加载，不清理OS磁盘缓存，不代表断电冷机',
 measure_policy='perf_counter_ns；CUDA前端/语义同步；检测/SAM/CLIP独立计时，嵌套总耗时不相加',
 exclusivity_policy='正式GPU逐子进程执行，启动前及每0.5秒nvidia-smi记录；任何其它CUDA PID/采样错误取消独占信用；短于采样间隔的外部活动无法排除',
 subprocess_execution_policy='仅等待450主矩阵、450缺一/750条件及GPU传感器186故障全部结束后运行，不并发模型；失败不重试',
 scopes=['开发固定4场景12RGB、72相同语义请求；排序含受控符号、属性含真实crop，非自然道路准确率',
 '逐阶段固定输入微基准，不是异步完整导航吞吐或硬实时','录制链重算不增加前端调用/控制执行/实际到达',
 'CPU试跑共享主机，不获独占性能或GPU信用','当前原始模型和旧Adapter均使用相同completion_v3及预算'],
 selection_deduplicated_before_inference=True,previous_protocol='runtime_profile_protocol_001; contains one duplicated rank input, semantic modes never executed',new_test_opened=False,goal_complete=False)
assert len(frontend)==12 and len(recorded)==10
source_paths=sorted({PROJECT/'conceptgraph/completion/runtime_profiling.py',Path(__file__).resolve(),
 PROJECT/'conceptgraph/scripts/streamlined_detections.py',PROJECT/'conceptgraph/completion/semantic_evaluation.py',
 PROJECT/'conceptgraph/completion/semantic_contract.py',PROJECT/'conceptgraph/vlm/qwen_semantic_delegate.py',
 PROJECT/'conceptgraph/vlm/qwen_vl_chat.py',PROJECT/'conceptgraph/vlm/qwen_cache.py',PROJECT/'conceptgraph/vlm/image_input.py'})
plan['source_sha256']={str(p.relative_to(PROJECT)):digest(p) for p in source_paths}
dump(root/'plan.json',plan)
with tarfile.open(root/'source_snapshot.tar.gz','x:gz') as archive:
 for p in source_paths:archive.add(p,arcname=str(p.relative_to(PROJECT)))
print(json.dumps(dict(plan=str(root/'plan.json'),frontend_inputs=len(frontend),semantic_requests=len(rows),model_calls=0,goal_complete=False)))
