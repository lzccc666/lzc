"""重算已有真实GPU公平对照，不新增推理或把符号候选冒充公开道路成绩。"""
import csv,json,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from conceptgraph.completion.stage9 import OUTPUT,digest,dump
from conceptgraph.completion.semantic_evaluation import metrics_for
R=Path(__file__).resolve().parents[1];O=R/'qwen_adapter_fair_comparison/existing_evidence_audit_001';S=OUTPUT/'main_experiments/fair_three_task_full_validation_001';D=OUTPUT/'semantic_dataset/structured_supervision_002'
def main():
 O.mkdir(exist_ok=False);release=json.loads((S/'release.json').read_text());manifest=json.loads((D/'manifest.json').read_text());assert digest(D/'manifest.json')==release['dataset_manifest_sha256']
 for name,sha in manifest['files'].items():assert digest(D/name)==sha
 allrows={r['sample_id']:r for r in map(json.loads,(D/'validation.jsonl').open())};rows=[allrows[i] for i in release['sample_ids']];original=json.loads((S/'report.json').read_text());tables=[];data=[];hashes={str(p):digest(p) for p in [S/'release.json',S/'report.json',D/'manifest.json',D/'validation.jsonl',Path(__file__)]}
 variants=['base_fullschema','old_adapter_fullschema','new_adapter_seed17','new_adapter_seed29','new_adapter_seed43']
 for variant in variants:
  path=S/variant/'raw_records.jsonl';records=list(map(json.loads,path.open()));assert len(records)==len(rows)==603;assert set(r['sample_id'] for r in records)==set(release['sample_ids']);recomputed=metrics_for(records,rows);hashes[str(path)]=digest(path)
  for task,m in recomputed.items():
   assert m['semantic_correct']==original['variants'][variant]['metrics'][task]['semantic_correct'];assert m['requests']==original['variants'][variant]['metrics'][task]['requests'];units=sorted({r['log_token'] for r in records if r['task']==task});a=np.array([[sum(r.get('semantic_correct',False) for r in records if r['task']==task and r['log_token']==u),sum(r['task']==task and r['log_token']==u for r in records)] for u in units]);rng=np.random.default_rng(17);b=a[rng.integers(0,len(units),(4000,len(units)))].sum(axis=1);ci=np.quantile(b[:,0]/b[:,1],[.025,.975]);row=dict(model=variant,task=task,correct=m['semantic_correct'],denominator=m['requests'],accuracy=m['accuracy'],format_valid=m['schema_valid'],format_valid_rate=m['schema_valid_rate'],ci95_low=float(ci[0]),ci95_high=float(ci[1]),cluster_units=len(units),mean_latency_s=float(np.mean([r['envelope']['latency']['total_s'] for r in records if r['task']==task])),data_kind='controlled_symbolic_candidates' if task=='candidate_rerank' else 'weak_supervised_real_crop_category' if task=='object_attributes' else 'constructed_language_heldout_expression_families',new_GPU_calls=0);tables.append(row)
  data.append(dict(variant=variant,records=603,raw_sha256=digest(path),recomputed_metrics=recomputed))
 with (O/'comparison.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)
 dump(O/'report.json',dict(all_metrics_recomputed_equal=True,models=5,rows_per_model=603,historical_actual_GPU_records=3015,new_GPU_calls=0,source_sha256=hashes,bootstrap=dict(seed=17,repetitions=4000,unit='原任务的log_token；排序为符号候选库，不增加真实场景数'),data_counts=manifest['audit']['counts'],training_seeds=[17,29,43],training_evidence=str(OUTPUT/'training/three_task_v2_formal_complete_001.json'),results=data,missing=['本Goal尚未新增训练；这里只复核历史训练与真实GPU原始输出','九类语义挑战未补齐','完整合作模块及五组单因素消融未执行','新封存自然指令与人审仍缺，不能把该验证成绩当新测试','颜色和细属性准确率不从类别成绩推断','自建仿真的下游负结果继续保留，不并入官方CARLA实验'],goal_complete=False))
 (O/'说明.md').write_text('# Qwen已有公平训练与对照复核\n\n已核验5模型×603条原始真实GPU输出，并重新计分；与原报告全部一致。此次新增GPU调用为0，不冒充重新训练或重新推理。\n\n- 数据为2398训练、603验证，训练/验证采集批次和表达家族隔离。新Adapter历史种子17/29/43各240步。\n- 解析为构造语言与完整字段严格计分；真实裁剪任务只评价类别；排序为符号候选，不是真实道路目标检索。\n- 解析、类别、排序的改善不能推出真实导航到达改善。历史自建仿真负结果保留。\n- 完整合作模块、九类挑战、真实公开数据语义消融及新封存测试仍待完成。详见report.json和comparison.csv。\n')
 print('verified 3015 historical real GPU records; new GPU calls=0')
if __name__=='__main__':main()
