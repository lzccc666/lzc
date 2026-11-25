"""等待冻结15回合结束后独立汇总；计时/到达/拒绝分别给分母，未成功不排除。"""
import csv,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import dump,digest
from audit_carla_choices_004 import audit
R=Path(__file__).resolve().parents[1];O=R/'carla_official/ablation_statistics_004'
def main():
 O.mkdir(exist_ok=False)
 while not (R/'json/carla_ablation_jobs_003/summary.json').exists():time.sleep(20)
 registration=R/'carla_official/ablation_registration_003.json';bank=json.loads(registration.read_text());rows=[];failed=[]
 for ep in bank['episodes']:
  for condition in bank['conditions']:
   e=R/'carla_official/ablation_batch_003'/ep['episode']/condition
   if not (e/'summary.json').exists() or not (e/'independent_audit_003/report.json').exists():failed.append(dict(episode=ep['episode'],condition=condition,path=str(e)));continue
   report=e/'independent_choice_audit_004/report.json';rows.append(json.loads(report.read_text()) if report.exists() else audit(e))
 table=[];metrics=['route_generated','first_proposal_correct_ever','correct_stable_lock_ever','entered_independent_region','geometric_identity_stop_arrival','strict_safe_arrival','no_collision','no_lane_violation','timeout']
 for condition in bank['conditions']:
  all_rows=[p for p in rows if p['condition']==condition];positive=[p for p in all_rows if p['positive_task']];negative=[p for p in all_rows if not p['positive_task']]
  for key in metrics+['no_match_any_proposal','no_match_any_stable_lock','no_match_correct_refusal']:
   cases=negative if key.startswith('no_match') else positive
   if not cases:continue
   values=np.asarray([float(p[key]) for p in cases]);draw=np.random.default_rng(17).integers(0,len(cases),(4000,len(cases)));low,high=np.percentile(values[draw].mean(axis=1),[2.5,97.5]);table.append(dict(condition=condition,metric=key,numerator=int(values.sum()),denominator=len(cases),value=float(values.mean()),ci95_low=float(low),ci95_high=float(high),unit='task',independent_configuration_warning='共享Town/起点/低速执行器；仅两个正任务、一个负任务，各一次运行，不足以评估泛化'))
 with (O/'comparison.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
 case_fields=['episode','condition','positive_task','final_status','source_perception_frames','execution_ticks','candidate_coverage_ever','first_proposal_correct_ever','correct_stable_lock_ever','wrong_stable_lock_ever','route_generated','entered_independent_region','geometric_identity_stop_arrival','strict_safe_arrival','no_collision','no_lane_violation','timeout','no_match_any_proposal','no_match_any_stable_lock','no_match_correct_refusal','mean_frontend_mapping_selection_plan_s','median_frontend_mapping_selection_plan_s','p95_frontend_mapping_selection_plan_s','wall_s','sim_s','maximum_acceleration_mps2']
 with (O/'episodes.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=case_fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 dump(O/'report.json',dict(processing_completed=len(rows),expected=len(bank['episodes'])*len(bank['conditions']),failed_or_missing=failed,actual_official_episodes=len(rows),distinct_frozen_tasks=len(bank['episodes']),conditions=bank['conditions'],perception_calls=sum(p['source_perception_frames'] for p in rows),execution_ticks=sum(p['execution_ticks'] for p in rows),source_sha256=digest(__file__),registration_sha256=digest(registration),audit_source_sha256=digest(R/'source/audit_carla_choices_004.py'),result_sha256={str(R/'carla_official/ablation_batch_003'/p['episode']/p['condition']/'independent_choice_audit_004/report.json'):digest(R/'carla_official/ablation_batch_003'/p['episode']/p['condition']/'independent_choice_audit_004/report.json') for p in rows},limitations=['官方0.9.16，实际RGB/深度/LiDAR、物理执行、下一帧反馈；非自建仿真','仅清晰类别规则几何链；复杂Qwen未激活，不能冒称整个语义架构的闭环消融','无历史维护实际仅关闭动态TTL，不是所有历史/运动/关联同时删除','当前帧条件同时受统一3次观测证据门限制，无法形成历史确认；此为模块交互诊断，不能独立证明融合模块的新颖性','类别指令未使用对象关系；无关系条件差异可能来自物理/规划时序，不能将单次加速度差归因于关系模块贡献','每任务仅一个固定种子，CI极粗且不代表独立道路泛化','车辆是25Nm扭矩上限的低速Model3，并非原厂Tesla动力学；执行限值保持不变','旧correct_object_selected实际上检查候选覆盖，新审计单列首位提议、稳定锁定、指定区域/停止/朝向/保持与严格限值','负任务不进入正任务到达率分母；无路径不能被叫作成功安全导航'],goal_complete=False))
 (O/'结果说明.md').write_text('本批是官方CARLA的15次真实闭环开发验证，包含2个正任务和1个无匹配任务，每个任务5种条件。结果见comparison.csv，逐任务和实际计时见episodes.csv，原始传感器与执行记录保留在ablation_batch_003。\n\n严格到达要求独立身份、指定终区、朝向、速度、连续保持、无碰撞、不越界及执行限值全部通过。几何到达但加速度超限的回合保留为严格失败。旧汇总的“correct_object_selected”只证明候选覆盖，已通过新审计区分首位提议和锁定。\n\n本批未使用复杂Qwen，也未达到原课题50任务×3种子或新封存测试要求。简单类别不激活关系推理，不能把无关系消融的物理波动当作关系模块优势。完整局限和哈希见report.json。\n')
 print(json.dumps(dict(audited=len(rows),missing=len(failed),goal_complete=False)),flush=True)
if __name__=='__main__':main()
