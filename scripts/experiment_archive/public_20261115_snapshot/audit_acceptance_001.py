"""从已完成原始回执汇总验收状态；缺失证据不得用子任务完成替代。"""
import csv,datetime,json,shutil
from pathlib import Path
from conceptgraph.completion.stage9 import digest,dump
R=Path(__file__).resolve().parents[1];Q=R/'qwen_adapter_fair_comparison';O=R/'json/acceptance_audit_001'
def main():
 O.mkdir(exist_ok=False);proofs={}
 def load(name):
  p=R/name;proofs[str(p)]=digest(p);return json.loads(p.read_text())
 k=load('statistics/kitti_expanded_full_001/report.json');n=load('nuscenes_ablation/statistics_001/report.json');c=load('carla_official/ablation_statistics_004/report.json');h=load('json/protected_history_audit_001/report.json');q=load('qwen_adapter_fair_comparison/real_graph_completion_audit_001/report.json');a=load('qwen_adapter_fair_comparison/real_graph_weighted_pilot_analysis_001/report.json');p=load('qwen_adapter_fair_comparison/real_graph_permutation_002/report.json')
 manual=sum(len(list((R/sub).glob('*.json'))) for sub in ['annotation/human_submissions','annotation/human_natural_submissions'])
 training=[]
 for seed in [17,29,43]:
  t=load(f'qwen_adapter_fair_comparison/training/real_graph_seed{seed}_001/training_report.json');training.append(dict(seed=seed,steps=t['completed_steps'],status=t['status']))
 episodes=list(csv.DictReader((R/'carla_official/ablation_statistics_004/episodes.csv').open()));positives=[r for r in episodes if r['condition']=='full_rule_geometry' and not r['episode'].startswith('no_match')]
 total=sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink());free=shutil.disk_usage(R).free
 items=[dict(item='KITTI连续序列实际建图',status='pass',proof=dict(sequences=k['sequence_count'],frames=k['frame_count']),scope='离线连续建图/关联；非物理导航、非官方AP'),dict(item='nuScenes六条件对象关系消融',status='pass_with_scope_limit',proof=dict(scenes=n['scenes'],logs=n['logs'],frames_each=n['frames_per_condition'],actual_mapper_calls=n['actual_new_mapper_calls']),scope='六组复杂Qwen语义仍未全部接入'),dict(item='真实图三种子训练与同输入七模型比较',status='pass_development_only',proof=dict(training=training,actual_GPU_requests=q['actual_upstream_GPU_requests'],same_tokens=q['same_input_token_count_each_sample']),scope='不是新封存测试；目标选择没有证实正提升'),dict(item='真实图三种子五条件语义合作',status='pass_offline_development',proof=dict(seeds=3,conditions=5,queries_each=108,actual_CPU_cooperation=1620),scope='只给离线提议信用，无在线控制/到达'),dict(item='逆序压力测试与单种子损失试验',status='pass_development_diagnostic',proof=dict(permutation_GPU=p['new_actual_GPU_requests'],pilot_GPU=a['new_actual_GPU_eval_requests']),scope='试验负结果保留，不替换三种子主协议'),dict(item='官方CARLA五条件同清单',status='partial',proof=dict(complete=c['processing_completed'],expected=c['expected'],distinct_tasks=c['distinct_frozen_tasks'],full_positive_completed=len(positives),full_positive_strict_arrivals=sum(r['strict_safe_arrival']=='True' for r in positives)),scope='末次资源中断；规则几何已闭环，复杂Qwen尚未；不足原课题更大任务/种子覆盖'),dict(item='真实人工颜色细属性歧义审核',status='pending_human',proof=dict(actual_submission_files=manual),scope='不得由执行代理伪造审核人/提交'),dict(item='新封存全系统测试',status='not_started',proof=dict(opened=False),scope='最终方法尚未冻结；开发日志不得改名封存'),dict(item='正式论文科学图',status='pending_backend_choice',proof=dict(user_selected_backend=None),scope='nature-figure明确要求用户选Python/R后绘图'),dict(item='全新环境安装及全链复跑',status='not_done',proof=dict(clean_environment_install_passed=False,isolated_source_smoke_passed=True),scope='已装依赖的独立源码展开重计分不等于干净环境安装'),dict(item='历史保护',status='pass',proof=dict(files=h['protected_files_checked'],all_match=h['all_protected_paths_match_baseline']),scope='只读哈希比对，无删除修复'),dict(item='容量',status='near_budget_limit',proof=dict(project_output_bytes_at_audit=total,budget_bytes=10*2**30,remaining_bytes=10*2**30-total,physical_free_bytes=free),scope='提高项目产物上限的询问待答；没有自行使用物理空闲突破10GiB')]
 assert k['sequence_count']==10 and k['frame_count']==1000 and n['frames_per_condition']==1508 and q['actual_upstream_GPU_requests']==756
 assert total<10*2**30 and free>=20*2**30 and h['all_protected_paths_match_baseline']
 assert all(t['steps']==240 and t['status']=='completed' for t in training)
 dump(O/'report.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),items=items,proof_sha256=proofs,source_sha256=digest(__file__),all_requirements_met=False,goal_complete=False,stop_large_runs_until_budget_approved=True,thesis_prose_written=False,git_commit_or_push_performed=False,artifact_manifest_pending_at_this_audit=True));print(json.dumps(dict(items=len(items),manual_files=manual,complete_CARLA=c['processing_completed'],goal_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
