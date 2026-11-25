"""实际结果阶段快照002，完整任务仍为false；保留快照001。"""
import datetime,json,shutil
from pathlib import Path
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1]
def main():
 proofs=[R/n for n in ['kitti_tracking_expanded/completion_receipt_001.json','nuscenes_ablation/statistics_001/report.json','method_comparison/mapping_same_input_002/report.json','method_comparison/ab3dmot_statistics_002/report.json','method_comparison/recorded_retrieval_audit_001/report.json','method_comparison/planning_official_snapshots_001/report.json','statistics/xyz_center_supplement_001/report.json','tables/semantic_summary_001/report.json','carla_official/pipeline_episode_010/independent_audit_003/report.json','carla_official/ablation_registration_001.json']]
 assert all(p.exists() for p in proofs)
 state=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_complete=False,goal_status='active',completed=dict(KITTI=dict(sequences=10,continuous_frames=1000,actual_GPU_frontend=True,actual_mapping=True,graph_queries=3000,graph_query_independence='10序列×3类；相邻帧查询相关',arrival_evaluated=False),nuScenes=dict(scenes=38,logs=11,frames_per_condition=1508,object_relation_conditions=6,actual_mapper_calls=9048,full_final_semantics_integrated=False),external_tracking_adapter=dict(name='official AB3DMOT 2020 core + shared frontend adapted',unique_frames=2508,nuScenes=1508,KITTI=1000,actual_CPU_projection_and_tracker=True,author_full_benchmark_reproduction=False),recorded_retrieval_audit=dict(methods=4,queries=1115,positive=733,no_match=382,new_inference_calls=0),planning_comparison=dict(methods=3,actual_official_sensor_snapshots=24,unique_source_development_episodes=4,physical_control_evaluated=False),new_Qwen=dict(training_seeds=[17,29,43],actual_training_steps_per_seed=240,train_rows=540,validation_rows=180,development_holdout_rows=135,models=6,actual_GPU_requests=810,actual_offline_cooperation_decisions=2025,strata=9,scope='官方标注几何给定图+受控语言，不计感知/自然语言/闭环信用'),semantic_auxiliary=dict(base_target_only_correct=30,old_adapter_target_only_correct=30,new_seed17_target_only_correct=73,new_seed29_target_only_correct=73,new_seed43_target_only_correct=74,denominator=135,posthoc_auxiliary=True,strict_primary_also_preserved=True)),CARLA=dict(official_version='0.9.16',latest_completed_development='pipeline_episode_010',route=True,correctly_locked=True,geometric_identity_stop_arrival=True,strict_safe_arrival=False,strict_failure='一次实际加速度超1m/s2，实测峰值1.1260788m/s2',registered_ablation_conditions=5,registered_episodes=15,formal_execution_state='见json/carla_ablation_jobs_001逐任务状态；本快照不预授予完成信用',Qwen_complex_branch_verified=False),negative_results=['KITTI本方法召回.841高于AB3DMOT适配.657，但IDF1 .504低于.558，重复节点307高于58','完整语义合作约.75至.78，固定顺序消融.844，受限完整规则1.0；保留全部结果','CARLA010几何到达但存在一次加速度超限，严格安全到达不通过','历史三任务Adapter在新契约上严格4/135、编号9/135，跨任务退化保留'],required_remaining=['nuScenes完整最终语义合作与对象/关系消融集成同口径验收，目前只有对象关系六组和独立给定图诊断','真实感知候选上的最终语义模块、与强规则/特征方法共同候选比较；不能拿oracle图分数替代','官方CARLA五条件同清单实际完成并独立审计；当前批次仍是规则几何简单类别任务，不是完整Qwen复杂语义闭环','人工颜色/细属性/歧义记录仍为0，核验服务8877保持开放等待真实提交','方案冻结后开启整课题新封存测试；当前没有新封存整系统成绩','正式科学图待绘图工具偏好，全部可编辑数据先完成','干净环境软件依赖/输入哈希/代码快照/恢复步骤和完整交付验收'],resources=dict(project_free_bytes=shutil.disk_usage(R).free,output_budget_bytes=10*2**30,minimum_free_bytes=20*2**30,deletions=[]),proof_sha256={str(p):digest(p) for p in proofs})
 dump(R/'json/milestone_20261116_002.json',state)
 (R/'交付说明_阶段快照002.md').write_text('''# 后续实验阶段交付快照002

当前 **goal_complete=false**。实际运行和可核验结果如下，未撰写论文正文。

|实验|真实范围|结果入口|
|---|---|---|
|KITTI连续建图|10序列、1000帧，真实检测/分割/CLIP、融合、关联、场景图；3000条按帧固定类别查询|`kitti_tracking_expanded/index.html`、`结果与逐帧目录说明.md`|
|nuScenes对象/关系消融|38场景、11日志、6组各1508帧，共9048次真实建图更新|`nuscenes_ablation/statistics_001/metrics.csv`|
|建图内部结构比较|6种内部结构，同1508帧历史真实输出复核；本次新建图调用0|`method_comparison/mapping_same_input_002/`|
|官方AB3DMOT适配比较|本轮重新投影并实际运行官方跟踪核心：nuScenes1508帧、KITTI1000帧|`method_comparison/ab3dmot_statistics_002/`；原始帧在`ab3dmot_shared_full_001/`|
|真实记录检索对照|四方法同1115条旧开发查询，733有目标、382范围内无匹配；本轮重计分与聚类CI|`method_comparison/recorded_retrieval_audit_001/`|
|规划诊断|24份官方CARLA实际传感器快照×3规划器真实搜索；不授予新车辆执行信用|`method_comparison/planning_official_snapshots_001/`|
|三维中心补充|原XY匹配上的Z/XYZ误差及新增XYZ3m门限诊断|`statistics/xyz_center_supplement_001/`|
|新Adapter|540训练/180验证/135开发留出；三个种子各240实际步|`qwen_adapter_fair_comparison/training/graph_v2_seed*_001/`|
|Qwen六模型公平评测|同135请求×6模型，810次真实GPU推理，保留原始输出|`qwen_adapter_fair_comparison/official_graph_fair_gpu_001/`|
|语义五组开关|三个种子×五条件×135请求，2025次真实离线合作判断|`tables/semantic_summary_001/`与`qwen_adapter_fair_comparison/official_graph_decisions_001/`|
|官方CARLA开发010|真实传感器、物理控制、正确锁定、进入终区并停车2秒；一次加速度超限|`carla_official/pipeline_episode_010/independent_audit_003/report.json`|

## 可以成立的结论及限制

KITTI本方法在该局部协议的召回率84.10%，AB3DMOT共享前端适配65.74%；但身份一致性IDF1为50.35%对55.79%，重复对象307对58。不能只报召回率优势或称本方法所有指标最优。AB3DMOT输入是共享前端的部分可见点云轴对齐框，朝向不使用真值；这不是作者原检测器或官方排行榜复现。

新Adapter统一编号辅助正确数为73、73、74/135，基础/旧Adapter均30/135；严格契约主指标也完整保留。合作完整版约75%—78%，固定顺序消融84.44%，受限语法完整规则100%，说明目前复杂重排并非优势模块。该组使用给定官方几何图和受控证据字段，不能当真实感知、自然语言或道路导航成绩。

CARLA010几何身份停车到达成立，但实际加速度峰值1.126m/s²超过1m/s²限值，因此严格安全到达为false。车辆已改为低扭矩慢速平台，原厂动力学和各失败轮次均保留，不混报。五条件15个episode已预注册，执行状态逐项见`json/carla_ablation_jobs_001/`；未完成的结果不预记成功。该批以清晰类别规则为主，不能冒充Qwen复杂语义闭环。

## 仍不能验收的部分

nuScenes完整最终语义模块的同口径集成、真实候选的最终语义公平对照、CARLA完整复杂语义闭环及五条件完成审核、真实人工颜色/细属性/歧义核验、新封存整系统测试、正式图表、干净环境复现及逐项哈希仍需完成。核验服务仍开放于服务器127.0.0.1:8877，正确页面及端口转发方式见`annotation/核验页面使用说明002.md`；当前真实提交仍为0。

## 目录与复现

本批根目录为`/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1/public_experiment_completion_20261115/`。新runner在`source/`，历史生产代码未修改；输入数据、checkpoint和旧结果通过release/manifest哈希定位，不额外复制。运行默认拒绝覆盖已有目录。新环境须先恢复输入及权重路径，再按对应runner的`--help`和release执行；不是只有脚本即可凭空复现实验。

Python为`/root/miniconda3/envs/conceptgraph/bin/python`；固定线程`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`、`PYTHONDONTWRITEBYTECODE=1`。CPU重计分使用`CUDA_VISIBLE_DEVICES=''`，真实前端/Qwen使用0，禁止与训练重叠。实际安装包版本见`json/runtime_pip_packages_001.json`，仓库旧environment.yml不能代替当前运行版本。

CARLA固定官方0.9.16，安装与图形依赖在共享卷；新episode和日志保留。闭环期间禁止第二客户端推进世界时钟。阶段快照不等于最终验收文件，完整状态和输入回执哈希见`json/milestone_20261116_002.json`。

项目实验产物预算10GiB、余量低于20GiB停止大写入；CARLA官方安装/压缩包约27GiB在共享卷单列，不能声称全部新增物理存储也小于10GiB。缓存只列候选，无删除。
''',encoding='utf-8')
 print(json.dumps(dict(milestone=2,proofs=len(proofs),goal_complete=False),ensure_ascii=False))
if __name__=='__main__':main()
