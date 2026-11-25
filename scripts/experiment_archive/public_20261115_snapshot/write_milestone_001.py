"""阶段快照，历史不覆盖；不将部分成果标记为整个Goal完成。"""
import json,datetime,shutil,os
from pathlib import Path
from conceptgraph.completion.stage9 import dump,digest,PROJECT
R=Path(__file__).resolve().parents[1]
def main():
 O=R/'json/milestone_20261116_001.json';size=sum(p.lstat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink());q=R/'qwen_adapter_fair_comparison/existing_evidence_audit_001/report.json';ns=R/'nuscenes_ablation/statistics_001/report.json';kt=R/'kitti_tracking_expanded/completion_receipt_001.json';mp=R/'method_comparison/mapping_same_input_002/report.json';c=R/'carla_official/pipeline_episode_006/independent_audit_002/report.json';free=shutil.disk_usage(R).free
 state=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal_complete=False,goal_status='active',completed=dict(KITTI=dict(sequences=10,frames=1000,actual_GPU_frontend=True,actual_map=True,actual_query_graph=True,closed_loop_arrival=False,receipt=str(kt)),nuScenes=dict(scenes=38,logs=11,frames_per_condition=1508,object_relation_conditions=6,new_mapper_calls=9048,semantic_evaluated=False,receipt=str(ns)),mapping_comparison=dict(methods=6,unique_frames=1508,new_mapper_calls=0,internal_variants=True,external_reproductions_complete=False,receipt=str(mp)),qwen_existing_audit=dict(models=5,requests=3015,new_GPU_calls=0,new_training_this_goal=False,receipt=str(q))),CARLA=dict(official_version='0.9.16',server_port=2010,actual_physical_control=True,sensors_synchronized=True,development_geometric_arrival=True,strict_safe_arrival=False,latest_completed_episode='pipeline_episode_006',independent_audit=str(c),currently_running='pipeline_episode_007',final_Qwen_evaluated=False,formal_conditions_complete=False),required_remaining=['nuScenes相同图上的最终语义模块与独立语义消融','统一输入的目标选择和规划诊断比较，外部方法复现或明确不可比边界','九类语义任务、新Adapter的本轮独立监督与真实GPU运行、完整合作消融','官方CARLA实际轮角/速度/加减速均满足限制的到达及五条件统一清单','颜色/细属性/歧义的人工审核记录：当前未找到','方案冻结后开启新封存测试；当前未打开','正式图表、逐项文件哈希与干净环境复现交付'],storage=dict(project_free_gib=free/2**30,output_logical_bytes_without_symlink_follow=size,output_budget_bytes=10*2**30,large_writes_allowed=free>=20*2**30 and size<10*2**30,deleted_files=[]),proof_sha256={str(p):digest(p) for p in [q,ns,kt,mp,c]});dump(O,state)
 text='''# 实验交付进度与复现入口

当前总体验收：**goal_complete=false**。本目录为后续实验产物，历史文件、失败记录、模型和论文原稿未删除。当前不写论文正文。

## 已落地的结果

|部分|实际执行范围|入口|
|---|---|---|
|KITTI连续实验|10序列，每序列100连续帧，共1000帧；真实检测、分割、CLIP、融合、关联、图和固定查询|kitti_tracking_expanded/结果与逐帧目录说明.md；index.html；completion_receipt_001.json|
|nuScenes对象/关系单模块消融|38场景、11日志、每组1508帧，6组全量实际重建|nuscenes_ablation/statistics_001/结果说明.md、metrics.csv|
|建图结构比较|同一1508帧历史真实输出，6种内部结构；本轮重计分，无新建图调用|method_comparison/mapping_same_input_002/|
|已有Qwen公平对照核验|5模型×603原始输出，核验数据哈希并重新计分；本轮新增GPU调用0|qwen_adapter_fair_comparison/existing_evidence_audit_001/|
|官方CARLA|0.9.16，真实离屏RGB/深度/LiDAR/位姿/碰撞与物理控制已跑通|carla_official/server_connection_001.json；sensor_control_probe_002/|
|CARLA完整项目链|006已选对、锁定、进入目标区并停车保持2秒；独立审核发现7次加速与10次轮角变化超限，严格安全到达不通过|carla_official/pipeline_episode_006/independent_audit_002/report.json|

上述建图消融不含Qwen，内部结构比较不冒充外部论文复现，KITTI不声称实际到达。历史Adapter在部分下游任务中的负结果保留。人工颜色等核验仍待记录，新封存测试未打开。

## 复现环境和命令

项目根为`/root/autodl-tmp/lizichong`；Python为`/root/miniconda3/envs/conceptgraph/bin/python`。每次先设置：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/root/autodl-tmp/lizichong
```

源代码均在本目录`source/`，生产代码未由本轮新runner修改。正式运行目录默认拒绝覆盖；复跑应改为新run-id或新版本目录。下列命令中的`R`为本目录绝对路径：

```bash
R=/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1/public_experiment_completion_20261115
/root/miniconda3/envs/conceptgraph/bin/python "$R/source/kitti_tracking_gpu_001.py" --help
/root/miniconda3/envs/conceptgraph/bin/python "$R/source/nuscenes_strict_ablation_002.py" --help
/root/miniconda3/envs/conceptgraph/bin/python "$R/source/audit_carla_arrival_002.py" --episode pipeline_episode_006
```

最后一条审计命令也拒绝覆盖已有审计目录；已跑结果直接阅读即可，重新审计需新输出目录。KITTI完整选择、输入和方法参数见其release、manifest与逐帧说明；nuScenes六个strict运行均有release.json和帧级JSONL，所有版本哈希均在那里。

CARLA API需将共享卷`/autodl-fs/data/third_party/carla_0.9.16/python_site`加到PYTHONPATH。官方服务器以nobody运行；安装、隔离图形依赖与HOME/XDG在共享卷，软链接入口见`carla_official/`。不能在episode运行时启动另一个更改世界/时钟的客户端。

## 待补项与限制

完整语义合作、九类挑战与语义消融、本轮新训练及统一模型复测仍未完成；外部方法、规划比较与CARLA统一五条件尚未交付；CARLA到达必须通过独立速度、轮角、加减速、安全和身份审核；人工审核与封存测试仍待补。最终图表与全量干净环境复现须在这些结果完成后验收。

本文件是阶段性交付入口，不是整课题完成声明。具体状态见`json/milestone_20261116_001.json`。低于20GiB空闲或本目录到达10GiB实验输出预算时停止大写入，先留原始前缀和恢复命令。没有执行清理删除。
'''
 (R/'交付说明_阶段快照001.md').write_text(text)
 print(json.dumps(state,ensure_ascii=False))
if __name__=='__main__':main()
