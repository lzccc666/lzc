"""生成当前实验交付入口与验收快照；只写新交付目录，不覆盖历史阶段文件。"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

R = Path(__file__).resolve().parents[1]
UTC = datetime.now(timezone.utc).isoformat()

indexes = {
    'models': ('模型与Adapter入口', '模型训练和权重仍保存在 qwen_adapter_fair_comparison/；本目录只提供入口，避免复制大权重。', ['qwen_adapter_fair_comparison/training/real_graph_seed17_001/training_report.json', 'qwen_adapter_fair_comparison/training/real_graph_seed29_001/training_report.json', 'qwen_adapter_fair_comparison/training/real_graph_seed43_001/training_report.json']),
    'replay': ('连续回放入口', 'KITTI连续1000帧诊断和新封存回放索引。当前证据是连续建图/检索回放，不能解释为车辆实际到达。', ['kitti_tracking_expanded/graph_query_001/frame_index.csv', 'main_experiments/sealed_execution_075/evaluation/report.json']),
    'vehicle_execution': ('车辆执行入口', '官方CARLA 0.9.16的实际控制、车身碰撞和独立到达判定证据。用户已要求CARLA先保留10回合并停止扩展。', ['carla_official/carla_ten_041/report.json', 'carla_official/identity_geometry_audit_046/report.json']),
    'interactive_simulation': ('交互感知仿真入口', 'CARLA传感器—感知—建图—语义—规划—控制—新传感器闭环证据。当前规模和严格到达率有限，状态保留为未完全完成。', ['carla_official/carla_ten_041/report.json', 'carla_official/ablation_statistics_004/report.json']),
    'ablations': ('模块消融入口', 'nuScenes建图六条件、语义模块开关和封存图表结果。', ['nuscenes_ablation/statistics_001/report.json', 'figures/sealed_public_078/mapping_metrics.csv', 'figures/sealed_public_078/retrieval_all_metrics.csv']),
    'faults': ('故障与安全入口', '异常、过期、控制边界、无匹配和阻塞诊断。', ['carla_official/ten_bottlenecks_043/report.json', 'carla_official/current_body_causes_066/report.json']),
    'cache_audit': ('缓存与确定性入口', '缓存命中、输入哈希和模型调用核查。', ['json/sealed_delivery_verification_099/report.json', 'main_experiments/fair_three_task_full_validation_001/report.json']),
    'thesis': ('论文实验材料入口', '按用户要求暂不写论文正文；此处只保留可用于后续论文的图表、三线表、模块索引和实验事实。', ['tables/sealed_office_097/封存公开数据实验表.docx', 'figures/sealed_public_078/封存公开数据结果表.xlsx']),
}
for name, (title, intro, evidence) in indexes.items():
    d = R / name
    d.mkdir(exist_ok=True)
    body = [f'# {title}', '', intro, '', '## 当前证据', '']
    body.extend(f'- `{item}`' for item in evidence)
    (d / 'README.md').write_text('\n'.join(body) + '\n', encoding='utf-8')

report = {
    'generated_at_utc': UTC,
    'goal_complete': False,
    'source': 'build_delivery_index_101.py',
    'protected_history': {'files_checked': 255170, 'mismatches': 0, 'evidence': 'json/protected_history_audit_101/report.json'},
    'latest_tests': {'pytest': {'passed': 452, 'failed': 0, 'warnings': 5, 'command': 'PYTHONPATH=/root/autodl-tmp/lizichong:/autodl-fs/data/third_party/carla_0.9.16/python_site /root/miniconda3/envs/conceptgraph/bin/python -m pytest tests -q'}, 'ruff': {'status': 'known_preexisting_optional_branch_errors', 'count': 6, 'evidence': 'json/software_validation_092/static_checks.json'}},
    'sealed_public': {'scenes': 30, 'logs': 14, 'frames_per_mapping': 1196, 'models': 5, 'actual_gpu_calls': 4390, 'evidence': 'json/sealed_delivery_verification_099/report.json'},
    'human_natural': {'raw_records_imported': 1, 'qualifying_labels': 0, 'evidence': 'annotation/offline_imports/human_natural_lzc_20261116_001/source_binding_report.json'},
    'carla': {'episodes': 10, 'positive_tasks': 6, 'strict_safe_arrival': 1, 'status': '按用户上限停止', 'evidence': 'carla_official/carla_ten_041/report.json'},
    'git_sync': {'local_repo': '/root/autodl-tmp/lzc_sync_101/repo', 'commit': '9bfbddcb81430c0bd7f75303035cd3a0c2456212', 'branch': 'codex/thesis-completion-20261116', 'remote_push': 'blocked_authentication', 'evidence': 'source_bundle/github_sync_101/report.json'},
    'remaining': ['自然导航独立合格标签仍不足', 'CARLA严格闭环规模和正确到达证据有限', '部分主对比与参数分组尚未达到原始Goal的全部门槛', 'GitHub远程写入凭据待配置'],
}
(R / 'json/delivery_index_101.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

(R / 'experiment_report.md').write_text('''# 实验交付总报告（当前版本）\n\n本报告只汇总已经保存并核验的实验事实，不把未完成项目写成完成。\n\n## 已完成\n\n- KITTI：10 个连续序列片段、每段 100 帧，共 1000 帧，逐帧保留 RGB、深度、检测、建图和图查询入口。\n- nuScenes：38 个场景、11 个 log、1508 帧，完成六种建图条件评价。\n- 新封存公开实验：30 个场景、14 个 log、1196 帧建图输入；5 个模型共 4390 次真实 GPU 调用，输入请求和 GT 隔离核验通过。\n- Qwen 新 Adapter：3 个随机种子完成训练和封存推理；结果没有超过基础 Qwen，负结果保留。\n- CARLA：官方 0.9.16，已完成 10 回合真实传感器闭环审计；6 个有目标任务中严格安全到达 1 个，按用户要求停止扩展。\n- 工程：最新环境测试 452 项通过、0 失败、5 个警告；历史保护清单 255170 个文件 0 变化。\n\n## 主要结果\n\n完整建图方案在新封存建图表中对象召回率 28.52%、IDF1 29.34%、中心误差 1.157 m、关系 F1 12.77%。新 Adapter 三个种子在模板主集正目标选对率分别为 7.20%、0.38%、3.79%，基础 Qwen 为 14.77%；不能宣称微调带来正提升。\n\n## 证据入口\n\n- 图表：`figures/sealed_public_078/`\n- Word/PDF 三线表：`tables/sealed_office_097/`\n- 真实模型执行：`main_experiments/sealed_execution_075/`\n- 人工自然指令导入：`annotation/offline_imports/human_natural_lzc_20261116_001/`\n- CARLA：`carla_official/carla_ten_041/`\n- 最新验收：`json/delivery_index_101.json`\n\n## 当前限制\n\n自然导航人工记录目前只有 1 条，合格封存标签为 0；CARLA 仅保留 10 回合且严格到达证据有限；部分原始 Goal 要求的全量方法对比、参数泛化和实时故障矩阵仍未全部完成。因此 `goal_complete=false`。\n''', encoding='utf-8')

(R / 'remaining_issues.md').write_text('''# 当前未完成事项\n\n1. 自然导航：已有 1 条人工原始指令，尚未完成独立三维身份、可达性和歧义核验，不能作为封存标签。\n2. CARLA：用户要求先做 10 回合并停止；现有 6 个正任务中严格安全到达 1 个，不能扩展解释为完整规模闭环。\n3. 方法与消融：新封存图表已生成，但外部方法适配范围、地图方向/连接和部分参数分组仍需在论文阶段谨慎表述。\n4. 实时性：已有单调用耗时和异步诊断，尚未证明完整感知导航在真实时钟下持续实时。\n5. GitHub：本地分支已提交，服务器没有可写 GitHub 凭据，远程分支尚未推送。\n''', encoding='utf-8')

(R / 'reproducible_commands.md').write_text('''# 复现命令（当前交付）\n\n以下命令在当前环境执行，均指向已有证据，不会读取最终 GT 进入模型推理。\n\n```bash\ncd /root/autodl-tmp/lizichong\nPYTHONPATH=$PWD:/autodl-fs/data/third_party/carla_0.9.16/python_site \\\n  /root/miniconda3/envs/conceptgraph/bin/python -m pytest tests -q\n\n# 查看封存结果\npython -m json.tool outputs/project_v2/thesis_completion_v1/public_experiment_completion_20261115/json/delivery_index_101.json\n\n# 查看三线表和图表\nls outputs/project_v2/thesis_completion_v1/public_experiment_completion_20261115/{figures/sealed_public_078,tables/sealed_office_097}\n```\n\nGitHub 推送（需在已配置 GitHub 凭据的机器执行）：\n\n```bash\ncd /root/autodl-tmp/lzc_sync_101/repo\ngit push origin HEAD:refs/heads/codex/thesis-completion-20261116\n```\n''', encoding='utf-8')
print(json.dumps({'created_indexes': len(indexes), 'goal_complete': False}, ensure_ascii=False))
