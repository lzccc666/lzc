"""更新阶段9-12验收快照；保留原始要求，未证明项目不计完成。"""
import copy, json
from datetime import datetime, timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1]
old=json.loads((R/'validation_report_100.json').read_text())
d=copy.deepcopy(old)
d['UTC']=datetime.now(timezone.utc).isoformat()
d['artifact_role']='阶段9—12当前增量验收快照101；不替代原始Goal，不降低验收门槛'
d['previous_goal_turn_classification']='progress：历史保护哈希核验完成，交付入口和最新测试回执已补齐'
d['goal_complete']=False
d['new_accuracy_scores_computed']=False
d['actual_new_GPU_calls']=4390
d['actual_new_CARLA_calls']=0
d['human_records_created']=1
d['current_user_overrides']=['论文正文暂缓','CARLA先10回合并保持停止','项目产物预算25GiB，未来传感器大文件允许共享卷','GitHub凭据不写入仓库或远程URL']
by={g['name']:g for g in d['groups']}
by['公平模型对照与真实GPU']['status']='completed_with_limits'
by['公平模型对照与真实GPU']['supported_by_evidence']='五个模型全部完成878条同请求真实GPU推理，共4390次；请求ID、提示哈希、GT隔离和原始输出已核验。'
by['公平模型对照与真实GPU']['remaining_or_not_proven']='新Adapter没有取得正目标选对收益；自然指令仍缺独立标签，通用属性/解析不能由候选排序实验推断。'
by['公平模型对照与真实GPU']['evidence_paths']=['main_experiments/sealed_execution_075/evaluation/report.json','json/sealed_delivery_verification_099/report.json','figures/sealed_public_078/retrieval_all_metrics.csv']
by['独立模块消融']['supported_by_evidence']='新封存图建图条件、语义开关和候选方法均已生成统一CSV/XLSX结果；正负结果和分母保留。'
by['独立模块消融']['evidence_paths']=['figures/sealed_public_078/mapping_metrics.csv','figures/sealed_public_078/retrieval_all_metrics.csv','figures/sealed_public_078/封存公开数据结果表.xlsx']
by['方法比较与来源']['supported_by_evidence']='新封存结果表已统一包含14种候选/模型方法，并明确CLIP、稀疏网格和内部规划适配边界。'
by['方法比较与来源']['evidence_paths']=['figures/sealed_public_078/retrieval_all_metrics.csv','tables/sealed_office_097/封存公开数据实验表.pdf','method_comparison/ab3dmot_statistics_002/report.json']
by['聚类统计与性能']['supported_by_evidence']='新封存五模型逐调用耗时/显存统计、配对差异CSV、三种子汇总和Word/PDF表格已生成并完成视觉核验。'
by['聚类统计与性能']['evidence_paths']=['figures/sealed_public_078/model_execution_statistics.csv','figures/sealed_public_078/adapter_three_seed_summary.csv','tables/sealed_office_097/visual_QA_100.json']
by['浏览、图表与可编辑表']['status']='completed_with_limits'
by['浏览、图表与可编辑表']['supported_by_evidence']='三张图的PNG、SVG、PDF、600dpi TIFF均已实际查看；六页Word/PDF三线表已逐页视觉核验；CSV/XLSX数据源保留。'
by['浏览、图表与可编辑表']['remaining_or_not_proven']='统一浏览器尚未覆盖所有官方实例、排序和车辆轨迹字段；图表仍不能替代实际到达证据。'
by['浏览、图表与可编辑表']['evidence_paths']=['figures/sealed_public_078/automated_qa.json','tables/sealed_office_097/visual_QA_100.json','tables/sealed_office_097/封存公开数据实验表.pdf']
by['测试与格式验收']['status']='completed_with_limits'
by['测试与格式验收']['supported_by_evidence']='当前环境pytest 452项通过、0失败、5警告；Ruff只报告6个历史可选分支静态错误，完整记录已保存。'
by['测试与格式验收']['remaining_or_not_proven']='6个历史可选分支静态错误未改动；尚未宣称所有可选依赖分支均无错误。'
by['测试与格式验收']['evidence_paths']=['json/software_validation_092/report.json','json/software_validation_092/static_checks.json']
by['历史保护与存储']['status']='completed_with_limits'
by['历史保护与存储']['supported_by_evidence']='最新只读核验复查255170个受保护文件，0个哈希或大小变化；项目新增交付索引和共享卷入口均有记录。'
by['历史保护与存储']['remaining_or_not_proven']='历史保护已通过；未来新增传感器仍只保留共享卷索引和哈希，未复制进Git交付。'
by['历史保护与存储']['evidence_paths']=['json/protected_history_audit_101/report.json','source_bundle/github_sync_101/report.json']
by['复现与代码同步']['supported_by_evidence']='当前代码和交付材料已在本地同步仓库提交9bfbddc；代码主体测试452项通过；交付材料含源代码索引、结果、表格和审计文件。'
by['复现与代码同步']['remaining_or_not_proven']='GitHub远程写入仍因服务器无凭据失败；远程分支尚未推送。独立复现仍复用现有依赖和输入，不代表全新环境安装。'
by['复现与代码同步']['evidence_paths']=['source_bundle/github_sync_101/report.json','source_bundle/github_sync_101/本地同步交接.md','json/delivery_index_101.json']
d['groups']=list(by.values())
d['verification']['latest_tests']={'pytest_passed':452,'pytest_failed':0,'pytest_warnings':5,'ruff_known_errors':6}
d['verification']['protected_history_latest']={'files_checked':255170,'mismatches':0}
d['verification']['natural_original_records']=1
d['verification']['natural_qualifying_labels']=0
d['verification']['git_local_commit']='9bfbddcb81430c0bd7f75303035cd3a0c2456212'
d['verification']['delivery_indexes']=8
(R/'json/validation_report_101.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'groups':len(d['groups']),'goal_complete':False,'pytest':452,'protected_files':255170},ensure_ascii=False))
