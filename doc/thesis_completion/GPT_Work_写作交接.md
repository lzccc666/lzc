# 后续写作的证据交接

本文件提供事实和证据边界，不起草论文正文、目录或框架。

请先阅读[代码与实验交接](工作交接_代码与实验.md)、[验收状态](validation_report.json)及[实验报告](experiment_report.md)。所有结论从冻结结果和逐任务表取数；禁止把生成路径、选对对象、正确锁定和实际到达混用。新Adapter下游负结果、失败分母、未知标签、显式重建初态、重复记录与共享道路家族均需保留。

Word/Excel/PDF表格分别在`experiment_tables/development_evidence_003/`（15表）、`semantic_ablation_office_001/`（3表）和`fault_performance_office_002/`（7表）。表号为开发交付编号；此处不决定论文最终章节编号。CSV、来源哈希与逐条件记录随表保存。

不能写成已完成：真实人工审核、合格自然指令覆盖、至少20新场景/400指令的新封存主表。合成开发任务、KITTI有限子集、官方GT时间近似与实例关联代理必须注明。规则/基础Qwen达到最低仿真功能数量，不代表所有方法或道路工况有效。独立新venv已实际复现，但共享基础标准库及原生系统库，不称全新操作系统。

原论文稿、MEMORY.md、PAPER_MEMORY.md和阶段1—8均保持原样。本轮不输出新正文或框架，后续由用户开展写作。整个课题仍为goal_complete=false。
