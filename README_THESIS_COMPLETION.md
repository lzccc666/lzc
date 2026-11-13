# 三维语义场景图与车辆导航：代码和实验交付

本仓库同步当前项目代码及小型实验依据。整个课题仍为 **goal_complete=false**：真实人工审核和至少20新场景/400条有效指令的新封存测试尚未完成。论文正文、章节及框架由用户后续开展，本轮仅交付实验事实和结果表。

- [当前交接](doc/thesis_completion/工作交接_代码与实验.md)、[实验报告](doc/thesis_completion/experiment_report.md)、[未完成条件](doc/thesis_completion/remaining_issues.md)、[复现命令](doc/thesis_completion/reproducible_commands.md)、[写作证据边界](doc/thesis_completion/GPT_Work_写作交接.md)。
- 390项目源码；主450与新增缺一450真实执行已完整记账，各449完整/1失败。组合750复用300条主记录，不能按1200次独立执行计数。
- 每模型90正任务实际到达：规则50、基础Qwen39、新Adapter4。新Adapter下游无提升，负结果和全部失败保留；四类结果必须区分。
- GPU故障180次完成（179完整、1超时未知），三类独立审计结束；29次实际断流停车及恢复，10次报告到达获保持证书。范围限于冻结仿真工况。
- 固定输入正式性能：12次GPU前端、216次真实Qwen推理、216缓存请求且新增调用0；分阶段统计不代表闭环实时吞吐。
- 新Python环境实际CPU复现：38感知调用、187控制记录、正确到达保持2秒；主450四份输出逐字节一致。依赖复制隔离、CPU执行、CUDA未初始化；共享基础标准库与原生库，不是全新操作系统。
- KITTI静态64帧/33 drive与连续3固定序列/90帧已实际实验；不是官方AP或KITTI导航到达，覆盖限制与负结果见报告。
- 原机452测试通过；交付副本450通过、2项缺未公开真实crop数据明确跳过。业务范围Ruff及diff检查通过；255170受保护历史文件最终SHA核验通过。完整原始数据、模型、原论文稿、失败现场和运行环境保留在指定服务器目录，不上传凭据或大模型。

本次新增证据在 `experiments/completed_increment_029/`（故障、性能、表格）与 `completed_increment_030/`（隔离复现、工程检查、门户）。Word/Excel/PDF结果表在029下的`experiment_tables/fault_performance_office_002/`，更早表格和各阶段证据仍保留。源码辅助程序在`reproduction/`。

网页的完整RGB/深度/日志浏览需要原始数据服务；服务器最新入口为8768端口`/visualizations/thesis_portal_017/index.html`，实时交互8769，独立人工审核8770。Git内不包含完整传感器记录。

本仓库内验收快照冻结于推送之前，发布事实以Git提交记录及服务器同步回执为准。原先的分批进度README归档在`doc/thesis_completion/history_before_030/`，不能把历史“等待中”当成当前状态。
