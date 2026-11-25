# 阶段11：路线形状复用与完整矩阵冻结

当前状态：未完成阶段验收。新增planning_reuse从当前实际状态重新积分真实控制，检查当前道路/障碍、整车矩形与连续扫掠，不复用旧安全标志。身份、目标位置/航向及方向进度必须符合冻结门限，失败退回Hybrid。阻塞新障碍及当前状态回归通过。

单任务config00-positive0/seed17：规则43.60s到达，base58.03s到达，new98.78s（包含后台清理）超时。四项结果中路线、选对、稳定正确锁定三者均成立，实际到达仅规则/base成立，全部独立执行审计通过。三者重排应用均0；不是模型增益或路线复用因果性能证明。旧超时/崩溃run保留。

完整真实时钟realtime_three_model_full_001已冻结450次，50任务/10配置/3种子/规则-base-new，同前端、执行器及90s主循环预算。每试验独立进程，240s进程上限，整体24h/20GB，低于6GB停止，失败不自动重试。未完成前不产生最终率或置信区间。源码与权重冻结清单见release.json；当前运行期间只修改不在调用链中的独立实验/论文/审核工具。

输出：statistics/plan_reuse_pilot_001/、interactive_simulation/realtime_three_model_full_001/。goal_complete=false。
