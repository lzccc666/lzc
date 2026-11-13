# 真实屏障与迟到排序CPU开发实验交接

四次真实执行全部保留。第一版白色目标两次未形成故障前提，不能计通过；第二版红色目标规则与迟到排序两个条件暴露审核通过。红色条件合计48次真实感知前端、4931个控制区间、1次CPU基础Qwen调用。两者均为生成路线、选对对象、正确锁定为真，实际到达为假。

实际车辆先移动，再在观测间断时由原TTL刹停。停车后环境加入真实渲染屏障，代理通过实际深度/LiDAR估计障碍，清除受阻路线。迟到排序使用真实模型输出；该输出在新障碍发布后返回，因为候选/几何安全上下文改变被原处理逻辑拒绝。独立连续矩形车体和行驶方向审核通过，最终速度为零，没有新增到达信用。

规则条件的后轴中心线不穿屏障，但旧整车路径相撞，而且所有可接受终点姿态的车体均相撞；迟到排序条件的全部终点区域与起点分量被屏障隔断。独立解析证明和交叉采样见两个 geometry_witness_002/report.json，不能只用本规划器失败定义不可达。

限制：一个配置、固定种子的后验开发补充；先TTL停车再出现屏障，不代表运动中突发障碍安全。CPU感知TTL15秒、固定240秒观察窗，不混入正式GPU的1.5秒TTL实验、主450到达率、独占性能或新封存测试。

原审核NumPy计数导致的部分JSON、后轴点与整车判据修正、输入提示原文审核失败、手机菜单溢出均保留。最终输入审核003按原异步调度排序冻结JSON恢复提示键顺序，核对精确提示、候选安全证据与真实模型原文，未改原请求或重跑模型。最终查看器002采用短中文菜单名和宽度约束，桌面/手机共8个案例视口检查通过，114个请求全为GET，GT默认隐藏。

## 入口

- 实际完成汇总：faults/perceived_blockage_cpu_002/recovery_001/complete.json
- 两个原始独立审核：faults/perceived_blockage_cpu_002/{rule_blockage,delayed_actual_rank}/independent_audit_003/report.json
- 模型输入审核：faults/perceived_blockage_cpu_002/delayed_actual_rank/input_audit_003/report.json
- 完整回放：visualizations/perceived_blockage_cpu_002/index.html；8768只读服务下打开此路径。
- 浏览器验收：visualizations/perceived_blockage_browser_qa_002/report.json
- 结构化文件与历史失败清单：source_bundle/perceived_blockage_structured_validation_001.json
- 保护核验：source_bundle/perceived_blockage_protection_001.json

原worker、runner、release、观察图像和控制记录均随本增量包保留。按原协议可在新版本目录重新执行，不能覆盖原路径、把重跑当独立样本，或把CPU数据算成GPU故障验收。
