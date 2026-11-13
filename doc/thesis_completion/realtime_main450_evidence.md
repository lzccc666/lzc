# 三模型450次开发实验核验交接

主矩阵450次已完整结束并独立复核（2026-11-12T19:38:19.906280+00:00）。449个完整结果、1个240秒进程超时，失败保留在全部分母。每模型90次可达正试验，规则为83路线/84曾选对/82曾正确锁定/50实际到达，原始Qwen为66/72/66/39，新Adapter为59/33/32/4。选对和锁定为保存观测中的曾发生，分别重算，不能代替到达。新Adapter未提升，保留负结果；原始Qwen减新Adapter到达率差38.89个百分点，10配置配对95%区间18.89至60.00。

严格初态审计395车身证书、54缺零时刻观测未知、1缺完整结果，原93到达中86获严格车身/方向及2秒保持。另立种子初态重建版本：54例对应68条保存静止前缀，独立PCG64初态交叉复核、41523新增审核区间；复用旧292406区间后，共449完整试验333929区间。359行驶/90静止/1未知，121585运动区间全线段航向锥及243170次GEOS进度复核；93原到达全部通过重建版本车身、方向和保守身份/终点2秒保持，没有新增执行、模型或到达。54初态为重建，不能称已保存零时刻观测；严格旧未知和7个未决保持报告均保留。

当前450缺一语义消融已实际启动，本次快照保存8/450；后续合并复用主矩阵300记录为750行，不是750新增试验。正式GPU故障和独占性能继续等前置完成。真实人工及合格新测试指令分别以API快照为准，新封存仍关闭。实验与交付优先，不写论文正文/框架；最终实验核验后同步GitHub。业务380源码未改，原机424/纯代码422通过2数据skip的结果继续有效；本轮只新增辅助审核及交接脚本，不冒称重新全量测试。v14包不包含本轮450增量，当前新证据另见《主矩阵450_实验核验交接_001.md》。


|方法|有路线|曾选对对象|曾正确锁定|实际到达|
|---|---:|---:|---:|---:|
|规则|83/90|84/90|82/90|50/90|
|原始Qwen|66/90|72/90|66/90|39/90|
|新Adapter|59/90|33/90|32/90|4/90|

分母为各模型全部30个冻结可达正任务×3种子，含失败。全部450还包含每模型30无匹配及30独立不可接近试验，详见逐种类CSV。

## 独立核验证据

- `statistics/realtime_three_model_full_001/report.json`，SHA-256 `84343e10311932d4f5efac759ebfbdd0b9cfd57ea01a196ebc75192f8c6ac32a`。
- `statistics/main_matrix_evidence_check_001/report.json`，SHA-256 `fa12f3273131c9694e04d53a8e6b08601bc31f5e90ead02e1a9fcb1180950574`。
- `vehicle_execution/main_body_evidence_check_001/report.json`，SHA-256 `9d1e159f6879306f31c24788a72c6670c973b6a856c5ffd487ea14b662a7b28d`。
- `vehicle_execution/main_direction_evidence_check_001/report.json`，SHA-256 `ae42a4196c0181971aa6c859b26fe4c953b309e6c1c195e302ea28dd8600fd01`。
- `vehicle_execution/independent_body_main_seed_initial_001/report.json`，SHA-256 `497cdec3e658b5f4925b27b348ee37f96806e68fd196ca1331373b4b80e91f42`。
- `vehicle_execution/main_seed_initial_ledger_001/report.json`，SHA-256 `b916b48a8af0b5dd3d0f82f7113d1aca736897d368c730bfe00f4bc8c091f3b6`。
- `vehicle_execution/execution_direction_main_seed_initial_001/report.json`，SHA-256 `5e31a26d21dbe744dfc520b0828c6fee277fb4b9f08611d88c55270dd1fd29e2`。
- `vehicle_execution/main_seed_direction_evidence_check_001/report.json`，SHA-256 `893b8e5e471f41aa56d7b6573bbb85662a00add36adfd287e088940a004209fc`。
- `vehicle_execution/main_goal_hold_evidence_check_001/report.json`，SHA-256 `3efe0f20c829cb8a6a18e759be69d414683ca859f8f1db120ca06d27adacd456`。
- `vehicle_execution/main_goal_hold_evidence_check_002/report.json`，SHA-256 `6a67ef04850bcf423edb26c3c06c0f5bfa734fd176d66ee9d54d8a117b8661d2`。

## 复现和后续顺序

1. 保持正在运行的450缺一语义试验及其冻结源不变，完成750行合并统计；正式GPU六条件smoke及180故障、独占性能按既有等待器继续。
2. 真实人工自然表达与属性/歧义审核目前为0；30封存场景保持关闭，可靠指令与标签覆盖达标后才能冻结并开启。
3. 完善完整450与后续消融/故障的统一回放和最终可复现材料，做全量测试、完整历史保护/结构化文件/导出核验，再按已授权要求同步GitHub。
4. 本轮辅助审计固定输出目录，不可覆盖重跑。`main_goal_hold_evidence_check_001.py`支持`--body-run`、`--run-id`；重放审核不新增执行或到达。其它固定目录审核需复制为新版本，在独立副本指定唯一run-id并冻结来源。
5. 论文框架和正文留给后续GPT Work；本文件是实验与复现交接。v14包范围固定，本轮新结果在本机以上路径，不能声称v14已含新450原始数据。
