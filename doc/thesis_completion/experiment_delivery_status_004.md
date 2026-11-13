# 当前实验完成度与交付清单 v4

快照2026-11-12T17:32:27.474427+00:00；优先实验与复现交付，暂不写论文正文/框架。Goal与阶段9—12仍未整体验收。

|阶段|工作项|状态|
|---|---|---|
|9|任务与独立终点评价|部分完成|
|9|因果建图与关系|开发实验完成，封存未完成|
|9|全序列身份与返回|新增开发诊断已独立核验|
|10|独立监督与三种子Adapter|训练完成，下游验收未完成|
|10|语义消融|排队待实际执行|
|11|车辆规划与实际控制|最低到达功能通过，整体验收未完成|
|11|独立控制重算与连续车身|150规则+6CPU故障已审核，完整主450等待|
|11|独立连续方向与未检验连接范围|156保存试验实际完成，主450审核等待|
|11|真实时钟主矩阵|运行中|
|12|KITTI真实车载实验|对象定位开发诊断完成|
|12|感知故障|CPU试跑完成，GPU正式待执行|
|12|分阶段性能|CPU试跑完成，GPU正式待执行|
|12|新封存测试|尚未开启|
|12|自然语料覆盖与真实表达采集|覆盖审计和采集工具完成，有效人工数据待提交审核|
|12|人工审核|待真实提交验收|
|12|工程与复现|阶段性通过，最终验收未完成|
|12|论文正文|按用户最新要求暂缓|

## 任务与独立终点评价

已实现：已实现对象身份、安全区域、航向/速度/保持时间与碰撞/车道联合判定；路线、对象、锁定、到达分开。

待完成与限制：新封存合法目标/可达性与关键标签仍需完整验收。

证据：`task_protocol/protocol_v1.json`、`statistics/interactive_rule_full_002/report.json`。

## 因果建图与关系

已实现：38场景11log1508帧、1470过去LiDAR/38缺测；七方法5723GT观测全保留；关系、动静态、门限、动态/TTL、录制故障及地图消融已完成。

待完成与限制：官方GT晚33–42ms的近似仍须保留；可靠人工错合并/歧义审核和新封存比较待完成。

证据：`statistics/causal_mapping_full_001/report.json`、`mapping_evaluation/causal_relations_full_001/summary.json`、`statistics/causal_modules_full_001/report.json`。

## 全序列身份与返回

已实现：532身份条件、2028返回事件、1628尾段、154逐log行、208区间均核对；深度IDF1改善，但返回保持与相邻变化差区间跨零。

待完成与限制：自动中心关联不是人工实例/遮挡真值；IDF1受检测覆盖影响，不宣称全部身份能力提高。

证据：`statistics/temporal_identity_full_001/report.md`、`statistics/temporal_identity_full_001/audit_002/report.json`。

## 独立监督与三种子Adapter

已实现：v2 2398训练/603验证；三种子各240步；base/old/new三种子共五模型各603相同输入真实CUDA公平验证；三任务冻结请求0新增调用缓存复跑。

待完成与限制：可靠颜色/细属性依赖真实人工；真实验证仅2log、排序含6符号bank；不能把训练/结构化得分直接当道路检索或到达收益。

证据：`semantic_dataset/structured_supervision_002/manifest.json`、`statistics/fair_three_task_full_validation_001/report.json`。

## 语义消融

已实现：完整/关闭Qwen与缺解析/缺属性/缺重排协议已冻结；450新试验加300共享主条件，预期750条件。

待完成与限制：主矩阵完整结束后才执行450新试验并完整统计；共享300不增加独立试验。

证据：`source_bundle/semantic_ablation_after_main_001.py`。

## 车辆规划与实际控制

已实现：50任务×3种子×2规划器300上限执行已完成；规则全感知150次：90可达正试验68路线/90选对/79正确锁定/21到达，覆盖13正任务8配置。

待完成与限制：完整异步三模型对比、全部失败/拒绝统计、不可接近独立参考和新封存连续回放仍需验收；给定正确对象上限不计实际语义成绩。

证据：`statistics/kinematic_oracle_002/report.json`、`statistics/interactive_rule_full_002/report.json`。

## 独立控制重算与连续车身

已实现：全部33076区间获冻结静态几何数值证书；规则21与CPU故障6原到达均覆盖，不新增执行/到达。新实现不调用原积分/车身/扫掠/setup；940输入/产物SHA、468停车矩形/30屏障、全部任务种子与逐区间账目独立通过。

待完成与限制：不是第三套连续证书算法或形式化区间算术；不覆盖动态交通、语言身份与有向拓扑。主450等待完整summary后才自动CPU审核，当前不能以排队当完成。

证据：`task_protocol/independent_continuous_body_001.md`、`vehicle_execution/independent_body_rule_full_001/report.json`、`vehicle_execution/independent_body_fault_cpu_full_001/report.json`、`vehicle_execution/independent_body_evidence_check_001/report.md`、`vehicle_execution/independent_body_after_main_001/release.json`。

## 独立连续方向与未检验连接范围

已实现：150规则+6CPU故障全部33076区间审核：126行驶试验前向，30未行驶不计方向安全。27314运动区间由另一种全线段航向锥独立确认、54628次GEOS进度核对，788输入/产物SHA。原21+6到达同时有车身/方向证据，没有新增模型、执行或到达。

待完成与限制：主450方向审核已实际排队，4源冻结，等主矩阵与独立车身完整；运行后仍须独立账目/全局方向界复核。十配置为单向单车道，无多车道/路口连接过渡，连接准确率null，不提供实际连接性能；固定积分模型不代表真实动态交通安全。

证据：`task_protocol/execution_direction_audit_001.md`、`vehicle_execution/execution_direction_rule_full_001/report.json`、`vehicle_execution/execution_direction_fault_cpu_full_001/report.json`、`vehicle_execution/execution_direction_evidence_check_002/report.md`、`vehicle_execution/execution_direction_evidence_check_002/report.json`、`vehicle_execution/execution_direction_after_body_001/release.json`。

## 真实时钟主矩阵

已实现：冻结450次，快照已保存385条；统计、主矩阵独立车身及方向审核仍需完整结果，不把排队当完成。

待完成与限制：完整结束后统计对象/锁定/路线/到达、配置聚类区间、排队迟到过期及三模型公平性；进行中数字不作最终优劣结论。

证据：`interactive_simulation/realtime_three_model_full_001/release.json`、`interactive_simulation/realtime_three_model_full_001/trial_records.jsonl`。

## KITTI真实车载实验

已实现：固定64帧/33drive真实CPU前端；319GT下247→262匹配，召回77.43%→82.13%，drive配对区间支持提升；卡车误差退化保留。

待完成与限制：不称官方AP、完整跨帧图、KITTI导航到达或独占GPU性能；预训练重叠未知。

证据：`statistics/kitti_object_cpu_full_001/report.json`。

## 感知故障

已实现：六CPU条件63次真实前端/2270状态，单任务单配置；4秒断流但15秒TTL未过期停车的负结果保留。另有已知路径0.3秒TTL控制功能63状态。

待完成与限制：等待主450和语义450结束后六GPU试跑再180正式；CPU重复不增加独立覆盖，不能替代GPU1.5秒TTL验收。

证据：`faults/realtime_fault_cpu_smoke_002/audit_001/report.json`、`faults/realtime_fault_after_semantics_001/release.json`。

## 分阶段性能

已实现：12实际CPU前端，10录制观测/391状态重算；独立统计通过。三模型各72不同实际cache键截获预检，无模型/响应。

待完成与限制：全部前置实际进程退出且GPU空闲后才测12GPU前端、216无缓存语义+216缓存；当前预检不计推理/准确率。

证据：`performance/runtime_profile_cpu_smoke_001/audit_001/report.json`、`performance/runtime_profile_after_faults_001/release.json`。

## 新封存测试

已实现：30新场景14新log1196帧已分离；1166过去扫描/30缺测已审计；600构造主查询+278挑战及203预选回放准备。

待完成与限制：30场景14log1196帧保持封存；每场景20条有效自然指令、真实标签与独立参考尚未齐备。现有600构造+278挑战和439原文均不等于合格自然分布。

证据：`splits/split_audit_002/report.json`、`data_manifest/reserved_causal_inputs_full_001/audit_001/report.json`。

## 自然语料覆盖与真实表达采集

已实现：固定146候选2308原文全部核对官方属性；50m下66场景1097原文/364官方静态属性，15场景6log有20原文、0场景有20官方静态属性。原30场景选场不改，不能仅换选场满足要求。独立2308命令映射与1196原图SHA/尺寸/来源通过。

待完成与限制：自然表达与有效标签不足是外部依赖；不使用模板、机器改写或降低距离规则补足。30场景需各20条有效指令，即至少600；最低20场景400条口径不用于替代逐场景要求。

证据：`task_protocol/human_natural_collection_001.md`、`data_manifest/natural_candidate_coverage_001/report.md`、`data_manifest/natural_candidate_coverage_001/report.json`、`data_manifest/natural_data_preparation_audit_002/report.json`。

## 人工审核

已实现：既有对象核验及439原文审核入口保留；新增30场景1196真实帧自由表达采集页。原8768端口服务已兼容升级，桌面/手机12组正式帧检查通过；不产生测试假人工记录。

待完成与限制：真实人工审核与自由表达提交仍0；提交后仅原始待审核，身份、运动、颜色、歧义与可达性仍unknown。现有表单不能替代全部漏检和跨帧长链审核。

证据：`visualizations/causal_development_review_002/review_manifest.json`、`annotation/human_natural_collection_001/manifest.json`、`visualizations/natural_collection_browser_canonical_001/report.json`、`source_bundle/natural_collection_service_upgrade_001/report.json`。

## 工程与复现

已实现：原机401测试通过，新增方向14回归；范围Ruff/diff通过。当前374源码准备，最后已公开365；根项目索引与七组冻结源码不变。

待完成与限制：运行中统计不能代替最终JSON/配置/历史保护/浏览器/完整复现实验验收；所有后续结果并入最终源码和交付包。

证据：`source_bundle/tests_full_035.log`、`source_bundle/tests_full_036.log`、`source_bundle/ruff_completion_050.log`、`source_bundle/git_diff_check_036.log`、`source_bundle/async_no_match_test_before_001/manifest.json`、`baseline_verification.json`、`source_bundle/tests_full_037.log`、`source_bundle/ruff_051.log`、`source_bundle/diff_check_037.log`、`source_bundle/tests_natural_collection_002.log`、`source_bundle/tests_natural_collection_003.log`、`source_bundle/tests_full_038.log`、`source_bundle/ruff_052.log`、`source_bundle/diff_check_038.log`、`source_bundle/tests_execution_direction_001.log`。

## 论文正文

已实现：本阶段只整理实验状态、证据、复现命令和交付材料；旧阶段稿保留。

待完成与限制：用户后续通过GPT Work拟定整体框架并正式撰写；当前不新增论文文字章节，不标记论文或Goal完成。

证据：`GPT_Work_写作交接.md`。

## 后续顺序

1. 当前450主矩阵完整及独立统计。
2. 缺一语义450新试验/750条件及统计。
3. GPU故障6试跑通过后180全量。
4. 全部前置进程实际退出后GPU分阶段性能。
5. 人工/有效指令/独立参考齐备后冻结并开启新封存。
6. 完整统计、可视化、工程验收与最终Git/复现交付。

主矩阵完整后依次进行独立车身与方向CPU审核；它们不会自动补齐独立输入/账目复核。GPU语义、故障和性能按原冻结队列运行，不重复启动。自然指令和人工标签仍需真实提交审核，未达标时不解封新测试；最终核验后再统一同步GitHub。
