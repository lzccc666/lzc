# 实验完成度与交付清单（第12版）

当前仅完善实验、代码和复现交付，暂缓论文文字；Goal与四阶段仍未完成。

# 当前实验进度与KITTI连续诊断交接

更新时间：2026-11-12T22:17:59.322001+00:00。当前优先实验、业务代码、核验和复现交付；暂缓论文正文、章节及框架。Goal与阶段9—12仍未全部验收，新封存未开启。

已验证的开发工作：阶段9任务协议、因果建图、对象/关系/跨帧身份指标及独立审核；阶段10三种子新Adapter真实训练与公平原始Qwen/旧Adapter验证；阶段11实际感知—规划—车辆控制、完整主450对照与独立车身/方向/保持时间审核，以及浏览器可操作的CPU感知仿真；阶段12开发主表、配对聚类统计、失败证据查看、源码隔离导出和复现入口。它们不能代替尚未合格的新封存自然指令测试。

主450为50任务×3种子×3方法，449完整结果、1超时留在分母。每方法90正任务，规则/原始Qwen/新Adapter的生成路线分别83/66/59，曾选对84/72/33，正确稳定锁定82/66/32，实际正确到达50/39/4。规则到达覆盖27正任务/10配置，满足最低功能证据；新Adapter下游负结果保留。93原始到达已有分版本独立核验，其中54试验初态为显式种子重建，严格保存初态的口径仍单列，不伪称新观测或新增到达。

KITTI Object已有64帧/33 drive诊断，稳健深度匹配247/319→262/319；truck误差和Cyclist代理失败保留。新增KITTI Tracking固定0000/0005/0013各中心30帧，共90帧，先前端后隔离评价。首次001因官方标定分隔格式失败完整保留，002修正格式并检查原选样不变。下载279,493,425字节，未下载15/35GB完整大包；独立完整P2、OXTS及标签转换核验通过，1,760,358有效深度像素、539原始标签记录。1帧与Object64重合，不额外算独立样本。

|条件|同类XY≤3m匹配/GT|召回|匹配中心误差(m)|IDF1|
|---|---:|---:|---:|---:|
|单帧|4/74|5.41%|1.908|0.0172|
|原始融合|5/74|6.76%|1.784|0.0254|
|改进融合|63/74|85.14%|1.321|0.3569|

同一实际CPU前端90不同帧（smoke3＋新增87，非93次不同帧），统一名义0.1秒；单帧、原始融合、改进融合输入相同。独立NetworkX最大基数匹配、固定TrackEval身份计数、返回事件及4000次种子17序列配对bootstrap均通过（浮点独立区间绝对容差1e-12；首次逐位比较失败保留，未改模型或结果）。差异、置信区间、所有分母及未改善条件见statistics/kitti_tracking_full_001/report.json。全部74次主类别GT只在0005；其他两窗口主类别GT为0，仍保留原选样。有效正GT重采样仅2845/4000，召回/匹配误差差值区间退化，不作为跨场景显著性证据，见scope_audit_001/report.json。只有三个短窗口；序列不是已证实独立采集log，名义10Hz不是原始实测同步；中心门限不是人工身份标注或官方KITTI HOTA/AP。未匹配预测可能受DontCare影响，不全称误检。预训练与KITTI重合未知。到达/GPU性能/人工标签信用均0。

浏览器：visualizations/kitti_tracking_preview_001/index.html，提供90帧RGB、实际检测/SAM、稀疏深度、三维点云、条件切换及过去帧节点轨迹，默认隐藏官方GT。数据及指标独立审核见数据independent_audit_001及指标independent_audit_002/report.json；桌面/手机只读QA见visualizations/kitti_tracking_browser_qa_001/report.json。

必须继续：缺一语义450当前101/450快照，结束后保留原源集合断言失败，并以兼容003独立核验完整750组合；其后原队列继续真实GPU故障与独占性能。真实人工记录及合格新自然指令仍以8768 API为准，不能用自动证据代替；至少20场景×20有效指令、独立场景/log和完整新封存主对照仍是验收要求。完成这些后再做最终全量保护、最终复现包、代码Git同步及整体工作交接；当前不能宣布课题完成。

当前390业务源码已隔离准备。旧公开Git365源码commit f75725f8fd422ec619775bfa0407dc6733312488；当前版本未推送。旧v14交接包不包含主450、交互及本次连续KITTI增量，保持原包不覆盖。新的实验增量包是补充材料，不冒称整个Goal最终交付。


## 任务与独立终点评价

部分完成。已实现对象身份、安全区域、航向/速度/保持时间与碰撞/车道联合判定；路线、对象、锁定、到达分开。

待完成与限制：新封存合法目标/可达性与关键标签仍需完整验收。

证据：`task_protocol/protocol_v1.json`、`statistics/interactive_rule_full_002/report.json`。

## 因果建图与关系

开发实验完成，封存未完成。38场景11log1508帧、1470过去LiDAR/38缺测；七方法5723GT观测全保留；关系、动静态、门限、动态/TTL、录制故障及地图消融已完成。

待完成与限制：官方GT晚33–42ms的近似仍须保留；可靠人工错合并/歧义审核和新封存比较待完成。

证据：`statistics/causal_mapping_full_001/report.json`、`mapping_evaluation/causal_relations_full_001/summary.json`、`statistics/causal_modules_full_001/report.json`。

## 全序列身份与返回

新增开发诊断已独立核验。532身份条件、2028返回事件、1628尾段、154逐log行、208区间均核对；深度IDF1改善，但返回保持与相邻变化差区间跨零。

待完成与限制：自动中心关联不是人工实例/遮挡真值；IDF1受检测覆盖影响，不宣称全部身份能力提高。

证据：`statistics/temporal_identity_full_001/report.md`、`statistics/temporal_identity_full_001/audit_002/report.json`。

## 独立监督与三种子Adapter

训练与主450对照完成，下游负结果保留。3种子新Adapter已真实CUDA训练；主矩阵新模型4/90到达，base39/90、规则50/90。 主450负结果新增逐响应诊断：正任务74已投递解析均引入同名near锚点，52丢颜色；原模型90投递均保留原任务语义。

待完成与限制：不能将验证结构化指标提升代替导航提升；人工属性与新封存评价仍待完成。

证据：`semantic_dataset/structured_supervision_002/manifest.json`、`statistics/fair_three_task_full_validation_001/report.json`、`statistics/realtime_three_model_full_001/report.json`、`statistics/main_matrix_evidence_check_001/report.json`、`vehicle_execution/main_body_evidence_check_001/report.json`、`vehicle_execution/main_direction_evidence_check_001/report.json`、`vehicle_execution/independent_body_main_seed_initial_001/report.json`、`vehicle_execution/main_seed_initial_ledger_001/report.json`、`vehicle_execution/execution_direction_main_seed_initial_001/report.json`、`vehicle_execution/main_seed_direction_evidence_check_001/report.json`、`vehicle_execution/main_goal_hold_evidence_check_001/report.json`、`vehicle_execution/main_goal_hold_evidence_check_002/report.json`、`task_protocol/main_failure_chain_002.md`、`statistics/main_failure_chain_full_002/report.json`、`statistics/main_failure_chain_full_002/trials.csv`、`statistics/main_failure_chain_coverage_001/report.json`、`statistics/main_failure_chain_audit_001/report.json`。

## 语义消融

450缺一正在执行，源集合兼容审核已预检并排队。450缺一实际执行中，当前只读快照101/450。兼容003等待器保持原队列，完成桥接仍未写；后验诊断与交互会话不替代消融。

待完成与限制：必须等450实际结束、保留原源集合断言失败、完整750组合及独立复算后才计消融完成；不更改原release或重启模型。

证据：`source_bundle/semantic_ablation_after_main_001.py`、`statistics/realtime_three_model_full_001/report.json`、`statistics/main_matrix_evidence_check_001/report.json`、`vehicle_execution/main_body_evidence_check_001/report.json`、`vehicle_execution/main_direction_evidence_check_001/report.json`、`vehicle_execution/independent_body_main_seed_initial_001/report.json`、`vehicle_execution/main_seed_initial_ledger_001/report.json`、`vehicle_execution/execution_direction_main_seed_initial_001/report.json`、`vehicle_execution/main_seed_direction_evidence_check_001/report.json`、`vehicle_execution/main_goal_hold_evidence_check_001/report.json`、`vehicle_execution/main_goal_hold_evidence_check_002/report.json`、`task_protocol/semantic_source_compatibility_003.md`、`ablations/semantic_navigation_chain_003/preflight.json`、`ablations/semantic_navigation_chain_003/release.json`、`ablations/semantic_navigation_chain_003/worker_started.json`、`statistics/semantic_compatibility_audit_rehearsal_main_001/report.json`。

## 车辆规划与实际控制

最低正确到达功能通过，主450开发对照已完成。规则50到达覆盖27不同正任务10配置，base39覆盖17任务9配置，新模型4覆盖4任务3配置；93均有分版本车身、方向与保守2秒保持核验。旧逻辑规则21、地图39参考位姿分别保留，不混计。

待完成与限制：共享3走廊家族的合成开发场景；正式故障、新封存与最终整体验收仍未完成。

证据：`statistics/kinematic_oracle_002/report.json`、`statistics/interactive_rule_full_002/report.json`、`statistics/realtime_three_model_full_001/report.json`、`statistics/main_matrix_evidence_check_001/report.json`、`vehicle_execution/main_body_evidence_check_001/report.json`、`vehicle_execution/main_direction_evidence_check_001/report.json`、`vehicle_execution/independent_body_main_seed_initial_001/report.json`、`vehicle_execution/main_seed_initial_ledger_001/report.json`、`vehicle_execution/execution_direction_main_seed_initial_001/report.json`、`vehicle_execution/main_seed_direction_evidence_check_001/report.json`、`vehicle_execution/main_goal_hold_evidence_check_001/report.json`、`vehicle_execution/main_goal_hold_evidence_check_002/report.json`。

## 独立控制重算与连续车身

主450严格及种子重建审核完成。严格395已保存零时刻证书/54缺初态未知/1缺完整结果保留；显式种子重建54例后449完整试验333929区间，独立初态及账目复核通过，93原到达均覆盖。重建不称新增观测，无新增到达。

待完成与限制：保留严格54缺初态/1缺完整结果未知；新版本仅修复证据缺口，不新增执行/到达。

证据：`task_protocol/independent_continuous_body_001.md`、`vehicle_execution/independent_body_rule_full_001/report.json`、`vehicle_execution/independent_body_fault_cpu_full_001/report.json`、`vehicle_execution/independent_body_evidence_check_001/report.md`、`vehicle_execution/independent_body_after_main_001/release.json`、`statistics/realtime_three_model_full_001/report.json`、`statistics/main_matrix_evidence_check_001/report.json`、`vehicle_execution/main_body_evidence_check_001/report.json`、`vehicle_execution/main_direction_evidence_check_001/report.json`、`vehicle_execution/independent_body_main_seed_initial_001/report.json`、`vehicle_execution/main_seed_initial_ledger_001/report.json`、`vehicle_execution/execution_direction_main_seed_initial_001/report.json`、`vehicle_execution/main_seed_direction_evidence_check_001/report.json`、`vehicle_execution/main_goal_hold_evidence_check_001/report.json`、`vehicle_execution/main_goal_hold_evidence_check_002/report.json`。

## 独立连续方向与未检验连接范围

主450两版本及全线段锥核验完成。严格315行驶/80静止/55未知保留；种子重建版本359行驶/90静止/1缺完整结果，121585运动区间和243170次GEOS进度独立通过，93原到达覆盖。单向单车道无连接指标，不外推动态交通。

待完成与限制：十配置无多车道连接过渡，连接指标null。

证据：`task_protocol/execution_direction_audit_001.md`、`vehicle_execution/execution_direction_rule_full_001/report.json`、`vehicle_execution/execution_direction_fault_cpu_full_001/report.json`、`vehicle_execution/execution_direction_evidence_check_002/report.md`、`vehicle_execution/execution_direction_evidence_check_002/report.json`、`vehicle_execution/execution_direction_after_body_001/release.json`、`statistics/realtime_three_model_full_001/report.json`、`statistics/main_matrix_evidence_check_001/report.json`、`vehicle_execution/main_body_evidence_check_001/report.json`、`vehicle_execution/main_direction_evidence_check_001/report.json`、`vehicle_execution/independent_body_main_seed_initial_001/report.json`、`vehicle_execution/main_seed_initial_ledger_001/report.json`、`vehicle_execution/execution_direction_main_seed_initial_001/report.json`、`vehicle_execution/main_seed_direction_evidence_check_001/report.json`、`vehicle_execution/main_goal_hold_evidence_check_001/report.json`、`vehicle_execution/main_goal_hold_evidence_check_002/report.json`。

## 真实地图连接与车辆执行诊断

228条件完成并独立核验，非语义到达。38开发场景/11log、3种子、两规划器共228条件全保留；点A*18路线/18实际运动/9参考位姿到达（3场景），Hybrid固定3秒2000扩展预算下0路线。6351连续后轴区间、原独立车身输入与证书、官方连接、方向及2秒终点保持通过复核；18轨迹有36次完整见证的声明连接。桌面/手机10类条件检查通过。

待完成与限制：人工参考后轴起点、给定地图路线、冻结静态预测障碍，没有语言选对象或在线感知，语义到达信用为0；两算法搜索预算不同，不据此断言一般优劣。Hybrid减点A*参考位姿到达率差为−7.89个百分点，11log配对95%区间−18.18至0。21个非ready选场及不合法连接条件保留，不放宽道路/车身边界。 原60条UNKNOWN_INVALID_DECLARED_CHAIN实际均因连续重复采样触发严格方向入口，不是官方端点不连；原结果保留，显式规范化另立版本。

证据：`task_protocol/HDmap_vehicle_connection_001.md`、`task_protocol/HDmap_vehicle_connection_smoke_addendum_001.md`、`task_protocol/HDmap_vehicle_connection_control_smoke_001.md`、`vehicle_execution/HDmap_connection_tasks_001/manifest.json`、`vehicle_execution/HDmap_connection_full_001/summary.json`、`vehicle_execution/HDmap_connection_full_001/source_snapshot_receipt.json`、`vehicle_execution/HDmap_connection_independent_audit_001/report.json`、`source_bundle/HDmap_counter_schema_before_001/verification.json`、`visualizations/HDmap_connection_browser_qa_001/report.json`。

## 同预算中心线引导开发版本

228条件及规划/实际控制核验完成。原Hybrid与新引导各114条件，均3秒/2000扩展、交替顺序。原0路线，新18路线/18运动/18参考位姿到达。6527实际控制区间、36声明连接、5951计划区间及0.05米车身余量独立通过；按11log配对差+15.79个百分点，95%区间+5.56至+30.77。

待完成与限制：查看旧失败后的开发版本，同时改变中心线控制建议与逐步拓扑搜索，不能单独归因于一个组件；扩展工作单位不同、共享CPU不代表独占性能。地图给定人工后轴起点、静态预测障碍，语义到达信用0；新版本仅接入地图诊断，未替换冻结的完整感知导航版本。

证据：`task_protocol/reference_guided_vehicle_001.md`、`vehicle_execution/HDmap_reference_guided_full_001/summary.json`、`vehicle_execution/HDmap_reference_guided_audit_001/report.json`、`vehicle_execution/HDmap_reference_guided_plan_audit_001/report.json`、`visualizations/HDmap_reference_guided_browser_v1_fixed_001/report.json`。

## 无损重复采样规范化与完整配对

第二轮228条件及独立影响核验完成。只移除相邻完全相同XY，保留全部有向非零边、曲线、长度、道路和官方连接；不删非零短边。两方法共同规范化后，新39/114路线、实际运动及参考位姿到达，原Hybrid0/114；覆盖13场景6log12不同车道链。18743实际控制区间/78连接、17495计划区间与0.05米余量核验通过，按11log配对差+34.21个百分点（95%+15.79至+52.17）。原60严格退化入口条件中，新方法21转为参考位姿到达、9无路线，原Hybrid30无路线。其余168条件业务相同，42组保存状态和路径完全相同。

待完成与限制：同场景/路线复用已披露，三种子与版本重跑不增加独立样本。仍有全部失败/未知；没有规划器独立可达性真值。全部39为给定地图的参考位姿，语义正确到达0；新封存与完整导航组合验收待做。旧页面候选区间显示0的字段兼容错误已修，旧QA未覆盖此字段，当前v1_fixed/003逐项核对并保留旧记录。

证据：`task_protocol/reference_guided_canonical_002.md`、`vehicle_execution/HDmap_reference_guided_full_002/summary.json`、`vehicle_execution/HDmap_reference_guided_audit_002/report.json`、`vehicle_execution/HDmap_reference_guided_plan_audit_002/report.json`、`vehicle_execution/HDmap_reference_guided_canonical_impact_001/report.json`、`visualizations/HDmap_reference_guided_browser_qa_003/report.json`、`visualizations/HDmap_reference_guided_viewer_schema_erratum_001.json`。

## 真实时钟主矩阵

450开发条件结束并独立核验。主矩阵450次已完整结束并独立复核（2026-11-12T19:38:19.906280+00:00）。449个完整结果、1个240秒进程超时，失败保留在全部分母。每模型90次可达正试验，规则为83路线/84曾选对/82曾正确锁定/50实际到达，原始Qwen为66/72/66/39，新Adapter为59/33/32/4。选对和锁定为保存观测中的曾发生，分别重算，不能代替到达。新Adapter未提升，保留负结果；原始Qwen减新Adapter到达率差38.89个百分点，10配置配对95%区间18.89至60.00。

待完成与限制：新Adapter负结果保留，独立消融仍在跑；当前开发合成任务不是封存自然语言验收。

证据：`interactive_simulation/realtime_three_model_full_001/release.json`、`interactive_simulation/realtime_three_model_full_001/trial_records.jsonl`、`statistics/realtime_three_model_full_001/report.json`、`statistics/main_matrix_evidence_check_001/report.json`、`vehicle_execution/main_body_evidence_check_001/report.json`、`vehicle_execution/main_direction_evidence_check_001/report.json`、`vehicle_execution/independent_body_main_seed_initial_001/report.json`、`vehicle_execution/main_seed_initial_ledger_001/report.json`、`vehicle_execution/execution_direction_main_seed_initial_001/report.json`、`vehicle_execution/main_seed_direction_evidence_check_001/report.json`、`vehicle_execution/main_goal_hold_evidence_check_001/report.json`、`vehicle_execution/main_goal_hold_evidence_check_002/report.json`。

## KITTI真实车载实验

对象定位开发诊断完成。固定64帧/33drive真实CPU前端；319GT下247→262匹配，召回77.43%→82.13%，drive配对区间支持提升；卡车误差退化保留。

待完成与限制：不称官方AP、完整跨帧图、KITTI导航到达或独占GPU性能；预训练重叠未知。

证据：`statistics/kitti_object_cpu_full_001/report.json`。

## 感知故障

CPU试跑完成，GPU正式待执行。六CPU条件63次真实前端/2270状态，单任务单配置；4秒断流但15秒TTL未过期停车的负结果保留。另有已知路径0.3秒TTL控制功能63状态。

待完成与限制：等待主450和语义450结束后六GPU试跑再180正式；CPU重复不增加独立覆盖，不能替代GPU1.5秒TTL验收。

证据：`faults/realtime_fault_cpu_smoke_002/audit_001/report.json`、`faults/realtime_fault_after_semantics_001/release.json`。

## 分阶段性能

CPU试跑完成，GPU正式待执行。12实际CPU前端，10录制观测/391状态重算；独立统计通过。三模型各72不同实际cache键截获预检，无模型/响应。

待完成与限制：全部前置实际进程退出且GPU空闲后才测12GPU前端、216无缓存语义+216缓存；当前预检不计推理/准确率。

证据：`performance/runtime_profile_cpu_smoke_001/audit_001/report.json`、`performance/runtime_profile_after_faults_001/release.json`。

## 新封存测试

尚未开启。30新场景14新log1196帧已分离；1166过去扫描/30缺测已审计；600构造主查询+278挑战及203预选回放准备。

待完成与限制：30场景14log1196帧保持封存；每场景20条有效自然指令、真实标签与独立参考尚未齐备。现有600构造+278挑战和439原文均不等于合格自然分布。

证据：`splits/split_audit_002/report.json`、`data_manifest/reserved_causal_inputs_full_001/audit_001/report.json`。

## 自然语料覆盖与真实表达采集

覆盖审计和采集工具完成，有效人工数据待提交审核。固定146候选2308原文全部核对官方属性；50m下66场景1097原文/364官方静态属性，15场景6log有20原文、0场景有20官方静态属性。原30场景选场不改，不能仅换选场满足要求。独立2308命令映射与1196原图SHA/尺寸/来源通过。

待完成与限制：自然表达与有效标签不足是外部依赖；不使用模板、机器改写或降低距离规则补足。30场景需各20条有效指令，即至少600；最低20场景400条口径不用于替代逐场景要求。

证据：`task_protocol/human_natural_collection_001.md`、`data_manifest/natural_candidate_coverage_001/report.md`、`data_manifest/natural_candidate_coverage_001/report.json`、`data_manifest/natural_data_preparation_audit_002/report.json`。

## 人工审核

待真实提交验收。既有对象核验及439原文审核入口保留；新增30场景1196真实帧自由表达采集页。原8768端口服务已兼容升级，桌面/手机12组正式帧检查通过；不产生测试假人工记录。

待完成与限制：真实人工审核与自由表达提交仍0；提交后仅原始待审核，身份、运动、颜色、歧义与可达性仍unknown。现有表单不能替代全部漏检和跨帧长链审核。

证据：`visualizations/causal_development_review_002/review_manifest.json`、`annotation/human_natural_collection_001/manifest.json`、`visualizations/natural_collection_browser_canonical_001/report.json`、`source_bundle/natural_collection_service_upgrade_001/report.json`。

## 工程与复现

源码及主450查看器新增验收通过，最终验收未完成。390源码；原机452通过，纯代码450通过2真实数据skip；业务Ruff通过。连续KITTI90帧独立数据/指标/浏览器核验完成。

待完成与限制：后续消融/故障展示、最终全部结构化文件与历史保护、完整复现包、全部代码Git同步仍待完成；v14不含新增量。

证据：`source_bundle/tests_full_035.log`、`source_bundle/tests_full_036.log`、`source_bundle/ruff_completion_050.log`、`source_bundle/git_diff_check_036.log`、`source_bundle/async_no_match_test_before_001/manifest.json`、`baseline_verification.json`、`source_bundle/tests_full_037.log`、`source_bundle/ruff_051.log`、`source_bundle/diff_check_037.log`、`source_bundle/tests_natural_collection_002.log`、`source_bundle/tests_natural_collection_003.log`、`source_bundle/tests_full_038.log`、`source_bundle/ruff_052.log`、`source_bundle/diff_check_038.log`、`source_bundle/tests_execution_direction_001.log`、`source_bundle/tests_full_040.log`、`source_bundle/ruff_054.log`、`source_bundle/diff_check_040.log`、`source_bundle/tests_full_041.log`、`source_bundle/tests_full_042.log`、`source_bundle/ruff_056.log`、`source_bundle/diff_check_042.log`、`visualizations/realtime_main450_preview_003/manifest.json`、`visualizations/realtime_main450_preview_003/release.json`、`visualizations/realtime_main450_browser_qa_001/report.json`、`source_bundle/realtime_main450_mobile_preflight_failure_001.json`、`source_bundle/realtime_main450_mobile_preflight_pass_003.json`、`source_bundle/tests_full_044.log`、`source_bundle/ruff_057.log`、`source_bundle/github_sync_001/tests_export_016.log`、`source_bundle/github_sync_001/ruff_export_016.log`、`visualizations/thesis_portal_013/manifest.json`、`visualizations/thesis_portal_013/completed_evidence.json`、`visualizations/thesis_portal_browser_qa_017/report.json`、`task_protocol/unified_portal_main450_001.md`、`task_protocol/semantic_source_compatibility_003.md`、`ablations/semantic_navigation_chain_003/preflight.json`、`ablations/semantic_navigation_chain_003/release.json`、`ablations/semantic_navigation_chain_003/worker_started.json`、`statistics/semantic_compatibility_audit_rehearsal_main_001/report.json`、`source_bundle/tests_full_046.log`、`source_bundle/ruff_061.log`、`source_bundle/github_sync_001/tests_export_017.log`、`source_bundle/github_sync_001/ruff_export_017.log`。

## 完整主450可视化

开发450查看器与代表条件浏览器验收完成。实际RGB/检测/16位有效深度、三维点与节点、关系、三个排序、道路/车身/计划和保存执行轨迹；两个时间轴，GT默认隐藏。450清单/1350证据链接与24桌面手机代表条件检查通过。

待完成与限制：未逐帧人工审核；尚未完成的语义消融/GPU故障需在实际结束后加入；查看不是新执行、新数据或新到达。

证据：`visualizations/realtime_main450_preview_003/manifest.json`、`visualizations/realtime_main450_preview_003/release.json`、`visualizations/realtime_main450_browser_qa_001/report.json`、`source_bundle/realtime_main450_mobile_preflight_failure_001.json`、`source_bundle/realtime_main450_mobile_preflight_pass_003.json`、`source_bundle/tests_full_044.log`、`source_bundle/ruff_057.log`、`source_bundle/github_sync_001/tests_export_016.log`、`source_bundle/github_sync_001/ruff_export_016.log`。

## 统一实验与交付入口

门户013桌面手机及失败降级验收通过。门户016保留原区域/参考/主450/交互入口，加入连续KITTI90帧；桌面手机及全部静态HTTP链接只读验收通过。

待完成与限制：后续语义消融/GPU故障/独占性能须完成后再整合；入口不是全Goal完成证明。

证据：`visualizations/thesis_portal_013/manifest.json`、`visualizations/thesis_portal_013/completed_evidence.json`、`visualizations/thesis_portal_browser_qa_017/report.json`、`task_protocol/unified_portal_main450_001.md`。

## 主450对象、候选与语义失败诊断

450全量后验诊断及独立复核完成。57,460帧、1,317投递响应、2,672SHA、27区间复算通过。明确30,538帧空mapping源于尚未选择，重建全节点后原四项成绩不变；缺结果与响应/事件分母单列。

待完成与限制：不是人工实例真值或模块因果效应；不改冻结方法，需待缺一消融、真实人工与新封存验收。

证据：`task_protocol/main_failure_chain_002.md`、`statistics/main_failure_chain_full_002/report.json`、`statistics/main_failure_chain_full_002/trials.csv`、`statistics/main_failure_chain_coverage_001/report.json`、`statistics/main_failure_chain_audit_001/report.json`、`statistics/main_failure_chain_full_001/interrupted.json`。

## 可启动、单步与安全停车的交互控制台

一个固定开发任务的桌面手机实际操作及独立核验通过。14真实CPU前端/67控制区间/52运动区间/10操作；暂停状态固定、7步制动到零，RGB与深度各11种。独立连续车身与方向核验通过。

待完成与限制：不是50任务逐项浏览器验收，不并入主450；用户中断无到达，不是GPU实时性能或自然语言封存测试。

证据：`task_protocol/live_navigation_console_001.md`、`交互仿真_操作与复现_001.md`、`visualizations/live_navigation_browser_qa_001/report.json`、`vehicle_execution/live_navigation_independent_audit_001/report.json`、`source_bundle/live_navigation_process_probe_001.json`、`source_bundle/tests_full_048.log`、`source_bundle/github_sync_001/tests_export_018.log`、`source_bundle/ruff_063.log`。

## 论文正文

按用户最新要求暂缓。本阶段只整理实验状态、证据、复现命令和交付材料；旧阶段稿保留。

待完成与限制：用户后续通过GPT Work拟定整体框架并正式撰写；当前不新增论文文字章节，不标记论文或Goal完成。

证据：`GPT_Work_写作交接.md`。

## KITTI连续车载外部诊断

固定90帧实际CPU前端、融合与独立身份评价通过。三序列各30帧；单帧、原融合、改进融合，统一名义0.1秒；独立完整P2/OXTS、NetworkX及固定TrackEval核验，保留一个旧Object重合帧。

待完成与限制：三个开发短窗口不能替代新封存、人工真值或导航到达；所有负结果及DontCare影响单列。

证据：`KITTI连续车载_实验与复现交接_001.md`、`statistics/kitti_tracking_full_001/report.json`、`statistics/kitti_tracking_full_001/independent_audit_002/report.json`、`visualizations/kitti_tracking_browser_qa_001/report.json`。
