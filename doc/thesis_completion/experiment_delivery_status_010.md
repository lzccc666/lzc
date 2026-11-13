# 实验完成度与交付清单（第10版）

主450失败环节诊断更新（2026-11-12T21:07:34.759744+00:00），仅实验与复现交接，不写论文正文/框架。450试验、57,460感知帧、1,317已保存投递响应、2,672原输入/产物SHA、27项配置配对区间已独立核对。001首次诊断发现旧evaluation_identity在selected=None时直接保存空mapping，不能据此判漏检；原失败完整保留。002从保存节点与冻结目标重建全部类别/2.5m Hungarian关联；30,538个原空映射帧其实有可关联节点，13,922个selected非空帧重建与原保存映射完全相同。重建不是新增观测或人工真值；所有原完整试验四项业务结果精确不变，1失败前缀不给完整成功信用。

每方法90可达正条件中，三方法都曾建图关联到可接受目标；安全检查前曾含正确候选为规则90/base90/new54，通过安全检查曾含正确候选84/73/33，原最终Top1正确84/72/33、连续3帧selected正确82/66/32、实际到达50/39/4均保持原值。base的安全候选73含1失败前缀，完整Top1分子仍72。不同时间存在事件不可当严格单调漏斗；安全前候选已受解析/属性影响，不是独立纯规则候选。

可达正条件中base保存90个解析响应均保留颜色且无额外锚点/关系，其中1例缺完整事件日志，完整事件应用只知89加1未知；新Adapter90次后端入口只有74响应投递并应用，16未投递，不能给未知输出语义评分。74已投递均有同名near锚点、52丢失目标颜色，仅22保留颜色，完整任务解析诊断0/74；格式通过74/74不等于语义正确。全部种类中新Adapter134已投递解析均有同名near锚点，不能据此声称全部150后端原文都相同。正任务属性base335/new233已投递响应均为unknown；新Adapter有3次排序实际应用但没有改变任何帧的最终首位，base完整事件排序应用0。不能将选择变差全部归因于重排。

训练1400解析有145条无锚点/关系且含颜色，验证180有24条；与本合成任务精确查询文字重合均0。训练7条、验证1条近关系的锚点/目标文字同名，不自动证明指代相同或标签错误。监督来自声明语法而非人工自然指令，这些分布与应用现象只是后验解释；模块因果贡献仍等正在运行的450缺一，当前快照61/450。GPU故障/独占性能继续原队列，人工和合格封存指令按API记录，新封存未开启，Goal为false。业务381源码未改，原424测试/导出422通过2数据skip继续有效，本轮辅助诊断实际执行和独立审计通过，不冒称重复全量测试。


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

450缺一正在执行，源集合兼容审核已预检并排队。49/450快照；共同66源码不变、多1个未调用文件。003预检与主450独立统计回归通过，版本化等待器已运行；桥接未写。 最新只读快照61/450；后验诊断不替代消融。

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

源码及主450查看器新增验收通过，最终验收未完成。381源码已准备；原机424通过，纯代码422通过2真实数据skip；业务Ruff及差异检查通过。450只读查看器与桌面/手机检查通过。 统一入口013及65证据链接已验收，源集合兼容等待器单列，不计实验完成。

待完成与限制：后续消融/故障展示、最终全部结构化文件与历史保护、完整复现包、全部代码Git同步仍待完成；v14不含新增量。

证据：`source_bundle/tests_full_035.log`、`source_bundle/tests_full_036.log`、`source_bundle/ruff_completion_050.log`、`source_bundle/git_diff_check_036.log`、`source_bundle/async_no_match_test_before_001/manifest.json`、`baseline_verification.json`、`source_bundle/tests_full_037.log`、`source_bundle/ruff_051.log`、`source_bundle/diff_check_037.log`、`source_bundle/tests_natural_collection_002.log`、`source_bundle/tests_natural_collection_003.log`、`source_bundle/tests_full_038.log`、`source_bundle/ruff_052.log`、`source_bundle/diff_check_038.log`、`source_bundle/tests_execution_direction_001.log`、`source_bundle/tests_full_040.log`、`source_bundle/ruff_054.log`、`source_bundle/diff_check_040.log`、`source_bundle/tests_full_041.log`、`source_bundle/tests_full_042.log`、`source_bundle/ruff_056.log`、`source_bundle/diff_check_042.log`、`visualizations/realtime_main450_preview_003/manifest.json`、`visualizations/realtime_main450_preview_003/release.json`、`visualizations/realtime_main450_browser_qa_001/report.json`、`source_bundle/realtime_main450_mobile_preflight_failure_001.json`、`source_bundle/realtime_main450_mobile_preflight_pass_003.json`、`source_bundle/tests_full_044.log`、`source_bundle/ruff_057.log`、`source_bundle/github_sync_001/tests_export_016.log`、`source_bundle/github_sync_001/ruff_export_016.log`、`visualizations/thesis_portal_013/manifest.json`、`visualizations/thesis_portal_013/completed_evidence.json`、`visualizations/thesis_portal_browser_qa_017/report.json`、`task_protocol/unified_portal_main450_001.md`、`task_protocol/semantic_source_compatibility_003.md`、`ablations/semantic_navigation_chain_003/preflight.json`、`ablations/semantic_navigation_chain_003/release.json`、`ablations/semantic_navigation_chain_003/worker_started.json`、`statistics/semantic_compatibility_audit_rehearsal_main_001/report.json`、`source_bundle/tests_full_046.log`、`source_bundle/ruff_061.log`、`source_bundle/github_sync_001/tests_export_017.log`、`source_bundle/github_sync_001/ruff_export_017.log`。

## 完整主450可视化

开发450查看器与代表条件浏览器验收完成。实际RGB/检测/16位有效深度、三维点与节点、关系、三个排序、道路/车身/计划和保存执行轨迹；两个时间轴，GT默认隐藏。450清单/1350证据链接与24桌面手机代表条件检查通过。

待完成与限制：未逐帧人工审核；尚未完成的语义消融/GPU故障需在实际结束后加入；查看不是新执行、新数据或新到达。

证据：`visualizations/realtime_main450_preview_003/manifest.json`、`visualizations/realtime_main450_preview_003/release.json`、`visualizations/realtime_main450_browser_qa_001/report.json`、`source_bundle/realtime_main450_mobile_preflight_failure_001.json`、`source_bundle/realtime_main450_mobile_preflight_pass_003.json`、`source_bundle/tests_full_044.log`、`source_bundle/ruff_057.log`、`source_bundle/github_sync_001/tests_export_016.log`、`source_bundle/github_sync_001/ruff_export_016.log`。

## 统一实验与交付入口

门户013桌面手机及失败降级验收通过。完整主450、KITTI、身份及HD地图开发诊断统一入口；论文安排暂缓，原审核/新自然表达/合格测试数量分开，读取失败显示未知。

待完成与限制：后续语义消融/GPU故障/独占性能须完成后再整合；入口不是全Goal完成证明。

证据：`visualizations/thesis_portal_013/manifest.json`、`visualizations/thesis_portal_013/completed_evidence.json`、`visualizations/thesis_portal_browser_qa_017/report.json`、`task_protocol/unified_portal_main450_001.md`。

## 主450对象、候选与语义失败诊断

450全量后验诊断及独立复核完成。57,460帧、1,317投递响应、2,672SHA、27区间复算通过。明确30,538帧空mapping源于尚未选择，重建全节点后原四项成绩不变；缺结果与响应/事件分母单列。

待完成与限制：不是人工实例真值或模块因果效应；不改冻结方法，需待缺一消融、真实人工与新封存验收。

证据：`task_protocol/main_failure_chain_002.md`、`statistics/main_failure_chain_full_002/report.json`、`statistics/main_failure_chain_full_002/trials.csv`、`statistics/main_failure_chain_coverage_001/report.json`、`statistics/main_failure_chain_audit_001/report.json`、`statistics/main_failure_chain_full_001/interrupted.json`。

## 论文正文

按用户最新要求暂缓。本阶段只整理实验状态、证据、复现命令和交付材料；旧阶段稿保留。

待完成与限制：用户后续通过GPT Work拟定整体框架并正式撰写；当前不新增论文文字章节，不标记论文或Goal完成。

证据：`GPT_Work_写作交接.md`。
