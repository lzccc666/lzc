# 阶段6变更记录：集成与回归

- 完成日期：2026-09-01
- 状态：已完成并通过最终验收
- 项目根目录：`/root/autodl-tmp/lizichong`
- Git基线：`eab10f5aebfe558d1e50fae543efcce3c27e613f`
- 基线标签：`v1.0-offline-baseline`
- 运行环境：Conda `conceptgraph`，Python 3.10.20

## 1. 实施范围

本阶段完整实施了 `doc/project_optimization_plan.md` 的“阶段6：集成与回归”。工作范围包括：

- 复用阶段2的双模式目标选择、安全环带、连通性、A*、路径平滑和连续碰撞检查；
- 复用阶段3的结构化关系边、旧三元组兼容、正反向关系解析、增量更新和可视化入口；
- 复用阶段4的严格schema、隔离磁盘缓存、禁用模式、超时、异常、非法schema、非法候选和规则回退；
- 复用阶段5的 `objects_all_frames` 快照回放、`StreamlinedMapper.update(frame)` 内存接口、在线状态机、重规划和事务回滚；
- 在 `scene-0916` 和 `scene-0003` 上执行隔离的Stage2至Stage5真实回归；
- 生成兼容性矩阵、性能、确定性、故障注入、基线核验、代码审查和最终验收报告。

本阶段没有运行目标检测或深度生成，没有训练、下载或无缓存调用Qwen，没有解压nuScenes，没有处理论文，没有删除用户文件，没有提交或推送Git。nuScenes仍按录制序列回放；规划路径不会改变后续传感器观测，本结果不表述为闭环因果实验。

## 2. 开始前基线

任何Stage6代码和回归产物修改前，已写入：

`/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/baseline_before_stage6.json`

记录内容包括：

- HEAD、分支、HEAD标签、完整dirty清单、tracked diff和status指纹；
- 当前测试 `126 collected / 126 passed`，仅有2条torchvision Beta API上游警告；
- 阶段1至阶段5输出目录、报告、旧数据集、checkpoint、`MEMORY.md`、`PAPER_MEMORY.md` 和既有文档指纹；
- 所有 `exp_qwen2vl_semantics_v1`、`exp_streamlined_yolo_outdoor_objects_v1` 和 `objects_all_frames` 聚合指纹；
- 两个真实场景的快照、RGB、深度、缓存检测和位姿指纹；
- Stage2的 `first_reachable=24/9`、`joint_utility=21/11`；
- Stage3两场最小、平均、最终活跃边一致率均为1.0；
- Stage4禁用、缓存、规则回退和严格schema有效数0；
- Stage5四组双运行确定性哈希、回放帧数、地图对照和故障注入结果。

基线文件写入后立即重新解析，SHA-256为：

`704b2945561c7b5781cf2dc4730bf5ace930c04858a22fe957bf20af6dc20710`

## 3. 新增和修改文件

新增：

```text
conceptgraph/scripts/run_project_v2_integration_regression.py
conceptgraph/hydra_configs/project_v2_integration_regression.yaml
tests/integration/__init__.py
tests/integration/test_project_v2_integration.py
doc/change_logs/stage6_integration_and_regression.md
```

修改：

```text
conceptgraph/scripts/run_stage4_offline_qwen_eval.py
```

Stage4只增加以下精确白名单：

`/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/regression_stage4`

没有放宽为任意输出目录；原 `qwen_eval` 和Stage5隔离回归路径行为保持不变。阶段2至阶段5核心算法均未重写，也没有无关重构。

## 4. 统一入口和输出保护

新入口：

```bash
conda run -n conceptgraph python -m \
  conceptgraph.scripts.run_project_v2_integration_regression
```

只重新汇总既有真实回归并执行静态门禁：

```bash
conda run -n conceptgraph python -m \
  conceptgraph.scripts.run_project_v2_integration_regression --finalize-only
```

运行器只接受精确输出根：

`/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression`

其他绝对或相对输出路径均被拒绝。JSON采用同目录临时文件加 `os.replace` 原子写入。Stage2、Stage3和Stage4回归分别写入 `regression_stage2`、`regression_stage3` 和 `regression_stage4` 子目录；快照与端到端回放直接写入Stage6的 `snapshot_replay` 和 `end_to_end_replay`，没有覆盖Stage1至Stage5产物。

## 5. 兼容性矩阵

兼容性矩阵位于：

`/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/compatibility_matrix.json`

最终包含19个有诊断字段的必需组合。每行记录输入格式、入口、场景、成功状态、实际选择节点、路径有效性、关系来源、fallback、绝对输出路径和失败诊断。覆盖范围为：

- 两场旧完整场景图与 `first_reachable`；
- 两场结构化边与 `joint_utility`；
- 两场旧三元组加载和正向导航；
- 同一 `structured_edges.json` 同时供导航和 `visualize_scenegraph` 使用，完整边内容一致；
- 两场 `objects_all_frames` 快照回放；
- 两场 `StreamlinedMapper.update(frame)` 与 `OnlineSemanticNavigator` 端到端回放；
- Qwen关闭纯规则模式；
- Qwen开启且首次、复跑均命中隔离缓存；
- Qwen超时、异常、非法schema、非法候选ID和迟到结果回退；
- 旧入口、新入口、CLI、旧pickle、新JSON/JSONL和内存接口smoke。

`scene-0003` 的旧三元组正向查询成功；旧格式不具备结构化反向语义，因此反向查询保留 `NO_SEMANTIC_CANDIDATES` 诊断。结构化边的同一反向查询成功选择节点14。这是明确记录的新旧能力差异，不把旧格式缺失能力伪装成成功。

## 6. Stage2真实回归

Stage2隔离输出：

`/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/regression_stage2`

结果：

| 场景 | first_reachable | joint_utility | 两模式成功 | 基线一致 |
| --- | ---: | ---: | --- | --- |
| scene-0916 | 24 | 21 | 是 | 是 |
| scene-0003 | 9 | 11 | 是 | 是 |

四条成功路径均重新实例化离线流水线、重新执行目标选择和连续栅格碰撞检查，节点选择完全复现且路径无碰撞。每个 `evaluated_candidates` 均显式包含：

```text
score/text_score
relation_score
path_cost
min_clearance_m
fallback_distance_m
dynamic_risk
rejection_reason
selection_utility
selection_components
semantic_rank
goal_candidates
```

因此Top-K语义、关系、路径代价、clearance、fallback、动态风险、拒绝原因和最终联合效用均可审计。

## 7. Stage3真实回归

Stage3隔离输出：

`/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/regression_stage3`

两场结构化边构建均成功：

| 场景 | 最小一致率 | 平均一致率 | 最终一致率 | 结构化边 | 旧边 |
| --- | ---: | ---: | ---: | ---: | ---: |
| scene-0916 | 1.0 | 1.0 | 1.0 | 63 | 20 |
| scene-0003 | 1.0 | 1.0 | 1.0 | 59 | 21 |

每场均实际查找有向边和 `near` 对称边，验证正向、反向、对称正向和对称反向解析。旧pickle加载后全部标记 `edge_format=legacy`、`metadata_available=false`。结构化查询均使用 `edge_source=structured_edge` 且没有几何fallback。

可视化入口实际读取：

`/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/regression_stage3/stage3_full_rebuild/scene-0916/structured_edges.json`

输出：

`/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/structured_edge_visualization/scene-0916/scenegraph_summary.json`

可视化的63条完整边内容与导航使用的结构化边逐条一致。

## 8. Stage4真实回归

Stage4隔离输出：

`/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/regression_stage4`

运行前只复制既有 `qwen_eval` 数据集、缓存、processor smoke和Stage4基线到隔离目录。没有下载、训练或无缓存推理。

| 场景 | 首次缓存命中 | 首次后端调用 | 复跑缓存命中 | 复跑后端调用 | 严格schema有效数 | 规则/Qwen回退选择 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scene-0916 | 3/3 | 0 | 3/3 | 0 | 0 | 21/21 |
| scene-0003 | 3/3 | 0 | 3/3 | 0 | 0 | 11/11 |

Qwen关闭时后端调用为0并返回 `QWEN_DISABLED` 规则结果。超时、后端异常、非法schema、非法候选ID和迟到版本结果均实际注入；导航安全选择不变，关闭后跟踪任务为0。

严格schema有效数仍为0。报告将其明确记录为模型指标；规则回退后的正确节点不计为模型schema有效率或模型准确率。

## 9. Stage5双真实场景回放

### 9.1 快照回放

| 场景 | 帧数 | 目标出现 | 确认 | 首次规划 | 规划次数 | 失败帧 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scene-0916 | 38（3-40） | 3 | 5 | 5 | 11 | 0 |
| scene-0003 | 39（1-39） | 1 | 3 | 3 | 7 | 0 |

### 9.2 端到端回放

| 场景 | 帧数 | 目标出现 | 确认 | 首次规划 | 规划次数 | 失败/恢复 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scene-0916 | 41 | 2 | 4 | 4 | 11 | 2/1 |
| scene-0003 | 40 | 0 | 2 | 2 | 12 | 0/0 |

目标确认前没有路径，非规划帧不保存完整路径，末帧Qwen待处理任务为0。七类触发器 `target_confirmed`、`target_changed`、`better_target_persisted`、`path_blocked`、`vehicle_moved`、`periodic_interval` 和 `manual_request` 均在四组回放中出现并保留接受/拒绝诊断。

最终地图对照：

| 场景 | 实际/参考节点 | 中心中位误差 | 4 m内比例 | 静态类别最大误差 | 离群点 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| scene-0916 | 45/44 | 1.0839 m | 81.82% | 0.0076 m | 8 | 通过 |
| scene-0003 | 24/23 | 0.0052 m | 95.65% | 0.0094 m | 1 | 通过 |

旧轻量快照没有 `motion_state` 和 `is_short_term`，所以 `reference_dynamic_count` 和动态差值保持 `null`，没有伪造动态参考成功。

## 10. 确定性

两个模式、两个场景各连续运行2次。排除耗时和异步完成时刻后，状态、目标、触发器、失败码和路径哈希完全一致：

| 模式 | 场景 | 内容SHA-256 |
| --- | --- | --- |
| snapshot_replay | scene-0916 | `17d13b1b1fb720a083ee0800bd5658d112e15b35557fe4a98687af89ef9a89cc` |
| snapshot_replay | scene-0003 | `819be499a8d1c488d51c1588b49c43d7a6d33f2ac166193d748bcde2d7714de2` |
| end_to_end_replay | scene-0916 | `66353d3390a77401716f5092ef8f445b3cc957074286df0709c69370ff0538b1` |
| end_to_end_replay | scene-0003 | `be4c580affb155a623266bb2bef44045983a5ca78a3e78437827ee54961aa660` |

## 11. 性能统计

性能报告：

`/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/performance_report.json`

报告按模式、场景和规划/非规划帧拆成8组。所有已执行指标均包含 `count`、`mean`、`median`、`p95`、`max` 和 `total`。中位数摘要如下，单位为秒：

| 模式/场景 | 帧类型 | mapping | relation | ranking | planning | total frame |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| snapshot/0916 | 规划 | 不执行 | 0.00291 | 0.00202 | 0.31757 | 0.33012 |
| snapshot/0916 | 非规划 | 不执行 | 0.00334 | 0.00260 | 0.00000025 | 0.02104 |
| snapshot/0003 | 规划 | 不执行 | 0.00361 | 0.00142 | 0.70627 | 0.72022 |
| snapshot/0003 | 非规划 | 不执行 | 0.00317 | 0.00144 | 0.00000027 | 0.01539 |
| end-to-end/0916 | 规划 | 0.06904 | 0.00342 | 0.00236 | 0.41139 | 0.63203 |
| end-to-end/0916 | 非规划 | 0.07175 | 0.00467 | 0.00297 | 0.00000025 | 0.18254 |
| end-to-end/0003 | 规划 | 0.04072 | 0.00291 | 0.00155 | 0.80485 | 0.98826 |
| end-to-end/0003 | 非规划 | 0.04239 | 0.00410 | 0.00183 | 0.00000026 | 0.14342 |

Stage4隔离缓存任务耗时为真实缓存读取与schema回退耗时，两个场景首次3个任务均为亚毫秒量级，后端调用为0。

- CPU峰值RSS：1289.79 MB；
- `detection_time_s=null`，状态为 `not_executed_cached_input`；
- snapshot的 `mapping_time_s=null`，状态为 `not_executed_snapshot_input`；
- 逐帧回放Qwen关闭，`qwen_time_s=null`，状态为 `not_executed_qwen_disabled`；
- 没有实际CUDA工作负载，`peak_gpu_memory_mb=null`，状态为 `not_recorded_no_actual_cuda_workload`。

未执行项没有伪造为0。

## 12. 故障注入

统一报告：

`/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/failure_injection_report.json`

实际覆盖：

- 损坏JSON和JSONL被解析器拒绝，故障fixture在检查后移除；
- 损坏gzip快照被拒绝；
- 空对象返回结构化失败并保持序列可继续；
- 非法位姿触发完整事务回滚，后续帧恢复；
- 全阻塞占据图触发规划失败，恢复地图后继续；
- 重复帧返回深拷贝的幂等结果，不产生第二次规划；
- 乱序帧返回 `OUT_OF_ORDER_FRAME` 且不消费帧号；
- Qwen超时、异常、非法schema和非法候选ID全部规则回退；
- 迟到重排因 `object_version/candidate_version` 不匹配被丢弃；
- 关闭导航器后待处理任务为0。

## 13. 测试与静态检查

新增集成测试共8项，覆盖：

- Stage6精确输出根保护；
- 性能统计与部分缺失语义；
- Top-K完整诊断字段；
- 旧三元组与新结构化关系序列化及正反向查询；
- 重复帧、乱序帧、事务回滚、迟到结果和任务关闭；
- Qwen超时、异常和非法schema回退；
- 兼容矩阵诊断字段；
- JSON/JSONL最终重解析计数。

执行命令：

```bash
conda run -n conceptgraph pytest -q tests/integration/test_project_v2_integration.py
conda run -n conceptgraph pytest -q tests
conda run -n conceptgraph ruff check <全部当前改动Python文件>
git diff --check
```

当前结果：

- Stage6专项：8 passed；
- `tests`全量：134 passed，2条torchvision Beta API上游警告；
- 全部当前改动Python文件Ruff：通过；
- `git diff --check`：通过；
- Stage6输出下最终JSON/JSONL重新解析：131个JSON、8个JSONL、1628行，错误0。本文写入后的最终门禁会重新统计文件数。

没有修复与本项目阶段无关的历史Ruff问题。

## 14. 代码审查

结构化代码审查报告：

`/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/code_review_report.json`

审查范围为阶段1至阶段6全部当前改动，重点检查旧API/CLI、输入原地修改、共享可变状态、Qwen安全边界、状态机与异步竞态、异常帧事务、确定性、输出保护和报告语义。

发现并修复：

1. Stage4输出白名单缺少Stage6精确隔离路径；只加入一个精确路径，没有开放任意输出根。
2. Stage5原始性能字段不能直接满足Stage6命名和缺失语义；新增规范化报告，未执行项使用 `null+status`。
3. 初版Stage6验收器错误要求旧三元组的所有反向查询都成功；修正为旧格式加载成功且至少一个正向导航成功，同时保留反向能力差异诊断。
4. 初版兼容矩阵把Stage2节点与Stage3结构化查询证据组合展示；修正为记录实际结构化查询的节点、路径和边文件。
5. 初版结构化边/可视化一致性只比较数量；修正为对规范排序后的完整边内容逐条比较。
6. 初版故障矩阵缺少非法候选ID单列；新增真实 `ILLEGAL_CANDIDATE_ID` 注入与安全节点回退。

修复后专项测试、Ruff、碰撞复查、Stage3验证和最终汇总均重新执行。无未解决阻断问题。

## 15. 关键产物

统一输出根：

`/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression`

关键文件：

```text
/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/baseline_before_stage6.json
/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/compatibility_matrix.json
/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/integration_report.json
/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/stage6_comparison.json
/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/performance_report.json
/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/determinism_report.json
/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/failure_injection_report.json
/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/regression_report.json
/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/code_review_report.json
/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/baseline_verification.json
/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/validation_report.json
/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/snapshot_replay/scene-0916
/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/snapshot_replay/scene-0003
/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/end_to_end_replay/scene-0916
/root/autodl-tmp/lizichong/outputs/project_v2/integration_regression/end_to_end_replay/scene-0003
```

## 16. 基线保护和限制

最终基线核验将重新计算全部保护目录、数据集组、文档文件及双场景输入。当前核验为 `passed=true`，Stage1至Stage5旧产物和真实输入指纹均未变化。

剩余限制：

- 既有Qwen adapter严格schema有效率为0，当前必须依赖规则回退；
- Stage6只验证两个真实录制场景，不代表完整数据集统计；
- 旧轻量快照没有动态参考字段，不能提供motion_state动态真值对照；
- 端到端回放只消费缓存检测，不验证在线检测器性能；
- nuScenes录制回放不是规划改变后续观测的闭环实验。

## 17. Git状态

分支仍为 `main`，HEAD仍为 `eab10f5aebfe558d1e50fae543efcce3c27e613f`，标签仍为 `v1.0-offline-baseline`。工作树在Stage6开始前已包含阶段1至阶段5和用户文件的未提交改动，完成后仍为dirty；这些内容全部保留，没有回滚、删除、提交或推送。
