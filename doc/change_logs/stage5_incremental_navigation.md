# 阶段5：增量导航

## 1. 完成范围

本阶段完整实施了 `doc/project_optimization_plan.md` 的“阶段5：增量导航”。实现把阶段2的目标选择与安全规划、阶段3的增量关系图、阶段4的可选Qwen语义代理接入逐帧内存流程，并完成双真实场景的快照回放和端到端回放。

本阶段只消费已有本地逐帧对象快照、RGB-D、位姿和缓存检测结果，没有运行检测或深度生成，没有解压nuScenes、训练Qwen、实现阶段6、修改论文、删除用户文件、提交或推送Git。所有新增实验产物位于：

```text
/root/autodl-tmp/lizichong/outputs/project_v2/incremental_navigation
```

## 2. 主要实现

新增文件：

- `conceptgraph/navigation/online.py`：`OnlineSemanticNavigator`、内存对象转换、异步Qwen调度和逐帧事务式故障恢复。
- `conceptgraph/navigation/state_machine.py`：目标确认、锁定迟滞、短暂消失、更优候选持续确认和目标失效状态机。
- `conceptgraph/navigation/replan_policy.py`：确定性触发器优先级、debounce、cooldown、同帧去重和审计。
- `conceptgraph/scripts/run_incremental_semantic_navigation.py`：双模式、双场景、双运行回放，故障注入、地图对照和验收报告。
- `conceptgraph/hydra_configs/incremental_navigation.yaml`：状态机、重规划、回放范围和地图对照容差。
- `tests/navigation/test_online_state_machine.py`、`test_replan_policy.py`、`test_snapshot_replay.py`：阶段5专项测试。

复用和扩展文件：

- `conceptgraph/navigation/pipeline.py` 新增 `set_in_memory_inputs()`，允许节点、结构化边和占据栅格直接从内存传入；规划仍由阶段2 `TargetSelector` 执行。
- `conceptgraph/navigation/planner.py` 新增在基础BEV上叠加当前对象的接口，继续使用安全环带、起点连通分量、A*、路径平滑和连续碰撞检查。
- `conceptgraph/navigation/types.py` 新增严格的 `OnlineNavigationUpdate` 和 `OnlineNavigationFailure`。
- `conceptgraph/slam/streamlined_mapping.py` 抽取 `MappingFrame`、`MappingUpdate` 和 `StreamlinedMapper.update(frame)`；旧Hydra CLI改为调用同一核心，保留原入口。
- `conceptgraph/scripts/run_stage4_offline_qwen_eval.py` 只额外放行精确的阶段5隔离回归目录，避免覆盖既有Stage4产物。

逐帧数据流如下：

```text
已有快照或StreamlinedMapper.update(frame)
  -> 内存SceneGraphNode与changed_object_ids
  -> IncrementalSceneGraph.update_snapshot()
  -> 规则候选排序与锁定状态机
  -> ReplanPolicy审计触发器
  -> Stage2 TargetSelector + 安全环带 + A* + 连续碰撞检查
  -> OnlineNavigationUpdate
```

在线循环不读取最终场景图文件。逐帧输出保存状态、候选、目标、触发器、关系局部更新、占据图变化、耗时、Qwen事件和结构化失败；只有规划或重规划帧保存完整路径，其他帧只保存路径点数和路径哈希。

## 3. 状态机与锁定策略

状态全集为：

```text
SEARCHING
PROVISIONAL
LOCKED
PLANNING
TRACKING
REPLANNING
UNREACHABLE
```

规则解析在 `set_query()` 内立即完成。默认配置要求候选连续3次观测、置信度至少0.20且持续3帧后才能锁定。锁定前不生成路径。锁定目标允许单帧短暂消失，长期消失4帧后按配置退回 `SEARCHING`；也可配置为 `UNREACHABLE`。

略高分候选不会切换目标。挑战者必须置信度至少0.45、比分差至少0.12并连续满足3帧，才产生 `better_target_persisted` 并替换锁定目标。`LOCKED`、`PLANNING` 和 `REPLANNING` 可能在一次 `update()` 内完成，因此完整瞬态保存在 `state_trace`。正常真实回放没有以 `UNREACHABLE` 作为帧末稳态；该状态由全阻塞规划失败故障注入和专项测试实际覆盖。

重复帧返回上次更新的深拷贝，内容完全幂等；乱序帧返回 `OUT_OF_ORDER_FRAME` 且不消费帧号。单帧异常会回滚状态机、重规划策略、关系图、节点、占据图、轨迹、版本号和路径，再以 `FRAME_PROCESSING_EXCEPTION` 记录，后续帧继续运行。

## 4. 重规划规则

固定优先级从高到低为：

```text
path_blocked
target_changed
better_target_persisted
target_confirmed
manual_request
vehicle_moved
periodic_interval
```

默认cooldown为2帧；安全关键和明确用户事件可绕过cooldown。每类触发器都有独立debounce和持续事件去重，同一帧最多进入一次规划。没有触发器时不调用A*。

占据图更新后只检查车辆当前位置之后的剩余路径，并复用连续碰撞检查；只有路径实际被阻断才触发 `path_blocked`。目标中心相对上次规划位置变化至少0.75 m才触发 `target_changed`。所有接受和拒绝均记录触发器及原因，例如 `cooldown_active`、`debounced_active_event`、`no_locked_target` 或 `lower_priority_than:*`。

## 5. Qwen异步安全边界

Qwen只负责异步查询解析或对阶段2已经验证可达的候选提出建议，不改变对象可达性、碰撞检测、A*路径或阶段2权威选择。真实阶段5回放默认关闭Qwen，规则导航结果不依赖模型。

开启时使用有界线程池调度，不等待异步任务，因此 `set_query()` 和逐帧 `update()` 不被模型阻塞。候选集合显著变化且满足提交间隔时才提交重排。阶段7已把原来的全量版本绑定修订为任务级有效期：query parse绑定query版本，对象属性绑定对象身份，候选重排绑定query和已验证候选集合，避免正常逐帧对象变化使所有异步结果失效。取消、迟到、安全候选变化、超时、异常和缓存命中均进入审计。关闭导航器时取消待执行任务并等待执行器结束，不遗留跟踪任务。

专项测试覆盖异步非阻塞、超时、异常、立即关闭取消、迟到重排丢弃、缓存命中和关闭后零任务。Stage4隔离回归进一步证明两个场景首次及复跑均3/3缓存命中、后端调用为0，异常代理全部规则回退。

## 6. 两级真实回放

### 6.1 snapshot replay

快照回放只读取既有 `objects_all_frames`，不运行检测、深度或对象融合：

| 场景 | 帧范围/帧数 | 首次目标出现 | 首次确认 | 首次规划 | 规划次数 | 帧末状态分布 | 运行时间 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| scene-0916 | 3-40 / 38 | 3 | 5 | 5 | 11 | SEARCHING 1、PROVISIONAL 4、TRACKING 33 | 5.469 s |
| scene-0003 | 1-39 / 39 | 1 | 3 | 3 | 7 | PROVISIONAL 2、TRACKING 37 | 5.381 s |

两场均满足目标出现前无路径、确认后首次规划、后续跟踪和有审计的重规划。`scene-0916` 实际覆盖目标长期消失、回到搜索并重新确认；`scene-0003` 覆盖目标变化和周期触发拒绝。

### 6.2 end-to-end replay

端到端回放只消费已有本地RGB-D、位姿和缓存检测，每帧执行 `StreamlinedMapper.update(frame)` 后直接把当前对象传给导航器：

| 场景 | 帧数 | 首次目标出现/确认/规划 | 规划次数 | 失败帧/恢复 | 运行时间 |
| --- | ---: | ---: | ---: | ---: | ---: |
| scene-0916 | 41 | 2 / 4 / 4 | 11 | 2 / 1 | 12.137 s |
| scene-0003 | 40 | 0 / 2 / 2 | 12 | 0 / 0 | 14.758 s |

`scene-0916` 的前两帧没有可用对象，分别记录为可恢复的 `EMPTY_OBJECTS`，后续完成目标确认和规划。`scene-0003` 实际覆盖2次持续更优目标切换、1次剩余路径阻断和多次目标几何变化。

最终对象状态与既有最后快照按类别做一对一最小代价匹配。验收同时约束节点数差、逐类别数量差、中心中位误差、4 m内匹配比例和静态类别最大误差：

| 场景 | 实际/参考节点 | 中心中位误差 | 4 m内比例 | 静态类别最大误差 | 动态离群点 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| scene-0916 | 45 / 44 | 1.0839 m | 81.82% | 0.0076 m | 8 | 通过 |
| scene-0003 | 24 / 23 | 0.0052 m | 95.65% | 0.0094 m | 1 | 通过 |

既有轻量快照未保存 `motion_state` 和 `is_short_term`，因此不能伪造参考动态数量。报告保留实际动态数量、全部匹配对、最大误差和动态离群点；`scene-0916` 最大动态关联误差为17.8344 m，`scene-0003` 为4.6672 m。

nuScenes是录制序列。规划路径不会改变后续传感器观测，本阶段不作闭环因果声明。

## 7. 故障注入

`failure_injection/failure_injection_report.json` 实际覆盖：

- 损坏gzip快照被隔离并记录 `BadGzipFile`；
- 空对象返回 `EMPTY_OBJECTS`；
- 非法位姿返回 `FRAME_PROCESSING_EXCEPTION`，已确认状态和路径不被破坏；
- 全占据栅格触发 `path_blocked` 并产生结构化规划失败/`UNREACHABLE`；
- 恢复占据图后继续处理下一帧；
- Qwen异常被审计但不成为导航失败；
- Qwen超时由专项测试验证规则回退。

故障注入总结果为通过。

## 8. 性能与确定性

非规划帧的 `planning_s` 中位数接近0。逐帧总耗时中位数分别为：

- snapshot scene-0916：18.620 ms；scene-0003：10.920 ms。
- end-to-end scene-0916：20.818 ms；scene-0003：13.885 ms。
- 端到端单帧建图中位数分别为72.224 ms和41.975 ms。
- 关系局部更新中位数为3.286-4.478 ms；候选排序中位数为1.413-2.790 ms。

两个模式、两个场景均连续运行2次。排除耗时和异步完成时刻后，状态、目标、触发器、失败码和路径哈希逐组完全相同，四组确定性结果全部通过。内容哈希见 `determinism_report.json`。

## 9. 测试、回归与代码审查

修改前测试基线为110项。最终结果：

```text
conda run -n conceptgraph pytest -q tests
126 passed, 2 warnings in 4.70s
```

两条警告仅来自torchvision Beta API。所有当前改动Python文件Ruff通过，`git diff --check`通过。

兼容回归写入独立的阶段5子目录，没有覆盖旧产物：

- Stage2：两个场景、`first_reachable`和`joint_utility`共4次成功；联合模式保持scene-0916节点21、scene-0003节点11。
- Stage3：两个场景总体成功，增量/同帧全量重建的最小、平均和最终活跃边一致率均为1.0，结构化JSON/PKL均生成。
- Stage4：隔离缓存回归通过；两场首次和复跑均3/3命中、后端0次，规则/Qwen回退选择保持21/11，严格schema有效数仍为0，故障回退通过。

人工代码审查修复了：锁定目标不再作为当前语义候选时的误规划、异常帧的完整内存事务回滚、Qwen查询四版本绑定、端到端地图重复最近邻偏差、动态参考字段缺失的伪成功风险、故障注入缺少恢复帧，以及最终JSON自解析未纳入总验收。未发现剩余阻断问题。

`validation_report.json` 最终为 `passed=true`，并重新解析126个JSON、8个JSONL共1628行，无序列化错误。

## 10. 基线与Git状态

修改前 `baseline_before_stage5.json` 记录了Git HEAD、110项测试、Stage2的21/11、Stage3一致率、Stage4缓存结果，以及阶段1至4产物、旧数据集、checkpoint、MEMORY、PAPER_MEMORY、原文档和两场真实输入指纹。

最终 `baseline_verification.json` 为通过：所有保护文件、保护目录、两场快照和位姿指纹均未变化。回归使用阶段5隔离目录，不写回旧Stage2/3/4输出。

工作树在阶段5开始前已经包含阶段1至4的未提交改动，完成后仍为dirty。没有创建提交或推送，也没有回退或覆盖用户已有改动。

## 11. 复现命令

```bash
conda run -n conceptgraph python -m conceptgraph.scripts.run_incremental_semantic_navigation \
  --mode all --scenes scene-0916 scene-0003 --determinism-runs 2

conda run -n conceptgraph pytest -q tests
```

关键入口报告：

```text
outputs/project_v2/incremental_navigation/stage5_comparison.json
outputs/project_v2/incremental_navigation/state_transition_report.json
outputs/project_v2/incremental_navigation/replan_report.json
outputs/project_v2/incremental_navigation/performance_report.json
outputs/project_v2/incremental_navigation/determinism_report.json
outputs/project_v2/incremental_navigation/regression_report.json
outputs/project_v2/incremental_navigation/baseline_verification.json
outputs/project_v2/incremental_navigation/validation_report.json
```
