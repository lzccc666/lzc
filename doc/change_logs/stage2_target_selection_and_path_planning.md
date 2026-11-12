# 阶段2变更记录：目标选择与路径规划

## 1. 实施范围

本阶段完整实现了 `doc/project_optimization_plan.md` 中“阶段2：目标选择与路径规划”。实现仅复用现有离线场景图、对象地图、BEV占据栅格、车辆位姿和旧导航结果，没有运行目标检测、深度生成、建图、Qwen训练或推理、全量导航及nuScenes解压。

未修改阶段3至阶段6业务逻辑，未处理论文内容，未提交或推送Git，也未覆盖任何现有实验结果。新增真实运行产物全部位于：

```text
/root/autodl-tmp/lizichong/outputs/project_v2/offline_navigation
```

## 2. 新增与修改文件

新增：

```text
conceptgraph/navigation/target_selector.py
conceptgraph/scripts/run_stage2_offline_navigation.py
tests/__init__.py
tests/navigation/__init__.py
tests/navigation/conftest.py
tests/navigation/test_target_selector.py
tests/navigation/test_reachable_goal.py
tests/navigation/test_planner.py
doc/change_logs/stage2_target_selection_and_path_planning.md
```

修改：

```text
conceptgraph/hydra_configs/semantic_navigation.yaml
conceptgraph/navigation/__init__.py
conceptgraph/navigation/io.py
conceptgraph/navigation/pipeline.py
conceptgraph/navigation/planner.py
conceptgraph/navigation/semantic.py
conceptgraph/navigation/types.py
```

主要接口变化：

- `CandidateTarget` 增加语义排名、可达目标列表、路径指标、clearance、fallback、联合效用、动态风险、评分分项和拒绝原因。
- 新增 `ReachableGoalCandidate`，保存理想环带点、实际目标点、连通性、clearance、A*路径、fallback、碰撞复查和拒绝诊断。
- 新增 `TargetSelectionResult`，统一返回 `success`、`mode`、`selected_candidate`、`evaluated_candidates`、`failure_code`、`failure_reason`、`scoring_weights` 和 `timing`。
- `SemanticNavigationPipeline.select()` 是结构化成功/失败接口；原 `run()` 继续在成功时返回 `NavigationPlan`，旧调用入口无需修改。
- `save_selection_result()` 可保存成功或失败结果；保存的失败JSON可直接重新解析。

## 3. 候选目标生成规则

`joint_utility` 对Top-K语义对象执行以下确定性流程：

1. 校验二维包围盒中心和尺寸；字段缺失返回 `missing_bbox`，非有限值或非正尺寸返回 `degenerate_bbox`。
2. 从轴对齐二维包围盒中心向固定角度发射射线，求射线与包围盒边界的交点，再沿射线外移类别相关接近距离，形成安全环带理想点。
3. 将理想点转换到栅格；检查边界、障碍物及膨胀区域、起点连通分量、类别最小clearance和对象安全距离。
4. 直接点无效时，仅在类别 `max_fallback_distance_m` 内搜索满足全部约束的起点连通栅格。即使启用 `allow_global_fallback`，硬性fallback上限也不会取消，因此不存在无限制全局fallback。
5. 按fallback距离、clearance、行列坐标进行稳定排序，对实际栅格目标去重；重复点记录 `duplicate_goal`。
6. 每个未被前置约束拒绝的目标点都执行现有A*；路径平滑后对每一段经过的连续栅格重新做碰撞检查，不通过则拒绝。
7. 保存路径代价、物理长度、整条路径最小clearance、fallback距离、扩展节点数和结构化拒绝原因。

类别接近距离、clearance、候选数和fallback上限均在 `semantic_navigation.yaml` 的 `target_selection` 下配置。配置同时接受类别映射或标量缺省值。

## 4. 联合效用与确定性

联合模式在所有可行“对象-目标点”组合上计算：

```text
utility = w_semantic  * semantic_score_normalized
        + w_relation  * relation_score_normalized
        - w_path      * path_cost_normalized
        + w_clearance * clearance_score_normalized
        - w_fallback  * fallback_penalty_normalized
        - w_dynamic   * dynamic_risk_normalized
```

真实运行使用的权重为：

```json
{
  "semantic_score": 0.35,
  "relation_score": 0.15,
  "path_cost": 0.25,
  "clearance_score": 0.10,
  "fallback_penalty": 0.10,
  "dynamic_risk": 0.05
}
```

每个Top-K候选保存 `raw`、`normalized` 和 `weighted` 三组评分分项。动态风险由类别惩罚、动态运动标记和短期目标标记共同产生。对象内部先按效用选择目标点，再在对象间选择最高效用候选。

语义候选按“分数降序、node_id升序”排序；目标点和最终对象使用显式数值舍入及行列坐标作为稳定平局规则。重复运行排除耗时字段后，13个阶段2 JSON文件的聚合SHA-256均为：

```text
c63501496764429afed0fab63db1aa2a4c5992258afd4714c7f040c7499132e5
```

## 5. first_reachable兼容方式

旧配置没有 `target_selection` 时自动解析为 `first_reachable`。该模式保留原语义候选顺序、对象包围盒中心目标、旧 `goal_search_radius_m`、旧 `allow_goal_fallback` 和原A*代价配置；找到第一个成功路径后立即停止，后续候选不会进入 `evaluated_candidates`。

修改前和修改后的代表性真实结果：

| 场景 | 选择节点 | 目标点 | A*代价 | 修改前路径长度 | 修改后路径长度 |
|---|---:|---|---:|---:|---:|
| scene-0916 | 24 | [683.9319, 1801.2654] | 56.2042 | 19.5795 m | 19.0676 m |
| scene-0003 | 9 | [266.0712, 912.4029] | 43.9210 | 15.7626 m | 15.7626 m |

两处场景的对象、目标点、语义排名、A*代价和fallback诊断均保持一致。`scene-0916` 的路径长度变化来自新增连续碰撞复查：旧平滑折线未通过复查，系统回退到已验证无碰撞的捷径路径。这是安全约束修正，不是目标选择变化。

## 6. 测试与代码审查

在Conda环境 `conceptgraph` 中执行：

```bash
conda run -n conceptgraph pytest -q tests/navigation
conda run -n conceptgraph pytest -q tests
conda run -n conceptgraph ruff check conceptgraph/navigation conceptgraph/scripts/run_stage2_offline_navigation.py tests/navigation
git diff --check
```

最终结果：

- 导航专项测试：17项通过。
- `tests` 全量测试：25项通过。
- Ruff静态检查：通过。
- `git diff --check`：通过。
- 新旧配置加载：通过。
- 结构化失败JSON往返解析：通过。
- 真实结果确定性复跑：通过。

测试覆盖包围盒安全环带、边界和障碍物、起点连通性、clearance、fallback硬上限、A*不可达、路径质量联合选择、动态风险、连续碰撞复查、全部失败、`first_reachable`停止行为、确定性、缺失/退化bbox、空场景图、无可行驶区域和字典形式bbox加载。

代码审查中发现并修复：

- `first_reachable` 曾把未实际评估的候选写入 `evaluated_candidates`。
- 真实加载节点缺失bbox字段时，诊断曾落为 `degenerate_bbox`。
- 字典形式的 `bbox.center` 和 `bbox.extent` 未被正确读取。
- 标量类别配置曾被错误替换为默认值。
- 零代价路径在平局排序中曾被当作无穷大。
- 两处无效导入/局部变量触发静态检查。

## 7. 真实离线运行结果

复现命令：

```bash
cd /root/autodl-tmp/lizichong
conda run -n conceptgraph python -m conceptgraph.scripts.run_stage2_offline_navigation --scene-count 2
```

| 场景 | 模式 | 成功 | 节点/语义排名 | 路径代价 | 路径长度 | 最小clearance | fallback | 联合效用 |
|---|---|---|---|---:|---:|---:|---:|---:|
| scene-0916 | first_reachable | 是 | 24 / 1 | 56.2042 | 19.0676 m | 0.5 m | 2.0386 m | 0.2815 |
| scene-0916 | joint_utility | 是 | 21 / 4 | 2.6405 | 1.1180 m | 3.0 m | 0.2563 m | 0.5658 |
| scene-0003 | first_reachable | 是 | 9 / 1 | 43.9210 | 15.7626 m | 0.5 m | 2.0013 m | 0.2825 |
| scene-0003 | joint_utility | 是 | 11 / 3 | 30.1688 | 11.7047 m | 2.0 m | 0.2971 m | 0.5264 |

两个场景的联合模式都发生选择变化。`scene-0916` 在完整评估Top-5后选择语义第4名，获得更低路径代价、更高clearance和更小fallback；`scene-0003` 的语义过滤后只有3个候选，联合模式完整评估3个并选择第3名。

结果文件：

```text
/root/autodl-tmp/lizichong/outputs/project_v2/offline_navigation/stage2_first_reachable/
/root/autodl-tmp/lizichong/outputs/project_v2/offline_navigation/stage2_joint_utility/
/root/autodl-tmp/lizichong/outputs/project_v2/offline_navigation/stage2_comparison.json
```

每个场景目录包含 `target_selection_result.json`、`semantic_navigation_plan.json` 和 `resolved_config.json`。

## 8. 基线指纹核验

修改前后指纹完全一致：

| 基线范围 | SHA-256 |
|---|---|
| 全部 `exp_qwen2vl_semantics_v1` 文件 | `658ff381d98d0791781de0664189be10f8d9249b0b30f82945ff5935656d0456` |
| 全部 `exp_streamlined_yolo_outdoor_objects_v1` 文件 | `eb29d2a83865a245672e99bb84165942e312eb16a2bcd76e269c8c9829a8f0ff` |
| `outputs/experiment_summary` 与 `outputs/qwen_finetune_dataset` | `1ff02e537e7272c8b2db05652232d7cc41c04cf3b713fd9dbb91c682fb52ccc5` |
| `MEMORY.md` | `f1db9a84f9b586fff70f61d09cd3bdda78d4b2cac953fbde8af45533ed034fd6` |
| `PAPER_MEMORY.md` | `2a978751be1efd888a130444bc31f146666c7f4fcb62e37c2f92aa2fe196f2b2` |
| 阶段1 `scene_scores.json` | `c697df2ca95dacf4d03b93756231fa0f2e979529c1b90c1cc98ab8686e6fb19b` |
| 阶段1 `selected_scenes.json` | `aded1666a4fa94148b3f7223de17e58a7f5ca698fce6b8826b228d9ff20b7681` |
| 阶段1 `integrity_report.json` | `45aeb181e33658c7212255ec3f78d957fafcb24a6183ba19d21c732098ba2260` |

## 9. 数据限制与Git状态

- 阶段1清单中的 `scene-1094` 当前没有对应本地 `dataset/nuscenes/scene-1094` 目录；运行器会记录缺失输入并继续检查其他入选场景。本次排名前两位 `scene-0916` 和 `scene-0003` 输入完整，因此满足至少两个真实场景的验收要求。
- 现有BEV地图是旧实验生成的离线占据栅格；本阶段只读取并叠加已有场景对象障碍，不重新建图。
- 安全环带基于场景图提供的轴对齐二维包围盒；若上游未来提供朝向包围盒，可在目标生成模块内扩展，不影响流水线接口。
- 动态风险依赖现有 `motion_state`、`is_potential_dynamic` 和 `is_short_term` 元数据，缺失时仅使用类别缺省惩罚。

当前分支仍为 `main`。工作树在开始前已有阶段1和用户文件的未提交改动；这些改动全部保留，未回滚、未删除。阶段2新增/修改也保持未提交状态，没有创建提交或推送。
