# 项目统一分阶段优化方案

## 1. 范围与原则

当前只处理项目代码、数据处理、模型训练、运行结果和测试，不包含论文写作工作。

实施原则：

- 当前稳定代码作为离线基线，不直接覆盖；
- 优先复用已有逐帧地图结果验证增量导航；
- 每个阶段都保留旧模式和兼容接口；
- 每个阶段都有独立测试和独立输出目录；
- 先完成可独立验证的模块，再接入逐帧主流程；
- Qwen异常不能阻断建图和导航规则主链路；
- nuScenes用于录制序列上的增量处理和路径规划，不模拟路径改变后续传感器观测。

## 2. 当前基线

- Git分支：`main`
- 基线提交：`eab10f5`
- 本地标签：`v1.0-offline-baseline`
- 远端 `origin/main` 已包含基线提交；标签因当前环境缺少GitHub凭据尚未推送。
- 未跟踪的 `MEMORY.md` 和 `PAPER_MEMORY.md` 不纳入代码版本。

现有结果继续保留：

```text
outputs/experiment_summary/
outputs/qwen_finetune_dataset/
checkpoint/qwen_lora_semantic_delegate/
dataset/nuscenes/scene-*/exp_qwen2vl_semantics_v1/
dataset/nuscenes/scene-*/objects_all_frames/
```

后续统一写入：

```text
outputs/project_v2/
dataset/nuscenes/scene-*/exp_project_v2_*/
```

## 3. 总体阶段

项目修改分为六个阶段。

| 阶段 | 模块 | 主要结果 |
|---|---|---|
| 1 | 工程基础与数据工具 | 测试骨架、输出隔离、停车场场景manifest |
| 2 | 目标选择与路径规划 | Top-K候选联合选择和严格可达点约束 |
| 3 | 场景图关系边 | 有向、稳定、稀疏且可增量更新的关系边 |
| 4 | Qwen语义模块 | 多模态数据、结构化输出、缓存和规则回退 |
| 5 | 增量导航 | 指令先解析、逐帧目标搜索和事件触发重规划 |
| 6 | 集成与回归 | 完整链路、性能统计、失败诊断和结果整理 |

推荐执行顺序为阶段1至阶段6。阶段2、3、4先在离线模式中验证，阶段5再将三个模块接入逐帧流程。

### 3.1 阶段变更记录

每个阶段完成验收后，必须在 `doc/change_logs/` 中新增或更新一份阶段变更记录。记录是阶段完成条件的一部分，不能只保留在对话或终端输出中。

变更记录至少包含：

- 实施范围与明确未修改的范围；
- 新增、修改文件及接口或配置变化；
- 核心算法、兼容行为和确定性约束；
- 单元测试、真实数据运行和代码审查结果；
- 生成产物的绝对路径及可复现命令；
- 已知数据限制、未解决问题和Git状态；
- 现有基线结果未被覆盖的验证依据。

阶段1记录：

```text
doc/change_logs/stage1_engineering_foundation_and_data_tools.md
```

阶段2完成后必须生成：

```text
doc/change_logs/stage2_target_selection_and_path_planning.md
```

## 4. 阶段1：工程基础与数据工具

### 4.1 目标

建立后续修改的测试和输出边界，同时把当前停车场场景关键词筛选升级为结构化场景评分工具。

### 4.2 主要修改

新增测试目录：

```text
tests/navigation/
tests/scenegraph/
tests/vlm/
tests/nuscenes/
```

新增输出目录约定：

```text
outputs/project_v2/tests/
outputs/project_v2/subset/
outputs/project_v2/offline_navigation/
outputs/project_v2/incremental_navigation/
outputs/project_v2/qwen_eval/
outputs/project_v2/performance/
```

数据工具涉及：

```text
conceptgraph/scripts/select_nuscenes_parking_scenes.py
conceptgraph/scripts/extract_nuscenes_scene_blobs.py
conceptgraph/scripts/preprocess_nuscenes_to_minimal.py
```

新增建议：

```text
conceptgraph/nuscenes/scene_selector.py
conceptgraph/scripts/build_nuscenes_subset_manifest.py
tests/nuscenes/test_scene_selector.py
```

结构化场景评分读取：

```text
scene.json
sample.json
sample_annotation.json
instance.json
attribute.json
category.json
```

评分内容：

- `vehicle.parked` 数量和比例；
- `cycle.without_rider` 数量；
- barrier、traffic cone和bicycle rack数量；
- 静态目标持续帧数；
- 动态目标比例；
- CAM_FRONT与LIDAR_TOP文件完整率；
- 轨迹长度和有效帧数。

脚本增加 `--dry-run`、缺失文件报告、处理后完整性检查和确定性排序。

### 4.3 阶段输出

```text
outputs/project_v2/subset/scene_scores.json
outputs/project_v2/subset/selected_scenes.json
outputs/project_v2/subset/integrity_report.json
```

### 4.4 完成条件

- 新测试目录可以独立运行；
- 新输出不会覆盖当前基线；
- 场景排序在相同输入下完全一致；
- 场景manifest中的帧数与实际最小数据目录一致；
- 缺失数据时返回明确错误，不生成不完整场景。

## 5. 阶段2：目标选择与路径规划

### 5.1 当前行为

当前系统先按语义分数排列节点，再从高到低检查候选。对象包围盒中心作为语义目标位置，规划器在其附近寻找起点连通的可行驶栅格。找到第一个能够生成A*路径的候选后立即停止。

主要问题：

- 没有比较Top-K候选各自的路径质量；
- 全局fallback可能距离对象过远；
- 可达点主要参考中心点，没有利用对象包围盒边界；
- 目标选择失败主要通过异常字符串表达；
- 平滑路径缺少完整的连续碰撞复查。

### 5.2 涉及文件

```text
conceptgraph/navigation/types.py
conceptgraph/navigation/semantic.py
conceptgraph/navigation/planner.py
conceptgraph/navigation/pipeline.py
conceptgraph/hydra_configs/semantic_navigation.yaml
```

新增建议：

```text
conceptgraph/navigation/target_selector.py
tests/navigation/test_target_selector.py
tests/navigation/test_reachable_goal.py
tests/navigation/test_planner.py
```

### 5.3 数据结构修改

扩展 `CandidateTarget`：

```text
semantic_rank
goal_candidates
path_cost
path_length_m
min_clearance_m
fallback_distance_m
selection_utility
rejection_reason
```

新增 `ReachableGoalCandidate`：

```text
goal_xy
goal_rc
distance_to_object_m
clearance_m
connected_to_start
fallback_reason
```

### 5.4 算法修改

- 对Top-K语义候选全部执行可达点搜索；
- 以对象二维包围盒边界和安全环带生成可达点，而不是只使用中心点；
- 按类别配置合理接近距离；
- 默认禁止无限制全局fallback；
- 对每个候选运行A*并记录路径指标；
- 联合语义、关系、路径代价、clearance、fallback距离和动态风险选择最终目标；
- 所有候选失败时返回结构化失败对象；
- 保留 `first_reachable` 模式复现当前基线。

联合选择分数：

```text
utility = semantic_score
        + relation_score
        - path_cost
        + clearance_score
        - fallback_penalty
        - dynamic_risk
```

实际权重全部写入配置并保存到结果JSON。

### 5.5 完成条件

- 目标点位于起点连通的可行驶区域；
- 目标点与对象包围盒满足安全距离约束；
- fallback超过阈值时明确拒绝该候选；
- 平滑路径经过连续栅格碰撞检查；
- 每个Top-K候选都有完整诊断；
- 旧的 `first_reachable` 模式仍能运行；
- `doc/change_logs/stage2_target_selection_and_path_planning.md` 已按真实代码、测试和输出完成记录。

## 6. 阶段3：场景图关系边

### 6.1 当前问题

当前关系边在最终场景图生成时一次性计算，方向依赖世界坐标轴，边结构只保留基本三元组；交互图没有充分使用边置信度和稳定性信息。

### 6.2 涉及文件

```text
conceptgraph/scripts/build_scenegraph_from_map_stub.py
conceptgraph/navigation/io.py
conceptgraph/navigation/types.py
conceptgraph/navigation/semantic.py
conceptgraph/scripts/visualize_scenegraph.py
conceptgraph/scripts/visualize_scenegraph_3d.py
conceptgraph/scripts/visualize_scenegraph_3d_interactive.py
```

新增建议：

```text
conceptgraph/scenegraph/relations.py
conceptgraph/scenegraph/incremental_graph.py
tests/scenegraph/test_relations.py
tests/scenegraph/test_edge_pruning.py
```

### 6.3 边结构

扩展 `SceneGraphEdge`：

```text
source
target
relation
directed
confidence
distance_m
coordinate_frame
relation_type
first_seen_frame
last_seen_frame
observation_count
```

### 6.4 算法修改

- `near`使用对称关系；
- `left_of/right_of/in_front_of/behind`使用有向关系；
- 支持世界坐标、自车航向和局部轨迹坐标；
- 检测并清除互相矛盾的方向关系；
- 分关系类型执行Top-K稀疏化；
- 静态边累积稳定性，动态边记录有效帧范围；
- 增量模式只重算新增或发生位置变化节点周围的边；
- 导航优先使用关系边元数据，实时几何计算作为fallback；
- 继续兼容当前三元组pickle读取。

### 6.5 可视化修改

- 关系类型映射颜色和线型；
- 置信度映射边宽与透明度；
- 有向关系显示箭头；
- 支持类型和置信度过滤；
- 点击节点时只突出局部邻接边；
- hover显示距离、置信度、坐标系和稳定性；
- 导航模式只突出当前查询实际使用的关系链。

### 6.6 完成条件

- 方向关系的正向和反向定义一致；
- 不同时保留互相矛盾的高置信度边；
- 节点度和总边数受配置约束；
- 新旧边格式均可加载；
- 增量边更新与同帧全量重建在容差内一致；
- 可视化能够隐藏无关边并查看边详细信息。

## 7. 阶段4：Qwen语义模块

### 7.1 当前问题

当前已经有Qwen delegate和LoRA训练脚本，但导航主流程没有真正使用delegate。对象属性微调主要输入文本元数据，没有真正使用对象裁剪图；训练和验证随机混合样本，可能包含同场景数据泄漏。

### 7.2 涉及文件

```text
conceptgraph/vlm/qwen_semantic_delegate.py
conceptgraph/vlm/qwen_vl_chat.py
conceptgraph/scripts/build_qwen_finetune_dataset.py
scripts/train_qwen_lora_semantic_delegate.py
conceptgraph/navigation/semantic.py
conceptgraph/navigation/pipeline.py
```

新增建议：

```text
conceptgraph/vlm/schemas.py
conceptgraph/vlm/qwen_cache.py
tests/vlm/test_qwen_schemas.py
tests/vlm/test_qwen_fallback.py
```

### 7.3 修改内容

- 为指令解析、对象属性和候选重排定义明确JSON schema；
- 统一记录原始输出、解析结果、耗时和回退原因；
- 对象属性训练真正加载对象crop图像；
- 按 `scene_id` 划分训练、验证和测试；
- 支持分任务训练和任务采样权重；
- 导航开始时调用一次 `parse_query`；
- 候选集合显著变化时调用 `rerank_candidates`；
- Qwen失败、超时或格式错误时使用规则结果；
- 相同指令和相同对象版本启用磁盘缓存；
- 增量模式使用异步调用，禁止Qwen阻塞逐帧主循环。

### 7.4 完成条件

- 不启用Qwen时与规则基线兼容；
- schema错误和推理异常能稳定回退；
- 缓存命中时不重复推理；
- Qwen异常不会终止建图和规划；
- 训练、验证和测试不存在scene泄漏；
- 评估结果按任务和场景分别输出。

## 8. 阶段5：增量导航

### 8.1 当前问题

建图代码已经逐帧融合对象并能保存 `objects_all_frames`，但导航代码只读取最终场景图文件并缓存节点、边和占据地图。当前缺少建图状态直接传给导航的内存接口。

### 8.2 新增模块

```text
conceptgraph/navigation/online.py
conceptgraph/navigation/state_machine.py
conceptgraph/navigation/replan_policy.py
conceptgraph/scripts/run_incremental_semantic_navigation.py
tests/navigation/test_online_state_machine.py
tests/navigation/test_replan_policy.py
```

### 8.3 核心接口

```python
class OnlineSemanticNavigator:
    def set_query(self, query_text: str) -> None:
        ...

    def update(
        self,
        frame_idx: int,
        vehicle_pose,
        objects,
        changed_object_ids,
        occupancy_update=None,
    ) -> "OnlineNavigationUpdate":
        ...
```

输出包含：

```text
frame_idx
state
candidate_ids
selected_node_id
target_confidence
path_world
replanned
replan_reason
timings
failure
```

### 8.4 状态机

```text
SEARCHING
PROVISIONAL
LOCKED
PLANNING
TRACKING
REPLANNING
UNREACHABLE
```

行为约束：

- 目标未出现时保持SEARCHING，不生成虚假路径；
- 候选连续满足观测数和置信度阈值后才LOCKED；
- 首次LOCKED后生成路径；
- 后续帧继续检查剩余路径和目标状态；
- 短暂候选抖动不能切换已锁定目标；
- 路径阻断、目标变化、车辆移动或周期事件触发重规划；
- 相同状态和相同帧不能重复规划。

### 8.5 两步实现

第一步实现 `snapshot_replay`：

- 读取已有 `objects_all_frames`；
- 不重新运行检测和对象融合；
- 验证目标搜索、锁定、状态机和重规划；
- 快速发现在线架构问题。

第二步实现 `end_to_end_replay`：

- 将 `streamlined_mapping.py` 的单帧逻辑拆成 `mapper.update(frame)`；
- 每帧融合后直接传递对象状态和变化ID；
- 局部更新关系边；
- Qwen异步处理稳定节点和候选变化；
- 保存逐帧轻量状态，只有重规划帧保存完整路径和图片。

### 8.6 重规划触发器

```text
target_confirmed
target_changed
better_target_persisted
path_blocked
vehicle_moved
periodic_interval
manual_request
```

### 8.7 完成条件

- snapshot replay可从首帧运行到末帧；
- 目标首次出现前没有规划路径；
- 目标确认后生成第一条有效路径；
- 后续地图变化能触发或拒绝重规划并给出原因；
- 任意单帧异常被记录后可以继续处理；
- 固定输入产生确定性状态序列和路径结果。

## 9. 阶段6：集成与回归

### 9.1 单元测试

```text
指令解析
候选过滤和评分
Top-K目标选择
目标附近可达点
A*与路径平滑
关系方向和边稀疏化
Qwen schema、缓存与fallback
在线状态转换
重规划触发器
```

### 9.2 集成测试

```text
旧完整场景图离线导航
新规划器离线导航
objects_all_frames增量回放
完整逐帧建图与增量导航
旧关系边格式兼容
Qwen关闭、开启和异常回退
```

### 9.3 运行时指标

```text
detection_time_s
mapping_time_s
relation_update_time_s
query_ranking_time_s
qwen_time_s
planning_time_s
total_frame_time_s
peak_gpu_memory_mb
```

### 9.4 最终完成条件

- 当前离线模式可继续运行；
- 新规划器输出Top-K完整诊断；
- 新关系边同时服务导航和可视化；
- Qwen具备超时、缓存和规则回退；
- 增量模式具备稳定状态机和事件触发重规划；
- 所有新增测试通过；
- 所有新结果写入 `project_v2` 目录；
- 基线代码和结果未被覆盖。

## 10. 实际开发顺序

1. 建立测试目录、公共fixture和输出隔离检查。
2. 使用现有 `objects_all_frames` 做增量导航最小原型，验证状态机接口。
3. 完成Top-K目标选择和规划器修改，并接入离线及增量模式。
4. 完成关系边新结构、全量构建和增量更新。
5. 完成Qwen schema、图像训练数据、缓存、回退和主流程接入。
6. 完成结构化场景筛选和新增场景处理。
7. 重构建图单帧接口，完成end-to-end replay。
8. 运行完整回归和性能统计，整理项目最终结果。

先做第2步的增量导航最小原型，是为了尽早验证“指令先给出、建图过程中搜索目标并重规划”能否在当前逐帧地图数据上成立；验证通过后再投入规划器、关系边和Qwen的较大改造。
