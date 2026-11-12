# 阶段3：场景图关系边

## 目标与结果

本阶段将旧的三元组关系升级为可复用的结构化关系边，并完成关系语义、坐标系、置信度、冲突消解、稀疏化、增量更新、导航集成、旧格式兼容和真实场景可视化。实现仅使用已有场景图、对象地图、逐帧对象快照、车辆轨迹和阶段2导航输入，没有运行检测、深度生成、建图或视觉语言模型推理，也没有覆盖 `dataset/nuscenes/scene-*` 下的旧产物。

统一结果位于：

- `outputs/project_v2/scenegraph_relations/stage3_full_rebuild/`
- `outputs/project_v2/scenegraph_relations/stage3_incremental/`
- `outputs/project_v2/scenegraph_relations/stage3_visualizations/`
- `outputs/project_v2/scenegraph_relations/stage3_comparison.json`

## 主要代码

新增的可复用模块包括：

- `conceptgraph/scenegraph/relations.py`：结构化边、坐标系解析、关系构建、反向语义、置信度、冲突、稀疏化和序列化。
- `conceptgraph/scenegraph/incremental_graph.py`：新增、移动、状态变化、失效、帧推进和快照导出的局部更新接口。
- `conceptgraph/scenegraph/visualization.py`：固定关系样式、过滤、导航链识别和 hover 文本。
- `conceptgraph/hydra_configs/scenegraph_relations.yaml`：全部阈值、权重、度限制和可视化样式。
- `conceptgraph/scripts/run_stage3_offline_relations.py`：真实离线输入解析、两种构建、导航查询、统计和可视化调度。
- `conceptgraph/scripts/qa_stage3_visualizations.py`：离屏加载交互 HTML，并自动验证关系过滤、置信度过滤和节点聚焦。

导航侧复用 `SceneGraphEdge` 和统一反向语义 API，修改了 `navigation/io.py`、`navigation/types.py`、`navigation/semantic.py` 和 `navigation/pipeline.py`。三个既有可视化入口均支持结构化边和过滤参数；`build_scenegraph_from_map_stub.py` 保留旧辅助入口，同时转用关系模块生成新边。

## 关系语义

全项目采用“只保存一个规范化事实、查询时解析反向语义”的策略：

- `near` 是对称关系，只保存 `source < target` 的一条无向边；正反向查询均返回 `near`。
- 横向事实只存 `left_of`。查询反向时由 API 将 `A left_of B` 解释为 `B right_of A`。
- 纵向事实只存 `in_front_of`。查询反向时由 API 将 `A in_front_of B` 解释为 `B behind A`。
- 有向边不会被写入无条件双向索引；导航只在调用反向语义 API 后获得反向事实。
- self-loop 以 `self_loop` 原因拒绝；重复边按规范键合并，并累加观测次数及有效帧。
- 同一无序节点对、同一方向轴上的相反事实进入同一冲突组，只保留一个确定性胜者。

所有边按关系顺序、源节点和目标节点稳定排序。相同输入不会依赖字典遍历顺序。

## 坐标系

坐标变换集中在 `relations.py`：

- `world`：世界 `+Y` 为前方，`-X` 为左侧，`+X` 为右侧。
- `ego_heading`：归一化车辆航向为前轴，左轴为 `(-heading_y, heading_x)`。
- `local_trajectory`：取离自车当前位置最近的轨迹点，用相邻有效轨迹点形成的局部切线作为前轴，左轴同样逆时针旋转 90 度。

默认请求 `local_trajectory`，退化时按配置回退到 `ego_heading`。若回退坐标也不可用，构建结果返回结构化失败码和诊断，不用异常字符串表达正常业务失败。每条边保存实际 `coordinate_frame`、请求坐标系和 `coordinate_frame_fallback_reason`。

## 置信度、冲突与稀疏化

置信度为六个归一化分量的加权平均：

`confidence = sum(weight_i * component_i) / sum(weight_i)`

默认权重为距离 0.30、方向角裕量 0.20、观测次数 0.15、跨帧稳定性 0.15、坐标系可靠性 0.10、动态状态 0.10。具体分量为：

- 距离：`max(0, 1 - distance / relation_threshold)`。
- 角裕量：方向偏角离 40 度阈值越远分数越高；`near` 固定为 1。
- 观测次数：相对 `static_stability_frames=4` 截断到 `[0, 1]`。
- 稳定性：联合有效帧跨度和观测次数并截断到 `[0, 1]`。
- 坐标系：实际坐标系及回退链提供可靠性分数。
- 动态状态：静态为 1，任一端动态时为 0.70。

最低置信度为 0.35。距离阈值默认为 `near=8 m`、方向关系 `12 m`，方向可判定的最小距离为 0.5 m。

冲突胜者依次按置信度、稳定性、距离及稳定键排序；超过 `conflict_margin=0.03` 记为低置信度淘汰，否则记为确定性 tie-break 淘汰。被删除边保存胜者、置信度差和结构化拒绝原因。

稀疏化先按关系类型分别执行 Top-K：`near=8`，各方向关系为 6；随后限制每节点最大出度 16、最大入度 16，最后限制全图最大 512 条边。`near` 对两个端点同时计数，方向边按真实出入方向计数。每个阶段都保存裁剪前后数量及逐边拒绝原因。

## 增量更新

`IncrementalSceneGraph` 维护规范候选边和最终边快照。每次更新只把以下节点放入受影响集合：

- 新增节点；
- 位移严格超过 0.10 m 的节点；
- 动静态状态变化的节点；
- 失效或删除节点。

仅重算至少包含一个受影响节点的节点对，其他候选边复用。静态边在重复观测时累加 `observation_count` 和稳定性；动态边维护 `first_seen_frame`、`last_seen_frame` 和有效帧范围，超过 3 帧未更新后失效。每帧结果包含受影响节点、重算节点对、重算边、复用边、过期动态边和耗时。

真实逐帧回放固定使用 `world` 坐标系，原因是车辆航向变化会让所有既有方向事实同时改变，已不再是局部依赖；同帧全量对照使用相同坐标系。最终场景全量图仍按默认 `local_trajectory` 构建。两场景所有有效帧的增量/同帧全量一致率均为 1.0。

## 旧格式与导航

加载器支持：

- 旧 `(source, relation, target)` 三元组 pickle；
- 新 `SceneGraphEdge` 列表 pickle；
- 新结构化 JSON；
- 带 `edges` 字段的结构化构建结果。

旧三元组会补齐兼容默认值，并在元数据中标记 `edge_format=legacy`、`metadata_available=false`。结构化边也可另存为旧三元组 pickle，供原入口继续读取。

导航评分优先读取结构化边的方向、置信度、距离、坐标系、观测次数和有效帧；`near` 对称查询，有向反向查询通过统一 API 解析。只有旧边缺少元数据、结构化边缺失或低于最低置信度时才计算实时几何关系。输出保存实际关系链、存储事实方向、边来源和 fallback 原因。阶段2的 `first_reachable` 与 `joint_utility` 回归均在两个场景成功，选择结果与阶段2记录一致。

## 真实场景结果

| 场景 | 帧数 | 节点 | 旧边 | 新边 | 新关系分布 | 对称/有向 | 增量重算/复用 | 最低/平均/最终一致率 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| scene-0916 | 38（3-40） | 44 | 20 | 63 | near 20, left_of 18, in_front_of 25 | 20/43 | 185/977 | 1.0/1.0/1.0 |
| scene-0003 | 39（1-39） | 23 | 21 | 59 | near 25, left_of 15, in_front_of 19 | 25/34 | 233/759 | 1.0/1.0/1.0 |

真实全量构建没有出现冲突或触发裁剪；这表示两段数据在当前阈值下没有冲突候选或超限节点，不表示相关逻辑未执行。冲突、关系 Top-K、最大出入度和总边数均由合成测试覆盖。

导航对比：

- `scene-0916`：`black traffic sign in front of green traffic sign` 的旧/新选择均为节点 5；反向查询选择节点 3。新导航直接使用置信度 0.6125 的 `5 in_front_of 3` 存储事实，两个查询均无几何 fallback。
- `scene-0003`：`black car in front of gray bus` 的旧/新选择均为节点 11，新导航使用置信度 0.6988 的 `11 in_front_of 14`。反向查询 `gray bus behind black car` 中，旧边无法选择，新结构化关系选择节点 14，因此记录为选择变化。两个新查询均无几何 fallback。

## 可视化验收

每个场景均生成 BEV 静态 PNG、3D 静态 PNG、可交互 HTML、导航关系链高亮 PNG，以及全量/增量统计差异图。固定样式为：`near` 灰色实线，`left_of` 蓝色点线箭头，`in_front_of` 绿色虚线箭头；边宽和透明度映射置信度。

交互 HTML 支持关系复选框、最低置信度滑块、节点点击局部聚焦和完整边 hover 元数据。离屏 Qt 环境没有可用 WebGL，因此页面自动切换到使用同一真实数据的 SVG/BEV 交互回退图；两场景均验证加载、关系过滤、置信度过滤和节点聚焦有效，页面没有显示 WebGL 错误。正常浏览器有 WebGL 时仍优先显示 Plotly 3D。

自动验收记录：

- `outputs/project_v2/scenegraph_relations/visualization_qa.json`
- `outputs/project_v2/scenegraph_relations/validation_report.json`

## 测试与确定性

- 场景图专项测试：34 passed。
- `tests` 全量测试：59 passed。
- 目标文件 Ruff：通过。
- `git diff --check`：通过。
- 新旧 JSON/PKL：解析与往返通过。
- 阶段2双模式、双场景回归：4 次运行全部成功。

连续两次真实实验的两个结构化 JSON、两个结构化 PKL 均为字节级一致。剔除耗时和内容哈希字段后的总报告内容哈希为：

`4772f00ab3fb8f6f04d9475bc8196bb0ed4f14e8c71e2fa235ff43b530612391`

详细边文件哈希见 `outputs/project_v2/scenegraph_relations/determinism_report.json`。

## 基线保护与限制

修改前基线见 `baseline_before_stage3.json`，完成后核验见 `baseline_verification.json`。全部 `exp_qwen2vl_semantics_v1`、全部 `exp_streamlined_yolo_outdoor_objects_v1`、阶段1、阶段2、`MEMORY.md`、`PAPER_MEMORY.md`、原有 5 个文档和两个场景 10 个旧可视化文件的指纹均未变化。新增阶段3文档会按预期改变 `doc/` 的聚合指纹，因此核验按“原文件逐个不变”判断。

数据限制如下：

- `scene-0916` 的可用快照从第 3 帧开始，`scene-0003` 从第 1 帧开始，报告只统计磁盘上真实存在的连续快照。
- 离屏验收机缺少可用 WebGL，已验证功能等价的自动 SVG/BEV 回退；没有阻塞实际 HTML 交互。
- 真实场景未触发冲突和裁剪，相关行为由确定性合成测试验证。

没有遗留的功能性阻塞问题。本阶段未提交或推送 Git，也未修改阶段4至阶段6业务逻辑。
