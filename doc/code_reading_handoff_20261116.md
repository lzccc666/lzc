# 代码阅读交接记录（2026-11-16）

## 本轮阅读范围

已进入远程项目 `/root/autodl-tmp/lizichong`，建立了源码清单，并优先阅读论文相关核心模块：

- `conceptgraph/slam/streamlined_mapping.py`
- `conceptgraph/scenegraph/incremental_graph.py`
- `conceptgraph/scenegraph/relations.py`
- `conceptgraph/vlm/schemas.py`
- `conceptgraph/vlm/qwen_semantic_delegate.py`
- `conceptgraph/navigation/target_selector.py`
- `conceptgraph/navigation/planner.py`
- `conceptgraph/navigation/vehicle_geometry.py`
- `conceptgraph/completion/frontend.py`
- `conceptgraph/completion/semantic_contract.py`
- `conceptgraph/completion/semantic_evaluation.py`
- `conceptgraph/navigation/pipeline.py`
- 相关测试目录和实验脚本索引

项目的 `conceptgraph`、`scripts`、`tests` 中共有约320个项目相关 Python/ shell 文件；`Grounded-Segment-Anything`、`gradslam`、`chamferdist` 等目录属于第三方或外部依赖，不能直接作为论文原创实现统计。

## 已确认的代码实现

1. 建图入口将逐帧图像、深度、内参、位姿、检测结果和时间戳组织为 `MappingFrame`，通过 `StreamlinedMapper.update()` 进行检测转三维、对象融合、变化对象识别和动态短期对象 TTL 清理。时间戳要求严格递增。
2. 关系图由 `IncrementalSceneGraph` 维护，只对新增、移动、动态状态变化和删除节点影响的邻域重算；关系配置包括方向阈值、距离阈值、置信度、静态稳定帧数、动态边 TTL、入/出度和总边数限制。
3. 关系边明确区分对称 `near` 和有向 `left_of/right_of/in_front_of/behind`，附带坐标系、时间、观测次数、距离和置信度；关系可拒绝、过期、冲突处理并保留统计信息。
4. Qwen 接口使用严格 Pydantic schema，分别处理 `query_parse`、`object_attributes`、`candidate_rerank`；模型调用具备确定性生成、磁盘缓存、图像哈希、超时、异步执行、失败日志和规则回退。
5. 语义输出必须经过规范化和 schema 校验，类别、颜色、关系、运动状态有显式别名映射；模型输出合法不等于候选选对，代码层面保留任务级失败原因。
6. 规划和车辆检查分层：目标选择、接近区域、道路/车道约束、离线搜索、矩形车身占用、控制状态限制和连续扫掠审计分别实现，不能把规划器返回路径直接解释为实际车辆到达。

## 与论文的对应关系

论文第4章对应对象深度恢复、跨帧融合、动态时效和关系图维护；第5章对应语义契约、候选检索、目标/接近位置选择和离线规划；第6章对应 `completion` 下的固定输入评价、消融、故障、KITTI 和统计脚本。第三方检测、分割、视觉语言和基础建图框架只作为依赖或已有方法归属。

## 尚需继续阅读

下一轮继续逐文件阅读 `conceptgraph/completion` 中生成证据的统计、故障、KITTI、地图约束和自然指令脚本，并对照测试用例确认每个论文数字的实际生成路径；随后更新论文方法描述中任何与源码不一致的措辞。
