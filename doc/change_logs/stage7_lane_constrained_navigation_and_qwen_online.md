# 车道级约束导航：CPU 实验验收与 GPU 待办

本轮按用户最新要求完成 CPU 可执行部分，真实 CUDA Qwen 实验明确保留为待办。CPU 验收状态：**通过**；原始完整 GPU 目标尚未完成。

最终运行目录：`/root/autodl-tmp/lizichong/outputs/project_v2/lane_qwen_experiment/runs/cpu_v3`。其余早期或中断运行目录保留，不作为最终结论来源。

## 实现与安全边界

- 从 nuScenes expansion HD Map 读取 lane/connector polygon、真实 token、arcline centerline、局部 heading、起终点与 incoming/outgoing。scene-0916 保留25条局部 lane/connector，scene-0003 保留20条。局部地图边界之外的连接单独记录，禁止生成虚构连接。
- 严格模式在 `(row, column, lane ID)` 上搜索；重叠 polygon 保留全部车道归属。`lane_id_grid` 的主ID用于索引/预览，不能代替规划使用的多车道状态。方向、拓扑、中心线偏离代价和每帧对象障碍覆盖同时参与规划。
- 起点和安全接近目标都受8m投影上限约束，目标还须通过类别安全距离与clearance检查。报告保留原始期望点、实际目标点及投影距离。车道连接须存在于真实 outgoing，并位于两段 arcline 端点2m以内；方向误差上限75°。
- 独立验证器按 polygon 边界交点切分整条线段，并检查全部闭栅格交点，覆盖线段中途越界与对角触碰。方向用中心线局部切向验证；拓扑诊断独立于方向诊断，避免早期方向失败掩盖后续非法切换。
- 平滑结果非法时拒绝该结果，重新执行车道搜索并禁用该次平滑；禁止用普通空间路径代替。普通 A* 也修复了对角穿角，平滑碰撞检查和最小栅格clearance统计使用 supercover；普通路径可能因此发生变化，旧产物保留作基线。
- nuScenes 输入为相机位姿时，车道模式取相机 +Z 的世界平面投影作为前向。普通模式的既有入口和默认配置保留；全量测试覆盖兼容性。

## 双场景结果

静态四类地图对照共8项，使用同一查询“前往小汽车附近”和首帧位姿；自由空间条件仍保留对象障碍，只去除HD Map区域约束。可行驶区域、纯车道区域、方向+拓扑逐项加入限制。所有成功路径均重新做连续碰撞检查。

| 场景 | 地图条件 | 结果 | 独立完整车道验证 |
|---|---|---|---|
| scene-0003 | drivable_area | 成功 | 0.00% |
| scene-0003 | free_space | 成功 | 0.00% |
| scene-0003 | lane_area | 成功 | 0.00% |
| scene-0003 | lane_constrained | NO_LEGAL_LANE_PATH | — |
| scene-0916 | drivable_area | 成功 | 0.00% |
| scene-0916 | free_space | 成功 | 0.00% |
| scene-0916 | lane_area | 成功 | 0.00% |
| scene-0916 | lane_constrained | 成功 | 100.00% |

在线回放逐帧读取既有检测、RGB/depth与录制位姿，重新执行增量建图及导航；没有重新运行目标检测或深度估计。每个普通/车道条件、每个场景均连续执行两次，共8次回放；所有接受的新路径以及复用路径的前方剩余段，均按该帧保存的实际 occupancy 独立复核。

| 场景 | 条件 | 成功规划/尝试 | 车道遵循率 | 逆向段数 | 非法拓扑转换数 | 碰撞率 | 越界长度合计/m |
|---|---|---:|---:|---:|---:|---:|---:|
| scene-0003 | drivable_qwen_off | 12/12 | 8.33% | 12 | 14 | 0.00% | 29.276 |
| scene-0916 | drivable_qwen_off | 11/11 | 45.45% | 8 | 0 | 0.00% | 20.204 |
| scene-0003 | lane_qwen_off | 5/13 | 100.00% | 0 | 0 | 0.00% | 0.000 |
| scene-0916 | lane_qwen_off | 1/3 | 100.00% | 0 | 0 | 0.00% | 0.000 |

遵循率的分母是实际成功规划条数；零成功路径记为未定义，不能写成100%。越界长度按每次规划的整条路径累计，重复规划可能覆盖同一路段，不能解释为车辆实际行驶里程。逆向段数是连续采样中的负方向事件数；75°方向容差违规样本另行记录。非法转换数是每条路径独立拓扑状态发生断开的次数；有离开车道的路径同时以越界指标判无效。

车道约束会降低目标可达性。拒绝不满足方向、拓扑或安全距离的目标属于本实验的安全性与可达性取舍；当前样本量和查询数量不足以支持总体性能优势结论。静态失败诊断另存于 `static_failure_analysis.json`，分别检查投影、安全距离及HD Map有向连通性；该诊断不会向导航返回放宽约束的路径。普通模式的低车道遵循率是对照结果，不等同于碰撞失败。

| 场景 | 条件 | 规划p95/s | 帧处理p95/s | 路径均长/m | 最小栅格clearance/m | 目标投影/回退均距/m |
|---|---|---:|---:|---:|---:|---:|
| scene-0003 | drivable_qwen_off | 3.327 | 3.513 | 10.294 | 0.500 | 0.302 |
| scene-0916 | drivable_qwen_off | 6.740 | 4.283 | 14.460 | 0.500 | 0.178 |
| scene-0003 | lane_qwen_off | 13.415 | 5.882 | 25.722 | 0.500 | 0.292 |
| scene-0916 | lane_qwen_off | 38.841 | 2.400 | 7.106 | 2.500 | 5.535 |

`frame_runtime_s` 包含建图和导航，不含随后独立审计与NPZ落盘；scene `runtime_s` 包括整个回放的审计、初始化和收尾。CPU线程数由复现实验命令限制为1，耗时只代表本次执行环境。clearance来自膨胀后占据栅格，不代表真实车辆的物理控制误差。

逐帧对象检索 Top-1/MRR 没有独立目标真值，当前保留 `null` 和原因；没有把规则生成的排序当作正确答案。既有阶段4 held-out 结果也没有冒充本轮在线指标。后续须采用来源明确的对象级标注基准。

## 故障、确定性和检查

- 两场景 × 异常/超时/非法schema/空输出：8/8实际故障注入保持基线路径或同一失败原因。注入后端明确标为synthetic fault，不加载Qwen，不计为模型效果。
- 全量 tests：169通过；其中本轮车道与故障专项21项通过。上游torchvision Beta API产生2条警告。
- 当前改动Python文件 Ruff：77个文件通过；`git diff --check`通过。
- 普通/车道 × 两场景的结构化内容哈希均两次一致；不把耗时字段纳入内容哈希。
- 全部受保护文件完整SHA-256比对：2492个文件通过，目录成员检查通过。原始基线未被改写；允许新增本阶段变更记录。此核验范围以原基线的保护清单为准。
- JSON/JSONL严格重解析：见 `json_parse_report.json`；独立路径审计：`independent_path_verification.json`；完整JUnit和Ruff证据：`verification_checks.json`、`full_tests.xml`。
- Git保留开始前的大量dirty改动；本轮没有提交、推送或删除用户文件。不要把最终 `git diff --stat` 的全部内容归因于本轮。

## 产物与复现

地图位于 `lane_maps/<scene>/`，含 lane NPZ、topology JSON和预览。最终实验报告位于 `/root/autodl-tmp/lizichong/outputs/project_v2/lane_qwen_experiment`；路径、选择与逐帧障碍栅格位于最终运行目录。图表位于 `visualizations/cpu_v3/`：四地图路径对照、在线规则路径对照、真实场景故障回退对照、Qwen待办状态和被拒平滑的结构化证据；均消费同一份规划地图与保存结果。

```bash
cd /root/autodl-tmp/lizichong
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.scripts.run_lane_qwen_experiment --run-id cpu_reproduce_01 --skip-qwen
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.scripts.run_lane_failure_injection --run-id reproduce_01 --static-run outputs/project_v2/lane_qwen_experiment/runs/cpu_reproduce_01
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.scripts.visualize_lane_qwen_experiment --run-dir outputs/project_v2/lane_qwen_experiment/runs/cpu_reproduce_01
/root/miniconda3/envs/conceptgraph/bin/python -m pytest tests -q --junitxml=outputs/project_v2/lane_qwen_experiment/full_tests.xml
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.scripts.finalize_lane_cpu_experiment --run-dir outputs/project_v2/lane_qwen_experiment/runs/cpu_reproduce_01 --reuse-baseline-verification
```

每次选择新的run-id，脚本拒绝覆盖已存在的运行目录。最终验收会再次核对实际帧、障碍栅格、哈希、测试和指纹；不会只依赖报告中的通过标记。

## GPU待办和结论边界

用户明确要求本轮跳过GPU部分。当前 CUDA可见设备数为0，真实Qwen后端调用0次；本地模型与adapter文件已检查。base/adapter真实推理、首次后端与缓存复跑、三类语义应用、真实schema/fallback/显存/耗时及候选重排图均尚未完成，详见 `gpu_todo.json` 与可直接续用的 `next_gpu_goal_prompt.md`。

本实验是nuScenes录制场景回放，不能表述为真实车辆闭环控制。车道合规性来自HD Map和确定性规划，不代表摄像头车道线识别。Qwen只允许语义解析、属性补充和已验证候选重排，不能直接生成或修改最终路径。

主要实现文件：

- `conceptgraph/navigation/lane_constraints.py`、`lane_geometry.py`、`lane_validation.py`、`lane_audit.py`
- `conceptgraph/navigation/target_selector.py`、`pipeline.py`、`online.py`、`planner.py`
- `conceptgraph/scripts/build_lane_constrained_bev.py`、`run_lane_qwen_experiment.py`、`lane_experiment_metrics.py`
- `conceptgraph/scripts/run_lane_failure_injection.py`、`finalize_lane_cpu_experiment.py`、`visualize_lane_qwen_experiment.py`
- `tests/navigation/test_lane_constraints.py`、`test_lane_qwen_safety.py`
