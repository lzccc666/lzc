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


---

以下为此前保留记录；当前状态以上述第12版与对应原始证据为准。

# 复现实验与续跑（进行中）

在项目根目录执行；所有新run-id必须唯一。不要重写历史。需要激活conceptgraph环境；系统python不用于模型。

```bash
export LD_LIBRARY_PATH=/root/miniconda3/envs/conceptgraph/lib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.frontend --run-id frontend_corrected_repro_001 --scene scene-0038
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.stage9 --run-id corrected_mapping_repro_001 --scene scene-0038 --frontend-run frontend_corrected_repro_001
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.relations_audit --mapping-run corrected_mapping_repro_001 --run-id relations_repro_001
/root/miniconda3/envs/conceptgraph/bin/python -m pytest -q tests
python outputs/project_v2/thesis_completion_v1/source_bundle/baseline_guard.py verify
```

已完成旧前端开发运行：dev_full_001和dev_factorial_002，各38场景1508帧。运行前源码指纹见各release.json，原始逐帧输出和场景报告保留。失败run不得覆盖。

前端可按release指纹使用--resume；建图已完成场景可续跑，中断场景拒绝覆盖，应新建run-id。尚不存在完整导航、训练和主表的一键复现入口，不能据此声明完整课题可复现。

## 本轮新增入口（均使用新run-id）

真实人工核验服务，浏览器入口为 `http://localhost:8767/visualizations/review.html`。服务提交写入本轮 `annotation/human_submissions/`；自动证据与实际人工记录分开。远程访问需转发8767端口。

```bash
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.review_server --port 8767
```

独立标签准备允许在方法冻结前执行；下列命令不运行新测试方法。新封存评测仍需完整release，不能仅因标签文件存在而开启。

```bash
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.reserved_annotation --run-id reserved_annotation_repro_001 --inputs reserved_inputs_002 --selection reserved_test_scenes_v2.json
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.structured_ranking --run-id structured_supervision_repro_001
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.semantic_training_v2 --dataset structured_supervision_002 --run-id three_task_v2_repro_seed17_001 --seed 17
```

正式三种子训练入口对应17、29、43，每次运行独立目录，不能并发启动多个Qwen模型进程。正式全验证在三种子全部完成后使用：

```bash
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.semantic_evaluation --run-id fair_three_task_repro_001 --dataset structured_supervision_002 --validation-per-task 0 --prompt-selection base_prompt_calibration_001 --training-series three_task_v2_formal
```

运动学与完整三维感知是不同实验；以下GPU闭环使用隔离的EGL运行库和真实YOLO/SAM/CLIP。实验资产SHA与来源记录位于 `interactive_simulation/` 和隔离依赖清单，不需改动系统驱动。

```bash
export LD_LIBRARY_PATH=/root/autodl-fs/third_party/graphics_runtime/lib:/root/miniconda3/envs/conceptgraph/lib
export PYOPENGL_PLATFORM=egl
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.interactive_navigation --run-id interactive_recovery_repro_001 --task config00-positive0 --task config00-no_match --task config00-inaccessible --seed 17 --recovery-wait-s 4
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.realtime_navigation --run-id realtime_rule_repro_001 --model rule --task config00-positive0 --maximum-wall-s 45
```

`interactive_navigation`为固定逻辑时钟；`realtime_navigation`有独立真实墙钟控制线程、观测TTL及有界后台语义队列。当前安全候选链支持解析、真实crop属性和确定性安全检查通过后的重排，`--semantic-tasks`可独立选择三个任务；旧解析实验可显式使用`--legacy-parse-only-chain --semantic-tasks query_parse`。已实际验证后台解析与属性，三任务完整模型应用及全量比较以各run真实报告为准。`--model base`或`--model new`需要GPU空闲并保持一个Qwen模型进程；故障注入可声明 `--semantic-ttl-s 1 --injected-post-delay-s 3`，保留迟到记录，不把注入等待当模型推理耗时。每个任务使用独立子进程，避免中断污染后续任务。

论文阶段稿及第6章三线表由冻结JSON直接生成，运行库独立放在本轮source_bundle/document_runtime，未修改conda依赖。006版为38页、9张表、5张图，含完整五模型三任务结果。图表顺序及表注分页检查通过；原稿数学公式仍需正式转换，不是终稿。浏览器只读复用已有Chromium。

```bash
export PLAYWRIGHT_BROWSERS_PATH=/root/autodl-tmp/lizichong/outputs/project_v2/lane_qwen_expansion_v1/browser_runtime
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.manuscript_delivery --run-id manuscript_delivery_repro_001 --figure-run figures_development_003
```

阶段整合稿不是最终论文。完整图件、全部主表、封存评价、独立消融和最终引用核对仍需完成；当前不能声明全部课题一键复现。

## 指纹续跑、公平统计与开发检索

公平续跑只接受原release、原源码快照、原验证请求及原权重。已完成版本核对后跳过，中断版本追加完整前缀；不会覆盖原报告或更改提示。下例针对尚未完成的原run，完成后拒绝重复运行。

```bash
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.semantic_evaluation_resume --run-id fair_three_task_full_validation_001 --wall-budget-s 6600
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.semantic_statistics --input-run fair_three_task_full_validation_001 --run-id fair_statistics_repro_001
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.retrieval_baselines --run-id retrieval_repro_001 --wall-budget-s 1800
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.retrieval_statistics --input-run retrieval_repro_001 --run-id retrieval_statistics_repro_001
```

检索使用实际CPU CLIP文字编码器和已有真实图像特征，所有方法相同前端及查询。稀疏二维网格是VLMaps式明确适配；不等同完整复现，不给实际路线、锁定或到达信用。不同独立身份关联协议的限制写入报告。

## 真实录制输入扰动

```bash
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.fault_robustness --run-id recorded_fault_repro_smoke_001 --scene scene-0012 --scene scene-0016 --seeds 17 --wall-budget-s 1200
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.fault_robustness --run-id recorded_fault_repro_full_001 --wall-budget-s 6600
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.fault_statistics --input-run recorded_fault_repro_full_001 --run-id recorded_fault_statistics_repro_001
```

六条件包含无扰动、30%检测随机缺失、80%LiDAR随机缺失、0.20m深度白噪声、0.25m/2°位姿白噪声及三帧前端中断；模型只接触扰动后的当前输入，官方真值仅用于隔离评价。两个开发场景的80帧无扰动指标与原冻结结果逐帧完全相等。完整运行按38场景、3种子、6条件冻结，种子不增加独立log数。

## 统一证据查看与真实时钟三模型smoke

浏览器入口：`http://localhost:8767/visualizations/navigation_demo_004/index.html`。163条保存试验、7106帧，含完整三模型002批9次和所有失败/中断；22条到达包含重复开发复测，不增加独立任务。默认隐藏GT评估层，展示只使用当帧和过去执行。需要8767核验服务；远程访问需转发端口。浏览器回放速度不等于实验实时性能。

```bash
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.navigation_demo --run-id navigation_demo_repro_001 --source-run interactive_rule_full_002 --source-run realtime_rule_geometry_smoke_003 --source-run realtime_rule_verified_smoke_002 --source-run realtime_base_three_tasks_smoke_001
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.realtime_batch --run-id realtime_three_model_repro_001 --after-evaluation-run fair_three_task_full_validation_001 --wait-budget-s 3600 --wall-budget-s 6000
```

三模型smoke为3个事前指定任务×rule/base/new×seed17共9次，依固定seed17乱序串行执行；它不替代50任务×3种子的完整主比较。总报告未完成或GPU有其他模型进程时不会并发启动Qwen。001排队阶段中断保留。002已完成9次：3个无匹配试验因感知快于控制周期、重复时间戳而退出，规则正任务到达而base/new正任务超时；未出现SIGSEGV。003在真实心跳取帧修复后按原9次完整顺序重新冻结，不覆盖002或只复测成功样例。每子进程90秒墙钟任务预算、240秒外部超时、候选规划10秒、语义TTL30秒、传感器TTL1.5秒；异常均入记录，不自动重跑挑优。

## 原始自然语言指令及准确传感器准备

Talk2Car原始数据固定提交 `3fde52aee3341312196ab9b2f999337eff0693a0`，源文件、SHA、完整数据许可与归属位于 `data_manifest/talk2car_source_001/`。数据许可为CC BY-NC-SA 4.0；README中的软件MIT说明不替代数据许可。本项目仅作非商业论文研究。

`data_manifest/natural_distribution_audit_001/`保存439条原始英文，覆盖30场景14log、360个不同相机帧。上游`sample_token`实际指向相机sample_data，必须以官方加载器口径解析。仅47条对应现有关键帧，392条引用非关键帧；静态官方属性176条也不能直接算作已审核接近导航指令。3帧smoke真实RGB/LiDAR准备、数值和叠图检查已通过，全量准确准备以complete.json为准。只使用相机之前100ms内的真实LiDAR，不替代相机、不填充未知深度。

```bash
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.natural_inputs --run-id natural_original_inputs_repro_smoke_001 --max-cameras 3 --wall-budget-s 900
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.natural_inputs --run-id natural_original_inputs_repro_full_001 --wall-budget-s 1800
```

全量新增存储预算700MB、smoke25MB，源归档只读流式访问，不保存重复原始LiDAR。此入口不运行检测、语义、检索或封存方法；人工属性、任务适配和自然分布评估仍待完成。


## 路线复用、完整真实时钟矩阵与参数检查（检查点011）

复用仅保留路线形状；每次从当前实际车辆状态重新积分，并按当前道路/障碍检查矩形车身、5cm余量和连续扫掠。端点、身份、航向不符或检查失败即退回Hybrid规划。单任务开发pilot规则/base到达、new超时，重排均无实际应用，不据此推断总体收益。

```bash
export LD_LIBRARY_PATH=/root/autodl-fs/third_party/graphics_runtime/lib:/root/miniconda3/envs/conceptgraph/lib
export PYOPENGL_PLATFORM=egl PYTHONFAULTHANDLER=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.realtime_matrix --run-id realtime_three_model_full_001 --preset main --wall-budget-s 86400 --storage-budget-gb 20 --minimum-free-gb 6
# 已存在且中断的批次只在源码、任务与权重指纹一致时使用：
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.realtime_matrix --run-id realtime_three_model_full_001 --preset main --resume
```

当前批为50个冻结开发任务×3种子×3方法450次，逐进程隔离单模型。所有已尝试失败留在分母，续跑不自动重试。源码冻结清单见该批release.json；批次结束前不得改清单内依赖。需要新版本时先保存当前批，不篡改release。语义减解析/减属性/减重排的完整开发矩阵入口为相同模块的`--preset semantic-minus`，当前尚未运行；完整/关闭Qwen参考需核验同版本主矩阵，不把入口存在记成消融完成。

```bash
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.selection_parameters --run-id selection_parameters_dev_001
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.mapping_parameters --run-id association_parameters_smoke_001 --scene scene-0038 --wall-budget-s 180
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.mapping_parameters --run-id association_parameters_dev_full_001 --wall-budget-s 1800
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.mapping_parameter_statistics --input-run association_parameters_dev_full_001 --run-id association_parameters_dev_full_001 --wait-budget-s 1900
```

选择端9设置×1115查询已完成，Top-K仅改变候选覆盖；0.7拒绝阈值全部拒绝正任务，保留负结果。关联参数只改变推理sim_threshold=0.25/0.45，0.35复用同观测/种子/配置完整基线；评价3m门限不变。两类检查均不修改正在运行的导航。CPU参数实验与导航共享主机，不能作为独占性能表。原目录已存在时请使用新的run-id；不要覆盖失败证据。

## 自然指令真实人工审核与论文009

```bash
# 已生成439卡片，无需覆盖重新生成；服务端口8768与旧8767并存
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.review_server --port 8768
export PLAYWRIGHT_BROWSERS_PATH=/root/autodl-tmp/lizichong/outputs/project_v2/lane_qwen_expansion_v1/browser_runtime
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.manuscript_delivery --run-id manuscript_delivery_009 --figure-run figures_development_003
/root/miniconda3/envs/conceptgraph/bin/python outputs/project_v2/thesis_completion_v1/source_bundle/document_qa_004.py --run-id manuscript_delivery_009
```

自然核验页：http://localhost:8768/annotation/natural_command_review_001/index.html；远程访问需转发8768。主语料439原文、360原图、410不同二维裁剪均不作人工结论，真实记录独立存annotation/human_natural_submissions/。缺少审核者/依据、源哈希不符会拒绝；当前人工0，浏览器测试未写入假审核。

论文009为39页、11表、5图及12处公式的阶段稿，公式同时导出可编辑OMML和离线MathML；本机没有Word/LibreOffice渲染，未声明全39页最终视觉验收或全文引用已核验。007因浏览器环境路径缺失中断、008括号拉伸的证据保留。


已启动后续缺一消融调度：`source_bundle/semantic_ablation_after_main_001.py`等待完整450次主矩阵结束并核验同版，再启动`realtime_semantic_minus_full_001`的450次；预算另24小时/20GB，开始前至少26GB可用，全部失败保留。运行链完成前同样保护主矩阵release中的源码依赖。联合`semantic_ablation_combined_001`仅是300共享基线+450新增的750原始记录合集，不是750个独立任务。

```bash
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.realtime_matrix_statistics --input-run realtime_three_model_full_001 --run-id realtime_three_model_full_001 --wait-budget-s 87000
```

完整统计器先用12条实际已完成试验核验读取，再设置等待；缺少结果仍保留四项业务null并留在分母，轨迹前缀不计完整安全信用。统计回归防止给缺失数据或无独立审计的自报到达信用。最终报告仅在全部冻结试验有尝试记录后产生；等待不是统计完成。

## 开发地图、动态维护与接近点消融（检查点012）

以下均为实际已完成的CPU开发实验入口；使用新的run-id复现，不覆盖已有产物。它们不打开新封存方法测试，也不与正在运行的GPU主矩阵争用模型进程。共享CPU的耗时不计独占性能。

```bash
export LD_LIBRARY_PATH=/root/miniconda3/envs/conceptgraph/lib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.map_constraint_ablation --run-id map_constraints_repro_smoke_001 --query-limit 2 --wall-budget-s 180
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.map_constraint_ablation --run-id map_constraints_repro_full_001 --wall-budget-s 7200 --case-budget-s 3
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.map_constraint_statistics --input-run map_constraints_repro_full_001 --run-id map_constraints_statistics_repro_001
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.mapping_module_ablation --run-id mapping_modules_repro_smoke_001 --scene scene-0038 --wall-budget-s 180
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.mapping_module_ablation --run-id mapping_modules_repro_full_001 --wall-budget-s 1800
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.mapping_module_motion_statistics --input-run mapping_modules_repro_full_001 --run-id mapping_modules_motion_repro_001
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.goal_selection_ablation --run-id goal_selection_repro_smoke_001 --scene scene-0038 --wall-budget-s 180
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.goal_selection_ablation --run-id goal_selection_repro_full_001 --wall-budget-s 3600
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.road_group_statistics --run-id road_groups_repro_001
```

六地图条件最多每场景选6条，实际为212查询、176不同几何查询、1272规划，不能把上限228当实际分母。正确对象参考点由隔离GT给定；全部静态矩形姿态未通过，点路径不构成安全接近或实际到达证据。动态维护/TTL为114条件记录，其中76次新重建、38份同配置完整基线复用；建图模块消融结束时会自动生成同run-id的statistics报告，无须调用关联阈值统计器。接近点/首达/联合为1115查询、3345记录；1115组实际候选池和382份上下文SHA核验通过，联合38/733、首达36/733，差的区间含0。去接近点37个已选终点均不足预测框外2m，条件仅离线，未执行。完整推理输入与所有失败分别保留。

## 新论文012、科研图与统一入口

012整合稿实际为44页、16表、7图、12处可编辑公式。图表来自冻结开发报告；新封存主表、三模型完整导航和全部正文文献核对仍未完成。011因合并图注的段首空白导出失败，保留目录和日志；012修复后成功。

科研图新源包为 `thesis/figures_constraints_002/`，需先有声明的 `figure_contract.json` 再运行绘图器。首包001的自动刻度检查未通过，不能取代002；重复绘制应复制契约到新目录，不覆盖旧图。合并入口只接受已通过源SHA和实际QA的图包。

```bash
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.merge_figures --run-id figures_development_repro_001 --source figures_development_003 --source figures_constraints_002
export PLAYWRIGHT_BROWSERS_PATH=/root/autodl-tmp/lizichong/outputs/project_v2/lane_qwen_expansion_v1/browser_runtime
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.manuscript_delivery --run-id manuscript_delivery_repro_012 --figure-run figures_development_repro_001
/root/miniconda3/envs/conceptgraph/bin/python outputs/project_v2/thesis_completion_v1/source_bundle/document_qa_006.py --run-id manuscript_delivery_repro_012 --expected-tables 16 --expected-figures 7
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.thesis_portal --run-id thesis_portal_repro_001
```

统一入口：[论文及实验证据](http://localhost:8768/visualizations/thesis_portal_001/index.html)，需运行核验服务并转发8768端口。页面实时读取实际保存状态，当前未完成批次不显示总体结论；论文入口从validation_report读取最新实际版本。桌面/手机QA通过，全部请求GET，没有写入人工记录。原三维浏览器仍为163保存试验，运行中的完整矩阵结束后必须继续更新。

最新工程检查：267项tests通过、5条依赖警告；新代码Ruff和git diff --check通过。严格运行快照44163份JSON/JSONL、579639条JSONL记录，0非法；跳过当时活动子试验与追加记录，不能称最终验收。最新全量历史保护核验255170文件，0改动/缺失。运行中的450主比较及后续450缺一消融继续冻结其66项源码依赖；不得修改后再沿用旧release。

## 独立静态区域与因果输入（检查点013）

下列入口对应已真实执行的全量工作；复现请换新run-id。区域为隔离评价端标签准备，不能作为方法输入；全部878查询的可达性仍unknown。1166真实过去扫描与30首帧缺失分开报告，空掩码没有测量信用。

```bash
export LD_LIBRARY_PATH=/root/miniconda3/envs/conceptgraph/lib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.reference_regions --run-id reference_regions_reserved_full_003 --wall-budget-s 3600
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.reference_region_audit --run-id reference_regions_reserved_full_003 --audit-id audit_001
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.reference_region_preview --run-id reference_regions_reserved_001 --input-run reference_regions_reserved_full_003
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.reserved_causal_inputs --run-id reserved_causal_inputs_full_001 --wall-budget-s 3600
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.causal_input_audit --run-id reserved_causal_inputs_full_001
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.reserved_causal_inputs --run-id reserved_causal_inputs_full_001 --wall-budget-s 3600 --resume
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.causal_overlap_audit --run-id causal_cross_input_audit_002
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.thesis_portal --run-id thesis_portal_002 --region-run reference_regions_reserved_full_003 --region-preview reference_regions_reserved_001
export PLAYWRIGHT_BROWSERS_PATH=/root/autodl-tmp/lizichong/outputs/project_v2/lane_qwen_expansion_v1/browser_runtime
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.manuscript_delivery --run-id manuscript_delivery_015 --figure-run figures_development_005
/root/miniconda3/envs/conceptgraph/bin/python outputs/project_v2/thesis_completion_v1/source_bundle/document_qa_007.py --run-id manuscript_delivery_015 --expected-tables 18 --expected-figures 7
```

门户更新为[论文、真实实验与独立标签证据](http://localhost:8768/visualizations/thesis_portal_002/index.html)。PDF015的6页实际目视记录见同目录pdf_visual_qa_001/findings.json；014缺字与013导出失败均保留。新输入需要纳入未来封存release并验证缺测处理，不能据转换完成宣称全部传感器覆盖或新方法评测完成。原开发1508帧也有帧内未来LiDAR，已有结果仅作离线配对诊断；本轮交互渲染闭环的独立真实时钟传感链未改变。

最新工程证据为tests_full_015.log（273通过）、ruff_completion_026.log、git_diff_check_017.log及json_snapshot_004/report.json（50241文件/664166记录0非法）。快照时之后的文件未包含，最终验收仍须重新执行。部分标签生成失败、跨输入集合错误预期和文档失败不删除、不以修订结果覆盖。


## KITTI新增真实实验及写作交接（2026-11-12）

已完成：`kitti_object_uniform64_001`准备64帧/33原始drive，`kitti_object_cpu_smoke_001`真实3帧，`kitti_object_cpu_full_001`真实64帧；主表319个car/truck/pedestrian，匹配247→262，251共同支持误差1.758→1.540m。卡车负结果与深度未知保留。`statistics/kitti_object_cpu_full_001/`为自定义3m中心诊断，不是官方AP、封存测试或导航到达。

下面使用全新run-id复现。仅在项目根目录及已有模型环境运行；CPU旁路不会另占GPU，但主机并行负载不计独占性能。原始zip只读取选定member，不解压全部，不写公共卷。

```bash
export LD_LIBRARY_PATH=/root/miniconda3/envs/conceptgraph/lib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.kitti_transfer prepare --run-id kitti_object_uniform64_repro_001 --frames 64
CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.kitti_transfer infer --run-id kitti_object_cpu_repro_001 --agent-manifest outputs/project_v2/thesis_completion_v1/data_manifest/kitti_object_uniform64_repro_001/agent_manifest.json --wall-budget-s 10800 --storage-budget-gb 2
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.kitti_evaluate --input-run kitti_object_uniform64_repro_001 --inference-run kitti_object_cpu_repro_001 --run-id kitti_object_cpu_repro_001
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.kitti_preview --input-run kitti_object_uniform64_repro_001 --inference-run kitti_object_cpu_repro_001 --statistics-run kitti_object_cpu_repro_001 --run-id kitti_object_preview_repro_001
```

正式证据页：`http://localhost:8768/visualizations/kitti_object_cpu_full_002/index.html`。001页生成器重复解压NPZ导致开销，主动终止保留；修正为每帧只解压一次后002完成。QA001在生成结束前访问404，保留原错误；QA002桌面/手机、翻页、GT隐藏、深度叠加、无溢出、全GET均通过，并实际查看截图。没有生成任何人工标注记录。

本轮最终代码检查快照：全量tests286通过/5条依赖警告（tests_full_018.log）、Ruff与diff检查通过（ruff_completion_030.log、git_diff_check_021.log）、历史255170文件全量SHA0变化（protection_verify_008.log）。运行中66源文件保持一致，完整主矩阵及后续缺一仍未结束，因此非整个Goal最终验收。

写作交接见 `GPT_Work_写作交接.md`，执行与Git交接见 `工作交接_代码与实验.md`。用户已授权同步到 `https://github.com/lzccc666/lzc`；只同步当前代码和必要小型交接证据，不发布原始数据、权重、原论文与凭据。成功与否以本机 `source_bundle/github_sync_001/push_report.json` 的远端核验为准，不能仅以本地提交判断已同步。


## 开发集过去LiDAR输入、独立审计与真实重建（新增）

现有全量运行：`source_bundle/development_causal_pipeline_full_001/`。仅CPU、单个顺序流水线，与主矩阵重叠时无独占性能信用。以下命令已真实启动；新复现实验必须换run-id，已有输出不覆盖。

```bash
LD_LIBRARY_PATH=/root/miniconda3/envs/conceptgraph/lib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.development_causal_pipeline --run-id full_001
LD_LIBRARY_PATH=/root/miniconda3/envs/conceptgraph/lib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.development_causal_downstream --run-id full_001 --mapping-run causal_mapping_full_001 --wait-for-pipeline full_001
```

第一条顺序执行：原开发38场景1508帧过去LiDAR转换（3GiB/7200s）→独立原始表及像素重投影审计（1800s）→复用同RGB真实前端的七条件重建（3GiB/10800s）→4000次log配对统计。第二条只在完整建图结束后运行关系、正确解析上限、四方法检索和配对统计（1GiB，分阶段900/900/1800/600s）。磁盘均保留6GiB。单模型GPU主矩阵及66源码不得改动。

只核验已完成转换而不重复读归档的入口：保持release所有源码与输入SHA不变，调用development_causal_inputs对应的`--run-id development_causal_inputs_full_001 --wall-budget-s 7200 --storage-budget-gb 3 --minimum-free-gb 6 --resume`。若源代码更新，应使用新的run-id及新release；不可覆盖旧结果。其他失败需先查看interrupted/failed记录，再按各stage command和新run-id重做，不把排队计完成。

试运行证据：data_manifest/development_causal_inputs_smoke_001/audit_001/report.json、statistics/causal_mapping_smoke_001/report.json、mapping_evaluation/causal_relations_smoke_001/summary.json、main_experiments/causal_retrieval_smoke_001/report.json、statistics/causal_retrieval_smoke_002/report.json。检索统计001的无正查询NaN写出失败保留，002以null表示未知；缺测和候选缺失均保留在原GT分母。官方sample GT仍存在晚于相机的时间近似，见协议，不将此批称封存测试或完整导航。


开发causal完整输入及独立审计现已结束：38场景11log1508帧，1470真实过去扫描、38缺测，未来0，年龄3.870–63.661ms；1508帧独立重投影像素差0，7659文件SHA通过。实际resume 0归档读取/0新增输出。官方sample GT偏移实测33.136–42.356ms，保留评价时间近似。

新增显式输入版本入口：fault_robustness --mapping-run；mapping_module_ablation --mapping-run和--baseline-fault-run（必须帧SHA/seed17/配置相同）；map_constraint_ablation和goal_selection_ablation --retrieval-run。默认旧版本保留，禁止用旧未来LiDAR完整条件作新causal消融基线。source_bundle/causal_ablations_after_downstream_001.py已启动等待，同版完整causal下游结束后顺序做684扰动、114模块记录、六地图及三目标接近条件；协议development_causal_ablations_001.md。真实smoke均完成，未把排队当结果。全量tests294通过，5依赖警告。全部阶段仍未验收。


## 过去LiDAR敏感性与关系边消融

本机 `source_bundle/causal_sensitivity_sequence_001.py --preset full` 已在执行/等待；禁止重复启动或使用已有run-id覆盖。先检查其release、各exit和complete，以及同源 `faults/causal_fault_full_001/summary.json`。它只用CPU、不改变冻结导航与参数，不开放新测试。全量命令也可从相应 `*_command.json`逐项复现；异机先准备相同输入与协议。

```bash
python -m conceptgraph.completion.selection_parameters --run-id YOUR_NEW_RUN --retrieval-run causal_retrieval_full_001
python -m conceptgraph.completion.development_selection_ablation --run-id YOUR_NEW_ABLATION --retrieval-run causal_retrieval_full_001
python -m conceptgraph.completion.road_group_statistics --run-id YOUR_NEW_GROUP --retrieval-run causal_retrieval_full_001
python -m conceptgraph.completion.mapping_parameters --run-id YOUR_NEW_ASSOCIATION --mapping-run causal_mapping_full_001 --baseline-fault-run causal_fault_full_001 --wall-budget-s 3600
python -m conceptgraph.completion.mapping_parameter_statistics --input-run YOUR_NEW_ASSOCIATION --run-id YOUR_NEW_STATS
```

关联基线须同观测SHA、seed17、配置，不能替换旧未来LiDAR结果。9参数实际10035行；关系边条件保留几何回退，不能称关闭全部关系推理。零正查询指标必须null，真实负查询验证见 `source_bundle/causal_sensitivity_negative_only_001`。门户004展示最新past主结果，旧门户及失败浏览器命名冲突日志保留。


过去LiDAR动静态与错误证据已实际完成，生成器源码及指纹在 `source_bundle/causal_diagnostics_full_001`。异机准备同一mapping和official_motion标签后，可使用新run-id：

```bash
python -m conceptgraph.completion.motion_audit --mapping-run causal_mapping_full_001 --run-id YOUR_NEW_MOTION
python -m conceptgraph.completion.inspection --mapping-run causal_mapping_full_001 --run-id YOUR_NEW_EVIDENCE --method one_to_one_depth_quality
```

本机补充配对统计由 `source_bundle/causal_motion_statistics_001.py`生成，真实crop/掩膜核验由 `source_bundle/causal_evidence_audit_001.py`执行；原件保留，不能重复运行现有目录。共同匹配分析为初步计数后的解释性分析，不是预注册独立主表；全部自动证据，不改变人工提交数量或旧审核服务。

## 版本化过去LiDAR人工核验

正式访问 `http://localhost:8768/visualizations/causal_development_review_002/index.html`；先转发8768端口。旧开发身份审核不能直接归入新图。新来源、方法和节点校验测试311通过，实际人工数量仍以 `/api/review_status` 为准。完整操作、无检测对象和跨帧证据限制见 `task_protocol/development_human_review_versioning_001.md`。

已有服务正在8768运行，勿重复启动。仅在其确实退出后，用下面命令恢复；保持记录目录不变。

```bash
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.review_server --port 8768
```

如需重新生成只读审核页面必须使用新的run-id；证据和审计SHA改变时先独立复核，不覆盖001/002。

```bash
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.development_review --run-id YOUR_NEW_REVIEW_ID
```

本轮过去LiDAR684录制故障、各114动态模块/关联门限完整结束；五步敏感性队列有complete.json。全帧关联和配对统计独立重算脚本为 `source_bundle/causal_fault_audit_001.py`、`causal_module_parameter_audit_001.py`，其输出已存在，原脚本故意拒绝覆盖，复现应另建输出版本或在隔离复现目录执行。不是启动脚本后即视为实验完成。


## 相机时间评价敏感性与完整地图复核

已真实完成的运行ID拒绝覆盖。相机时间评价只使用固定过去LiDAR开发预测与隔离官方标注中心；不运行模型/封存导航。重现已有结果须另选run-id，并保持原协议和元数据SHA。原官方时间主成绩不被本补充替代。

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  /root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.camera_time_evaluation \
  --run-id camera_time_gt_dev_reproduction_001
```

实际运行`camera_time_gt_dev_full_001`，报告与原始关联记录在`mapping_evaluation/camera_time_gt_dev_full_001/`。完整参数以该release.json与--help为准；独立审核源码为`source_bundle/camera_time_gt_audit_001.py`。六地图已有完整统计和独立显式车身复算，审核源码`source_bundle/causal_map_audit_002.py`；该审核器固定输出audit_002，已有结果不可覆盖，首次边界失败audit_001须保留。三处数值边界差异记录不应改判通过。


## 真实时钟传感器故障与CPU试跑

正式GPU故障当前只排队，不可在主矩阵/缺一语义活跃时手动启动。先查看`faults/realtime_fault_after_semantics_001/release.json`、`source_bundle/realtime_fault_after_semantics_001.log`及进程；已保存run-id拒绝覆盖。等待器已运行，不要再次执行。程序入口`python -m conceptgraph.completion.realtime_faults --help`；矩阵脚本和冻结预算见`task_protocol/realtime_sensor_faults_001.md`。

已完成的独立输入/执行复核可阅读`faults/realtime_fault_cpu_smoke_002/audit_001/report.json`及`report.md`；来源脚本`source_bundle/realtime_fault_cpu_audit_001.py`使用排他新目录，原路径已完成，不应直接覆盖重跑。`realtime_fault_cpu_smoke_002.py`的原run-id已完成，重新实验须独立新版本和预算。第一批001的真实兼容失败保留。

查看器：`http://localhost:8768/visualizations/realtime_fault_cpu_preview_002/index.html`。63张有效检测图来自已保存真实输出重绘，前端新增调用0。源生成脚本002和浏览器QA002留存；001排他写入/Playwright参数失败也保留。

CPU渲染隔离运行库不是全局安装。真实使用的环境变量、OSMesa/PyOpenGL来源、SHA及成功探针见`source_bundle/software_render_runtime_001/accepted_001.json`、`probe_004.log`和`realtime_fault_cpu_smoke_002.py`。库未进入公共Git，异机需从官方源重建；CPU TTL15秒与正式GPU1.5秒分开。4秒断流在CPU批未触发过期停车；仅TTL0.3秒已知路径功能试验显示停车/恢复，不能替代正式GPU证据。


## 分阶段性能测量

当前CPU已经实际完成，固定输出`performance/runtime_profile_cpu_frontend_001`与`runtime_profile_cpu_recorded_001`、独立统计`runtime_profile_cpu_smoke_001/audit_001`。源脚本`source_bundle/runtime_profile_cpu_smoke_001.py`与`runtime_profile_CPU_audit_001.py`使用排他run-id；原输出不可覆盖。新复验须先声明新版本/预算和独立路径。

正式独占GPU队列`source_bundle/runtime_profile_after_faults_001.py`已经启动，先等待原主矩阵/缺一/GPU故障真实进程均退出，检查PID启动tick与CUDA空闲后才执行；不要再次启动。协议/固定输入见`task_protocol/runtime_performance_001.md`和`performance/runtime_profile_protocol_002/plan.json`。001含一个重复语义输入而未运行Qwen，保留原清单；002与已完成CPU输入完全一致，实际业务cache.get截获核对三模型各72个不同键，没有运行模型或写响应。

业务CLI：`python -m conceptgraph.completion.runtime_profiling --help`。CPU必须在进程启动前设`CUDA_VISIBLE_DEVICES=''`；GPU默认监测唯一计算进程，发现他进程取消独占信用。每模型/任务首次、暖态、缓存分开，嵌套计时不相加。源码及Adapter冻结，旧run-id排他拒绝；没有自动重试或丢弃慢/失败请求。全量完成后仍须独立复核统计和全部工程验收。


## 全序列身份与返回开发诊断（仅实验交付，不新增论文正文）

代码：`conceptgraph/completion/temporal_identity_evaluation.py`；已完成`statistics/temporal_identity_full_001/`，官方参考独立审计002通过。禁止覆盖已有run-id；下面只在新命名下重做开发评价，不打开封存。

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.temporal_identity_evaluation --mapping-run causal_mapping_full_001 --run-id temporal_identity_replication_UNIQUE
```

原执行器`source_bundle/temporal_identity_cpu_001.py`先1场景smoke再38全量，CPU900s/1GiB/至少6GiB余量，与主矩阵共享CPU窗口已记资源事件。参考下载器`fetch_identity_reference_001.py`及固定manifest/官方MIT LICENSE、NumPy兼容差异与独立审核器`temporal_identity_audit_002.py`随本机交付；首次audit_001零键比较失败保留。审核器路径固定原已完成run，独立重做需在副本指定新run/审计目录，不覆盖原审计。

评价独立于推理；同类XY<=3m，GT晚相机33–42ms近似保留。单帧身份按帧隔离；全序列GT只用于评价，不传训练/语义缓存/建图。IDF1受覆盖影响，不能替代人工错合并判定；列表缺席不等于确定遮挡，未知和未返回尾段均保留。简明已完成/待完成与队列顺序见`实验完成度与交付清单_001.md`。


## 独立控制重算与连续车身审核

当前已实际完成`vehicle_execution/independent_body_rule_full_001/`（150条/30806区间）及`independent_body_fault_cpu_full_001/`（6条/2270区间）。协议`task_protocol/independent_continuous_body_001.md`；输入/账目/940文件SHA独立核对`vehicle_execution/independent_body_evidence_check_001/report.json`。全部33076区间获静态数值证书，原到达21与6分别交叉核对，不增加到达或独立任务。

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/envs/conceptgraph/bin/python outputs/project_v2/thesis_completion_v1/source_bundle/independent_body_audit_001.py --cohort rule --run-id independent_body_rule_replication_UNIQUE
```

先固定首任务`--limit 1`试跑，再全量；CPU900秒/1GiB/最低6GiB余量。故障队列用`--cohort fault_cpu`；完整450主矩阵只在对应summary存在后用`--cohort main`。已有run-id排他拒绝，失败和unknown不重置。等待器`source_bundle/independent_body_after_main_001.py`已经运行，PID必须实查，不重复启动；它只有CPU审核，不加载模型或抢占CUDA。

新实现不调用原积分/矩形/扫掠/setup，以复数sinc圆弧与速度界包络给静态整车证书，固定2e-6m数值余量、最多12级二分；未决不记安全。GEOS双精度及分段恒定平均运动模型的限制保留。此审核不独立重评语言身份、区域保持或有向拓扑，不产生新到达。原机最新374测试通过，首次既有异步事件断言失败与修复前源码见source_bundle/async_no_match_test_before_001；tests_full_035失败、036通过分别保留。


## 本轮自然指令覆盖与真实表达采集（不写论文）

`data_manifest/natural_candidate_coverage_001`是实际2308原文元数据审计；`natural_data_preparation_audit_002`独立检查通过。001审计先因RGB符号链接路径字符串不同失败，002按解析后的同一文件核对，不修改来源数据或SHA；失败文件保留。首次HTTP单元测试因环境代理502失败，修复本机测试绕过代理；浏览器首次缺旧runtime路径失败，复用只读历史runtime后通过。上述均不产生人工记录。

现有服务：`python -m conceptgraph.completion.natural_collection_server --port 8768`；三类入口兼容。当前实际PID见`source_bundle/natural_collection_service_upgrade_001/report.json`，不重复启动。回退旧服务命令在同目录before.json。先终止自己确认身份的当前服务再回退，不占用同一端口并行启动。

- 新自由表达：`http://127.0.0.1:8768/annotation/human_natural_collection_001/index.html`
- Talk2Car原文审核：`http://127.0.0.1:8768/annotation/natural_command_review_001/index.html`
- 状态：`/api/natural_collection_status`只计自由表达原始提交，`/api/review_status`计既有标签审核。

新表达入库`annotation/human_generated_natural_submissions/`，默认目标身份/运动/颜色/歧义/可达性unknown，不直接进入封存测试。字面去重不能证明表达家族隔离。历史测试数量对应各自代码版本，当前以`tests_full_037.log`实际387通过为准。

普通回归：`CUDA_VISIBLE_DEVICES='' python -m pytest -q tests`。临时目录正向HTTP测试不会写正式人工目录。真实页面QA需设置只读历史`PLAYWRIGHT_BROWSERS_PATH`，并使用新的run-id，参见`source_bundle/natural_collection_browser_qa_001.py`。新环境不含原始RGB时须先按原数据许可准备清单中的文件，HTML和清单不能代替数据。


## 保存执行的连续方向审核

`conceptgraph.completion.execution_direction_audit`不调用规划器/原车身审核。实际150规则与6CPU故障报告位于`vehicle_execution/execution_direction_rule_full_001`及`execution_direction_fault_cpu_full_001`；每步保守最近段方向界、进度范围、初始实际观测及原执行SHA均保存。方向模块采用固定保存模型圆弧，不能替代完整动力学或碰撞审核。

`source_bundle/execution_direction_audit_001.py --cohort rule|fault_cpu|main --run-id 新的唯一目录`是实际入口；默认900秒/1GiB/至少6GiB剩余/CUDA禁用，与主矩阵共享CPU。主450已由`execution_direction_after_body_001.py`等待原独立车身队列完整，不重复启动。当前排队不等于完成，完成后仍需独立核验。

`execution_direction_evidence_check_002.py`实际用全中心线切向的更保守航向锥复核156试验27314运动区间，GEOS独立核对54628个进度。001同结果先完成但辅助代码有E731风格问题，原脚本/结果/失败Ruff日志保留；002只改为具名函数后再次实际核验。两次复核不增加独立样本数量。普通回归`tests_full_038.log`401通过。

十配置共用三个道路生成族，均单向单车道。没有实际连接过渡，连接准确率null、未检验；30未行驶试验不算方向安全。全局航向锥只为本批保存轨迹提供另一种方向界，不是第三种车辆积分或动态交通证明。


## 真实开发地图连接诊断（非语义导航）

入口`source_bundle/HDmap_vehicle_connection_prepare_001.py`生成38开发场景的固定lane→connector→lane参考任务；各场景只选原有因果上下文中最早保存帧，不保证frame_idx为0。全量原始任务/失败理由见`vehicle_execution/HDmap_connection_tasks_001`。没有新封存访问。首个元数据失败与首个ready任务的规划试跑均保留；另给定中心线的控制功能试跑实际运动4.86m后安全停车，未到达，不计规划或语义成绩。

正式执行入口`HDmap_vehicle_connection_run_002.py --mode full --run-id 唯一新目录`，需要原控制试跑记录；默认共享CPU、隐藏CUDA、总3600秒/1GiB/剩余6GiB，38×3×2=228。Hybrid单任务3秒/2000扩展，点A*为既有离散入口而非相同搜索预算，不能将本诊断当公平时延或算法普遍优劣比较。相同车辆/道路/障碍、配对初始扰动和控制守卫。900步上限，0.75m/12度/0.1m每秒/出口lane/安全条件保持2秒才记REFERENCE_POSE_REACHED，不计语言对象正确到达。

原228结果固定在`HDmap_connection_full_001`，其原始7源快照及SHA在同目录。实验结束后仅修正输出计数：原declared_lane_transitions_exercised为候选控制区间数，不能称不同连接数。当前源码增加完整可回溯车道状态见证，18轨迹实际共36次连接；修前字节/差异/独立匹配见`source_bundle/HDmap_counter_schema_before_001`。复现原始输出必须在独立临时目录使用该原始快照，禁止覆盖七组活跃冻结业务源。使用当前源码的新run有等价控制和新的计数字段，不声称原release字节相同。

`HDmap_connection_independent_audit_001.py`读取既有228结果，采用另一种圆弧中点方框界复核全部6351后轴区间，独立回溯完整lane序列、官方原始连接与多边形、终点2秒保持及11log配对bootstrap。审核程序固定输出目录，不可直接覆盖重跑；复制为新版本并指定唯一结果目录。需要依法取得的nuScenes地图、相机上下文等原始依赖，包不替代原始数据。重复审核不增加执行/模型/到达。

8768端口服务上的`/visualizations/HDmap_connection_preview_001/index.html`展示全部228条件，播放只读实际保存车辆状态。`HDmap_connection_browser_qa_001.py`实际通过桌面/手机10条件、首中末状态/播放/无运动禁播/无横向溢出；不提交人工记录。当前原机完整测试408通过，不运行真实模型。新测试与所有最终验收仍未完成。


## 同预算引导与无损规范化开发版本

运行入口为`source_bundle/reference_guided_vehicle_run_001.py`（严格原始采样）与`reference_guided_vehicle_run_002.py`（显式相邻重复XY规范化），均`--mode smoke|full --run-id 新的唯一目录`。两者均38×3×2=228，全量前需固定首任务/首ready两方法4条件烟测；默认总3600秒/1GiB/最低6GiB、隐藏CUDA。脚本绑定原任务及特定试跑目录，复现需在独立副本修改run-id及对应试跑依赖并另立release，不能直接覆盖本机旧结果。原Hybrid/新引导均3秒2000扩展并交替顺序，所有前导工作计入预算；扩展单位不同，不做最优性或独占性能结论。

模块`navigation/reference_guided_vehicle.py`只沿真实积分提出路线，`completion/canonical_vehicle_corridor.py`保留源输入并明确移除相邻完全相同XY，验证有序非零边/集合/长度不变，不修补道路或连接。v1/v2正式运行前8/9份源码完整快照与SHA位于各full目录。它们未替换正在运行的三模型冻结链，不得改七组活跃冻结业务。

`reference_guided_vehicle_audit_001.py/002.py`独立重核实际控制、官方地图/完整车道见证/连续后轴/方向/保持；`reference_guided_plan_audit_001.py/002.py`核对计划状态可由有界控制产生，并用加0.05米矩形车身的独立连续审核。该控制见证由状态推导，不冒充保存了原规划建议命令；实际执行的原始命令另保存。`reference_guided_canonical_impact_001.py`逐键核对前后228条件、重复计数和未受影响168条件/42组路径状态。辅助审核固定输出目录，复跑应复制为新版本使用独立目录，不覆盖旧证据。

当前回放：8768服务下`/visualizations/HDmap_reference_guided_preview_003/index.html`为v2，`/visualizations/HDmap_reference_guided_preview_v1_fixed_001/index.html`为v1。候选区间/完整连接均与原记录核对，可切局部轨迹/全车道，播放只读保存的实际积分。旧preview001/002候选计数字段错误及旧QA范围见勘误，不能当最新版。新测试仍关闭，语义到达信用0，论文正文留后续GPT Work。


## 主450保存证据及初态重建增量

主矩阵450次已完整结束并独立复核（2026-11-12T19:38:19.906280+00:00）。449个完整结果、1个240秒进程超时，失败保留在全部分母。每模型90次可达正试验，规则为83路线/84曾选对/82曾正确锁定/50实际到达，原始Qwen为66/72/66/39，新Adapter为59/33/32/4。选对和锁定为保存观测中的曾发生，分别重算，不能代替到达。新Adapter未提升，保留负结果；原始Qwen减新Adapter到达率差38.89个百分点，10配置配对95%区间18.89至60.00。

严格初态审计395车身证书、54缺零时刻观测未知、1缺完整结果，原93到达中86获严格车身/方向及2秒保持。另立种子初态重建版本：54例对应68条保存静止前缀，独立PCG64初态交叉复核、41523新增审核区间；复用旧292406区间后，共449完整试验333929区间。359行驶/90静止/1未知，121585运动区间全线段航向锥及243170次GEOS进度复核；93原到达全部通过重建版本车身、方向和保守身份/终点2秒保持，没有新增执行、模型或到达。54初态为重建，不能称已保存零时刻观测；严格旧未知和7个未决保持报告均保留。

当前450缺一语义消融已实际启动，本次快照保存8/450；后续合并复用主矩阵300记录为750行，不是750新增试验。正式GPU故障和独占性能继续等前置完成。真实人工及合格新测试指令分别以API快照为准，新封存仍关闭。实验与交付优先，不写论文正文/框架；最终实验核验后同步GitHub。业务380源码未改，原机424/纯代码422通过2数据skip的结果继续有效；本轮只新增辅助审核及交接脚本，不冒称重新全量测试。v14包不包含本轮450增量，当前新证据另见《主矩阵450_实验核验交接_001.md》。

辅助入口在`source_bundle/main_matrix_evidence_check_001.py`、`main_body_evidence_check_001.py`、`main_direction_evidence_check_001.py`。严格原审核54缺初态与7到达未决保留。`main_seed_initial_reconstruction_001.py --run-id 新目录 [--limit 2]`重建；`main_seed_initial_ledger_001.py`显式PCG64复核。`main_seed_direction_audit_001.py --cohort main --run-id 新目录 [--limit 10]`及`main_seed_direction_evidence_check_001.py`审核方向。`main_goal_hold_evidence_check_001.py --body-run independent_body_main_seed_initial_001 --run-id 新目录`审核保守终点保持。固定输入/输出的辅助脚本复跑需在独立副本另立版本，不能直接覆盖本机证据。所有辅助工作隐藏CUDA、单CPU线程，不新增模型/执行/到达。


## 主450只读查看器与第8版实验交接

复现入口（在项目根目录、既有conceptgraph环境中，使用独立run-id；不得覆盖已存在目录）：

```bash
export LD_LIBRARY_PATH=/root/miniconda3/envs/conceptgraph/lib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
/root/miniconda3/envs/conceptgraph/bin/python -m pytest -q tests
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.realtime_matrix_demo --run-id 新唯一smoke目录 --limit 6 --wall-budget-s 900
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.realtime_matrix_demo --run-id 新唯一全量目录 --wall-budget-s 900
```

复现依赖完整主450原始感知与控制产物及已生成的分版本独立证书。无reuse模式重新核验全部输入；`--reuse-viewer 已完成查看器目录名`仅在旧生成物和指纹一致时复用原核验清单，必须另用run-id，不能把复用称重跑模型。总元数据预算128MiB、最低剩余6GiB、只读CPU共享资源，不接管GPU队列。查看通过HTTP服务，当前8768已运行，远程需转发该端口：

`http://localhost:8768/visualizations/realtime_main450_preview_003/index.html`

真实人工开发审核：`http://localhost:8768/visualizations/causal_development_review_002/index.html`；自然表达审核：`http://localhost:8768/annotation/natural_command_review_001/index.html`。人工填写审核后才产生真实记录，自动浏览检查不代填。新封存测试仍关闭。

后续顺序：450新增缺一消融及独立统计→GPU故障smoke与180全量→独占GPU性能→可靠人工/新测试数据审核、方法冻结后新封存→完整可视化/保护/复现/导出核验→GitHub同步与最终实验交接。当前不写论文文字。


## 统一入口013及源集合兼容003

统一演示入口：`http://localhost:8768/visualizations/thesis_portal_013/index.html`，需转发远程8768端口。原主450专门查看器仍为`/visualizations/realtime_main450_preview_003/index.html`。门户复现使用唯一run-id：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.thesis_portal --run-id 新唯一门户目录 --region-run reference_regions_reserved_full_003 --region-preview reference_regions_reserved_001 --reference-run reference_reachability_reserved_full_002 --reference-preview reference_reachability_reserved_002
```

源集合兼容等待器已运行，`ablations/semantic_navigation_chain_003/worker_started.json`记录PID/ticks，不要在同目录重复启动。003脚本--prepare-only/--resume是两阶段入口，固定run目录，只适合原实例或隔离副本重现；新运行必须使用新版本/路径，先检查进程和全部指纹。冻结原66与额外文件均不可改。旧`semantic_ablation_after_main_001.py`将保留原字典全等断言失败；只有真正450完成、750组合及独立审核通过，才写原等待器要求的兼容桥接`ablations/semantic_navigation_chain_001/complete.json`。桥接显式标记original_script_success=false，不冒称原脚本成功。完成前桥接必须不存在。

独立复算入口`source_bundle/semantic_compatibility_evidence_check_003.py`有--input-run、--run-id、--expected、--methods参数，先用实际主450完成回归，再对实际750运行；不能用部分消融记录替代全量。无新增模型/执行/到达。所有后续GPU队列仍按原冻结设置串行。


## 主450失败环节诊断

主450失败环节诊断更新（2026-11-12T21:07:34.759744+00:00），仅实验与复现交接，不写论文正文/框架。450试验、57,460感知帧、1,317已保存投递响应、2,672原输入/产物SHA、27项配置配对区间已独立核对。001首次诊断发现旧evaluation_identity在selected=None时直接保存空mapping，不能据此判漏检；原失败完整保留。002从保存节点与冻结目标重建全部类别/2.5m Hungarian关联；30,538个原空映射帧其实有可关联节点，13,922个selected非空帧重建与原保存映射完全相同。重建不是新增观测或人工真值；所有原完整试验四项业务结果精确不变，1失败前缀不给完整成功信用。

每方法90可达正条件中，三方法都曾建图关联到可接受目标；安全检查前曾含正确候选为规则90/base90/new54，通过安全检查曾含正确候选84/73/33，原最终Top1正确84/72/33、连续3帧selected正确82/66/32、实际到达50/39/4均保持原值。base的安全候选73含1失败前缀，完整Top1分子仍72。不同时间存在事件不可当严格单调漏斗；安全前候选已受解析/属性影响，不是独立纯规则候选。

可达正条件中base保存90个解析响应均保留颜色且无额外锚点/关系，其中1例缺完整事件日志，完整事件应用只知89加1未知；新Adapter90次后端入口只有74响应投递并应用，16未投递，不能给未知输出语义评分。74已投递均有同名near锚点、52丢失目标颜色，仅22保留颜色，完整任务解析诊断0/74；格式通过74/74不等于语义正确。全部种类中新Adapter134已投递解析均有同名near锚点，不能据此声称全部150后端原文都相同。正任务属性base335/new233已投递响应均为unknown；新Adapter有3次排序实际应用但没有改变任何帧的最终首位，base完整事件排序应用0。不能将选择变差全部归因于重排。

训练1400解析有145条无锚点/关系且含颜色，验证180有24条；与本合成任务精确查询文字重合均0。训练7条、验证1条近关系的锚点/目标文字同名，不自动证明指代相同或标签错误。监督来自声明语法而非人工自然指令，这些分布与应用现象只是后验解释；模块因果贡献仍等正在运行的450缺一，当前快照61/450。GPU故障/独占性能继续原队列，人工和合格封存指令按API记录，新封存未开启，Goal为false。业务381源码未改，原424测试/导出422通过2数据skip继续有效，本轮辅助诊断实际执行和独立审计通过，不冒称重复全量测试。

复现见《主450失败环节_实验诊断_001.md》的分母与复现。三种诊断均不重跑模型，不给予新到达信用。


## 当前可启动交互仿真

交互实验交付更新（2026-11-12T21:33:31.648770+00:00）：新增可实际启动、单步0.5秒、连续运行、暂停和制动停车的CPU交互入口。桌面/手机实际操作同一预定任务config00-positive0/seed17，共14次真实前端、67控制区间/52运动区间、10条已应用操作、11种RGB与11种深度；暂停状态保持，7步有界制动到零。独立复数圆弧、连续矩形车身和全圆弧方向审核通过，111原始文件SHA一致。终止USER_STOPPED，生成路线/曾选对/曾正确锁定为是、实际到达为否；不并入主450或增加独立任务、人工标签、Qwen/GPU/性能信用。

当前385源码；原机439通过，纯代码437通过2真实数据skip，业务Ruff通过。统一查看入口已验收thesis_portal_015（桌面/手机69链接），新增交互服务需转发8769，已有审核服务8768保留。历史v14不包含本次交互及主450增量；当前源码尚未推送，待最终实验核验后同步。缺一新增450当前快照75/450，其余故障/独占性能继续原队列；人工及合格自然指令以API为准，新封存未开，Goal与四阶段仍为false。本阶段不写论文正文或框架。

具体启动、预算、日志和异机依赖见《交互仿真_操作与复现_001.md》。


## 连续KITTI复现命令（使用独立run-id，不覆盖已有结果）

数据准备的首次官方格式失败001与修正002、原选样、网络Range/ETag/CRC证据均保留。已有90帧无需重复下载。独立环境先按来源、许可和manifest准备本地模型、依赖和同一数据；不得把原始RGB/点云/标签或模型权重提交Git。

```bash
cd /root/autodl-tmp/lizichong
export LD_LIBRARY_PATH=/root/miniconda3/envs/conceptgraph/lib
export CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.kitti_tracking_experiment frontend --agent-manifest outputs/project_v2/thesis_completion_v1/data_manifest/kitti_tracking_center90_002/agent_manifest.json --run-id kitti_tracking_frontend_reproduce_001 --wall-budget-s 3600 --storage-budget-gb 3
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.kitti_tracking_experiment map --agent-manifest outputs/project_v2/thesis_completion_v1/data_manifest/kitti_tracking_center90_002/agent_manifest.json --frontend-run kitti_tracking_frontend_reproduce_001 --run-id kitti_tracking_mapping_reproduce_001
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.kitti_tracking_experiment evaluate --input-run kitti_tracking_center90_002 --mapping-run kitti_tracking_mapping_reproduce_001 --run-id kitti_tracking_statistics_reproduce_001
```

再复跑须更换全部独立run-id。固定002数据准备和首次数据/浏览器audit脚本含保护性exist_ok=False，跨机先核对实际路径、预建目录和版本；不能直接覆盖已有记录。模型大文件SHA见前端release；异机共享CPU耗时不作为当前GPU性能。
