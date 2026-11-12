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
