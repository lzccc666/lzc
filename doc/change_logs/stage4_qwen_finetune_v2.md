# Stage 4 Qwen 语义委托器 LoRA 微调 v2

- 完成日期：2026-09-02
- 项目根目录：`/root/autodl-tmp/lizichong`
- 输出根目录：`/root/autodl-tmp/lizichong/outputs/project_v2/qwen_eval/finetune_v2`
- 模型：本地 `checkpoint/Qwen2-VL-2B-Instruct`
- 设备：NVIDIA GeForce RTX 4090
- 状态：训练、独立验证、一次性封存测试和双场景系统回归均通过

## 1. 实施范围

本阶段实际训练新的导航语义 LoRA，而不是沿用旧 adapter 或把规则回退计入模型准确率。新增了 v2 提示协议、数据审计、真实 BF16 LoRA 训练、可恢复检查点、逐任务严格指标、两随机种子汇总、一次性封存测试、误差与确定性报告，以及 `scene-0916`、`scene-0003` 的无缓存真实后端回归。

旧 `stage4` 提示仍是委托器默认值；新提示必须显式设置 `prompt_profile="finetune_v2"`。部署回归显式使用与训练一致的 BF16 和 448 像素图像长边，旧调用的默认推理精度和原图行为不变。

本阶段没有下载模型或数据，没有运行检测、深度生成、建图、nuScenes 解压或论文流程，没有删除用户文件，也没有提交或推送 Git。基础模型、旧 adapter、原始 Stage 4 数据集及 Stage 1--6 产物均只读保护。

## 2. 新增与修改

新增：

- `conceptgraph/vlm/qwen_finetune_v2.py`：v2 提示、数据审计、严格原始输出指标、硬门槛和误差分析。
- `scripts/train_qwen_lora_semantic_delegate_v2.py`：BF16 LoRA 训练、验证、断点恢复、冻结变体评测和封存测试保护。
- `conceptgraph/scripts/finalize_qwen_finetune_v2.py`：正式种子聚合、最佳 adapter 晋升、封存测试和基线报告整理。
- `conceptgraph/scripts/run_qwen_finetune_v2_downstream_regression.py`：双场景、三任务、无缓存真实后端回归。
- `conceptgraph/scripts/run_qwen_finetune_v2_qa.py`：故障测试、Stage 4/5/6 回归、全量测试和静态检查汇总。
- `tests/vlm/test_qwen_finetune_v2.py`：提示、颜色顺序、增强、指标、委托器和图像缩放测试。

修改：

- `conceptgraph/vlm/qwen_semantic_delegate.py`：增加显式 `finetune_v2` 提示档位，默认 `stage4` 行为保持不变。
- `conceptgraph/vlm/qwen_vl_chat.py`：增加默认关闭的推理 dtype 和最长图像边选项，并显式设置生成 `pad_token_id`。

## 3. 数据审计

原始数据共 735 条、11 个 scene，scene 泄漏为 0，sample ID 重复为 0，manifest 不一致为 0，426 张 crop 全部存在、可解码且内容哈希一致。

| 任务 | train | validation | sealed test |
| --- | ---: | ---: | ---: |
| query_parse | 115 | 31 | 9 |
| object_attributes | 354 | 62 | 10 |
| candidate_rerank | 115 | 31 | 8 |
| 合计 | 584 | 124 | 27 |

处理器审计仅检查 train/validation 的 708 条样本，没有读取测试标签。最大输入长度为 1486，小于 2048；所有对象样本均产生真实视觉 token，示例 `pixel_values=[576,1176]`、`image_grid_thw=[1,3]`，没有截断监督答案。

颜色稀缺问题只用训练行构造 90 条“有序颜色对复制协议”文本增强，覆盖 10 种规范颜色的 90 个有序二元组。增强行明确标记为 synthetic，只训练 JSON 复制约束，不声称图像包含对应颜色；原始数据、validation 和 test 均未修改。最终有效训练行数为 674。

### 弱监督与重复披露

标签不是人工真值：对象类别与检测标签、运动状态与元数据、颜色与 caption 派生结果的相等率均为 100%；rerank 正确目标原本位于输入首位的比例为 92.8571%，同时具有最高既有组合分数的比例为 88.9610%。

scene 隔离成立，但精确输入重复组在 query_parse、candidate_rerank、object_attributes 中分别为 18、13、0，跨 split 还存在感知近似 crop。因此本报告中的准确率只代表既有弱监督分布内能力，不构成开放世界或独立人工标注泛化结论。

## 4. 训练配置与可恢复性

正式训练固定配置除 seed 外完全一致，配置哈希为：

`fdeea3d3bc2c9a582313989cc90b3d46a857aaeab5876e121c96328cf0175590`

核心配置为 BF16、LoRA `r=16`、`alpha=32`、`dropout=0.05`、学习率 `1e-4`、micro batch 1、梯度累积 8、梯度裁剪 1.0、最长 2048 token、每 80 步验证、最多 240 步并按全部硬门槛早停。任务权重为 query 1.5、object 1.0、rerank 1.2，稀有多颜色原始样本提升系数为 64；synthetic 行不参与该提升。

可训练参数 18,464,768，总参数 2,227,450,368，可训练比例 0.828964%。正式两次训练均在第 80 步早停，峰值 PyTorch 分配显存约 7242 MiB：

| seed | 训练时间 | 最佳检查点 | 验证门槛 |
| ---: | ---: | --- | --- |
| 73 | 377.880 s | `runs/seed_73/checkpoints/step_0080` | 全部通过 |
| 79 | 368.052 s | `runs/seed_79/checkpoints/step_0080` | 全部通过 |

总训练时间 745.931 s，均值 372.966 s，标准差 4.914 s。每个检查点包含 adapter、processor、optimizer、scheduler、Python/NumPy/Torch CPU/CUDA RNG 和训练曲线，可继续恢复。探索运行与正式两种子统计分开记录，没有混入最终均值。

最佳 adapter 按验证综合分数选择，完全相同时以较小 seed 破平，因此来自 seed 73。测试集没有参与训练、早停、超参数选择或最佳模型选择。权重 SHA-256 为：

`4b7117a93272b46ac67c697a0d6a961ae067ed0c4e4eb327ef3717141623610e`

## 5. 基线与验证结果

所有指标只统计原始模型输出；schema 错误、非法候选、超时或异常一律记失败，规则 fallback 不获得准确率信用。

| validation 指标 | 基础模型 | 旧 adapter | 新 LoRA 两 seed mean/std/worst |
| --- | ---: | ---: | ---: |
| 总 schema 有效率 | 0.7258 | 0.3548 | 1.0000 / 0 / 1.0000 |
| 最低单任务 schema 有效率 | 0.0323 | 0.1935 | 1.0000 / 0 / 1.0000 |
| query macro-F1 | 0.3716 | 0.3787 | 1.0000 / 0 / 1.0000 |
| query exact match | 0 | 0 | 1.0000 / 0 / 1.0000 |
| object category macro-F1 | 0.7877 | 0.3429 | 1.0000 / 0 / 1.0000 |
| object color macro-F1 | 0.5774 | 0.2682 | 1.0000 / 0 / 1.0000 |
| object motion macro-F1 | 0.9845 | 0.6647 | 1.0000 / 0 / 1.0000 |
| rerank Top-1 | 0.8065 | 0.1935 | 0.9032 / 0 / 0.9032 |
| rerank MRR | 0.8677 | 0.1935 | 0.9323 / 0 / 0.9323 |
| rerank 候选合法率 | 0.9677 | 0.1935 | 1.0000 / 0 / 1.0000 |
| schema/非法 fallback 率 | 0.2742 | 0.6452 | 0 / 0 / 0 |

两个正式 seed 都通过总 schema 不低于 0.95、各任务 schema 不低于 0.90、query macro-F1 不低于 0.90、query exact 不低于 0.80、对象 category/color/motion 不低于 0.85/0.80/0.85、rerank Top-1/MRR 不低于 0.85/0.90、候选合法率 100% 和 fallback 不高于 5% 的全部门槛。

## 6. 一次性封存测试

在配置、两个正式检查点和最佳 adapter 全部冻结后，运行器创建 `sealed_test_release.json`，只开启一次测试集。评测执行 27 条测试样本、2 个 seed、每个 seed 2 次确定性重复，共 108 次原始生成；此后运行器拒绝再次执行测试评测。

两个 seed 的总 schema、三个单任务 schema、query macro-F1/exact、对象 category/color/motion、rerank Top-1/MRR/候选合法率全部为 1.0000，fallback 为 0。全部聚合指标的两 seed mean=1.0000、std=0、worst=1.0000；四次原始输出内容哈希相同：

`119e7b2e837f7a3a908f05263bfa7f44adf79722973bc7b1e1f3e463bc8481cc`

误差分析中两个 seed 的失败或非精确样本数均为 0，schema 错误计数为空。

## 7. 双场景系统回归

最佳 adapter 在 `scene-0916` 和 `scene-0003` 上分别执行 query_parse、candidate_rerank 和真实 crop object_attributes，共 6 次真实后端调用。缓存关闭，生成非采样，BF16，图像长边 448。

| 场景 | 原始 schema | fallback | 候选合法 | 选择节点 | 路径 |
| --- | --- | ---: | --- | ---: | --- |
| scene-0916 | 3/3 | 0 | 是 | 21 | 连续、无碰撞 |
| scene-0003 | 3/3 | 0 | 是 | 11 | 连续、无碰撞 |

两个场景都属于训练 split，所以这里只声明系统集成与导航安全回归通过，不把它们计作封存泛化准确率。Qwen 只能在已验证候选内建议重排，Stage 2 联合效用、可达性、A* 和连续碰撞检查仍是权威路径决策。

## 8. 故障、回归与质量门禁

委托器专项覆盖 Qwen 关闭、超时、异常、空输出、非法 schema、非法候选和超时后的迟到结果；全部情况都不采纳不合格输出，并保留规则导航安全结果。

- v2 新增专项：10 passed。
- 故障与迟到结果专项：17 passed。
- Stage 4/5/6 相关测试：102 passed。
- `tests` 全量：144 passed。
- 目标改动文件 Ruff：通过。
- `git diff --check`：通过。
- 唯一警告为 2 条既有 torchvision Beta API 提示。

机器可读结果写入 `qa_report.json` 并合并到最终 `validation_report.json`；既有 Stage 4/5/6 报告仅被读取，未被重写。

## 9. 关键产物

- `best_adapter/`：最佳 seed 73 adapter 和完整可恢复状态。
- `training_report.json`：两正式 seed 及独立探索运行记录。
- `validation_report.json`：两 seed 验证均值、标准差、最差值和最终门禁。
- `heldout_evaluation.json`：一次性封存测试汇总。
- `per_task_metrics.json`：封存集逐任务指标。
- `per_scene_metrics.json`：由冻结原始预测重算的 validation/test 逐 scene 指标。
- `error_analysis.json`：混淆和失败样本分析。
- `determinism_report.json`：两 seed、两重复内容与指标哈希。
- `downstream_regression.json`：双场景六次无缓存真实调用及路径检查。
- `data_audit.json`、`processor_length_audit.json`：数据、标签代理、重复、crop 和 token 审计。
- `baseline_verification.json`：保护对象复核。
- `qa_report.json`：最终测试、故障和静态检查结果。

## 10. 基线保护与限制

基础模型、旧 adapter、Stage 1--6 保护目录、Stage 4 数据集和 `scene-0003` 输入聚合哈希与开始前完全一致。`scene-0916` 输入和旧 `qwen_eval`（排除本目录）的聚合 manifest 哈希与开始时记录不一致，但文件数、总字节完全一致，Stage 4 关键文件单体哈希一致，且没有任何文件 mtime 晚于本轮基线。报告因此记录 `strict_all_manifest_hashes_match=false`，同时以取证条件确认 `forensic_no_overwrite_check_passed=true`；该警告没有被隐藏或改写为严格哈希一致。

主要限制如下：

- 数据标签是自动弱监督代理，不是独立人工标注真值。
- query 和 rerank 存在跨 scene 模板/输入重复，测试满分不能外推为开放世界能力。
- 封存测试只有 27 条样本，置信区间有限。
- 双场景端到端回归属于训练 scene，只证明部署链路、schema、候选边界和路径安全行为。
- Conda 环境需要优先加载 `/root/miniconda3/envs/conceptgraph/lib` 的 `libstdc++`；`libtinfo` 版本提示和 torchvision Beta 警告不影响结果。
