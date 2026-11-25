# 阶段4：Qwen语义模块

## 目标与实施范围

本阶段在阶段2目标选择和阶段3结构化关系边之上加入可选的 Qwen2-VL 语义代理，完成严格结构化输出、磁盘缓存、规则回退、导航接入、异步接口、多模态数据集和训练入口，并使用本地 Qwen2-VL 与既有 LoRA adapter 做双真实场景离线验收。

本阶段没有运行目标检测、深度生成、建图、nuScenes 解压、阶段5状态机或无界训练；没有覆盖旧数据集、基础模型、LoRA adapter、旧实验结果或阶段1至阶段3产物；没有修改论文内容，也没有提交或推送 Git。

统一输出根目录为：

`/root/autodl-tmp/lizichong/outputs/project_v2/qwen_eval`

## 文件和接口变化

新增文件：

- `conceptgraph/vlm/schemas.py`：三个严格 Pydantic schema、规范化和鲁棒 JSON 提取。
- `conceptgraph/vlm/qwen_cache.py`：规范 JSON、SHA-256 键、校验和及原子磁盘缓存。
- `conceptgraph/hydra_configs/qwen_semantic.yaml`：默认关闭的模型、任务、缓存、超时和确定性生成配置。
- `conceptgraph/scripts/run_stage4_offline_qwen_eval.py`：双真实场景规则/Qwen/缓存/故障注入评估。
- `tests/vlm/test_qwen_schemas.py`、`test_qwen_cache.py`、`test_qwen_fallback.py`、`test_qwen_navigation_integration.py`。

修改文件：

- `conceptgraph/vlm/qwen_semantic_delegate.py`：改为可注入 backend、惰性加载、严格校验、缓存、超时、规则回退和异步接口。
- `conceptgraph/vlm/qwen_vl_chat.py`：仅加载本地模型和 adapter，固定非采样生成，并真正传入图像。
- `conceptgraph/vlm/__init__.py`：导出 schema、缓存和 delegate 接口。
- `conceptgraph/navigation/semantic.py`：将已验证的 Qwen 查询字段转换为现有 `ParsedNavigationQuery`。
- `conceptgraph/navigation/pipeline.py`：导航开始时只解析一次；仅把已验证、已有合法路径的候选交给 Qwen 作建议性重排。
- `conceptgraph/hydra_configs/semantic_navigation.yaml`：组合默认关闭的 `qwen_semantic` 配置。
- `conceptgraph/scripts/build_qwen_finetune_dataset.py`：真实 crop、scene 划分、分任务输出、审计和确定性哈希。
- `scripts/train_qwen_lora_semantic_delegate.py`：读取 split manifest、真实多模态处理、任务权重及逐任务/逐场景指标。

旧 CLI 名称仍可迁移；旧数据集和旧 adapter 路径及其子目录被显式保护。新训练默认写入：

`/root/autodl-tmp/lizichong/outputs/project_v2/qwen_eval/sanity_adapter`

本次只执行真实 AutoProcessor smoke，没有执行训练。

## 严格 schema

统一 `schema_version` 为 `1.0.0`，三个 schema 均使用严格类型和 `extra=forbid`：

- `query_parse`：目标/锚点文本、最多 4 个类别和颜色、规范关系、两个布尔偏好及 `[0,1]` 置信度。
- `object_attributes`：摘要、规范类别、1 至 8 个候选标签、颜色、可选车身类型、动静态状态及 `[0,1]` 置信度。
- `candidate_rerank`：非负选中节点、非空且无重复的排序、可选解释及 `[0,1]` 置信度；首项必须等于选中节点。

关系统一为 `near/in_front_of/behind/left_of/right_of`，颜色、类别、车身类型、运动状态和布尔词均由集中映射规范化。解析支持裸 JSON、Markdown JSON 代码块和单个带前后文本的 JSON；多个不同 JSON、错误顶层类型、额外字段、错误类型、越界置信度和非法候选 ID 均拒绝。

每次调用统一记录 `raw_output`、`normalized_output`、`validation_errors`、`output_schema_valid`、`used_qwen`、`fallback_used`、`fallback_reason`、`failure_code`、前后端耗时、缓存状态和模型/adapter 标识。异常中的绝对路径会脱敏，审计不保存 crop 绝对路径或完整输入提示。

## 缓存与确定性

缓存键使用规范 JSON 和 SHA-256，联合以下维度：

- task 和 schema 版本；
- 模型和 adapter 版本；
- 提示版本与规范化输入；
- 图像内容哈希和对象版本；
- 候选内容及顺序；
- 结构化边版本；
- 缓存格式版本。

字符串先做 NFC 规范化，字典键稳定排序，禁止 NaN 和无穷值。缓存记录保存 payload 校验和；损坏、格式版本不匹配、不完整和校验和错误均诊断为未命中并重新计算。写入采用同目录临时文件、`fsync`、`os.replace` 和目录 `fsync`，同进程写入由锁串行化。

真实场景缓存复跑的三个任务全部命中，新增 backend 调用数为 0。首次和缓存语义快照哈希均为：

`872878a58671adf2fcf9b2d5d99e6d032b1e59c74ef7ebd5f1d4b09912a035a1`

数据集连续重建时三个核心文件的联合 SHA-256 前后均为：

`3f2d4e5ea76f1cda1583617391120b0f3a1337f25da08eaba64e83b8c1796518`

## 回退与异步行为

稳定回退覆盖：`QWEN_DISABLED`、`IMAGE_UNAVAILABLE`、`EMPTY_OUTPUT`、`SCHEMA_VALIDATION_FAILED`、`ILLEGAL_CANDIDATE_ID`、`BACKEND_TIMEOUT`、`BACKEND_EXCEPTION`、`NO_CANDIDATES` 和导航侧 `DELEGATE_EXCEPTION`。

回退值来自当前规则解析、规则对象属性或阶段2候选顺序。`infer_relation` 只保留兼容入口；存在阶段3结构化事实时直接返回该事实，不调用 Qwen，也不允许语言模型覆盖几何关系。

同步 backend 运行在有界单线程执行器中，超时后退休该执行器，避免后续请求被旧长任务阻塞；并发替换、关闭和任务计数受锁保护。`parse_query_async`、`describe_object_async` 和 `rerank_candidates_async` 使用独立后台执行器。延迟 backend 测试验证事件循环不阻塞，超时会回退，取消和 `close/aclose` 后无追踪中的异步任务。

## 导航集成

导航先执行一次现有规则解析，再在 delegate 存在时只调用一次 `parse_query`。只有通过严格 schema 的 Qwen 输出才更新 `ParsedNavigationQuery` 的语义字段；Qwen 关闭、失败或非法时直接保留规则对象。

候选重排发生在阶段2完成联合效用、可达点搜索、A* 和连续碰撞检查之后。输入只包含 `reachable=True` 且至少有一条合法路径的候选，返回 ID 必须属于该集合。重排结果仅写入审计，`selection_applied=false`；阶段2联合效用与已验证路径始终决定实际路线，因此 Qwen 无法改变可达性、碰撞检查或 A* 结果。

## 数据集与训练入口

数据集位于：

`/root/autodl-tmp/lizichong/outputs/project_v2/qwen_eval/dataset`

共 735 条、11 个场景：

| 任务 | 样本数 |
| --- | ---: |
| object_attributes | 426 |
| query_parse | 155 |
| candidate_rerank | 154 |

scene 级确定性划分为训练 8、验证 2、测试 1：测试为 `scene-0128`，验证为 `scene-0133/scene-0757`，其余 8 个场景用于训练。scene 泄漏数和 manifest 分配不一致数均为 0。split manifest 哈希为：

`edb04efa6b32548d68f3208bafeaf5ed3686ac6d66e4237da00899f839a1106d`

数据集内容哈希为：

`eb2082b18a11203c4e49e07a89324aa516a64366c61710ac68ad388f65c17b06`

426 个对象样本的 crop 均真实存在、可解码并带内容哈希；缺失和损坏数均为 0。`candidate_rerank` 保存真实候选节点、阶段2分数、阶段3关系元数据、既有已验证目标和标签来源，不再产生空候选。

训练加载器严格核对 scene、split 和 sample ID 是否登记在 manifest 中，以任务权重做确定性采样，并分别输出逐任务和逐场景指标。对象属性样本以 PIL 图像进入 `AutoProcessor`。真实 smoke 使用 `scene-0003` 的 crop，生成 `pixel_values=[1248,1176]`、`image_grid_thw=[1,3]` 和 63 个监督 token，证明不是只把路径写进 JSON。

所有标签来自已有自动规则、旧 Qwen 场景图或既有导航结果，仅属于弱监督。颜色、细分类、关系和正确目标可能带有系统性噪声，不等价于人工真值。

## 双真实场景推理

实际加载：

- 模型：`Qwen2-VL-2B-Instruct@local-qwen2-vl-2b-instruct`
- adapter：`qwen_lora_semantic_delegate@local-semantic-delegate-lora`
- 设备：RTX 4090，`cuda:0`
- 生成：`do_sample=false`，`max_new_tokens=192`
- 网络下载：关闭

每个场景运行规则基线、Qwen query 解析、真实 crop 对象属性、候选重排、三任务故障注入和缓存复跑：

| 场景 | 规则选择 | Qwen 导航选择 | 首次 backend 调用 | 缓存复跑调用 | 故障回退 |
| --- | ---: | ---: | ---: | ---: | --- |
| scene-0916 | 21 | 21 | 3 | 0 | 通过，仍选 21 |
| scene-0003 | 11 | 11 | 2（查询跨场景命中） | 0 | 通过，仍选 11 |

真实 backend 总调用数为 5。6 个首次任务样本的延迟中位数为 4.7213 秒，P95 为 7.3703 秒，最大 7.7047 秒。缓存复跑每场景命中率均为 1.0。

代码审查修复完成后又执行了一次无缓存真实查询探针，最终 delegate 成功加载同一模型和 adapter，backend 调用数严格为 1，结果按预期因 schema 不合格安全回退。

既有 adapter 的 6 个原始输出均未通过本阶段更严格的 schema，主要表现为重复/额外字段或解释文本。因此原始 `schema-valid`、字段准确率、rerank Top-1/MRR 均为 0，fallback 率为 1.0；规则回退后的有效字段准确率、Top-1 和 MRR 均为 1.0。该结果证明真实模型被调用且安全边界有效，但不证明既有 adapter 已具备新 schema 的可直接部署质量。后续如需提升模型采纳率，应基于本阶段 scene 隔离数据重新训练并独立评估，不能把回退指标当作模型指标。

总对比报告剔除耗时后的确定性内容哈希为：

`f6046d10f62983fd057512c6db8c86e08a859990df6b07ceff1b4df679e53aed`

## 回归、测试与代码审查

- VLM 专项：51 passed。
- `tests` 全量：110 passed，只有 torchvision Beta API 的 2 条上游警告。
- 阶段4及全部当前改动 Python 文件 Ruff：通过。
- `git diff --check`：通过。
- 阶段2双模式真实回归：`scene-0916` 为 `24→21`，`scene-0003` 为 `9→11`，四次选择全部成功。
- 阶段3真实回归：两场景成功，增量/全量最终一致率均为 1.0，四个结构化关系查询全部成功且几何 fallback 为 0。
- `/root/autodl-tmp/lizichong/outputs/project_v2/qwen_eval` 下 108 个 JSON 文件全部可重新解析。

人工审查修复了零值候选效用排序、显式 scene 顺序规范化、manifest 外样本拒绝、旧产物子目录保护、异常绝对路径脱敏，以及并发超时执行器替换/关闭竞争。修复后重新执行专项测试、全量测试、Ruff、Stage 2/3 回归、数据重建和真实 processor smoke，未发现剩余功能性问题。

仓库全量 `ruff check conceptgraph scripts tests` 仍报告 136 个既有历史问题，集中在未参与本阶段的 `ai2thor`、`llava`、旧脚本和工具模块；本阶段没有以无关大范围格式化修改它们。阶段4文件及当前工作树所有改动 Python 文件均通过 Ruff。

## 产物与复现

关键绝对路径：

- `/root/autodl-tmp/lizichong/outputs/project_v2/qwen_eval/baseline_before_stage4.json`
- `/root/autodl-tmp/lizichong/outputs/project_v2/qwen_eval/dataset/`
- `/root/autodl-tmp/lizichong/outputs/project_v2/qwen_eval/cache/`
- `/root/autodl-tmp/lizichong/outputs/project_v2/qwen_eval/rule_baseline/`
- `/root/autodl-tmp/lizichong/outputs/project_v2/qwen_eval/qwen_inference/`
- `/root/autodl-tmp/lizichong/outputs/project_v2/qwen_eval/failure_fallback/`
- `/root/autodl-tmp/lizichong/outputs/project_v2/qwen_eval/stage4_comparison.json`
- `/root/autodl-tmp/lizichong/outputs/project_v2/qwen_eval/determinism_report.json`
- `/root/autodl-tmp/lizichong/outputs/project_v2/qwen_eval/baseline_verification.json`
- `/root/autodl-tmp/lizichong/outputs/project_v2/qwen_eval/validation_report.json`

复现命令需在项目根目录执行；本机需显式优先使用 Conda 环境的 `libstdc++`：

```bash
export LD_LIBRARY_PATH=/root/miniconda3/envs/conceptgraph/lib
conda run -n conceptgraph python -m conceptgraph.scripts.build_qwen_finetune_dataset
conda run -n conceptgraph python scripts/train_qwen_lora_semantic_delegate.py --processor-smoke-only
conda run -n conceptgraph python -m conceptgraph.scripts.run_stage4_offline_qwen_eval
conda run -n conceptgraph pytest -q tests
conda run -n conceptgraph ruff check conceptgraph/vlm conceptgraph/navigation/semantic.py conceptgraph/navigation/pipeline.py conceptgraph/scripts/build_qwen_finetune_dataset.py conceptgraph/scripts/run_stage4_offline_qwen_eval.py scripts/train_qwen_lora_semantic_delegate.py tests/vlm
git diff --check
```

阶段2/3 回归：

```bash
conda run -n conceptgraph python -m conceptgraph.scripts.run_stage2_offline_navigation --scene-count 2 --output-root outputs/project_v2/qwen_eval/regression_stage2
conda run -n conceptgraph python -m conceptgraph.scripts.run_stage3_offline_relations --scene-count 2 --output-root outputs/project_v2/qwen_eval/regression_stage3 --skip-visualizations
```

## 基线保护、限制与 Git 状态

修改前记录位于 `baseline_before_stage4.json`，完成后核验位于 `baseline_verification.json`。阶段1至阶段3三个输出目录、旧 Qwen 数据集、旧 Qwen 评估、基础模型、既有 LoRA adapter、`MEMORY.md`、`PAPER_MEMORY.md`、原有 5 个文档和两个真实场景输入的 SHA-256 均保持不变，`baseline_verification.passed=true`。

资源限制与未解决问题：

- 真实推理只覆盖两个场景、每场景三个任务，是有界离线验收，不代表完整数据集模型评估。
- 未执行 LoRA 训练；只验证了训练数据、manifest、任务采样和真实图像处理链路。
- 既有 adapter 尚不能稳定产生新严格 schema，当前必须依赖规则回退；这是模型质量限制，不是导航安全阻塞。
- 基础 shell 偶尔优先加载系统 `libstdc++.so.6`，验收命令通过显式 `LD_LIBRARY_PATH` 固定到 `conceptgraph` Conda 环境；伴随的 `libtinfo` 版本提示不影响 Python、测试或推理结果。

当前分支为 `main`，HEAD 为 `eab10f5aebfe558d1e50fae543efcce3c27e613f`。工作树在阶段4开始前已包含阶段1至阶段3和用户文档的未提交改动；这些内容全部保留。本阶段未提交、未推送、未删除用户文件。
