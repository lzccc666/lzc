# 最新复现与交付入口

当前完整机器实验的结果以交接024为准；下方历史命令中的等待状态已过时，原run-id不可覆盖。重跑必须选新目录，并保留原SHA、随机种子及失败分母。原始权重/传感器数据不随Git上传。

- 当前源码：`source_bundle/source_checkpoint_038/source_current.tar.gz`，390文件；业务运行源码与037相同，仅一个测试夹具修正。
- 新环境实测回执：`reproduction/isolated_environment_after_GPU_005/complete.json`。准备/隔离/实际导航的完整argv在其jobs字段；安装004、追加CLIP的来源哈希见`reproduction/isolated_cpu_environment_004/`。固定绝对路径依赖和共享标准库/原生库明确保留。
- 便携参数入口：`reproduction/standalone_portable_release_001/prepare_portable_navigation_001.py --help`，传入release、源码根、模型、资产、图形运行时和新destination。不得把主表统计软链接暴露给导航推理；本次在导航结束后才添加统计只读输入。
- 全量测试：`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m pytest -q tests`；452通过。业务检查：`python -m ruff check conceptgraph/completion conceptgraph/navigation tests`、`git diff --check`。
- GPU故障原180、语义缺一450和主450已有冻结release/原始记录，重跑入口沿用下列协议；不要挑选成功种子。
- 最新门户生成器：`source_bundle/thesis_portal_completed_017.py`；故障查看器`gpu_fault_evidence_viewer_004.py`；结果表`fault_performance_office_002.py`。这些程序有不覆盖已有目标的保护。

以下为归档版本的完整命令资料，保留参数来源；其中“进行中”不代表当前状态。历史原文件位于`source_bundle/status_aliases_before_024/reproducible_commands.md`。

---

# 实验复现与交付入口（快照020）

当前Goal未完成，正文暂缓。以下示例使用已经核验的已有Python环境；不宣称新系统或新虚拟环境已复现。所有新增输出放在本Goal目录，已有失败和结果不得覆盖。完整历史训练、录制建图、KITTI和全量实验命令已原样保留在 `source_bundle/status_aliases_before_020/reproducible_commands.md`。

## 阅读已完成证据

- 当前进度及分项：`实验完成度与交付清单_020.md/json`、`validation_report.json`。
- 15张开发实验表：`experiment_tables/development_evidence_003/README.md`，同目录含Word、Excel、PDF、逐表CSV与来源。
- 主450全部尝试：`statistics/realtime_three_model_full_001/trials.csv`；449完整、1缺失，失败在分母。
- 主450查看器：8768服务下的 `/visualizations/realtime_main450_preview_003/index.html`。
- KITTI：当前清单的“KITTI连续车载外部诊断”条目；原始证据 `KITTI连续车载_实验与复现交接_001.md`。
- 屏障/迟到排序：`真实屏障与迟到排序_CPU交接_001.md`及 `/visualizations/perceived_blockage_cpu_002/index.html`。
- 已有便携小任务：`reproduction/standalone_portable_release_001/README.md`。

## 在同一机器准备一个全新的源码完整任务

这个入口从390源码发布包重建，不复制历史帧、检测或轨迹；三个前端权重和第三方资产从合法本地路径只读使用。先确认余量，原GPU正式队列活跃时不要另启GPU模型。不要复用示例输出目录；如果已存在，换新的独立版本号。

```bash
TASK_PROJECT=/root/autodl-tmp/lizichong
TASK_OUT="$TASK_PROJECT/outputs/project_v2/thesis_completion_v1"
TASK_RELEASE="$TASK_OUT/reproduction/standalone_portable_release_001"
TASK_PY=/root/miniconda3/envs/conceptgraph/bin/python
mkdir "$TASK_OUT/reproduction/portable_user_source_001"
tar -xzf "$TASK_RELEASE/source_current.tar.gz" -C "$TASK_OUT/reproduction/portable_user_source_001"
"$TASK_PY" "$TASK_RELEASE/prepare_portable_navigation_001.py"   --release "$TASK_RELEASE/release.json"   --source-root "$TASK_OUT/reproduction/portable_user_source_001"   --model-dir "$TASK_PROJECT/checkpoint"   --asset /root/autodl-fs/third_party/urban_scene_assets/ToyCar_002.glb   --runtime-root "$TASK_OUT/source_bundle/software_render_runtime_001"   --python "$TASK_PY"   --extra-library-path /root/autodl-fs/third_party/graphics_runtime/lib   --extra-library-path /root/miniconda3/envs/conceptgraph/lib   --destination "$TASK_OUT/reproduction/portable_user_run_001"   --execute
```

若只要预检，移除 `--execute` 并使用独立新目录；预检不算实际运行。检查 `launcher_exit.json` 与实际case的 `result.json`、`import_audit.json`、`execution.jsonl`、`perception.jsonl` 和 `thread_observations.jsonl`。Python退出0不等于正确到达。路线、对象、锁定和实际到达分开；安全和保持须独立复算。已完成参考在 `reproduction/portable_executed_001/independent_audit_001/report.json`，停车保持扩展为后验开发诊断，不回填主450。

## 从全部冻结450记录重算主统计

无需模型推理，但必须保留原450 release、全部尝试账目和449原case；摘要不能代替原始记录。输出run-id须尚未使用。

```bash
cd /root/autodl-tmp/lizichong
LD_LIBRARY_PATH=/root/miniconda3/envs/conceptgraph/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.realtime_matrix_statistics   --input-run realtime_three_model_full_001 --run-id main_statistics_reproduce_001
```

结果为配置聚类开发统计；不能把种子、帧或模型调用当独立城市样本。尚在运行的缺一矩阵不能用此命令生成终态统计。

## 原活跃队列与人工入口

原队列自动按450语义消融→6例GPU故障smoke→180故障→独占性能执行。不要因观察超时、旧快照或等待文件而重启；先核对PID及/proc启动时刻。进度在 `vehicle_execution/semantic_minus_execution_full_001/progress.json`，原任务账目在 `interactive_simulation/realtime_semantic_minus_full_001/trial_records.jsonl`；最终统计/故障/性能只有实际完整结束才计完成。

服务8768的人工核验为 `/annotation/natural_command_review_001/index.html`，独立自然表达采集为 `/annotation/human_natural_collection_001/index.html`，开发当前对象图审核为 `/visualizations/causal_development_review_002/index.html`。远程使用端口映射；网页具备提交功能不等于已收到真实人工记录。交互导航服务8769见 `交互仿真_操作与复现_001.md`。

## 工程核验

普通测试不加载真实大模型。最近原机452通过，纯代码导出450通过、2个真实数据跳过；最终队列结束后再进行所需终态检查。

```bash
cd /root/autodl-tmp/lizichong
LD_LIBRARY_PATH=/root/miniconda3/envs/conceptgraph/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/conceptgraph/bin/python -m pytest -q tests
git diff --check
```

新的Python环境仍待实际准备/执行；472包版本清单不是自动安装成功证明。当前完整复制site-packages约11GiB，会挤占剩余正式实验冻结余量，因此不在原队列运行时做全量复制。模型、资产与系统图形依赖的SHA和外部路径见便携release；不自动下载或写回历史模型。
