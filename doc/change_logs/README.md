# 分阶段变更记录

本目录保存 `doc/project_optimization_plan.md` 各实施阶段完成后的工程变更记录。总计划描述预期工作，本目录记录实际完成内容、验证证据和数据限制。

## 记录规则

- 每个阶段使用一份独立Markdown文件，文件名以 `stageN_` 开头。
- 阶段验收、代码审查和真实输出完成后再将状态标记为完成。
- 记录实际修改，不复制尚未实施的计划内容。
- 命令、输出路径、测试数量和数据差异必须使用真实运行结果。
- 后续修复若改变该阶段行为，应同步更新原记录并增加修订说明。
- 不在变更记录中写论文内容、密钥、凭据或大段运行日志。

## 当前记录

| 阶段 | 状态 | 记录文件 |
|---|---|---|
| 阶段1：工程基础与数据工具 | 已完成 | `stage1_engineering_foundation_and_data_tools.md` |
| 阶段2：目标选择与路径规划 | 已完成 | `stage2_target_selection_and_path_planning.md` |
| 阶段3：场景图关系 | 已完成 | `stage3_scenegraph_relations.md` |
| 阶段4：Qwen语义模块与微调v2 | 已完成 | `stage4_qwen_semantic_module.md`、`stage4_qwen_finetune_v2.md` |
| 阶段5：增量导航 | 已完成 | `stage5_incremental_navigation.md` |
| 阶段6：集成与回归 | 已完成 | `stage6_integration_and_regression.md` |
| 阶段7：Qwen在线闭环 | 代码完成、GPU实跑待执行 | `stage7_qwen_online_closed_loop.md` |

后续Goal的停止条件必须包含：更新本目录的对应记录，并核对记录中的文件、测试和真实输出与最终工作树一致。
