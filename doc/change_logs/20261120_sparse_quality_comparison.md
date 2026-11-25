# 稀疏观测配置对照变更

## 问题与结果

原入口丢弃不足5个实测深度点的检测，稀疏位置与三维形状绑定。新增显式实验模块A，将质量加权位置与点云形状分开，保持原充分点云的主关联。在相同38验证/调试场景1508帧中，召回39.49%→46.88%，F1 43.31%→47.44%，IDF1 39.10%→42.37%。召回增量通过预登记三次比较校正门槛，停止B2/B3实施。默认未改变。

汽车与近距离召回回退，重复节点增加，详见[完整报告](../../outputs/reports/recall_config_goal_20261120_v1/本轮配置对照与早停结论.md)。本轮只证明模块A整体收益，未分离内部机制贡献。

## 代码

- `conceptgraph/slam/streamlined_mapping.py`：提取默认观测与默认分配钩子，默认逻辑保持原入口。
- `conceptgraph/slam/utils.py`：增加可选过滤诊断、观测策略、稀疏回调；默认参数不改变旧行为。
- `conceptgraph/slam/audited_mapper.py`：真实入口诊断及确认快照。
- `conceptgraph/slam/sparse_observations.py`：深度簇质量评分、真实点反投影、协方差位置层、独立证据确认和身份交接。
- `conceptgraph/slam/quality_mapper.py`：模块A显式实验入口，B2/B3仅登记、未实施。
- `conceptgraph/completion/config_comparison.py`：原评价实现的计数汇总与日志级配对统计。
- `scripts/run_mapping_quality_comparison.py`：串行完整重建、输入与源码冻结、原范围评价。
- `scripts/summarize_quality_configuration.py`：固定统计、分组、自动案例与数值停止门槛。
- `scripts/verify_mapping_quality_evidence.py`：同配置核验，逐帧预测确定性及内部关联/确认记录。
- `scripts/finalize_quality_comparison.py`：交付审计、关键保护文件和模型指纹、预算、测试与静态检查。
- `tests/test_sparse_quality_mapping.py`：实际入口与位置层、身份、统计的边界测试。

## 运行与验证

环境使用`/root/miniconda3/envs/conceptgraph/bin/python`，固定`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`。入口运行时传入独立run_id和`--condition B0`或`--condition B1`；冻结命令、配置、输入清单与源码见对应run的`release.json`和`source_snapshot`。

64项相关测试、目标文件Ruff和Git空白检查通过。B0逐帧四项计数及汇总身份指标与原始基线一致。B0/B1各38场景1508帧同配置重放的预测与身份逐字段一致；关联、确认与失效记录已保存。总墙钟约34分钟，在2小时预算内。详见最终核验报告，不声称重做全仓库测试或全部历史文件指纹。

保护历史数据、模型、失败记录、论文与MEMORY类文件；未访问新测试选方法，未训练/下载或开启CARLA，未提交/推送Git。
