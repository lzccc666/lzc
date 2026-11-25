# RAM与Grounding DINO对照

本轮回答：把当前YOLO检测前端换成Grounding DINO，或者先用RAM自动列类别再交给Grounding DINO，是否更适合当前车载对象地图。只做这一项对照，不恢复已停止的旧Goal。

**本轮已完成38场景、1508帧完整三维主对照。当前YOLO综合F1为43.31%，固定类别DINO为38.84%，RAM＋DINO为38.87%；暂不替换现有默认前端。** 人话结论与全部表格见 [实验结论与交付](../../outputs/reports/ram_dino_comparison_20261118/实验结论与交付.md)。

**清理后状态（用户明确要求）：继续使用改进YOLO；本轮新增RAM权重、下载残片和逐帧检测/分割/特征文件已删除，实际释放6.27 GiB。** 下文目录格式说明实验执行时的产物；其中`detections/*.pkl.gz`现已清理。全部汇总、逐帧地图/匹配`frames.jsonl.gz`、代码快照、配置、日志和失败记录保留。原有DINO/YOLO/SAM/CLIP/Qwen资产未清理。逐项清单、原哈希和完成记录见`outputs/reports/ram_dino_comparison_20261118/cleanup_20261118/`。

## 三种方案分别是什么

| 方案 | 用人话解释 |
|---|---|
| 当前改进YOLO | 已知要找六类交通目标，直接检测它们 |
| 固定六类Grounding DINO | 告诉另一个检测器找同样六类目标，用来单独比较检测器 |
| RAM＋Grounding DINO | RAM先看图列出它认为有哪些类别，再让DINO寻找这些类别；不偷偷补入缺失的交通类别 |

后两组接入相同的MobileSAM分割、CLIP特征、距离输入和连续对象建图。原始ConceptGraphs论文还包含其他模型和任务，因此不能把本轮称为原论文完整复现。

## 完成范围和查看入口

- 完整二维诊断：38个验证场景中的154个固定抽样帧，587次标准目标出现。结果在 `outputs/results/nuscenes/validation/iterations/ram_dino_probe_20261118_v2/`。
- 完整三维主对照：`outputs/results/nuscenes/validation/iterations/ram_dino_full_20261118_v3/`，目标范围为同38场景1508帧。只有根目录report.json生成后才表示全部完成。
- 合并当前YOLO的比较：`outputs/results/nuscenes/validation/iterations/ram_dino_comparison_20261118_v1/`。
- 独立对象覆盖：`outputs/results/nuscenes/validation/iterations/ram_dino_object_coverage_20261118_v1/`。
- 日志、类别遗漏诊断、运行环境与交付核验：`outputs/reports/ram_dino_comparison_20261118/`。

主对照使用事前列出的DINO推荐框门限0.25。0.1门限的完整二维结果保留；由于候选显著增多、增加分割和特征成本，其三维补充实验只保留已跑前缀，不冒充完整三维比较。v3只省去不会进入推荐配置地图的候选的SAM/CLIP计算，原始0.1框、模型分类分数和RAM提示仍保存。没有按成绩挑场景，没有放宽三米评分规则。

## 每帧存什么

```text
ram_dino_full_20261118_v3/
  release.json                             模型哈希、输入哈希、配置与预算
  source_snapshot/                         实际执行代码快照
  scene-xxxx/
    detections/
      000000_dino_fixed.pkl.gz             固定类别检测、分割、特征和原始分数
      000000_ram_dino.pkl.gz               RAM提示、检测、分割、特征和原始分数
    frames.jsonl.gz                        两种地图每帧节点、隔离评分和耗时
    report.json                            场景结果
  report.json                              完整汇总
```

检测文件中的 `frame_provenance.diagnostics` 保留RAM原始标签、实际提示类别、原始框和模型对各类别的分数。混合短语按模型的完整类别分数选择，不随便归第一类；car door、bus stop不当作车辆，van/SUV按nuScenes官方定义归到car。

## 模型和代码在哪里

| 内容 | 位置 |
|---|---|
| RAM来源与哈希、分词器配置（权重已按用户要求清理） | `checkpoint/ram_dino_comparison_v1/` |
| 原有DINO权重 | `checkpoint/groundingdino_swint_ogc.pth` |
| RAM/DINO适配、类别解码、接入同一SAM/CLIP | `conceptgraph/completion/ram_dino_frontend.py` |
| 固定抽样二维诊断 | `scripts/probe_ram_dino.py` |
| 完整真实检测分割和连续建图 | `scripts/run_ram_dino_mapping.py` |
| 同场景比较、采集日志统计区间和分组 | `scripts/summarize_ram_dino_comparison.py` |
| 类别解码和建图接口检查 | `tests/test_ram_dino_frontend.py` |

RAM原模型约5.6GB，清理前SHA-256已与作者发布文件核对。本轮没有微调RAM、DINO、YOLO或SAM。DINO检查点包含完整BERT权重，初始化后确认无missing_keys，不用另下载的BERT权重参与推理。接入失败记录保留，下载中断的part文件已按用户要求清理。

## 复现命令

以下仅用于确需重跑时执行，run-id必须换成未使用过的新名称。正常查结果请直接打开已有report，不反复运行。

清理后重跑前需按`checkpoint/ram_dino_comparison_v1/assets.json`恢复RAM权重并核对哈希；已删除的原始检测文件不能直接读取。原始交付核验描述的是清理前状态，以清理记录为当前存储状态依据。

```bash
cd /root/autodl-tmp/lizichong
export LD_PRELOAD=/root/miniconda3/envs/conceptgraph/lib/libstdc++.so.6
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

/root/miniconda3/envs/conceptgraph/bin/python scripts/probe_ram_dino.py --run-id <新的二维对照ID>
/root/miniconda3/envs/conceptgraph/bin/python scripts/run_ram_dino_mapping.py --native-only --wall-budget-s 2400 --run-id <新的完整建图ID>
/root/miniconda3/envs/conceptgraph/bin/python scripts/summarize_ram_dino_comparison.py --run-id <新的汇总ID>
```

C++运行库设置仅影响当前进程，避免已有SciPy与系统运行库加载顺序冲突；没有升级或卸载项目环境。DINO采用官方800短边／1333最长边预处理，当前YOLO采用960输入，比较报告同时披露耗时；不把它们说成相同像素计算量。
