# 阶段1变更记录：工程基础与数据工具

- 完成日期：2026-08-30
- 状态：已完成并通过验收
- 项目根目录：`/root/autodl-tmp/lizichong`
- 基线提交：`eab10f5`（`v1.0-offline-baseline`）
- 运行环境：Conda `conceptgraph`，Python 3.10.20

## 1. 实施范围

本阶段建立了新增代码的测试骨架和 `outputs/project_v2` 输出边界，并将停车场场景筛选从描述关键词计数升级为基于nuScenes结构化元数据、实例属性和本地传感器文件完整性的确定性评分工具。

本阶段未修改目标检测、建图、Qwen训练或推理、阶段2至阶段6业务逻辑，也未解压完整nuScenes数据。

## 2. 文件变更

新增：

```text
conceptgraph/nuscenes/__init__.py
conceptgraph/nuscenes/scene_selector.py
conceptgraph/scripts/build_nuscenes_subset_manifest.py
tests/README.md
tests/nuscenes/test_scene_selector.py
tests/navigation/.gitkeep
tests/scenegraph/.gitkeep
tests/vlm/.gitkeep
doc/change_logs/README.md
doc/change_logs/stage1_engineering_foundation_and_data_tools.md
```

修改：

```text
conceptgraph/scripts/select_nuscenes_parking_scenes.py
doc/project_optimization_plan.md
```

旧筛选脚本继续作为可用入口，但改为调用新的结构化评分模块；默认输出迁移到 `outputs/project_v2/subset`，不会覆盖 `outputs/experiment_summary` 基线。

## 3. 结构化指标

评分模块读取并关联以下9张表：

```text
scene.json
sample.json
sample_annotation.json
instance.json
attribute.json
category.json
calibrated_sensor.json
sensor.json
sample_data.json
```

每个场景记录：

- 场景名称、描述、声明样本数、元数据样本数、sample链长度、轨迹帧长度和持续时间；
- CAM_FRONT、LIDAR_TOP的元数据关键帧数、本地文件数和完整配对帧数；
- `vehicle.parked` 数量及其在 `vehicle.*` 标注中的比例；
- `cycle.without_rider`、barrier、traffic cone和bicycle rack数量；
- 静态目标标注数、唯一实例数、持续帧数和持续比例；
- 动态目标标注数、唯一实例数和动态标注比例；
- 数据完整率、缺失文件诊断、各子项得分、总分和选择理由；
- 本地源数据是否满足转换条件，以及已有最小场景的color/depth/pose帧数对照。

无效token、重复token、sample链异常、缺表、缺少关键帧记录、非法文件名和本地文件缺失均输出结构化诊断。

## 4. 评分与排序

总分上限100分：

| 子项 | 分值 | 规则 |
|---|---:|---|
| 停放车辆 | 20 | 65%为上限40条的parked标注数量，35%为parked/vehicle比例 |
| 静态目标多样性 | 15 | 60%为上限10个的唯一静态实例，40%为5类静态目标覆盖率 |
| 静态持续性 | 15 | 静态实例覆盖唯一sample数的平均比例 |
| 低动态程度 | 15 | `1 - dynamic_target_ratio`；无标注场景不得分 |
| 数据完整率 | 25 | 本地CAM_FRONT和LIDAR_TOP文件数除以期望文件数 |
| 序列长度 | 5 | 元数据样本数，上限40帧 |
| 描述上下文 | 5 | 停车上下文匹配，上限2次，仅作为辅助项 |

排序依次使用：本地源数据完整条件、总分、停放车辆比例、静态持续性、静态目标数、场景名。选择manifest只包含CAM_FRONT和LIDAR_TOP全帧齐备的场景。输出不写入时间戳等易变字段，相同输入可逐字节复现。

## 5. CLI与输出约束

CLI为 `conceptgraph/scripts/build_nuscenes_subset_manifest.py`，支持：

```text
--metadata-root
--version
--data-roots
--num-scenes
--out-dir
--dry-run
--overwrite
```

`--out-dir` 必须位于 `/root/autodl-tmp/lizichong/outputs/project_v2` 内。正式输出采用原子替换，默认拒绝已有文件，只有显式传入 `--overwrite` 才允许替换阶段1自己的结果。`--dry-run` 读取并评分真实元数据，但不创建或修改输出。

建立的输出目录：

```text
outputs/project_v2/tests/
outputs/project_v2/subset/
outputs/project_v2/offline_navigation/
outputs/project_v2/incremental_navigation/
outputs/project_v2/qwen_eval/
outputs/project_v2/performance/
```

## 6. 测试与代码审查

测试使用临时小型JSON fixture和伪传感器文件，不依赖完整nuScenes，覆盖评分、排序、缺失表、缺失传感器文件、无效token、属性映射、类别映射、输出越界拒绝和确定性。

最终结果：

```text
conda run -n conceptgraph pytest -q tests
8 passed
```

同时通过CLI `--help`、`python -m py_compile`、`black --check`、`flake8 --ignore=E501,W503` 和 `git diff --check`。

代码审查后修复了版本目录优先级、空标注场景的低动态理由、静态持续性重复sample计数和覆盖检查时机，并在修复后重新运行全部测试和真实数据生成。

## 7. 真实nuScenes结果

元数据目录：

```text
/root/autodl-tmp/lizichong/dataset/nuscenes_trainval_raw/v1.0-trainval
```

数据根目录：

```text
/root/autodl-tmp/lizichong/dataset/nuscenes_trainval_raw
/root/autodl-tmp/lizichong/dataset/nuscenes
```

解析了850个scene、34,149个sample、1,166,187个sample annotation、64,386个instance和2,631,083个sample data。元数据无效引用为0，缺失关键帧元数据为0。

850个场景中15个具备完整本地源数据。请求选择12个，实际结果：

| 排名 | 场景 | 总分 | 完整帧 | parked数量 | 动态比例 |
|---:|---|---:|---:|---:|---:|
| 1 | scene-0916 | 88.476 | 41/41 | 1647 | 0.208 |
| 2 | scene-0003 | 85.808 | 40/40 | 787 | 0.350 |
| 3 | scene-0168 | 85.526 | 40/40 | 1406 | 0.145 |
| 4 | scene-0655 | 84.852 | 41/41 | 1818 | 0.134 |
| 5 | scene-0133 | 83.763 | 40/40 | 1054 | 0.231 |
| 6 | scene-0128 | 83.524 | 40/40 | 832 | 0.190 |
| 7 | scene-0045 | 83.296 | 39/39 | 1108 | 0.243 |
| 8 | scene-0553 | 75.795 | 41/41 | 168 | 0.597 |
| 9 | scene-0757 | 74.113 | 41/41 | 177 | 0.541 |
| 10 | scene-0103 | 73.129 | 40/40 | 745 | 0.589 |
| 11 | scene-0061 | 71.895 | 39/39 | 30 | 0.435 |
| 12 | scene-1094 | 60.468 | 40/40 | 48 | 0.932 |

另外3个完整但未进入前12名的场景为 `scene-0796`、`scene-1077` 和 `scene-1100`。

## 8. 结果文件

```text
/root/autodl-tmp/lizichong/outputs/project_v2/subset/scene_scores.json
/root/autodl-tmp/lizichong/outputs/project_v2/subset/selected_scenes.json
/root/autodl-tmp/lizichong/outputs/project_v2/subset/integrity_report.json
```

最终SHA-256：

```text
c697df2ca95dacf4d03b93756231fa0f2e979529c1b90c1cc98ab8686e6fb19b  scene_scores.json
aded1666a4fa94148b3f7223de17e58a7f5ca698fce6b8826b228d9ff20b7681  selected_scenes.json
45aeb181e33658c7212255ec3f78d957fafcb24a6183ba19d21c732098ba2260  integrity_report.json
```

使用 `--overwrite` 重跑后，三个SHA-256保持不变。

## 9. 数据限制

- 非本地场景共有67,092个CAM_FRONT或LIDAR_TOP文件尚未提取；工具只报告缺失，没有解压原始压缩包。
- 现有11个最小场景目录中9个的color、depth和pose帧数与源数据一致。
- `scene-0553` 和 `scene-0757` 的源数据均为41个完整配对帧，但现有color、depth和poses均为40帧。差异已写入完整性报告，本阶段未重新转换或覆盖现有目录。

没有未解决的阶段1代码问题，上述内容属于已有数据覆盖限制。

## 10. 基线与Git状态

受保护范围包括现有experiment summary、Qwen数据和checkpoint，以及所有 `exp_qwen2vl_semantics_v1` 与 `objects_all_frames` 目录。实施前后的联合内容指纹均为：

```text
a0e8d743f387448c9bc8b9a622d5b9c41230e48e63c5163ed8ab52f50554ef12
```

阶段1未提交、未推送Git。用户原有 `MEMORY.md`、`PAPER_MEMORY.md` 和其他未提交内容未被删除或回滚。生成结果位于Git忽略的 `outputs/project_v2`。

## 11. 复现命令

```bash
conda run -n conceptgraph python conceptgraph/scripts/build_nuscenes_subset_manifest.py \
  --metadata-root /root/autodl-tmp/lizichong/dataset/nuscenes_trainval_raw/v1.0-trainval \
  --version v1.0-trainval \
  --data-roots \
    /root/autodl-tmp/lizichong/dataset/nuscenes_trainval_raw \
    /root/autodl-tmp/lizichong/dataset/nuscenes \
  --num-scenes 12 \
  --out-dir /root/autodl-tmp/lizichong/outputs/project_v2/subset \
  --dry-run
```

正式重建已有阶段1结果时，将 `--dry-run` 替换为 `--overwrite`。
