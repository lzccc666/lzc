# M12 KITTI跨数据集诊断

## 解决的问题与输入

KITTI Object真实相机/LiDAR及Tracking连续帧、标定/位姿，官方标签仅进入隔离评价。

## 实际工作流程

Object分prepare/infer/evaluate；Tracking转换连续相机与位姿，运行真实CPU前端和无GT建图，再做中心/身份诊断。预览默认隐藏GT。

## 输出及下游接口

静态64帧/33 drive及固定3序列/90帧结果、匹配/误差/身份指标和可核验图像证据。

## 代码位置

| 文件 | 主要入口（固定版本） |
|---|---|
| [本地 completion/kitti_transfer.py](../../code/conceptgraph/completion/kitti_transfer.py) | [source_files:27](https://github.com/lzccc666/lzc/blob/b8d3d18a6c7b53c5322ea5308c37bcb283fa8f1e/conceptgraph/completion/kitti_transfer.py#L27)、[source_hashes:35](https://github.com/lzccc666/lzc/blob/b8d3d18a6c7b53c5322ea5308c37bcb283fa8f1e/conceptgraph/completion/kitti_transfer.py#L35)、[budget_guard:39](https://github.com/lzccc666/lzc/blob/b8d3d18a6c7b53c5322ea5308c37bcb283fa8f1e/conceptgraph/completion/kitti_transfer.py#L39)、[prepare:49](https://github.com/lzccc666/lzc/blob/b8d3d18a6c7b53c5322ea5308c37bcb283fa8f1e/conceptgraph/completion/kitti_transfer.py#L49)、[observations:137](https://github.com/lzccc666/lzc/blob/b8d3d18a6c7b53c5322ea5308c37bcb283fa8f1e/conceptgraph/completion/kitti_transfer.py#L137) |
| [本地 completion/kitti_evaluate.py](../../code/conceptgraph/completion/kitti_evaluate.py) | [evaluate_frame:19](https://github.com/lzccc666/lzc/blob/b8d3d18a6c7b53c5322ea5308c37bcb283fa8f1e/conceptgraph/completion/kitti_evaluate.py#L19)、[aggregate:56](https://github.com/lzccc666/lzc/blob/b8d3d18a6c7b53c5322ea5308c37bcb283fa8f1e/conceptgraph/completion/kitti_evaluate.py#L56)、[common_supported_errors:72](https://github.com/lzccc666/lzc/blob/b8d3d18a6c7b53c5322ea5308c37bcb283fa8f1e/conceptgraph/completion/kitti_evaluate.py#L72)、[main:86](https://github.com/lzccc666/lzc/blob/b8d3d18a6c7b53c5322ea5308c37bcb283fa8f1e/conceptgraph/completion/kitti_evaluate.py#L86) |
| [本地 completion/kitti_tracking_experiment.py](../../code/conceptgraph/completion/kitti_tracking_experiment.py) | [source_hashes:31](https://github.com/lzccc666/lzc/blob/b8d3d18a6c7b53c5322ea5308c37bcb283fa8f1e/conceptgraph/completion/kitti_tracking_experiment.py#L31)、[verify_hashes:43](https://github.com/lzccc666/lzc/blob/b8d3d18a6c7b53c5322ea5308c37bcb283fa8f1e/conceptgraph/completion/kitti_tracking_experiment.py#L43)、[guard:49](https://github.com/lzccc666/lzc/blob/b8d3d18a6c7b53c5322ea5308c37bcb283fa8f1e/conceptgraph/completion/kitti_tracking_experiment.py#L49)、[load_agent:58](https://github.com/lzccc666/lzc/blob/b8d3d18a6c7b53c5322ea5308c37bcb283fa8f1e/conceptgraph/completion/kitti_tracking_experiment.py#L58)、[freeze:71](https://github.com/lzccc666/lzc/blob/b8d3d18a6c7b53c5322ea5308c37bcb283fa8f1e/conceptgraph/completion/kitti_tracking_experiment.py#L71) |
| [本地 completion/kitti_tracking_preview.py](../../code/conceptgraph/completion/kitti_tracking_preview.py) | [main:46](https://github.com/lzccc666/lzc/blob/b8d3d18a6c7b53c5322ea5308c37bcb283fa8f1e/conceptgraph/completion/kitti_tracking_preview.py#L46) |

## 实验证据与可写边界

[仓库证据](https://github.com/lzccc666/lzc/tree/b8d3d18a6c7b53c5322ea5308c37bcb283fa8f1e/doc/thesis_completion/KITTI_tracking_handoff.md)。证据目录中的开发对比、消融和限制需一起阅读。

不是官方3D AP或KITTI导航到达。74次受支持Tracking GT全部集中在0005的4实例，另外465次不支持标签保留；不能当3个独立正序列。预训练重叠未被排除。

## 后续写作材料归属

第6章外部数据诊断，明确支持类别、指标门限、序列覆盖与负结果。

机制解释是设计动机；只有对应独立实验支持时，才能写成已证实的性能改进。上游基础模型、框架和本项目改动须分别归属。
