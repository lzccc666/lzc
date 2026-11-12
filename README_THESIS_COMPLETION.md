# 三维语义场景图与车辆目标接近导航：代码及实验交接

这是 `/root/autodl-tmp/lizichong` 的当前代码快照，保留上游ConceptGraphs许可与归属。用户后续使用GPT Work拟定论文框架和撰写；**课题仍有未完成验收，goal_complete=false**。

从[执行与代码交接](doc/thesis_completion/工作交接_代码与实验.md)和[写作证据交接](doc/thesis_completion/GPT_Work_写作交接.md)开始阅读。[验收快照](doc/thesis_completion/validation_snapshot.json)说明仍在运行的450次真实时钟主矩阵、后续450次语义缺一、新封存测试与人工审核依赖。快照不自动随服务器实验更新。

KITTI已实际完成固定64帧、33原始drive的YOLO/MobileSAM/CLIP与LiDAR定位诊断。[完整结果](experiments/kitti_object_uniform64/report.md)与[原始统计JSON](experiments/kitti_object_uniform64/report.json)保留改进、卡车负结果和未知深度。这是自定义三维中心诊断，**不是官方KITTI AP或完整导航到达**。

- 业务代码：`conceptgraph/completion/`、`navigation/`、`scenegraph/`、`slam/`、`vlm/`。
- 验证：当前全量294测试通过，Ruff和diff检查通过，见`reproduction/verification/`。
- 纯代码导出环境没有原始阶段4数据，2个真实crop集成测试会明确skip；其余292项正常运行。服务器有数据时这2项必须通过，不用合成图像冒充真实crop。
- 复现命令：[reproducible_commands.md](doc/thesis_completion/reproducible_commands.md)。KITTI入口为`python -m conceptgraph.completion.kitti_transfer`；先prepare、再CPU infer，独立运行`kitti_evaluate`和`kitti_preview`。
- 环境：Python3.10，实际安装版本见`reproduction/runtime_environment.json`。部分pip依赖在原机为本地构建，`requirements-runtime-lock.txt`是可携带的部分锁，不代表异机环境已完整验收。
- 轻量仿真准备：`python reproduction/run_rule_demo.py --prepare-only`；可复现CPU规划上限`python -m conceptgraph.completion.kinematic_benchmark --run-id local_oracle_001 --task config00-positive0 --seed 17 --methods hybrid_astar`。
- 完整感知规则演示需用户另备本地YOLO/MobileSAM/CLIP权重、CUDA和EGL：`PYOPENGL_PLATFORM=egl python reproduction/run_rule_demo.py --execute --run-id local_rule_001`。仅在GPU没有其他实验时运行，实际到达以保存结果为准，不保证每次成功。

原始数据、大模型/Adapter、原论文、历史记忆、本机字体与凭据未发布。`experiments/`仅含小型统计与已许可的派生数值；原始KITTI图像、点云与标签仍按CC BY-NC-SA 3.0单独获取。`reproduction/assets/`的ToyCar是CC0，来源与归属随文件保留。

服务器本轮结果目录为`outputs/project_v2/thesis_completion_v1/`，其中大量原始证据没有上传；文档中此目录下的路径指向服务器文件，不能假设公共代码库包含完整训练和实验数据。原项目checkout、分支、索引与历史结果保留，本快照没有发布未经审计的原Git历史。

过去LiDAR开发全量已结束：38场景11log1508帧（1470实测、38显式缺测），全部重投影0像素差，7659文件SHA核验。[建图结果](experiments/development_causal/mapping_full_report.md)保留5723个GT观测及130个缺测首帧失败。原融合召回17.23%、匹配精确率16.49%、中心误差1.696m；深度质量30.11%、43.68%、1.303m。身份变化率差区间跨0，不称改善。官方sample GT晚于相机33–42ms的评价近似仍在。

同批1115查询（733正/382负）的[四方法检索](experiments/development_causal/retrieval_full_report.json)已结束：类别最近/CLIP/简化VLMaps式网格/联合选对176/124/103/43、负误选131/275/298/8。结构化关系F1 0.18780低于直接几何0.19030，未知端点保留。不是完整VLMaps复现，也不提供正确锁定或车辆到达信用。

[9参数敏感性](experiments/development_causal/selection_parameters_full_report.json)实际10035记录，[关系边/证据消融](experiments/development_causal/selection_ablation_full_report.json)3345记录。确认5次选对34/733；拒绝分0.7选对和误选均0。去结构化关系边后52/733、对基线差异区间[0,2.57]pp，仍保留几何回退，不能称关闭全部关系或证明结构化边收益。[道路分组](experiments/development_causal/road_groups_full_report.json)的弯道仅5正查询2log，不宣称泛化。

本次导出357份当前项目源码，字节和执行权限逐项核验。原机294测试通过；导出环境另有2个真实数据依赖skip。业务范围Ruff通过；2个未改动上游脚本广域Ruff仍17条历史问题，完整性审计明确记录。公开CSV仅CRLF改LF，逐字段相同，原机实验CSV不动。

`reproduction/causal_ablation_sequence.py`与`causal_sensitivity_sequence.py`为本机冻结顺序队列，须先准备相同数据、模型和输出协议，不能当无数据的一键实验。新past录制扰动及其余动态/地图/关联消融仍推进；真实时钟450主矩阵和后续450缺一仍未结束，人工标签和新封存等仍未验收。门户004在服务器8768端口提供当前与历史范围分开的只读证据入口，快照不会自动更新。
