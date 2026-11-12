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


开发集新增过去LiDAR转换及独立重投影审计，参见`conceptgraph/completion/development_causal_*.py`和`doc/thesis_completion/development_causal_mapping_001.md`。相同RGB与检测SHA核验后复用原真实前端；缺测不补未来深度，新前端调用0。3帧真实smoke及独立投影/七条件建图/四检索已运行，首段对象0/2的负结果保留。38场景1508帧正式流水线和关系/检索继续有界运行；完成状态以服务器complete/report为准，公开快照不将排队视为完成。官方sample GT仍晚于相机33–42ms，评价近似需披露。

完整开发causal输入与独立审计已完成：1470实测/38缺测、未来0、全部1508帧重投影0差异。七条件建图及后续下游/消融仍执行中；新消融入口要求明确输入版本，复用基线严格检查相同帧SHA、seed17和配置。`reproduction/causal_ablation_sequence.py`是本机顺序队列源码，运行需要前置数据及输出协议，不是无数据的一键主实验。

过去LiDAR七条件开发重建已完整结束：[结果](experiments/development_causal/mapping_full_report.md)。原融合召回17.23%、精确率16.49%、中心误差1.696m；深度质量30.11%、43.68%、1.303m，11log配对区间支持定位改善，身份变化改善未确定。全5723次GT观测保留，包括38缺测首帧的130次失败；不是新封存或导航到达。
