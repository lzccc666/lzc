# 三维语义场景图与车辆导航：代码及实验交接

当前 `goal_complete=false`，阶段9—12仍有必需条件未满足。论文框架和正式写作由用户后续GPT Work开展；本仓库提供当前项目代码与小型证据，不包含原始数据、权重、历史论文原稿或凭据。

- [工作交接](doc/thesis_completion/工作交接_代码与实验.md)、[写作证据边界](doc/thesis_completion/GPT_Work_写作交接.md)、[复现命令](doc/thesis_completion/reproducible_commands.md)、[验收快照](doc/thesis_completion/validation_snapshot.json)。
- 当前363份项目源码逐项字节及执行权限核对；原机 `pytest -q tests` 336项通过。纯代码环境有2项依赖未公开真实crop的测试明确skip；不能用假crop补齐。业务范围Ruff通过，2个未改动上游脚本的17条广域Ruff历史问题保持记录。
- 过去LiDAR开发主表为38场景11log1508帧，1470实测/38缺测、未来0，独立重投影0像素差。原融合召回17.23%、匹配精确率16.49%、误差1.696m；深度质量30.11%、43.68%、1.303m。5723是跨帧GT观测，身份变化率差区间跨0，官方GT晚约33–42ms的近似继续披露。
- [四方法检索](experiments/development_causal/retrieval_full_report.json)：1115查询、733正/382负，类别最近/CLIP/简化VLMaps式网格/联合选对176/124/103/43，负误选131/275/298/8。结构化关系F1低于直接几何，去结构化边仍保留几何回退；不声称完整VLMaps复现或所有模块有收益。
- [684次录制扰动](experiments/development_causal/faults_readable_report.md)及27144帧记录独立重算结束。LiDAR丢失80%召回差−15.84pp，位姿扰动中心误差+0.1887m；只是离线合成抗扰动，不是闭环安全或车辆到达。
- [动态/TTL](experiments/development_causal/dynamic_modules_full_report.json)和[关联门限](experiments/development_causal/mapping_parameters_full_report.json)各114条件，分别76新CPU重建+38同版seed17复用，共享基线不增加独立样本。减动态维护召回−5.21pp，减TTL精确率−3.89pp。门限0.25精确率收益区间跨0，未重现旧未来输入下的明确收益；0.45精确率与身份变化明显退化。两批全部关联/身份/区间[独立重算](experiments/development_causal/modules_parameters_independent_audit.json)通过。
- [相机时间中心敏感性](experiments/development_causal/camera_time_readable_report.md)：5723固定分母，5311中心插值/412回退，七方法原预测不变。深度质量对原融合召回差+12.98pp、11log CI[8.20,18.20]pp，定位优势仍在；深度质量自身因时间参考改变的召回/误差差区间跨0。插值是中心线性模型，尺寸/朝向/可见性仍有近似，不是精确相机时间GT，不替代原主表或冻结标签。
- [六地图完整结果](experiments/development_causal/maps_readable_report.md)：212查询1272条件，完整约束94路线、49同时车道与道路合法、2/94静态矩形通过，未执行车辆。三处逐姿态浮点边界差异和首次严格比较失败保留，完整路线车身通过/失败一致，没有改判原违规。
- KITTI实际64帧33原始drive真实CPU前端，3m中心召回247/319→262/319；卡车误差2.768→2.888m的负结果保留。这不是官方AP、跨帧图或KITTI导航到达，详见 `experiments/kitti_object_uniform64/` 及交接文档中的实际路径。
- [版本化人工核验](doc/thesis_completion/development_human_review_versioning_001.md)绑定过去LiDAR证据/源预测/crop、所选方法和节点。服务器8768正式入口 `/visualizations/causal_development_review_002/index.html`，门户008分开当前past与历史future。真实人工仍0，浏览器只核对读取及4个必定拒绝的错误SHA请求；没有伪造有效人工记录。

[真实时钟传感器故障](experiments/realtime_fault_cpu/readable_report.md)六条件CPU试跑：真实前端63调用、2270控制状态，单任务/单种子均到达，不增加独立覆盖。4秒断流实际发生，但CPU15秒TTL下未触发过期停车，负结果保留；正式GPU1.5秒TTL的6试跑和180全量仅排队。[已知路径控制功能](experiments/realtime_fault_cpu/control_functional.json)另用0.3秒TTL、63状态、0模型调用验证停车/恢复与阻塞锁存，不代替完整感知故障验收。首次gobs元数据不兼容失败保留，修复后独立重跑，未覆盖原run。

450次真实时钟三模型主矩阵仍运行，随后450次语义缺一按冻结队列执行；六地图1272条件已结束，目标接近点3345条件也已完整结束，联合与首达1115组完整推理完全相同，未观察联合收益。真实人工、自然指令有效覆盖、新封存、独占性能和完整最终验收均仍缺。分别报告生成路线、选对对象、正确锁定、实际到达，不能混用。

公开CSV仅CRLF转LF且逐字段一致，原机实验文件不动。复现脚本是本机有界队列或完整结果审核器，须另备许可数据、模型与同版协议；已存在的run-id拒绝覆盖。Git快照与统计快照不自动随服务器更新，后续以新发布回执和完整结果为准。
