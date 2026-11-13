# 小规模完整导航复现入口

本目录不依赖原项目的历史实验输出。`source_current.tar.gz`是检查点033的390个业务源码/配置；`release.json`记录逐文件SHA与权限、固定任务config00-positive0/seed17、三个真实前端权重和外部资产SHA。代码改动由额外的停车保持扩展明确提供，不改冻结主实验。未包含模型权重、第三方车辆资产或Python/OSMesa环境；这些依赖必须自行准备到合法本地位置，并逐项校验。

001原版真实执行未到达，且Ultralytics将入口线程设置重设为8；原始失败保留。002是查看该失败后设计的资源修正＋停车保持组合诊断，不能作为未调参新测试或单组件因果收益，也不能回填主450成绩。复用既往成功规划的感知接近点，在同一感知节点附近低速停车后持续制动，不读取独立目标真值。

复现步骤：

1. 用`tar -xzf source_current.tar.gz -C /新建的源码目录`解压源码；不得将旧实验对象图、轨迹或检测缓存复制进去。
2. 根据`standalone_environment_distributions_001.txt`和项目环境说明准备Python3.10环境。该清单记录实际安装发行包版本，不是自动保证跨系统安装成功的conda镜像。另准备`software_render_runtime_001`同结构的PyOpenGL与OSMesa依赖，并提供所需动态库目录。
3. 在本目录运行下列命令，将占位路径替换为真实依赖位置。先省略`--execute`只校验和准备；需实际执行时选一个尚不存在的新输出目录并加入`--execute`。准备器拒绝覆盖已有目录。实际运行逐次重新渲染并调用YOLO/MobileSAM/CLIP，不使用旧帧或旧检测。

```bash
/环境/bin/python prepare_portable_navigation_001.py \
  --release release.json \
  --source-root /新建的源码目录 \
  --model-dir /合法模型目录 \
  --asset /合法资产目录/ToyCar_002.glb \
  --runtime-root /隔离图形依赖/software_render_runtime_001 \
  --python /环境/bin/python \
  --extra-library-path /补充动态库目录 \
  --destination /新建的输出目录 \
  --execute
```

模型目录需有`yolov8l-world.pt`、`mobile_sam.pt`与release中指定层级的CLIP safetensors。程序不会自动下载、安装或将权重写回原目录。输出中的checkpoint为显式链接，权重逐项检查SHA。原始资产复制至新输出目录，内容必须匹配release；代理只访问当前/过去实际感知和声明的道路先验。

运行结果位于新输出目录的`checkout/outputs/project_v2/thesis_completion_v1/interactive_simulation/live_navigation_sessions_001/`。分别检查`result.json`的路线、对象、正确锁定、实际到达四项；Python退出码0不等于到达。`import_audit.json`验证业务模块实际从新源码导入，`thread_observations.jsonl`验证每次前端CPU单线程且CUDA未初始化。900秒墙钟、30仿真秒、80前端、500MB以及最低6GB余量预算仍有效。

现有独立审核使用保存的RGB/深度/检测SHA、NetworkX身份关联、重新计算的自行车区间/矩形车身/方向，以及独立区域内连续至少2秒保持；原始运行和审核路径见上级交付报告。此处复现使用已有Python和图形依赖，不能称已在新操作系统或新虚拟环境完成。

本入口不增加独立场景、人工标签、新封存样本或GPU性能证据。整个Goal仍未完成；当前阶段不写论文正文。
