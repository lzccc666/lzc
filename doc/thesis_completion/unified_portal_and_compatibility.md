# 统一实验入口与消融兼容交接

实验交付更新（2026-11-12T20:47:08.882508+00:00）：暂缓论文正文、章节和框架，先完善实验与复现。统一入口已升级为`visualizations/thesis_portal_013/index.html`，直接展示主450四项分离结果，并链接完整450日志回放、KITTI64帧/33drive、532身份条件及228真实地图控制诊断。桌面/手机检查、65静态证据链接、缺失/损坏响应显示未知及刷新恢复均通过；数字分数和方法名称在手机不拆行。旧入口、009错路径失败、010脚本错误、011原样式、012及QA016不恰当行高判据全部保留，不覆盖历史。

发现并处理后续汇总阻碍：主release66文件，缺一release是共同66份SHA全同+新增地图引导模块。任务、权重、选模、协议相同，新增模块由目录全收录引入，未在共同冻结源码显式引用，隐藏CUDA的实际入口导入探针也未加载。原字典全等断言将失败，原程序/原release不改。新003等待器PID 288085已核对起始ticks，只有原450子进程及父进程实际退出、原断言失败日志保留、450完整账目/指纹核对、750行组合统计和独立复算全部通过后，才新增明确标识兼容版本的完成桥接。当前桥接不存在，不能称消融完成。独立复算入口已用主450真实记录验证66项统计、2320源/产物SHA，不运行模型或车辆。002误将模型.eval()当内置eval的预检失败已保留，003修复只涉及审核器。

缺一新增450本次快照49/450；300主记录只复用，不称750新增。正式GPU故障和独占性能原队列不变。项目424通过、纯代码422通过2真实数据skip，业务Ruff和差异检查通过；381源码已准备，最后公开365，最终实验核验后统一同步GitHub。人工与合格新测试仍以API快照为准，封存未开启，Goal仍false。旧v14包不含主450、查看器及本轮增量。

统一演示入口：`http://localhost:8768/visualizations/thesis_portal_013/index.html`，需转发远程8768端口。原主450专门查看器仍为`/visualizations/realtime_main450_preview_003/index.html`。门户复现使用唯一run-id：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.thesis_portal --run-id 新唯一门户目录 --region-run reference_regions_reserved_full_003 --region-preview reference_regions_reserved_001 --reference-run reference_reachability_reserved_full_002 --reference-preview reference_reachability_reserved_002
```

源集合兼容等待器已运行，`ablations/semantic_navigation_chain_003/worker_started.json`记录PID/ticks，不要在同目录重复启动。003脚本--prepare-only/--resume是两阶段入口，固定run目录，只适合原实例或隔离副本重现；新运行必须使用新版本/路径，先检查进程和全部指纹。冻结原66与额外文件均不可改。旧`semantic_ablation_after_main_001.py`将保留原字典全等断言失败；只有真正450完成、750组合及独立审核通过，才写原等待器要求的兼容桥接`ablations/semantic_navigation_chain_001/complete.json`。桥接显式标记original_script_success=false，不冒称原脚本成功。完成前桥接必须不存在。

独立复算入口`source_bundle/semantic_compatibility_evidence_check_003.py`有--input-run、--run-id、--expected、--methods参数，先用实际主450完成回归，再对实际750运行；不能用部分消融记录替代全量。无新增模型/执行/到达。所有后续GPU队列仍按原冻结设置串行。
