# GPU故障证据查看入口

当前页面覆盖已经完成的六条件GPU试跑，不代表180次完整批次已经完成。

通过项目核验服务访问：`http://127.0.0.1:8768/visualizations/gpu_fault_evidence_smoke_003/index.html`。远程浏览器需使用对应端口转发。

6次冻结试验、494帧保存观测；有效检测图由故障后实际送入建图的检测和掩膜绘制。四类结果分别为生成路线、选对对象、正确锁定、实际到达。显示每例是否触发故障以及当前帧是否生效；未触发不算抗故障成功。点击页面中的三个独立审计链接可检查事件时钟、传感器数值、整车和到达证据。评估层默认隐藏。

桌面与手机12项检查及494帧来源哈希核验：`../gpu_fault_evidence_smoke_browser_002/report.json`。初版手机链接溢出失败记录保留在 `../gpu_fault_evidence_smoke_browser_001/failure.json`，修正版使用可换行的证据链接。原001和002查看包保留；003使用紧凑JSON以减少完整批次交付空间。

生成器：`../../source_bundle/gpu_fault_evidence_viewer_003.py`；浏览器核验：`../../source_bundle/gpu_fault_viewer_browser_001.py`。完整批次排队记录：`../gpu_fault_viewer_queue_001/release.json`，须等待监督器和三个独立审计退出并验证完成后运行。预期完整输出为 `../gpu_fault_evidence_full_001/`，尚未生成，不获得完成信用。

查看与生成均不调用模型，不增加实际到达或独立任务数。开发仿真结果不是新封存测试结果，goal_complete=false。原始传感器、试验和失败记录均未修改。
