# 交互仿真操作与复现

本页是实验操作说明，不是论文正文。现有统一查看页仍只读，新交互服务的开始和控制按钮会创建独立开发会话。

## 启动与操作

本机服务已在8769运行。远程先转发8769，再打开 http://127.0.0.1:8769/ 。原人工审核/日志查看服务8768独立保留。

从冻结50任务和17/29/43种子中选择，点击“开始新运行”，等待真实CPU模型与第一帧。默认暂停。单步推进0.5仿真秒，连续运行后可暂停；“停车并结束”按原2m/s²上限逐步制动并保存状态。暂停只冻结仿真时间，速度仍可非零；以终点实际零速为停车证据。前端忙时操作排队；不是实时物理或真实异步Qwen性能测试。

```bash
cd /root/autodl-tmp/lizichong
LD_LIBRARY_PATH=/root/miniconda3/envs/conceptgraph/lib CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.live_navigation_server --port 8769
```

已有服务时不重复启动；单活动worker、服务文件锁以及PID/start_ticks检查防止第二个会话争用。重启服务会识别尚存worker，不自动重跑旧会话。单会话900秒墙钟、30秒仿真、80前端、500MB，最低6GB空闲；暂停计墙钟预算。

## 输入、依赖与产物

同一InteractiveRoadEnvironment重新渲染当前位姿，原YOLO/SAM/CLIP真实前端、PerceptionVehicleAgent、Hybrid A*和车辆积分。仅CPU，OSMesa/PyOpenGL隔离依赖见source_bundle/software_render_runtime_001/manifest.json与accepted_001.json；软件渲染真实进程探针另有证据。模型需合法取得并按项目检测配置放置，ToyCar_002资产来源与SHA见interactive_environment.py及已有资产清单。仓库/交接包不替代模型、授权数据和渲染运行库；异机先重建这些依赖，不声称仅clone即可运行真实前端。

每会话位于interactive_simulation/live_navigation_sessions_001/独立run-id，保存release、source_snapshot、worker_process、原始initial_state、commands与processed_commands、RGB/16位深度/传感器证据、检测/分割、实际crop、perception、execution、终点及车身审核。state.json为原子更新的当前状态，最终结果和原始记录不覆盖。界面图像来自本次新渲染，检测来自本次真实前端；深度颜色图只是有效深度显示副本。

Agent只接收指令、道路先验和当前实际观测；冻结对象身份、可达性与安全区域仅供环境/隔离评价。不能修改查询却沿用旧目标标签。当前只接受冻结指令；自然指令测试需完成真实审核后另行冻结。

## 实际验证与限制

交互实验交付更新（2026-11-12T21:33:31.648770+00:00）：新增可实际启动、单步0.5秒、连续运行、暂停和制动停车的CPU交互入口。桌面/手机实际操作同一预定任务config00-positive0/seed17，共14次真实前端、67控制区间/52运动区间、10条已应用操作、11种RGB与11种深度；暂停状态保持，7步有界制动到零。独立复数圆弧、连续矩形车身和全圆弧方向审核通过，111原始文件SHA一致。终止USER_STOPPED，生成路线/曾选对/曾正确锁定为是、实际到达为否；不并入主450或增加独立任务、人工标签、Qwen/GPU/性能信用。

当前385源码；原机439通过，纯代码437通过2真实数据skip，业务Ruff通过。统一查看入口已验收thesis_portal_015（桌面/手机69链接），新增交互服务需转发8769，已有审核服务8768保留。历史v14不包含本次交互及主450增量；当前源码尚未推送，待最终实验核验后同步。缺一新增450当前快照75/450，其余故障/独占性能继续原队列；人工及合格自然指令以API为准，新封存未开，Goal与四阶段仍为false。本阶段不写论文正文或框架。

本次会话：live_cpu_1794518748728770037_3042f6a3。操作验证见visualizations/live_navigation_browser_qa_001/report.json；独立控制/连续车身/方向见vehicle_execution/live_navigation_independent_audit_001/report.json。后者不调用运行端积分、扫掠或规划器判定成功，不新增控制执行。

自动化浏览器操作不是人工标注；只验证一个任务的交互操作，不增加场景覆盖或正式到达。全部主450成绩保持原值。页面可选全部冻结任务，不声称已经逐个通过交互测试。

验证脚本绑定首次会话和固定输出目录。复跑必须在独立副本另设版本/唯一目录，不能覆盖本机旧记录；不要直接原样运行固定run-id的QA脚本。
