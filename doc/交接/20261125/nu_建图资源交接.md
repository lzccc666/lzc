# nuScenes 建图与资源交接（2026-11-25T14:05:28.279979+00:00）

**当前必须停在交接状态。** 用户要求规则改完后先不要跑完整实验，先保存并推送 GitHub。唯一 tracked 状态是 `doc/交接/20261125/执行状态.json`，`user_handoff_hold.active=true`，不自动过期。原 ignored DATA 下也保留 `user_handoff_hold_v1.json`。缺失状态同样不能推断获得执行许可。仅人类后续明确要求继续后，接手者才可修改状态；源认证/资源门/原关系定义确认不能解除此暂停。

## 已完成及其边界

1. 八方向核心已实现并封存：`conceptgraph/scenegraph/relation_single8_20261125.py`，SHA `66efdde5b633fdc8a7f9939007cfa20f35d766314a3d274d22b608ba5f7ea124`。无序物体对按 stable ID 的 `str(ID)` 字典序规范化；一条边的标签表示 **source 相对 target**。世界XY距离为 `distance_m`，距离 `(1e-6,12]m` 的全部有效位置对各一条边；同点≤1e-6不强猜方向、不合并节点。没有 near 边、没有反向存储边、没有旧confidence/topk/degree剪枝。动态和运动unknown节点只要XY有效均可连边。
2. 前向用 CAM_FRONT 光轴 worldXY 投影**代理**，不是物体车头、也不是已验证真实车身朝向。8个45°半开扇区，前[-22.5°,22.5°)，左为正；API `direction_for_delta`、`build_edges_from_centers`、`canonical_fact`、`query_pair`。逆查询反转标签；GT映射后ID规范序改变必须反转标签。
3. 旧8场景多关系pilot完整保留；新纯CPU重边独立run `nuscenes_full_20261125_single8_pilot_v1` 已完成 **8场景/318时刻/7条件2226方法帧**，新增GPU调用0，未读GT。非关系字段逐值完全一致（节点/身份/中心/运动/观测/时间/lineage/原节点资格均不变）。允许变化字段仅 payload `edges, relation_contract, relation_statistics, relation_provider_sha256`，root新增相应contract/provider；旧payload不存在额外旧边缓存计数。所有新边数量由新统计重算。
4. 新8重边方法累计边数：single334、原融合1390、R1 863、S2_1 746、S2_3 2830、S2_6 20271、共享scan辅助20373。单位是跨318时刻累计的边帧，不是唯一永久边。7条件有效节点累计24209，含动态/unknown，不只静态停车节点。稀疏节点无形状仍无车辆安全授权。
5. 新GT评价入口 `single8_evaluate.py`、辅助评价、统计已写；保全GT关系分母与端点匹配条件分母，单无序对计一次，映射反序fixture真实函数通过。节点3m同类XY一对一/估计和verified运动各分母保留。**未执行官方全场景新评价，未产生新版850成绩。**
6. 核心+8重边已独审通过：`outputs/reports/local_scope_audit_20261125/single8_pilot_independent_v1.json`；本线语法及真实评价反序fixture：`outputs/reports/nuscenes_full_20261125/single8_handoff_CPU_check_v1.json`。2315导航新GT/解析由另一线交接，不把历史93/733当新版图成绩。

## 当前数据/进程与换机注意

本机官方metadata实际850场景/34149时刻/68采集log；6相机RGB204894张。包内可用成员仍必须以逐包证书为准，不能以JSON列名替代传感器存在。源索引状态快照见 `nu_进程索引快照.json`，当前完整包 1,2,3,4,5,6,7,8/10，活动PID {"9": 29826, "10": 29914}；队列为空，**本轮不新开其他任务**。唯一活跃协调器PID26371，仅认证在读08/09/10，deadline 2026-11-25 14:32UTC，既有C索引线程不能kill。它在deadline会SIGSTOP活跃进程并写checkpoint保内存；若需交接前停止，应保留其owner和进度，不直接杀子进程。若自然完成，写全10证书，不会启动GPU/full。该本机准备进程无法迁移内存到另一台机器。

本机root `/root/.codex/worktrees/lzc-integration-20261125/lizichong`；dataset真实root `/root/autodl-tmp/lizichong/dataset`，outputs真实root `/root/autodl-tmp/lizichong/outputs`。共享原包 `/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval01_blobs.tgz`…10，约294GB。**原包、原始metadata、模型权重、完整预测/索引不在Git中**；新机器clone代码后需自行有合法相同数据，或访问原机器。不要照抄旧主机绝对symlink。源证书绑定archive stat/SHA及真实member，换路径需要合法新绑定，不可手改旧证书假称相同。索引隔离依赖 `/tmp/nuscenes_full_index_20261125` 的 indexed_gzip1.10.3，不污染冻结环境；该/tmp也不会随Git迁移。

## 有效目录与命令

- 索引/源元数据：`dataset/processed/nuscenes_full_20261125_v1/`，progress：`outputs/reports/nuscenes_full_20261125/phaseA_progress_v3.json`；已关闭包证书 `archives/archiveNN_complete.json`，实际index路径由证书指向。03–05旧deadline失败完整保留，v3恢复明确import已CRC导出的旧index，不冒充重新CRC。
- 旧8节点来源run：`outputs/results/nuscenes/validation/iterations/nuscenes_full_20261125_resource_v1/`，新8单关系：同目录 `nuscenes_full_20261125_single8_pilot_v1/`。source/model完全复用已付GPU节点。新完整预定run **`nuscenes_full_20261125_single8_main_v1` 尚未创建**，旧 `nuscenes_full_20261125_main_v1` 也不存在/不得重启。
- 可只读验证：`PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/conceptgraph/bin/python scripts/relation_single8_independent_20261125.py --help`；具体独审已闭收据优先读取，不为交接重复大实验。
- 新关系重边命令历史已执行：`scripts/nuscenes_full_20261125_single8_reedge.py`（exclusive创建run，**不可在同名已关闭run再次运行**）。
- 新评价命令：`scripts/nuscenes_full_20261125_single8_evaluate.py freeze/evaluate`，辅助 `...single8_sharedscan_evaluate.py`，统计 `...single8_statistics.py freeze/run`；完整输入/协议/预测未齐之前不能evaluate。新GT源码只评价端导入，不能进入方法输入。

## 正式850仍待接线验证，不能直接运行草稿

`conceptgraph/completion/nuscenes_full_20261125_resource_runtime_v8.py` 是**明确draft**，现最前端已读tracked hold并硬拒；即使未来取消hold，也要求 `full850_execution_validated=true` 才执行。它尚未完整切换新single8 context，不能把它叫做已可用正式入口。`single8_context.py` 仅建立新DATA及合法原源链接/新manifest；**新method_manifest还未创建**。

后续接手者在用户明确继续后，首先完成一个新的版本入口（不改旧65冻结业务实体）：CPU initializer同时把 base/aux 两处 CurrentRelationCache 接共享single8；base.node_record补写新contract/provider/statistics；同进程双sceneCPU零差核节点；newprepare写新source/GT/资源methodmanifest；newpromotion绑定新8关系而非旧多关系。当前new8 accepted_attempt下缺供部分通用消费者读取的attempt/complete.json，需要独立metadata适配并验证，不能覆盖原节点/边结果。原runtime8 idempotent_promote仍用旧manifest比较，不可直接用于新single8。

资源/调度已有真实小验可复用：CPU worker重复interop=1幂等（4→5→4→5同PID79帧全部7方法除mapper_seconds外exact）；资源guard adapter只替换32h/48h/5.25GiB三个执行常量且证明可逆；单reader RAM队列1预取2scene558源成员SHA精确、峰值约67MB。**未把这些工程小验冒充新版850/resume长跑成功。** 新resume/watch版本由导航线交接，同样必须服从tracked hold；当前没有自动full启动器在运行。

## 时间、GPU、存储与统计

有效资源policy见小证据 `effective_resource_policy_v8.json`：累计实际GPU驻留≤32h，整个任务从2026-11-25 11:02UTC起≤48h（Nov27 11:02UTC绝对deadline），nu持久≤5.25GiB，其他全部≤.75GiB，合计≤6GiB；全任务临时≤1GiB，我线磁盘512MiB+预取RAM128MiB合算640MiB。真实outputs磁盘不足20GiB停新增大文件，保全进度，不删共享数据。当前GPU进程为空。旧8累计真实占卡1028.120147s，1908正常+48固定配对=1956调用，历史失败也计一次；首次240的逐模型时间缺失明确未知，不能按零。完整204894主图只复用同一次前端，额外48只pilot，不默认每scene多6次。旧12h/36h门不通过和错误passed原件均保留；新32h/48h/5.25GiB必须实guard与源码绑定，不能只改报告。

新8数据输出实际合计约24.4MB（所有7方法318时刻），较旧丰富多谓词边更小，但**全10索引实际字节+新8序列化+尾部大场景风险尚未完成新正式资源投影**，不可用旧4.43GiB直接放行。五预定主比较（orig−single、R1−orig、S2_1−R1、S2_3−S2_1、S2_6−S2_3）按log配对bootstrap10000/seed17、召回99%Bonferroni；P/F1/ID/重复/中心误差95%描述CI，全范围分母及失败场景保留，1/3/6已用测试不选参。辅助sharedscan第7条件单表，6583时刻多scan混杂需保留来源而不能归因全相机收益。

## Git所带证据与未带内容

`nu_源码SHA清单.json`列本线新旧入口SHA，`nu_小证据/`带规则、8CPU摘要/独审、真实资源policy/clock/hold/合成fixture。完整逐帧节点/边、旧失败、原包、完整index/模型都留原路径，不删不覆写。当前仅完成规则和小试，不声称全场景对照完成或第一创新点所有质量瓶颈已解决。
