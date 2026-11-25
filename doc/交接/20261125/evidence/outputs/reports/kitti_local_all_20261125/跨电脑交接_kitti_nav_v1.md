# KITTI、导航任务与Qwen换机交接（暂停状态）

当前用户要求暂停完整实验，先保存进度并同步GitHub。唯一执行状态见 `doc/交接/20261125/执行状态.json`：人类未明确“继续”前，不启动watch、full、GPU、训练或KITTI单8补跑；旧关系release不解除换机hold。新机器先读此状态和主交接，不能自动恢复旧Goal或旧控制器。

## 已完成的KITTI完整对照

Tracking本机实际11序列共1030帧，三方法已完整CPU重放并独审。不是21条完整原始序列。GT3257；同类、initial-IMU world XY≤3m，限定原类别/相机深度/标签范围。

| 方法 | TP/GT | 召回 | 匹配预测占比 | 重复代理/预测 | 自定义IDF1 |
|---|---:|---:|---:|---:|---:|
| 单帧 |846/3257|25.97%|846/3932=21.52%|42/3932=1.07%|2.36%|
| 原融合 |1235/3257|37.92%|1235/8865=13.93%|75/8865=0.85%|14.57%|
| 稳健方案 |2739/3257|84.10%|2739/7002=39.12%|307/7002=4.38%|50.32%|

匹配预测占比不是覆盖全部未标注区域的完整精度，不是官方AP/HOTA。稳健重复率回退必须保留：307次中199为last_seen旧于当前、108为当前更新；298 ID曾在过去输出、9首次导出，249 car/58 pedestrian；0002/0007合217次。旧last_seen不等于错误节点，不能据此断言物理重复或关联错误因果。诊断与424逐条记录见 `outputs/reports/kitti_local_all_20261125/Tracking重复率回退_只读补充.md` 和 duplicate_proxy_occurrences_v1.jsonl。

Object完整有标签7481帧，mean/robust共享真实新GPU前端，0失败。三类GT33191；同类、camera0完整XYZ≤3m，不能与Tracking的XY成绩直接比高低。

| 方法 | TP/GT | 精确率 | 召回率 | F1 | 无深度预测 |
|---|---:|---:|---:|---:|---:|
| mean |24228/33191|63.78%|73.00%|68.08%|1288|
| robust |26332/33191|69.32%|79.33%|73.99%|1553|

car占GT84.19%；robust类召回car88.18%、truck35.23%、pedestrian31.69%，macro51.70%，须与micro79.33%并列说明。robust无深度+265、已匹配3m内均误差1.1576→1.1712m为负项。Cyclist→bicycle代理另表，不混三类。Object单帧无身份稳定性/融合/动静成绩。7518无本机GT的testing传感器帧已列范围，不做无评分收益GPU推理。GPU累计56.99分钟且已释放。

结果/全类别/固定距离桶：`outputs/results/kitti/validation/iterations/kitti_local_all_20261125_tracking_v1/`、`kitti_object_all_20261125_v3/`；中文主表、微宏平均和独审在 `outputs/reports/kitti_local_all_20261125/`、`kitti_object_all_20261125/`、`local_scope_audit_20261125/`。原64仅缓存后处理复现，不能替代7481真前端。

## SemanticKITTI适用性盘点

22序列43552 LiDAR帧，00–10共23201点GT；完整认证22400组RGB2+LiDAR+pose+点标签输入配对（10序列），只是输入就绪，未跑22400建图。03的801 LiDAR/点GT存在，但Depth包缺对应raw0067 RGB；11–21有LiDAR/pose/calib、无公开点GT，现devkit无已证raw对应，不能笼统说没数据。pose是提供方SuMa离线轨迹，不是实测OXTS或因果/车头/动静真值。点标签不是3D框中心，须另立指标后才能做新实验；本轮不造点簇框GT。

完整轻量清单 `outputs/reports/kitti_local_all_20261125/semantic_RGB_LiDAR_pose_pair_inventory_v3.json`、`SemanticKITTI_22seq_applicability_final_v2.json`、`SemanticKITTI适用性_最终盘点.md`，独审passed；600s partial和13.6min续查证据均保留。

**KITTI单8关系overlay与GT关系评价：未开始，0新GPU/0新完整CPU实验。** 待用户继续后，先核Tracking每帧标定front代理和已审匹配、Object camera0地面(z,-x)与原XYZ匹配索引，再冻结≤15minCPU/≤128MiB预算。Object只能称单帧空间关系诊断，不能称时序地图。

## 已完成的新八方向导航接口与2315题真值

共享provider `conceptgraph/scenegraph/relation_single8_20261125.py`：无序pair≤1条，8×45°半开扇区，前0/左正，稳定ID字典序source/target，relation描述source相对target，worldXY距离(1e-6,12]。无near/反向重复边；反向只查询inverse。相机光轴XY是参考朝向代理，不是对象车头。

我的matcher `conceptgraph/navigation/single8_task_20261125.py` 支持复合中文优先；一条front_left可答左/前/左前，近按距离默认6m、明确米数按文字。两个通道分开：普通对象几何检索允许所有motion态但不证明静止；确认静止通道保原raw/深度门，单帧无历史不作零分基线宣传。0路线/区域成功/实际到达主张。

新数据 `dataset/processed/nu_task_single8_20261125_v1/`：全部2315原题ID/文字/帧保留、502独立官方GT/原pose齐。827方向题仅按预定8扇区及原参照/静止GT重算，**没有用模型输出改GT**；1488类/最近题接受集逐值不变，原60pending完整保留。

| 协议 | legacy1115 正/负 | unified1200 正/负/pending |
|---|---:|---:|
| 原历史 |733/382|141/999/60|
| 新single8 |695/420|138/1002/60|

64条接受集变化：41正变负仍作为负例保留，23仍正但可接受对象变化，绝不删题/混分母。证据为 evaluation_only/{GT_context,task_labels,label_change_ledger}.jsonl，manifest SHA `3a1372080afa896a8c4f122f5c646091f30479d7884ba18fdeb5731138340740`。独审 `single8_task_GT_independent_v1.json` 用独立最近射线/词语拆分重算并passed。新方法infer/评分尚未运行，不能搬历史93/733或9/141作全量新成绩。

安全小合同命令（仅用于已授权的小检查）：`CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/conceptgraph/bin/python -m scripts.nu_task_single8_20261125 contract`。prepare已完成，重复prepare会拒绝，勿覆盖旧包。infer/evaluate入口已保存但**暂停中禁止执行**；只接受未来 `nuscenes_full_20261125_single8_main_v1/<scene>/frames.jsonl.gz`，旧main与旧8pilot不当新正式流。新nu八场景318帧×7纯CPU重边由nu线交付；其分母/GT evaluator不扩反向，匹配ID顺序翻转经canonical_fact再inverse。

## Qwen实际状态

40条真实候选重排样本单seed17短训已完成；16条人工审核文字不入训练，仅已见解析回归，不能当16个场景/轨迹GT。完整171验证中95正任务有59漏候选、仅36目标可选；规则v2选对20/95，base9/95，旧adapter12/95，新adapter8/95，**本轮微调无收益，Qwen未作为优于规则的主选择器**。新adapter故障误选53/57、语言严格schema1/16且全字段0/16，不能用大量拒选或宽松格式刷分，也不能据schema失败说理解能力为零。

详细证据 `outputs/reports/qwen_iteration_20261124/首轮Qwen完整验证.md` 与comparison_summary.json；逐请求 `outputs/results/qwen/validation/iterations/qnext_20261124_comparison_v1/`；adapter `outputs/results/qwen/train/iterations/qnext_20261124_seed17_v1/best_adapter/`。未新增训练。后续若用户继续，先固定无答案暗示的输出契约，验证schema/拒选，再决定是否短训；不自动长训。

## 监听、续跑草稿与暂停边界

历史watch v1/v2/v3均停止，最后v3进程28307已退出；0/38正式场景、0/2315消费，累计active子进程0s。没有新watch4运行。其36h旧注册/失败/停止记录保留，不能复用旧main路径监听。未来需新single8 watcher，48h应沿原11:02起点；active1800s/128MiB不清零，真实终态退出保留pending。用户换机hold优先且不自动过期。

resume v1 `conceptgraph/completion/nu_pipeline_resume_20261125.py` 是已冻结旧namespace/12h36h/runtime6技术实现，只做CPU锁/PID/坏SHA合同；**不是新single8推荐入口、未真实全量/cache修复运行**。v2同前缀 `_v2.py` 为交接草稿：缓存闭合、发布前移旧marker留档、已闭repair_work恢复、孤儿worker只读证据等代码已保存，但仍待新single8 DATA/method/runtime和32h/48h/5.25GiB实际对齐与独审。当前v2除selftest外全部CLI硬阻断，先查tracked执行状态（缺失/active即拒），再查换机hold；不能调用旧release自动恢复。真实长跑/实际缓存CPU恢复/最终评价overlay都未验证，勿宣称完成。

有效资源历史文件是旧source DATA的effective_resource_policy_v8.json/overall_execution_clock_v2.json：GPU32h、整体48h、nu5.25GiB/其余0.75GiB、TMP512MiB，原2026-11-25 11:02起点不重置。它们不覆盖当前用户暂停；恢复时须重新核时钟是否已过期和实际磁盘，禁止自动延长。正式新method_manifest尚未创建，不能从旧method_manifest伪造。

## 换电脑如何续接

若新电脑Codex仍SSH到同一服务器：沿用服务器真实输入/权重/结果，进入唯一主树 `/root/.codex/worktrees/lzc-integration-20261125/lizichong`，确认Git最终main与本地兼容链接。不要回到另一个陈旧work分支重新做方法。

若新电脑是全新本地环境：clone Git只获得源码和小交接证据，不会自动获得NAS原始RGB/LiDAR、模型权重、被忽略的dataset/outputs全流和adapter。先按主交接来源索引核实际数据/权重是否可达；缺失时只能审代码/论文/已有摘要，不能宣称实验可直接续跑，也不自动下载/复制大数据。GPU锁绝对路径是原服务器约定，迁移到新服务器必须登记新路径并重核资源。

人类明确继续后的顺序：核执行状态与最终Git commit → 核数据/权重/所有已冻SHA → 完成new single8 bootstrap/GT scorer/resume/watcher接线与小CPU独审 → 固定新正式source与budget/release → 再启动全量。KITTI单8和其真实scorer反序fixture仍待做。原历史失败、默认、论文、MEMORY/PAPER_MEMORY不覆盖。

本补充来源/小证据SHA见同目录 `跨电脑交接_来源收据_v1.json`。本线已封笔，无后台watch/resume/GPU；主线全局进程和最终push由主交接统一确认。
