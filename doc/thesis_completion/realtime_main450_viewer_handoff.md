# 主450回放与复现交接

实验与复现交接更新（2026-11-12T20:14:07.197860+00:00）：当前不写论文正文、章节或框架。三模型主450开发条件已结束，449完整加1失败；每模型90可达正条件的“路线/曾选对/曾正确锁定/实际到达”为：规则83/84/82/50，原始Qwen66/72/66/39，新Adapter59/33/32/4。新模型下游未提升，负结果保留。93原到达已有分版本独立车身、方向及2秒保持核验，54初态是显式种子重建；不新增执行或到达信用。

新增完整450只读查看器：`visualizations/realtime_main450_preview_003/index.html`。57,460感知帧、334,607控制记录（含失败前缀），450清单和1,350逐项证据链接已查；桌面/手机12不同条件共24次、144个状态及轨迹前缀检查通过，16位深度5种PNG过滤器逐像素对照PIL、真实深度和384世界点独立核对。按处理完成墙钟对齐已完成感知，首个结果之前不显示未来观测；轨迹来自保存控制状态。旧001/002展示问题、失败记录及修复003均保留，详见勘误JSON。浏览器核验是代表条件检查，不是450次逐帧人工审核。

项目424测试通过（5条依赖警告）、纯代码副本422通过2项真实数据skip，业务Ruff和差异检查通过；首次043命令遗漏动态库且误收集第三方测试的失败日志保留，不记为通过。当前381源码已准备，最后公开365源码版本；最终实验核验后再同步GitHub。缺一语义新增450正在跑，本次快照29/450；GPU故障和独占性能继续按队列等待。人工/合格测试数量以以下清单API快照为准，新封存保持关闭，Goal为false。v14旧包不含主450及本次查看器增量。

复现入口（在项目根目录、既有conceptgraph环境中，使用独立run-id；不得覆盖已存在目录）：

```bash
export LD_LIBRARY_PATH=/root/miniconda3/envs/conceptgraph/lib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
/root/miniconda3/envs/conceptgraph/bin/python -m pytest -q tests
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.realtime_matrix_demo --run-id 新唯一smoke目录 --limit 6 --wall-budget-s 900
/root/miniconda3/envs/conceptgraph/bin/python -m conceptgraph.completion.realtime_matrix_demo --run-id 新唯一全量目录 --wall-budget-s 900
```

复现依赖完整主450原始感知与控制产物及已生成的分版本独立证书。无reuse模式重新核验全部输入；`--reuse-viewer 已完成查看器目录名`仅在旧生成物和指纹一致时复用原核验清单，必须另用run-id，不能把复用称重跑模型。总元数据预算128MiB、最低剩余6GiB、只读CPU共享资源，不接管GPU队列。查看通过HTTP服务，当前8768已运行，远程需转发该端口：

`http://localhost:8768/visualizations/realtime_main450_preview_003/index.html`

真实人工开发审核：`http://localhost:8768/visualizations/causal_development_review_002/index.html`；自然表达审核：`http://localhost:8768/annotation/natural_command_review_001/index.html`。人工填写审核后才产生真实记录，自动浏览检查不代填。新封存测试仍关闭。

后续顺序：450新增缺一消融及独立统计→GPU故障smoke与180全量→独占GPU性能→可靠人工/新测试数据审核、方法冻结后新封存→完整可视化/保护/复现/导出核验→GitHub同步与最终实验交接。当前不写论文文字。
