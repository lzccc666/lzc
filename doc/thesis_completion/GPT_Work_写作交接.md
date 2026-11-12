# 毕业论文写作交接：代码与实验证据

课题：面向智能车辆的三维语义场景图构建与导航方法研究。

本文件用于后续GPT Work拟定论文框架与写作，**不是全课题完成证明**。当前 `goal_complete=false`，阶段9—12仍有必需验收未完成。后续新增运行结果以实时 `validation_report.json`、各run的release/summary与原始逐任务记录为准，不以这份快照替代最终统计。保留负结果、失败、unknown与未开启的封存测试身份。

## 先读这些文件

1. `工作交接_代码与实验.md`：本机路径、执行入口、Git交付及仍在运行的任务。
2. `validation_report.json` 与 `remaining_issues.md`：每项的完成范围与剩余条件。
3. `reproducible_commands.md`：真实入口、续跑条件与预算，不能复用已经存在的run-id覆盖结果。
4. 本文件列出的独立统计JSON、原始记录与SHA。写作中的每个数字必须回到对应分母与范围。
5. `thesis/manuscript_delivery_016/完整论文阶段整合稿.md`：可作为代码和旧稿整合的参考材料；不要把它当作已经完成的论文框架或终稿。用户将另行拟定框架和撰写。

本轮输出根目录是 `/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1/`，以下相对路径均基于此目录。原始第1—3章、原始Word、MEMORY/PAPER_MEMORY和阶段1—8结果只读，不能覆盖。

## 已有证据能够支持的事实

|模块|实际完成的实验|可以使用的结论与限制|主要证据|
|---|---|---|---|
|真实前端|38开发场景、11个log、1508帧YOLO/MobileSAM/CLIP真实重跑|修复SAM原地修改YOLO框的别名问题；检测框、掩膜、真实crop和图像哈希一致性可追溯|`mapping_evaluation/frontend_corrected_dev_001/`及`source_bundle/tests_full_018.log`|
|对象定位与融合|同前端、多条件全量开发重建|原融合召回19.48%、匹配精确率17.80%、中心误差1.662m；深度支持条件32.12%、42.99%、1.250m。身份变化24→44，不能声称全部对象维护指标都改善|`statistics/corrected_mapping_001/report.json`、`statistics/robust_motion_001/report.json`|
|关系|直接几何、结构化与增量关系独立评估|直接F1约0.2020，结构化约0.1993，增量约0.1988。实现一致率不是语义正确率，不能宣称复杂关系必然更准确|`mapping_evaluation/relations_depth_both_axes_001/`、`mapping_evaluation/relations_exactgeometry_001/`、`ablations/relation_evidence_dev_001/report.json`|
|语义模型|新v2 Adapter种子17/29/43各240步真实CUDA训练；五模型各603个相同输入请求|新三种子完整解析均150/180，真实crop类别227/226/231（分母243），受控排序57/51/51（分母180）。只有2个真实验证log，排序含符号bank；不能外推真实道路目标或导航收益|`main_experiments/fair_three_task_full_validation_001/report.json`、`statistics/fair_three_task_full_validation_001/report.json`|
|检索|38场景、1115查询，733正/382负|最近类别/CLIP/稀疏VLMaps式适配/联合规则正选对211/149/132/45，负误选141/308/330/8。联合规则用较低覆盖换较少误选，不是正目标准确率最优；稀疏适配不是完整VLMaps复现|`statistics/retrieval_baselines_dev_full_001/report.json`|
|车辆规划上限|50冻结任务×3种子×2规划器=300次|给定正确对象时Hybrid在90正试验到达70次，点A*62次；配置聚类差异区间跨0。GT辅助，不计真实语义导航成绩|`statistics/kinematic_oracle_002/report.json`|
|真实感知执行功能|50任务×3种子=150次规则闭环；10配置共享3种道路家族|90可达正试验路线68、选对90、稳定正确锁定79、实际到达21；13不同正任务、8配置满足最低功能门槛。无匹配与不可接近各30试验分开；原地等待不计自主识别不可达或运动安全信用|`statistics/interactive_rule_full_002/report.json`、`interactive_simulation/interactive_rule_full_002/summary.json`|
|KITTI新增|固定64帧、33个原始drive、真实CPU前端，两条件共享输入|car/truck/pedestrian共319个GT；3m中心召回247/319→262/319，即77.43%→82.13%；251个共同支持的二维匹配中心误差1.758m→1.540m。卡车误差2.768m→2.888m，没有改善；低深度支持16→18保留|`statistics/kitti_object_cpu_full_001/report.json`、`records.json`、`common_supported_pairs.json`|

KITTI方法差异按33个drive配对4000次bootstrap：召回差+4.70个百分点，95%区间[+1.48,+9.07]；共同支持中心误差差−0.218m，区间[−0.444,−0.047]。这仅是自定义三维点中心诊断，**不是官方KITTI AP，也没有KITTI导航到达**。Cyclist/bicycle框语义不一致，单列代理诊断；其二维匹配0/20不能隐藏。YOLO/CLIP上游预训练与KITTI重叠未知。

## 必须保留的范围与负结果

- 原1508开发帧LiDAR比相机晚约33—42ms，旧融合、关系、检索和故障实验只能写为离线配对诊断。新预留1196帧已转换为1166真实过去扫描与30显式缺测，独立核验通过，但没有运行封存方法；不要把旧数据回写成严格因果在线证据。
- 联合选择38/733与首个可达36/733，配对差异区间包含0。去安全接近点只离线，不实际执行关闭保护的条件。
- 六地图条件共1272次点规划；方向连接约束减少路线覆盖但提高点车道合法率，仍有3条连接审计失败；这些姿态未通过矩形整车检查。不能把点路径合法解释为整车可执行。
- 运动状态来自多帧。单帧crop不能作为可靠静动态真值；颜色/细属性/歧义没有人工记录时保持unknown。
- 新Adapter初版存在提示过拟合，真实负结果已保留。v2结构化验证改善不能替代完整下游主比较；当前尚不能声称新Adapter提高导航到达率。
- 当前三维环境是控制动作改变位姿、RGB、16束LiDAR的真实渲染与低速自行车模型执行，使用CC0 ToyCar。不是CARLA、真实车辆、完整动力学或任意动态交通安全验证。
- 路线生成、选对语言目标、稳定正确锁定、实际安全到达是四个不同指标，必须分别写出分子/分母。GT上限、缓存命中、模型返回、排序应用、改变选择也不能互相替代。

## 尚不能写成“已完成”的部分

1. 真实时钟三模型450试验主矩阵仍运行；随后450语义缺一试验排队。完整与关闭Qwen共享部分条件，不能把重复运行当成新的独立任务。
2. 30场景14新log的新测试尚未开启。600构造主指令与278挑战指令不能当作已经满足自然分布400条有效静止目标指令。
3. Talk2Car原始439条英文覆盖30场景360帧，但只有176条带官方静态属性且仍需任务适配；仅3场景有至少20条原文。真实人工审核记录目前为0。
4. 独立静态区域307/422正查询获证；独立条件控制参考在878查询中仅81条有证、797条unknown。条件为静止起步、当前障碍冻结、90秒与2秒保持，不是待测方法成功或记录车辆真实状态的到达证据。
5. 独占冷启动/无缓存模型/前端/建图/规划/执行性能还没有完整验收；共享主机或GPU结果不能并入独占性能主表。
6. 全文引文主张、学校模板、全部47页与本机Word视觉验收尚不完整。85条连续参考文献不等于85项正文支持已经核验。

## 现有写作材料的可信范围

016版实际导出了47页PDF、可编辑Word、19张第6章表、7张图、11处OMML公式、85条连续引用。版式与部分页面已检查，全部参考文献正文支持仍未完成。使用其表格时优先读取对应CSV和冻结JSON，旧稿中的早期阶段数字不应覆盖最新独立统计。

`thesis/claim_citation_review_015.json` 保存11处限缩和11组细节复查。ALIGN表示学习不能写为检测器；A*原论文条件不能外推到有预算的实现；本项目是前向车辆搜索适配，不能称完整复现Dolgov两阶段与倒车策略；稀疏VLMaps式网格不能称复现LSeg像素映射。错arXiv、反爬HTTP200、仅元数据页面与下载失败均保留，不能充当全文内容证据。

写作时可围绕已有方法和真实改动组织论证，但创新性和最终结论必须由一手文献比较与完整结果支持。请先依据证据制定框架，再起草；不得把剩余项用预期结果补齐，也不得将此交接材料直接当作正式论文正文。

代码仓库已同步：https://github.com/lzccc666/lzc ，远端main提交`60cd334a1df6a77e7a4652ad0244db43a5588e01`。详细入口在仓库`README_THESIS_COMPLETION.md`。运行中的矩阵与未提交人工标注不因公开代码而变为完成。文献抓取009中38/59/65的HTTP429空body误用.json后缀，005扫描失败保留；类型核验002将其明确列为原始失败响应，不能作为JSON元数据或正文支持。


## 开发集过去LiDAR续跑补充（2026-11-12）

新增development_causal_inputs/audit/mapping/statistics/pipeline/downstream模块。固定38开发场景11log1508相机帧，逐帧复用相同RGB真实检测，新前端调用为0；按相机之前100ms内最近实测扫描重投影，缺测保留，旧未来LiDAR结果继续仅作离线诊断。真实3帧试运行已完成转换、独立原始表选帧/标定/像素重投影（0不一致）、七建图条件及关系/四检索调用；该片段对象0/2，无正查询的检索指标为null，失败NaN写出原件保留并修复。全量293 tests通过，5依赖警告。

完整流水线已真实启动（source_bundle/development_causal_pipeline_full_001），转换后自动执行独立审计、38场景七条件重建、11log配对统计；causal_downstream_full_001等待其完成后运行关系、选择上限、四方法检索及统计。不能把排队当完成，最终状态查各complete/report。官方sample GT仍较相机晚约33–42ms，必须披露评价时间近似。本批不提供路径、锁定、实际到达或新封存信用。

可查日志：source_bundle/development_causal_pipeline_full_001.log、source_bundle/development_causal_pipeline_full_001/{convert,audit,mapping,statistics}.log、source_bundle/causal_downstream_full_001.log。输入输出与审计位于data_manifest/development_causal_inputs_full_001；结果位于mapping_evaluation/causal_mapping_full_001、statistics/causal_mapping_full_001。旧GPT_Work_论文写作交接包_v1.zip保留原检查点，不包含此后新增实验；工作目录与validation_report.json是动态进展入口。

新增代码快照已于2026-11-12T12:09:31.779896+00:00同步GitHub main `7706d7358b5cd74a33feda6d5fe32c8b48a6d3e7`（增量源文件总数356；当前原机293测试，隔离导出291通过/2真实数据依赖跳过）。原checkout索引保持，真实完整实验仍在后台；不是Goal完成证明。


开发causal完整输入及独立审计现已结束：38场景11log1508帧，1470真实过去扫描、38缺测，未来0，年龄3.870–63.661ms；1508帧独立重投影像素差0，7659文件SHA通过。实际resume 0归档读取/0新增输出。官方sample GT偏移实测33.136–42.356ms，保留评价时间近似。

新增显式输入版本入口：fault_robustness --mapping-run；mapping_module_ablation --mapping-run和--baseline-fault-run（必须帧SHA/seed17/配置相同）；map_constraint_ablation和goal_selection_ablation --retrieval-run。默认旧版本保留，禁止用旧未来LiDAR完整条件作新causal消融基线。source_bundle/causal_ablations_after_downstream_001.py已启动等待，同版完整causal下游结束后顺序做684扰动、114模块记录、六地图及三目标接近条件；协议development_causal_ablations_001.md。真实smoke均完成，未把排队当结果。全量tests294通过，5依赖警告。全部阶段仍未验收。


完整38场景过去LiDAR重建统计已结束（statistics/causal_mapping_full_001）：5723个跨帧GT观测全保留，38个缺测帧130个GT观测均0匹配。原融合986/5723（17.23%）、匹配精确率16.49%、中心误差1.696m；深度质量1723/5723（30.11%）、43.68%、1.303m。召回差+12.88pp，11log配对95%CI[8.12,18.08]pp；中心误差差−0.393m，CI[−0.543,−0.270]m。原始身份变化30/615、深度44/1312，率差区间跨0，不称身份改善。稳健运动条件1720/5723、44.05%、1.282m，保留召回略降与身份未知。以上是开发集同输入方法比较；官方sample GT时间近似仍在，未开启新封存，也未做车辆到达。

旧未来LiDAR诊断的19.48%→32.12%等数据仅保留原离线口径，不能混用为当前过去LiDAR成绩。新统计包含七条件及实测/缺测分组、10556帧—方法行与4000次log配对bootstrap；关系及四检索在同批输入上顺序推进，后续完整录制故障/消融按冻结队列执行。
