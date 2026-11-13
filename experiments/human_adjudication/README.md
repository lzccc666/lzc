# 真实自然表达采集之后的独立标签审核

当前实际人工表达和审核记录均为0；页面、10项合成夹具测试和浏览器流程验收通过，不代表已经取得人工标签。没有开启新封存测试。

## 使用顺序

1. 映射8768端口，先到`/annotation/human_natural_collection_001/index.html`观察原始连续RGB并提交本人自然表达。此处保存原始表达，不是有效标签。
2. 映射8770端口，打开`/annotation/human_natural_adjudication_001/index.html`。该页面读取8768已保存的原始表达，切换条目后绑定原始记录和SHA。不要在8770重复写一份表达。
3. 审核者查看指令绑定帧及需要的时序证据。官方投影框默认关闭，可开启帮助核对官方实例ID；其为近似对齐的3D框投影，不是像素掩膜。选择所有可接受实例，填写类别、所需细属性、关系及锚点、否定、歧义、运动与观察依据。无法确认保留unknown，不凭作者提示点自动指定实例。
4. 静止主任务至少查看3帧、时间跨度至少1秒，审核者本人判断静止范围；单帧官方属性不能替代该核验。无匹配仅限已观察前缀，需查看完整前缀，不等于整条道路没有目标。所有实际所需语义字段须完整勾选，颜色、细属性和歧义不能凭自动标签补齐。
5. 提交为追加记录。修订指向上一个review_id，不覆盖旧记录。页面依靠审核者真实观察、自述和说明，不声称独立验证了审核者身份。需要独立复核时保留多条真实记录及分歧，不由代理代签。

## 统计与验收界限

标签已审核、检索标签可用、有效封存测试三者分开。后台对缺失语义、未知颜色、类别不符、错误锚点、单帧静止和审核版本冲突进行检查；无法自动保证审核者解释正确或完整。实际需复查可接受集合及所有关键字段。

当前qualified_test_queries仍为0，即使将来有单条标签通过，也不会自动解封。仍需逐场景≥20有效指令、最低20场景，总计≥400，既有30场景计划则≥600；划分/表达家族审计、标签可靠性、连续回放预选、独立运动学可达性、方法/模型/指标协议冻结均另行验收。未获得可达证书保持unknown，不因待评价规划器失败宣称不可达。

官方真值只供本标注服务和隔离评估，未传给感知导航agent；后续画面若用于标签审核，不能因此让在线方法读取未来观测。该新增服务不改8768采集代码，也不改原GPU冻结链。

## 证据和服务

服务PID 430242、启动时刻 1383380054，进程信息见service_process.json。检查/api/adjudication_status；正式审核保存在annotation/human_natural_adjudication_records/。源表达在annotation/human_generated_natural_submissions/。

10项后端/HTTP测试见reproduction/human_adjudication_fixture_validation_001/；桌面与390像素手机只读浏览器验收见visualizations/human_adjudication_browser_qa_003/，22请求均GET。完整表单切换/选择/提交只在reproduction/human_adjudication_browser_fixture_002/合成服务执行，1次POST明确标synthetic_fixture；正式服务POST为0，人工记录新增0。

首轮浏览器字段禁用判断失败日志和初版页面history_001保留。页面第2版补齐真实提交切换事件；第3版在切换表达时清空旧标签、依据与确认勾选，合成浏览器已验证；最终核验见source_bundle/human_adjudication_final_verification_002.json。新辅助文件尚未最终Git同步。

当前服务器运行时可直接使用，无需重复启动。若进程已经终止，先确认原PID/tick及端口；后续重启使用相同脚本并另存日志，不覆盖已有启动记录：

`PYTHONPATH=/root/autodl-tmp/lizichong OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' /root/miniconda3/envs/conceptgraph/bin/python /root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1/source_bundle/human_query_adjudication_001.py --port 8770`

该入口不生成论文正文，不添加模型调用或真实车辆到达次数。goal_complete=false。
