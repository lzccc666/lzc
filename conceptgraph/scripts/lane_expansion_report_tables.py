"""Chinese tables with actual denominators for the final evidence report."""

from __future__ import annotations

from collections import Counter


def rate(measure):
    if not measure or measure.get('value') is None:
        return '不适用（分母0）'
    return f"{measure['value']:.3f}（n={measure['denominator']}）"


def report_tables(navigation, retrieval, ablation, qwen, stats, compatibility):
    roles = {'legacy_regression': '原场景回归', 'development': '新增开发', 'sealed_test': '固定独立测试'}
    lines = ['', '## 实际指标与分母', '', '以下表格在矩阵完整前均为进度快照。固定测试集曾在V1发布后被访问；V2安全修订及确认/拒绝分母修订分别在method_protocol_v2.json和reporting_protocol_v2.json披露。没有按测试效果选择场景、查询、提示或模型超参数。', '', '路径成功表示一条预选查询回放中至少产生一条接受路径，不等于目标选择正确、到达目标或车辆完成控制。目标确认只对有匹配目标的查询计分；无匹配查询单列拒绝率。所有关联遗漏正查询均保留为失败。', '', '### 全查询对象检索', '', '| 集合 | 条件 | scene数 | 查询数 | Top-1 | MRR | 候选召回 | 正目标确认 | 无匹配正确拒绝 |', '|---|---|---:|---:|---|---|---|---|---|']
    for group in retrieval.get('summaries', []):
        metric = group['overall_query_means']
        values = [rate(metric.get(k)) for k in ('top1', 'mrr', 'candidate_recall', 'target_confirmation_correct', 'correct_rejection')]
        lines.append(f"| {roles[group['role']]} | {group['method']} | {group['scene_count']} | {group['query_condition_count']} | "+' | '.join(values)+' |')
    test_groups = [group for group in retrieval.get('summaries', []) if group['role'] == 'sealed_test']
    if retrieval.get('all_complete') and len(test_groups) == 6:
        identical = all(len({group['overall_query_means'][metric]['value'] for group in test_groups}) == 1 for metric in ('top1', 'mrr', 'candidate_recall'))
        if identical:
            metric = test_groups[0]['overall_query_means']
            lines += ['', f"固定测试集的六种条件在全查询检索上得到相同结果：Top-1 {rate(metric['top1'])}，MRR {rate(metric['mrr'])}，候选召回 {rate(metric['candidate_recall'])}。本基准未观察到Qwen的全查询检索收益。低候选召回和关联覆盖限制了可用语义输入；合法schema或任务成功应用不能直接证明导航质量提高。这也不足以证明模型总体等效。", '']
    lines += ['', '检索表的无匹配正确拒绝指检索返回空列表，正目标确认指几何选择后的目标是否关联到可接受官方实例，两者与下表的在线目标锁定拒绝分别统计。本轮测试为自定义采集log隔离划分，并非nuScenes官方隐藏test；官方train/val身份另行保留。', '', '### 逐帧回放的查询与规划尝试', '', '| 集合 | 条件 | 查询数 | 查询路径成功率 | scene宏平均 | 成功规划 / 尝试 | 正目标确认 | 无匹配锁定拒绝 |', '|---|---|---:|---|---|---|---|---|']
    for group in navigation.get('summaries', []):
        members = [r for r in navigation['rows'] if r['role'] == group['role'] and r['method'] == group['method']]
        attempts = sum(r['planning_attempts'] for r in members)
        successes = sum(r['successful_plans'] for r in members)
        lines.append(f"| {roles[group['role']]} | {group['method']} | {len(members)} | {rate(group['overall_query_means']['query_success'])} | {rate(group['scene_macro_means']['query_success'])} | {successes}/{attempts} | {rate(group['overall_query_means']['target_confirmation_correct'])} | {rate(group['overall_query_means']['no_match_correct_rejection'])} |")
    lines += ['', '全基准有149条正目标查询、351条无匹配查询；测试集为28条正查询、72条无匹配查询。测试集10条首帧关系查询均无匹配，正目标关系检索准确率没有定义；不能据此得出正目标关系推理能力结论。另有2条关系正查询分别来自原场景与新增开发场景。', '', '规划观察可达率采用成功规划/实际尝试，与查询路径成功率分别报告；它不是独立可达性真值。整帧拒绝、等待和异常不一定构成新的规划尝试，逐帧失败事件另行汇总。', '', '### 四地图静态对照', '', '| 集合 | 地图约束 | 查询条件数 | 路径成功率 | scene宏平均 |', '|---|---|---:|---|---|']
    for group in ablation.get('summaries', []):
        lines.append(f"| {roles[group['role']]} | {group['method']} | {group['query_condition_count']} | {rate(group['overall_query_means']['query_success'])} | {rate(group['scene_macro_means']['query_success'])} |")
    lines += ['', '### 固定测试集配对差值及95%置信区间', '', '| 范围 | 指标 | 差值方向 | scene数 | 配对查询数 | 差值 | 95% CI |', '|---|---|---|---:|---:|---:|---|']
    for comparison in stats.get('comparisons', []):
        if comparison['role'] != 'sealed_test':
            continue
        selected = comparison['scope'] == 'online' and comparison['metric'] == 'query_success'
        selected = selected or comparison['scope'] == 'all_query_retrieval' and comparison['metric'] in {'top1', 'mrr'} and 'off_' in comparison['difference']
        if not selected:
            continue
        interval = comparison['ci95']
        ci = '不适用' if interval is None else f"[{interval[0]:.4f}, {interval[1]:.4f}]"
        delta = comparison['mean_difference']
        lines.append(f"| {comparison['scope']} | {comparison['metric']} | {comparison['difference']} | {comparison['scene_count']} | {comparison['paired_queries']} | {delta if delta is not None else '不适用'} | {ci} |")
    lines += ['', '固定seed=17，按scene聚类bootstrap 2000次；先配对同scene同query，再计算scene宏差值。只存在10个测试scene、4个采集log，另报log聚类敏感性。正目标指标只使用有定义分母的scene；大量无匹配和低对象关联覆盖会造成地板效应，退化为零的区间不能证明模型等效。全部分层结果、缺失对和其他指标的区间见statistical_report.json与stratified_report.json。', '', '## 真实Qwen调用与应用', '', '| 集合 / 条件 | 首次后端调用 | 首次合法成功响应 | 解析 / 属性 / 重排应用 | 复跑新增后端 | 复跑命中 | 首次回退 |', '|---|---:|---:|---|---:|---:|---:|']
    for key, total in qwen.get('totals', {}).items():
        apps = total['application_counts']
        counts = ' / '.join(str(apps.get(k+'_applied', 0)) for k in ('query_parse', 'object_attributes', 'candidate_rerank'))
        lines.append(f"| {key} | {total['backend_call_count']} | {total['successful_backend_response_count']} | {counts} | {total['cache_repeat_backend_call_count']} | {total['cache_repeat_hits']} | {total['fallback_count']} |")
    lines += ['', '表中是144项在线条件范围的调用；全查询静态检索及功能探针单独保存在对应报告中，不混入该表。合法schema、真实模型成功和在线实际应用是不同事件，缓存命中与规则fallback均不算首次模型成功。单候选重排可以产生合法建议和应用，但不代表排序改变或指标改善。', '']
    if compatibility:
        apps = compatibility['applications']
        lines += [f"base的既有stage4完整schema提示另在开发/原训练场景scene-0003上验证：解析/属性/重排实际应用分别为{apps.get('query_parse_applied', 0)}/{apps.get('object_attributes_applied', 0)}/{apps.get('candidate_rerank_applied', 0)}。该功能性补充{'通过' if compatibility['passed'] else '未通过'}，不并入固定测试对照，也不替换主矩阵finetune_v2提示下的schema失败。base的三类任务证据分属主矩阵的属性/重排与单独开发提示的解析；没有宣称同一提示协议全部成功。主矩阵反映finetune_v2固定提示下的表现，不能由此推断base一般无法解析指令或adapter具备普遍优势。", '']
    else:
        lines += ['base开发场景的既有完整schema提示兼容性验证仍待完成；主矩阵中schema失败继续作为回退保留。', '']
    lines += ['## 失败、等待与安全修订', '', '| 集合 / 条件 | 逐帧失败事件 | 阻塞路径立即失效帧数 |', '|---|---|---:|']
    for group in navigation.get('summaries', []):
        members = [r for r in navigation['rows'] if r['role'] == group['role'] and r['method'] == group['method']]
        failures = Counter()
        for row in members:
            failures.update(row.get('frame_failure_codes', {}))
        invalidated = sum(len(row.get('blocked_path_invalidated_frames', [])) for row in members)
        lines.append(f"| {group['role']}/{group['method']} | "+', '.join(f'{k}: {v}' for k, v in sorted(failures.items()))+f" | {invalidated} |")
    lines += ['', '开发scene-0038旧回放曾在去抖期间复用两帧阻塞路径，原失败报告保留；两条受影响查询的车道条件在新目录重跑。V2不放宽任何几何阈值：无法立即重规划时清空路径并明确返回PATH_BLOCKED_REPLAN_DEFERRED。所有选中的首次/复跑接受路径及每帧复用剩余段仍须对实际occupancy独立检查；未产生路径的帧不计作100%路径安全。', '', '光照、天气、停车/道路等分层由官方描述和真实数据可用性确定；未注明天气不能视作晴天。物体密度及类别/关系复杂度在选场后才做定量覆盖审计，不能宣称这些维度已作事前定量分层。4对相近路段及4对近似crop均披露，不能宣称跨地域或外观独立泛化。', '', '交互浏览器见[实际地图与回放](visualizations/index.html)，同时提供官方GT关联、真实障碍栅格、投影、方向与合法连接、拒绝平滑、首次/缓存、故障回退和场景配对区间。拒绝平滑的坐标由保存的原始A*路径、实际栅格及冻结参数确定性重建，并逐项核对保存的拒绝判定；不能把重建曲线称作原本存档的坐标。', '']
    return lines
