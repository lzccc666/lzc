"""统一只读入口：动态显示真实状态，不把进行中批次或目录存在当成完成。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


HTML = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>三维语义场景图与车辆导航 · 实验证据</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f7;color:#182e3b;font:15px/1.65 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1160px;margin:auto;padding:28px 22px}h1{font-size:26px;line-height:1.4;margin:0 0 10px}h2{font-size:19px;margin:0 0 12px}h3{font-size:16px;margin:0 0 8px}.muted{color:#566774;font-size:13px;overflow-wrap:anywhere}.status{display:inline-block;background:#f6ead6;border:1px solid #d0b68a;padding:5px 12px;border-radius:5px}section{margin:22px 0;padding:20px;background:white;border:1px solid #d2dce2;border-radius:7px;min-width:0}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.grid section{margin:0}a{color:#1a618c;overflow-wrap:anywhere}p{margin:9px 0}.links{display:flex;gap:14px;flex-wrap:wrap}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #dce4e8;padding:9px;overflow-wrap:anywhere}th:first-child{width:120px}.four{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}.four div{padding:12px;border:1px solid #d7e0e5;border-radius:5px;font-size:13px}.four strong{display:block;font-size:15px}.scroll{overflow-x:auto}button{font:inherit;padding:7px 14px;color:#1a4b65;background:#edf3f7;border:1px solid #b6c8d3;border-radius:4px;cursor:pointer}.error{color:#9b3b30}code{overflow-wrap:anywhere}footer{color:#5a6e79;font-size:13px;margin-top:22px}@media(max-width:760px){main{padding:18px 12px}h1{font-size:22px}.grid{grid-template-columns:1fr}.four{grid-template-columns:repeat(2,1fr)}section{padding:15px}td,th{padding:7px}}
#main-outcomes td{white-space:nowrap;overflow-wrap:normal}#main-outcomes th:first-child{width:auto}#main-outcomes tbody th{white-space:nowrap;overflow-wrap:normal}</style><main>
<h1>面向智能车辆的三维语义场景图构建与导航方法研究</h1>
<div id="goal" class="status">正在读取真实验收状态</div><p id="updated" class="muted"></p>
<p>当前入口汇集已完成开发实验、真实日志回放、后续队列与人工核验。论文正文与框架按用户安排暂缓。新封存测试的开放和完成状态以验收文件为准。</p>
<div class="four"><div><strong>生成路线</strong>规划器给出目标路径。</div><div><strong>选对对象</strong>独立关联到可接受实例。</div><div><strong>正确锁定</strong>持续观测确认正确身份。</div><div><strong>实际到达</strong>执行后满足身份、区域、姿态、速度与保持时间。</div></div>
<div class="grid"><section><h2>实时保存进度</h2><p id="main-progress">读取中</p><p id="main-last" class="muted"></p><p>50个冻结任务 × 3种子 × 规则/原始Qwen/新Adapter，共450次。结果表来自完整冻结记录；当前追加进度只作运行记录查看。</p><div class="links"><a href="/interactive_simulation/realtime_three_model_full_001/release.json">运行协议与源码指纹</a><a href="/interactive_simulation/realtime_three_model_full_001/trial_records.jsonl">全部已尝试记录</a><a href="/statistics/realtime_three_model_full_001/report.json">完整450统计</a></div><p class="muted">三种缺一语义条件新增450次；源集合兼容审核后与300份共享记录组成750条件。等待和复用不增加独立任务数。</p></section>
<section><h2>真实人工核验与自然表达采集</h2><p id="human">读取中</p><p id="collection">读取中</p><p>439条原始自然指令及开发证据可提交审核。自动证据、官方标签和真实人工记录分别保存；unknown不补写成肯定标签。</p><div class="links"><a href="/annotation/natural_command_review_001/index.html">审核原始自然指令</a><a href="/annotation/human_natural_collection_001/index.html">采集新的自然表达</a><a href="/visualizations/causal_development_review_002/index.html">打开过去LiDAR开发核验页</a><a href="/visualizations/review.html">旧开发核验页（旧深度输入）</a></div><p class="muted">审核会写入单独的追加记录。当前同意使用页面不等于已产生审核结论。</p></section></div>
<section><h2>实验交付与复现入口</h2><div id="paper-links" class="links"></div><p id="paper-status" class="muted"></p><div class="links"><a href="/visualizations/realtime_main450_preview_003/index.html">完整450感知与车辆执行查看器</a><a href="/visualizations/navigation_demo_004/index.html">历史163条开发记录</a><a href="/实验完成度与交付清单_012.md">第12版实验清单</a><a href="/主矩阵450_回放与复现交接_002.md">主450复现交接</a><a href="/reproducible_commands.md">复现及续跑命令</a><a href="/experiment_report.md">实验记录</a><a href="/remaining_issues.md">剩余必需工作</a><a href="/validation_report.json">逐项验收证据</a><a href="/baseline_verification.json">历史产物保护核验</a></div><p class="muted">查看器中的回放速度不是实时性能；主450与历史163条开发记录分开，重复复测不增加独立样本。当前390源码准备，GitHub同步待最终实验核验；旧v14包不含主450与新查看器增量。</p></section>
<section><h2>历史离线开发消融：旧未来LiDAR输入</h2><p>以下保留旧诊断口径；不能与新的过去LiDAR成绩混用，也不获得因果在线感知信用。</p><div class="scroll"><table><thead><tr><th>实验</th><th>真实记录与限制</th><th>证据</th></tr></thead><tbody>
<tr><td>六种地图约束</td><td>38场景、212查询、1272次规划；176个不同几何查询。正确对象参考点由GT给定；完整约束仍有3条独立连接审计失败，静态矩形姿态全部未通过，未执行控制。</td><td><a href="/statistics/map_constraints_dev_full_001/report.json">统计</a><br><a href="/thesis/figures_constraints_002/图6-6.pdf">矢量图</a></td></tr>
<tr><td>动态维护与过期</td><td>114条件记录，76次新重建、38份同版基线复用；减动态维护召回下降4.93个百分点。运动unknown保留，未证明车辆到达收益。</td><td><a href="/statistics/mapping_modules_dev_full_001/report.json">总体统计</a><br><a href="/statistics/mapping_modules_motion_dev_full_001/report.json">官方动静态组</a><br><a href="/thesis/figures_constraints_002/图6-7.pdf">矢量图</a></td></tr>
<tr><td>首达、联合与接近点</td><td id="selection-progress">全量结果读取中；进行中不计完成。</td><td><a href="/ablations/goal_selection_dev_full_001/release.json">冻结协议</a><br><a href="/ablations/goal_selection_dev_full_001/report.json">全量统计（完成后）</a></td></tr>
<tr><td>道路探索分组</td><td>1115开发查询按当前位姿与HD地图归组；弯道正查询仅5条、2log，不支持道路泛化结论。</td><td><a href="/performance/road_groups_dev_001/report.json">分组统计</a></td></tr>
<tr><td>引用版本核验</td><td>被引PDF优先于存在冲突的注册作者顺序。连续85个引用不表示全部正文主张已核验，失败下载与被拒中间版本保留。</td><td><a href="/thesis/citation_corrections_007.json">条目核验</a><br><a href="/thesis/claim_citation_review_008.json">正文范围核验</a></td></tr>
</tbody></table></div></section>
<section><h2>后续实验与当前证据</h2><p id="minus-progress">缺一语义记录读取中</p><p id="fault-progress">GPU故障结果读取中</p><p id="performance-progress">独占性能结果读取中</p><p id="refresh-time" class="muted"></p><div class="links"><a href="/interactive_simulation/realtime_semantic_minus_full_001/release.json">缺一冻结协议</a><a href="/ablations/semantic_navigation_chain_003/preflight.json">源集合兼容预检</a><a href="/task_protocol/semantic_source_compatibility_003.md">版本化汇总补充协议</a><a href="/faults/realtime_fault_after_semantics_001/release.json">故障队列冻结协议</a><a href="/performance/runtime_profile_after_faults_001/release.json">独占性能队列协议</a></div><p class="muted">保存记录不证明进程目前存活；完成以实际全量结果及独立审核为准。CPU已完成试跑单列，不能替代GPU正式实验。</p></section><section><h2>逐项验收状态</h2><button id="refresh">刷新实际状态</button><p id="error" class="error"></p><div class="scroll"><table><thead><tr><th>项目</th><th>当前状态与未完成条件</th></tr></thead><tbody id="statuses"></tbody></table></div></section>
<section><h2>启动新的感知导航交互运行</h2><p>可选择冻结开发任务，实际运行CPU感知、建图、规划与控制；支持单步、连续运行、暂停与制动停车。新运行保存独立日志，不并入主450成绩。</p><div class="links"><a id="live-navigation" href="http://127.0.0.1:8769/" target="_blank">打开交互控制台（8769端口）</a><a href="/交互仿真_操作与复现_001.md">操作与复现说明</a><a href="/visualizations/live_navigation_browser_qa_001/report.json">桌面手机实际操作验证</a><a href="/vehicle_execution/live_navigation_independent_audit_001/report.json">独立控制与连续几何核验</a></div><p class="muted">远程使用需另行转发8769。CPU固定仿真时钟与共享硬件耗时不代表实时性能；暂停保留速度，停车以实际零速为准。自动浏览器操作不是人工标注。</p></section>
<footer>本页只读；交互控制台的开始与控制按钮会产生新的开发运行。数据直接来自保存结果。远程访问需转发当前核验服务8768端口。缺失结果显示未完成，不按零值或成功补齐。</footer>
</main><script>
const $=id=>document.getElementById(id);async function json(url){const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw Error('尚无完整文件：'+url);return r.json()}
function link(parent,label,path){const a=document.createElement('a');a.textContent=label;a.href=path;parent.appendChild(a)}
const liveNavigationURL=new URL('/',location.href);liveNavigationURL.port='8769';document.getElementById('live-navigation').href=liveNavigationURL.href;
async function records(url){const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw Error('追加记录未就绪');const s=await r.text(),lines=s.split('\\n');if(!s.endsWith('\\n'))lines.pop();return lines.filter(x=>x.trim()).map(x=>JSON.parse(x))}
async function optional(url){const r=await fetch(url,{cache:'no-store'});if(r.status===404)return null;if(!r.ok)throw Error('请求失败 '+r.status);return r.json()}
async function refresh(){
 $('error').textContent='';$('refresh').disabled=true;
 try{
 const results=await Promise.allSettled([json('/validation_report.json'),json('/api/review_status'),records('/interactive_simulation/realtime_three_model_full_001/trial_records.jsonl'),json('/ablations/goal_selection_dev_full_001/summary.json'),json('/api/natural_collection_status'),records('/interactive_simulation/realtime_semantic_minus_full_001/trial_records.jsonl'),optional('/faults/realtime_fault_after_semantics_001/complete.json'),optional('/performance/runtime_profile_after_faults_001/complete.json'),optional('/ablations/semantic_navigation_chain_003/complete.json')]);
 const [validation,human,trials,selection,collection,minus,fault,performance,semanticComplete]=results;
 if(validation.status==='fulfilled'){const v=validation.value;$('goal').textContent=v.goal_complete?'全部Goal已验收':'Goal仍在进行 · 必需条件未全部满足';$('updated').textContent='验收状态更新时间：'+v.updated_utc+'；新封存测试：'+(v.new_test_opened?'以对应release为准':'未开启');$('statuses').replaceChildren();for(const [name,s] of Object.entries(v.statuses)){const tr=document.createElement('tr'),a=document.createElement('th'),b=document.createElement('td');a.textContent=name;const strong=document.createElement('strong');strong.textContent=s.status;b.appendChild(strong);const p=document.createElement('p');p.textContent=s.remaining;b.appendChild(p);for(const f of s.evidence){link(b,'证据','/'+f);b.appendChild(document.createTextNode(' '))}tr.append(a,b);$('statuses').appendChild(tr)}$('paper-links').replaceChildren();const paper=v.statuses['论文'];$('paper-status').textContent=v.manuscript_work_deferred_by_user?'当前暂缓论文正文和框架。先完成实验与复现，历史论文材料只作后续参考。':(paper?.remaining??'论文状态未知');if(!v.manuscript_work_deferred_by_user&&paper){const manifest=paper.evidence.find(x=>x.endsWith('/delivery_manifest.json'));if(manifest){const prefix='/'+manifest.slice(0,manifest.lastIndexOf('/'));for(const [name,suffix] of [['可编辑Word','docx'],['PDF阶段稿','pdf'],['完整Markdown','md']])link($('paper-links'),name,prefix+'/完整论文阶段整合稿.'+suffix)}}}else{$('goal').textContent='无法读取验收文件，不能推断完成';$('updated').textContent='验收状态未知';$('statuses').replaceChildren();$('paper-links').replaceChildren();$('paper-status').textContent='论文安排未知，未推断完成';$('error').textContent=String(validation.reason)}
 if(human.status==='fulfilled'){$('human').textContent='真实审核提交：开发 '+human.value.development_submissions+'；原始自然指令 '+human.value.natural_submissions+'。提交数量不等于可靠标签覆盖。'}else $('human').textContent='真实审核提交数量未知。';
 if(collection.status==='fulfilled'){const c=collection.value;$('collection').textContent='新自然表达 '+c.human_authored_submissions+' 条；合格新测试指令 '+c.qualified_test_queries+' 条。'+c.scenes+'场景采集目标至少'+c.full_campaign_minimum+'条，每场景至少'+c.minimum_per_scene+'条，仍需独立标签审核。'}else $('collection').textContent='新自然表达及合格测试指令数量未知。';
 if(trials.status==='fulfilled'){const rows=trials.value,complete=rows.filter(r=>r.result!==null).length;$('main-progress').textContent='已保存尝试记录 '+rows.length+' / 450；完整结果 '+complete+'，缺完整结果 '+(rows.length-complete)+'。'+(rows.length===450&&document.querySelector('#completed-main')?'全450开发条件已完成，失败保留，结果见独立核验表。':'不能以当前记录推断全量完成。');const r=rows.at(-1);$('main-last').textContent=r?'最近保存：'+r.variant+' · '+r.task+' · seed'+r.seed+' · '+r.status:'尚无保存试验'}else{$('main-progress').textContent='当前记录数量未知，不能推断完成。';$('main-last').textContent='';}
 $('selection-progress').textContent=selection.status==='fulfilled'?'已完成 '+selection.value.query_cases+' 查询、'+selection.value.records+' 条件记录。联合与首达共享实际规划池；去保护条件未执行；不计实际到达。':'全量结果未知，不能推断完成。';
 if(minus.status==='fulfilled'){const rows=minus.value;$('minus-progress').textContent='缺一语义新增记录 '+rows.length+' / 450；其中完整结果 '+rows.filter(r=>r.result!==null).length+'。'+(semanticComplete.status==='fulfilled'&&semanticComplete.value?.combined_trial_records===750?'750行组合汇总已生成，300主记录复用，独立审核见完成证据。':'750行完整汇总及验收尚未就绪；300主记录的复用不增加独立任务数。');}else $('minus-progress').textContent='缺一语义记录数量未知，不能推断完成。';
 $('fault-progress').textContent=fault.status==='fulfilled'?(fault.value?.full_trials===180&&fault.value?.smoke_trials===6?'正式GPU故障：6试跑及180全量结果已生成，独立安全与恢复审核仍须对应证据。':'正式GPU故障完成文件未就绪；CPU单任务六条件试跑单列，不替代正式实验。'):'GPU故障结果未知，不能推断完成。';
 $('performance-progress').textContent=performance.status==='fulfilled'?(performance.value?.actual_Qwen_calls===216&&performance.value?.actual_GPU_frontend_calls===12?'独占性能：12前端、216无缓存Qwen及216缓存请求已生成；不等于完整导航实时吞吐。':'独占GPU性能完成文件未就绪；已有CPU试跑单列。'):'独占GPU性能结果未知，不能推断完成。';
 $('refresh-time').textContent='页面读取时间：'+new Date().toISOString()+'；不依据记录数量判断进程存活。';
 }finally{$('refresh').disabled=false;}
}
$('refresh').onclick=()=>refresh().catch(e=>$('error').textContent=String(e));refresh().catch(e=>$('error').textContent=String(e));
</script>'''


def causal_section():
    """只从已完成报告读取过去LiDAR结果，固定生成时的证据指纹。"""
    paths = [OUTPUT/'statistics/causal_mapping_full_001/report.json',
        OUTPUT/'statistics/causal_retrieval_full_001/report.json',
        OUTPUT/'data_manifest/development_causal_inputs_full_001/audit_001/report.json',
        OUTPUT/'mapping_evaluation/causal_relations_full_001/summary.json']
    if not all(path.exists() for path in paths):
        return '', None
    mapping = json.loads(paths[0].read_text())
    retrieval = json.loads(paths[1].read_text())
    relations = json.loads(paths[3].read_text())
    methods = mapping['methods']
    order = ['category_nearest_current_detection', 'clip_current_crop',
        'vlmaps_style_sparse_clip_grid', 'existing_object_joint_rule']
    # 方法标识与源报告一致；缺失时停止生成，不能用预期数字替代结果。
    actual_order = []
    for prefix in order:
        found = [name for name in retrieval['methods'] if name == prefix]
        if len(found) != 1:
            raise ValueError('四方法检索标识不完整或重复：'+prefix)
        actual_order.append(found[0])
    correct = '/'.join(str(retrieval['methods'][name]['positive_selection']['numerator']) for name in actual_order)
    false = '/'.join(str(retrieval['methods'][name]['no_match_false_selection']['numerator']) for name in actual_order)
    def relation_f1(name):
        total = {key: sum(scene['methods'][name][key] for scene in relations['scenes'])
            for key in ['true_positive', 'predicted', 'truth']}
        denominator = total['predicted']+total['truth']
        return f"{2*total['true_positive']/denominator:.5f}" if denominator else 'unknown'
    def metrics(name):
        values = methods[name]['all']
        return (f"召回{100*values['recall']['value']:.2f}%，匹配精确率"
            f"{100*values['matched_prediction_precision']['value']:.2f}%，中心误差"
            f"{values['center_error_m_mean']['value']:.3f}m")
    note = ('<section><h2>当前开发主结果：相机之前的实测LiDAR</h2>'
        '<p>38场景、11采集log、1508帧；1470帧有过去扫描，38帧缺测保留。'
        '独立重投影逐像素0差异，无未来深度。官方sample GT仍较相机晚33–42ms，评价时间近似继续披露。</p>'
        '<div class="scroll"><table><thead><tr><th>条件</th><th>已完成结果</th></tr></thead><tbody>'
        f'<tr><td>原始融合</td><td>{metrics("original_fusion")}</td></tr>'
        f'<tr><td>深度质量</td><td>{metrics("one_to_one_depth_quality")}</td></tr>'
        '<tr><td>范围</td><td>5723次跨帧GT观测全部保留，缺测首帧130次GT观测均0匹配；身份变化率差区间跨0。'
        '开发定位改善不等于正确锁定或实际到达。</td></tr></tbody></table></div>'
        f'<p>{retrieval["paired_queries"]}条查询。类别最近/CLIP/简化VLMaps式网格/联合规则选对{correct}，'
        f'负误选{false}；关系结构化F1 {relation_f1("structured")}，直接几何{relation_f1("direct_geometry")}，未知端点保留。</p>'
        '<div class="links"><a href="/statistics/causal_mapping_full_001/report.md">七条件建图与区间</a>'
        '<a href="/statistics/causal_retrieval_full_001/report.json">四方法完整检索</a>'
        '<a href="/mapping_evaluation/causal_relations_full_001/summary.json">关系与未知端点</a>'
        '<a href="/data_manifest/development_causal_inputs_full_001/audit_001/report.json">独立输入审计</a></div>')
    sensitivity = OUTPUT/'performance/causal_selection_parameters_full_001/report.json'
    selection = OUTPUT/'ablations/causal_selection_ablation_full_001/report.json'
    roads = OUTPUT/'performance/causal_road_groups_full_001/report.json'
    if all(path.exists() for path in [sensitivity, selection, roads]):
        paths += [sensitivity, selection, roads]
        parameters = json.loads(sensitivity.read_text())
        ablations = json.loads(selection.read_text())
        groups = json.loads(roads.read_text())
        conditions = parameters['conditions']
        observations = conditions['minimum_observations_5']['positive_selection']
        rejected = conditions['rejection_score_0.7']
        depth = conditions['minimum_depth_support_fraction_0.6']
        without = ablations['conditions']['without_relation_edges']
        curve = groups['road_groups'].get('curved_lane', {}).get('existing_object_joint_rule', {})
        note += ('<h3>同输入参数、关系边与道路分组</h3>'
            f'<p>{len(conditions)}参数×{parameters["paired_queries"]}查询='
            f'{len(conditions)*parameters["paired_queries"]}条，'
            f'{len(ablations["conditions"])}选择条件{len(ablations["conditions"])*ablations["paired_queries"]}条；'
            f'基线逐查询复现。确认5次选对{observations["count"]}/{observations["denominator"]}；'
            f'拒绝分0.7选对{rejected["positive_selection"]["count"]}、误选{rejected["no_match_false_selection"]["count"]}。'
            f'深度支持比例0.6选对{depth["positive_selection"]["count"]}/{depth["positive_selection"]["denominator"]}、'
            f'负误选{depth["no_match_false_selection"]["count"]}/{depth["no_match_false_selection"]["denominator"]}，差异区间含0。'
            f'去结构化边后{without["positive_selected_correct"]}/{without["positive_queries"]}，'
            '仍保留几何回退，其差异区间下界0，不能声称关系边改善。'
            f'弯道仅{curve.get("positive_queries", "unknown")}条正查询、'
            f'{curve.get("positive_selection", {}).get("contributing_logs", "unknown")}log，不支持道路泛化结论。</p><div class="links">'
            '<a href="/performance/causal_selection_parameters_full_001/report.json">全部参数与配对区间</a>'
            '<a href="/ablations/causal_selection_ablation_full_001/report.json">关系边/证据消融</a>'
            '<a href="/performance/causal_road_groups_full_001/report.json">道路分组与样本量</a>'
            '<a href="/source_bundle/causal_sensitivity_audit_001/report.json">全查询完整性复核</a></div>')
    fault = OUTPUT/'statistics/causal_fault_full_001/report.json'
    fault_audit = OUTPUT/'faults/causal_fault_full_001/audit_001/report.json'
    if fault.exists() and fault_audit.exists():
        values = json.loads(fault.read_text())
        audit = json.loads(fault_audit.read_text())
        if not audit['passed'] or audit['statistics_sha256'] != digest(fault):
            raise ValueError('故障统计与独立重算不一致')
        paths += [fault, fault_audit]
        def fault_delta(condition, metric, factor=1):
            row = next(row for row in values['paired_differences']
                if row['condition'] == condition and row['metric'] == metric)
            return (f'{factor*row["difference"]:+.3f} '
                f'[{factor*row["ci95"][0]:+.3f}, {factor*row["ci95"][1]:+.3f}]')
        note += ('<h3>过去LiDAR录制故障：完整684次重建</h3>'
            f'<p>{audit["actual_frame_condition_records"]}条实际帧条件记录全部独立重算，'
            '38场景、11log、三种子不增加独立场景数。'
            f'LiDAR丢失80%时召回差{fault_delta("lidar_dropout_80pct", "recall", 100)}个百分点；'
            f'位姿扰动中心误差差{fault_delta("pose_noise_025m_2deg", "center_error_m_mean")}m。'
            '括号为配对95%区间；合成缺失/白噪声未经真实传感器标定，不能代替闭环故障恢复。</p>'
            '<div class="links"><a href="/statistics/causal_fault_full_001/report.json">六条件统计与区间</a>'
            '<a href="/faults/causal_fault_full_001/audit_001/report.md">可读结果表</a>'
            '<a href="/faults/causal_fault_full_001/audit_001/report.json">全帧独立重算</a></div>')
    camera = OUTPUT/'mapping_evaluation/camera_time_gt_dev_full_001/report.json'
    camera_audit = camera.parent/'audit_001/report.json'
    if camera.exists() and camera_audit.exists():
        values = json.loads(camera.read_text())
        audit = json.loads(camera_audit.read_text())
        if not audit['passed'] or audit['source_report_sha256'] != digest(camera):
            raise ValueError('相机时间评价与独立重算不一致')
        paths += [camera, camera_audit]
        comparison = next(row for row in values['method_contrasts']
            if row['condition'] == 'camera_center_interpolation_proxy'
            and row['method'] == 'one_to_one_depth_quality'
            and row['baseline'] == 'original_fusion' and row['metric'] == 'recall')
        note += ('<h3>相机时间中心插值：固定分母的补充敏感性</h3>'
            f'<p>仍为{values["fixed_GT_observations"]}个GT观测、全部七种原预测。'
            f'{values["alignment_states"]["interpolated"]}个中心在相邻官方标注间线性插值，'
            f'{values["alignment_states"]["fallback"]}个无前驱保留原中心。'
            '只改变评价中心，不改输入、方法或主表；尺寸、朝向与可见性仍有原时间近似。'
            f'新参考下深度质量相对原融合召回差{100*comparison["value"]:+.2f}pp，'
            f'11log配对95%区间[{100*comparison["ci95"][0]:.2f}, '
            f'{100*comparison["ci95"][1]:.2f}]pp。'
            '深度质量自身因评价中心改变的召回/误差差异区间均跨0。'
            '线性轨迹假设未获人工验证，不能称精确相机时间真值或新增到达证据。</p>'
            '<div class="links"><a href="/task_protocol/camera_time_gt_sensitivity_001.md">补充评价协议</a>'
            '<a href="/mapping_evaluation/camera_time_gt_dev_full_001/report.json">七方法全部区间与回退</a>'
            '<a href="/mapping_evaluation/camera_time_gt_dev_full_001/audit_001/report.md">独立重算与可读结果</a></div>')
    maps = OUTPUT/'statistics/causal_maps_full_001/report.json'
    if maps.exists():
        values = json.loads(maps.read_text())
        full = values['conditions']['lane_direction_connection']
        paths.append(maps)
        rectangle = full['static_rectangle_legal_among_routes']
        note += ('<h3>六地图约束：完整过去LiDAR诊断</h3>'
            f'<p>{values["queries"]}查询、{values["cases"]}条件、'
            f'{values["unique_geometric_queries"]}个不同几何查询。完整约束生成'
            f'{full["route_generated"]["count"]}条路线，'
            f'{full["full_road_legal_all_queries"]["count"]}条同时通过车道与可行驶区域检查；'
            f'仅{rectangle["count"]}/{rectangle["denominator"]}路线通过静态矩形姿态检查。'
            '正确对象参考点由隔离GT给定，没有控制执行、正确锁定或实际到达信用。</p>'
            '<div class="links"><a href="/statistics/causal_maps_full_001/report.json">六条件全部分母与配对区间</a></div>')
        map_audit = OUTPUT/'ablations/causal_maps_full_001/audit_002/report.json'
        if map_audit.exists():
            audit = json.loads(map_audit.read_text())
            if not audit['passed'] or audit['source_statistics_sha256'] != digest(maps):
                raise ValueError('地图统计与复核版本不一致')
            paths.append(map_audit)
            note += ('<p><a href="/ablations/causal_maps_full_001/audit_002/report.md">'
                '完整记录、显式矩形几何与区间复算</a>；沿用既有独立车道审计器，'
                '未新增第二套车道审计实现。逐姿态数值边界差异及首次失败保留，'
                '未把边界违规改判为通过。</p>')
    for title, report_path in [
            ('动态维护/TTL完整建图消融', 'statistics/causal_modules_full_001/report.json'),
            ('关联门限完整敏感性', 'statistics/causal_mapping_parameters_full_001/report.json')]:
        report = OUTPUT/report_path
        if report.exists():
            paths.append(report)
            note += (f'<p><a href="/{html.escape(report_path)}">{title}</a>：'
                '完整报告已生成，详见同版记录、分母、负结果与区间；不计车辆执行到达。</p>')
    note += '<p class="muted">关联门限、动态模块及地图消融按冻结队列推进；未完成的不按成功补齐。新封存测试仍未开启，本页无车辆到达信用。</p></section>'
    return note, {str(path.relative_to(OUTPUT)): digest(path) for path in paths}


def completed_section():
    """仅整合有完整同版来源和独立审计的开发结果，保留所有失败及适用范围。"""
    main = OUTPUT/'interactive_simulation/realtime_three_model_full_001'
    stats = OUTPUT/'statistics/realtime_three_model_full_001/report.json'
    if not stats.exists():
        return '', None
    paths = [main/'release.json', main/'summary.json', main/'trial_records.jsonl', stats,
        OUTPUT/'statistics/main_matrix_evidence_check_001/report.json',
        OUTPUT/'vehicle_execution/main_goal_hold_evidence_check_002/report.json',
        OUTPUT/'visualizations/realtime_main450_preview_003/manifest.json',
        OUTPUT/'visualizations/realtime_main450_browser_qa_001/report.json']
    summary, report = json.loads(paths[1].read_text()), json.loads(stats.read_text())
    audit = json.loads(paths[4].read_text())
    hold = json.loads(paths[5].read_text())
    view = json.loads(paths[6].read_text())
    browser = json.loads(paths[7].read_text())
    assert summary['attempted'] == summary['expected'] == report['trials'] == audit['trials'] == 450
    assert digest(paths[0]) == report['source_release_sha256']
    assert digest(paths[1]) == report['source_summary_sha256']
    assert digest(paths[2]) == summary['trial_records_sha256'] == report['source_records_sha256']
    assert audit['passed'] and hold['passed'] and browser['passed']
    assert browser['viewer_manifest_sha256'] == digest(paths[6]) and len(view['cases']) == 450
    fields = ['route_generated', 'selected_object_evaluated_correct',
              'stable_correct_lock', 'actual_full_perception_arrival']
    rows = []
    for method, label in [('rule', '规则'), ('base_full', '原始Qwen'), ('new_full', '新Adapter')]:
        values = report['methods'][method]['by_kind']['reachable_positive']['four_outcomes']
        assert all(values[field]['denominator'] == 90 for field in fields)
        cells = ''.join(f'<td data-field="{field}">{int(values[field]["numerator"])}/90</td>' for field in fields)
        rows.append(f'<tr data-method="{method}"><th>{label}</th>{cells}</tr>')
    records = [json.loads(line) for line in paths[2].read_text().splitlines()]
    complete_results = sum(row['result'] is not None for row in records)
    assert complete_results == 449 and hold['counts']['reported_arrivals_with_conservative_hold'] == 93
    note = ('<section id="completed-main"><h2>完整450开发结果与真实日志回放</h2>'
        '<p>450条件全部保留：449完整结果、1次进程超时。10配置共享3道路家族，50任务各3种子；'
        '每模型90可达正条件，另有30无匹配和30不可接近条件。以下为生成时已核验快照，属于合成开发任务。</p>'
        '<div class="scroll"><table id="main-outcomes"><thead><tr><th>方法</th><th>生成路线</th>'
        '<th>曾选对对象</th><th>曾正确锁定</th><th>实际到达</th></tr></thead><tbody>'
        +''.join(rows)+'</tbody></table></div>'
        '<p>新Adapter下游没有提升，负结果完整保留。93原报告到达获分版本车身、方向及2秒保持核验；'
        '54例初态为显式种子重建，严格缺零时刻观测未知仍保留。查看记录不增加执行或到达。</p>'
        '<div class="links"><a href="/visualizations/realtime_main450_preview_003/index.html">打开全部450回放</a>'
        '<a href="/statistics/realtime_three_model_full_001/trials.csv">逐条件CSV</a>'
        '<a href="/statistics/main_matrix_evidence_check_001/report.json">450分母与统计复算</a>'
        '<a href="/vehicle_execution/main_goal_hold_evidence_check_002/report.json">分版本独立到达核验</a>'
        '<a href="/visualizations/realtime_main450_browser_qa_001/report.json">桌面与手机浏览器验收</a></div>'
        f'<p class="muted">{view["perception_records"]:,}感知帧与{view["control_records"]:,}保存控制记录，'
        '含失败前缀；按完成墙钟对齐过去感知。代表条件浏览检查不等于全部逐帧人工审核。</p></section>')
    kitti = OUTPUT/'statistics/kitti_object_cpu_full_001/report.json'
    if kitti.exists():
        values = json.loads(kitti.read_text())
        # 完整统计已绑定真实前端完成清单；按已有实际位置校验，不推断不存在的路径。
        completion = OUTPUT/'data_manifest/kitti_object_uniform64_001/complete.json'
        assert digest(completion) == values['source_complete_sha256']
        assert not values['gt_used_for_inference'] and not values['complete_navigation_test']
        paths += [kitti, completion]
        original, robust = [values['metrics'][key] for key in ['mask_points_mean', 'mask_points_robust_mean']]
        common = values['common_supported_pairs']
        note += ('<section id="kitti"><h2>KITTI真实车载对象诊断</h2>'
            f'<p>{values["frame_count"]}真实帧、{values["raw_drive_count"]}个drive，'
            f'{original["truth"]}个car/truck/pedestrian对象。均值深度匹配'
            f'{original["matches_3d"]}/{original["truth"]}（{100*original["recall_3d"]:.2f}%），'
            f'稳健均值{robust["matches_3d"]}/{robust["truth"]}（{100*robust["recall_3d"]:.2f}%）；'
            f'共同有深度的{common["count"]}对象中心误差'
            f'{common["mean_error_m"]["mask_points_mean"]:.3f}→'
            f'{common["mean_error_m"]["mask_points_robust_mean"]:.3f}m。</p>'
            '<p>CPU跨数据集开发诊断，采用同类别3m中心匹配，非官方KITTI AP、连续跟踪或导航。'
            'truck误差未改善，Cyclist代理0/20；预训练重合未知，不能称未见数据泛化。</p><div class="links"><a href="/visualizations/kitti_object_cpu_full_002/index.html">原图、检测与深度查看</a>'
            '<a href="/statistics/kitti_object_cpu_full_001/report.json">完整分母、分组及区间</a></div></section>')
    tracking = OUTPUT/'statistics/kitti_tracking_full_001/report.json'
    tracking_audit = tracking.parent/'independent_audit_002/report.json'
    tracking_qa = OUTPUT/'visualizations/kitti_tracking_browser_qa_001/report.json'
    if tracking.exists() and tracking_audit.exists() and tracking_qa.exists():
        values = json.loads(tracking.read_text())
        assert json.loads(tracking_audit.read_text())['passed'] and json.loads(tracking_qa.read_text())['passed']
        paths += [tracking, tracking_audit, tracking_qa]
        summary = []
        for method, name in [('single_frame', '单帧'), ('original_fusion', '原融合'),
                             ('one_to_one_depth_robust_motion', '改进融合')]:
            obj = values['object']['methods'][method]
            identity_score = values['identity']['methods'][method]['all_gated']['metrics']['IDF1']['value']
            summary.append(f'{name}匹配{obj["counts"]["true_positive"]}/{obj["counts"]["truth"]}，IDF1 {identity_score:.4f}')
        note += ('<section id="kitti-tracking"><h2>KITTI连续车载外部开发诊断</h2>'
            f'<p>固定{values["sequence_count"]}个序列、{values["frame_count"]}帧；'
            + '；'.join(summary) + '。</p>'
            '<p>实际CPU前端、隔离建图、官方轨迹标签与固定TrackEval独立复核。'
            '名义0.1秒，非原始采集时间戳；1帧与旧Object64重合；三个窗口非独立采集log、非封存测试；全部74次主类别GT只在0005，不能宣称三场景召回泛化。'
            '未匹配预测不全称误检，不计导航到达、人工或GPU性能信用。</p>'
            '<div class="links"><a href="/visualizations/kitti_tracking_preview_001/index.html">90帧RGB、深度、三维对象与身份轨迹</a>'
            '<a href="/statistics/kitti_tracking_full_001/report.json">分母、分组与序列配对区间</a>'
            '<a href="/statistics/kitti_tracking_full_001/independent_audit_002/report.json">独立匹配与身份复核</a></div></section>')
    identity = OUTPUT/'statistics/temporal_identity_full_001/report.json'
    identity_audit = identity.parent/'audit_002/report.json'
    if identity.exists() and identity_audit.exists():
        values, audit = json.loads(identity.read_text()), json.loads(identity_audit.read_text())
        assert audit['passed'] and audit['source_report_sha256'] == digest(identity)
        paths += [identity, identity_audit]
        before = values['methods']['original_fusion']['all_gated']['metrics']['IDF1']['value']
        after = values['methods']['one_to_one_depth_quality']['all_gated']['metrics']['IDF1']['value']
        note += ('<section><h2>完整序列身份与真实地图控制诊断</h2>'
            f'<p>532个身份条件、{audit["checked"]["return_events"]}个返回事件、'
            f'{audit["checked"]["censored_tails"]}个尾段独立复核；深度方法IDF1 '
            f'{before:.4f}→{after:.4f}。返回保持及相邻身份变化的差异区间跨0，'
            '自动中心关联不是人工遮挡真值。</p><div class="links">'
            '<a href="/statistics/temporal_identity_full_001/report.md">身份统计与范围</a>'
            '<a href="/statistics/temporal_identity_full_001/audit_002/report.json">独立身份复核</a></div>')
        guided = OUTPUT/'vehicle_execution/HDmap_reference_guided_full_002/summary.json'
        guided_audit = OUTPUT/'vehicle_execution/HDmap_reference_guided_audit_002/report.json'
        if guided.exists() and guided_audit.exists():
            values, audit = json.loads(guided.read_text()), json.loads(guided_audit.read_text())
            assert audit['passed'] and values['attempted'] == values['expected'] == audit['trials'] == 228
            assert values['actual_semantic_arrival_credit'] == 0
            paths += [guided, guided_audit]
            baseline, proposed = [values['by_method'][key] for key in ['hybrid_astar', 'reference_guided']]
            note += (f'<p>同预算地图诊断：各{baseline["trials"]}条件，原Hybrid '
                f'{baseline["reference_pose_reached"]}、中心线引导{proposed["reference_pose_reached"]}参考位姿到达；'
                f'{audit["totals"]["control_intervals"]:,}实际控制区间独立通过。'
                '地图给定、人工后轴起点及静态预测障碍；开发后验改进，语义到达信用0，未替换主450版本。</p>'
                '<div class="links"><a href="/visualizations/HDmap_reference_guided_preview_003/index.html">完整228地图诊断回放</a>'
                '<a href="/vehicle_execution/HDmap_reference_guided_audit_002/report.json">整车、方向及连接复核</a></div>')
        note += '</section>'
    return note, {str(path.relative_to(OUTPUT)): digest(path) for path in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--region-run')
    parser.add_argument('--region-preview')
    parser.add_argument('--reference-run')
    parser.add_argument('--reference-preview')
    args = parser.parse_args()
    dest = OUTPUT/'visualizations'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    content = HTML
    completed_note, completed_evidence = completed_section()
    content = content.replace('<div class="grid">', completed_note+'<div class="grid">', 1)
    causal_note, causal_evidence = causal_section()
    content = content.replace('<section><h2>历史离线开发消融：旧未来LiDAR输入</h2>',
        causal_note+'<section><h2>历史离线开发消融：旧未来LiDAR输入</h2>')
    region_evidence = None
    if args.region_run and args.region_preview:
        region = OUTPUT/'annotation'/args.region_run
        summary = json.loads((region/'summary.json').read_text())
        audit = json.loads((region/'audit_001/report.json').read_text())
        preview = OUTPUT/'visualizations'/args.region_preview
        if (not audit['passed'] or audit['generator_summary_sha256'] != digest(region/'summary.json')
                or not (preview/'manifest.json').exists()):
            raise ValueError('静态区域入口需要本次结果的独立审计和实际预览')
        note = ('<section><h2>预留标签：独立静态终点区域</h2><p>'
            f"{summary['scenes']}场景、{summary['queries']}条构造指令；"
            f"{summary['positive_queries']}条正查询中"
            f"{summary['query_endpoint_status']['certified_static_regions']}条有保守静态证书，"
            f"{summary['query_endpoint_status']['unknown_no_certificate']}条无证书、保持未知。"
            f"{summary['certified_target_regions']}个区域通过独立地图/官方盒/车身包络核验。</p>"
            '<p>静态区域证书本身不证明从初始状态可达；后续条件控制参考须查看独立验收。没有测试方法调用、感知导航执行或人工记录。</p>'
            '<div class="links">'
            f'<a href="/visualizations/{html.escape(args.region_preview)}/index.html">RGB与区域证据查看器</a>'
            f'<a href="/annotation/{html.escape(args.region_run)}/summary.json">原始标签汇总</a>'
            f'<a href="/annotation/{html.escape(args.region_run)}/audit_001/report.json">独立区域核验</a>'
            '</div></section>')
        content = content.replace('<section><h2>逐项验收状态</h2>', note+'<section><h2>逐项验收状态</h2>')
        region_evidence = dict(summary_sha256=digest(region/'summary.json'),
            audit_sha256=digest(region/'audit_001/report.json'), preview_manifest_sha256=digest(preview/'manifest.json'))
    reference_evidence = None
    if bool(args.reference_run) != bool(args.reference_preview):
        raise ValueError('独立控制参考需要结果与预览成对提供')
    if args.reference_run:
        reference=OUTPUT/'annotation'/args.reference_run
        accepted=json.loads((reference/'acceptance_001.json').read_text())
        preview=OUTPUT/'visualizations'/args.reference_preview
        manifest=json.loads((preview/'manifest.json').read_text())
        if (not accepted['conditional_static_reference_certificates_accepted']
                or accepted['audit_sha256'] != digest(reference/'audit_001/report.json')
                or accepted['preview_manifest_sha256'] != digest(preview/'manifest.json')
                or manifest['reference_audit_sha256'] != accepted['audit_sha256']):
            raise ValueError('条件参考需要相同结果、独立审计与已核验预览')
        note=('<section><h2>预留标签：独立条件静态可达参考</h2>'
            f"<p>{accepted['queries']}条查询中，{accepted['query_status']['conditionally_reachable_static_from_rest']}条"
            f"有条件参考、{accepted['query_status']['unknown']}条可达性未知，覆盖{accepted['certified_scenes']}场景。"
            f"{accepted['accepted_witnesses']}条控制参考、{accepted['independently_checked_states']}个状态独立复算通过。</p>"
            '<p>已知后轴起点、静止及零转角起步、官方sample冻结障碍，90秒内进入独立区域并保持2秒。'
            '参考不计被评方法的路线、选对、锁定或实际到达；未知不判不可接近。</p><div class="links">'
            f'<a href="/visualizations/{html.escape(args.reference_preview)}/index.html">RGB、地图与控制参考滑杆</a>'
            f'<a href="/annotation/{html.escape(args.reference_run)}/acceptance_001.json">条件参考验收与范围</a>'
            '<a href="/task_protocol/conditional_reachability_reference_001.md">独立参考协议</a></div></section>')
        content=content.replace('<section><h2>逐项验收状态</h2>',note+'<section><h2>逐项验收状态</h2>')
        reference_evidence=dict(acceptance_sha256=digest(reference/'acceptance_001.json'),
            audit_sha256=accepted['audit_sha256'],preview_manifest_sha256=digest(preview/'manifest.json'))
    (dest/'index.html').write_text(content)
    shutil.copy2(__file__,dest/'generator_source.py')
    dump(dest/'manifest.json',dict(source_sha256=digest(Path(__file__)),
        html_sha256=digest(dest/'index.html'),created_utc=datetime.now(timezone.utc).isoformat(),
        completed_main_trials=450 if completed_evidence else None, manuscript_text_written=False,
        natural_review_and_collection_separate=True, read_only=True,live_status_from_saved_evidence=True,
        missing_results_not_success=True,human_submission_written_by_portal=False,goal_complete=False))
    if completed_evidence:
        dump(dest/'completed_evidence.json', completed_evidence)
    if region_evidence:
        dump(dest/'region_evidence.json', region_evidence)
    if reference_evidence:
        dump(dest/'reference_evidence.json', reference_evidence)
    if causal_evidence:
        dump(dest/'causal_evidence.json', causal_evidence)
    print('http://localhost:8768/visualizations/'+args.run_id+'/index.html')


if __name__ == '__main__':
    main()
