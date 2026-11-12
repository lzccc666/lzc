"""统一只读入口：动态显示真实状态，不把进行中批次或目录存在当成完成。"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import shutil

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


HTML = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>三维语义场景图与车辆导航 · 实验证据</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f7;color:#182e3b;font:15px/1.65 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1160px;margin:auto;padding:28px 22px}h1{font-size:26px;line-height:1.4;margin:0 0 10px}h2{font-size:19px;margin:0 0 12px}h3{font-size:16px;margin:0 0 8px}.muted{color:#566774;font-size:13px;overflow-wrap:anywhere}.status{display:inline-block;background:#f6ead6;border:1px solid #d0b68a;padding:5px 12px;border-radius:5px}section{margin:22px 0;padding:20px;background:white;border:1px solid #d2dce2;border-radius:7px;min-width:0}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.grid section{margin:0}a{color:#1a618c;overflow-wrap:anywhere}p{margin:9px 0}.links{display:flex;gap:14px;flex-wrap:wrap}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #dce4e8;padding:9px;overflow-wrap:anywhere}th:first-child{width:120px}.four{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}.four div{padding:12px;border:1px solid #d7e0e5;border-radius:5px;font-size:13px}.four strong{display:block;font-size:15px}.scroll{overflow-x:auto}button{font:inherit;padding:7px 14px;color:#1a4b65;background:#edf3f7;border:1px solid #b6c8d3;border-radius:4px;cursor:pointer}.error{color:#9b3b30}code{overflow-wrap:anywhere}footer{color:#5a6e79;font-size:13px;margin-top:22px}@media(max-width:760px){main{padding:18px 12px}h1{font-size:22px}.grid{grid-template-columns:1fr}.four{grid-template-columns:repeat(2,1fr)}section{padding:15px}td,th{padding:7px}}
</style><main>
<h1>面向智能车辆的三维语义场景图构建与导航方法研究</h1>
<div id="goal" class="status">正在读取真实验收状态</div><p id="updated" class="muted"></p>
<p>当前入口汇集开发实验、运行中的完整矩阵、论文阶段稿与人工核验。新封存测试的开放和完成状态以验收文件为准。</p>
<div class="four"><div><strong>生成路线</strong>规划器给出目标路径。</div><div><strong>选对对象</strong>独立关联到可接受实例。</div><div><strong>正确锁定</strong>持续观测确认正确身份。</div><div><strong>实际到达</strong>执行后满足身份、区域、姿态、速度与保持时间。</div></div>
<div class="grid"><section><h2>完整真实时钟比较</h2><p id="main-progress">读取中</p><p id="main-last" class="muted"></p><p>50个冻结任务 × 3种子 × 规则/原始Qwen/新Adapter，共450次。完整结束前不展示总体优劣结论。</p><div class="links"><a href="/interactive_simulation/realtime_three_model_full_001/release.json">运行协议与源码指纹</a><a href="/interactive_simulation/realtime_three_model_full_001/trial_records.jsonl">全部已尝试记录</a><a href="/statistics/realtime_three_model_full_001/report.json">全量统计（完成后）</a></div><p class="muted">随后三种缺一语义条件新增450次，核验同版后与300份共享记录组成750条件；等待和复用不增加独立任务数。</p></section>
<section><h2>真实人工核验</h2><p id="human">读取中</p><p>439条原始自然指令及开发证据可提交审核。自动证据、官方标签和真实人工记录分别保存；unknown不补写成肯定标签。</p><div class="links"><a href="/annotation/natural_command_review_001/index.html">打开自然指令核验页</a><a href="/visualizations/causal_development_review_002/index.html">打开过去LiDAR开发核验页</a><a href="/visualizations/review.html">旧开发核验页（旧深度输入）</a></div><p class="muted">审核会写入单独的追加记录。当前同意使用页面不等于已产生审核结论。</p></section></div>
<section><h2>论文与可复现入口</h2><div id="paper-links" class="links"></div><p id="paper-status" class="muted"></p><div class="links"><a href="/visualizations/navigation_demo_004/index.html">三维感知与车辆执行查看器</a><a href="/reproducible_commands.md">复现及续跑命令</a><a href="/experiment_report.md">实验记录</a><a href="/remaining_issues.md">剩余必需工作</a><a href="/validation_report.json">逐项验收证据</a><a href="/baseline_verification.json">历史产物保护核验</a></div><p class="muted">查看器中的回放速度不是实时性能；163条保存开发试验包含重复复测，不增加独立样本。运行中的新矩阵尚未全部加入该查看器。</p></section>
<section><h2>历史离线开发消融：旧未来LiDAR输入</h2><p>以下保留旧诊断口径；不能与新的过去LiDAR成绩混用，也不获得因果在线感知信用。</p><div class="scroll"><table><thead><tr><th>实验</th><th>真实记录与限制</th><th>证据</th></tr></thead><tbody>
<tr><td>六种地图约束</td><td>38场景、212查询、1272次规划；176个不同几何查询。正确对象参考点由GT给定；完整约束仍有3条独立连接审计失败，静态矩形姿态全部未通过，未执行控制。</td><td><a href="/statistics/map_constraints_dev_full_001/report.json">统计</a><br><a href="/thesis/figures_constraints_002/图6-6.pdf">矢量图</a></td></tr>
<tr><td>动态维护与过期</td><td>114条件记录，76次新重建、38份同版基线复用；减动态维护召回下降4.93个百分点。运动unknown保留，未证明车辆到达收益。</td><td><a href="/statistics/mapping_modules_dev_full_001/report.json">总体统计</a><br><a href="/statistics/mapping_modules_motion_dev_full_001/report.json">官方动静态组</a><br><a href="/thesis/figures_constraints_002/图6-7.pdf">矢量图</a></td></tr>
<tr><td>首达、联合与接近点</td><td id="selection-progress">全量结果读取中；进行中不计完成。</td><td><a href="/ablations/goal_selection_dev_full_001/release.json">冻结协议</a><br><a href="/ablations/goal_selection_dev_full_001/report.json">全量统计（完成后）</a></td></tr>
<tr><td>道路探索分组</td><td>1115开发查询按当前位姿与HD地图归组；弯道正查询仅5条、2log，不支持道路泛化结论。</td><td><a href="/performance/road_groups_dev_001/report.json">分组统计</a></td></tr>
<tr><td>引用版本核验</td><td>被引PDF优先于存在冲突的注册作者顺序。连续85个引用不表示全部正文主张已核验，失败下载与被拒中间版本保留。</td><td><a href="/thesis/citation_corrections_007.json">条目核验</a><br><a href="/thesis/claim_citation_review_008.json">正文范围核验</a></td></tr>
</tbody></table></div></section>
<section><h2>逐项验收状态</h2><button id="refresh">刷新实际状态</button><p id="error" class="error"></p><div class="scroll"><table><thead><tr><th>项目</th><th>当前状态与未完成条件</th></tr></thead><tbody id="statuses"></tbody></table></div></section>
<footer>只读入口；数据直接来自保存结果。远程访问需转发当前核验服务8768端口。缺失结果显示未完成，不按零值或成功补齐。</footer>
</main><script>
const $=id=>document.getElementById(id);async function json(url){const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw Error('尚无完整文件：'+url);return r.json()}
function link(parent,label,path){const a=document.createElement('a');a.textContent=label;a.href=path;parent.appendChild(a)}
async function refresh(){
 $('error').textContent='';
 const results=await Promise.allSettled([json('/validation_report.json'),json('/api/review_status'),fetch('/interactive_simulation/realtime_three_model_full_001/trial_records.jsonl',{cache:'no-store'}).then(async r=>{if(!r.ok)throw Error('完整矩阵记录未就绪');const s=await r.text();const lines=s.split('\\n');if(!s.endsWith('\\n'))lines.pop();return lines.filter(x=>x.trim()).map(x=>JSON.parse(x))}),json('/ablations/goal_selection_dev_full_001/summary.json')]);
 const [validation,human,trials,selection]=results;
 if(validation.status==='fulfilled'){const v=validation.value;$('goal').textContent=v.goal_complete?'全部Goal已验收':'Goal仍在进行 · 必需条件未全部满足';$('updated').textContent='验收状态更新时间：'+v.updated_utc+'；新封存测试：'+(v.new_test_opened?'以对应release为准':'未开启');$('statuses').replaceChildren();for(const [name,s] of Object.entries(v.statuses)){const tr=document.createElement('tr'),a=document.createElement('th'),b=document.createElement('td');a.textContent=name;const strong=document.createElement('strong');strong.textContent=s.status;b.appendChild(strong);const p=document.createElement('p');p.textContent=s.remaining;b.appendChild(p);for(const f of s.evidence){link(b,'证据','/'+f);b.appendChild(document.createTextNode(' '))}tr.append(a,b);$('statuses').appendChild(tr)}const paper=v.statuses['论文'];$('paper-status').textContent=paper.remaining;$('paper-links').replaceChildren();const manifest=paper.evidence.find(x=>x.endsWith('/delivery_manifest.json'));if(manifest){const prefix='/'+manifest.slice(0,manifest.lastIndexOf('/'));for(const [name,suffix] of [['可编辑Word','docx'],['PDF阶段稿','pdf'],['完整Markdown','md']])link($('paper-links'),name,prefix+'/完整论文阶段整合稿.'+suffix)}}else{$('goal').textContent='无法读取验收文件，不能推断完成';$('error').textContent=String(validation.reason)}
 if(human.status==='fulfilled'){$('human').textContent='真实提交：开发 '+human.value.development_submissions+'；自然指令 '+human.value.natural_submissions+'。数量不等于可靠标签覆盖。'}else $('human').textContent='提交数量未知，服务尚未返回。';
 if(trials.status==='fulfilled'){const rows=trials.value;$('main-progress').textContent='已保存尝试记录 '+rows.length+' / 450；尚未据部分样本计算最终结论。';const r=rows.at(-1);$('main-last').textContent=r?'最近保存：'+r.variant+' · '+r.task+' · seed'+r.seed+' · '+r.status:'尚无保存试验'}else $('main-progress').textContent='当前完成数量未知，请查运行日志。';
 $('selection-progress').textContent=selection.status==='fulfilled'?'已完成 '+selection.value.query_cases+' 查询、'+selection.value.records+' 条件记录。联合与首达共享实际规划池；去保护条件未执行；不计实际到达。':'全量仍进行中，尚无完整summary；32查询/96条件试跑已保存，不替代全量。';
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
        html_sha256=digest(dest/'index.html'),read_only=True,live_status_from_saved_evidence=True,
        missing_results_not_success=True,human_submission_written_by_portal=False,goal_complete=False))
    if region_evidence:
        dump(dest/'region_evidence.json', region_evidence)
    if reference_evidence:
        dump(dest/'reference_evidence.json', reference_evidence)
    if causal_evidence:
        dump(dest/'causal_evidence.json', causal_evidence)
    print('http://localhost:8768/visualizations/'+args.run_id+'/index.html')


if __name__ == '__main__':
    main()
