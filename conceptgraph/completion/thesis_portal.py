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
<section><h2>真实人工核验</h2><p id="human">读取中</p><p>439条原始自然指令及开发证据可提交审核。自动证据、官方标签和真实人工记录分别保存；unknown不补写成肯定标签。</p><div class="links"><a href="/annotation/natural_command_review_001/index.html">打开自然指令核验页</a><a href="/visualizations/review.html">打开开发证据核验页</a></div><p class="muted">审核会写入单独的追加记录。当前同意使用页面不等于已产生审核结论。</p></section></div>
<section><h2>论文与可复现入口</h2><div id="paper-links" class="links"></div><p id="paper-status" class="muted"></p><div class="links"><a href="/visualizations/navigation_demo_004/index.html">三维感知与车辆执行查看器</a><a href="/reproducible_commands.md">复现及续跑命令</a><a href="/experiment_report.md">实验记录</a><a href="/remaining_issues.md">剩余必需工作</a><a href="/validation_report.json">逐项验收证据</a><a href="/baseline_verification.json">历史产物保护核验</a></div><p class="muted">查看器中的回放速度不是实时性能；163条保存开发试验包含重复复测，不增加独立样本。运行中的新矩阵尚未全部加入该查看器。</p></section>
<section><h2>新增离线开发消融</h2><div class="scroll"><table><thead><tr><th>实验</th><th>真实记录与限制</th><th>证据</th></tr></thead><tbody>
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
    print('http://localhost:8768/visualizations/'+args.run_id+'/index.html')


if __name__ == '__main__':
    main()
