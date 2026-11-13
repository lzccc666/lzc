"""完整228条件的实际控制地图查看器；不生成论文图或伪造动画状态。"""
import json
from pathlib import Path

from conceptgraph.completion.stage9 import OUTPUT, digest, dump

PAGE = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>中心线引导与原Hybrid配对诊断</title><style>*{box-sizing:border-box}body{font:16px system-ui;margin:0;background:#f1f4f7;color:#162b3d}main{max-width:1320px;margin:auto;padding:18px}h1{font-size:25px}.card{background:white;border-radius:10px;padding:16px;margin:14px 0}.choices{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}label{display:block}button,select,input{font:inherit;max-width:100%;padding:8px}canvas{width:100%;height:auto;border:1px solid #dce5ec}#time{width:100%}#status,#details{white-space:pre-wrap;overflow-wrap:anywhere}.muted{color:#50677a}.legend{line-height:1.8}a{color:#1263a5}</style>
<main><h1>真实开发地图：同预算规划对照</h1><p>38场景、3种子、2规划器的全部228次条件均可查看。后轴从公开的地图参考位置起步，障碍为保存的当前预测静态快照；这是给定地图路线的运动学诊断，参考位姿到达不代表选对对象、正确锁定或语义导航到达。</p>
<div class="card choices"><label>场景<select id="scene"></select></label><label>规划器<select id="method"><option value="hybrid_astar">原 Hybrid A*</option><option value="reference_guided">中心线引导</option></select></label><label>视图<select id="zoom"><option value="trajectory">实际路线附近</option><option value="corridor">完整车道链</option></select></label><label>扰动种子<select id="seed"><option>17</option><option>29</option><option>43</option></select></label></div>
<div class="card"><p id="status"></p><canvas id="map" width="1200" height="700" aria-label="官方车道、计划与实际车辆状态"></canvas><p class="legend">浅蓝：入口车道　浅紫：连接段　浅绿：出口车道　灰框：保存预测障碍　蓝线：计划　橙线：实际轨迹　黑色矩形：实际车辆　绿色圈：参考终点</p><input id="time" type="range" min="0" value="0" aria-label="保存控制状态"><button id="previous">上一状态</button> <button id="play">播放保存状态</button> <button id="next">下一状态</button><p id="details"></p></div>
<div class="card"><p id="bounds" class="muted"></p><a id="raw" target="_blank">查看原始试验结果</a> · <a id="independent" target="_blank">查看独立复核</a><p class="muted">播放只读取真实保存的车辆积分状态，不补造位姿或控制。失败、缺条件和未运动的记录均保留；没有状态时不播放。候选连接区间数与实际连接次数分开显示。</p></div></main>
<script>
const $=id=>document.getElementById(id);let cards=[],current=null,states=[],task=null,context=null,geometry=null,plan=null,timer=null,epoch=0,checks=null;
$('zoom').addEventListener('change',()=>{if(current)draw()});
const names={unknown_artificial_start_exceeds_30m_bound:'参考起步距录制相机超过30米，条件未知',unknown_entry_or_exit_lane_too_short:'入口或出口车道过短',unknown_no_declared_outgoing_connector:'无已声明出口连接段',unknown_no_aligned_nearby_entry_lane:'附近没有航向相容入口车道',unknown_connector_has_no_declared_exit_lane:'连接段缺已声明出口车道',UNKNOWN_INVALID_DECLARED_CHAIN:'连接端点或几何不满足条件',NO_ROUTE:'未生成路线',SAFETY_GUARD_STOPPED:'执行安全门制动停止',REFERENCE_POSE_REACHED:'参考位姿到达（语义到达信用为0）',INVALID_INITIAL_BODY_OR_ENTRY_LANE:'初始车身或入口车道无效'};
function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='播放保存状态'}
async function read(p){let r=await fetch('../../'+p);if(!r.ok)throw Error(p+' '+r.status);return r.json()}
async function choose(){stop();let e=++epoch;let chosen=cards.find(c=>c.scene===$('scene').value&&c.method===$('method').value&&c.seed===Number($('seed').value));$('status').textContent='正在读取原始证据……';let [t,c]=await Promise.all([read(chosen.task_path),read(chosen.context_path)]);let values=[[],null,null];if(chosen.states_path)values=await Promise.all([read(chosen.states_path),read(chosen.plan_path),read(chosen.geometry_path)]);if(e!==epoch)return;current=chosen;task=t;context=c;[states,plan,geometry]=values;$('time').max=Math.max(0,states.length-1);$('time').value=0;$('play').disabled=states.length<2;$('previous').disabled=!states.length;$('next').disabled=!states.length;let a=checks.find(c=>c.task_id===current.task_id&&c.seed===current.seed&&c.method===current.method);current.audit=a;$('status').textContent=`${current.scene} · ${current.method} · seed ${current.seed}\n${names[current.status]||current.status}；实际运动 ${current.motion?'有':'无'}；独立确认参考位姿 ${a.independently_confirmed_reference_pose?'是':'否'}`;$('bounds').textContent='两方法共用只移除连续完全相同XY采样的显式规范化走廊，曲线与有向非零边保持。两版本均3秒/2000扩展，按固定奇偶交替顺序。扩展工作单位不同，共享CPU不代表独占性能。查看旧失败后的开发改进，同时改变建议控制与拓扑搜索，不将增益单独归因于一个组件。无未来感知或语言模型调用。';$('raw').href='../../'+current.result_path;$('independent').href='../../vehicle_execution/HDmap_reference_guided_audit_002/report.json';draw()}

function draw(){let canvas=$('map'),ctx=canvas.getContext('2d');ctx.fillStyle='#f8fafc';ctx.fillRect(0,0,canvas.width,canvas.height);let lanes=(task.lane_tokens||[]).map(t=>context.lane_geometries[t]).filter(Boolean);if(!lanes.length)lanes=Object.values(context.lane_geometries).slice(0,60);let points=lanes.flatMap(l=>l.polygon);if(task.start_xy)points.push(task.start_xy,task.goal_xy);if($('zoom').value==='trajectory'){let local=[...(plan?plan.path:[]),...states.map(s=>[s.x,s.y])];if(task.start_xy)local.push(task.start_xy,task.goal_xy);if(local.length)points=local}let xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);let xmin=Math.min(...xs)-5,xmax=Math.max(...xs)+5,ymin=Math.min(...ys)-5,ymax=Math.max(...ys)+5;let scale=Math.min((canvas.width-60)/(xmax-xmin),(canvas.height-60)/(ymax-ymin));const xy=p=>[30+(p[0]-xmin)*scale,canvas.height-30-(p[1]-ymin)*scale];function path(ps,stroke,width,fill){if(!ps||!ps.length)return;ctx.beginPath();ps.forEach((p,i)=>{let q=xy(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});if(fill){ctx.closePath();ctx.fillStyle=fill;ctx.fill()}ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.stroke()}
lanes.forEach((l,i)=>{path(l.polygon,'#8398a8',1,['#d8eaf5','#e5dcf2','#d9eedf'][i%3]);let line=l.centerline;path(line,'#7d8e9f',1);let m=line[Math.floor(line.length/2)];if(m){let p=xy(m),h=m[2];ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(p[0]+16*Math.cos(h),p[1]-16*Math.sin(h));ctx.strokeStyle='#576b80';ctx.stroke()}});
let obs=geometry?geometry.obstacles:context.obstacles_geojson;for(let o of obs){let polys=o.type==='Polygon'?[o.coordinates]:o.type==='MultiPolygon'?o.coordinates:[];for(let p of polys)path(p[0],'#656b73',1,'#b2b8bf')}
if(plan)path(plan.path,'#1767b5',2);let index=Number($('time').value);path(states.slice(0,index+1).map(s=>[s.x,s.y]),'#eb7a21',4);if(task.goal_xy){let p=xy(task.goal_xy);ctx.beginPath();ctx.arc(...p,.75*scale,0,Math.PI*2);ctx.strokeStyle='#17814e';ctx.lineWidth=3;ctx.stroke()}
if(states.length){let s=states[index];let body=[[-.9,-.925],[3.7,-.925],[3.7,.925],[-.9,.925]].map(([x,y])=>[s.x+x*Math.cos(s.yaw_rad)-y*Math.sin(s.yaw_rad),s.y+x*Math.sin(s.yaw_rad)+y*Math.cos(s.yaw_rad)]);path(body,'#101f2e',2,'#36495b');$('details').textContent=`状态 ${index+1}/${states.length} · 保存仿真时间 ${s.time_s.toFixed(1)}秒 · 速度 ${s.speed_mps.toFixed(3)}米/秒\n全程候选连接控制区间 ${current.audit.connection_candidate_intervals??0}；全程完整见证过渡 ${current.audit.accepted_distinct_connections} 次；语义正确到达信用 0。`}else{$('details').textContent='本条件没有实际控制状态；不能记执行安全或到达。'}}
$('scene').onchange=()=>choose().catch(error);$('method').onchange=()=>choose().catch(error);$('seed').onchange=()=>choose().catch(error);function error(e){$('status').textContent='读取失败：'+e.message} $('time').oninput=()=>{stop();draw()};$('previous').onclick=()=>{stop();$('time').value=Math.max(0,Number($('time').value)-1);draw()};$('next').onclick=()=>{stop();$('time').value=Math.min(states.length-1,Number($('time').value)+1);draw()};$('play').onclick=()=>{if(timer){stop();return}$('play').textContent='暂停';timer=setInterval(()=>{if(Number($('time').value)>=states.length-1){stop();return}$('time').value=Number($('time').value)+1;draw()},100)};
Promise.all([fetch('cards.json').then(r=>r.json()),fetch('audit_cases.json').then(r=>r.json())]).then(async([c,a])=>{cards=c;checks=a;$('scene').replaceChildren(...[...new Set(cards.map(c=>c.scene))].map(s=>{let o=document.createElement('option');o.value=s;o.textContent=s;return o}));await choose()}).catch(error);
</script></html>'''


def main():
    root = OUTPUT/'visualizations/HDmap_reference_guided_preview_003'
    root.mkdir(exist_ok=False)
    source = OUTPUT/'vehicle_execution/HDmap_reference_guided_full_002'
    tasks = {t['task_id']: t for t in json.loads((OUTPUT/'vehicle_execution/HDmap_connection_tasks_001/tasks.json').read_text())}
    cards = []
    for row in map(json.loads, (source/'trials.jsonl').open()):
        task = tasks[row['task_id']]
        task_path = root/'tasks'/f"{row['scene']}.json"
        if not task_path.exists():
            dump(task_path, task)
        folder = source/f"{row['index']:03d}_{row['scene']}_seed{row['seed']}_{row['method']}"
        cards.append(dict(task_id=row['task_id'], scene=row['scene'], seed=row['seed'], method=row['method'],
            status=row['status'], motion=row['actual_vehicle_motion'], task_path=str(task_path.relative_to(OUTPUT)),
            context_path=str(Path(task['context_path']).relative_to(OUTPUT)), result_path=str((folder/'result.json').relative_to(OUTPUT)),
            states_path=str((folder/'states.json').relative_to(OUTPUT)) if (folder/'states.json').exists() else None,
            plan_path=str((folder/'plan.json').relative_to(OUTPUT)), geometry_path=str((folder/'geometry.json').relative_to(OUTPUT))))
    dump(root/'cards.json', cards)
    audit = OUTPUT/'vehicle_execution/HDmap_reference_guided_audit_002/cases.jsonl'
    dump(root/'audit_cases.json', [json.loads(line) for line in audit.open()])
    (root/'index.html').write_text(PAGE)
    dump(root/'manifest.json', dict(cases=len(cards), scenes=len(tasks), cards_sha256=digest(root/'cards.json'),
        page_sha256=digest(root/'index.html'), audit_cases_sha256=digest(root/'audit_cases.json'),
        source_trial_records_sha256=digest(source/'trials.jsonl'), source_module_sha256=digest(__file__),
        source_independent_audit_sha256=digest(audit), simulated_interpolated_states_created=0,
        model_calls=0, semantic_arrival_credit=0, goal_complete=False))
    print(json.dumps(dict(cases=len(cards), scenes=len(tasks), goal_complete=False)))


if __name__ == '__main__':
    main()
