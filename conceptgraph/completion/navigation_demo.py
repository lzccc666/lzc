"""为全部保存的感知导航试验生成只读浏览器证据查看器；不重新运行方法。"""
from __future__ import annotations

import argparse
import bisect
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from urllib.parse import quote

import numpy as np
from PIL import Image

from conceptgraph.completion.kinematic_benchmark import SPEC
from conceptgraph.completion.stage9 import OUTPUT, dump, digest


def url(path):
    path=Path(path).resolve()
    return '/'+quote(str(path.relative_to(OUTPUT)),safe='/')


def lines(path):
    rows=[]
    if not path.exists():
        return rows,None
    for index,line in enumerate(path.read_text().splitlines()):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            return rows,dict(line=index+1,reason='原始文件末尾不完整；只显示已保存完整前缀，不能判定试验成功')
    return rows,None


def build_case(case,task,config):
    records,partial=lines(case/'perception.jsonl')
    execution,execution_partial=lines(case/'execution.jsonl')
    result=json.loads((case/'result.json').read_text()) if (case/'result.json').exists() else None
    times=[r['state']['time_s'] for r in execution]
    frames=[]
    checked=0
    for frame in records:
        sensor=frame['sensor']
        for key in ['rgb','depth']:
            if digest(sensor[key])!=sensor[key+'_sha256']:
                raise ValueError('saved actual sensor evidence changed')
            checked+=1
        depth=np.asarray(Image.open(sensor['depth']),dtype=float)/1000.
        v,u=np.nonzero(depth>0)
        take=np.linspace(0,len(v)-1,min(len(v),384)).astype(int) if len(v) else np.array([],int)
        z=depth[v[take],u[take]]
        camera=np.column_stack([(u[take]-320)*z/320,(v[take]-180)*z/320,z])
        pose=np.asarray(sensor['camera_pose_cv'])
        world=camera@pose[:3,:3].T+pose[:3,3]
        overlay=case/'detection_vis'/f"{frame['frame_idx']:06d}.jpg"
        observed_time=sensor['state']['time_s']
        state_index=bisect.bisect_right(times,observed_time)
        nodes=[]
        for n in frame['nodes']:
            node=dict(n)
            metadata=dict(node['metadata'])
            metadata['crop_urls']=[url(p) for p in metadata.get('crop_paths',[]) if Path(p).exists()]
            node['metadata']=metadata
            nodes.append(node)
        frames.append(dict(frame_idx=frame['frame_idx'],sim_time_s=observed_time,
            sensed_wall_s=frame.get('sensed_wall_elapsed_s'),completed_wall_s=frame.get('completed_wall_elapsed_s'),
            rgb=url(sensor['rgb']),detection_vis=url(overlay) if overlay.exists() else None,
            sparse_depth_pixels=np.round(np.column_stack([u[take],v[take],z]),3).tolist(),
            world_lidar_sample=np.round(world,3).tolist(),actual_lidar_pixels=len(v),state=sensor['state'],
            nodes=nodes,relations=frame['relations'],candidates=frame['candidates'],lock=frame['lock'],
            selected_node_id=frame['selected_node_id'],planning=frame['planning'],path=frame['path_world'],
            obstacles=frame['known_obstacle_polygons'],parsed_query=frame['parsed_query'],
            semantic_audit=frame.get('semantic_audit'),execution=execution[:state_index],
            evaluation_only_identity=frame.get('evaluation_only_identity'),
            evaluation_only_matches=frame.get('evaluation_only_matches',{})))
    # 执行数据只保存一份；每帧用前缀长度查看，避免把未来轨迹混入当帧展示。
    for frame in frames:
        frame['execution_prefix_length']=len(frame.pop('execution'))
    sources={p.name:digest(p) for p in [case/'perception.jsonl',case/'execution.jsonl',case/'result.json'] if p.exists()}
    semantic=case/'semantic_events.json'
    return dict(task=task,config=config,seed=int(case.name.removeprefix('seed')),
        run=case.parents[1].name,result=result,process_complete=result is not None,
        status=result['status'] if result else 'PROCESS_INTERRUPTED_NO_RESULT',
        partial=partial,execution_partial=execution_partial,
        frames=frames,execution=execution,raw_sources={p.name:url(p) for p in
            [case/'perception.jsonl',case/'execution.jsonl',case/'result.json'] if p.exists()},
        semantic_events_url=url(semantic) if semantic.exists() else None,
        semantic_response_urls=[url(p) for p in sorted((case/'semantic_responses').glob('*.json'))],
        sources_sha256=sources,sensor_files_verified=checked,
        depth_display='真实投影LiDAR最多均匀抽样384点，仅供浏览；推理使用原始完整深度',
        gt_layer='官方合成身份关联和冻结到达区域仅作隔离评估；默认隐藏')


HTML=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>感知导航证据查看</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f1f4f6;color:#152c3b;font:15px/1.55 system-ui,"Noto Sans CJK SC",sans-serif}header,main{max-width:1440px;margin:auto;padding:22px}header{padding-bottom:6px}h1{font-size:25px;margin:0}h2{font-size:16px;margin:0 0 12px}.muted{color:#526674;font-size:13px;overflow-wrap:anywhere}.bar{display:flex;gap:12px;flex-wrap:wrap;align-items:center;padding:14px 0}select,button{font:inherit;padding:7px;border:1px solid #bbcbd4;border-radius:5px;background:white}select#case{max-width:100%;flex:1;min-width:200px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{background:white;padding:16px;border:1px solid #d3dfe5;border-radius:8px;min-width:0}.wide{grid-column:1/-1}img,canvas{display:block;width:100%;height:auto;background:#eaf0f3}canvas{touch-action:none}#space{height:430px}input[type=range]{width:100%}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:320px;overflow:auto;color:#293f4c}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;border-bottom:1px solid #dce5ea;padding:5px;overflow-wrap:anywhere}.outcomes{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}.outcome{padding:7px 12px;border:1px solid #c6d6e0;border-radius:5px}.links a{margin-right:15px}a{color:#17658b}#error{color:#9c2e18}.node{padding:7px 0;border-bottom:1px solid #e0e7eb}.node img{display:inline-block;width:90px;margin:5px;vertical-align:middle}details{margin:10px 0}@media(max-width:800px){.grid{grid-template-columns:1fr}header,main{padding:12px}#space{height:330px}.wide{grid-column:auto}h1{font-size:21px}}</style>
<header><h1>感知导航证据查看</h1><p class="muted">完整保存的成功、失败和拒绝均可查看。这里回放真实控制及传感器日志；播放速度不表示实时性能。新封存测试尚未开启。</p></header>
<main><div class="bar"><label>筛选 <select id="filter"><option value="all">全部试验</option><option value="arrived">实际正确到达</option><option value="failed">失败或中断</option><option value="no_match">无匹配</option><option value="inaccessible">独立不可接近</option></select></label><select id="case" aria-label="选择保存试验"></select></div><p id="query"></p><div id="outcomes" class="outcomes"></div><p id="status" class="muted"></p><p id="error" role="alert"></p>
<div class="card"><div class="bar"><button id="play">播放保存帧</button><span id="frameLabel"></span><label><input id="gt" type="checkbox">显示隔离评估层</label><label><input id="lidar" type="checkbox" checked>RGB叠加实测深度点</label></div><input id="frame" aria-label="感知帧" type="range" min="0" value="0" max="0"></div>
<div class="grid" style="margin-top:16px"><section class="card"><h2>车载RGB与有效LiDAR</h2><canvas id="rgb" width="640" height="360"></canvas><p id="sensor" class="muted"></p></section><section class="card"><h2>保存的检测／SAM输出图</h2><img id="det" alt="当帧实际检测和分割输出"></section><section class="card wide"><h2>世界坐标对象、道路、计划与实际执行</h2><canvas id="space" width="1200" height="430"></canvas><p class="muted">拖动改变观察角度。蓝色道路、橙色计划、深绿色实际轨迹；灰色为实测障碍，框为当帧估计对象，红框为实际车辆外形。评估层默认隐藏。空间图仅为日志几何展示，不是完整环境渲染。</p></section><section class="card"><h2>对象与真实crop</h2><div id="nodes"></div></section><section class="card"><h2>候选、关系及语义</h2><div id="candidates"></div><details><summary>关系链和证据</summary><pre id="relations"></pre></details><details><summary>解析与安全候选排序记录</summary><pre id="semantics"></pre></details><p id="semanticLinks" class="links"></p></section><section class="card wide"><h2>车辆状态与独立评价</h2><pre id="execution"></pre><div id="sources" class="links"></div></section></div></main>
<script>
const $=id=>document.getElementById(id);let manifest,data,frameIndex=0,request=0,playing=null,yaw=-.65,pitch=.9,drag=null;
const pretty=v=>JSON.stringify(v,null,2),esc=v=>String(v??'unknown').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function stop(){clearInterval(playing);playing=null;$('play').textContent='播放保存帧'}
async function choose(){stop();const token=++request;const res=await fetch($('case').value);if(!res.ok)throw Error('无法读取该试验');const loaded=await res.json();if(token!==request)return;data=loaded;frameIndex=0;$('frame').max=Math.max(0,data.frames.length-1);$('frame').value=0;$('query').textContent=data.task.query_text;const r=data.result;const labels=[['route_generated','生成路线'],['selected_object_evaluated_correct','选对对象'],['stable_correct_lock','正确锁定'],['actual_full_perception_arrival','实际到达']];$('outcomes').innerHTML=labels.map(([k,l])=>'<span class="outcome">'+l+'：'+(r?(r[k]?'是':'否'):'结果未保存')+'</span>').join('');$('status').textContent=`${data.run} · ${data.task.task_id} · seed${data.seed} · ${data.status}；${data.process_complete?'有保存终止结果':'进程中断，不能推断终止表现'}`;$('sources').innerHTML=Object.entries(data.raw_sources).map(([k,v])=>`<a href="${v}" target="_blank">${esc(k)}</a>`).join('');$('semanticLinks').innerHTML=(data.semantic_events_url?`<a href="${data.semantic_events_url}" target="_blank">后台语义队列原始记录</a>`:'')+data.semantic_response_urls.map((v,i)=>`<a href="${v}" target="_blank">模型响应${i+1}</a>`).join('');render()}
function filter(){const f=$('filter').value;const rows=manifest.cases.filter(r=>f==='all'||f==='arrived'&&r.arrived||f==='failed'&&(r.status==='PROCESS_INTERRUPTED_NO_RESULT'||!r.arrived&&r.kind==='reachable_positive')||r.kind===f);$('case').innerHTML=rows.map(r=>`<option value="${r.file}">${esc(r.run)} / ${esc(r.task_id)} / seed${r.seed} / ${esc(r.status)}</option>`).join('');if(rows.length)choose().catch(fail)}
function fail(e){$('error').textContent=e.message}
function render(){if(!data)return;const f=data.frames[frameIndex];if(!f){$('error').textContent='本试验没有完整保存的感知帧；请查看原始中断记录。';return}$('error').textContent='';$('frameLabel').textContent=`感知帧${f.frame_idx} / ${data.frames.length} · 仿真${f.sim_time_s.toFixed(1)}秒`;$('sensor').textContent=`实际有效深度像素${f.actual_lidar_pixels}；浏览抽样${f.sparse_depth_pixels.length}点，颜色由近到远为深浅蓝。观测墙钟：${f.sensed_wall_s===null?'固定逻辑时钟':f.sensed_wall_s.toFixed(2)+'秒'}；完成墙钟：${f.completed_wall_s===null?'未记录':f.completed_wall_s.toFixed(2)+'秒'}`;
const rgb=new Image;const idx=frameIndex;rgb.onload=()=>{if(idx!==frameIndex||f!==data.frames[frameIndex])return;const ctx=$('rgb').getContext('2d');ctx.drawImage(rgb,0,0,640,360);if($('lidar').checked)for(const [u,v,z] of f.sparse_depth_pixels){const l=20+Math.min(1,z/65)*65;ctx.fillStyle=`hsl(202 90% ${l}%)`;ctx.fillRect(u-1.3,v-1.3,2.6,2.6)}};rgb.onerror=()=>fail(Error('实际RGB未能加载'));rgb.src=f.rgb;$('det').src=f.detection_vis||f.rgb;
$('nodes').innerHTML=f.nodes.map(n=>{const m=n.metadata,truth=$('gt').checked?(f.evaluation_only_matches[String(n.node_id)]||'unknown'):'评估层隐藏';return `<div class="node"><b>节点${n.node_id} · ${esc(n.category)} · ${esc(n.motion)}</b><br><span class="muted">${esc(m.object_identity)} / 官方关联：${esc(truth)}</span><br>中心 ${n.center.map(v=>v.toFixed(2)).join(', ')} m；观测${m.num_detections}次<br>${(m.crop_urls||[]).slice(-2).map(p=>`<a href="${p}" target="_blank"><img src="${p}" alt="实际输入crop"></a>`).join('')}<details><summary>深度与属性来源</summary><pre>${esc(pretty({depth:m.depth_evidence,attributes:m.attributes}))}</pre></details></div>`}).join('')||'当帧没有已估计对象';$('candidates').innerHTML='<table><tr><th>节点</th><th>保存的联合分数</th></tr>'+f.candidates.map(c=>`<tr><td>${c.node_id}</td><td>${c.score.toFixed(4)}</td></tr>`).join('')+'</table><p class="muted">无候选时不补造评分；分数不等于对象正确率。</p>';$('relations').textContent=pretty(f.relations);$('semantics').textContent=pretty({parsed:f.parsed_query,lock:f.lock,planning:f.planning,semantic_audit:f.semantic_audit});const last=data.execution[Math.max(0,f.execution_prefix_length-1)];$('execution').textContent=pretty({observed_state:f.state,last_saved_control_at_or_before_observation:f.execution_prefix_length?last:null,current_evaluation_identity:$('gt').checked?f.evaluation_only_identity:'评估层隐藏',final_result:data.result});drawSpace(f)}
function drawSpace(f){const c=$('space'),ctx=c.getContext('2d'),w=c.width,h=c.height;ctx.clearRect(0,0,w,h);const road=data.config.road_polygon_xy,center=data.config.centerline_xy;const mx=center.reduce((s,p)=>s+p[0],0)/center.length,my=center.reduce((s,p)=>s+p[1],0)/center.length;function raw(p){const x=p[0]-mx,y=p[1]-my,z=p[2]||0;return [Math.cos(yaw)*x-Math.sin(yaw)*y,Math.sin(pitch)*(Math.sin(yaw)*x+Math.cos(yaw)*y)-Math.cos(pitch)*z]}const rr=road.map(raw),xs=rr.map(p=>p[0]),ys=rr.map(p=>p[1]);const sc=Math.min((w-160)/(Math.max(...xs)-Math.min(...xs)+4),(h-90)/(Math.max(...ys)-Math.min(...ys)+8));const cx=(Math.max(...xs)+Math.min(...xs))/2,cy=(Math.max(...ys)+Math.min(...ys))/2;const project=p=>{const q=raw(p);return [w/2+(q[0]-cx)*sc,h/2+(q[1]-cy)*sc]};function line(points,color,width=1,fill=null){if(!points.length)return;ctx.beginPath();points.forEach((p,i)=>{const q=project(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});if(fill){ctx.closePath();ctx.fillStyle=fill;ctx.fill()}ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}line(road,'#7a9eb3',1,'#dfebf2');line(center,'#8496a1',1);for(const o of f.obstacles)line(o,'#8899a3',.7,'#d1dade');for(const p of f.world_lidar_sample){const q=project(p);ctx.fillStyle='#3983a780';ctx.fillRect(q[0],q[1],1.5,1.5)}line(f.path,'#c77730',3);line(data.execution.slice(0,f.execution_prefix_length).map(r=>[r.state.x,r.state.y]),'#156f61',3);for(const n of f.nodes){const corners=[];for(const z of [-1,1])for(const [x,y] of [[-1,-1],[1,-1],[1,1],[-1,1]])corners.push([n.center[0]+x*n.extent[0]/2,n.center[1]+y*n.extent[1]/2,n.center[2]+z*n.extent[2]/2]);const color=n.node_id===f.selected_node_id?'#9d571d':'#335e82';line([...corners.slice(0,4),corners[0]],color);line([...corners.slice(4,8),corners[4]],color);for(let i=0;i<4;i++)line([corners[i],corners[i+4]],color);const p=project(n.center);ctx.fillStyle=color;ctx.font='14px sans-serif';ctx.fillText(n.node_id+': '+n.category,p[0],p[1]-5)}const s=f.state,body=[[-.9,-.925],[3.7,-.925],[3.7,.925],[-.9,.925],[-.9,-.925]].map(([x,y])=>[s.x+Math.cos(s.yaw_rad)*x-Math.sin(s.yaw_rad)*y,s.y+Math.sin(s.yaw_rad)*x+Math.cos(s.yaw_rad)*y,.15]);line(body,'#a33d36',2,'#a33d3630');if($('gt').checked)for(const region of data.task.regions)line(region.polygon_xy,'#7852a0',2,'#7852a025');ctx.fillStyle='#203d50';ctx.font='14px sans-serif';ctx.fillText('世界坐标（m） · 当帧与过去执行 · 评估层'+($('gt').checked?'显示':'隐藏'),18,26)}
$('filter').onchange=filter;$('case').onchange=()=>choose().catch(fail);$('frame').oninput=()=>{frameIndex=Number($('frame').value);render()};$('gt').onchange=render;$('lidar').onchange=render;$('play').onclick=()=>{if(playing){stop();return}$('play').textContent='暂停';playing=setInterval(()=>{if(frameIndex>=data.frames.length-1){stop();return}$('frame').value=++frameIndex;render()},700)};$('space').onpointerdown=e=>{drag=[e.clientX,e.clientY];$('space').setPointerCapture(e.pointerId)};$('space').onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.006;pitch=Math.max(.15,Math.min(1.5,pitch+(e.clientY-drag[1])*.006));drag=[e.clientX,e.clientY];if(data?.frames[frameIndex])drawSpace(data.frames[frameIndex])};$('space').onpointerup=()=>drag=null;fetch('manifest.json').then(r=>{if(!r.ok)throw Error('请通过核验服务8767端口打开页面');return r.json()}).then(v=>{manifest=v;filter()}).catch(fail);
</script></html>'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--source-run',action='append',default=[])
    parser.add_argument('--reuse-case-package')
    args=parser.parse_args()
    dest=OUTPUT/'visualizations'/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    (dest/'cases').mkdir()
    tasks_release=OUTPUT/'interactive_simulation/synthetic_tasks_v1.json'
    release=json.loads(tasks_release.read_text())
    tasks={t['task_id']:t for t in release['tasks']}
    configs={c['config_id']:c for c in release['configs']}
    rows=[]
    files={}
    reused_manifest=None
    if args.reuse_case_package:
        old=OUTPUT/'visualizations'/args.reuse_case_package
        previous=json.loads((old/'manifest.json').read_text())
        if previous['task_release_sha256']!=digest(tasks_release) or previous['vehicle_spec']!=asdict(SPEC):
            raise ValueError('frozen tasks or vehicle changed')
        for name,sha in previous['files'].items():
            if not name.startswith('cases/'):
                continue
            if digest(old/name)!=sha:
                raise ValueError('previous case package changed')
            shutil.copy2(old/name,dest/name)
            files[name]=sha
        rows=previous['cases']
        reused_manifest=dict(path=str(old/'manifest.json'),sha256=digest(old/'manifest.json'))
    if not args.source_run and not args.reuse_case_package:
        raise ValueError('at least one source run or a verified case package required')
    for run in args.source_run:
        source=OUTPUT/'interactive_simulation'/run
        for case in sorted(source.glob('*/seed*')):
            if not (case/'perception.jsonl').exists():
                continue
            task=tasks[case.parent.name]
            data=build_case(case,task,configs[task['config_id']])
            name=f"{run}_{task['task_id']}_{case.name}.json"
            dump(dest/'cases'/name,data)
            files['cases/'+name]=digest(dest/'cases'/name)
            result=data['result']
            rows.append(dict(run=run,task_id=task['task_id'],seed=data['seed'],kind=task['kind'],
                status=data['status'],arrived=bool(result and result['actual_full_perception_arrival']),
                file='cases/'+name,frames=len(data['frames']),sources_sha256=data['sources_sha256']))
    (dest/'index.html').write_text(HTML)
    shutil.copy2(__file__,dest/'generator_source.py')
    files['index.html']=digest(dest/'index.html')
    files['generator_source.py']=digest(dest/'generator_source.py')
    dump(dest/'manifest.json',dict(cases=rows,source_runs=args.source_run,vehicle_spec=asdict(SPEC),
        task_release_sha256=digest(tasks_release),source_sha256=digest(__file__),files=files,
        reused_verified_case_package=reused_manifest,
        evaluation_gt_hidden_by_default=True,gt_in_agent=False,goal_complete=False))
    print(json.dumps(dict(cases=len(rows),frames=sum(r['frames'] for r in rows),
        url='http://localhost:8767/visualizations/'+args.run_id+'/index.html'),ensure_ascii=False))


if __name__=='__main__':
    main()
