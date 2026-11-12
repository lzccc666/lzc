"""只读检查独立静态终点区域、RGB及HD/官方盒；不写人工或测试结果。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>静态终点区域证据核验</title><style>
*{box-sizing:border-box}body{margin:0;background:#eef2f6;color:#182432;font:15px/1.6 system-ui,sans-serif}
main{max-width:1380px;margin:auto;padding:18px}h1{font-size:25px;margin:0 0 8px}.card{background:white;padding:16px;border-radius:12px;margin:12px 0;min-width:0}
.notice{border-left:5px solid #c57513}.tools{display:flex;gap:10px;flex-wrap:wrap;align-items:center}select,button{font:inherit;padding:7px;max-width:100%;border:1px solid #aebbc6;border-radius:5px;background:white}
#query{width:100%;min-width:0}.views{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start}.views>div{min-width:0}img,canvas{width:100%;height:auto;display:block;border:1px solid #d5dde3;border-radius:6px}canvas{aspect-ratio:4/3;background:#fff}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#f5f7f9;padding:12px;max-height:350px;overflow:auto}a{color:#2059a4;overflow-wrap:anywhere}.muted{color:#536576}#status{font-weight:600}#queryText{font-size:19px}
@media(max-width:800px){main{padding:10px}.views{grid-template-columns:1fr}h1{font-size:21px}.card{padding:12px}}
</style><main><h1>独立静态终点区域证据</h1>
<div class="card notice"><strong>静态区域证书 ≠ 可达证明 ≠ 实际到达。</strong><p>区域由隔离评价端的官方盒和地图产生，未运行任何预留测试方法。全体可达性仍为 unknown，未生成控制轨迹或人工记录。这里检查官方 sample 时刻的冻结盒，不保证未来动态交通安全。</p><div id="audit"></div></div>
<div class="card"><div class="tools"><label>场景 <select id="scene"><option value="all">全部场景</option></select></label><label>状态 <select id="filter"><option value="all">全部</option><option value="certified_static_regions">有静态区域证书</option><option value="unknown_no_certificate">未知：未找到证书</option><option value="not_applicable_no_matching_target">范围内无匹配</option></select></label><button id="previous">上一条</button><button id="next">下一条</button><span id="counter"></span></div><p><select id="query" aria-label="选择指令"></select></p><p id="queryText"></p><div id="status"></div></div>
<div class="card views"><div><p>原始相机图像；完整RGB与标注帧一致</p><img id="rgb" alt="当前独立标签所对应的原始RGB"></div><div><p>世界坐标俯视证据；蓝点为相机位置，并非车辆后轴初始位置</p><canvas id="map" width="1000" height="750"></canvas><p class="muted">浅蓝：车道；灰色：全部类别官方盒；橙色：可接受静止目标；绿色：安全区域及整车误差包络。地图孔洞保留。未绘制计划路径或实际轨迹。</p></div></div>
<div class="card"><h2>当前证据</h2><pre id="details"></pre><div id="sources"></div><p><a href="../thesis_portal_002/index.html">返回课题总入口</a></p></div></main>
<script>
const state={rows:[],visible:[],index:0,serial:0};const $=id=>document.getElementById(id);
async function get(url){const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw Error(`${r.status}: ${url}`);return r.json()}
function sourceLink(label,path){const a=document.createElement('a');a.textContent=label;a.href=path;a.target='_blank';return a}
function polygonPaths(geometry){if(!geometry)return[];if(geometry.type==='Polygon')return[geometry.coordinates];if(geometry.type==='MultiPolygon')return geometry.coordinates;return[]}
function draw(context,targets,row){const canvas=$('map'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);if(!context){ctx.fillStyle='#536576';ctx.font='24px sans-serif';ctx.fillText('无匹配目标：不生成终点区域或可达结论',30,80);return}
 const camera=row.camera_xy,points=[camera];for(const t of targets){for(const poly of polygonPaths(t.target_polygon)){for(const p of poly[0])points.push(p)}for(const r of t.regions)points.push(r.nominal_rear_axle_xy)}
 let minX=Math.min(...points.map(p=>p[0]))-9,maxX=Math.max(...points.map(p=>p[0]))+9,minY=Math.min(...points.map(p=>p[1]))-9,maxY=Math.max(...points.map(p=>p[1]))+9;
 const scale=Math.min((canvas.width-40)/(maxX-minX),(canvas.height-40)/(maxY-minY));const center=[(minX+maxX)/2,(minY+maxY)/2];function xy(p){return[canvas.width/2+(p[0]-center[0])*scale,canvas.height/2-(p[1]-center[1])*scale]}
 function geometry(g,fill,stroke,width=1){for(const poly of polygonPaths(g)){ctx.beginPath();for(const ring of poly){ring.forEach((p,i)=>{const q=xy(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.closePath()}ctx.fillStyle=fill;ctx.fill('evenodd');if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.stroke()}}}
 geometry(context.drivable_geojson,'#edf0f2','#b1bbc3');geometry(context.lane_union_geojson,'#e3edf8','#91b0cf');for(const o of context.objects)geometry(o.polygon_geojson,'#78849188','#67747e');
 for(const t of targets){geometry(t.target_polygon,'#ed9a4988','#bf5b12',2);for(const r of t.regions){geometry(r.envelope_geojson,'#1baf6a25','#16814e',1);geometry({type:'Polygon',coordinates:[r.polygon_xy]},'#05a752','#086c3b',2);const a=xy(r.nominal_rear_axle_xy),b=xy([r.nominal_rear_axle_xy[0]+2*Math.cos(r.heading_rad),r.nominal_rear_axle_xy[1]+2*Math.sin(r.heading_rad)]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.strokeStyle='#147644';ctx.lineWidth=2;ctx.stroke()}}
 const c=xy(camera);ctx.fillStyle='#205cbe';ctx.beginPath();ctx.arc(...c,6,0,Math.PI*2);ctx.fill();ctx.font='16px sans-serif';ctx.fillText('相机',c[0]+9,c[1]-7);ctx.fillStyle='#263b4e';ctx.fillText('比例参考 10m',25,canvas.height-20);ctx.beginPath();ctx.moveTo(150,canvas.height-25);ctx.lineTo(150+10*scale,canvas.height-25);ctx.lineWidth=3;ctx.strokeStyle='#263b4e';ctx.stroke();}
async function show(){const serial=++state.serial,row=state.visible[state.index];if(!row){$('queryText').textContent='该筛选无指令';$('counter').textContent='0 / 0';return}
 $('query').value=row.query_id;$('queryText').textContent=row.query_text;$('counter').textContent=`${state.index+1} / ${state.visible.length}`;$('status').textContent='加载证据中';$('rgb').src=row.rgb_url;
 try{let context=null,targets=[];if(row.frame_data_url){const data=await get(row.frame_data_url);context=await get(data.context_url);targets=data.targets.filter(t=>row.accepted_instance_tokens.includes(t.instance_token))}if(serial!==state.serial)return;
 draw(context,targets,row);const titles={certified_static_regions:'有保守静态区域证书；可达性 unknown',unknown_no_certificate:'未找到区域证书；可达性 unknown，不能判不可接近',not_applicable_no_matching_target:'指定官方可见范围无匹配；不等于全道路无目标'};$('status').textContent=titles[row.endpoint_status];
 $('details').textContent=JSON.stringify({query_id:row.query_id,scene:row.scene,frame_idx:row.frame_idx,cohort:row.cohort,accepted_official_instances:row.accepted_instance_tokens,static_regions:targets.reduce((s,t)=>s+t.regions.length,0),rear_axle_position_region_m:'0.4 × 0.2',heading_tolerance_deg:5,additional_body_clearance_m:.1,reference_point_distance_to_target_box_m:[2,8],camera_timestamp_s:row.camera_timestamp_s,official_sample_timestamp_s:context?.official_annotation_timestamp_s??null,sample_minus_camera_s:context?.annotation_minus_camera_s??null,reachability:'unknown',route:null,correct_stable_lock:null,actual_arrival:null,human_verified:false,scope:context?.temporal_scope??'当前官方可见范围；无目标不生成区域'},null,2);
 $('sources').replaceChildren(sourceLink('原始区域标签',row.source_query_url));if(row.frame_data_url){$('sources').append(' · ',sourceLink('当前官方盒与矢量地图',`/annotation/INPUT_RUN/${row.context_path}`))}}
 catch(error){if(serial===state.serial){$('status').textContent='证据读取失败；不提供通过结论';$('details').textContent=String(error)}}}
function filter(){state.visible=state.rows.filter(r=>($('scene').value==='all'||r.scene===$('scene').value)&&($('filter').value==='all'||r.endpoint_status===$('filter').value));state.index=0;$('query').replaceChildren(...state.visible.map(r=>{const o=document.createElement('option');o.value=r.query_id;o.textContent=`${r.query_id}：${r.query_text}`;return o}));show()}
get('index.json').then(data=>{state.rows=data.queries;$('audit').textContent=`${data.total_queries} 条指令，独立区域审计${data.audit_passed?'通过':'尚未通过'}；方法调用 0；真实到达证据 0。`;for(const s of [...new Set(state.rows.map(r=>r.scene))].sort()){const o=document.createElement('option');o.value=s;o.textContent=s;$('scene').append(o)}filter()}).catch(e=>{$('status').textContent=String(e)});
$('scene').addEventListener('change',filter);$('filter').addEventListener('change',filter);$('query').addEventListener('change',()=>{state.index=state.visible.findIndex(r=>r.query_id===$('query').value);show()});$('previous').addEventListener('click',()=>{state.index=Math.max(0,state.index-1);show()});$('next').addEventListener('click',()=>{state.index=Math.min(state.visible.length-1,state.index+1);show()});
</script></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--input-run', default='reference_regions_reserved_full_003')
    parser.add_argument('--audit-id', default='audit_001')
    args = parser.parse_args()
    source = OUTPUT/'annotation'/args.input_run
    summary = json.loads((source/'summary.json').read_text())
    audit = json.loads((source/args.audit_id/'report.json').read_text())
    if not audit['passed'] or audit['generator_summary_sha256'] != digest(source/'summary.json'):
        raise ValueError('need a valid independent audit for these exact region results')
    dest = OUTPUT/'visualizations'/args.run_id
    dest.mkdir(parents=True, exist_ok=False)
    (dest/'data').mkdir()
    original = OUTPUT/'annotation'/json.loads((source/'release.json').read_text())['input_annotation_run']
    queries = {q['query_id']: q for name in ['main_queries.jsonl', 'challenge_queries.jsonl']
               for q in map(json.loads, (original/name).open())}
    frames = {(f['scene'], f['frame_idx']): f for f in map(json.loads, (original/'evaluation_only_frames.jsonl').open())}
    targets = defaultdict(list)
    for record in map(json.loads, (source/'targets.jsonl').open()):
        context = json.loads((source/record['context']['path']).read_text())
        target = next(o for o in context['objects'] if o['instance_token'] == record['instance_token'])
        targets[(record['scene'], record['frame_idx'])].append(dict(instance_token=record['instance_token'],
            target_polygon=target['polygon_geojson'], regions=record['regions'], context_path=record['context']['path']))
    for (scene, frame_idx), records in targets.items():
        dump(dest/f'data/{scene}_{frame_idx:04d}.json', dict(targets=records,
            context_url=f'/annotation/{args.input_run}/{records[0]["context_path"]}'))
    rows = []
    for row in map(json.loads, (source/'queries.jsonl').open()):
        key = row['scene'], row['frame_idx']
        frame, query = frames[key], queries[row['query_id']]
        rgb = Path(frame['rgb'])
        rows.append(dict(query_id=row['query_id'], scene=row['scene'], frame_idx=row['frame_idx'],
            query_text=query['query_text'], cohort=row['cohort'], endpoint_status=row['endpoint_status'],
            accepted_instance_tokens=row['accepted_instance_tokens'], rgb_url='/'+str(rgb.relative_to(OUTPUT)),
            camera_xy=np_camera_xy(frame), camera_timestamp_s=frame['camera_timestamp_s'],
            frame_data_url=f'data/{row["scene"]}_{row["frame_idx"]:04d}.json' if key in targets else None,
            context_path=targets[key][0]['context_path'] if key in targets else None,
            source_query_url=f'/annotation/{args.input_run}/queries.jsonl'))
    dump(dest/'index.json', dict(queries=rows, total_queries=len(rows), audit_passed=audit['passed'],
        source_summary=summary, source_audit=audit, read_only=True, human_records_written=0,
        methods_executed=0, new_test_opened=False))
    (dest/'index.html').write_text(HTML.replace('INPUT_RUN', args.input_run))
    shutil.copy2(__file__, dest/'preview_source.py')
    dump(dest/'manifest.json', dict(source_sha256=digest(__file__),
        source_summary_sha256=digest(source/'summary.json'), source_audit_sha256=digest(source/args.audit_id/'report.json'),
        files={str(p.relative_to(dest)): digest(p) for p in dest.rglob('*') if p.is_file()},
        queries=len(rows), read_only=True, human_records_written=0, model_calls=0, goal_complete=False))
    print(json.dumps(dict(queries=len(rows), input_run=args.input_run, read_only=True), ensure_ascii=False))


def np_camera_xy(frame):
    return [float(frame['camera_pose'][i][3]) for i in range(2)]


if __name__ == '__main__':
    main()
