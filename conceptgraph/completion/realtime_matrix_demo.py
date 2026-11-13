"""完整真实时钟矩阵的只读查看器；直接读取冻结原记录，不复制感知或运行模型。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import struct
import time
from urllib.parse import quote

from PIL import Image

from conceptgraph.completion.navigation_demo import HTML as BASE_HTML
from conceptgraph.completion.stage9 import OUTPUT, PROJECT, digest, dump


def source_url(path):
    return '/'+quote(str(Path(path).resolve().relative_to(OUTPUT.resolve())), safe='/')


def read_records(path, *, allow_failed_tail=False):
    records, partial = [], None
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if not allow_failed_tail or index != len(lines)-1:
                raise
            partial = dict(line=index+1, reason='failed_process_incomplete_tail')
    return records, partial


def validate_depth_header(path):
    with path.open('rb') as handle:
        header = handle.read(33)
    if header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError('depth evidence is not PNG')
    width, height, bits, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', header[16:29])
    if (width, height, bits, color, compression, filtering, interlace) != (640, 360, 16, 0, 0, 0, 0):
        raise ValueError('unexpected saved depth format or calibration')


def build_case(record, task, config, source_files, check_budget):
    root = Path(record['raw_case'])
    frames, partial = read_records(root/'perception.jsonl', allow_failed_tail=record['result'] is None)
    execution, execution_partial = read_records(root/'execution.jsonl', allow_failed_tail=record['result'] is None)
    raw_urls = {}
    for name in ['perception.jsonl', 'execution.jsonl', 'result.json', 'semantic_events.json', 'independent_execution_audit.json']:
        path = root/name
        if path.exists():
            source_files[str(path.relative_to(OUTPUT))] = digest(path)
            raw_urls[name] = source_url(path)
    if record['result'] is not None:
        assert digest(root/'result.json') == record['result_sha256']
        assert json.loads((root/'result.json').read_text()) == record['result']
    counts = dict(sensor_RGB_files=0, sensor_depth_files=0, detector_files=0,
                  sensor_evidence_files=0, detection_visualizations=0)
    for frame in frames:
        check_budget()
        assert not frame['gt_sent_to_agent']
        sensor = frame['sensor']
        for name in ['rgb', 'depth']:
            path = Path(sensor[name])
            value = digest(path)
            assert value == sensor[name+'_sha256']
            source_files[str(path.relative_to(OUTPUT))] = value
            counts['sensor_RGB_files' if name == 'rgb' else 'sensor_depth_files'] += 1
        with Image.open(sensor['rgb']) as image:
            assert image.size == (640, 360)
        validate_depth_header(Path(sensor['depth']))
        detector = root/'detections'/f"{frame['frame_idx']:06d}.pkl.gz"
        assert digest(detector) == frame['detector_sha256']
        source_files[str(detector.relative_to(OUTPUT))] = frame['detector_sha256']
        counts['detector_files'] += 1
        evidence = root/'sensor_evidence'/f"{frame['frame_idx']:06d}.json"
        assert json.loads(evidence.read_text()) == sensor
        source_files[str(evidence.relative_to(OUTPUT))] = digest(evidence)
        counts['sensor_evidence_files'] += 1
        visual = root/'detection_vis'/f"{frame['frame_idx']:06d}.jpg"
        assert visual.is_file()
        source_files[str(visual.relative_to(OUTPUT))] = digest(visual)
        counts['detection_visualizations'] += 1
    complete_times = [frame['completed_wall_elapsed_s'] for frame in frames]
    assert all(a < b for a, b in zip(complete_times, complete_times[1:]))
    assert all(a['wall_elapsed_s'] < b['wall_elapsed_s'] for a, b in zip(execution, execution[1:]))
    return dict(run=record['run_id'], method=record['variant'], task=task, seed=record['seed'],
        config={key: config[key] for key in ['config_id', 'road_polygon_xy', 'centerline_xy', 'shared_generator_family']},
        status=record['status'], result=record['result'], raw_urls=raw_urls,
        process_complete=record['result'] is not None, process_timeout=record['process_timeout'],
        frames=len(frames), control_records=len(execution), partial=partial, execution_partial=execution_partial,
        semantic_response_urls=[source_url(path) for path in sorted((root/'semantic_responses').glob('*.json'))],
        raw_prefix_url=source_url(root), input_files_verified=counts,
        raw_perception_sha256=source_files[str((root/'perception.jsonl').relative_to(OUTPUT))],
        raw_execution_sha256=source_files[str((root/'execution.jsonl').relative_to(OUTPUT))])


JAVASCRIPT = r'''
const $=id=>document.getElementById(id), pretty=v=>JSON.stringify(v,null,2);
const esc=v=>String(v??'unknown').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let manifest, data, frames=[], execution=[], frameIndex=0, tickIndex=0, playing=null;
let loadToken=0, renderToken=0, currentView=null, currentDepth={count:0,sample:[],world:[]}, ready=false, yaw=-.65, pitch=.9, drag=null;
const depthCache=new Map();
function fail(error){$('error').textContent=error.message;document.body.dataset.renderReady='error'}
function stop(){clearInterval(playing);playing=null;$('play').textContent='播放保存记录'}
function url(path){const prefix=manifest.output_root+'/';if(!path.startsWith(prefix))throw Error('记录路径越过冻结输出目录');return '/'+path.slice(prefix.length).split('/').map(encodeURIComponent).join('/')}
function parseLines(text,allowPartial){const lines=text.trimEnd().split(/\r?\n/).filter(Boolean),rows=[];for(let i=0;i<lines.length;i++){try{rows.push(JSON.parse(lines[i]))}catch(error){if(!allowPartial||i!==lines.length-1)throw error}}return rows}
function upper(values,value){let lo=0,hi=values.length;while(lo<hi){const mid=(lo+hi)>>1;if(values[mid]<=value)lo=mid+1;else hi=mid}return lo}
async function getText(path,expected){const response=await fetch(path);if(!response.ok)throw Error('无法读取保存记录：'+path);const text=await response.text();if(expected&&globalThis.crypto?.subtle){const actual=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text))),v=>v.toString(16).padStart(2,'0')).join('');if(actual!==expected)throw Error('保存记录指纹不一致')}return text}
async function choose(){
 stop();const token=++loadToken;++renderToken;ready=false;document.body.dataset.caseReady='loading';$('error').textContent='';
 const response=await fetch($('case').value);if(!response.ok)throw Error('试验清单无法读取');const loaded=await response.json();
 const [perception,controls]=await Promise.all([getText(loaded.raw_urls['perception.jsonl'],loaded.raw_perception_sha256),getText(loaded.raw_urls['execution.jsonl'],loaded.raw_execution_sha256)]);
 if(token!==loadToken)return;
 data=loaded;frames=parseLines(perception,!data.process_complete);execution=parseLines(controls,!data.process_complete);
 if(frames.length!==data.frames||execution.length!==data.control_records)throw Error('保存前缀条数与冻结清单不同');
 frameIndex=0;tickIndex=0;depthCache.clear();$('gt').checked=false;
 $('frame').max=Math.max(0,frames.length-1);$('frame').value=0;$('tick').max=Math.max(0,execution.length-1);$('tick').value=0;
 $('query').textContent=data.task.query_text+' · '+data.task.task_id+' · '+data.method+' · seed'+data.seed;
 const labels=[['route_generated','生成路线'],['selected_object_evaluated_correct','曾选对对象'],['stable_correct_lock','曾正确锁定'],['actual_full_perception_arrival','实际到达']];
 $('outcomes').innerHTML=labels.map(([key,label])=>'<span class="outcome">'+label+'：'+(data.result?(data.result[key]?'是':'否'):'未知（缺终止结果）')+'</span>').join('');
 $('status').textContent=data.run+' · '+data.status+'；感知'+frames.length+'帧，控制'+execution.length+'条；'+(data.process_complete?'有保存终止结果':'240秒进程超时，仅展示完整保存前缀，缺结果保留未知');
 const evidence=data.independent_evidence;$('caseEvidence').textContent=evidence?'初态证据：'+(evidence.initial_is_reconstructed?'冻结种子与已保存静止前缀重建；原严格审计缺零时刻观测，未知记录保留':data.process_complete?'已保存零时刻感知状态':'缺完整结果，不能确认全部执行')+'。'+(data.result?.actual_full_perception_arrival?'原报告到达的独立终点保持：'+(evidence.arrival_hold_certificate?'通过':'未确认'):'未报告到达，不增加到达信用'):'本版本没有逐试验初态补充记录';
 $('sources').innerHTML=Object.entries({...data.raw_urls,...(evidence?.proof_urls||{})}).map(([key,value])=>'<a href="'+value+'" target="_blank">'+esc(key)+'</a>').join('');
 $('semanticLinks').innerHTML='<span class="muted">全试验模型响应（包括后续响应）：</span> '+data.semantic_response_urls.map((value,index)=>'<a href="'+value+'" target="_blank">响应'+(index+1)+'</a>').join('');
 ready=true;document.body.dataset.caseReady=data.run;await render();
}
function filter(){
 stop();const f=$('filter').value,m=$('method').value,c=$('config').value,s=$('seed').value;
 const rows=manifest.cases.filter(row=>(m==='all'||row.method===m)&&(c==='all'||row.config_id===c)&&(s==='all'||String(row.seed)===s)&&(f==='all'||f==='arrived'&&row.arrived||f==='failed'&&(row.result_missing||row.kind==='reachable_positive'&&!row.arrived)||row.kind===f));
 $('matchCount').textContent='匹配'+rows.length+' / '+manifest.cases.length+'次试验';
 $('case').innerHTML=rows.map(row=>'<option value="'+row.file+'">'+esc(row.method)+' / '+esc(row.task_id)+' / seed'+row.seed+' / '+esc(row.status)+'</option>').join('');
 if(rows.length)choose().catch(fail);else {++loadToken;++renderToken;ready=false;data=null;frames=[];execution=[];currentView=null;$('query').textContent='';$('outcomes').innerHTML='';$('status').textContent='';$('nodes').textContent='';$('candidates').textContent='';for(const id of ['orders','relations','semantics','execution','finalResult','sensor','semanticLinks','sources','frameLabel','tickLabel','caseEvidence'])$(id).textContent='';for(const id of ['rgb','space'])$(id).getContext('2d').clearRect(0,0,$(id).width,$(id).height);$('det').hidden=true;$('play').disabled=true;$('error').textContent='当前筛选没有记录';document.body.dataset.caseReady='empty'}
}
function paeth(a,b,c){const p=a+b-c,pa=Math.abs(p-a),pb=Math.abs(p-b),pc=Math.abs(p-c);return pa<=pb&&pa<=pc?a:pb<=pc?b:c}
async function decodeDepthPng(buffer){
 const bytes=new Uint8Array(buffer),view=new DataView(buffer);if(bytes.length<33||view.getUint32(0)!==0x89504e47||view.getUint32(4)!==0x0d0a1a0a)throw Error('深度图不是PNG');
 let width=0,height=0,parts=[],size=0;
 for(let offset=8;offset+12<=bytes.length;){const length=view.getUint32(offset),type=String.fromCharCode(...bytes.subarray(offset+4,offset+8));if(offset+12+length>bytes.length)throw Error('深度PNG数据截断');
  if(type==='IHDR'){width=view.getUint32(offset+8);height=view.getUint32(offset+12);if(bytes[offset+16]!==16||bytes[offset+17]!==0||bytes[offset+18]!==0||bytes[offset+19]!==0||bytes[offset+20]!==0)throw Error('仅接受无交错16位灰度深度PNG')}
  if(type==='IDAT'){parts.push(bytes.slice(offset+8,offset+8+length));size+=length}offset+=12+length;if(type==='IEND')break;
 }
 if(!width||!height||width*height>2000000)throw Error('深度PNG尺寸异常');
 const compressed=new Uint8Array(size);let offset=0;for(const part of parts){compressed.set(part,offset);offset+=part.length}
 const raw=new Uint8Array(await new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer());
 const stride=width*2;if(raw.length!==(stride+1)*height)throw Error('深度PNG解压尺寸错误');const pixels=new Uint16Array(width*height);let previous=new Uint8Array(stride);
 for(let y=0;y<height;y++){const filter=raw[y*(stride+1)],line=raw.slice(y*(stride+1)+1,(y+1)*(stride+1));if(filter>4)throw Error('非法PNG行过滤器');
  for(let x=0;x<stride;x++){const a=x>=2?line[x-2]:0,b=previous[x],c=x>=2?previous[x-2]:0;line[x]=(line[x]+(filter===0?0:filter===1?a:filter===2?b:filter===3?Math.floor((a+b)/2):paeth(a,b,c)))&255}
  for(let x=0;x<width;x++)pixels[y*width+x]=(line[2*x]<<8)|line[2*x+1];previous=line;
 }
 return {width,height,pixels};
}
async function sparseDepth(frame){
 const path=url(frame.sensor.depth);if(depthCache.has(path))return depthCache.get(path);
 const response=await fetch(path);if(!response.ok)throw Error('实测深度文件未能读取');const decoded=await decodeDepthPng(await response.arrayBuffer());
 if(decoded.width!==manifest.calibration.width||decoded.height!==manifest.calibration.height)throw Error('深度图尺寸与标定不同');
 const nonzero=[];for(let i=0;i<decoded.pixels.length;i++)if(decoded.pixels[i])nonzero.push(i);
 const k=Math.min(384,nonzero.length),sample=[],world=[],pose=frame.sensor.camera_pose_cv,cal=manifest.calibration;
 for(let i=0;i<k;i++){const index=nonzero[k===1?0:Math.floor(i*(nonzero.length-1)/(k-1))],u=index%decoded.width,v=Math.floor(index/decoded.width),z=decoded.pixels[index]/1000;
  sample.push([u,v,z]);const camera=[(u-cal.cx)*z/cal.fx,(v-cal.cy)*z/cal.fy,z];world.push(pose.slice(0,3).map(row=>row[3]+row[0]*camera[0]+row[1]*camera[1]+row[2]*camera[2]));
 }
 if(nonzero.length!==frame.sensor.lidar_pixels)throw Error('有效深度像素数不符');const result={count:nonzero.length,sample,world};depthCache.set(path,result);if(depthCache.size>3)depthCache.delete(depthCache.keys().next().value);return result;
}
function loadImage(path){return new Promise((resolve,reject)=>{const image=new Image;image.onload=()=>resolve(image);image.onerror=()=>reject(Error('保存图像无法读取：'+path));image.src=path})}
function displayFrame(){
 const control=$('mode').value==='control';$('frame').disabled=control||!frames.length;$('tick').disabled=!control||!execution.length;
 if(control){const row=execution[tickIndex];if(!row)return {frame:null,state:null,prefix:0,index:-1,control:true};
  const index=upper(frames.map(frame=>frame.completed_wall_elapsed_s),row.wall_elapsed_s)-1;
  return {frame:index>=0?frames[index]:null,state:row.state,prefix:tickIndex+1,index,control:true,row};
 }
 const frame=frames[frameIndex];return {frame:frame||null,state:frame?.sensor.state||null,prefix:frame?upper(execution.map(row=>row.state.time_s),frame.sensor.state.time_s):0,index:frameIndex,control:false};
}
async function render(){
 if(!ready)return;const token=++renderToken;document.body.dataset.renderReady='loading';const v=displayFrame();currentView=v;
 $('frameLabel').textContent=v.control?'控制记录'+(tickIndex+1)+' / '+execution.length+' · 模型时间'+(v.state?.time_s.toFixed(1)||'未知')+'秒':'感知帧'+(v.frame?.frame_idx??'未知')+' / '+frames.length+' · 观测模型时间'+(v.state?.time_s.toFixed(1)||'未知')+'秒';
 $('tickLabel').textContent=v.control?'墙钟'+v.row.wall_elapsed_s.toFixed(3)+'秒；只显示此前已完成的感知':'切换实际控制状态可检查感知帧之间及末尾制动';
 $('play').disabled=v.control?!execution.length:!frames.length;
 const frame=v.frame;let depth={count:0,sample:[],world:[]};
 if(frame){const [image,decoded]=await Promise.all([loadImage(url(frame.sensor.rgb)),sparseDepth(frame)]);if(token!==renderToken)return;depth=decoded;
  const context=$('rgb').getContext('2d');context.clearRect(0,0,640,360);context.drawImage(image,0,0,640,360);
  if($('lidar').checked)for(const [u,w,z] of depth.sample){context.fillStyle='hsl(202 90% '+(20+Math.min(1,z/65)*65)+'%)';context.fillRect(u-1.3,w-1.3,2.6,2.6)}
  const detector=data.raw_prefix_url+'/detection_vis/'+String(frame.frame_idx).padStart(6,'0')+'.jpg';const detected=await loadImage(detector);if(token!==renderToken)return;$('det').src=detected.src;$('det').hidden=false;
  $('sensor').textContent='保存观测'+frame.frame_idx+'：'+depth.count+'个有效深度像素，浏览抽样'+depth.sample.length+'点；观测墙钟'+frame.sensed_wall_elapsed_s.toFixed(3)+'秒，处理完成'+frame.completed_wall_elapsed_s.toFixed(3)+'秒。'+(v.control?'此处是该控制时刻最近已完成的感知，不能还原未记录的中间发布。':'');
 }else{const ctx=$('rgb').getContext('2d');ctx.clearRect(0,0,640,360);ctx.fillStyle='#314f63';ctx.font='19px sans-serif';ctx.fillText('该时刻尚无已完成感知',40,170);$('det').hidden=true;$('det').removeAttribute('src');$('sensor').textContent='仅显示已保存控制状态；没有补造当时的RGB、深度或对象。'}
 const nodes=frame?.nodes||[];$('nodes').innerHTML=nodes.map(node=>{const m=node.metadata,identity=$('gt').checked?(frame.evaluation_only_matches[String(node.node_id)]||'unknown'):'隔离评估层隐藏';return '<div class="node"><b>节点'+node.node_id+' · '+esc(node.category)+' · '+esc(node.motion)+'</b><br><span class="muted">'+esc(m.object_identity)+' / 评价关联：'+esc(identity)+'</span><br>中心 '+node.center.map(value=>value.toFixed(2)).join(', ')+' m<br>'+(m.crop_paths||[]).slice(-2).map(path=>'<a href="'+url(path)+'" target="_blank"><img src="'+url(path)+'" alt="当时保存的真实crop"></a>').join('')+'<details><summary>深度、属性和来源</summary><pre>'+esc(pretty({depth:m.depth_evidence,attributes:m.attributes}))+'</pre></details></div>'}).join('')||'该时刻没有可显示的已处理对象';
 const candidates=frame?.candidates||[];$('candidates').innerHTML='<table><tr><th>节点</th><th>保存的联合分数</th></tr>'+candidates.map(candidate=>'<tr><td>'+candidate.node_id+'</td><td>'+(Number.isFinite(candidate.score)?candidate.score.toFixed(4):'unknown')+'</td></tr>').join('')+'</table>';
 const audit=frame?.semantic_audit;$('orders').textContent=pretty({rule_order:audit?.rule_order??null,safety_verified_order:audit?.safety_verified_order??null,final_order:audit?.final_order??null,changed_order:audit?.changed_order??null,changed_final_first:audit?.changed_final_first??null});
 $('relations').textContent=pretty(frame?.relations??[]);$('semantics').textContent=pretty({parsed:frame?.parsed_query??null,lock:frame?.lock??null,planning:frame?.planning??null,semantic_audit:audit??null});
 $('execution').textContent=pretty({mode:v.control?'保存控制状态':'保存感知观测',observed_sensor_state:frame?.sensor.state??null,displayed_vehicle_state:v.state,control_prefix_length:v.prefix,last_saved_control:v.prefix?execution[v.prefix-1]:null,current_evaluation_identity:$('gt').checked?frame?.evaluation_only_identity??null:'隔离评估层隐藏'});
 $('finalResult').textContent=pretty(data.result??{status:data.status,terminal_outcomes:'unknown_missing_complete_result',process_timeout:data.process_timeout});
 currentDepth=depth;drawSpace(v,depth);$('error').textContent='';document.body.dataset.renderReady='ready';
}
function drawSpace(view,depth){
 const canvas=$('space'),w=Math.max(1,canvas.clientWidth),h=Math.max(1,canvas.clientHeight),ratio=window.devicePixelRatio||1;
 if(canvas.width!==Math.round(w*ratio)||canvas.height!==Math.round(h*ratio)){canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio)}
 const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);const road=data.config.road_polygon_xy,center=data.config.centerline_xy,mx=center.reduce((s,p)=>s+p[0],0)/center.length,my=center.reduce((s,p)=>s+p[1],0)/center.length;
 const raw=p=>{const x=p[0]-mx,y=p[1]-my,z=p[2]||0;return [Math.cos(yaw)*x-Math.sin(yaw)*y,Math.sin(pitch)*(Math.sin(yaw)*x+Math.cos(yaw)*y)-Math.cos(pitch)*z]};
 const corners=road.map(raw),xs=corners.map(p=>p[0]),ys=corners.map(p=>p[1]),local=$('spaceView').value==='vehicle'&&view.state;
 const localCenter=local?raw([view.state.x,view.state.y,0]):null,scale=local?Math.min(w-40,h-50)/18:Math.min((w-120)/(Math.max(...xs)-Math.min(...xs)+4),(h-80)/(Math.max(...ys)-Math.min(...ys)+8)),cx=local?localCenter[0]:(Math.max(...xs)+Math.min(...xs))/2,cy=local?localCenter[1]:(Math.max(...ys)+Math.min(...ys))/2;
 const project=p=>{const q=raw(p);return [w/2+(q[0]-cx)*scale,h/2+(q[1]-cy)*scale]};
 function line(points,color,width=1,fill=null){if(!points?.length)return;ctx.beginPath();points.forEach((p,i)=>{const q=project(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});if(fill){ctx.closePath();ctx.fillStyle=fill;ctx.fill()}ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
 line(road,'#7a9eb3',1,'#dfebf2');line(center,'#8496a1',1);const frame=view.frame;
 for(const obstacle of frame?.known_obstacle_polygons||[])line(obstacle,'#8899a3',.7,'#d1dade');
 for(const point of depth.world){const q=project(point);ctx.fillStyle='#3983a780';ctx.fillRect(q[0],q[1],1.5,1.5)}
 line(frame?.path_world||[],'#c77730',3);line(execution.slice(0,view.prefix).map(row=>[row.state.x,row.state.y]),'#156f61',3);
 for(const node of frame?.nodes||[]){const box=[];for(const z of [-1,1])for(const [x,y] of [[-1,-1],[1,-1],[1,1],[-1,1]])box.push([node.center[0]+x*node.extent[0]/2,node.center[1]+y*node.extent[1]/2,node.center[2]+z*node.extent[2]/2]);const color=node.node_id===frame.selected_node_id?'#9d571d':'#335e82';line([...box.slice(0,4),box[0]],color);line([...box.slice(4),box[4]],color);for(let i=0;i<4;i++)line([box[i],box[i+4]],color);const p=project(node.center);ctx.fillStyle=color;ctx.font='14px sans-serif';ctx.fillText(node.node_id+': '+node.category,p[0],p[1]-5)}
 if(view.state){const s=view.state,spec=manifest.vehicle_spec,front=spec.front_from_rear_axle_m,rear=spec.rear_overhang_m,half=spec.width_m/2;const body=[[-rear,-half],[front,-half],[front,half],[-rear,half],[-rear,-half]].map(([x,y])=>[s.x+Math.cos(s.yaw_rad)*x-Math.sin(s.yaw_rad)*y,s.y+Math.sin(s.yaw_rad)*x+Math.cos(s.yaw_rad)*y,.15]);line(body,'#a33d36',2,'#a33d3630')}
 if($('gt').checked)for(const region of data.task.regions)line(region.polygon_xy,'#7852a0',2,'#7852a025');ctx.fillStyle='#203d50';ctx.font='14px sans-serif';ctx.fillText('世界坐标（m） · 保存计划/实际轨迹 · '+(view.control?'控制状态':'感知观测'),18,26);
}
for(const id of ['filter','method','config','seed'])$(id).onchange=filter;
$('case').onchange=()=>choose().catch(fail);$('mode').onchange=()=>{stop();render().catch(fail)};
$('frame').oninput=()=>{frameIndex=Number($('frame').value);render().catch(fail)};$('tick').oninput=()=>{tickIndex=Number($('tick').value);render().catch(fail)};
$('gt').onchange=()=>render().catch(fail);$('lidar').onchange=()=>render().catch(fail);
$('play').onclick=()=>{if(playing){stop();return}if(!ready)return;$('play').textContent='暂停';playing=setInterval(()=>{const control=$('mode').value==='control';if(control){if(tickIndex>=execution.length-1){stop();return}$('tick').value=++tickIndex}else{if(frameIndex>=frames.length-1){stop();return}$('frame').value=++frameIndex}render().catch(error=>{stop();fail(error)})},$('mode').value==='control'?200:700)};
$('space').onpointerdown=e=>{drag=[e.clientX,e.clientY];$('space').setPointerCapture(e.pointerId)};$('space').onpointermove=e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.006;pitch=Math.max(.15,Math.min(1.5,pitch+(e.clientY-drag[1])*.006));drag=[e.clientX,e.clientY];render().catch(fail)};$('space').onpointerup=()=>drag=null;
window.addEventListener('resize',()=>{if(ready&&currentView)drawSpace(currentView,currentDepth)});
$('spaceView').onchange=()=>{if(ready&&currentView)drawSpace(currentView,currentDepth)};
window.MATRIX_VIEW={decodeDepthPng,get snapshot(){return {run:data?.run,method:data?.method,mode:$('mode').value,frameIndex,tickIndex,perceptionFrame:currentView?.frame?.frame_idx??null,perceptionCompletedWall:currentView?.frame?.completed_wall_elapsed_s??null,displayedState:currentView?.state??null,prefix:currentView?.prefix??0,controlWall:currentView?.row?.wall_elapsed_s??null,ready:document.body.dataset.renderReady==='ready',GT:$('gt').checked}}};
fetch('manifest.json').then(response=>{if(!response.ok)throw Error('请通过核验服务打开此页面');return response.json()}).then(value=>{manifest=value;$('method').innerHTML='<option value="all">全部方法</option>'+manifest.methods.map(value=>'<option>'+esc(value)+'</option>').join('');$('config').innerHTML='<option value="all">全部配置</option>'+manifest.configs.map(value=>'<option>'+esc(value)+'</option>').join('');$('seed').innerHTML='<option value="all">全部种子</option>'+manifest.seeds.map(value=>'<option>'+value+'</option>').join('');filter()}).catch(fail);
'''


def html_document(statistics):
    page = BASE_HTML.split('<script>', 1)[0]
    page = page.replace('</style>', '.links{display:flex;flex-wrap:wrap;gap:8px 15px;overflow-wrap:anywhere}.links a{margin:0;max-width:100%;overflow-wrap:anywhere}</style>', 1)
    page = page.replace('感知导航证据查看', '三模型450次开发实验回放')
    page = page.replace('这里回放真实控制及传感器日志；', '终止汇总与当时观测分开显示。这里回放真实控制及传感器日志；')
    page = page.replace('<main>', '<main><section class="card"><h2>完整开发对比，所有失败保留</h2>'
        '<p>450次冻结试验，449个完整结果、1次进程超时。10配置共享3道路家族；每模型90次可达正试验。'
        '这是合成颜色开发任务，尚非自然指令封存测试。新Adapter的负结果保留。</p>'
        '<div style="overflow-x:auto"><table><tr><th>方法</th><th>生成路线</th><th>曾选对对象</th><th>曾正确锁定</th><th>实际到达</th></tr>'
        +''.join('<tr><td>'+method+'</td>'+''.join('<td>'+str(int(report['by_kind']['reachable_positive']['four_outcomes'][field]['numerator']))+'/90</td>'
        for field in ['route_generated', 'selected_object_evaluated_correct', 'stable_correct_lock', 'actual_full_perception_arrival'])+'</tr>'
        for method, report in statistics['methods'].items())+'</table></div>'
        '<p class="muted">93原报告到达获得分版本独立车身、方向及2秒终点保持核验。'
        '54例初态另由冻结种子及保存静止前缀重建，原严格未知保留，不称新增零时刻观测。'
        '轨迹、渲染、帧和复核均不增加独立任务或到达。</p></section>', 1)
    page = page.replace('<label>筛选 ', '<label>方法 <select id="method"></select></label><label>配置 <select id="config"></select></label><label>种子 <select id="seed"></select></label><label>筛选 ', 1)
    page = page.replace('<p id="query">', '<p id="matchCount" class="muted"></p><p id="query">', 1)
    page = page.replace('<p id="error"', '<p id="caseEvidence" class="muted"></p><p id="error"', 1)
    page = page.replace('<canvas id="space"', '<div class="bar"><label>空间范围 <select id="spaceView"><option value="scene">整段道路</option><option value="vehicle">车辆附近</option></select></label></div><canvas id="space"', 1)
    page = page.replace('<button id="play">播放保存帧</button>', '<label>时间轴 <select id="mode"><option value="observation">感知观测</option><option value="control">实际控制状态</option></select></label><button id="play">播放保存记录</button>', 1)
    page = page.replace('<input id="frame" aria-label="感知帧" type="range" min="0" value="0" max="0">',
        '<input id="frame" aria-label="感知帧" type="range" min="0" value="0" max="0">'
        '<p id="tickLabel" class="muted"></p><input id="tick" aria-label="实际控制状态" type="range" min="0" value="0" max="0" disabled>', 1)
    page = page.replace('<details><summary>关系链和证据</summary>', '<details open><summary>规则顺序、安全验证后及语义最终顺序</summary><pre id="orders"></pre></details><details><summary>关系链和证据</summary>', 1)
    page = page.replace('<div id="sources" class="links"></div>', '<details><summary>整场终止结果（不是当前帧）</summary><pre id="finalResult"></pre></details><div id="sources" class="links"></div>'
        '<p class="links"><a href="/主矩阵450_实验核验交接_001.md">完整核验与限制</a>'
        '<a href="/statistics/realtime_three_model_full_001/trials.csv">全部450逐试验结果</a></p>', 1)
    return page+'<script>'+JAVASCRIPT+'</script></html>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--reuse-viewer')
    parser.add_argument('--wall-budget-s', type=float, default=900.)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError('limit must be positive')
    started = time.monotonic()
    root = OUTPUT/'interactive_simulation/realtime_three_model_full_001'
    release = json.loads((root/'release.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    assert summary['attempted'] == summary['expected'] == 450
    assert digest(root/'trial_records.jsonl') == summary['trial_records_sha256']
    records, partial = read_records(root/'trial_records.jsonl')
    assert partial is None and len(records) == len(release['trials']) == 450
    assert [row['run_id'] for row in records] == [row['run_id'] for row in release['trials']]
    tasks_path = OUTPUT/'interactive_simulation/synthetic_tasks_v1.json'
    task_release = json.loads(tasks_path.read_text())
    assert digest(tasks_path) == release['task_release_sha256']
    tasks = {task['task_id']: task for task in task_release['tasks']}
    configs = {config['config_id']: config for config in task_release['configs']}
    stats_path = OUTPUT/'statistics/realtime_three_model_full_001/report.json'
    statistics = json.loads(stats_path.read_text())
    assert statistics['source_records_sha256'] == summary['trial_records_sha256']
    body_root = OUTPUT/'vehicle_execution/independent_body_main_seed_initial_001'
    body_records, _ = read_records(body_root/'cases.jsonl')
    hold_root = OUTPUT/'vehicle_execution/main_goal_hold_evidence_check_002'
    hold_records, _ = read_records(hold_root/'cases.jsonl')
    body = {(row['method'], row['task'], row['seed']): row for row in body_records}
    holds = {(row['variant'], row['task'], row['seed']): row for row in hold_records}
    assert len(body) == len(holds) == 450
    destination = OUTPUT/'visualizations'/args.run_id
    destination.mkdir(exist_ok=False)
    (destination/'cases').mkdir()
    source_paths = [Path(__file__).resolve(), PROJECT/'conceptgraph/completion/navigation_demo.py',
                    PROJECT/'conceptgraph/completion/interactive_environment.py']
    frozen = {str(path.relative_to(PROJECT)): digest(path) for path in source_paths}
    dump(destination/'release.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=frozen,
        main_release_sha256=digest(root/'release.json'), main_records_sha256=summary['trial_records_sha256'],
        selected_limit=args.limit, selection='原冻结顺序前limit或全部450，不选择方法成功子集',
        wall_budget_s=args.wall_budget_s, storage_budget_bytes=128*1024**2, minimum_free_bytes=6*1024**3,
        CUDA_visible='', shared_CPU=True, actual_model_calls=0, new_executions=0, new_arrivals=0, goal_complete=False))
    sources = {str(path.relative_to(OUTPUT)): digest(path) for path in [root/'release.json', root/'summary.json', root/'trial_records.jsonl', tasks_path, stats_path, body_root/'cases.jsonl', body_root/'report.json', hold_root/'cases.jsonl', hold_root/'report.json']}
    previous = None
    if args.reuse_viewer:
        previous_root = OUTPUT/'visualizations'/args.reuse_viewer
        previous = json.loads((previous_root/'manifest.json').read_text())
        old_release = json.loads((previous_root/'release.json').read_text())
        assert old_release['main_records_sha256'] == summary['trial_records_sha256']
        assert previous['task_release_sha256'] == digest(tasks_path)
        assert len(previous['cases']) == len(records[:args.limit])
        assert digest(previous_root/'generator_source.py') == old_release['source_sha256'][str(Path(__file__).resolve().relative_to(PROJECT))]
        for name, value in previous['files'].items():
            assert digest(previous_root/name) == value
        sources.update(json.loads((previous_root/'input_files.json').read_text()))
        sources[str((previous_root/'manifest.json').relative_to(OUTPUT))] = digest(previous_root/'manifest.json')
    rows, files, totals = [], {}, {}

    def check_budget():
        if time.monotonic()-started > args.wall_budget_s or shutil.disk_usage(OUTPUT).free < 6*1024**3:
            raise RuntimeError('viewer preparation budget exhausted')

    try:
        for index, record in enumerate(records[:args.limit]):
            check_budget()
            task = tasks[record['task']]
            assert all(digest(PROJECT/path) == value for path, value in frozen.items())
            if previous:
                old_row = previous['cases'][index]
                assert old_row['run'] == record['run_id']
                data = json.loads((previous_root/old_row['file']).read_text())
            else:
                data = build_case(record, task, configs[record['config_id']], sources, check_budget)
            key = record['variant'], record['task'], record['seed']
            proof_name = f"{index:04d}_{record['task']}_seed{record['seed']}_{record['variant']}"
            proof_urls = {label: source_url(path) for label, path in {
                '严格初态审核': OUTPUT/'vehicle_execution/independent_body_main_full_001'/proof_name/'report.json',
                '分版本车身审核': body_root/proof_name/'report.json',
                '分版本方向审核': OUTPUT/'vehicle_execution/execution_direction_main_seed_initial_001'/proof_name/'report.json'}.items()}
            for path in proof_urls.values():
                assert path.startswith('/vehicle_execution/')
            data['independent_evidence'] = dict(initial_is_reconstructed=body[key]['initial_state_provenance'] == 'frozen_seed_reconstruction_and_saved_stationary_prefix',
                initial_state_provenance=body[key]['initial_state_provenance'], strict_status=body[key]['original_strict_status'],
                arrival_hold_certificate=holds[key].get('hold_certificate'), proof_urls=proof_urls)
            name = f'cases/{index:04d}.json'
            dump(destination/name, data)
            files[name] = digest(destination/name)
            rows.append(dict(index=index, run=record['run_id'], method=record['variant'],
                task_id=record['task'], config_id=record['config_id'], kind=record['kind'], seed=record['seed'],
                status=record['status'], arrived=record['actual_arrival'], result_missing=record['result'] is None,
                frames=data['frames'], control_records=data['control_records'], file=name))
            for key, value in data['input_files_verified'].items():
                totals[key] = totals.get(key, 0)+value
            print(json.dumps(dict(case=index+1, frames=data['frames'], status=record['status'])), flush=True)
        (destination/'index.html').write_text(html_document(statistics))
        shutil.copy2(__file__, destination/'generator_source.py')
        for name in ['index.html', 'generator_source.py', 'release.json']:
            files[name] = digest(destination/name)
        dump(destination/'input_files.json', sources)
        files['input_files.json'] = digest(destination/'input_files.json')
        assert sum(path.stat().st_size for path in destination.rglob('*') if path.is_file()) < 128*1024**2
        dump(destination/'manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
            cases=rows, files=files, methods=sorted({row['method'] for row in rows}),
            configs=sorted({row['config_id'] for row in rows}), seeds=task_release['seeds'],
            output_root=str(OUTPUT.resolve()), task_release_sha256=digest(tasks_path), vehicle_spec=task_release['vehicle'],
            calibration=dict(width=640, height=360, fx=320, fy=320, cx=320, cy=180),
            sensor_input_totals=totals, verified_source_files=len(sources), input_manifest_sha256=digest(destination/'input_files.json'),
            verified_input_manifest_reused_from=args.reuse_viewer,
            perception_records=sum(row['frames'] for row in rows), control_records=sum(row['control_records'] for row in rows),
            is_smoke=args.limit is not None, expected_full_cases=450, depth_display='原始16位毫米PNG，零值无效，浏览均匀抽样至多384点',
            control_timeline='只显示处理完成墙钟不晚于该控制记录的感知，不推断未保存中间发布',
            wall_CPU_shared_s=time.monotonic()-started, actual_model_calls=0, new_executions=0, new_arrivals=0,
            independent_test=False, new_test_opened=False, goal_complete=False))
        print(json.dumps(dict(cases=len(rows), verified_source_files=len(sources), sensor_files=totals,
                             goal_complete=False), ensure_ascii=False), flush=True)
    except BaseException as error:
        dump(destination/'interrupted.json', dict(error=repr(error), cases=len(rows), goal_complete=False))
        raise


if __name__ == '__main__':
    main()
