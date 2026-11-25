'use strict';
const $=id=>document.getElementById(id);
let manifest=null, frames=[], activeScene=null, current=null, generation=0;
let yaw=.35, elevation=.65, zoom=1, dragging=null;
const cache=new Map(), canvas=$('space'), ctx=canvas.getContext('2d');
function sourceURL(path){
  const prefix=manifest.old_output_prefix+'/';
  if(!path.startsWith(prefix))throw Error('原始文件不在已声明的证据目录内');
  return '/'+path.slice(prefix.length);
}
async function getFrames(url){
  if(cache.has(url))return cache.get(url);
  const response=await fetch(url);if(!response.ok)throw Error('读取记录失败：'+response.status);
  if(typeof DecompressionStream==='undefined')throw Error('浏览器缺少gzip读取支持，请使用新版Chrome或Edge');
  const stream=response.body.pipeThrough(new DecompressionStream('gzip'));
  const text=await new Response(stream).text();
  const rows=text.trim().split('\n').filter(Boolean).map(x=>JSON.parse(x));
  if(cache.size>=4)cache.delete(cache.keys().next().value);
  cache.set(url,rows);return rows;
}
async function load(){
  const token=++generation;$('status').className='';$('status').textContent='正在读取所选方案的已有建图记录……';
  try{
    const selected=manifest.scenes.find(x=>x.id===$('scene').value);
    const data=await getFrames(selected.mapping_urls[$('condition').value]);
    if(token!==generation)return;
    if(data.length!==selected.frames)throw Error('记录帧数与清单不一致');
    frames=data;activeScene=selected;$('seek').max=frames.length-1;
    $('seek').value=Math.min(Number($('seek').value),frames.length-1);show();
  }catch(error){$('status').className='bad';$('status').textContent=error.message;}
}
function link(label,url){const a=document.createElement('a');a.textContent=label;a.href=url;a.target='_blank';a.rel='noopener';return a;}
function show(){
  if(!frames.length)return;
  const i=Number($('seek').value);current=frames[i];const preview=activeScene.previews.find(x=>x.frame_idx===current.frame_idx);
  $('counter').textContent=`${i+1} / ${frames.length}（原记录帧号 ${current.frame_idx}）`;
  $('prev').disabled=i===0;$('next').disabled=i===frames.length-1;
  $('rgb').src=sourceURL(current.rgb);
  $('detection').src=sourceURL(current.detections.replace(/[^/]+$/,'detections.jpg'));
  $('depth').src=preview.path;
  $('status').className='';
  $('status').textContent=`${current.scene} · ${current.objects.length} 个保存的对象 · ${current.edges.length} 条关系\n`+
    (preview.depth_available?`本帧有效深度像素：${preview.valid_depth_pixels}`:'本帧没有符合时间要求的LiDAR深度，原始缺测状态已保留。');
  $('state').textContent='本页未读取官方目标答案，也不评价是否选对或实际到达。对象运动状态为系统估计，不是单帧图像真值。';
  $('details').textContent=JSON.stringify({场景:current.scene,帧号:current.frame_idx,原图:current.rgb,原始深度:current.depth,检测与分割:current.detections,对象:current.objects,语义编号:current.semantic_ids,关系:current.edges},null,2);
  $('links').replaceChildren(link('完整建图记录（gzip）',activeScene.mapping_urls[$('condition').value]),document.createTextNode(' · '),link('原始深度PNG',sourceURL(current.depth)),document.createTextNode(' · '),link('检测/掩膜原始数据',sourceURL(current.detections)));
  draw();
}
function draw(){
  const w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);if(!current)return;
  const pose=current.camera_pose, origin=[pose[0][3],pose[1][3],pose[2][3]];
  const heading=Math.atan2(pose[1][2],pose[0][2]);
  function project(p){
    const dx=p[0]-origin[0],dy=p[1]-origin[1],z=p[2]-origin[2];
    const forward=Math.cos(heading)*dx+Math.sin(heading)*dy;
    const left=-Math.sin(heading)*dx+Math.cos(heading)*dy;
    const x=left*Math.cos(yaw)+forward*Math.sin(yaw);
    const y=forward*Math.cos(yaw)-left*Math.sin(yaw);
    return [w/2+x*8*zoom,h*.82-(y*Math.sin(elevation)+z*Math.cos(elevation))*8*zoom];
  }
  function stroke(a,b,color,width=1){const x=project(a),y=project(b);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...x);ctx.lineTo(...y);ctx.stroke();}
  for(let distance=0;distance<=60;distance+=10){
    const a=[origin[0]+distance*Math.cos(heading)-20*Math.sin(heading),origin[1]+distance*Math.sin(heading)+20*Math.cos(heading),origin[2]-1.5];
    const b=[origin[0]+distance*Math.cos(heading)+20*Math.sin(heading),origin[1]+distance*Math.sin(heading)-20*Math.cos(heading),origin[2]-1.5];stroke(a,b,'#e0e7ec');
  }
  const nodes=new Map(current.objects.map(o=>[current.semantic_ids[o.node_id],o]));
  for(const edge of current.edges.slice(0,200)){const a=nodes.get(edge.source),b=nodes.get(edge.target);if(a&&b)stroke(a.center,b.center,'#c8d4dd',.7);}
  const colors={car:'#2d7194',truck:'#946b35',bus:'#367e69',pedestrian:'#835987',bicycle:'#6b7351',motorcycle:'#836050'};
  for(const obj of current.objects){
    const corners=[];for(let x of [-1,1])for(let y of [-1,1])for(let z of [-1,1])corners.push(obj.center.map((v,k)=>v+[x,y,z][k]*obj.extent[k]/2));
    const color=colors[obj.category]||'#566879';
    for(let a=0;a<8;a++)for(let b=a+1;b<8;b++)if([1,2,4].includes(a^b))stroke(corners[a],corners[b],color,1.3);
    const p=project(obj.center);ctx.fillStyle=color;ctx.beginPath();ctx.arc(p[0],p[1],2.2,0,Math.PI*2);ctx.fill();ctx.font='12px sans-serif';ctx.fillText(`${current.semantic_ids[obj.node_id]} ${obj.category}`,p[0]+4,p[1]-4);
  }
  ctx.fillStyle='#44586c';ctx.font='14px sans-serif';ctx.fillText('相机所在位置附近 · 参考网格间距10米',16,24);
}
canvas.addEventListener('pointerdown',e=>{dragging=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(!dragging)return;yaw+=(e.clientX-dragging[0])*.007;elevation=Math.max(.1,Math.min(1.5,elevation+(e.clientY-dragging[1])*.005));dragging=[e.clientX,e.clientY];draw();});
canvas.addEventListener('pointerup',()=>{dragging=null;});
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.35,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
$('scene').addEventListener('change',load);$('condition').addEventListener('change',load);$('seek').addEventListener('input',show);
$('prev').addEventListener('click',()=>{$('seek').value=Math.max(0,Number($('seek').value)-1);show();});
$('next').addEventListener('click',()=>{$('seek').value=Math.min(frames.length-1,Number($('seek').value)+1);show();});
for(const id of ['rgb','detection','depth'])$(id).addEventListener('error',()=>{$('status').className='bad';$('status').textContent+='\n有图像读取失败，请检查HTTP服务或原始证据入口。';});
fetch('manifest.json').then(r=>{if(!r.ok)throw Error('清单读取失败');return r.json();}).then(data=>{
  manifest=data;for(const scene of data.scenes){const option=document.createElement('option');option.value=scene.id;option.textContent=scene.id;$('scene').append(option);}
  for(const item of data.conditions){const option=document.createElement('option');option.value=item.id;option.textContent=item.label;$('condition').append(option);}return load();
}).catch(error=>{$('status').className='bad';$('status').textContent='无法打开：'+error.message+'。请通过HTTP服务访问本页，不能直接双击HTML。';});
