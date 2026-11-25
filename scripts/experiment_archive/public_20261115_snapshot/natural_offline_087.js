'use strict';
const DATA=JSON.parse(document.getElementById('embedded').textContent);
const $=id=>document.getElementById(id);
const KEY='thesis_natural_offline_087_'+DATA.campaign_sha256;
let rows=[],scene=null,frame=null,point=null,img=null,epoch=0,seen=[];
const norm=s=>s.normalize('NFKC').toLocaleLowerCase().replace(/\s/g,'');
const message=s=>{$('status').textContent=s;};
function check(r){
 if(!r||r.campaign!==DATA.campaign||r.campaign_sha256!==DATA.campaign_sha256)throw Error('不是本采集批次的记录');
 let s=DATA.cards.find(c=>c.scene===r.scene),f=s&&s.frames.find(x=>x.frame_idx===r.frame_idx);
 if(!f||r.camera_token!==f.camera_token||r.rgb_sha256!==f.rgb_sha256)throw Error('场景或图片标识不匹配');
 for(const [key,min,max] of [['author',1,120],['command',4,1000],['note',1,2000]]){
  if(typeof r[key]!=='string'||r[key].trim().length<min||r[key].trim().length>max)throw Error('请填写昵称、至少4字的指令和观察依据');
 }
 if(norm(r.command).length<4)throw Error('指令内容过短');
 if(r.human_authorship_attested!==true)throw Error('请勾选本人看图后自行撰写');
 if(!['zh','en','mixed','other'].includes(r.language)||!['unknown','approach_static_target','no_match','inaccessible','other'].includes(r.author_intent))throw Error('语言或意图无效');
 const p=r.author_target_point_normalized;
 if(p!==null&&(!Array.isArray(p)||p.length!==2||p.some(n=>typeof n!=='number'||!Number.isFinite(n)||n<0||n>1)))throw Error('点击坐标不在图内');
}
function counts(){
 const n=rows.filter(r=>r.scene===scene?.scene).length;
 $('count').textContent=`本机已保存 ${rows.length} 条（当前场景 ${n} 条），尚未发送到服务器`;
}
function persist(){
 try{localStorage.setItem(KEY,JSON.stringify(rows));$('storage').textContent='已存入本机浏览器；请导出JSON备份并发回聊天。';}
 catch(e){$('storage').textContent='浏览器不允许缓存：当前记录只在本页内存，请立即导出JSON，关闭前务必备份。';}
 counts();
}
function draw(){if(!img)return;let c=$('rgb'),ctx=c.getContext('2d');ctx.drawImage(img,0,0);if(point){ctx.beginPath();ctx.arc(point[0]*c.width,point[1]*c.height,13,0,Math.PI*2);ctx.lineWidth=4;ctx.strokeStyle='#f04421';ctx.stroke();}}
function frameChanged(){
 frame=scene.frames[Number($('frame').value)];point=null;img=null;$('submit').disabled=true;
 const e=++epoch,cur=new Image;
 $('frameInfo').textContent=`${scene.scene} · 原始帧号 ${frame.frame_idx} · 试用片段 ${Number($('frame').value)+1}/${scene.frames.length} · 此场景完整序列 ${scene.total_original_frames} 帧`;
 cur.onload=()=>{if(e!==epoch)return;img=cur;$('rgb').width=cur.naturalWidth;$('rgb').height=cur.naturalHeight;seen.push({scene:scene.scene,frame_idx:frame.frame_idx,viewed_at:new Date().toISOString()});draw();$('submit').disabled=false;};
 cur.onerror=()=>{if(e===epoch)message('内嵌图片加载失败，请重新下载完整HTML文件。');};cur.src=frame.image_data;
}
function sceneChanged(){scene=DATA.cards.find(s=>s.scene===$('scene').value);$('frame').max=scene.frames.length-1;$('frame').value=0;frameChanged();counts();}
$('scene').onchange=sceneChanged;$('frame').oninput=frameChanged;
$('previous').onclick=()=>{$('frame').value=Math.max(0,Number($('frame').value)-1);frameChanged();};
$('next').onclick=()=>{$('frame').value=Math.min(scene.frames.length-1,Number($('frame').value)+1);frameChanged();};
$('rgb').onclick=e=>{if(!img)return;const b=$('rgb').getBoundingClientRect();point=[Math.max(0,Math.min(1,(e.clientX-b.left)/b.width)),Math.max(0,Math.min(1,(e.clientY-b.top)/b.height))];draw();};
$('clearPoint').onclick=()=>{point=null;draw();};
$('submit').onclick=()=>{
 if(!img)return;
 const r={campaign:DATA.campaign,campaign_sha256:DATA.campaign_sha256,scene:scene.scene,
 frame_idx:frame.frame_idx,camera_token:frame.camera_token,rgb_sha256:frame.rgb_sha256,
 author:$('author').value.trim(),command:$('command').value.trim(),note:$('note').value.trim(),
 language:$('language').value,author_intent:$('intent').value,human_authorship_attested:$('attest').checked,
 author_target_point_normalized:point,submitted_at:new Date().toISOString(),collection_mode:'offline_087',
 observed_frames_before_submission:seen.slice(),qualifies_for_sealed_test:false};
 try{check(r);if(rows.some(x=>x.scene===r.scene&&norm(x.command)===norm(r.command)))throw Error('同场景这条指令已经保存，请勿重复');
 rows.push(r);persist();message(`第 ${rows.length} 条已保存在本机。完成后请导出JSON发回聊天。`);
 $('command').value='';$('note').value='';$('attest').checked=false;
 }catch(e){message('未保存：'+e.message);}
};
$('export').onclick=()=>{
 if(!rows.length){message('还没有记录，请先填写并保存。');return;}
 const data={format:DATA.export_format,campaign:DATA.campaign,campaign_sha256:DATA.campaign_sha256,
 exported_at:new Date().toISOString(),records:rows,qualifies_for_sealed_test:false,
 label_status:'仅为作者原始表达，目标、运动与歧义尚待独立核验'};
 const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json;charset=utf-8'}));
 const a=document.createElement('a');a.href=url;a.download='自然导航指令_人工填写_'+new Date().toISOString().replace(/[:.]/g,'-')+'.json';
 document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),3000);
 message(`已请求下载 ${rows.length} 条记录。请确认文件出现在下载文件夹，再发回聊天。`);
};
$('restore').onclick=()=>$('import').click();
$('import').onchange=async()=>{try{
 const file=$('import').files[0];if(!file)return;if(file.size>5*1024*1024)throw Error('记录文件超过5MB');
 const data=JSON.parse(await file.text());if(data.format!==DATA.export_format||!Array.isArray(data.records))throw Error('请选择本页导出的JSON文件');
 data.records.forEach(check);const fresh=rows.slice();let n=0;
 for(const r of data.records)if(!fresh.some(x=>x.scene===r.scene&&norm(x.command)===norm(r.command))){fresh.push(r);n++;}
 rows=fresh;persist();message(`恢复 ${n} 条，原记录保留；重复记录未重复计数。`);
 }catch(e){message('恢复失败：'+e.message);}finally{$('import').value='';}};
try{const saved=JSON.parse(localStorage.getItem(KEY)||'[]');if(!Array.isArray(saved))throw Error('本机缓存格式异常');saved.forEach(check);rows=saved;}
catch(e){$('storage').textContent='未读取缓存：'+e.message+'。如已有导出记录，可点击恢复。';}
$('scene').replaceChildren(...DATA.cards.map(s=>{const o=document.createElement('option');o.value=s.scene;o.textContent=s.scene;return o;}));
if(rows.length)$('author').value=rows[rows.length-1].author;
sceneChanged();
