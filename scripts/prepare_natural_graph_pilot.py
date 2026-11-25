"""固定抽样20幅验证图，制作双击即用的离线人工指令采集页面。"""
import base64
from collections import defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

from PIL import Image

from conceptgraph.completion.stage9 import PROJECT, available_development, digest


PAGE = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>首批20张图：人工导航指令</title><style>
body{font:17px system-ui;margin:0;background:#eef2f6;color:#17324d}main{max-width:1300px;margin:auto;padding:18px}.box{background:white;padding:18px;border-radius:12px;margin:15px 0}canvas{width:100%;height:auto;cursor:crosshair}input,select,textarea,button{font:inherit;padding:9px}textarea{width:96%;min-height:70px}label{display:block;margin:12px 0}button{cursor:pointer}#status{white-space:pre-wrap;color:#174b35}.small{font-size:14px;color:#546578}.row{display:flex;gap:12px;flex-wrap:wrap}h1{font-size:26px}</style>
<main><h1>看图写指令：首批20张，每图两条</h1>
<div class="box"><p>每条做三件事：①用自己的话写希望车辆做什么；②点所指目标；③说明看到的依据。看不清可选“不确定”，两个都符合可选“有歧义”。不需要算距离或画框。</p>
<p>先写自己的话，画面不显示模型答案。面包车可选“面包车”；道路仅作为位置或参照描述。不要从单张图片猜是否静止、是否安全停车。</p>
<label>填写人 <input id="author" maxlength="80"></label><div class="row"><button id="prev">上一张</button><select id="card"></select><button id="next">下一张</button></div><p id="progress"></p></div>
<div class="box"><canvas id="image"></canvas><p class="small">红色圆点＝目标；蓝色圆点＝参照物。目标和参照物的多个点击均保留。写第二条前切换到第二条，再点击该条的目标。</p>
<label>正在填写 <select id="slot"><option value="0">第1条</option><option value="1">第2条</option></select></label>
<label>① 导航指令 <textarea id="command" placeholder="请按自己的日常表达填写，不要求套用句式"></textarea></label>
<label>情况 <select id="kind"><option value="target">能明确指出目标</option><option value="ambiguous">有多个可能目标</option><option value="unknown">看不清／不确定</option><option value="no_match">画面中确实没有所说目标</option></select></label>
<label>② 点击模式 <select id="mode"><option value="target_points">点击目标（可多个）</option><option value="anchor_points">点击参照物（可多个）</option></select> <button id="clear">清除此条点击</button></label>
<p id="points"></p><div class="row"><label>类别 <select id="category"><option value="unknown">不确定／不适用</option><option>小汽车</option><option>面包车</option><option>卡车</option><option>公交车</option><option>摩托车</option><option>自行车</option><option>行人</option><option>其他</option></select></label>
<label>颜色 <select id="color"><option value="unknown">看不清／不适用</option><option>白</option><option>黑</option><option>灰</option><option>银</option><option>红</option><option>蓝</option><option>绿</option><option>黄</option><option>橙</option><option>棕</option><option>其他</option></select></label></div>
<label>③ 观察依据／疑问 <textarea id="note" placeholder="写目标在画面哪里、旁边有什么；有疑问直接写出来"></textarea></label>
<label><input type="checkbox" id="attest"> 我本人看图后填写；看不清的内容没有猜测。</label><button id="save">确认保存这一条</button><p id="status"></p></div>
<div class="box"><button id="export">导出审核记录JSON</button> <label>恢复以前的记录 <input type="file" id="import" accept=".json"></label><p>每完成几张就导出备份。全部填完后把JSON发到聊天中。网页或端口打不开时，解压离线包，双击index.html即可填写，无需安装软件。</p><p class="small">保存按钮会检查必填项；导出包括已完成记录与未完成草稿。原始点击待绑定和核验后才能用于训练，不会自动成为正式测试标签。</p></div></main>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),key='lzc_'+data.campaign;
let state={campaign:data.campaign,manifest_sha256:data.manifest_sha256,author:'',records:{}},ci=0,si=0,img=null;
const fields=['command','kind','category','color','note','attest'];
function blank(){return {command:'',kind:'target',category:'unknown',color:'unknown',note:'',attest:false,target_points:[],anchor_points:[],confirmed:false}}
function record(){let k=data.cards[ci].id+':'+si;return state.records[k]||(state.records[k]={...blank(),card_id:data.cards[ci].id,slot:si})}
function persist(){try{localStorage.setItem(key,JSON.stringify(state))}catch(e){$('status').textContent='浏览器不能自动保存；请立即导出JSON备份。'}}
function collect(){let r=record();for(let f of fields)r[f]=f==='attest'?$(f).checked:$(f).value;state.author=$('author').value.trim();persist()}
function progress(){let n=Object.values(state.records).filter(r=>r.confirmed).length;$('progress').textContent=`第${ci+1}/20张；已确认${n}/40条。请分批导出备份。`}
function draw(){if(!img)return;let c=$('image'),x=c.getContext('2d');c.width=img.naturalWidth;c.height=img.naturalHeight;x.drawImage(img,0,0);for(let [f,col] of [['target_points','#e83030'],['anchor_points','#1767d1']])for(let p of record()[f]){x.beginPath();x.arc(p[0]*c.width,p[1]*c.height,10,0,Math.PI*2);x.strokeStyle=col;x.lineWidth=4;x.stroke()}$('points').textContent=`目标点${record().target_points.length}个；参照点${record().anchor_points.length}个。`}
function load(){let r=record();for(let f of fields){if(f==='attest')$(f).checked=r[f];else $(f).value=r[f]}$('author').value=state.author;$('card').value=ci;$('slot').value=si;$('status').textContent=r.confirmed?'这一条已确认保存。':'';img=null;let n=new Image;n.onload=()=>{img=n;draw()};n.src=data.cards[ci].image;progress()}
for(let f of fields)$(f).oninput=()=>{record().confirmed=false;collect();progress()};$('author').oninput=collect;
$('card').onchange=()=>{collect();ci=Number($('card').value);si=0;load()};$('slot').onchange=()=>{collect();si=Number($('slot').value);load()};
$('prev').onclick=()=>{collect();ci=Math.max(0,ci-1);si=0;load()};$('next').onclick=()=>{collect();ci=Math.min(data.cards.length-1,ci+1);si=0;load()};
$('image').onclick=e=>{if(!img)return;let b=$('image').getBoundingClientRect();record()[$('mode').value].push([(e.clientX-b.left)/b.width,(e.clientY-b.top)/b.height]);record().confirmed=false;collect();draw();progress()};
$('clear').onclick=()=>{record().target_points=[];record().anchor_points=[];record().confirmed=false;collect();draw();progress()};
$('save').onclick=()=>{collect();let r=record();if(!state.author||r.command.trim().length<4||!r.note.trim()||!r.attest){$('status').textContent='尚未确认：请填写姓名、指令、观察依据，并勾选本人填写。';return}if(['target','ambiguous'].includes(r.kind)&&!r.target_points.length){$('status').textContent='请点击至少一个目标；看不清可选择不确定。';return}r.confirmed=true;r.updated_at=new Date().toISOString();persist();$('status').textContent='已确认保存。完成几张后请导出JSON备份。';progress()};
$('export').onclick=()=>{collect();let out={...state,exported_at:new Date().toISOString(),cards:data.cards.map(({image,...c})=>c)},a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));a.download='人工导航标注_'+new Date().toISOString().slice(0,10)+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
$('import').onchange=async()=>{try{let v=JSON.parse(await $('import').files[0].text());if(v.campaign!==data.campaign||v.manifest_sha256!==data.manifest_sha256)throw Error('批次不一致');if(!v.records||typeof v.author!=='string')throw Error('记录格式错误');for(let [k,r] of Object.entries(v.records)){if(!data.cards.some(c=>k===c.id+':0'||k===c.id+':1'))throw Error('未知画面');for(let f of ['target_points','anchor_points'])if(!Array.isArray(r[f])||r[f].some(p=>!Array.isArray(p)||p.length!==2||p.some(x=>typeof x!=='number'||!Number.isFinite(x)||x<0||x>1)))throw Error('点击坐标错误')}state=v;persist();load();$('status').textContent='已恢复导入记录。'}catch(e){$('status').textContent='导入失败：'+e.message}};
$('card').replaceChildren(...data.cards.map((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${i+1}. ${c.scene} 帧${c.frame_idx}`;return o}));
try{let v=JSON.parse(localStorage.getItem(key));if(v&&v.manifest_sha256===data.manifest_sha256)state=v}catch(e){}load();
</script></html>'''


def main():
    root=PROJECT/'dataset/annotations/natural_graph_v004/pilot_20261120_001'
    if shutil.disk_usage(PROJECT).free<20*1024**3:raise OSError('磁盘低于保护线')
    root.mkdir(parents=True,exist_ok=False)
    reference=PROJECT/'outputs/results/nuscenes/validation/iterations/vehicle_frontend_full_20261118_v1'
    logs=defaultdict(list)
    for s in available_development():logs[s['log_token']].append(s)
    chosen=[]
    while len(chosen)<20:
        for log in sorted(logs):
            if logs[log] and len(chosen)<20:chosen.append(logs[log].pop(0))
    cards=[]
    for i,s in enumerate(chosen):
        with gzip.open(reference/s['scene']/'frames.jsonl.gz','rt') as stream:rows=[json.loads(x) for x in stream]
        row=rows[len(rows)//2]
        path=Path(row['rgb'])
        with Image.open(path) as image:
            image=image.convert('RGB');image.thumbnail((1280,900));output=io.BytesIO();image.save(output,format='JPEG',quality=88)
        cards.append(dict(id=f'card-{i+1:02d}',scene=s['scene'],log_token=s['log_token'],frame_idx=row['frame_idx'],
            source_rgb=str(path),source_sha256=digest(path),display_sha256=hashlib.sha256(output.getvalue()).hexdigest(),
            image='data:image/jpeg;base64,'+base64.b64encode(output.getvalue()).decode()))
    public=[{k:v for k,v in c.items() if k!='image'} for c in cards]
    manifest=dict(campaign='natural_graph_v004_pilot_20261120_001',cards=public,images=20,planned_commands=40,
        split='training_validation_workflow_pilot_not_test',selection='round_robin_acquisition_log_then_scene_sorted_middle_frame_no_model_outcome_selection',
        model_or_GT_overlay_shown=False,annotations_created=0,minimum_free_gib=20,maximum_new_bytes=20000000,
        source_sha256=digest(__file__),notes='当前轮仅流程试填；人工点击待官方实例绑定复核，不自动认定训练标签')
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    payload=dict(campaign=manifest['campaign'],manifest_sha256=digest(root/'manifest.json'),cards=cards)
    (root/'index.html').write_text(PAGE.replace('__DATA__',json.dumps(payload,ensure_ascii=False)))
    (root/'填写说明.txt').write_text('解压后双击index.html，用浏览器打开。每张图写两条自己的指令并点击目标，确认保存。每完成几张导出JSON备份，最后将JSON发回聊天。没有网络也能填写；点击是待核验提示，不是自动标准答案。\n')
    with zipfile.ZipFile(root/'人工标注20图_离线包.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['index.html','manifest.json','填写说明.txt']:z.write(root/name,name)
    assert sum(f.stat().st_size for f in root.iterdir())<20000000
    print(str(root))


if __name__=='__main__':main()
