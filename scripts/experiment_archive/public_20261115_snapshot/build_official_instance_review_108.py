from __future__ import annotations
import base64,json,html
from pathlib import Path
R=Path(__file__).resolve().parents[1]
root=R/'annotation/offline_imports/human_natural_lzc_20261116_001'
out=root/'official_instance_review_108'; out.mkdir(exist_ok=True)
report=json.loads((root/'official_instance_audit_107/report.json').read_text())
img=Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1/data_manifest/reserved_causal_inputs_full_001/converted/scene-0248/color/000000.jpg')
b64=base64.b64encode(img.read_bytes()).decode()
items=[]
for i,row in enumerate(report['projected_candidates']):
 x1,y1,x2,y2=row['bbox_2d_xyxy'];
 # Keep only boxes that intersect the image; clip for display.
 if x2<0 or y2<0 or x1>1600 or y1>900: continue
 items.append({'i':i,'category':row['category'],'token':row['annotation_token'],'x1':max(0,min(1600,x1)),'y1':max(0,min(900,y1)),'x2':max(0,min(1600,x2)),'y2':max(0,min(900,y2)),'visibility':row['visibility_token'],'lidar':row['num_lidar_pts']})
boxes=json.dumps(items,ensure_ascii=False)
page=f'''<!doctype html><meta charset="utf-8"><title>官方实例独立核验 108</title>
<style>body{{font-family:Arial,sans-serif;background:#f5f7fa;margin:20px;color:#17202a}} .wrap{{max-width:1100px;margin:auto;background:white;padding:18px;border-radius:10px;box-shadow:0 2px 12px #ccd}} canvas{{max-width:100%;border:1px solid #789}} .row{{margin:10px 0}} label{{display:block;margin:7px 0}} textarea{{width:100%;height:90px}} button{{padding:10px 18px;background:#1769aa;color:#fff;border:0;border-radius:5px}} .small{{color:#566;font-size:13px}} </style>
<div class="wrap"><h2>scene-0248 第0帧：官方实例独立核验</h2>
<p>原始指令：<b>停到前方远处的黑色轿车后面</b>。红色十字是填写者点击位置；彩框是官方三维标注投影，仅用于事后核验，不会进入模型推理。</p>
<canvas id="cv" width="1600" height="900"></canvas>
<form id="f"><div class="row"><b>你看到的目标属于哪种情况？</b>
<label><input type="radio" name="decision" value="official_match" required> 与某个官方车辆框匹配（在备注中写框编号）</label>
<label><input type="radio" name="decision" value="unannotated_visible"> 看到目标，但官方框没有覆盖它</label>
<label><input type="radio" name="decision" value="no_target"> 没有看到所说的黑色轿车</label>
<label><input type="radio" name="decision" value="unclear"> 画面太远或太模糊，无法判断</label></div>
<div class="row">依据（请写你实际看到的内容）：<textarea id="note" required></textarea></div>
<button>下载核验记录 JSON</button></form><p class="small">当前自动核验结论：官方 vehicle.car 投影框与点击位置没有保守匹配，状态为待人工判断。不要根据此页自动修改封存测试分母。</p></div>
<script>const boxes={boxes}; const click={json.dumps(report['click_px'])}; const img=new Image(); img.onload=()=>{{const c=document.getElementById('cv'),x=c.getContext('2d');x.drawImage(img,0,0);x.strokeStyle='red';x.lineWidth=4;x.beginPath();x.moveTo(click[0]-16,click[1]);x.lineTo(click[0]+16,click[1]);x.moveTo(click[0],click[1]-16);x.lineTo(click[0],click[1]+16);x.stroke();boxes.forEach((b,j)=>{{x.strokeStyle=b.category==='vehicle.car'?'#00a000':'#0055ff';x.lineWidth=3;x.strokeRect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1);x.fillStyle=x.strokeStyle;x.font='20px Arial';x.fillText(j+': '+b.category,b.x1+4,Math.max(22,b.y1+22));}})}};img.src='data:image/jpeg;base64,{b64}';
document.getElementById('f').onsubmit=e=>{{e.preventDefault();const rec={{format:'official_instance_review_108',created_at:new Date().toISOString(),scene:'scene-0248',frame_idx:0,decision:document.querySelector('[name=decision]:checked').value,note:document.getElementById('note').value,click_px:click,displayed_boxes:boxes,automatic_status:'待独立复核'}};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(rec,null,2)],{{type:'application/json'}}));a.download='官方实例核验记录_scene-0248.json';a.click();}};</script>'''
(out/'index.html').write_text(page,encoding='utf-8')
(out/'README.md').write_text('# 官方实例独立核验页\n\n直接打开 `index.html`，选择你实际看到的情况并下载 JSON。该记录只作为人工复核候选，不自动成为封存测试标签。\n',encoding='utf-8')
print(json.dumps({'path':str(out/'index.html'),'displayed_boxes':len(items),'image_embedded':True},ensure_ascii=False))
