"""原始自然指令的人工任务范围/目标属性核验；只读官方标签，不运行封存方法。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from conceptgraph.completion.stage9 import OUTPUT, digest, dump


CAMPAIGN = 'natural_command_review_001'
SCOPES = {'unknown', 'static_target_approach', 'moving_target_action', 'driving_action', 'other'}
MOTIONS = {'unknown', 'static', 'dynamic'}
VISIBILITY = {'unknown', 'visible', 'not_visible'}
AMBIGUITY = {'unknown', 'clear', 'ambiguous'}
COLORS = {'unknown', 'red', 'blue', 'white', 'black', 'gray', 'silver', 'green', 'yellow', 'orange', 'brown', 'multiple'}


def validate_natural_submission(record, *, output=OUTPUT):
    """核验真实提交的来源和字段；不把结构通过当作语义正确或已获双人一致。"""
    if not isinstance(record, dict) or record.get('campaign') != CAMPAIGN:
        raise ValueError('非法自然指令核验批次')
    root = output/'annotation'/CAMPAIGN
    manifest = root/'manifest.json'
    if record.get('campaign_sha256') != digest(manifest):
        raise ValueError('核验批次指纹不一致')
    declared = json.loads(manifest.read_text())
    if digest(root/'cards.json') != declared['cards_sha256']:
        raise ValueError('核验卡片来源已变更')
    cards = json.loads((root/'cards.json').read_text())
    card = next((c for c in cards if c['command_token'] == record.get('command_token')), None)
    if card is None:
        raise ValueError('指令不在固定清单')
    for field, path in [('source_sha256', output/'data_manifest/natural_distribution_audit_001/evaluation_only_natural_commands.jsonl'),
                        ('rgb_sha256', output/card['rgb']), ('crop_sha256', root/card['crop'])]:
        if record.get(field) != card[field] or digest(path) != card[field]:
            raise ValueError('原始证据指纹不一致：'+field)
    reviewer = str(record.get('reviewer', '')).strip()
    note = str(record.get('note', '')).strip()
    if not reviewer or len(reviewer) > 120 or not note or len(note) > 5000:
        raise ValueError('请填写真实审核者及核验依据；看不清时保留unknown')
    for field, allowed in [('task_scope', SCOPES), ('observed_motion', MOTIONS),
                           ('target_visibility', VISIBILITY), ('target_ambiguity', AMBIGUITY), ('observed_color', COLORS)]:
        if record.get(field) not in allowed:
            raise ValueError('非法审核字段：'+field)
    if record['task_scope'] == 'static_target_approach' and record['observed_motion'] != 'static':
        raise ValueError('静止目标接近的确认需同时核验目标静止；否则范围保留unknown')
    return dict(campaign=CAMPAIGN, campaign_sha256=digest(manifest), command_token=card['command_token'],
        scene=card['scene'], camera_token=card['camera_token'], official_instance_token=card['instance_token'],
        source_sha256=card['source_sha256'], rgb_sha256=card['rgb_sha256'], crop_sha256=card['crop_sha256'],
        reviewer=reviewer, note=note, **{field: record[field] for field in
            ['task_scope', 'observed_motion', 'target_visibility', 'target_ambiguity', 'observed_color']},
        annotation_origin='human_browser_submission', automatic_label=False,
        label_correctness_independently_verified=False, reachability='unknown',
        client_submitted_at=record.get('submitted_at'))


PAGE = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>自然指令人工核验</title><style>body{font:16px system-ui;margin:0;background:#f4f5f7;color:#18222c}main{max-width:1200px;margin:auto;padding:20px}h1{font-size:24px}.panel{background:white;padding:18px;border-radius:10px;margin:14px 0}label{display:block;margin:10px 0}select,input,textarea,button{font:inherit;padding:8px;max-width:100%;box-sizing:border-box}textarea{width:100%;height:80px}button{cursor:pointer;margin:4px}canvas{width:100%;height:auto;background:#eee}#crop{max-height:230px;max-width:100%;image-rendering:auto}#command{font-size:20px;line-height:1.6;overflow-wrap:anywhere}.muted{color:#536579;overflow-wrap:anywhere}.fields{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:12px}#status{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#1260b8}</style>
<main><h1>原始自然指令：任务范围、目标及属性人工核验</h1>
<p>阅读原始英文指令并观察红框及原始裁剪。判断它是否要求接近静止对象；等待行人、跟随移动对象或一般转弯指令不自动计入主任务。看不清或无法确认时保留“未知”。官方运动属性只供参考，不能单凭一张图证明长期静止。</p>
<p><a href="../../visualizations/review.html">开发对象错误核验</a> · 数据：Talk2Car原始指令，CC BY-NC-SA 4.0；Deruyttere等，EMNLP 2019。此页面仅用于标注，未调用封存测试方法。</p>
<div class="panel"><label>真实审核者 <input id="reviewer" autocomplete="name"></label><label>筛选 <select id="filter"><option value="all">全部439条</option><option value="static">官方静态属性176条</option><option value="dynamic">官方动态属性215条</option><option value="unknown">官方运动未知48条</option></select></label>
<button id="previous">上一条</button><select id="choice" aria-label="选择待核验指令"></select><button id="next">下一条</button><p id="counter" class="muted"></p></div>
<div class="panel"><p id="command"></p><p id="meta" class="muted"></p><canvas id="rgb" aria-label="原始相机图与官方目标框"></canvas><p>官方二维框的真实像素裁剪（放大不增加证据）</p><img id="crop" alt="官方目标原始裁剪"></div>
<div class="panel"><div class="fields">
<label>指令任务范围 <select id="task_scope"><option value="unknown">未知/证据不足</option><option value="static_target_approach">接近静止目标</option><option value="moving_target_action">移动目标交互/等待/跟随</option><option value="driving_action">一般行驶/转弯/通过</option><option value="other">其他任务</option></select></label>
<label>目标运动核验 <select id="observed_motion"><option value="unknown">未知/单帧不能确认</option><option value="static">结合指令及官方证据可确认静止</option><option value="dynamic">移动</option></select></label>
<label>框内目标可见性 <select id="target_visibility"><option value="unknown">未知</option><option value="visible">目标可见</option><option value="not_visible">无法看到目标</option></select></label>
<label>指令指向是否明确 <select id="target_ambiguity"><option value="unknown">未知/需要更多证据</option><option value="clear">明确指向框内实例</option><option value="ambiguous">存在歧义</option></select></label>
<label>目标主要颜色 <select id="observed_color"><option value="unknown">未知/不可辨认</option><option value="red">红</option><option value="blue">蓝</option><option value="white">白</option><option value="black">黑</option><option value="gray">灰</option><option value="silver">银</option><option value="green">绿</option><option value="yellow">黄</option><option value="orange">橙</option><option value="brown">棕</option><option value="multiple">多色/无单一主色</option></select></label>
</div><label>核验依据（必填；不要仅复述自动属性）<textarea id="note"></textarea></label><button id="submit">提交当前审核记录</button><p id="status" role="status"></p><p class="muted">每次提交独立保存，修订不覆盖旧记录。提交数量不等于一致性或有效标签数量；此页面不标注车辆可达性。</p></div></main>
<script>let cards=[],visible=[],position=0,manifestSha='';const $=s=>document.getElementById(s);const fields=['task_scope','observed_motion','target_visibility','target_ambiguity','observed_color'];
function current(){return visible[position]}
function show(){let c=current();if(!c)return;$('choice').value=String(position);$('command').textContent=c.command;$('meta').textContent=`${c.scene} · ${c.upstream_split} · 类别 ${c.category} · 官方运动 ${c.official_motion} · ${c.command_token}`;$('counter').textContent=`${position+1}/${visible.length}；卡片仅为自动证据准备，尚需真实提交。`;fields.forEach(f=>$(f).value='unknown');$('note').value='';$('status').textContent='';$('crop').src=c.crop;let image=new Image;image.onload=()=>{let canvas=$('rgb');canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;let ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);ctx.lineWidth=4;ctx.strokeStyle='#ff293d';ctx.strokeRect(...c.box_xywh)};image.src='../../'+c.rgb}
function filter(){let value=$('filter').value;visible=cards.filter(c=>value==='all'||c.official_motion===value);position=0;$('choice').replaceChildren(...visible.map((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${i+1} · ${c.scene} · ${c.command_token.slice(0,8)}`;return o}));show()}
$('filter').onchange=filter;$('choice').onchange=()=>{position=Number($('choice').value);show()};$('previous').onclick=()=>{position=Math.max(0,position-1);show()};$('next').onclick=()=>{position=Math.min(visible.length-1,position+1);show()};$('reviewer').value=localStorage.getItem('thesis_natural_reviewer')||'';
$('submit').onclick=async()=>{let c=current(),record={campaign:'natural_command_review_001',campaign_sha256:manifestSha,command_token:c.command_token,reviewer:$('reviewer').value,note:$('note').value,source_sha256:c.source_sha256,rgb_sha256:c.rgb_sha256,crop_sha256:c.crop_sha256,submitted_at:new Date().toISOString()};fields.forEach(f=>record[f]=$(f).value);$('submit').disabled=true;try{let r=await fetch('/api/natural_reviews',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(record)});let data=await r.json();if(!r.ok)throw Error(data.error||r.status);$('status').textContent='已保存真实人工提交：'+data.submission_id;localStorage.setItem('thesis_natural_reviewer',record.reviewer)}catch(e){$('status').textContent='未保存：'+e.message}finally{$('submit').disabled=false}};
Promise.all([fetch('cards.json').then(r=>r.json()),fetch('manifest.sha256').then(r=>r.text())]).then(([c,s])=>{cards=c;manifestSha=s.trim();filter()}).catch(e=>{$('status').textContent='加载失败：'+e.message});</script></html>'''


def prepare():
    dest = OUTPUT/'annotation'/CAMPAIGN
    dest.mkdir(parents=True, exist_ok=False)
    (dest/'crops').mkdir()
    source = OUTPUT/'data_manifest/natural_distribution_audit_001/evaluation_only_natural_commands.jsonl'
    sensors = OUTPUT/'data_manifest/natural_original_inputs_full_001/sensor_provenance.jsonl'
    cameras = {r['camera_token']: r for r in map(json.loads, sensors.read_text().splitlines()) if r['channel'] == 'CAM_FRONT'}
    cards = []
    for row in map(json.loads, source.read_text().splitlines()):
        command = row['command']
        camera = cameras[command['sample_token']]
        rgb = Path(camera['output'])
        if digest(rgb) != camera['output_sha256']:
            raise ValueError('original natural RGB changed')
        image = Image.open(rgb)
        x, y, width, height = command['2d_box']
        crop_box = (max(0, int(x)), max(0, int(y)), min(image.width, int(x+width)), min(image.height, int(y+height)))
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            raise ValueError('upstream target box has no valid original pixels')
        identity = hashlib.sha256(json.dumps([camera['camera_token'], crop_box]).encode()).hexdigest()
        crop = dest/'crops'/f'{identity}.png'
        if not crop.exists():
            image.crop(crop_box).save(crop)
        cards.append(dict(command_token=command['command_token'], scene=row['scene'],
            camera_token=camera['camera_token'], command=command['command'], upstream_split=command['upstream_split'],
            category=command['obj_name'], official_motion=row['official_motion'],
            official_attributes=row['official_attributes'], instance_token=row['official_annotation']['instance_token'],
            box_xywh=command['2d_box'], clipped_crop_box=crop_box, rgb=str(rgb.relative_to(OUTPUT)),
            rgb_sha256=camera['output_sha256'], crop=str(crop.relative_to(dest)), crop_sha256=digest(crop),
            source_sha256=digest(source), human_verified=False, method_inference_executed=False))
    dump(dest/'cards.json', cards)
    (dest/'index.html').write_text(PAGE)
    dump(dest/'manifest.json', dict(campaign=CAMPAIGN, cards=len(cards),
        scenes=len({c['scene'] for c in cards}), unique_cameras=len(cameras), unique_crops=len(list((dest/'crops').glob('*.png'))),
        source_sha256=digest(source), sensor_provenance_sha256=digest(sensors),
        cards_sha256=digest(dest/'cards.json'), page_sha256=digest(dest/'index.html'),
        source_module_sha256=digest(__file__), attribution='Talk2Car, Deruyttere et al., EMNLP 2019',
        data_license='CC BY-NC-SA 4.0; original nuScenes sensor terms also apply',
        source_scope='原始439条全部呈现，不按方法成败筛选；仅任务范围与可见属性人工核验',
        human_submissions_at_creation=0, automatic_annotations_are_not_human=True,
        method_inference_executed=False, independent_test_opened=False, goal_complete=False))
    (dest/'manifest.sha256').write_text(digest(dest/'manifest.json')+'\n')
    (dest/'generator_source.py').write_text(Path(__file__).read_text())
    print(json.dumps(dict(output=str(dest), cards=len(cards), human_submissions=0, goal_complete=False), ensure_ascii=False))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
