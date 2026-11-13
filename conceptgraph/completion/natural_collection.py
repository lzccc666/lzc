"""从封存场景原始RGB采集人工自由表达，采集记录与有效标签分开。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import threading
import time
import unicodedata
import uuid

from conceptgraph.completion.stage9 import OUTPUT, digest, dump

CAMPAIGN = 'human_natural_collection_001'
STORAGE = 'human_generated_natural_submissions'
INTENTS = {'unknown', 'approach_static_target', 'no_match', 'inaccessible', 'other'}
LANGUAGES = {'zh', 'en', 'mixed', 'other'}
COLLECTION_LOCK = threading.Lock()


def normalized_command(command):
    """只识别字面重复，不声称已完成表达家族隔离。"""
    return ''.join(unicodedata.normalize('NFKC', command).casefold().split())


def _campaign(output):
    root = output/'annotation'/CAMPAIGN
    manifest = json.loads((root/'manifest.json').read_text())
    if digest(root/'cards.json') != manifest['cards_sha256']:
        raise ValueError('采集画面索引指纹不一致')
    return root, json.loads((root/'cards.json').read_text())


def validate_collection(record, *, output=OUTPUT):
    """校验实际画面与作者自述；不判断文字、目标或作者身份真实正确。"""
    output = Path(output).resolve()
    if not isinstance(record, dict) or record.get('campaign') != CAMPAIGN:
        raise ValueError('非法自然表达采集批次')
    root, cards = _campaign(output)
    if record.get('campaign_sha256') != digest(root/'manifest.json'):
        raise ValueError('采集批次指纹不一致')
    frame_idx = record.get('frame_idx')
    if type(frame_idx) is not int:
        raise ValueError('帧序号必须为整数')
    scene = next((s for s in cards if s['scene'] == record.get('scene')), None)
    frame = next((f for f in scene['frames'] if f['frame_idx'] == frame_idx), None) if scene else None
    if frame is None:
        raise ValueError('场景或帧不在固定采集清单')
    image = (output/frame['rgb']).resolve()
    if not image.is_relative_to(output):
        raise ValueError('相机证据越过输出目录')
    if (record.get('rgb_sha256') != frame['rgb_sha256']
            or record.get('camera_token') != frame['camera_token']
            or digest(image) != frame['rgb_sha256']):
        raise ValueError('相机证据指纹不一致')
    for field, minimum, maximum in [('author', 1, 120), ('command', 4, 1000), ('note', 1, 2000)]:
        value = record.get(field)
        if not isinstance(value, str) or not minimum <= len(value.strip()) <= maximum:
            raise ValueError('请填写有效的作者、原始自然表达和观察依据')
    command = record['command'].strip()
    if len(normalized_command(command)) < 4:
        raise ValueError('自然表达内容过短')
    if record.get('human_authorship_attested') is not True:
        raise ValueError('需要作者确认由本人观察画面后自行撰写')
    if record.get('language') not in LANGUAGES or record.get('author_intent') not in INTENTS:
        raise ValueError('非法语言或作者意图自述')
    point = record.get('author_target_point_normalized')
    if point is not None and (not isinstance(point, list) or len(point) != 2 or any(
            type(v) not in {int, float} or not math.isfinite(v) or not 0 <= v <= 1 for v in point)):
        raise ValueError('目标提示点必须在画面内；无法确定时保持空值')
    return dict(campaign=CAMPAIGN, campaign_sha256=digest(root/'manifest.json'),
        scene=scene['scene'], scene_token=scene['scene_token'], log_token=scene['log_token'],
        frame_idx=frame_idx, camera_token=frame['camera_token'], sample_token=frame['sample_token'],
        camera_timestamp=frame['camera_timestamp'], rgb=frame['rgb'], rgb_sha256=frame['rgb_sha256'],
        author=record['author'].strip(), command=command, normalized_command=normalized_command(command),
        language=record['language'], author_intent=record['author_intent'], note=record['note'].strip(),
        author_target_point_normalized=point, human_authorship_attested=True,
        authorship_independently_verified=False, annotation_origin='human_authored_browser_submission',
        automatic_label=False, target_identity=None, acceptable_target_set=None, observed_color='unknown',
        observed_motion='unknown', ambiguity='unknown', reachability='unknown',
        semantic_validity='pending_review', label_correctness_independently_verified=False,
        qualifies_for_sealed_test=False, client_submitted_at=record.get('submitted_at'))


def save_collection(record, *, output=OUTPUT):
    """追加原始人工记录；同场景字面重复不增加样本数，旧记录不覆盖。"""
    output = Path(output).resolve()
    checked = validate_collection(record, output=output)
    directory = output/'annotation'/STORAGE
    with COLLECTION_LOCK:
        for path in directory.glob('*.json'):
            previous = json.loads(path.read_text())
            if previous['scene'] == checked['scene'] and previous['normalized_command'] == checked['normalized_command']:
                raise ValueError('同场景已有相同表达；重复提交不会增加自然指令数量')
        directory.mkdir(parents=True, exist_ok=True)
        identity = uuid.uuid4().hex
        checked.update(submission_id=identity, server_received_utc=datetime.now(timezone.utc).isoformat(),
                       review_status='human_authored_source_and_schema_checked_pending_label_review')
        destination = directory/f'{identity}.json'
        temporary = directory/f'{identity}.pending'
        dump(temporary, checked)
        if destination.exists():
            raise FileExistsError(destination)
        temporary.rename(destination)
    return dict(saved=True, submission_id=identity, path=str(destination), sha256=digest(destination),
                qualifies_for_sealed_test=False)


def collection_status(*, output=OUTPUT):
    output = Path(output)
    _, cards = _campaign(output)
    records = [json.loads(p.read_text()) for p in (output/'annotation'/STORAGE).glob('*.json')]
    counts = Counter(r['scene'] for r in records)
    return dict(campaign=CAMPAIGN, human_authored_submissions=len(records),
        qualified_test_queries=0, qualified_count_status='当前采集流程未完成独立标签审核，原始提交不计入有效测试',
        by_scene={s['scene']: counts[s['scene']] for s in cards}, scenes=len(cards),
        minimum_per_scene=20, full_campaign_minimum=len(cards)*20,
        original_Talk2Car_reviews_included=False, duplicate_rule='同场景NFKC、大小写与空白归一后字面相同拒绝；未证明表达家族独立',
        goal_complete=False)


PAGE = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>真实自然指令采集</title><style>
*{box-sizing:border-box}body{margin:0;font:16px system-ui;background:#f3f5f7;color:#172a3b}main{max-width:1200px;margin:auto;padding:18px}h1{font-size:25px}.panel{background:white;padding:16px;border-radius:10px;margin:12px 0}label{display:block;margin:10px 0}input,select,textarea,button{font:inherit;padding:9px;max-width:100%}textarea{width:100%;min-height:90px}button{cursor:pointer}canvas{width:100%;height:auto;background:#ddd;touch-action:manipulation}input[type=range]{width:100%}.fields{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.muted{color:#52667a;font-size:14px}p,#status,#frameInfo{overflow-wrap:anywhere}#status{white-space:pre-wrap}a{color:#0a58a8}</style>
<main><h1>看真实连续画面，写自己的自然指令</h1>
<p>请按日常表达，写下你希望车辆根据画面完成的事情。优先描述靠近静止目标，也可记录确实存在的歧义、无匹配或不可接近情况。请自行观察和撰写，不照抄模板或改写模型输出；无法确认的信息保持未知。</p>
<p><a href="../natural_command_review_001/index.html">Talk2Car原文审核</a> · <a href="../../visualizations/review.html">开发对象证据核验</a></p>
<div class="panel"><div class="fields"><label>真实作者 <input id="author" autocomplete="name" maxlength="120"></label><label>场景 <select id="scene"></select></label></div><p id="counts">正在读取采集数量……</p><p class="muted">本页仅保存原始表达。20条是每场景的最低有效指令要求；30场景需至少600条，原始提交数不代表审核通过数。模板、自动改写、相同表达及重复帧不能充当独立样本。</p></div>
<div class="panel"><canvas id="rgb" aria-label="真实原始相机画面；点击可提示目标位置"></canvas><p id="frameInfo"></p>
<input id="frame" type="range" min="0" value="0" aria-label="时间帧"><button id="previous">上一帧</button> <button id="play">播放</button> <button id="next">下一帧</button> <button id="clearPoint">清除目标提示点</button><p class="muted">滑动或播放查看时序。可点击画面提示所指目标；点击只是作者提示，不是官方实例标签。画面不显示模型预测或官方目标框。</p></div>
<div class="panel"><label>本人观察后写出的原始指令 <textarea id="command" maxlength="1000"></textarea></label><div class="fields">
<label>语言 <select id="language"><option value="zh">中文</option><option value="en">英文</option><option value="mixed">混合</option><option value="other">其他</option></select></label>
<label>作者意图自述（待核验）<select id="intent"><option value="unknown">未知/暂不分类</option><option value="approach_static_target">接近静止目标</option><option value="no_match">当前观察范围无匹配</option><option value="inaccessible">目标存在但可能不可接近</option><option value="other">其他</option></select></label></div>
<label>观察依据与疑问（写明所指对象及不确定信息）<textarea id="note" maxlength="2000"></textarea></label>
<label><input id="attest" type="checkbox"> 我确认由本人观察画面后自行撰写，未使用自动生成或改写文本。</label><button id="submit">提交原始指令，进入待审核队列</button><p id="status" role="status"></p><p class="muted">指令会绑定提交时的当前帧。目标身份、静止状态、颜色、歧义和可达性仍需另行核验；作者自述不是已验证标签。每次保存新记录，刷新页面不会覆盖旧提交。</p></div></main>
<script>
let cards=[],campaignSha='',currentScene=null,currentFrame=null,point=null,timer=null,loadedImage=null,imageEpoch=0,stats=null;
const $=id=>document.getElementById(id);
function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='播放'}
function draw(){if(!loadedImage)return;let c=$('rgb'),ctx=c.getContext('2d');ctx.drawImage(loadedImage,0,0);if(point){ctx.beginPath();ctx.arc(point[0]*c.width,point[1]*c.height,12,0,2*Math.PI);ctx.strokeStyle='#e94025';ctx.lineWidth=4;ctx.stroke()}}
function frameChanged(){let f=currentScene.frames[Number($('frame').value)];currentFrame=f;point=null;loadedImage=null;let epoch=++imageEpoch;$('frameInfo').textContent=`${currentScene.scene} · 帧 ${f.frame_idx+1}/${currentScene.frames.length} · 相机时间 ${f.camera_timestamp} μs`;$('submit').disabled=true;let img=new Image;img.onload=()=>{if(epoch!==imageEpoch)return;loadedImage=img;$('rgb').width=img.naturalWidth;$('rgb').height=img.naturalHeight;draw();$('submit').disabled=false};img.onerror=()=>{if(epoch===imageEpoch)$('status').textContent='画面读取失败，不能提交。'};img.src='../../'+f.rgb}
function showCounts(){if(!stats||!currentScene)return;$('counts').textContent=`全批次原始人工表达 ${stats.human_authored_submissions} 条；当前场景 ${stats.by_scene[currentScene.scene]||0} 条；有效测试指令尚待独立标签审核。`}
async function refresh(){let r=await fetch('/api/natural_collection_status');if(!r.ok)throw Error('数量接口 '+r.status);stats=await r.json();showCounts()}
function sceneChanged(){stop();currentScene=cards.find(c=>c.scene===$('scene').value);$('frame').max=currentScene.frames.length-1;$('frame').value=0;$('status').textContent='';frameChanged();showCounts()}
$('scene').onchange=sceneChanged;$('frame').oninput=()=>{stop();frameChanged()};$('previous').onclick=()=>{stop();$('frame').value=Math.max(0,Number($('frame').value)-1);frameChanged()};$('next').onclick=()=>{stop();$('frame').value=Math.min(currentScene.frames.length-1,Number($('frame').value)+1);frameChanged()};
$('play').onclick=()=>{if(timer){stop();return}$('play').textContent='暂停';timer=setInterval(()=>{if(Number($('frame').value)>=currentScene.frames.length-1){stop();return}$('frame').value=Number($('frame').value)+1;frameChanged()},700)};
$('rgb').onclick=e=>{if(!loadedImage)return;let r=$('rgb').getBoundingClientRect();point=[Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))];draw()};$('clearPoint').onclick=()=>{point=null;draw()};$('author').value=localStorage.getItem('thesis_natural_author')||'';
$('submit').onclick=async()=>{stop();if(!loadedImage)return;const f=currentFrame,s=currentScene;let record={campaign:'human_natural_collection_001',campaign_sha256:campaignSha,scene:s.scene,frame_idx:f.frame_idx,camera_token:f.camera_token,rgb_sha256:f.rgb_sha256,author:$('author').value,command:$('command').value,note:$('note').value,language:$('language').value,author_intent:$('intent').value,human_authorship_attested:$('attest').checked,author_target_point_normalized:point,submitted_at:new Date().toISOString()};$('submit').disabled=true;try{let response=await fetch('/api/natural_collections',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(record)});let result=await response.json();if(!response.ok)throw Error(result.error||response.status);$('status').textContent='原始表达已保存，标签待审核：'+result.submission_id;localStorage.setItem('thesis_natural_author',record.author);$('command').value='';$('note').value='';$('attest').checked=false;await refresh()}catch(e){$('status').textContent='未保存：'+e.message}finally{$('submit').disabled=!loadedImage}};
Promise.all([fetch('cards.json').then(r=>r.json()),fetch('manifest.sha256').then(r=>r.text())]).then(async([data,sha])=>{cards=data;campaignSha=sha.trim();$('scene').replaceChildren(...cards.map(s=>{let option=document.createElement('option');option.value=s.scene;option.textContent=s.scene;return option}));sceneChanged();await refresh()}).catch(e=>{$('status').textContent='初始化失败：'+e.message});
</script></html>'''


def prepare():
    started = time.monotonic()
    if shutil.disk_usage(OUTPUT).free < 6*1024**3:
        raise RuntimeError('采集页准备需至少6GiB剩余空间')
    root = OUTPUT/'annotation'/CAMPAIGN
    root.mkdir(exist_ok=False)
    selection = OUTPUT/'splits/reserved_test_scenes_v2.json'
    provenance = OUTPUT/'data_manifest/reserved_causal_inputs_full_001/sensor_provenance.jsonl'
    sources_path = OUTPUT/'data_manifest/new_scene_sources_03_04.json'
    sources = json.loads(sources_path.read_text())
    cameras = {r['camera_token']: r for r in map(json.loads, provenance.open()) if r['channel'] == 'CAM_FRONT'}
    cards, alignments = [], {}
    for scene in json.loads(selection.read_text())['scenes']:
        if time.monotonic()-started > 300:
            raise TimeoutError('采集页准备超过300秒预算，保留已有文件')
        scene_root = OUTPUT/'data_manifest/reserved_causal_inputs_full_001/converted'/scene['scene']
        alignment = scene_root/'frame_alignment.json'
        alignments[str(alignment.relative_to(OUTPUT))] = digest(alignment)
        rows = json.loads(alignment.read_text())
        by_sample = {f['sample_token']: f for f in sources[scene['scene']]['sequence']}
        frames = []
        for row in rows:
            camera = by_sample[row['sample_token']]['sensors']['CAM_FRONT']
            evidence = cameras[camera['token']]
            image = scene_root/'color'/f"{row['frame_idx']:06d}.jpg"
            assert image.resolve() == Path(evidence['output']).resolve()
            assert digest(image) == evidence['output_sha256'] == evidence['member_sha256']
            assert camera['timestamp'] == row['camera_timestamp']
            frames.append(dict(frame_idx=row['frame_idx'], sample_token=row['sample_token'],
                          camera_token=camera['token'], camera_timestamp=row['camera_timestamp'],
                          rgb=str(image.relative_to(OUTPUT)), rgb_sha256=evidence['output_sha256']))
        assert len(frames) == scene['frame_count'] and [f['frame_idx'] for f in frames] == list(range(len(frames)))
        cards.append(dict(scene=scene['scene'], scene_token=scene['scene_token'], log_token=scene['log_token'], frames=frames))
    dump(root/'cards.json', cards)
    (root/'index.html').write_text(PAGE)
    dump(root/'manifest.json', dict(campaign=CAMPAIGN, scenes=len(cards), logs=len({s['log_token'] for s in cards}),
        frames=sum(len(s['frames']) for s in cards), selection_sha256=digest(selection),
        sensor_provenance_sha256=digest(provenance), source_sequence_sha256=digest(sources_path),
        alignment_sha256=alignments, cards_sha256=digest(root/'cards.json'), page_sha256=digest(root/'index.html'),
        source_sha256=digest(__file__), created_utc=datetime.now(timezone.utc).isoformat(),
        budget_wall_s=300, max_new_storage_bytes=10000000, new_RGB_copies=0,
        source='nuScenes原始RGB；沿用既有封存场景与采集顺序，不以方法结果选图',
        human_authored_records_at_creation=0, original_Talk2Car_not_rewritten=True,
        model_predictions_or_GT_overlays_shown=False, method_inference_executed=False,
        new_test_opened=False, qualifying_labels_created=0, goal_complete=False))
    if sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) > 10000000:
        raise RuntimeError('采集页元数据超过10MB预算，保留文件并拒绝完成')
    (root/'manifest.sha256').write_text(digest(root/'manifest.json')+'\n')
    (root/'generator_source.py').write_text(Path(__file__).read_text())
    print(json.dumps(dict(campaign=CAMPAIGN, scenes=len(cards), frames=sum(len(s['frames']) for s in cards),
                         actual_human_submissions=0, goal_complete=False), ensure_ascii=False))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
