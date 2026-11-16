"""打包真实原始相机图供人工离线填写；不生成指令或标签。"""
from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent
OUT = ROOT/'annotation/natural_offline_087'
ASSET = ROOT/'source/natural_offline_087.js'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    campaign = OLD/'annotation/human_natural_collection_001'
    manifest = json.loads((campaign/'manifest.json').read_text())
    assert sha(campaign/'cards.json') == manifest['cards_sha256']
    # 固定采集顺序前10场景，每场景原始前6个连续帧，未按检测或方法成绩选图。
    selected = []
    for card in json.loads((campaign/'cards.json').read_text())[:10]:
        frames = []
        for frame in card['frames'][:6]:
            path = OLD/frame['rgb']
            assert sha(path) == frame['rgb_sha256']
            frames.append(dict(frame, image_data='data:image/jpeg;base64,'+
                               base64.b64encode(path.read_bytes()).decode('ascii')))
        selected.append(dict(card, total_original_frames=len(card['frames']), frames=frames))
    payload = dict(campaign='human_natural_collection_001', campaign_sha256=sha(campaign/'manifest.json'),
                   cards=selected, export_format='thesis_human_natural_offline_087')
    template = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>自然指令填写：离线试用版</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:#172a3b;font:17px system-ui,"Microsoft YaHei",sans-serif}main{max-width:1100px;margin:auto;padding:20px}.panel{background:white;padding:16px;border-radius:10px;margin:14px 0}h1{font-size:27px}label{display:block;margin:10px 0}input,textarea,select,button{font:inherit;padding:9px;max-width:100%}textarea{width:100%;min-height:80px}canvas{width:100%;height:auto;background:#ddd;touch-action:manipulation}input[type=range]{width:100%}button{cursor:pointer;margin:4px}.muted{font-size:14px;color:#52667a}#status{white-space:pre-wrap;overflow-wrap:anywhere}.actions{display:flex;flex-wrap:wrap;align-items:center;gap:8px}#count{font-weight:bold}input[type=file]{display:none}
</style><main><h1>看图，用自己的话写导航指令</h1>
<p>这是离线试用页。图片已经放在本文件内，不需要联网、登录或端口转发。</p>
<div class="panel"><strong>操作：</strong>填写昵称 → 看图并点击想靠近的目标 → 写自己的指令和观察依据 → 勾选确认 → 点“保存这一条”。先填10条，然后点“导出记录”，把下载的 JSON 文件发回聊天。
<p>不知道怎么判断就写“不确定”。不要求每张图都有目标，不要把看不到的内容编进去。这里没有预填指令或模型答案。</p>
<div class="actions"><button id="export">导出记录（JSON）</button><button id="restore">从已导出的JSON恢复</button><input type="file" id="import" accept=".json,application/json"><span id="count"></span></div>
<p class="muted" id="storage">请定期导出，浏览器缓存不能替代文件备份。</p></div>
<div class="panel"><label>你的姓名或固定昵称 <input id="author" maxlength="120" autocomplete="name"></label>
<label>场景 <select id="scene"></select></label><canvas id="rgb" aria-label="真实原始相机画面；点击提示目标"></canvas>
<p id="frameInfo"></p><input id="frame" type="range" min="0" value="0" aria-label="连续帧">
<button id="previous">上一帧</button><button id="next">下一帧</button><button id="clearPoint">清除点击点</button>
<p class="muted">每场景仅含开头6个连续帧，约2.5秒。点击点只是你的提示，并非已核验的对象标签。短片不足以判断静止时，请写不确定。</p></div>
<div class="panel"><label>你会对车说的原始指令 <textarea id="command" maxlength="1000"></textarea></label>
<label>观察依据或疑问 <textarea id="note" maxlength="2000"></textarea></label>
<label>语言 <select id="language"><option value="zh">中文</option><option value="en">英文</option><option value="mixed">混合</option><option value="other">其他</option></select></label>
<label>你的意图（不确定可保留默认项）<select id="intent"><option value="unknown">不确定 / 暂不分类</option><option value="approach_static_target">靠近静止目标</option><option value="no_match">当前画面无匹配目标</option><option value="inaccessible">目标存在但可能无法靠近</option><option value="other">其他</option></select></label>
<label><input id="attest" type="checkbox"> 我确认由本人看图后自行撰写，没有使用自动生成或改写文本。</label>
<button id="submit" disabled>保存这一条（到本机浏览器）</button><p id="status" role="status"></p>
<p class="muted">保存到本机后还需导出JSON并发回聊天。你之前20张颜色审核无需重做。本试用包不等于完成全部自然指令测试；对象、运动、歧义等还需另行核验。不同写法指同一个物体也不等于多个独立场景。</p></div></main>
<script id="embedded" type="application/json">__PAYLOAD__</script>
<script>__SCRIPT__</script></html>'''
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    script = ASSET.read_text()
    assert '</script' not in script.casefold()
    page = template.replace('__PAYLOAD__', encoded).replace('__SCRIPT__', script)
    assert len(page.encode('utf-8')) < 20*1024**2
    OUT.mkdir(exist_ok=False)
    html = OUT/'自然指令填写_离线试用.html'
    html.write_text(page)
    evidence = [{k: v for k, v in card.items() if k != 'frames'} | {
        'frames': [{k: v for k, v in f.items() if k != 'image_data'} for f in card['frames']]
    } for card in selected]
    report = dict(UTC=datetime.now(timezone.utc).isoformat(), scenes=len(selected),
                  original_frames_per_scene=6, frames=sum(len(c['frames']) for c in selected),
                  selection='固定原采集顺序前10场景，每场景前6连续帧；不按方法成绩筛选',
                  original_image_bytes_embedded_without_resizing=True,
                  original_campaign_manifest_sha256=payload['campaign_sha256'],
                  image_evidence=evidence, html_sha256=sha(html), html_bytes=html.stat().st_size,
                  source_sha256=sha(Path(__file__)), script_sha256=sha(ASSET),
                  commands_prefilled=0, human_records_created=0, automatic_labels_created=0,
                  existing_files_modified=False, server_or_forward_required=False,
                  actual_new_GPU_calls=0, actual_new_CARLA_calls=0, goal_complete=False)
    with (OUT/'receipt.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ['scenes','frames','html_bytes','human_records_created','goal_complete']}))


if __name__ == '__main__':
    main()
