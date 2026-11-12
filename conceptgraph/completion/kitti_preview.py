"""真实KITTI帧、SAM轮廓、LiDAR与隔离GT的只读证据页。"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import pickle

import cv2
import numpy as np

from conceptgraph.completion.kitti_transfer import METHODS
from conceptgraph.completion.stage9 import OUTPUT, digest, dump

HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>KITTI真实前端证据</title>
<style>body{margin:20px auto;padding:0 16px;max-width:1440px;font:16px/1.65 system-ui;color:#173042;background:#f5f7fa}h1{font-size:27px}h2{font-size:20px}.notice,.card{background:white;border:1px solid #d7e0e8;border-radius:10px;padding:16px;margin:12px 0}.notice{border-left:5px solid #bf6900}.layout{display:grid;grid-template-columns:3fr 2fr;gap:16px}canvas{width:100%;height:auto;background:white;border:1px solid #dde3e8;box-sizing:border-box}button,select{font:inherit;padding:5px;max-width:100%}label{display:inline-block;margin:7px 12px 7px 0}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;border-bottom:1px solid #dde3e8;padding:6px;white-space:nowrap}code{overflow-wrap:anywhere}.error{color:#9a2700}a{color:#0859a8}@media(max-width:800px){.layout{display:block}h1{font-size:23px}body{margin:12px auto}}</style>
<h1>KITTI 真实车载感知与三维定位证据</h1>
<div class="notice">这是固定64帧的外部跨数据集开发诊断，复用真实YOLO、MobileSAM、CLIP与实测LiDAR。<b>没有跨帧融合、语言目标锁定或实际到达信用</b>。青色框为仅供评价的官方GT，未进入模型输入；深度不足显示unknown。数据：KITTI，CC BY-NC-SA 3.0。</div>
<div id="summary" class="card">正在读取已保存结果……</div>
<div class="card"><button id="prev">上一帧</button> <select id="frame" aria-label="帧选择"></select> <button id="next">下一帧</button><br><label><input id="gt" type="checkbox" checked>官方GT（评价专用）</label><label><input id="depth" type="checkbox">实测稀疏LiDAR（抽样显示）</label><label><input id="mask" type="checkbox" checked>SAM实际掩膜轮廓</label><div id="frame-info"></div></div>
<div class="layout"><div class="card"><h2>RGB、检测与深度证据</h2><canvas id="rgb"></canvas><p>橙：检测框及SAM轮廓；青：官方框；深度点仅为显示抽样，统计使用全部有效点。</p></div><div class="card"><h2>相机0坐标平面（米）</h2><canvas id="map" width="550" height="530"></canvas><p>横向x，前向z；灰圆：原始掩膜点均值；紫菱形：稳健深度均值；青方块：GT完整框中心。可见表面中心与完整框中心存在偏差。</p></div></div>
<div class="card"><h2>当前帧的真实检测</h2><div class="scroll"><table><thead><tr><th>编号/类别</th><th>置信度</th><th>原始支持点</th><th>稳健支持点</th><th>原始中心(x,y,z)m</th><th>稳健中心(x,y,z)m</th></tr></thead><tbody id="rows"></tbody></table></div><p>显示car、truck、pedestrian、bicycle；Cyclist/bicycle代理单列评价。全14类原始输出仍保存在检测文件。</p><a id="raw">当前帧原始定位记录</a> · <a href="../../statistics/kitti_object_cpu_full_001/report.md">完整统计与限制</a> · <a href="../thesis_portal_003/index.html">项目总览</a></div><p id="error" class="error"></p>
<script>
let data,current=0,renderVersion=0;const el=id=>document.getElementById(id),colors=['#8c9298','#8533ab'];
function centerText(v){return v?v.map(x=>x.toFixed(2)).join(', '):'unknown'}
async function show(){const n=++renderVersion,r=data.frames[current];el('frame').value=String(current);el('frame-info').textContent=`图像 ${r.frame_id}；原始采集 ${r.drive}；RGB输入SHA已核验；真实前端调用1次`;
el('raw').href=r.raw;let html='';for(const p of r.predictions){const a=p.methods.mask_points_mean,b=p.methods.mask_points_robust_mean;html+=`<tr><td>${p.prediction_index} / ${p.category}</td><td>${p.confidence.toFixed(3)}</td><td>${a.depth_evidence.retained_pixels}</td><td>${b.depth_evidence.retained_pixels}</td><td>${centerText(a.center_camera0)}</td><td>${centerText(b.center_camera0)}</td></tr>`}el('rows').innerHTML=html;
const image=new Image();image.src=r.rgb;await image.decode();if(n!==renderVersion)return;let canvas=el('rgb');canvas.width=image.width;canvas.height=image.height;let c=canvas.getContext('2d');c.drawImage(image,0,0);c.lineWidth=2;c.font='15px sans-serif';
if(el('depth').checked)for(const p of r.depth_pixels){c.fillStyle=`hsl(${Math.min(240,p[2]*4)},90%,42%)`;c.fillRect(p[0],p[1],2,2)}
for(const p of r.predictions){c.strokeStyle='#e67100';const b=p.bbox;c.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);c.fillStyle='#a54b00';c.fillText(`${p.prediction_index} ${p.category}`,Math.max(0,b[0]),Math.max(16,b[1]-3));if(el('mask').checked)for(const path of p.contours){if(!path.length)continue;c.beginPath();c.moveTo(...path[0]);for(const q of path.slice(1))c.lineTo(...q);c.closePath();c.stroke()}}
if(el('gt').checked)for(const g of r.truth){c.strokeStyle='#007e8c';const b=g.bbox;c.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);c.fillStyle='#00606a';c.fillText(`GT ${g.category}`,Math.max(0,b[0]),Math.min(image.height-3,b[3]+15))}
drawMap(r)}
function drawMap(r){const c=el('map').getContext('2d');c.clearRect(0,0,550,530);const pt=p=>[275+p[0]*8,490-p[2]*7];c.lineWidth=1;c.font='13px sans-serif';c.fillStyle='#333';for(let z=0;z<=60;z+=10){c.strokeStyle='#dde3e8';c.beginPath();c.moveTo(25,490-z*7);c.lineTo(525,490-z*7);c.stroke();c.fillText(`${z}m`,2,490-z*7)}c.fillText('x=0',277,516);c.fillText('z ↑',277,20);c.strokeStyle='#8798a5';c.beginPath();c.moveTo(275,490);c.lineTo(275,25);c.stroke();if(el('gt').checked)for(const g of r.truth){const [x,y]=pt(g.center_camera0);c.fillStyle='#007e8c';c.fillRect(x-4,y-4,8,8)}for(const p of r.predictions){data.methods.forEach((method,j)=>{const a=p.methods[method].center_camera0;if(!a)return;const [x,y]=pt(a);c.fillStyle=colors[j];c.beginPath();if(j===0)c.arc(x,y,4,0,Math.PI*2);else{c.moveTo(x,y-6);c.lineTo(x+6,y);c.lineTo(x,y+6);c.lineTo(x-6,y);c.closePath()}c.fill();c.fillText(String(p.prediction_index),x+6,y-3)})}}
function safeShow(){show().catch(e=>{el('error').textContent=String(e)})}el('prev').onclick=()=>{current=(current+data.frames.length-1)%data.frames.length;safeShow()};el('next').onclick=()=>{current=(current+1)%data.frames.length;safeShow()};el('frame').onchange=e=>{current=+e.target.value;safeShow()};for(const id of ['gt','depth','mask'])el(id).onchange=safeShow;
fetch('evidence.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(d=>{data=d;el('summary').textContent=`已完成 ${d.frames.length} 帧 / ${d.drive_count} 个采集drive；主表GT ${d.truth_count}；实际到达0。自定义中心匹配，不是官方KITTI AP。`;el('frame').innerHTML=d.frames.map((r,i)=>`<option value="${i}">${r.frame_id} — ${r.drive}</option>`).join('');safeShow()}).catch(e=>el('error').textContent=String(e));
</script></html>'''


def web_path(path):
    return '/' + str(Path(path).resolve().relative_to(OUTPUT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run', required=True)
    parser.add_argument('--inference-run', required=True)
    parser.add_argument('--statistics-run', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    source = OUTPUT/'data_manifest'/args.input_run
    inference = OUTPUT/'mapping_evaluation'/args.inference_run
    report = OUTPUT/'statistics'/args.statistics_run/'report.json'
    stats = json.loads(report.read_text())
    if stats['inference_sha256'] != digest(inference/'summary.json'):
        raise ValueError('统计与真实推理SHA不一致')
    agent = json.loads((source/'agent_manifest.json').read_text())
    truth = {r['frame_id']: r for r in json.loads((source/'evaluation_only/truth.json').read_text())['frames']}
    summary = json.loads((inference/'summary.json').read_text())
    frames = {r['frame_id']: r for r in agent['frames']}
    rows = []
    for saved in summary['frames']:
        if digest(saved['observation']) != saved['observation_sha256']:
            raise ValueError('观察记录改变')
        obs = json.loads(Path(saved['observation']).read_text())
        name = obs['frame_id']
        detected = Path(saved['observation']).parent/'detections.pkl.gz'
        if digest(detected) != saved['detection_sha256']:
            raise ValueError('SAM轮廓来源改变')
        with gzip.open(detected, 'rb') as handle:
            result = pickle.load(handle)
        predictions = []
        for pred in obs['predictions']:
            if pred['category'] not in {'car', 'truck', 'pedestrian', 'bicycle'}:
                continue
            mask = np.asarray(result['mask'][pred['prediction_index']], dtype=np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            simplified = [cv2.approxPolyDP(c, 1.5, True)[:, 0].tolist() for c in contours if cv2.contourArea(c) >= 3]
            predictions.append(pred | dict(contours=simplified))
        with np.load(frames[name]['projected_lidar']) as raw:
            depth = raw['depth']
            yy, xx = np.nonzero(depth)
            pixels = [[int(x), int(y), round(float(depth[y, x]), 2)] for x, y in zip(xx[::10], yy[::10])]
        gt = [r for r in truth[name]['objects'] if r['category'] in {'Car', 'Truck', 'Pedestrian', 'Cyclist'}
              and 1 < r['center_camera0'][2] < 65]
        rows.append(dict(frame_id=name, drive=obs['drive'], rgb=web_path(frames[name]['rgb']),
                         raw=web_path(saved['observation']), predictions=predictions,
                         depth_pixels=pixels, truth=gt))
    dest = OUTPUT/'visualizations'/args.run_id
    dest.mkdir(exist_ok=False, parents=True)
    dump(dest/'evidence.json', dict(frames=rows, methods=list(METHODS), drive_count=stats['raw_drive_count'],
        truth_count=stats['metrics'][METHODS[0]]['truth'], actual_arrivals=0,
        gt_evaluation_only=True, report_sha256=digest(report),
        source_sha256=digest(__file__)))
    html = HTML.replace('../../statistics/kitti_object_cpu_full_001/', f'../../statistics/{args.statistics_run}/')
    (dest/'index.html').write_text(html)
    dump(dest/'manifest.json', dict(frame_count=len(rows), source_sha256=digest(__file__),
        evidence_sha256=digest(dest/'evidence.json'), html_sha256=digest(dest/'index.html'),
        report=str(report), report_sha256=digest(report),
        actual_frontend_inference_credited=True, actual_navigation_credit=False,
        human_review_records_created=0))
    print(json.dumps(dict(preview=str(dest), frames=len(rows)), ensure_ascii=False))


if __name__ == '__main__':
    main()
