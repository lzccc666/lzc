"""KITTI连续开发实验的只读浏览器；默认隐藏官方评价真值。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from conceptgraph.completion.stage9 import OUTPUT, digest, dump

HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>KITTI连续车载证据</title>
<style>body{font:16px/1.65 system-ui;background:#f4f7fa;color:#193348;max-width:1450px;margin:20px auto;padding:0 16px}h1{font-size:26px}h2{font-size:20px}.card{background:#fff;border:1px solid #dae3ec;padding:16px;border-radius:10px;margin:12px 0;min-width:0}.notice{border-left:5px solid #b86c00}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.grid .card{margin:0}img,canvas{width:100%;height:auto;box-sizing:border-box;display:block;max-width:100%}canvas{border:1px solid #dde6ed}label{display:inline-block;margin:5px 10px 5px 0}select,button,input{font:inherit;max-width:100%;box-sizing:border-box}button,select{padding:5px}a{color:#075b95;overflow-wrap:anywhere}.scroll{overflow:auto}table{border-collapse:collapse;font-size:14px;width:100%}th,td{text-align:left;border-bottom:1px solid #dde6ed;white-space:nowrap;padding:6px}#error{color:#9c2600}#frame{width:min(100%,540px)}@media(max-width:800px){.grid{display:block}.grid .card{margin:12px 0}h1{font-size:23px}body{margin:10px auto}}</style>
<h1>KITTI 连续车载感知、融合与身份诊断</h1>
<div class="card notice">固定3个序列各30帧，共90帧；真实CPU YOLO/MobileSAM/CLIP与实测LiDAR。<b>开发诊断；不计导航到达，不是KITTI官方HOTA/AP。</b>使用0.1秒名义采样周期，没有原始传感器时间戳；1帧与此前Object64重合；74次主类别GT均在0005，其他两序列主类别GT为0。官方GT只供评价，默认隐藏。</div>
<div class="card" id="summary">正在读取冻结结果……</div>
<div class="card"><label>序列 <select id="sequence"></select></label><label>条件 <select id="method"></select></label><br><button id="prev">上一帧</button> <button id="play">播放</button> <button id="next">下一帧</button> <input id="frame" aria-label="帧序号" type="range" min="0" max="29" value="0"><br><label><input id="gt" type="checkbox">显示官方评价GT</label><label><input id="trace" type="checkbox" checked>过去帧节点轨迹</label><label>三维方位 <input id="angle" aria-label="三维方位角" type="range" min="0" max="360" value="35"></label><p id="info"></p></div>
<div class="grid"><div class="card"><h2>实际RGB</h2><img id="rgb" alt="实际KITTI相机2图像"><h2>实际检测与SAM</h2><img id="det" alt="实际检测和掩膜"><h2>实测稀疏深度</h2><img id="depth" alt="稀疏LiDAR深度"><p>深度0为无证据；伪彩色按1–65米固定范围。统计使用全部有效点，三维图仅抽样显示。</p></div><div class="card"><h2>初始IMU世界XY / 过去帧身份</h2><canvas id="map" width="700" height="560"></canvas><h2>当前稀疏点云与对象三维中心</h2><canvas id="world" width="700" height="450"></canvas><p>紫点：当前条件对象；蓝点：实测点云；青色：仅供评价的官方中心；黑色：相机位姿。轨迹来自当前条件实际保存节点ID。</p></div></div>
<div class="card"><h2>当前对象与评价关联</h2><div class="scroll"><table><thead><tr><th>节点ID</th><th>类别</th><th>世界中心(m)</th><th>观测次数</th><th>运动状态</th><th>名义速度(m/s)</th><th>GT关联（开启后）</th></tr></thead><tbody id="objects"></tbody></table></div><p id="score"></p><a id="raw">当前序列原始评价记录</a> · <a id="report">统计与限制</a> · <a id="audit">独立评价复核</a> · <a id="dataaudit">独立数据复核</a></div><p id="error"></p>
<script>
const el=id=>document.getElementById(id);let data,frames=[],index=0,timer=null;
function stop(){clearInterval(timer);timer=null;el('play').textContent='播放'}
function seq(){frames=data.frames.filter(f=>f.sequence===el('sequence').value);index=0;el('frame').max=frames.length-1;stop();show()}
function show(){try{const f=frames[index],m=el('method').value,p=f.predictions[m],truth=el('gt').checked?f.truth:[],metrics=f.metrics[m];el('frame').value=index;
el('info').textContent=`序列 ${f.sequence}，原始帧 ${f.original_frame_id}；窗口 ${index+1}/${frames.length}；名义时刻 ${f.timestamp_s.toFixed(1)}s；当前节点 ${p.length}；有效LiDAR像素 ${f.depth_pixels}。`;
el('rgb').src=f.rgb;el('det').src=f.detection;el('depth').src=f.depth_visual;el('raw').href=f.raw;
const body=el('objects');body.replaceChildren();for(let i=0;i<p.length;i++){const o=p[i],match=metrics.matches.find(x=>x.prediction_index===i),tr=document.createElement('tr');let associated='隐藏';if(el('gt').checked)associated=match?`${f.truth[match.truth_index].instance_token} / ${match.distance_m.toFixed(2)}m`:'未匹配';
for(const value of [o.node_id,o.category,o.center.map(x=>x.toFixed(2)).join(', '),o.num_detections,o.motion_state,o.velocity_mps.toFixed(2),associated]){const td=document.createElement('td');td.textContent=String(value);tr.append(td)}body.append(tr)}
el('score').textContent=el('gt').checked?`同类XY≤3m中心匹配 ${metrics.true_positive}/${metrics.truth} GT；未匹配预测 ${metrics.unmatched_predictions}（含未支持标签影响，不全称误检）。`:'官方评价结果已隐藏；图像、节点和历史轨迹来自隔离推理输出。';
const xy=el('map'),c=xy.getContext('2d');c.clearRect(0,0,xy.width,xy.height);let observed=frames.flatMap(f=>f.predictions[m].map(p=>p.center)).concat(frames.map(f=>f.camera));if(!observed.length)observed=[[0,0,0]];let xs=observed.map(p=>p[0]),ys=observed.map(p=>p[1]);let xmin=Math.min(...xs)-3,xmax=Math.max(...xs)+3,ymin=Math.min(...ys)-3,ymax=Math.max(...ys)+3;
const scale=Math.min(620/(xmax-xmin),460/(ymax-ymin)),loc=p=>[40+(p[0]-xmin)*scale,520-(p[1]-ymin)*scale];c.font='13px system-ui';c.fillStyle='#456';c.fillText('世界X / Y，米；独立初始IMU参考',12,22);c.strokeStyle='#d9e4ec';
for(let x=Math.ceil(xmin/10)*10;x<xmax;x+=10){let a=loc([x,ymin]),b=loc([x,ymax]);c.beginPath();c.moveTo(...a);c.lineTo(...b);c.stroke();c.fillText(String(x),a[0],545)}for(let y=Math.ceil(ymin/10)*10;y<ymax;y+=10){let a=loc([xmin,y]),b=loc([xmax,y]);c.beginPath();c.moveTo(...a);c.lineTo(...b);c.stroke();c.fillText(String(y),4,a[1])}
function dot(ctx,q,color,r,label){ctx.fillStyle=color;ctx.beginPath();ctx.arc(q[0],q[1],r,0,2*Math.PI);ctx.fill();if(label)ctx.fillText(label,q[0]+5,q[1]-5)}
if(el('trace').checked){const tracks=new Map();for(const past of frames.slice(0,index+1))for(const o of past.predictions[m]){const id=m==='single_frame'?`${past.frame_idx}:${o.node_id}`:String(o.node_id);if(!tracks.has(id))tracks.set(id,[]);tracks.get(id).push(o.center)}c.strokeStyle='#af8ac4';for(const points of tracks.values()){c.beginPath();points.forEach((p,i)=>{const q=loc(p);i?c.lineTo(...q):c.moveTo(...q)});c.stroke()}}
for(const o of truth)dot(c,loc(o.center),'#088b9a',5,String(o.track_id));for(const o of p)dot(c,loc(o.center),'#893db0',4,String(o.node_id));dot(c,loc(f.camera),'#111',5,'相机');
const canvas=el('world'),w=canvas.getContext('2d');w.clearRect(0,0,700,450);let a=Number(el('angle').value)*Math.PI/180;const proj=p=>{const x=p[0]-f.camera[0],y=p[1]-f.camera[1],z=p[2]-f.camera[2],u=Math.cos(a)*x-Math.sin(a)*y,v=Math.sin(a)*x+Math.cos(a)*y;return [350+u*6,350-v*3-z*9]};w.font='12px system-ui';for(const q of f.points)dot(w,proj(q),'#529bc1',1.2);for(const o of truth)dot(w,proj(o.center),'#088b9a',5,String(o.track_id));for(const o of p)dot(w,proj(o.center),'#893db0',4,String(o.node_id));dot(w,proj(f.camera),'#111',5,'相机');
}catch(e){el('error').textContent=e.message}}
fetch('evidence.json').then(r=>{if(!r.ok)throw Error('证据读取失败');return r.json()}).then(d=>{data=d;for(const s of d.sequences){const o=document.createElement('option');o.value=o.textContent=s;el('sequence').append(o)}for(const m of d.methods){const o=document.createElement('option');o.value=m;o.textContent=d.names[m];el('method').append(o)}el('method').value='one_to_one_depth_robust_motion';el('summary').textContent=d.summary;for(const k of ['report','audit','dataaudit'])el(k).href=d.links[k];seq()}).catch(e=>el('error').textContent=e.message);
el('sequence').onchange=seq;for(const id of ['method','gt','trace','angle'])el(id).oninput=()=>show();el('prev').onclick=()=>{stop();index=Math.max(0,index-1);show()};el('next').onclick=()=>{stop();index=Math.min(frames.length-1,index+1);show()};el('frame').oninput=()=>{stop();index=Number(el('frame').value);show()};el('play').onclick=()=>{if(timer){stop();return}el('play').textContent='暂停';timer=setInterval(()=>{if(index>=frames.length-1){stop();return}index++;show()},500)};
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statistics-run', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    statistics = OUTPUT/'statistics'/args.statistics_run
    report = json.loads((statistics/'report.json').read_text())
    if not json.loads((statistics/'independent_audit_002/report.json').read_text())['passed']:
        raise ValueError('独立复核未通过')
    destination = OUTPUT/'visualizations'/args.run_id
    destination.mkdir(parents=True, exist_ok=False)

    def web(path):
        return os.path.relpath(Path(path), destination)

    rows, sources = [], {}
    for path, expected in report['source_evaluation_frames_sha256'].items():
        if digest(path) != expected:
            raise ValueError('评价记录SHA不同')
        sources[path] = expected
        for line in open(path):
            f = json.loads(line)
            for field in ['input_sha256', 'frontend_output_sha256']:
                for p, sha in f[field].items():
                    if digest(p) != sha:
                        raise ValueError('原始证据SHA不同')
                    sources[p] = sha
            depth = np.asarray(Image.open(f['depth']), dtype=float)/1000
            valid = depth > 0
            normalized = np.clip((depth-1)/64*255, 0, 255).astype(np.uint8)
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
            colored[~valid] = 0
            visual = destination/'depth'/f['sequence']/f"{f['original_frame_id']:06d}.jpg"
            visual.parent.mkdir(exist_ok=True, parents=True)
            if not cv2.imwrite(str(visual), colored):
                raise OSError('深度图保存失败')
            y, x = np.nonzero(valid)
            stride = max(1, len(x)//600)
            x, y = x[::stride], y[::stride]
            camera_points = np.column_stack([x, y, np.ones(len(x))])@np.linalg.inv(np.asarray(f['intrinsic'])).T
            camera_points *= depth[y, x, None]
            pose = np.asarray(f['pose'])
            points = camera_points@pose[:3, :3].T+pose[:3, 3]
            rows.append(dict(sequence=f['sequence'], frame_idx=f['frame_idx'], original_frame_id=f['original_frame_id'],
                timestamp_s=f['camera_timestamp_s'], rgb=web(f['rgb']), detection=web(f['detection_visualization']),
                depth_visual=web(visual), raw=web(path), depth_pixels=int(valid.sum()),
                predictions=f['predictions'], truth=f['evaluation_only_official'], metrics=f['metrics'],
                points=points.round(3).tolist(), camera=pose[:3, 3].tolist()))
    methods = ['single_frame', 'original_fusion', 'one_to_one_depth_robust_motion']
    names = dict(zip(methods, ['单帧对象', '原始融合（名义0.1秒）', '一对一＋深度质量＋稳健运动']))
    summary = []
    for method in methods:
        m = report['object']['methods'][method]
        recall = m['metrics']['recall']
        summary.append(f"{names[method]}：匹配{m['counts']['true_positive']}/{m['counts']['truth']}，召回{recall:.2%}" if recall is not None else f'{names[method]}：召回unknown')
    dump(destination/'evidence.json', dict(frames=rows, methods=methods, names=names,
        sequences=sorted({f['sequence'] for f in rows}), summary='；'.join(summary),
        links=dict(report=web(statistics/'report.json'), audit=web(statistics/'independent_audit_002/report.json'),
            dataaudit=web(OUTPUT/'data_manifest/kitti_tracking_center90_002/independent_audit_001/report.json'))))
    (destination/'index.html').write_text(HTML)
    dump(destination/'manifest.json', dict(frames=len(rows), sequences=3,
        statistics_sha256=digest(statistics/'report.json'), evidence_sha256=digest(destination/'evidence.json'),
        html_sha256=digest(destination/'index.html'), source_sha256=digest(__file__),
        original_evidence_sha256=sources, official_GT_hidden_by_default=True,
        human_review_records_created=0, navigation_arrivals=0, goal_complete=False))
    print(json.dumps(dict(viewer=str(destination), frames=len(rows))))


if __name__ == '__main__':
    main()
