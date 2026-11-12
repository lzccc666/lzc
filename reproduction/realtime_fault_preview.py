"""复用保存导航查看器，加入六条件证据及有效检测图；不调用任何感知模型。"""
from pathlib import Path
import hashlib
import html
import json
import gzip
import pickle
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image,ImageDraw

from conceptgraph.completion.stage9 import PROJECT,OUTPUT,digest,dump


def main():
    source=OUTPUT/'faults/realtime_fault_cpu_smoke_002'
    audit=json.loads((source/'audit_001/report.json').read_text())
    assert audit['passed']
    records=[json.loads(p.read_text()) for p in sorted(source.glob('*_record.json'))]
    run='realtime_fault_cpu_preview_002'
    command=[sys.executable,'-m','conceptgraph.completion.navigation_demo','--run-id',run]
    for record in records:command+=['--source-run',record['run_id']]
    subprocess.run(command,cwd=PROJECT,check=True)
    dest=OUTPUT/'visualizations'/run
    manifest=json.loads((dest/'manifest.json').read_text())
    metadata={r['run_id']:r for r in records};rows=[];images=0
    for case in manifest['cases']:
        record=metadata[case['run']];raw=Path(record['raw_case']);data=json.loads((dest/case['file']).read_text())
        for frame in data['frames']:
            index=frame['frame_idx']
            with gzip.open(raw/'detections'/f'{index:06d}.pkl.gz') as h:gobs=pickle.load(h)
            sensor=json.loads((raw/'sensor_evidence'/f'{index:06d}.json').read_text())
            image=np.asarray(Image.open(sensor['rgb']).convert('RGB')).copy()
            colors=[(230,120,40),(35,150,180),(180,70,150),(80,170,50)]
            for i,mask in enumerate(gobs['mask']):
                active=np.asarray(mask,dtype=bool)
                image[active]=np.rint(.72*image[active]+.28*np.asarray(colors[i%len(colors)])).astype(np.uint8)
            result=Image.fromarray(image);draw=ImageDraw.Draw(result)
            for i,bbox in enumerate(gobs['xyxy']):
                draw.rectangle(tuple(map(float,bbox)),outline=colors[i%len(colors)],width=2)
                category=gobs['classes'][int(gobs['class_id'][i])]
                draw.text((float(bbox[0])+2,max(0,float(bbox[1])-12)),str(category),fill=colors[i%len(colors)])
            relative=Path('effective_detections')/case['run']/f'{index:06d}.jpg'
            path=dest/relative;path.parent.mkdir(parents=True,exist_ok=True);result.save(path,quality=90)
            frame['detection_vis']='/'+str(path.relative_to(OUTPUT));manifest['files'][str(relative)]=digest(path)
            images+=1
        (dest/case['file']).write_text(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n');manifest['files'][case['file']]=digest(dest/case['file'])
        e=record['exposure'];file=json.dumps(case['file'])
        # 静态表只选择已保存case，复用原只读查看逻辑；没有入库或执行按钮。
        click=html.escape(f"document.getElementById('case').value={file};document.getElementById('case').dispatchEvent(new Event('change'))",quote=True)
        rows.append(f'<tr><td>{html.escape(record["condition"])}</td><td>{e["actual_sensor_frames"]}</td>'
            f'<td>{len(e["active_frames"])}</td><td>{e["recorded_control_ticks"]}</td><td>{html.escape(record["result"]["status"])}</td>'
            f'<td><button onclick="{click}">查看保存记录</button></td></tr>')
    note=('<section class="card"><h2>CPU规则真实前端：六条件单任务试跑</h2>'
        '<p>63次真实YOLO/MobileSAM/CLIP、2270控制tick，OSMesa软件渲染、Qwen关闭；'
        '六条件均为config00-positive0/seed17，只有一个任务、一个配置。'
        '实际到达读取独立评价与整车执行审计，重复条件不增加独立覆盖。</p>'
        '<p>本批传感器TTL为15秒；4秒断流真实发生，但未触发过期制动，负结果保留。'
        '正式GPU条件仍为1.5秒，当前尚未运行，不横比性能或宣称GPU故障验收完成。</p>'
        '<div style="overflow-x:auto"><table><thead><tr><th>条件</th><th>实际帧</th><th>注入帧</th><th>控制tick</th><th>终止</th><th>查看</th></tr></thead><tbody>'
        +''.join(rows)+'</tbody></table></div>'
        '<p class="muted">以下检测图按保存的故障后有效gobs重新绘制；原始未删减图和检测仍保存。'
        '点云使用有效深度及被评方法收到的相机位姿；评估层默认隐藏。播放是日志查看，未重新运行模型。</p>'
        '<p class="links"><a href="/faults/realtime_fault_cpu_smoke_002/audit_001/report.md">逐帧输入与执行复核</a>'
        '<a href="/task_protocol/realtime_sensor_faults_001.md">实验协议与限制</a>'
        '<a href="/faults/realtime_fault_after_semantics_001/release.json">正式GPU排队清单</a></p></section>')
    path=dest/'index.html';text=path.read_text().replace('<main>', '<main>'+note,1)
    text=text.replace('保存的检测／SAM输出图','保存的有效检测／SAM输出图（故障后）')
    path.write_text(text);manifest['files']['index.html']=digest(path)
    shutil.copy2(__file__,dest/'fault_wrapper_source.py');manifest['files']['fault_wrapper_source.py']=digest(__file__)
    manifest.update(fault_wrapper_sha256=digest(__file__),source_audit_sha256=digest(source/'audit_001/report.json'),
        effective_detection_images=images,new_model_calls=0,additional_unique_tasks=0,formal_GPU_faults_complete=False,goal_complete=False)
    (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cases=len(records),effective_detection_images=images,new_model_calls=0,goal_complete=False),ensure_ascii=False))


if __name__=='__main__':main()
