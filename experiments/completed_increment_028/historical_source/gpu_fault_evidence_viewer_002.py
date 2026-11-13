"""从完整GPU故障批次及独立审计生成只读证据页面；不运行感知或语义模型。"""

import argparse
from dataclasses import asdict
import gzip
import html
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw

from conceptgraph.completion.kinematic_benchmark import SPEC
from conceptgraph.completion.navigation_demo import HTML, build_case, url
from conceptgraph.completion.stage9 import OUTPUT, digest, dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--event-audit', required=True)
    parser.add_argument('--numeric-audit', required=True)
    parser.add_argument('--execution-audit', required=True)
    args = parser.parse_args()
    source = OUTPUT / 'interactive_simulation' / args.input_run
    records_path = source / 'trial_records.jsonl'
    summary = json.loads((source / 'summary.json').read_text())
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert summary['attempted'] == summary['expected'] == len(records)
    assert digest(records_path) == summary['trial_records_sha256']
    reports = [OUTPUT / p / 'report.json' for p in
               [args.event_audit, args.numeric_audit, args.execution_audit]]
    event, numeric, execution = [json.loads(p.read_text()) for p in reports]
    assert event['passed'] and event['complete_cohort']
    assert numeric['passed'] and numeric['complete_cohort']
    assert execution['audit_completed']
    assert all(r['source_run'] == args.input_run for r in [event, numeric, execution])
    assert event['audited_cases'] == numeric['cases'] == execution['cases'] == len(records)
    assert event['input_prefix_sha256'] == numeric['input_records_sha256'] == digest(records_path)
    tasks_path = OUTPUT / 'interactive_simulation/synthetic_tasks_v1.json'
    release = json.loads(tasks_path.read_text())
    tasks = {t['task_id']: t for t in release['tasks']}
    configs = {c['config_id']: c for c in release['configs']}
    destination = OUTPUT / 'visualizations' / args.run_id
    destination.mkdir(exist_ok=False)
    (destination / 'cases').mkdir()
    begun = time.monotonic()
    files, cases, table = {}, [], []
    total_bytes = 0
    images = 0
    with (destination / 'effective_detection_sources.jsonl').open('x') as ledger:
        for record in records:
            assert time.monotonic() - begun < 1800
            assert shutil.disk_usage(OUTPUT).free > 6_000_000_000
            raw = Path(record['raw_case'])
            task = tasks[record['task']]
            data = build_case(raw, task, configs[task['config_id']])
            if record['result'] is not None:
                assert digest(raw / 'result.json') == record['result_sha256']
                assert data['result'] == record['result']
            else:
                assert data['result'] is None
            exposure = record['fault_exposure']
            data['fault_evidence'] = exposure
            data['raw_sources'].update({p.name: url(p) for p in
                [raw / 'fault_events.jsonl', raw / 'fault_exposure.json'] if p.exists()})
            for frame in data['frames']:
                index = frame['frame_idx']
                detected = raw / 'detections' / f'{index:06d}.pkl.gz'
                sensor_path = raw / 'sensor_evidence' / f'{index:06d}.json'
                sensor = json.loads(sensor_path.read_text())
                with gzip.open(detected) as stream:
                    gobs = pickle.load(stream)
                pixels = np.asarray(Image.open(sensor['rgb']).convert('RGB')).copy()
                colors = [(230, 120, 40), (35, 150, 180), (180, 70, 150), (80, 170, 50)]
                for i, mask in enumerate(gobs['mask']):
                    active = np.asarray(mask, dtype=bool)
                    pixels[active] = np.rint(.72 * pixels[active] + .28 * np.asarray(colors[i % 4])).astype(np.uint8)
                image = Image.fromarray(pixels)
                draw = ImageDraw.Draw(image)
                for i, bbox in enumerate(gobs['xyxy']):
                    draw.rectangle(tuple(map(float, bbox)), outline=colors[i % 4], width=2)
                    label = gobs['classes'][int(gobs['class_id'][i])]
                    draw.text((float(bbox[0]) + 2, max(0, float(bbox[1]) - 12)), str(label), fill=colors[i % 4])
                relative = Path('effective_detections') / record['run_id'] / f'{index:06d}.jpg'
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                image.save(target, quality=90)
                files[str(relative)] = digest(target)
                frame['detection_vis'] = url(target)
                frame['fault_active'] = sensor['fault_active']
                total_bytes += target.stat().st_size
                images += 1
                ledger.write(json.dumps(dict(run=record['run_id'], frame_idx=index,
                    detection_source=str(detected), detection_sha256=digest(detected),
                    sensor_source=str(sensor_path), sensor_sha256=digest(sensor_path),
                    output_file=str(relative), output_sha256=files[str(relative)])) + '\n')
            name = 'cases/' + record['run_id'] + '.json'
            (destination / name).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
            files[name] = digest(destination / name)
            total_bytes += (destination / name).stat().st_size
            assert total_bytes < 2_000_000_000
            cases.append(dict(run=record['run_id'], task_id=record['task'], seed=record['seed'],
                kind=record['kind'], status=data['status'], arrived=bool(record['actual_arrival']),
                file=name, frames=len(data['frames']), sources_sha256=data['sources_sha256']))
            table.append('<tr><td>' + html.escape(record['fault']) + '</td><td>'
                + html.escape(record['task']) + f" / {record['seed']}</td><td>"
                + ('是' if exposure.get('actual_fault_exposed') else '否／未触发')
                + '</td><td>' + html.escape(data['status']) + '</td><td><button data-case="'
                + html.escape(name, quote=True) + '">查看记录</button></td></tr>')
    links = ''.join('<a href="' + url(p) + '">' + label + '</a> ' for p, label in
                    zip(reports, ['事件独立审计', '传感器数值审计', '整车与到达独立审计']))
    panel = ('<section class="card" id="faultPanel"><h2>GPU故障实验：保存证据</h2>'
        f'<p>{len(records)} 次冻结试验，{images} 帧有效检测图。传感器 TTL 1.5 秒，语义 TTL 30 秒。'
        '未触发故障的试验保留，不能获得抗故障成功信用；基线没有故障注入。</p>'
        '<p>检测图由故障后实际送入建图的检测与掩膜绘制。此处播放日志，不产生新的模型调用或到达。'
        '共享场景、重复条件与种子不增加独立任务数；本批属于开发实验，未开启新封存测试。</p>'
        '<p class="links">' + links + '</p><p id="faultFrame" class="muted"></p>'
        '<details><summary>全部试验与实际触发情况</summary><div style="overflow:auto;max-height:400px">'
        '<table><thead><tr><th>条件</th><th>任务／种子</th><th>实际触发</th><th>终止</th><th>查看</th>'
        '</tr></thead><tbody>' + ''.join(table) + '</tbody></table></div></details></section>')
    script = '''<script>
const faultOriginalRender=render;
render=function(){faultOriginalRender();const f=data?.frames[frameIndex];
 $('faultFrame').textContent=data?`当前条件：${data.fault_evidence.condition}；当帧故障生效：${f?.fault_active?'是':'否'}；实际触发：${data.fault_evidence.actual_fault_exposed?'是':'否／未触发'}`:'';};
document.querySelectorAll('[data-case]').forEach(b=>b.onclick=()=>{
 $('filter').value='all';filter();$('case').value=b.dataset.case;choose().catch(fail);});
</script>'''
    page = HTML.replace('<main>', '<main>' + panel, 1).replace('</html>', script + '</html>')
    page = page.replace('保存的检测／SAM输出图', '故障后有效检测／SAM输出图')
    (destination / 'index.html').write_text(page)
    shutil.copy2(__file__, destination / 'generator_source.py')
    for name in ['index.html', 'generator_source.py', 'effective_detection_sources.jsonl']:
        files[name] = digest(destination / name)
    dump(destination / 'manifest.json', dict(cases=cases, files=files, vehicle_spec=asdict(SPEC),
        source_runs=[r['run_id'] for r in records], source_matrix=args.input_run,
        input_records_sha256=digest(records_path), task_release_sha256=digest(tasks_path),
        audit_reports={str(p): digest(p) for p in reports}, effective_detection_images=images,
        source_sha256=digest(__file__), evaluation_gt_hidden_by_default=True,
        actual_model_calls=0, new_arrivals=0, new_test_opened=False, goal_complete=False,
        saved_case_and_image_bytes=total_bytes, execution_wall_s=time.monotonic() - begun))
    print(json.dumps(dict(cases=len(cases), images=images, saved_bytes=total_bytes,
                         actual_model_calls=0, goal_complete=False)), flush=True)


if __name__ == '__main__':
    main()
