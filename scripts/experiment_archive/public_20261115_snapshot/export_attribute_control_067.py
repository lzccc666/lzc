"""从新增真实结果生成可编辑交付表及可离线查看的裁剪证据页；不写论文正文。"""
import base64
from datetime import datetime, timezone
import html
import io
import json
from pathlib import Path
import zipfile
from xml.etree import ElementTree as ET

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font
from PIL import Image

from export_completed_tables_002 import p, table
from human_crop_fair_065 import sha, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tables/attribute_control_067'
ATTRIBUTE = ROOT / 'qwen_adapter_fair_comparison/human_crop_development_065'
BODY = ROOT / 'carla_official/current_body_causes_066'


def main():
    OUT.mkdir(exist_ok=False)
    source_paths = [ATTRIBUTE/'report.json', ATTRIBUTE/'comparison.csv', ATTRIBUTE/'per_sample.csv',
                    BODY/'report.json', BODY/'per_frame.csv', Path(__file__)]
    attribute = json.loads((ATTRIBUTE/'report.json').read_text())
    geometry = json.loads((BODY/'report.json').read_text())
    sections = []
    metrics = attribute['results']
    sections.append(('人工审核图：真实裁剪属性开发核验',
        ['模型', '格式通过/20', '单色正确/16', '颜色正确率', '类别正确/17', 'GPU均秒'],
        [[m['variant'], m['schema_valid'], m['color_correct'], f"{100*m['color_accuracy']:.2f}%",
          m['category_correct'], f"{m['mean_GPU_latency_s']:.3f}"] for m in metrics],
        '20张用户已审核裁剪，5模型共100次新GPU推理；每条图像、提示词、处理后输入张量均核验相同。'
        '单色以16张已知单色为分母，另外2张多色、2张未知不计算颜色准确率但全部保留输出。'
        '类别比较17张；模型同时收到检测器类别元数据，且这17张检测器原类别已全正确，因此该列不能证明独立视觉纠错能力。'
        '这是开发审核包，有选择偏差、单人审核及历史场景重合，不能用作独立测试或整体提升结论。'
        'real_graph新Adapter只训练排序，本次跨任务输出多无效，不能作为通用属性Adapter部署。'
        '未知细车型未重新标注，尚不计算细车型准确率；无三维身份、运动或到达信用。'))
    episodes = geometry['episodes']
    sections.append(('CARLA原始轨迹：当前车身检查的独立拆分',
        ['原回合', '观察帧', '初始车身拒绝', '独立复现', '道路导致', '障碍导致', '仅5cm余量超出'],
        [[e['job'], e['counts']['observed_frames'], e['counts']['reported_invalid_initial_body_frames'],
          e['counts']['reported_invalid_reproduced_frames'],
          e['reported_initial_failure_causes'].get('road_only', 0),
          e['reported_initial_failure_causes'].get('current_obstacle_only', 0),
          e['counts']['padding_only_road_violation_frames']] for e in episodes],
        '选取唯一成功回合1作参照，加全部4个已经正确选中、锁定和规划但未到达的正回合2/5/6/10。'
        '共2066个真实感知帧，523个规划初始车身拒绝全部通过当前状态/道路/障碍的独立相交检查复现，均为道路边界。'
        '最后一列按全部观察帧统计，与前几列仅针对规划报错的分母不同。'
        '第10回合的316个当前车身不合法帧全为5cm安全外扩超出道路；第2回合后期还有未外扩车身超出已声明走廊。'
        '不放宽车身安全余量；该分析未改变轨迹，不能证明修复后到达或唯一因果。未新增CARLA回合或GPU调用。'))
    body = [p('新增属性与车辆控制核验交付表'), p('仅实验资料，不是论文正文。所有失败、分母和适用范围保留。')]
    workbook = Workbook()
    workbook.remove(workbook.active)
    for index, (title, headers, values, note) in enumerate(sections, 1):
        sheet = workbook.create_sheet(f'{index:02d}')
        sheet.append([title])
        sheet.append(headers)
        for row in values:
            sheet.append(row)
        sheet.append([note])
        sheet.freeze_panes = 'B3'
        for column in sheet.columns:
            sheet.column_dimensions[column[0].column_letter].width = 27
            for cell in column:
                cell.font = Font(name='Noto Serif CJK SC', size=10)
                cell.alignment = Alignment(vertical='top', wrap_text=True)
        for cell in sheet[2]:
            cell.font = Font(name='Noto Serif CJK SC', size=10, bold=True)
        if index > 1:
            body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
        body.extend([p(title), table(headers, values), p(note)])
    workbook.save(OUT/'属性与控制核验表.xlsx')
    section = '<w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/><w:pgMar w:top="850" w:right="700" w:bottom="850" w:left="700"/></w:sectPr>'
    document = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'+''.join(body)+section+'</w:body></w:document>'
    ET.fromstring(document)
    template = ROOT/'tables/completed_office_002/已完成实验交付表.docx'
    source_paths.extend([template, Path(__file__).with_name('export_completed_tables_002.py')])
    with zipfile.ZipFile(template) as source, zipfile.ZipFile(OUT/'属性与控制核验表.docx', 'x', compression=zipfile.ZIP_DEFLATED) as dest:
        for name in source.namelist():
            dest.writestr(name, document.encode() if name == 'word/document.xml' else source.read(name))
    reference_path = ROOT/'annotation/human_attribute_reference_003/attributes.json'
    references = json.loads(reference_path.read_text())['rows']
    source_paths.append(reference_path)
    models = list(json.loads((ATTRIBUTE/'protocol.json').read_text())['variants'])
    output = {}
    for model in models:
        path = ATTRIBUTE/model/'outputs.jsonl'
        source_paths.append(path)
        output[model] = {row['sample_id']: row for row in map(json.loads, path.open())}
    page = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>人工裁剪属性开发核验</title>',
        '<style>body{font:16px system-ui;margin:24px;line-height:1.6;background:#fafafa}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;vertical-align:top}img{max-width:180px;max-height:130px}pre{white-space:pre-wrap;font-size:12px}.bad{color:#a11}</style>',
        '<h1>20张已审核图的属性核验</h1><p>同图同提示，共100次真实GPU调用。开发样本，有选择偏差，不是独立测试；只读展示，不是再次提交审核的页面。</p>',
        '<p>类别输入包含检测器标签，不能把类别成绩称为纯视觉识别。新real_graph Adapter只训练排序，本页保留它在属性任务上的无效输出。颜色主指标只计算16张已知单色。</p>',
        '<table><tr><th>实际模型裁剪</th><th>用户审核</th>' + ''.join('<th>'+html.escape(m)+'</th>' for m in models) + '</tr>']
    for reference in references:
        image_path = Path(reference['crop_file'])
        assert sha(image_path) == reference['crop_sha256']
        with Image.open(image_path) as image:
            preview = image.convert('RGB')
            preview.thumbnail((180, 130))
            data = io.BytesIO()
            preview.save(data, format='PNG')
        encoded = base64.b64encode(data.getvalue()).decode()
        sample = f"human-crop-{reference['sample_number']:03d}"
        page.append('<tr><td><img alt="实际模型裁剪" src="data:image/png;base64,'+encoded+'"><br>'+html.escape(reference['scene'])+'</td><td>'+html.escape(str({k: reference[k] for k in ['human_category', 'human_color', 'human_body_type']}))+'</td>')
        for model in models:
            prediction = output[model][sample]
            value = prediction['normalized_output'] or {}
            label = str({k: value.get(k) for k in ['object_tag', 'colors', 'body_type']})
            page.append('<td><span class="'+('' if prediction['schema_valid'] else 'bad')+'">'+('格式通过' if prediction['schema_valid'] else '格式无效')+'</span><br>'+html.escape(label)+'<details><summary>原始输出</summary><pre>'+html.escape(prediction['raw_output'] or '')+'</pre></details></td>')
        page.append('</tr>')
    page.append('</table></html>')
    with (OUT/'index.html').open('x') as f:
        f.write(''.join(page))
    assert len(load_workbook(OUT/'属性与控制核验表.xlsx').sheetnames) == 2
    with zipfile.ZipFile(OUT/'属性与控制核验表.docx') as z:
        ET.fromstring(z.read('word/document.xml'))
    write(OUT/'receipt.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
        tables=2, HTML_images=20, HTML_model_outputs=100,
        sources_sha256={str(path): sha(path) for path in source_paths},
        generated_sha256={p.name: sha(p) for p in OUT.iterdir() if p.is_file()},
        Word_XML_parsed=True, Excel_reopened=True, PDF_generated=False,
        workspace_dependency_tool_available=False, thesis_prose_written=False, goal_complete=False))
    for folder, description in [(ATTRIBUTE, sections[0][3]), (BODY, sections[1][3])]:
        with (folder/'核验说明_067.md').open('x') as f:
            f.write('# 适用范围与结论\n\n'+description+'\n\n原始报告和逐样本表保持不变。可编辑表及图像页见 '+str(OUT)+' 。\n')
    print(json.dumps(dict(tables=2, crops=20, real_model_outputs=100, goal_complete=False)))


if __name__ == '__main__':
    main()
