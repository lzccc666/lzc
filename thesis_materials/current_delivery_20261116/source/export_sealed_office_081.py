"""等现有图表导出结束后生成真实结果的Word/PDF表格；不写论文正文。"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from xml.etree import ElementTree as ET
import zipfile

from export_sealed_public_078 import LABELS
from sealed_resource_guard_074 import ResourceGuard

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT/'figures/sealed_public_078'
OUT = ROOT/'tables/sealed_office_081'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    with (OUT/name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def state(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None, None


def paragraph(value):
    return '<w:p><w:r><w:t xml:space="preserve">'+html.escape(str(value))+'</w:t></w:r></w:p>'


def table(headers, rows):
    width = 15438//len(headers)
    result = ['<w:tbl><w:tblPr><w:tblW w:w="15438" w:type="dxa"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="12"/><w:bottom w:val="single" w:sz="12"/>'
        '<w:left w:val="nil"/><w:right w:val="nil"/><w:insideH w:val="nil"/>'
        '<w:insideV w:val="nil"/></w:tblBorders></w:tblPr>']
    for index, row in enumerate([headers]+rows):
        result.append('<w:tr><w:trPr><w:cantSplit/>'+('<w:tblHeader/>' if index == 0 else '')+'</w:trPr>')
        for value in row:
            border = '<w:tcBorders><w:bottom w:val="single" w:sz="6"/></w:tcBorders>' if index == 0 else ''
            result.append(f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{border}</w:tcPr>'
                          +paragraph(value)+'</w:tc>')
        result.append('</w:tr>')
    return ''.join(result)+'</w:tbl>'


def formatted(value, *, percent=True):
    if value is None or value == '':
        return '未定义'
    return f'{float(value)*100:.2f}%' if percent else f'{float(value):.3f}'


def interval(row, *, percent=True):
    return formatted(row['value'], percent=percent)+' ['+formatted(row['ci95_low'], percent=percent)+', '+formatted(row['ci95_high'], percent=percent)+']'


def main(wait_pid=None):
    OUT.mkdir(exist_ok=False)
    template = ROOT/'tables/completed_office_002/已完成实验交付表.docx'
    sources = {str(Path(__file__)): sha(Path(__file__)), str(template): sha(template),
        str(ROOT/'source/export_sealed_public_078.py'): sha(ROOT/'source/export_sealed_public_078.py')}
    status, identity = state(wait_pid) if wait_pid else (None, None)
    if wait_pid:
        assert status is not None and status != 'Z'
        assert b'wait_and_export_sealed_079.py' in Path(f'/proc/{wait_pid}/cmdline').read_bytes()
    write('registration.json', dict(UTC=datetime.now(timezone.utc).isoformat(),
        wait_pid=wait_pid, wait_start_ticks=identity, sources_sha256=sources,
        selected_tables=['六配置单模块消融', '架构诊断', '模板主集检索', '模板挑战集检索',
                         '新Adapter三个种子', '模型推理性能'],
        maximum_wait_s=12*3600, document_export_budget_s=180,
        new_GPU_calls=0, new_CARLA_calls=0, thesis_prose_written=False, goal_complete=False))
    begun = time.monotonic()
    try:
        while wait_pid:
            current, start = state(wait_pid)
            if current is None or current == 'Z' or start != identity:
                break
            if time.monotonic()-begun > 12*3600:
                raise TimeoutError('等待图表截止，不中断原实验或图表进程')
            time.sleep(10)
        assert (INPUT/'export_wait_receipt_079.json').is_file()
        assert all(sha(Path(p)) == h for p, h in sources.items())
        guard = ResourceGuard(ROOT, OUT)
        tick = time.monotonic()
        guard(tick, 180)
        def read(name):
            path = INPUT/name
            sources[str(path)] = sha(path)
            with path.open(encoding='utf-8-sig') as f:
                return list(csv.DictReader(f))
        mapping = read('mapping_metrics.csv')
        retrieval = read('retrieval_all_metrics.csv')
        seeds = read('adapter_three_seed_summary.csv')
        execution = read('model_execution_statistics.csv')
        release = json.loads((ROOT/'main_experiments/sealed_execution_075/release.json').read_text())
        sources[str(ROOT/'main_experiments/sealed_execution_075/release.json')] = sha(ROOT/'main_experiments/sealed_execution_075/release.json')
        common = '30个封存场景、14个采集日志；每配置1196帧。括号内为日志聚类95%区间，4000次重采样。共享CPU建图耗时不作为独占实时速度。'
        sections = []
        for title, conditions in [('封存建图：严格单模块消融', release['conditions'][:6]),
                                  ('封存建图：架构诊断比较', ['full']+release['conditions'][6:])]:
            values = []
            for condition in conditions:
                row = [LABELS[condition]]
                for name in ['object_recall', 'object_precision', 'IDF1', 'center_error_m', 'relation_f1', 'duplicate_fraction']:
                    item = next(m for m in mapping if m['condition'] == condition and m['metric'] == name)
                    row.append(interval(item, percent=name != 'center_error_m'))
                values.append(row)
            sections.append((title, ['配置', '对象召回率↑', '对象精确率↑', 'IDF1↑', '中心误差/m↓', '关系F1↑', '重复比例↓'], values, common))
        for cohort, title in [('distribution_preserving', '模板主集'), ('challenge', '模板挑战集')]:
            values = []
            for method in release['retrieval_methods']:
                data = {m['metric']: m for m in retrieval if m['cohort'] == cohort and m['method'] == method}
                positive, negative = data['positive_target_correct'], data['no_match_false_selection']
                values.append([LABELS[method], positive['numerator']+'/'+positive['denominator'],
                    interval(positive), negative['numerator']+'/'+negative['denominator'],
                    interval(negative), interval(data['task_correct'])])
            sections.append((title+'：全部方法目标选择比较', ['方法', '正目标正确/n', '正目标正确率↑',
                '无匹配误选/n', '无匹配误选率↓', '全部任务正确率↑'], values,
                '每方法使用相同候选、指令和评价分母；这是模板查询，不能称人工自然指令。'
                '全部任务正确率包含无匹配，须同时看正目标和误选列；仅离线提议，不授予锁定或到达信用。'))
        names = dict(task_correct='全部任务正确率', positive_target_correct='正目标正确率',
                     no_match_false_selection='无匹配误选率')
        sections.append(('新Adapter三个正式种子的结果差异', ['集合', '指标', '每种子分母',
            '三个种子均值', '样本标准差', '最小值', '最大值'],
            [[('主集' if r['cohort'] == 'distribution_preserving' else '挑战集'), names[r['metric']],
              r['denominator_per_seed'], formatted(r['mean_over_three_seeds']),
              formatted(r['sample_std_over_three_seeds']), formatted(r['minimum_seed_value']),
              formatted(r['maximum_seed_value'])] for r in seeds],
            'seed17/29/43全部保留；种子重复不增加独立场景数。标准差描述种子差异，不替代日志聚类区间。'))
        sections.append(('五模型实际推理性能', ['模型', '处理/调用', '平均/s', '中位/s', 'P95/s', '峰值显存/MiB'],
            [[LABELS[r['variant']], r['requests']+'/'+r['actual_GPU_calls'],
              formatted(r['mean_GPU_latency_s'], percent=False), formatted(r['p50_GPU_latency_s'], percent=False),
              formatted(r['p95_GPU_latency_s'], percent=False), formatted(r['peak_GPU_MiB'], percent=False)] for r in execution],
            '处理数包括超长输入等未调用项，真实调用另列；延迟仅计实际GPU生成，不包括感知、建图或车辆执行。'
            '五模型逐进程串行，不据此声称端到端实时。格式有效率和冷启动、总耗时见同源Excel/CSV。'))
        body = [paragraph('封存公开数据实验结果表'), paragraph('只整理已完成实验数据，不是论文正文；完整分母、负结果和适用范围保留。')]
        for index, (title, headers, values, note) in enumerate(sections):
            if index:
                body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
            body.extend([paragraph(title), table(headers, values), paragraph(note)])
        section = '<w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/><w:pgMar w:top="700" w:right="700" w:bottom="700" w:left="700"/></w:sectPr>'
        document = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'+''.join(body)+section+'</w:body></w:document>'
        tree = ET.fromstring(document)
        assert len(tree.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tbl')) == 6
        docx = OUT/'封存公开数据实验表.docx'
        with zipfile.ZipFile(template) as source, zipfile.ZipFile(docx, 'x', compression=zipfile.ZIP_DEFLATED) as output:
            for name in source.namelist():
                output.writestr(name, document.encode() if name == 'word/document.xml' else source.read(name))
        profile = OUT/'libreoffice_profile'
        command = ['libreoffice', '-env:UserInstallation='+profile.as_uri(), '--headless',
                   '--convert-to', 'pdf', '--outdir', str(OUT), str(docx)]
        (OUT/'tmp').mkdir()
        env = dict(os.environ, TMPDIR=str(OUT/'tmp'), XDG_CACHE_HOME=str(OUT/'runtime_cache'))
        process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
        with (OUT/'PDF导出日志.txt').open('x') as f:
            f.write(process.stdout+'\n'+process.stderr)
        assert process.returncode == 0 and (OUT/'封存公开数据实验表.pdf').is_file()
        guard(tick, 180, force=True)
        assert all(sha(Path(p)) == h for p, h in sources.items())
        write('receipt.json', dict(UTC=datetime.now(timezone.utc).isoformat(), tables=6,
            sources_sha256=sources, generated_sha256={p.name: sha(p) for p in OUT.iterdir() if p.is_file()},
            Word_XML_parsed=True, PDF_generated=True, PDF_visual_QA_pending=True,
            workspace_dependency_tool_available=False, thesis_prose_written=False,
            actual_new_GPU_calls=0, actual_new_CARLA_calls=0, goal_complete=False))
        print('six actual sealed tables exported to editable Word and PDF; PDF visual review pending', flush=True)
    except BaseException as exc:
        write(f'failure_{time.time_ns()}.json', dict(error=repr(exc), traceback=traceback.format_exc(),
            no_experiment_restart=True, goal_complete=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    main(args.wait_pid)
