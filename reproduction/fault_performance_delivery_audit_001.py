"""逐项核验表格、来源和原始计时重算；不执行模型。"""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path('/root/autodl-tmp/lizichong/outputs/project_v2/thesis_completion_v1')
PACKAGE = ROOT / 'experiment_tables/fault_performance_office_002'
sys.path.insert(0, str(ROOT / 'source_bundle/document_runtime'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display(value, kind):
    if value is None:
        return '未知/无分母'
    if kind == 'float':
        return f'{value:.4f}'
    if kind == 'int':
        return str(int(value))
    return str(value)


def main():
    import fitz
    import openpyxl
    from docx import Document

    destination = PACKAGE / 'independent_audit_001'
    destination.mkdir(exist_ok=False)
    tables = json.loads((PACKAGE / 'tables.json').read_text())
    sources = json.loads((PACKAGE / 'sources.json').read_text())
    for name, value in sources.items():
        assert sha(ROOT / name) == sha(PACKAGE / value['package_path']) == value['sha256']
    book = openpyxl.load_workbook(PACKAGE / '开发实验结果表.xlsx', data_only=True)
    document = Document(PACKAGE / '开发实验结果表.docx')
    pdf = fitz.open(PACKAGE / '开发实验结果表.pdf')
    assert len(document.tables) == len(tables) == len(pdf) == 7
    cells = 0
    for ti, table in enumerate(tables):
        columns = table['columns']
        sheet = book[table['id']]
        word = document.tables[ti]
        text = ''.join(pdf[ti].get_text().split())
        assert table['id'] in text
        with (PACKAGE / 'csv' / (table['id'] + '.csv')).open(encoding='utf-8-sig') as stream:
            csv_rows = list(csv.DictReader(stream))
        assert len(csv_rows) == len(table['rows'])
        for ri, row in enumerate(table['rows']):
            for ci, column in enumerate(columns):
                value = row[column['key']]
                actual = sheet.cell(ri+2, ci+1).value
                if isinstance(value, float):
                    assert math.isclose(actual, value, rel_tol=1e-14, abs_tol=1e-12)
                else:
                    assert actual == value
                assert csv_rows[ri][column['key']] == ('' if value is None else str(value))
                rendered = display(value, column['kind'])
                assert word.rows[ri+1].cells[ci].text == rendered
                assert ''.join(rendered.split()) in text, (table['id'], rendered)
                cells += 1
        for x0, y0, x1, y1, *_ in pdf[ti].get_text('words'):
            assert x0 >= 0 and y0 >= 0 and x1 <= pdf[ti].rect.width+1 and y1 <= pdf[ti].rect.height+1
    assert book['故障180逐任务'].max_row == 181
    fault = json.loads((ROOT / 'statistics/realtime_faults_gpu_full_001/report.json').read_text())
    assert sum(fault['missing_complete_results'].values()) == 1
    assert sum(v['by_kind']['reachable_positive']['trials'] for v in fault['methods'].values()) == 180
    recomputed = 0
    for variant in ['base', 'old', 'new', 'frontend']:
        base = ROOT / f'performance/runtime_profile_gpu_{variant}_001'
        report = json.loads((base / 'report.json').read_text())
        groups = defaultdict(list)
        for line in (base / 'timings.jsonl').open():
            row = json.loads(line)
            groups[row['stage']+'/'+row['phase']].append(row)
        assert set(groups) == set(report['stages'])
        for key, rows in groups.items():
            values = sorted(r['duration_s'] for r in rows)
            pos = .95*(len(values)-1)
            lo, hi = math.floor(pos), math.ceil(pos)
            p95 = values[lo]+(values[hi]-values[lo])*(pos-lo)
            expected = report['stages'][key]
            assert expected['calls'] == len(rows)
            assert expected['failures'] == sum(not r['succeeded'] for r in rows)
            for field, actual in [('median_s', statistics.median(values)), ('p95_s', p95),
                                  ('summed_s', sum(values)),
                                  ('successful_calls_per_measured_s', sum(r['succeeded'] for r in rows)/sum(values))]:
                assert math.isclose(expected[field], actual, rel_tol=1e-12, abs_tol=1e-12), (key, field)
            recomputed += 1
    for i in [0, 5]:
        pdf[i].get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).save(destination / f'page_{i+1}.png')
    result = dict(passed=True, tables=7, rows=sum(len(t['rows']) for t in tables),
                  numeric_and_text_cells_checked=cells, PDF_pages=len(pdf),
                  sources_verified=len(sources), performance_stages_recomputed=recomputed,
                  failed_process_kept_in_denominator=True, actual_new_model_calls=0,
                  source_sha256=sha(Path(__file__)), goal_complete=False)
    with (destination / 'report.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
