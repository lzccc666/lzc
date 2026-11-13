"""从原始450运行记录与实际文件核对结果表；检查可编辑文件和PDF交付。"""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

PROJECT = Path('/root/autodl-tmp/lizichong')
OUT = PROJECT/'outputs/project_v2/thesis_completion_v1'
PACKAGE = OUT/'experiment_tables/development_evidence_003'
DEST = PACKAGE/'independent_audit_001'
sys.path.insert(0, str(OUT/'source_bundle/document_runtime'))


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def string_value(value, kind):
    if value is None:
        return '未知/无分母'
    if kind == 'bool':
        return ['否', '是'][bool(value)]
    if kind == 'rate':
        return format(value*100, '.2f')+'%'
    if kind == 'pp':
        return format(value*100, '+.2f')
    if kind == 'float':
        return format(value, '.4f')
    return str(int(value)) if kind == 'int' else str(value)


def main():
    from docx import Document
    import fitz
    import openpyxl
    from PIL import Image

    DEST.mkdir(exist_ok=False)
    try:
        manifest = read(PACKAGE/'manifest.json')
        assert not (PACKAGE/'interrupted.json').exists()
        assert all(digest(PACKAGE/p) == h for p, h in manifest['files'].items())
        sources = read(PACKAGE/'sources.json')
        assert len(sources) == 17
        assert all(digest(OUT/p) == d['sha256'] == digest(PACKAGE/d['package_path'])
                   for p, d in sources.items())
        tables = read(PACKAGE/'tables.json')
        assert len(tables) == 15 and sum(len(t['rows']) for t in tables) == 114
        cyclist = [r for r in tables[9]['rows'] if r['category'] == 'bicycle']
        assert [(r['truth'], r['matched'], r['pairs'], r['error']) for r in cyclist] == [(20, 6, 0, None), (20, 7, 0, None)]
        assert '普通/稳健3D匹配分别6/7' in tables[9]['note']
        assert '2D共同条件对均为0' in tables[9]['note']
        source_records = OUT/'interactive_simulation/realtime_three_model_full_001/trial_records.jsonl'
        shutil.copy2(source_records, DEST/'main450_trial_records.jsonl')
        records = [json.loads(line) for line in (DEST/'main450_trial_records.jsonl').read_text().splitlines()]
        assert len(records) == 450
        fields = ['route_generated', 'selected_object_evaluated_correct',
                  'stable_correct_lock', 'actual_full_perception_arrival']
        csv_rows = list(csv.DictReader((PACKAGE/'evidence/statistics/realtime_three_model_full_001/trials.csv').open()))
        csv_by_id = {(r['method'], r['task_id'], int(r['seed'])): r for r in csv_rows}
        assert len(csv_by_id) == len(records)
        raw_hashes = {}
        missing = []
        for record in records:
            key = (record['variant'], record['task'], record['seed'])
            row = csv_by_id[key]
            path = Path(record['raw_case'])/'result.json'
            if record['result'] is None:
                assert not path.exists() and row['result_missing'] == 'True'
                assert all(row[f] == '' for f in fields)
                missing.append(record['run_id'])
                continue
            actual = read(path)
            assert digest(path) == record['result_sha256']
            assert all(actual[f] == record['result'][f] and str(actual[f]) == row[f] for f in fields)
            raw_hashes[str(path.relative_to(OUT))] = digest(path)
        assert len(raw_hashes) == 449 and len(missing) == 1
        expected_counts = {'rule': [83, 84, 82, 50], 'base_full': [66, 72, 66, 39],
                           'new_full': [59, 33, 32, 4]}
        for method, expected in expected_counts.items():
            positive = [r for r in records if r['variant'] == method and r['kind'] == 'reachable_positive']
            assert len(positive) == 90
            counts = [sum(r['result'] is not None and r['result'][f] is True for r in positive) for f in fields]
            assert counts == expected
        save(DEST/'main449_raw_result_sha256.json', raw_hashes)

        workbook = openpyxl.load_workbook(PACKAGE/'开发实验结果表.xlsx', data_only=False)
        document = Document(PACKAGE/'开发实验结果表.docx')
        assert len(document.tables) == len(tables)
        pdf = fitz.open(PACKAGE/'开发实验结果表.pdf')
        assert len(pdf) == len(tables)
        cells_checked = 0
        page_checks = []
        for index, table in enumerate(tables):
            sheet = workbook[table['id']]
            word_table = document.tables[index]
            assert len(word_table.rows) == len(table['rows'])+1
            assert [c.value for c in sheet[1]] == [c['label'] for c in table['columns']]
            assert [c.text for c in word_table.rows[0].cells] == [c['label'] for c in table['columns']]
            data_csv = list(csv.DictReader((PACKAGE/'csv'/f"{table['id']}.csv").open(encoding='utf-8-sig')))
            assert len(data_csv) == len(table['rows'])
            pdf_page = pdf[index]
            page_text = ''.join(pdf_page.get_text().split())
            assert table['id']+''.join(table['title'].split()) in page_text
            assert '\ufffd' not in page_text
            assert abs(pdf_page.rect.width - 841.92) < 1 and abs(pdf_page.rect.height - 594.96) < 1
            for i, row in enumerate(table['rows'], 2):
                for j, column in enumerate(table['columns'], 1):
                    value = row[column['key']]
                    cell = sheet.cell(i, j)
                    if isinstance(value, float):
                        assert isinstance(cell.value, (float, int)) and math.isclose(cell.value, value, rel_tol=1e-12, abs_tol=1e-12)
                    else:
                        assert cell.value == value
                    assert data_csv[i-2][column['key']] == ('' if value is None else str(value))
                    text = string_value(value, column['kind'])
                    assert word_table.rows[i-1].cells[j-1].text == text
                    assert ''.join(text.split()) in page_text
                    cells_checked += 1
            words = pdf_page.get_text('words')
            outside = [w[:4] for w in words if w[0] < -0.1 or w[1] < -0.1
                       or w[2] > pdf_page.rect.width+.1 or w[3] > pdf_page.rect.height+.1]
            assert not outside
            assert all(pdf_page.search_for(p) for p in ['开发证据', 'goal_complete=false'])
            page_checks.append(dict(page=index+1, table=table['id'], rows=len(table['rows']),
                                   outside_text_boxes=len(outside), cell_display_present=True))
        assert workbook['原始450逐任务'].max_row == 451
        for row_index, row in enumerate(csv_rows, 2):
            assert [c.value or '' for c in workbook['原始450逐任务'][row_index]] == list(row.values())
        assert not any(row[c].data_type == 'f' for sheet in workbook for row in sheet for c in range(len(row)))
        pages = [0, 6, 9, 12]
        contact = Image.new('RGB', (1600, 1132), 'white')
        for i, page_index in enumerate(pages):
            pix = pdf[page_index].get_pixmap(matrix=fitz.Matrix(1.3, 1.3), alpha=False)
            image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
            image.thumbnail((800, 566))
            contact.paste(image, ((i % 2)*800, (i//2)*566))
        contact.save(DEST/'preview_main_fair_kitti_fault.png')
        checkpoint = read(OUT/'source_bundle/source_checkpoint_037/checkpoint.json')
        export = OUT/'source_bundle/github_sync_001/checkout'
        assert all(digest(PROJECT/p) == h == digest(export/p) for p, h in checkpoint['sources'].items())
        assert digest(PROJECT/'.git/index') == checkpoint['original_index_sha256']
        report = dict(passed=True, checked_utc=datetime.now(timezone.utc).isoformat(),
            original_raw_results_checked=449, original_missing_results_preserved=missing,
            original_main_records_sha256=digest(source_records),
            source_snapshots_checked=len(sources), tables=15, data_rows=114,
            editable_cells_checked=cells_checked, pdf_pages=15, pdf_page_checks=page_checks,
            xlsx_numeric_absolute_and_relative_tolerance=1e-12,
            raw_csv_values_unchanged=True, numeric_csv_json_not_rounded=True,
            ratio_difference_T09_explicitly_in_percentage_points=True,
            original_business_source_files_unchanged=390, export_business_sources_unchanged=True,
            original_index_unchanged=True, actual_new_model_calls=0, new_arrival_credit=0,
            manuscript_text_written=False, new_test_opened=False, goal_complete=False,
            source_sha256=digest(Path(__file__)))
        save(DEST/'report.json', report)
        print(json.dumps({k:v for k,v in report.items() if k!='pdf_page_checks'}, ensure_ascii=False))
    except BaseException as error:
        save(DEST/'interrupted.json', dict(type=type(error).__name__, message=str(error), goal_complete=False))
        raise


if __name__ == '__main__':
    main()
