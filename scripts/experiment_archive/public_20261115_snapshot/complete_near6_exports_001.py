"""在冻结统计关闭后生成语义图和可编辑表PDF，并以Python验证导出。"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

R = Path(__file__).resolve().parents[1]
O = R / 'json/near6_export_completion_001'
F = R / 'figures/corrected_semantics_001'
T = R / 'tables/full_semantics_near6_office_002'
CLEAN = R / 'reproduction/clean_cpu_install_001/venv/bin/python'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def main():
    followup = R / 'json/full_semantics_near6_followup_002/report.json'
    report = json.loads(followup.read_text())
    assert report['actual_CPU_cooperation'] == 9048
    policy = json.loads((R / 'json/resource_policy_002.json').read_text())
    used = sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink())
    assert used + 96 * 2**20 < policy['new_project_artifact_budget_bytes']
    assert shutil.disk_usage(R).free > policy['minimum_physical_free_GiB'] * 2**30
    O.mkdir(exist_ok=False)
    steps = []
    jobs = [
        ('semantic_figure', [sys.executable, str(R / 'source/plot_corrected_semantics_001.py')]),
        ('editable_table_PDF', [
            '/usr/bin/libreoffice', f'-env:UserInstallation={(O / "libreoffice_profile").as_uri()}',
            '--headless', '--convert-to', 'pdf', '--outdir', str(T),
            str(T / '修正邻近距离后全量语义消融.docx'),
        ]),
    ]
    assert not (T / '修正邻近距离后全量语义消融.pdf').exists()
    for name, command in jobs:
        tick = time.monotonic()
        log = O / f'{name}.log'
        with log.open('x') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    env=os.environ.copy(), timeout=300, check=False)
        steps.append(dict(action=name, exit_code=result.returncode,
                          wall_s=time.monotonic()-tick, log_sha256=sha(log)))
        assert result.returncode == 0, log
    # PDF QA is Python only. Pixel inspection is a subsequent, explicit step.
    qa_code = r'''
import hashlib,json
from pathlib import Path
import fitz
R=Path(__import__('sys').argv[1]);F=R/'figures/corrected_semantics_001';T=R/'tables/full_semantics_near6_office_002'
rows=[]
for path,prefix,expected_pages in [(F/'corrected_semantics.pdf',F/'corrected_semantics_pdf_qa',1),(T/'修正邻近距离后全量语义消融.pdf',T/'table_pdf_qa',2)]:
 doc=fitz.open(path);assert len(doc)==expected_pages
 text=''.join(page.get_text() for page in doc);assert '627' in text and '1508' in text
 assert any('\u4e00'<=c<='\u9fff' for c in text)
 previews=[]
 for i,page in enumerate(doc):
  image=prefix.parent/f'{prefix.name}_{i+1}.png';assert not image.exists();page.get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(image);previews.append(dict(path=str(image),sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
 rows.append(dict(path=str(path),pages=len(doc),text_characters=len(text),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),renders=previews));doc.close()
automated=json.loads((F/'automated_qa_001.json').read_text());assert all(not x['text_outside_canvas'] for x in automated['figures'])
out=R/'json/near6_export_completion_001/PDF_export_qa_001.json'
with out.open('x') as f:json.dump(dict(PDF_checks=rows,all_canvas_bounds_pass=True,visual_inspection_pending=True,source_font_sha256=hashlib.sha256((R/'figures/font_runtime_001/NotoSansSC-Regular.ttf').read_bytes()).hexdigest(),goal_complete=False),f,ensure_ascii=False,indent=2)
print(json.dumps(dict(PDF_files=len(rows),pages=sum(x['pages'] for x in rows))))
'''
    with (O / 'PDF_export_qa.log').open('x') as stream:
        result = subprocess.run([str(CLEAN), '-c', qa_code, str(R)], stdout=stream,
                                stderr=subprocess.STDOUT, timeout=120, check=False)
    assert result.returncode == 0, O / 'PDF_export_qa.log'
    save(O / 'report.json', dict(
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), steps=steps,
        source_sha256=sha(__file__), followup_sha256=sha(followup),
        actual_new_GPU_calls=0, source_data_inference_complete=True,
        exports_verified_by_Python=True, visual_inspection_pending=True,
        whole_project_sealed_test=False, thesis_prose_written=False, goal_complete=False,
    ))
    print('semantic figure, two editable tables and PDF exports checked', flush=True)


if __name__ == '__main__':
    main()
